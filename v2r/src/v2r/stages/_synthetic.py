"""Shared helpers for synthetic-mode stage outputs (schema-valid, source=synthesized)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema.io import (
    EGODEX_HAND_JOINTS,
    SMPLX_MAIN_JOINTS,
    poses_df,
    write_depth_png,
    write_npz,
    write_table,
)
from ..schema.models import (
    CameraInfo,
    Captions,
    ConsentRecord,
    SceneTags,
    Segment,
    SegmentsFile,
    SourceTag,
    VideoProbe,
)
from ..schema.io import write_json_model
from ..schema.rotations import axis_angle_to_quat, make_se3, se3_from_quat_pos
from ..schema.timeline import canonical_timestamps
from ..schema.workspace import EpisodeWorkspace
from .base import rng_for


def default_consent(episode_id: str) -> ConsentRecord:
    return ConsentRecord(
        consent_id=f"synthetic_{episode_id}",
        license="synthetic-dev-only",
        subject_ids=["synthetic_subject_0"],
        blur_applied=True,
        allow_blurred_video_redistribution=False,
        notes="Auto-generated consent for synthetic/CI mode only.",
    )


def probe_from_video(video_path: Path, fps: float = 30.0) -> VideoProbe:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = n_frames / max(vfps, 1e-6)
    return VideoProbe(
        width=width,
        height=height,
        fps=vfps,
        n_frames=n_frames,
        duration_s=duration_s,
        codec="h264",
        pix_fmt="yuv420p",
        corrupt_frames=0,
        original_path=str(video_path),
        shots=[(0.0, duration_s)],
    )


def load_probe(ws: EpisodeWorkspace) -> VideoProbe:
    from ..schema.io import read_json_model
    if ws.probe_path.is_file():
        return read_json_model(ws.probe_path, VideoProbe)
    if ws.video_path.is_file():
        return probe_from_video(ws.video_path)
    return VideoProbe(width=1280, height=720, fps=30.0, n_frames=90, duration_s=3.0)


def episode_timeline(ws: EpisodeWorkspace, cfg: V2RConfig) -> tuple[np.ndarray, np.ndarray, VideoProbe]:
    """Canonical timeline aligned with geometry/poses when present."""
    from ..schema.io import read_table, poses_arrays

    probe = load_probe(ws)
    if ws.poses_parquet.is_file():
        parr = poses_arrays(read_table(ws.poses_parquet))
        return parr["t"], parr["frame"], probe
    t = canonical_timestamps(probe.duration_s, cfg.pipeline.canonical_hz)
    frames = np.arange(len(t), dtype=np.int64)
    return t, frames, probe


def timeline(probe: VideoProbe, cfg: V2RConfig) -> tuple[np.ndarray, np.ndarray]:
    t = canonical_timestamps(probe.duration_s, cfg.pipeline.canonical_hz)
    frames = np.arange(len(t), dtype=np.int64)
    return t, frames


def synthesize_geometry(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, _probe = episode_timeline(ws, cfg)
    n = len(t)
    fx = probe.width * 0.9
    camera = CameraInfo(
        width=probe.width,
        height=probe.height,
        fx=fx,
        fy=fx,
        cx=probe.width / 2.0,
        cy=probe.height / 2.0,
        depth_scale=1000.0,
        depth_width=min(probe.width, 640),
        depth_height=min(probe.height, 360),
        scale_source="synthetic",
        scale_correction=1.0,
    )
    write_json_model(ws.camera_json, camera)

    T = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        yaw = 0.02 * i
        R = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0],
             [np.sin(yaw), np.cos(yaw), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        T[i] = make_se3(R, np.array([0.01 * i, 0.0, 1.5], dtype=np.float64))
    conf = rng.uniform(0.7, 0.95, n)
    valid = np.ones(n, dtype=bool)
    write_table(poses_df(t, frames, T, conf, valid, SourceTag.synthesized.value), ws.poses_parquet)

    dw, dh = camera.depth_width or probe.width, camera.depth_height or probe.height
    ws.depth_dir.mkdir(parents=True, exist_ok=True)
    for fi in frames:
        depth = np.full((dh, dw), 1.2, dtype=np.float64)
        depth += 0.02 * np.sin(fi / 10.0)
        write_depth_png(ws.depth_frame(int(fi)), depth, camera.depth_scale)

    # minimal scene point cloud
    pts = rng.uniform(-1, 1, (500, 3)) * np.array([2.0, 2.0, 0.5])
    pts[:, 2] += 0.5
    _write_ply(ws.scene_ply, pts)
    ws.scene_mesh_glb.write_bytes(b"glTF")  # placeholder bytes for layout check

    tracked_ratio = float(valid.mean())
    depth_cov = 0.75
    return {"tracked_ratio": tracked_ratio, "depth_coverage": depth_cov, "n_frames": n}


def _write_ply(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ply", "format ascii 1.0", f"element vertex {len(pts)}", "property float x",
             "property float y", "property float z", "end_header"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def synthesize_smplx(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, _probe = episode_timeline(ws, cfg)
    n = len(t)
    betas = rng.normal(0, 0.1, 10)
    pose = rng.normal(0, 0.05, (n, 55, 3))
    global_orient = rng.normal(0, 0.02, (n, 3))
    conf = rng.uniform(0.6, 0.9, n)
    valid = np.ones(n, dtype=bool)

    joints = np.zeros((n, len(SMPLX_MAIN_JOINTS), 3), dtype=np.float64)
    transl = np.zeros((n, 3), dtype=np.float64)
    # World-frame skeleton near geometry SCENE_CENTER (matches synthetic geometry stage)
    scene_center = np.array([0.0, 0.8, 0.6], dtype=np.float64)
    skel_world = np.zeros((len(SMPLX_MAIN_JOINTS), 3), dtype=np.float64)
    skel_world[0] = scene_center + [0.0, 0.0, -0.05]
    skel_world[15] = scene_center + [0.0, 0.0, 0.35]
    skel_world[16] = scene_center + [-0.2, 0.0, 0.25]
    skel_world[19] = scene_center + [0.2, 0.0, 0.25]
    for i in range(n):
        for j in range(len(SMPLX_MAIN_JOINTS)):
            off = skel_world[j] if np.any(skel_world[j]) else skel_world[0]
            joints[i, j] = off
        transl[i] = joints[i, 0]

    write_npz(
        ws.smplx_npz,
        t=t,
        betas=betas,
        pose=pose,
        transl=transl,
        global_orient=global_orient,
        conf=conf,
        valid=valid,
        source=np.array([SourceTag.synthesized.value] * n),
        joints_world=joints,
    )
    return {"n_frames": n, "mean_reproj_px": 4.0, "max_jitter_m_s2": 12.0}


def synthesize_hands(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, _probe = episode_timeline(ws, cfg)
    rows = []
    for hand in ("left", "right"):
        sign = -1.0 if hand == "left" else 1.0
        for fi, tt in zip(frames, t):
            for jidx, jname in enumerate(EGODEX_HAND_JOINTS):
                pos = np.array([sign * 0.3, 0.0, 0.9 - 0.01 * jidx], dtype=np.float64)
                q = axis_angle_to_quat(np.array([0.0, 0.0, 0.05 * jidx]))
                conf = float(rng.uniform(0.5, 0.95))
                valid = conf > 0.35
                rows.append({
                    "t": tt, "frame": fi, "hand": hand, "joint_idx": jidx, "joint_name": jname,
                    "px": pos[0], "py": pos[1], "pz": pos[2],
                    "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                    "conf": conf, "valid": valid,
                    "source": SourceTag.synthesized.value, "interpolated": False,
                })
    df = pd.DataFrame(rows)
    write_table(df, ws.hands_parquet)
    mean_conf = float(df["conf"].mean())
    invalid_ratio = float((~df["valid"]).mean())
    return {"mean_conf": mean_conf, "invalid_ratio": invalid_ratio}


def synthesize_objects(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, _probe = episode_timeline(ws, cfg)
    rows = []
    for oid in ("obj_0", "obj_1"):
        for fi, tt in zip(frames, t):
            pos = np.array([0.4, 0.1 * (1 if oid == "obj_0" else -1), 0.8], dtype=np.float64)
            q = axis_angle_to_quat(np.array([0.0, 0.0, 0.01 * fi]))
            conf = float(rng.uniform(0.4, 0.9))
            rows.append({
                "t": tt, "frame": fi, "object_id": oid,
                "px": pos[0], "py": pos[1], "pz": pos[2],
                "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                "conf": conf, "valid": conf > 0.3,
                "source": SourceTag.synthesized.value,
            })
    write_table(pd.DataFrame(rows), ws.tracks_parquet)
    for oid in ("obj_0", "obj_1"):
        ws.object_mesh_glb(oid).write_bytes(b"glTF")
    return {"mean_track_conf": float(pd.DataFrame(rows)["conf"].mean())}


def synthesize_contacts(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> None:
    t, frames, _probe = episode_timeline(ws, cfg)
    rows = []
    for fi, tt in zip(frames, t):
        for hand in ("left", "right"):
            dist = float(rng.uniform(0.0, 0.02))
            rows.append({
                "t": tt, "frame": fi, "hand": hand, "object_id": "obj_0",
                "contact": dist < 0.005, "min_dist_m": dist, "penetration_m": max(0.0, 0.005 - dist),
                "conf": float(rng.uniform(0.5, 0.9)), "valid": True,
                "source": SourceTag.estimated.value,
            })
    write_table(pd.DataFrame(rows), ws.contacts_parquet)


def synthesize_semantics(ws: EpisodeWorkspace, cfg: V2RConfig) -> None:
    probe = load_probe(ws)
    skill = cfg.verbs[0] if cfg.verbs else "idle"
    segs = SegmentsFile(
        segments=[Segment(start_s=0.0, end_s=probe.duration_s, skill=skill, text=f"Synthetic {skill}")],
        method="synthetic_changepoint",
        source=SourceTag.synthesized,
    )
    write_json_model(ws.segments_json, segs)
    write_json_model(
        ws.captions_json,
        Captions(short="Synthetic demo", medium="Synthetic manipulation demo clip.", long="Synthetic V2R pipeline demo episode.", source=SourceTag.synthesized),
    )
    write_json_model(
        ws.scene_tags_json,
        SceneTags(scene_type="tabletop", lighting="indoor", clutter=2, surfaces=["table"], source=SourceTag.synthesized),
    )


def synthesize_retarget(ws: EpisodeWorkspace, cfg: V2RConfig, robot: str, rng: np.random.Generator) -> None:
    from ..schema.models import RetargetMapping, RobotClass

    spec = cfg.robot(robot)
    t, frames, _probe = episode_timeline(ws, cfg)
    rd = ws.retarget_dir(robot)
    rd.mkdir(parents=True, exist_ok=True)
    qcols = ["t", "frame"] + list(spec.dof) + ["conf", "valid", "source"]
    qrows = []
    for tt, fi in zip(t, frames):
        row = {"t": tt, "frame": fi, "conf": 0.85, "valid": True, "source": SourceTag.synthesized.value}
        for j, name in enumerate(spec.dof):
            row[name] = float(0.1 * np.sin(tt + j))
        qrows.append(row)
    write_table(pd.DataFrame(qrows, columns=qcols), ws.qpos_parquet(robot))

    ee_rows = []
    for tt, fi in zip(t, frames):
        for hand in ("left", "right"):
            q = axis_angle_to_quat(np.array([0.0, 0.0, 0.1]))
            ee_rows.append({
                "t": tt, "frame": fi, "hand": hand,
                "px": 0.3, "py": 0.0, "pz": 0.9,
                "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                "gripper_aperture_m": 0.04,
                "conf": 0.8, "valid": True, "source": SourceTag.synthesized.value,
            })
    write_table(pd.DataFrame(ee_rows), ws.ee_parquet(robot))

    method = "gmr" if spec.robot_class == RobotClass.humanoid_wholebody else (
        "mink_diff_ik" if spec.robot_class == RobotClass.ee_manipulator else "base_twist_abstraction"
    )
    mapping = RetargetMapping(
        robot=robot,
        robot_class=spec.robot_class,
        retarget_method=method,
        retarget_version="0.1.0-synthetic",
        provenance="kinematic-retarget" if spec.robot_class != RobotClass.quadruped else "command-abstraction",
        key_body_map=dict(spec.key_body_map),
        notes="Synthetic retarget for CI/dev harness.",
    )
    write_json_model(ws.mapping_json(robot), mapping)


def episode_rng(ws: EpisodeWorkspace, stage: str) -> np.random.Generator:
    return rng_for(ws.episode_id, stage)
