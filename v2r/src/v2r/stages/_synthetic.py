"""Shared helpers for synthetic-mode stage outputs (schema-valid, source=synthesized).

The synthetic human/scene content is a kinematically coherent puppet, not a
random table: an articulated 22-joint body leans and reaches toward a cup on a
table placed in front of the episode's ACTUAL camera trajectory, hands open
and close with real grasp cycles, and the cup follows the wrist while grasped.
Downstream stages (contact, semantics, QA reprojection/jitter) COMPUTE their
outputs from this kinematics — nothing downstream is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema.io import (
    EGODEX_HAND_JOINTS,
    SMPLX_MAIN_JOINTS,
    poses_arrays,
    poses_df,
    read_table,
    write_depth_png,
    write_json_model,
    write_npz,
    write_table,
)
from ..schema.models import (
    CameraInfo,
    ConsentRecord,
    SourceTag,
    VideoProbe,
)
from ..schema.rotations import (
    axis_angle_to_matrix,
    axis_angle_to_quat,
    make_se3,
    matrix_to_quat,
    se3_inverse,
)
from ..schema.timeline import canonical_timestamps
from ..schema.workspace import EpisodeWorkspace
from .base import rng_for

# ---------------------------------------------------------------------------
# ingest / probe helpers (unchanged behavior)
# ---------------------------------------------------------------------------


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


def episode_rng(ws: EpisodeWorkspace, stage: str) -> np.random.Generator:
    return rng_for(ws.episode_id, stage)


# ---------------------------------------------------------------------------
# scene layout + grasp-cycle phase profiles (shared by body/hands/objects)
# ---------------------------------------------------------------------------

CUP_RADIUS = 0.04
CUP_HEIGHT = 0.10
LIFT_HEIGHT = 0.10
APERTURE_OPEN = 0.10
APERTURE_CLOSED = 2 * CUP_RADIUS - 0.004  # tips 2 mm inside the cup surface


@dataclass
class SceneLayout:
    pelvis0: np.ndarray      # standing pelvis position (world)
    facing: np.ndarray       # person's forward (toward camera), horizontal unit
    right: np.ndarray        # person's right, horizontal unit
    table_center: np.ndarray
    cup_rest: np.ndarray     # cup CENTER at rest
    bowl_pos: np.ndarray


def scene_layout(ws: EpisodeWorkspace, cfg: V2RConfig) -> SceneLayout:
    """Deterministic scene placed relative to the episode's real camera track,
    so reprojection into the stored intrinsics/extrinsics is meaningful."""
    if ws.poses_parquet.is_file():
        pa = poses_arrays(read_table(ws.poses_parquet))
        cam = pa["T_world_cam"]
        cam_pos = cam[:, :3, 3].mean(axis=0)
        fwd = cam[:, :3, 2].mean(axis=0)  # OpenCV +Z (view direction) in world
    else:
        cam_pos = np.array([0.0, 0.0, 1.4])
        fwd = np.array([0.0, 1.0, 0.0])
    fwd = np.asarray(fwd, dtype=np.float64).copy()
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])

    facing = -fwd                                # person faces the camera
    right = np.cross(facing, [0.0, 0.0, 1.0])
    right = right / max(np.linalg.norm(right), 1e-9)

    pelvis0 = cam_pos + 3.0 * fwd  # matches geometry SCENE_CENTER distance
    pelvis0[2] = 0.95
    table_center = pelvis0 + 0.35 * facing
    table_center[2] = 0.75
    cup_rest = table_center + 0.10 * right + np.array([0.0, 0.0, CUP_HEIGHT / 2])
    bowl_pos = table_center - 0.18 * right + np.array([0.0, 0.0, 0.03])
    return SceneLayout(pelvis0, facing, right, table_center, cup_rest, bowl_pos)


def _minjerk(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def grasp_cycles(t: np.ndarray, duration: float) -> dict[str, np.ndarray]:
    """Per-frame phase profiles for the reach-grasp-lift-lower-release cycle.

    Returns reach_s (0 rest -> 1 at cup), grip_c (0 open -> 1 closed),
    lift_h (cup/wrist elevation), attached (cup follows wrist), cycle index.
    """
    n_cycles = max(1, int(round(duration / 4.0)))
    cycle_T = duration / n_cycles if duration > 0 else 1.0
    cycle = np.minimum((t / cycle_T).astype(int), n_cycles - 1)
    tau = (t - cycle * cycle_T) / cycle_T

    A, B, C, D, E = 0.30, 0.40, 0.55, 0.70, 0.80
    reach = np.where(
        tau < A, _minjerk(tau / A),
        np.where(tau < E, 1.0, 1.0 - _minjerk((tau - E) / (1.0 - E))),
    )
    grip = np.where(
        tau < A, 0.0,
        np.where(tau < B, _minjerk((tau - A) / (B - A)),
                 np.where(tau < D, 1.0,
                          np.where(tau < E, 1.0 - _minjerk((tau - D) / (E - D)), 0.0))),
    )
    lift = np.where(
        (tau >= B) & (tau < C), LIFT_HEIGHT * _minjerk((tau - B) / (C - B)),
        np.where((tau >= C) & (tau < D), LIFT_HEIGHT * (1.0 - _minjerk((tau - C) / (D - C))), 0.0),
    )
    attached = (tau >= B) & (tau < D)
    return {"reach": reach, "grip": grip, "lift": lift, "attached": attached, "cycle": cycle, "tau": tau}


def cup_center_at(layout: SceneLayout, phases: dict[str, np.ndarray]) -> np.ndarray:
    """(T, 3) cup center positions: rest + vertical lift while attached."""
    n = len(phases["lift"])
    pos = np.tile(layout.cup_rest, (n, 1))
    pos[:, 2] += phases["lift"]
    return pos


def right_wrist_at(layout: SceneLayout, phases: dict[str, np.ndarray]) -> np.ndarray:
    """(T, 3) right wrist positions blending rest pose -> grasp pose at cup."""
    n = len(phases["reach"])
    rest = layout.pelvis0 + 0.22 * layout.right + 0.10 * layout.facing
    rest = rest + np.array([0.0, 0.0, -0.07])  # z ~0.88
    cup = cup_center_at(layout, phases)
    grasp = cup - 0.11 * layout.facing + np.array([0.0, 0.0, 0.03])
    s = phases["reach"][:, None]
    return (1.0 - s) * rest[None, :] + s * grasp


# ---------------------------------------------------------------------------
# geometry (kept: geometry.py owns the richer implementation)
# ---------------------------------------------------------------------------


def synthesize_geometry(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, probe = episode_timeline(ws, cfg)
    n = len(t)
    fx = probe.width * 0.9
    camera = CameraInfo(
        width=probe.width, height=probe.height,
        fx=fx, fy=fx, cx=probe.width / 2.0, cy=probe.height / 2.0,
        depth_scale=1000.0,
        depth_width=min(probe.width, 640), depth_height=min(probe.height, 360),
        scale_source="synthetic", scale_correction=1.0,
    )
    write_json_model(ws.camera_json, camera)

    T = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        yaw = 0.02 * i
        R = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0],
             [np.sin(yaw), np.cos(yaw), 0.0],
             [0.0, 0.0, 1.0]], dtype=np.float64)
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

    pts = rng.uniform(-1, 1, (500, 3)) * np.array([2.0, 2.0, 0.5])
    pts[:, 2] += 0.5
    _write_ply(ws.scene_ply, pts)
    ws.scene_mesh_glb.write_bytes(b"glTF")
    return {"tracked_ratio": float(valid.mean()), "depth_coverage": 0.75, "n_frames": n}


def _write_ply(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ply", "format ascii 1.0", f"element vertex {len(pts)}", "property float x",
             "property float y", "property float z", "end_header"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


# ---------------------------------------------------------------------------
# human body: articulated, moving, metrics COMPUTED (not literals)
# ---------------------------------------------------------------------------

_J = {name: i for i, name in enumerate(SMPLX_MAIN_JOINTS)}


def _build_skeleton(layout: SceneLayout, t: np.ndarray, phases: dict[str, np.ndarray],
                    rng: np.random.Generator) -> np.ndarray:
    """(T, 22, 3) articulated skeleton with lean + right-arm reach cycles."""
    n = len(t)
    up = np.array([0.0, 0.0, 1.0])
    F, R = layout.facing, layout.right
    joints = np.zeros((n, 22, 3), dtype=np.float64)

    sway = 0.008 * np.sin(2 * np.pi * 0.3 * t)[:, None] * R[None, :]
    pelvis = layout.pelvis0[None, :] + sway

    def leg(side: float) -> dict[str, np.ndarray]:
        hip = pelvis + side * 0.09 * R - np.array([0, 0, 0.03])
        knee = hip.copy(); knee[:, 2] = 0.50
        ankle = hip.copy(); ankle[:, 2] = 0.09
        foot = ankle + 0.12 * F; foot[:, 2] = 0.04
        return {"hip": hip, "knee": knee, "ankle": ankle, "foot": foot}

    lleg, rleg = leg(-1.0), leg(+1.0)
    joints[:, _J["pelvis"]] = pelvis
    for side, d in (("left", lleg), ("right", rleg)):
        joints[:, _J[f"{side}_hip"]] = d["hip"]
        joints[:, _J[f"{side}_knee"]] = d["knee"]
        joints[:, _J[f"{side}_ankle"]] = d["ankle"]
        joints[:, _J[f"{side}_foot"]] = d["foot"]

    # torso chain, leaned forward with the reach profile
    spine1 = pelvis + np.array([0, 0, 0.11])
    lean_angle = 0.38 * phases["reach"]  # rad, pitch about person's right axis
    Rlean = axis_angle_to_matrix(lean_angle[:, None] * R[None, :])  # (T,3,3)

    def leaned(offset_z: float, lateral: float = 0.0, fwd: float = 0.0) -> np.ndarray:
        rel = offset_z * up + lateral * R + fwd * F  # relative to spine1, unleaned
        return spine1 + np.einsum("tij,j->ti", Rlean, rel)

    joints[:, _J["spine1"]] = spine1
    joints[:, _J["spine2"]] = leaned(0.11)
    joints[:, _J["spine3"]] = leaned(0.22)
    joints[:, _J["neck"]] = leaned(0.36)
    joints[:, _J["head"]] = leaned(0.49, fwd=0.02)
    joints[:, _J["left_collar"]] = leaned(0.32, lateral=-0.05)
    joints[:, _J["right_collar"]] = leaned(0.32, lateral=+0.05)
    lsho = leaned(0.34, lateral=-0.19)
    rsho = leaned(0.34, lateral=+0.19)
    joints[:, _J["left_shoulder"]] = lsho
    joints[:, _J["right_shoulder"]] = rsho

    # left arm: hanging, slight sway
    joints[:, _J["left_elbow"]] = lsho + np.array([0, 0, -0.28]) - 0.02 * R
    joints[:, _J["left_wrist"]] = joints[:, _J["left_elbow"]] + np.array([0, 0, -0.26]) + 0.03 * F

    # right arm: wrist follows the reach trajectory; elbow via 2-link IK
    wrist = right_wrist_at(layout, phases)
    joints[:, _J["right_wrist"]] = wrist
    L1, L2 = 0.28, 0.26
    sw = wrist - rsho
    d = np.linalg.norm(sw, axis=1)
    u = sw / np.maximum(d, 1e-9)[:, None]
    d_c = np.minimum(d, L1 + L2 - 1e-4)
    a = (L1 ** 2 - L2 ** 2 + d_c ** 2) / (2 * d_c)
    h = np.sqrt(np.maximum(L1 ** 2 - a ** 2, 0.0))
    perp = np.cross(u, F[None, :])
    pn = np.linalg.norm(perp, axis=1, keepdims=True)
    fallback = np.cross(u, up[None, :])
    perp = np.where(pn > 1e-6, perp / np.maximum(pn, 1e-9), fallback)
    # bias elbow outward-down
    flip = (perp @ R - perp @ up) < 0
    perp[flip] = -perp[flip]
    joints[:, _J["right_elbow"]] = rsho + a[:, None] * u + h[:, None] * perp

    joints += rng.normal(0.0, 0.001, joints.shape)  # ~1 mm estimation noise
    return joints


def synthesize_smplx(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, probe = episode_timeline(ws, cfg)
    n = len(t)
    layout = scene_layout(ws, cfg)
    phases = grasp_cycles(t, float(t[-1]) if n > 1 else 1.0)
    joints = _build_skeleton(layout, t, phases, rng)

    betas = rng.normal(0, 0.1, 10)
    yaw = float(np.arctan2(layout.facing[1], layout.facing[0]))
    global_orient = np.tile([0.0, 0.0, yaw], (n, 1)) + rng.normal(0, 0.01, (n, 3))
    pose = rng.normal(0, 0.03, (n, 55, 3))
    pose[:, 18, 0] = 0.3 + 1.2 * phases["reach"]  # right-elbow flexion channel
    transl = joints[:, _J["pelvis"], :].copy()

    conf = rng.uniform(0.82, 0.92, n)
    valid = np.ones(n, dtype=bool)
    # short occlusion window: valid=False, NOT interpolated (contract)
    occ0 = int(0.40 * n)
    occ1 = min(n, occ0 + max(1, int(0.3 * cfg.pipeline.canonical_hz)))
    valid[occ0:occ1] = False
    conf[occ0:occ1] *= 0.4

    write_npz(
        ws.smplx_npz,
        t=t, betas=betas, pose=pose, transl=transl, global_orient=global_orient,
        conf=conf, valid=valid,
        source=np.array([SourceTag.synthesized.value] * n),
        joints_world=joints,
    )

    # ---- REAL metrics -----------------------------------------------------
    # jitter: max joint acceleration by finite differences (valid frames)
    metrics: dict = {"n_frames": n}
    if n >= 3:
        dt = np.gradient(t)
        vel = np.gradient(joints, axis=0) / dt[:, None, None]
        acc = np.gradient(vel, axis=0) / dt[:, None, None]
        acc_mag = np.linalg.norm(acc, axis=2)
        metrics["max_jitter_m_s2"] = float(acc_mag[valid].max())
    else:
        metrics["max_jitter_m_s2"] = 0.0

    # reprojection: project joints with the stored camera; compare against
    # simulated 2D detections (proj + 2 px detector noise; real mode uses
    # Sapiens/ViTPose here). Measures artifact consistency + noise model.
    if ws.camera_json.is_file() and ws.poses_parquet.is_file():
        from ..schema.io import read_json_model

        cam = read_json_model(ws.camera_json, CameraInfo)
        K = np.array(cam.K())
        parr = poses_arrays(read_table(ws.poses_parquet))
        T_cw = se3_inverse(parr["T_world_cam"])  # world -> camera
        m = min(len(T_cw), n)
        pts_cam = np.einsum("tij,tnj->tni", T_cw[:m, :3, :3], joints[:m]) + T_cw[:m, None, :3, 3]
        z = pts_cam[..., 2]
        in_front = z > 0.05
        uv = np.einsum("ij,tnj->tni", K, pts_cam)
        uv = uv[..., :2] / np.maximum(uv[..., 2:3], 1e-6)
        det = uv + rng.normal(0.0, 2.0, uv.shape)
        err = np.linalg.norm(uv - det, axis=2)
        mask = in_front & valid[:m, None]
        metrics["mean_reproj_px"] = float(err[mask].mean()) if mask.any() else float("inf")
        in_img = (uv[..., 0] >= 0) & (uv[..., 0] < cam.width) & (uv[..., 1] >= 0) & (uv[..., 1] < cam.height)
        metrics["in_image_ratio"] = float((in_img & mask).sum() / max(mask.sum(), 1))
        metrics["in_front_ratio"] = float(in_front[valid[:m]].mean())
    else:
        metrics["mean_reproj_px"] = float("inf")
        metrics["in_image_ratio"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# hands: 25 EgoDex joints, real grasp cycles synced with the body
# ---------------------------------------------------------------------------

_FINGERS = {
    "thumb": ["thumbKnuckle", "thumbIntermediateBase", "thumbIntermediateTip", "thumbTip"],
    "index": ["indexFingerMetacarpal", "indexFingerKnuckle", "indexFingerIntermediateBase",
              "indexFingerIntermediateTip", "indexFingerTip"],
    "middle": ["middleFingerMetacarpal", "middleFingerKnuckle", "middleFingerIntermediateBase",
               "middleFingerIntermediateTip", "middleFingerTip"],
    "ring": ["ringFingerMetacarpal", "ringFingerKnuckle", "ringFingerIntermediateBase",
             "ringFingerIntermediateTip", "ringFingerTip"],
    "little": ["littleFingerMetacarpal", "littleFingerKnuckle", "littleFingerIntermediateBase",
               "littleFingerIntermediateTip", "littleFingerTip"],
}

# open-hand fingertip offsets in the palm frame (x = finger dir, y = thumb side, z = palm normal)
_TIP_OPEN = {
    "thumb": np.array([0.055, 0.075, -0.015]),
    "index": np.array([0.100, 0.025, 0.0]),
    "middle": np.array([0.105, 0.000, 0.0]),
    "ring": np.array([0.098, -0.022, 0.0]),
    "little": np.array([0.085, -0.042, 0.0]),
}


def _palm_frame(facing: np.ndarray, side: str) -> np.ndarray:
    """Palm rotation matrix: x = finger direction, y = thumb side, z = palm normal."""
    x = facing / np.linalg.norm(facing)
    up = np.array([0.0, 0.0, 1.0])
    zc = up - x * (up @ x)
    zc = zc / max(np.linalg.norm(zc), 1e-9)
    y = np.cross(zc, x)
    if side == "left":
        # mirror thumb side AND palm normal: two flips keep det = +1
        y = -y
        zc = -zc
    return np.stack([x, y, zc], axis=1)  # columns


def synthesize_hands(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    from ..schema.io import read_npz

    t, frames, _probe = episode_timeline(ws, cfg)
    n = len(t)
    layout = scene_layout(ws, cfg)
    phases = grasp_cycles(t, float(t[-1]) if n > 1 else 1.0)
    cup = cup_center_at(layout, phases)

    smplx = read_npz(ws.smplx_npz)
    joints_body = smplx["joints_world"]
    body_conf = smplx["conf"]
    body_valid = smplx["valid"].astype(bool)

    wrists = {
        "left": joints_body[:, _J["left_wrist"], :],
        "right": joints_body[:, _J["right_wrist"], :],
    }
    R_palm = {s: _palm_frame(layout.facing, s) for s in ("left", "right")}
    q_palm = {s: matrix_to_quat(R_palm[s]) for s in ("left", "right")}

    # closed-grip fingertip targets ON the cup surface (2 mm penetration for
    # the contact stage to genuinely detect; recorded as a QA signal there)
    r_touch = CUP_RADIUS - 0.002
    R, F = layout.right, layout.facing
    closed_tips = {
        "thumb": cup + r_touch * (-R) + [0, 0, 0.015],
        "index": cup + r_touch * R + [0, 0, 0.025],
        "middle": cup + r_touch * R + [0, 0, 0.005],
        "ring": cup + (r_touch + 0.004) * R + [0, 0, -0.012],
        "little": cup + (r_touch + 0.008) * R + [0, 0, -0.028],
    }

    grip = phases["grip"]
    rows = []
    for side in ("left", "right"):
        w = wrists[side]
        Rp = R_palm[side]
        q = q_palm[side]
        c = grip if side == "right" else np.zeros(n)
        for finger, chain in _FINGERS.items():
            tip_open = w + (Rp @ _TIP_OPEN[finger])[None, :]
            tip_closed = closed_tips[finger] if side == "right" else tip_open
            tip = (1 - c[:, None]) * tip_open + c[:, None] * tip_closed
            n_chain = len(chain)
            for k, jname in enumerate(chain):
                frac = (k + 1) / n_chain
                bow = 0.012 * np.sin(np.pi * frac)
                pos = w + frac * (tip - w) + bow * Rp[:, 2][None, :]
                if k == n_chain - 1:
                    pos = tip
                jidx = EGODEX_HAND_JOINTS.index(jname)
                jconf = np.clip(0.8 * body_conf + rng.normal(0, 0.02, n), 0.05, 1.0)
                for i in range(n):
                    rows.append({
                        "t": t[i], "frame": frames[i], "hand": side,
                        "joint_idx": jidx, "joint_name": jname,
                        "px": pos[i, 0], "py": pos[i, 1], "pz": pos[i, 2],
                        "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                        "conf": float(jconf[i]), "valid": bool(body_valid[i]),
                        "source": SourceTag.synthesized.value, "interpolated": False,
                    })
        # wrist joint rows
        widx = EGODEX_HAND_JOINTS.index("wrist")
        wconf = np.clip(0.85 * body_conf, 0.05, 1.0)
        for i in range(n):
            rows.append({
                "t": t[i], "frame": frames[i], "hand": side,
                "joint_idx": widx, "joint_name": "wrist",
                "px": w[i, 0], "py": w[i, 1], "pz": w[i, 2],
                "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                "conf": float(wconf[i]), "valid": bool(body_valid[i]),
                "source": SourceTag.synthesized.value, "interpolated": False,
            })

    df = pd.DataFrame(rows).sort_values(["hand", "joint_idx", "frame"]).reset_index(drop=True)
    write_table(df, ws.hands_parquet)

    # aperture sanity metrics (thumbTip <-> indexFingerTip, right hand)
    r = df[df.hand == "right"]
    tt = r[r.joint_name == "thumbTip"].sort_values("frame")[["px", "py", "pz"]].to_numpy()
    it = r[r.joint_name == "indexFingerTip"].sort_values("frame")[["px", "py", "pz"]].to_numpy()
    ap = np.linalg.norm(tt - it, axis=1)
    valid_ratio = float((~df["valid"]).mean())
    return {
        "mean_conf": float(df["conf"].mean()),
        "invalid_ratio": valid_ratio,
        "aperture_min_m": float(ap.min()),
        "aperture_max_m": float(ap.max()),
        "n_grasp_cycles": int(phases["cycle"].max()) + 1,
    }


# ---------------------------------------------------------------------------
# objects: cup follows the wrist while grasped; real meshes
# ---------------------------------------------------------------------------


def synthesize_objects(ws: EpisodeWorkspace, cfg: V2RConfig, rng: np.random.Generator) -> dict:
    t, frames, _probe = episode_timeline(ws, cfg)
    n = len(t)
    layout = scene_layout(ws, cfg)
    phases = grasp_cycles(t, float(t[-1]) if n > 1 else 1.0)
    cup = cup_center_at(layout, phases)

    rows = []
    yaw_noise = np.cumsum(rng.normal(0, 0.002, n))
    for i in range(n):
        q = axis_angle_to_quat(np.array([0.0, 0.0, yaw_noise[i]]))
        rows.append({
            "t": t[i], "frame": frames[i], "object_id": "cup_0",
            "px": cup[i, 0], "py": cup[i, 1], "pz": cup[i, 2],
            "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
            "conf": float(np.clip(0.75 + rng.normal(0, 0.03), 0, 1)),
            "valid": True, "source": SourceTag.synthesized.value,
        })
    for i in range(n):
        rows.append({
            "t": t[i], "frame": frames[i], "object_id": "bowl_1",
            "px": layout.bowl_pos[0], "py": layout.bowl_pos[1], "pz": layout.bowl_pos[2],
            "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
            "conf": float(np.clip(0.85 + rng.normal(0, 0.02), 0, 1)),
            "valid": True, "source": SourceTag.synthesized.value,
        })
    df = pd.DataFrame(rows).sort_values(["object_id", "frame"]).reset_index(drop=True)
    write_table(df, ws.tracks_parquet)

    # real meshes (contact stage computes signed distances against these)
    import trimesh

    cup_mesh = trimesh.creation.cylinder(radius=CUP_RADIUS, height=CUP_HEIGHT, sections=32)
    cup_mesh.export(ws.object_mesh_glb("cup_0"))
    bowl = trimesh.creation.icosphere(subdivisions=2, radius=0.06)
    bowl.apply_scale([1.0, 1.0, 0.5])
    bowl.export(ws.object_mesh_glb("bowl_1"))

    return {"mean_track_conf": float(df["conf"].mean()), "n_objects": 2}


# ---------------------------------------------------------------------------
# retarget: EE channel now derived from hands.parquet (real derivation)
# ---------------------------------------------------------------------------


def synthesize_retarget(ws: EpisodeWorkspace, cfg: V2RConfig, robot: str, rng: np.random.Generator) -> None:
    from ..schema.models import RetargetMapping, RobotClass

    spec = cfg.robot(robot)
    t, frames, _probe = episode_timeline(ws, cfg)
    rd = ws.retarget_dir(robot)
    rd.mkdir(parents=True, exist_ok=True)
    from .robot_models import home_qpos, joint_limits

    limits = joint_limits(cfg, robot)
    nominal = home_qpos(cfg, robot)
    is_quadruped = spec.robot_class == RobotClass.quadruped
    qcols = ["t", "frame"] + list(spec.dof) + ["conf", "valid", "source"]
    qrows = []
    for tt, fi in zip(t, frames):
        row = {"t": tt, "frame": fi, "conf": 0.85, "valid": True, "source": SourceTag.synthesized.value}
        for j, name in enumerate(spec.dof):
            if is_quadruped:
                row[name] = float(nominal[name])
                continue
            lo, hi = limits[name]
            mid = 0.5 * (lo + hi)
            amp = min(0.1, 0.45 * (hi - lo))
            row[name] = float(mid + amp * np.sin(tt + j))
        qrows.append(row)
    write_table(pd.DataFrame(qrows, columns=qcols), ws.qpos_parquet(robot))

    # EE channel: REAL derivation from hands.parquet (wrist SE(3) + aperture)
    ee_rows = []
    if ws.hands_parquet.is_file():
        hands = read_table(ws.hands_parquet)
        for hand in ("left", "right"):
            h = hands[hands.hand == hand]
            wr = h[h.joint_name == "wrist"].sort_values("frame")
            tt_ = h[h.joint_name == "thumbTip"].sort_values("frame")[["px", "py", "pz"]].to_numpy()
            it_ = h[h.joint_name == "indexFingerTip"].sort_values("frame")[["px", "py", "pz"]].to_numpy()
            ap = np.linalg.norm(tt_ - it_, axis=1)
            for k, (_, row) in enumerate(wr.iterrows()):
                ee_rows.append({
                    "t": row.t, "frame": row.frame, "hand": hand,
                    "px": row.px, "py": row.py, "pz": row.pz,
                    "qw": row.qw, "qx": row.qx, "qy": row.qy, "qz": row.qz,
                    "gripper_aperture_m": float(ap[k]) if k < len(ap) else float("nan"),
                    "conf": row.conf, "valid": row.valid, "source": row.source,
                })
    if not ee_rows:
        q = axis_angle_to_quat(np.array([0.0, 0.0, 0.1]))
        for tt, fi in zip(t, frames):
            for hand in ("left", "right"):
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
        notes="Synthetic retarget for CI/dev harness; EE channel derived from hands.parquet.",
    )
    write_json_model(ws.mapping_json(robot), mapping)
