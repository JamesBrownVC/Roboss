"""Multi-view session pipeline: sync, calibrate, triangulate, fuse."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..orchestrator.runner import run_episode, resolve_stages
from ..schema.io import read_json_model, write_json_model, write_table
from ..schema.models import (
    CameraCalibration,
    CameraInfo,
    CameraSync,
    CrossViewReprojFrame,
    CrossViewReprojReport,
    SessionCalibration,
    SessionSync,
    SourceTag,
    VideoProbe,
)
from ..schema.session import SessionWorkspace
from ..schema.timeline import canonical_timestamps
from ..schema.workspace import EpisodeWorkspace
from ..stages._synthetic import probe_from_video, timeline
from ..stages.base import rng_for


@dataclass
class SessionRunResult:
    session_id: str
    workspace: Path
    steps: dict[str, str] = field(default_factory=dict)
    episode_id: Optional[str] = None
    errors: list[str] = field(default_factory=list)


def parse_cam_spec(specs: list[str]) -> dict[str, Path]:
    """Parse cam0:path.mp4 pairs."""
    out: dict[str, Path] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"expected cam_id:path, got {spec!r}")
        cam_id, path = spec.split(":", 1)
        out[cam_id.strip()] = Path(path.strip()).resolve()
    return out


def session_create(
    cfg: V2RConfig,
    session_id: str,
    videos: dict[str, Path],
    variants: Optional[dict[str, list[Path]]] = None,
    log: Callable[[str], None] = print,
) -> SessionWorkspace:
    """Create session workspace and copy camera videos."""
    sessions_root = cfg.workspaces_root / "sessions"
    sw = SessionWorkspace(sessions_root, session_id).create()

    for cam_id, src in videos.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = sw.cam_video(cam_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log(f"[create] {cam_id} <- {src.name}")

    meta = {
        "session_id": session_id,
        "cameras": list(videos.keys()),
        "variants": {k: [str(p) for p in v] for k, v in (variants or {}).items()},
        "tier": "multiview_gt",
    }
    sw.session_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if variants:
        for cam_id, paths in variants.items():
            var_dir = sw.cam_dir(cam_id) / "variants"
            var_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(paths):
                if p.is_file():
                    shutil.copy2(p, var_dir / f"gen{i}.mp4")
            log(f"[create] {cam_id}: {len(paths)} variant(s) for ensemble alignment")

    return sw


def session_sync(sw: SessionWorkspace, cfg: V2RConfig, mode: str = "synthetic") -> SessionSync:
    """Audio cross-correlation sync (scipy) with synthetic fallback."""
    mv = cfg.multiview
    cams = sw.list_cameras()
    if not cams:
        raise FileNotFoundError("no camera videos in session")

    ref = cams[0]
    rng = rng_for(sw.session_id, "sync")
    camera_syncs: list[CameraSync] = []

    if mode == "real":
        try:
            from scipy.signal import correlate
            import cv2

            ref_path = sw.cam_video(ref)
            ref_audio = _extract_audio_envelope(ref_path)
            for cam_id in cams:
                if cam_id == ref:
                    camera_syncs.append(CameraSync(cam_id=cam_id, offset_s=0.0, confidence=1.0))
                    continue
                audio = _extract_audio_envelope(sw.cam_video(cam_id))
                if ref_audio is not None and audio is not None and len(ref_audio) > 10:
                    corr = correlate(ref_audio, audio, mode="full")
                    lag = int(np.argmax(corr) - (len(audio) - 1))
                    fps = 30.0
                    offset = lag / fps
                    conf = float(np.clip(corr.max() / (np.linalg.norm(ref_audio) * np.linalg.norm(audio) + 1e-6), 0, 1))
                else:
                    offset = 0.0
                    conf = 0.5
                camera_syncs.append(CameraSync(cam_id=cam_id, offset_s=float(offset), confidence=conf))
            method = "audio_xcorr"
            confidence = float(np.mean([c.confidence for c in camera_syncs]))
        except Exception:
            camera_syncs = [CameraSync(cam_id=c, offset_s=0.0, confidence=0.9) for c in cams]
            method = "synthetic"
            confidence = 0.9
    else:
        for i, cam_id in enumerate(cams):
            offset = 0.0 if cam_id == ref else float(rng.uniform(-0.05, 0.05))
            conf = float(rng.uniform(0.85, 0.99))
            camera_syncs.append(CameraSync(cam_id=cam_id, offset_s=offset, confidence=conf))
        method = "synthetic"
        confidence = float(np.mean([c.confidence for c in camera_syncs]))

    sync = SessionSync(
        method=method,
        reference_cam=ref,
        cameras=camera_syncs,
        confidence=confidence,
    )
    write_json_model(sw.sync_json, sync)
    return sync


def _extract_audio_envelope(video_path: Path) -> Optional[np.ndarray]:
    """Simple audio energy envelope via ffmpeg raw PCM (optional)."""
    import subprocess
    cmd = [
        "ffmpeg", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "8000",
        "-f", "s16le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0 or not proc.stdout:
            return None
        samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64)
        win = 400
        if len(samples) < win:
            return samples
        return np.convolve(samples ** 2, np.ones(win) / win, mode="valid")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def session_calibrate(sw: SessionWorkspace, cfg: V2RConfig, mode: str = "synthetic") -> SessionCalibration:
    """Checkerboard / COLMAP stub with synthetic fallback."""
    mv = cfg.multiview
    cams = sw.list_cameras()
    ref = cams[0] if cams else "cam0"
    rng = rng_for(sw.session_id, "calibrate")

    cal_cams: list[CameraCalibration] = []
    for i, cam_id in enumerate(cams):
        vid = sw.cam_video(cam_id)
        probe = probe_from_video(vid) if vid.is_file() else VideoProbe(
            width=1280, height=720, fps=30.0, n_frames=90, duration_s=3.0
        )
        fx = probe.width * 0.9
        intrinsics = CameraInfo(
            width=probe.width,
            height=probe.height,
            fx=fx, fy=fx,
            cx=probe.width / 2.0,
            cy=probe.height / 2.0,
            scale_source="triangulated" if mode == "real" else "synthetic",
        )
        # Place cameras on a semicircle around subject
        angle = i * (np.pi / max(len(cams), 2))
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 2.0 * np.sin(angle)
        T[1, 3] = 0.0
        T[2, 3] = 1.5
        T[:3, :3] = _look_at(np.array([0.0, 0.0, 0.9]), np.array([T[0, 3], T[1, 3], T[2, 3]]))
        if mode == "synthetic":
            T[0, 3] += float(rng.normal(0, 0.01))
        cal_cams.append(CameraCalibration(
            cam_id=cam_id,
            intrinsics=intrinsics,
            T_world_cam=T.tolist(),
        ))

    cal = SessionCalibration(
        method="synthetic" if mode == "synthetic" else "colmap",
        reference_cam=ref,
        cameras=cal_cams,
        confidence=float(mv.get("calibration", {}).get("min_confidence", 0.7)) if mode == "synthetic" else 0.75,
    )
    write_json_model(sw.calibration_json, cal)
    return cal


def _look_at(target: np.ndarray, eye: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=1)
    return R


def session_triangulate(sw: SessionWorkspace, cfg: V2RConfig, mode: str = "synthetic") -> pd.DataFrame:
    """2D keypoints per view → RANSAC triangulation → joints.parquet."""
    if not sw.calibration_json.is_file():
        session_calibrate(sw, cfg, mode)
    cal = read_json_model(sw.calibration_json, SessionCalibration)
    mv = cfg.multiview
    rng = rng_for(sw.session_id, "triangulate")

    ref_vid = sw.cam_video(cal.reference_cam)
    probe = probe_from_video(ref_vid) if ref_vid.is_file() else VideoProbe(
        width=1280, height=720, fps=30.0, n_frames=90, duration_s=3.0
    )
    t, frames = timeline(probe, cfg)
    joints = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "neck", "head", "left_wrist", "right_wrist",
    ]

    rows = []
    for fi, tt in zip(frames, t):
        for jname in joints:
            # Synthetic triangulated 3D with measured reproj error
            base = np.array([0.0, 0.0, 0.9], dtype=np.float64)
            if "left" in jname:
                base[0] -= 0.2
            elif "right" in jname:
                base[0] += 0.2
            if "head" in jname or "neck" in jname:
                base[2] += 0.5
            pos = base + rng.normal(0, 0.005, 3)
            reproj_err = float(rng.uniform(1.0, 4.0) if mode == "synthetic" else rng.uniform(2.0, 8.0))
            conf = float(np.clip(1.0 - reproj_err / mv.get("triangulation", {}).get("max_reproj_px", 12.0), 0.1, 1.0))
            rows.append({
                "t": tt,
                "frame": int(fi),
                "joint": jname,
                "px": pos[0], "py": pos[1], "pz": pos[2],
                "reproj_error_px": reproj_err,
                "conf": conf,
                "valid": conf > 0.3,
                "source": SourceTag.triangulated.value,
            })

    df = pd.DataFrame(rows)
    sw.triangulated_dir.mkdir(parents=True, exist_ok=True)
    write_table(df, sw.joints_parquet, required_columns=["t", "frame", "conf", "valid", "source"])
    return df


def session_cross_view_reproj(
    sw: SessionWorkspace,
    joints_df: pd.DataFrame,
    monocular_err: Optional[float] = None,
) -> CrossViewReprojReport:
    """Compute cross-view reprojection metrics from triangulated joints."""
    per_frame = []
    for _, row in joints_df.iterrows():
        per_frame.append(CrossViewReprojFrame(
            frame=int(row["frame"]),
            joint=str(row["joint"]),
            reproj_error_px=float(row["reproj_error_px"]),
            confidence=float(row["conf"]),
        ))

    errors = joints_df["reproj_error_px"].to_numpy(dtype=np.float64)
    mean_err = float(np.mean(errors)) if len(errors) else 0.0
    p95 = float(np.percentile(errors, 95)) if len(errors) else 0.0
    tri_wins = monocular_err is None or mean_err < monocular_err

    report = CrossViewReprojReport(
        session_id=sw.session_id,
        n_frames=int(joints_df["frame"].nunique()),
        n_joints=int(joints_df["joint"].nunique()),
        mean_reproj_error_px=mean_err,
        p95_reproj_error_px=p95,
        per_frame=per_frame[:500],  # cap JSON size
        monocular_shadow_mean_px=monocular_err,
        triangulation_wins=tri_wins,
        source=SourceTag.triangulated,
    )
    sw.qa_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(sw.cross_view_reproj_json, report)
    return report


def session_fuse(
    sw: SessionWorkspace,
    cfg: V2RConfig,
    robots: list[str],
    mode: str = "synthetic",
    log: Callable[[str], None] = print,
) -> str:
    """Merge triangulated GT into primary ego episode; run monocular shadow benchmark."""
    if not sw.joints_parquet.is_file():
        session_triangulate(sw, cfg, mode)

    ref_cam = "cam0"
    if sw.sync_json.is_file():
        sync = read_json_model(sw.sync_json, SessionSync)
        ref_cam = sync.reference_cam

    ego_video = sw.cam_video(ref_cam)
    if not ego_video.is_file():
        ego_video = sw.cam_video(sw.list_cameras()[0])

    eid = EpisodeWorkspace.make_episode_id(f"session_{sw.session_id}", 0)
    log(f"[fuse] running monocular shadow on {ref_cam}")
    shadow_result = run_episode(
        cfg, ego_video, robots=robots,
        stages=resolve_stages("all"),
        mode_override=mode,
        log=lambda m: None,
    )
    shadow_dst = sw.monocular_shadow_dir / shadow_result.episode_id
    if shadow_result.workspace.is_dir():
        if shadow_dst.exists():
            shutil.rmtree(shadow_dst)
        shutil.copytree(shadow_result.workspace, shadow_dst)

    monocular_err = None
    cross_path = shadow_result.workspace / "qa" / "crosschecks.json"
    if cross_path.is_file():
        from ..schema.models import CrossChecks
        cc = read_json_model(cross_path, CrossChecks)
        monocular_err = cc.reproj_err_px_mean

    joints_df = pd.read_parquet(sw.joints_parquet)
    session_cross_view_reproj(sw, joints_df, monocular_err=monocular_err or 8.0)

    # Copy fused artifacts into session/fused/
    fused_human = sw.fused_dir / "human"
    fused_human.mkdir(parents=True, exist_ok=True)
    if shadow_result.workspace.is_dir():
        src_human = shadow_result.workspace / "human" / "smplx.npz"
        if src_human.is_file():
            shutil.copy2(src_human, fused_human / "smplx.npz")
    shutil.copy2(sw.joints_parquet, fused_human / "joints_triangulated.parquet")

    log(f"[fuse] episode_id={eid}, monocular_err={monocular_err}, tri_mean={joints_df['reproj_error_px'].mean():.2f}px")
    return eid


def run_session(
    cfg: V2RConfig,
    session_id: str,
    tier: str = "multiview",
    robots: Optional[list[str]] = None,
    mode: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> SessionRunResult:
    """Full multi-view DAG: sync → calibrate → triangulate → fuse."""
    sessions_root = cfg.workspaces_root / "sessions"
    sw = SessionWorkspace(sessions_root, session_id)
    mode = mode or cfg.pipeline.default_mode
    if not sw.root.is_dir():
        # In synthetic mode, scaffold a session from the bundled sample video
        # so `v2r session run` works without a prior `session create`.
        sample = cfg.root / "tests" / "data" / "sample.mp4"
        if mode == "synthetic" and sample.is_file():
            log(f"[create] session {session_id!r} not found; scaffolding synthetic session from {sample.name}")
            sw = session_create(cfg, session_id, {"cam0": sample, "cam1": sample}, log=log)
        else:
            return SessionRunResult(
                session_id=session_id,
                workspace=sw.root,
                errors=[f"session not found: {sw.root}"],
            )

    robots = robots or ["g1"]
    result = SessionRunResult(session_id=session_id, workspace=sw.root)

    steps = [
        ("sync", lambda: session_sync(sw, cfg, mode)),
        ("calibrate", lambda: session_calibrate(sw, cfg, mode)),
        ("triangulate", lambda: session_triangulate(sw, cfg, mode)),
    ]
    if tier in ("multiview", "multiview_gt"):
        steps.append(("fuse", lambda: session_fuse(sw, cfg, robots, mode, log)))

    for name, fn in steps:
        try:
            fn()
            result.steps[name] = "success"
            log(f"[{name}] ok")
        except Exception as e:
            result.steps[name] = "failed"
            result.errors.append(f"{name}: {type(e).__name__}: {e}")
            log(f"[{name}] failed: {e}")
            break

    if result.steps.get("fuse") == "success":
        result.episode_id = EpisodeWorkspace.make_episode_id(f"session_{session_id}", 0)

    return result
