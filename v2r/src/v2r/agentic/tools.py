"""Perception tools available to the agentic labeler.

Every tool returns a JSON-safe summary dict and (where applicable) writes
contract artifacts into the episode workspace tagged source='estimated'.
All tools degrade gracefully: a missing model/package yields
{"available": False, "reason": ...} instead of an exception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema.io import EGODEX_HAND_JOINTS, write_table
from ..schema.models import SourceTag
from ..schema.workspace import EpisodeWorkspace

MEDIAPIPE_POSE_JOINTS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

# MediaPipe hand landmark index -> EgoDex joint name (21 of 25; the four
# finger metacarpals are absent and get interpolated wrist->knuckle)
MP_HAND_TO_EGODEX = {
    0: "wrist",
    1: "thumbKnuckle", 2: "thumbIntermediateBase", 3: "thumbIntermediateTip", 4: "thumbTip",
    5: "indexFingerKnuckle", 6: "indexFingerIntermediateBase",
    7: "indexFingerIntermediateTip", 8: "indexFingerTip",
    9: "middleFingerKnuckle", 10: "middleFingerIntermediateBase",
    11: "middleFingerIntermediateTip", 12: "middleFingerTip",
    13: "ringFingerKnuckle", 14: "ringFingerIntermediateBase",
    15: "ringFingerIntermediateTip", 16: "ringFingerTip",
    17: "littleFingerKnuckle", 18: "littleFingerIntermediateBase",
    19: "littleFingerIntermediateTip", 20: "littleFingerTip",
}
_METACARPALS = {
    "thumb": None,  # thumb has no metacarpal joint in the EgoDex 25 set
    "indexFingerMetacarpal": "indexFingerKnuckle",
    "middleFingerMetacarpal": "middleFingerKnuckle",
    "ringFingerMetacarpal": "ringFingerKnuckle",
    "littleFingerMetacarpal": "littleFingerKnuckle",
}


def _per_second_bins(times: np.ndarray, values: np.ndarray,
                     duration_s: float, agg=np.mean) -> list[float]:
    """Aggregate (t, value) samples into 1-second bins over [0, duration]."""
    n_bins = max(int(np.ceil(duration_s)), 1)
    out = []
    for b in range(n_bins):
        m = (times >= b) & (times < b + 1)
        out.append(round(float(agg(values[m])), 3) if m.any() else 0.0)
    return out


def probe_video(video: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"available": False, "reason": f"cannot open {video}"}
    out = {
        "available": True,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    out["duration_s"] = out["n_frames"] / max(out["fps"], 1e-6)
    cap.release()
    return out


def sample_frames(video: Path, n: int = 6, max_width: int = 512,
                  save_dir: Path | None = None) -> tuple[list[bytes], list[float], dict]:
    """Evenly spaced JPEG frames for VLM analysis (+ optional review copies)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    idxs = np.unique(np.linspace(0, max(total - 1, 0), n).astype(int))
    jpegs: list[bytes] = []
    stamps: list[float] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(max_width * h / w)))
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            jpegs.append(buf.tobytes())
            stamps.append(i / fps)
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / f"agentic_frame_{int(i):06d}.jpg").write_bytes(buf.tobytes())
    cap.release()
    return jpegs, stamps, {"available": bool(jpegs), "n_sampled": len(jpegs)}


def motion_timeline(video: Path, bin_s: float = 0.5) -> dict:
    """Frame-difference motion energy per time bin + changepoint candidates."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"available": False, "reason": "cannot open video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(fps / 10))  # ~10 samples/s
    prev = None
    energies, times = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            g = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                energies.append(float(np.mean(np.abs(g - prev))))
                times.append(idx / fps)
            prev = g
        idx += 1
    cap.release()
    if not energies:
        return {"available": False, "reason": "no frames"}
    e = np.array(energies)
    t = np.array(times)
    bins = np.arange(0.0, t[-1] + bin_s, bin_s)
    binned = [float(e[(t >= b) & (t < b + bin_s)].mean()) if ((t >= b) & (t < b + bin_s)).any() else 0.0
              for b in bins]
    de = np.abs(np.diff(binned))
    thresh = np.mean(de) + 1.5 * np.std(de) if len(de) else 0.0
    changepoints = [round(float(bins[i + 1]), 2) for i in np.flatnonzero(de > max(thresh, 1e-6))]
    return {
        "available": True,
        "bin_s": bin_s,
        "energy_per_bin": [round(x, 2) for x in binned],
        "mean_energy": round(float(e.mean()), 2),
        "changepoint_candidates_s": changepoints,
    }


# ---------------------------------------------------------------------------
# MediaPipe pose
# ---------------------------------------------------------------------------


def track_human_pose(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig) -> dict:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        return {"available": False, "reason": f"mediapipe not importable: {e}"}

    model = cfg.root / "assets" / "models" / "pose_landmarker_lite.task"
    if not model.is_file():
        return {"available": False, "reason": f"missing {model}"}

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    rows = []
    present = 0
    total = 0
    wrist_path = []
    with mp_vision.PoseLandmarker.create_from_options(options) as lm:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = int(idx / fps * 1000)
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, ts_ms)
            total += 1
            if res.pose_landmarks and res.pose_world_landmarks:
                present += 1
                lms2d = res.pose_landmarks[0]
                lms3d = res.pose_world_landmarks[0]
                for j, name in enumerate(MEDIAPIPE_POSE_JOINTS):
                    p2, p3 = lms2d[j], lms3d[j]
                    conf = float(min(getattr(p2, "visibility", 0.5) or 0.5,
                                     getattr(p2, "presence", 1.0) or 1.0))
                    rows.append({
                        "t": idx / fps, "frame": idx, "joint_idx": j, "joint_name": name,
                        "u": p2.x, "v": p2.y,
                        "x": p3.x, "y": p3.y, "z": p3.z,
                        "conf": max(0.0, min(1.0, conf)), "valid": conf > 0.5,
                        "source": SourceTag.estimated.value,
                    })
                rw = lms3d[16]
                wrist_path.append((rw.x, rw.y, rw.z))
            idx += 1
    cap.release()
    if not rows:
        return {"available": True, "person_present_ratio": 0.0, "n_frames": total,
                "note": "no pose detected in any frame"}
    df = pd.DataFrame(rows)
    out = ws.human_dir / "pose2d3d_mediapipe.parquet"
    write_table(df, out)
    wp = np.array(wrist_path) if wrist_path else np.zeros((1, 3))
    motion = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum()) if len(wp) > 1 else 0.0
    duration_s = total / max(fps, 1e-6)
    nose = df[df["joint_name"] == "nose"]
    presence_1s = _per_second_bins(nose["t"].to_numpy(),
                                   np.ones(len(nose)), duration_s, agg=len)
    frames_per_s = max(fps, 1.0)
    wrist = df[df["joint_name"] == "right_wrist"].sort_values("t")
    speed_1s: list[float] = []
    if len(wrist) > 1:
        wt = wrist["t"].to_numpy()
        wxyz = wrist[["x", "y", "z"]].to_numpy()
        dt = np.diff(wt)
        speed = np.linalg.norm(np.diff(wxyz, axis=0), axis=1) / np.maximum(dt, 1e-6)
        speed_1s = _per_second_bins(wt[1:], speed, duration_s)
    presence_ps = [round(min(c / frames_per_s, 1.0), 2) for c in presence_1s]
    # a real tracked human yields sustained per-second presence; rapid
    # flicker between 0 and 1 is the signature of false positives (animals,
    # foam, foliage all trigger MediaPipe's person detector)
    flicker = float(np.mean(np.abs(np.diff([p > 0.5 for p in presence_ps]))) \
                    if len(presence_ps) > 1 else 0.0)
    return {
        "available": True,
        "artifact": ws.rel(out),
        "person_present_ratio": round(present / max(total, 1), 3),
        "mean_conf": round(float(df["conf"].mean()), 3),
        "n_frames": total,
        "right_wrist_path_length_m": round(motion, 3),
        "person_present_per_second": presence_ps,
        "right_wrist_speed_m_s_per_second": speed_1s,
        "presence_flicker": round(flicker, 2),
        "caveat": "detector fires on person-LIKE shapes (animals, statues, "
                  "textures); high flicker or implausible wrist speed means "
                  "false positive - verify against what you SEE in frames",
    }


# ---------------------------------------------------------------------------
# MediaPipe hands -> EgoDex 25-joint hands.parquet
# ---------------------------------------------------------------------------


def track_hands(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig) -> dict:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        return {"available": False, "reason": f"mediapipe not importable: {e}"}

    model = cfg.root / "assets" / "models" / "hand_landmarker.task"
    if not model.is_file():
        return {"available": False, "reason": f"missing {model}"}

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    rows = []
    frames_with_hands = 0
    total = 0
    apertures = []
    with mp_vision.HandLandmarker.create_from_options(options) as lm:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = int(idx / fps * 1000)
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, ts_ms)
            total += 1
            if res.hand_world_landmarks:
                frames_with_hands += 1
            for h_i, world in enumerate(res.hand_world_landmarks or []):
                handed = res.handedness[h_i][0]
                side = handed.category_name.lower()  # 'left' | 'right'
                score = float(handed.score)
                pts = {}
                for mp_idx, joint in MP_HAND_TO_EGODEX.items():
                    p = world[mp_idx]
                    pts[joint] = (p.x, p.y, p.z)
                for meta, knuckle in _METACARPALS.items():
                    if meta == "thumb" or knuckle is None:
                        continue
                    w0 = np.array(pts["wrist"])
                    k0 = np.array(pts[knuckle])
                    pts[meta] = tuple(w0 + 0.5 * (k0 - w0))
                ap = float(np.linalg.norm(np.array(pts["thumbTip"]) - np.array(pts["indexFingerTip"])))
                apertures.append(ap)
                for joint, (x, y, z) in pts.items():
                    interp = joint in _METACARPALS and joint != "thumb"
                    rows.append({
                        "t": idx / fps, "frame": idx, "hand": side,
                        "joint_idx": EGODEX_HAND_JOINTS.index(joint), "joint_name": joint,
                        "px": x, "py": y, "pz": z,
                        # hand-frame orientation is not observable from MP world
                        # landmarks; identity quat + reduced conf, never faked
                        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                        "conf": max(0.0, min(1.0, score * (0.6 if interp else 1.0))),
                        "valid": score > 0.5,
                        "source": SourceTag.estimated.value,
                        "interpolated": bool(interp),
                    })
            idx += 1
    cap.release()
    if not rows:
        return {"available": True, "hands_present_ratio": 0.0, "n_frames": total,
                "note": "no hands detected"}
    df = pd.DataFrame(rows).sort_values(["hand", "joint_idx", "frame"]).reset_index(drop=True)
    out = ws.human_dir / "hands_mediapipe.parquet"
    write_table(df, out)
    ap = np.array(apertures)
    ap_t = df[df["joint_name"] == "wrist"]["t"].to_numpy()[:len(ap)]
    duration_s = total / max(fps, 1e-6)
    aperture_1s = _per_second_bins(ap_t, ap, duration_s) if len(ap_t) else []
    return {
        "available": True,
        "artifact": ws.rel(out),
        "hands_present_ratio": round(frames_with_hands / max(total, 1), 3),
        "n_hand_observations": int(len(df) / 25),
        "aperture_m": {"min": round(float(ap.min()), 4), "max": round(float(ap.max()), 4),
                       "mean": round(float(ap.mean()), 4)},
        "aperture_m_per_second": aperture_1s,
        "note": "positions are hand-centered metric estimates (MediaPipe world "
                "landmarks); orientations identity (not observable), conf reduced",
    }


# ---------------------------------------------------------------------------
# Trajectory characterization (time-series descriptors for any moving point)
# ---------------------------------------------------------------------------


def characterize_trajectory(t: np.ndarray, pos: np.ndarray) -> dict:
    """Movement descriptors for a trajectory: speed profile, straightness,
    direction changes, smoothness (log dimensionless jerk), periodicity
    (autocorrelation of speed). `t` (n,), `pos` (n, d)."""
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if len(t) < 3:
        return {"n_samples": int(len(t)), "note": "too short to characterize"}
    # de-jitter before differentiating: raw tracker output makes jerk and
    # direction-change counts measure sensor noise, not movement
    if len(t) >= 7:
        try:
            from scipy.signal import savgol_filter

            win = min(len(t) - (1 - len(t) % 2), 9)
            pos = savgol_filter(pos, win, 2, axis=0)
        except Exception:  # noqa: BLE001
            pass
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-6, dt)
    vel = np.diff(pos, axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    path_len = float(np.sum(speed * dt))
    net_disp = float(np.linalg.norm(pos[-1] - pos[0]))
    straightness = net_disp / path_len if path_len > 1e-9 else 1.0

    # direction changes: >90 deg turns between successive MOVING samples.
    # Comparing across stopped gaps matters: at a reversal the speed passes
    # through zero, so consecutive-pair checks would never see the turn.
    moving_idx = np.flatnonzero(speed > max(float(np.max(speed)) * 0.2, 1e-9))
    n_turns = 0
    for a, b in zip(moving_idx[:-1], moving_idx[1:]):
        cos = float(np.dot(vel[b], vel[a]) / (speed[b] * speed[a] + 1e-12))
        if cos < 0.0:
            n_turns += 1

    # log dimensionless jerk (higher = smoother; ~ -5 natural reach, << -10 jittery)
    duration = float(t[-1] - t[0])
    smoothness = None
    if len(vel) >= 3 and path_len > 1e-9:
        jerk = np.diff(vel, axis=0) / dt[1:, None]
        jerk_int = float(np.sum(np.linalg.norm(jerk, axis=1) ** 2 * dt[1:]))
        mean_speed = path_len / max(duration, 1e-9)
        dlj = jerk_int * duration ** 3 / max(mean_speed ** 2, 1e-12)
        smoothness = round(float(-np.log(max(dlj, 1e-12))), 2)

    # periodicity from speed autocorrelation (waving/stirring/walking cycles)
    period_s, period_strength = None, 0.0
    s = speed - speed.mean()
    if len(s) >= 8 and float(np.std(s)) > 1e-9:
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        ac /= ac[0]
        mean_dt = float(np.mean(dt))
        for lag in range(2, len(ac) - 1):
            if ac[lag] > 0.3 and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
                period_s = round(lag * mean_dt, 2)
                period_strength = round(float(ac[lag]), 2)
                break
    return {
        "n_samples": int(len(t)),
        "duration_s": round(duration, 2),
        "path_length": round(path_len, 4),
        "net_displacement": round(net_disp, 4),
        "straightness": round(float(straightness), 2),
        "mean_speed": round(float(speed.mean()), 4),
        "max_speed": round(float(speed.max()), 4),
        "n_direction_changes": int(n_turns),
        "smoothness_log_dj": smoothness,
        "periodic": period_s is not None,
        "period_s": period_s,
        "period_strength": period_strength,
    }


# ---------------------------------------------------------------------------
# Dense optical flow (OpenCV DIS): camera vs subject motion per second
# ---------------------------------------------------------------------------


def optical_flow_timeline(video: Path, ws: EpisodeWorkspace | None = None) -> dict:
    """DIS dense flow at ~8 Hz on downscaled frames. Separates camera motion
    (median flow over the frame) from subject motion (residual after removing
    the camera component), per second."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"available": False, "reason": "cannot open video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, round(fps / 8))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev = None
    samples = []  # (t, cam_dx, cam_dy, subject_mag, moving_frac)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            g = cv2.cvtColor(cv2.resize(frame, (256, 144)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = dis.calc(prev, g, None)
                sample_dt = stride / fps
                cam = np.median(flow.reshape(-1, 2), axis=0) / sample_dt  # px/s
                resid = flow / sample_dt - cam[None, None, :]
                rmag = np.linalg.norm(resid, axis=2)
                samples.append((idx / fps, float(cam[0]), float(cam[1]),
                                float(rmag.mean()), float((rmag > 4.0).mean())))
            prev = g
        idx += 1
    cap.release()
    if not samples:
        return {"available": False, "reason": "no frames"}
    arr = np.array(samples)
    duration_s = arr[-1, 0] + 1e-6
    cam_mag = np.linalg.norm(arr[:, 1:3], axis=1)
    out = {
        "available": True,
        "units": "px/s at 256x144 analysis resolution",
        "camera_motion_px_s_per_second": _per_second_bins(arr[:, 0], cam_mag, duration_s),
        "subject_motion_px_s_per_second": _per_second_bins(arr[:, 0], arr[:, 3], duration_s),
        "moving_area_fraction_per_second": _per_second_bins(arr[:, 0], arr[:, 4], duration_s),
        "camera_moving": bool(np.median(cam_mag) > 3.0),
        "note": "subject motion is flow residual after removing median (camera) "
                "flow; high subject + low camera = something moves in a static shot",
    }
    if ws is not None:
        import json as _json

        (ws.qa_dir / "flow_timeline.json").write_text(
            _json.dumps(out, indent=1), encoding="utf-8")
        out["artifact"] = ws.rel(ws.qa_dir / "flow_timeline.json")
    return out


# ---------------------------------------------------------------------------
# Shot-cut detection (PySceneDetect)
# ---------------------------------------------------------------------------


def detect_scenes(video: Path) -> dict:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as e:
        return {"available": False, "reason": f"scenedetect not importable: {e}"}
    try:
        scene_list = detect(str(video), ContentDetector(threshold=27.0))
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"scenedetect failed: {e}"}
    shots = [{"start_s": round(s.get_seconds(), 2), "end_s": round(e.get_seconds(), 2)}
             for s, e in scene_list]
    return {
        "available": True,
        "n_shots": max(len(shots), 1),
        "shots": shots[:20],
        "single_shot": len(shots) <= 1,
        "note": "cuts mean the video is edited; treat each shot as its own "
                "temporal context for segmentation",
    }


# ---------------------------------------------------------------------------
# Action recognition prior (torchvision S3D, Kinetics-400, CPU)
# ---------------------------------------------------------------------------

_S3D_CACHE: dict = {}


def recognize_action(video: Path, n_windows: int = 3) -> dict:
    """Kinetics-400 top-5 per temporal window via torchvision S3D (8.3M params,
    CPU-feasible). A weak prior: 400 everyday/sports classes, not a
    manipulation taxonomy - use it to corroborate, never to overrule visuals."""
    try:
        import torch
        from torchvision.models.video import S3D_Weights, s3d
    except ImportError as e:
        return {"available": False, "reason": f"torchvision video not importable: {e}"}
    import cv2

    try:
        if "model" not in _S3D_CACHE:
            weights = S3D_Weights.KINETICS400_V1
            model = s3d(weights=weights)
            model.eval()
            _S3D_CACHE["model"] = model
            _S3D_CACHE["transform"] = weights.transforms()
            _S3D_CACHE["categories"] = weights.meta["categories"]
    except Exception as e:  # noqa: BLE001 - typically: no network for weights
        return {"available": False, "reason": f"S3D weights unavailable: {e}"}

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = total / max(fps, 1e-6)
    n_windows = max(1, min(n_windows, int(duration // 2) or 1))
    frames_per_window = 16
    windows = []
    for w in range(n_windows):
        w0 = w * total // n_windows
        w1 = (w + 1) * total // n_windows - 1
        idxs = np.linspace(w0, max(w1, w0), frames_per_window).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(cv2.resize(frame, (256, 256)),
                                       cv2.COLOR_BGR2RGB))
        if len(frames) >= 13:
            windows.append((w0 / fps, min((w1 + 1) / fps, duration), frames))
    cap.release()
    if not windows:
        return {"available": False, "reason": "not enough decodable frames"}

    import torch

    model = _S3D_CACHE["model"]
    transform = _S3D_CACHE["transform"]
    cats = _S3D_CACHE["categories"]
    results = []
    with torch.inference_mode():
        for t0, t1, frames in windows:
            clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # T,C,H,W
            batch = transform(clip).unsqueeze(0)
            probs = model(batch).softmax(dim=1)[0]
            top = torch.topk(probs, 5)
            results.append({
                "t_start_s": round(t0, 1), "t_end_s": round(t1, 1),
                "top5": [{"label": cats[i], "p": round(float(p), 3)}
                         for p, i in zip(top.values, top.indices)],
            })
    return {
        "available": True,
        "model": "torchvision S3D (Kinetics-400)",
        "windows": results,
        "note": "weak prior over 400 everyday/sports classes; corroborate with "
                "visuals - never overrule what you see with this classifier",
    }


# ---------------------------------------------------------------------------
# Motion primitives: multi-channel changepoint segmentation (ruptures)
# ---------------------------------------------------------------------------


def segment_motion_channels(grid_t: np.ndarray, channels: dict[str, np.ndarray],
                            min_seg_s: float = 0.5) -> list[dict]:
    """Segment a multi-channel motion signal into primitives via kernel
    changepoint detection (ruptures KernelCPD, rbf). Returns per-segment
    boundaries with per-channel means. Pure function (unit-testable)."""
    import ruptures as rpt

    names = [k for k, v in channels.items() if np.std(v) > 1e-9]
    if not names or len(grid_t) < 10:
        return [{"start_s": float(grid_t[0]), "end_s": float(grid_t[-1]),
                 "channels": {k: round(float(np.mean(v)), 4)
                              for k, v in channels.items()}}] if len(grid_t) else []

    def _smooth(v: np.ndarray) -> np.ndarray:
        # savgol over ~0.7 s: MediaPipe world coords jitter frame-to-frame
        try:
            from scipy.signal import savgol_filter

            win = min(len(v) - (1 - len(v) % 2), 7)
            if win >= 5:
                return savgol_filter(v, win, 2)
        except Exception:  # noqa: BLE001
            pass
        return v

    sig = np.column_stack([
        (lambda s: (s - s.mean()) / (s.std() + 1e-12))(_smooth(channels[k]))
        for k in names])
    dt = float(np.mean(np.diff(grid_t)))
    min_size = max(2, int(min_seg_s / max(dt, 1e-6)))
    algo = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(sig)
    # tuned empirically on real clips: linear-in-d penalties over-penalize
    # when flat/noisy channels are stacked in; sqrt(d)*log(n) recovers
    # human-meaningful phase boundaries at 2, 3 and 5 channels
    pen = float(np.sqrt(sig.shape[1]) * np.log(len(sig)))
    bkps = algo.predict(pen=pen)
    segments = []
    prev = 0
    for b in bkps:
        seg_slice = slice(prev, b)
        segments.append({
            "start_s": round(float(grid_t[prev]), 2),
            "end_s": round(float(grid_t[min(b, len(grid_t) - 1)]), 2),
            "channels": {k: round(float(np.mean(channels[k][seg_slice])), 4)
                         for k in channels},
        })
        prev = b
    return segments


def motion_primitives(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig) -> dict:
    """Fuse every available motion signal (optical flow subject motion, wrist
    speed, hand aperture, object-track speeds) onto a common 10 Hz grid and
    segment it into motion primitives via kernel changepoint detection.
    Run AFTER pose/hands/track so their artifacts feed extra channels."""
    probe = probe_video(video)
    if not probe.get("available"):
        return {"available": False, "reason": "cannot probe video"}
    duration = float(probe["duration_s"])
    grid_t = np.arange(0.0, max(duration, 0.2), 0.1)
    channels: dict[str, np.ndarray] = {}
    channel_sources: dict[str, str] = {}

    # channel 1: subject motion from dense flow (always computable)
    flow = optical_flow_timeline(video)
    if flow.get("available"):
        sm = np.array(flow["subject_motion_px_s_per_second"], dtype=float)
        sec_t = np.arange(len(sm)) + 0.5
        channels["flow_subject_px_s"] = np.interp(grid_t, sec_t, sm)
        channel_sources["flow_subject_px_s"] = "DIS optical flow residual"
        cam = np.array(flow["camera_motion_px_s_per_second"], dtype=float)
        channels["flow_camera_px_s"] = np.interp(grid_t, sec_t, cam)
        channel_sources["flow_camera_px_s"] = "DIS optical flow median"

    # channel 2: right-wrist speed from pose artifact (if the pose tool ran)
    pose_pq = ws.human_dir / "pose2d3d_mediapipe.parquet"
    wrist_char = None
    if pose_pq.is_file():
        try:
            pdf = pd.read_parquet(pose_pq)
            wrist = pdf[pdf["joint_name"] == "right_wrist"].sort_values("t")
            if len(wrist) > 3:
                wt = wrist["t"].to_numpy()
                wxyz = wrist[["x", "y", "z"]].to_numpy()
                dt = np.maximum(np.diff(wt), 1e-6)
                sp = np.linalg.norm(np.diff(wxyz, axis=0), axis=1) / dt
                channels["wrist_speed_m_s"] = np.interp(grid_t, wt[1:], sp,
                                                        left=0.0, right=0.0)
                channel_sources["wrist_speed_m_s"] = "MediaPipe pose artifact"
                wrist_char = characterize_trajectory(wt, wxyz)
        except Exception:  # noqa: BLE001 - artifact may be malformed
            pass

    # channel 3: hand aperture from hands artifact
    hands_pq = ws.human_dir / "hands_mediapipe.parquet"
    if hands_pq.is_file():
        try:
            hdf = pd.read_parquet(hands_pq)
            th = hdf[hdf["joint_name"] == "thumbTip"].sort_values("t")
            ix = hdf[hdf["joint_name"] == "indexFingerTip"].sort_values("t")
            m = min(len(th), len(ix))
            if m > 3:
                ap = np.linalg.norm(th[["px", "py", "pz"]].to_numpy()[:m]
                                    - ix[["px", "py", "pz"]].to_numpy()[:m], axis=1)
                channels["hand_aperture_m"] = np.interp(
                    grid_t, th["t"].to_numpy()[:m], ap, left=0.0, right=0.0)
                channel_sources["hand_aperture_m"] = "MediaPipe hands artifact"
        except Exception:  # noqa: BLE001
            pass

    # channel 4: mean object-track speed from track artifact
    tracks_pq = ws.objects_dir / "tracks_2d.parquet"
    if tracks_pq.is_file():
        try:
            tdf = pd.read_parquet(tracks_pq)
            speeds_t, speeds_v = [], []
            for _, grp in tdf.groupby("track_id"):
                grp = grp.sort_values("t")
                if len(grp) < 3:
                    continue
                gt = grp["t"].to_numpy()
                gxy = grp[["cx_norm", "cy_norm"]].to_numpy()
                sp = np.linalg.norm(np.diff(gxy, axis=0), axis=1) / \
                    np.maximum(np.diff(gt), 1e-6)
                speeds_t.append(gt[1:])
                speeds_v.append(sp)
            if speeds_t:
                all_t = np.concatenate(speeds_t)
                all_v = np.concatenate(speeds_v)
                order = np.argsort(all_t)
                channels["object_speed_frac_s"] = np.interp(
                    grid_t, all_t[order], all_v[order], left=0.0, right=0.0)
                channel_sources["object_speed_frac_s"] = "ByteTrack tracks artifact"
        except Exception:  # noqa: BLE001
            pass

    if not channels:
        return {"available": False, "reason": "no motion channels computable"}
    segments = segment_motion_channels(grid_t, channels)
    # coarse activity tag per segment so the LLM can scan quickly
    subj = channels.get("flow_subject_px_s")
    subj_hi = float(np.percentile(subj, 75)) if subj is not None else 0.0
    for seg in segments:
        lvl = seg["channels"].get("flow_subject_px_s", 0.0)
        seg["activity"] = ("high_motion" if subj_hi > 0 and lvl > subj_hi else
                           "low_motion" if lvl > max(subj_hi * 0.25, 0.5) else "static")
    out = {
        "available": True,
        "n_channels": len(channels),
        "channel_sources": channel_sources,
        "n_primitives": len(segments),
        "primitives": segments[:24],
        "note": "boundaries are statistical changepoints over all channels; "
                "channels present depend on which tools ran before this one",
    }
    if wrist_char:
        out["wrist_trajectory"] = wrist_char
    import json as _json

    path = ws.semantics_dir / "motion_primitives.json"
    path.write_text(_json.dumps(out, indent=1), encoding="utf-8")
    out["artifact"] = ws.rel(path)
    return out


# ---------------------------------------------------------------------------
# YOLO object detection (2D)
# ---------------------------------------------------------------------------


def detect_objects(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig, stride: int = 5) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        return {"available": False, "reason": f"ultralytics not importable: {e}"}

    weights = cfg.root / "yolo11n.pt"
    if not weights.is_file():
        return {"available": False, "reason": f"missing {weights}"}

    import cv2

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            res = model.predict(frame, verbose=False, conf=0.3)[0]
            for b in res.boxes:
                cls = res.names[int(b.cls[0])]
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                rows.append({
                    "t": idx / fps, "frame": idx, "class": cls,
                    "conf": float(b.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "valid": True, "source": SourceTag.estimated.value,
                })
        idx += 1
    cap.release()
    if not rows:
        return {"available": True, "n_detections": 0, "classes": {},
                "note": "no objects detected above conf 0.3"}
    df = pd.DataFrame(rows)
    out = ws.objects_dir / "detections_2d.parquet"
    write_table(df, out)
    counts = df["class"].value_counts().to_dict()
    # temporal presence per class: which seconds each class appears in
    presence: dict[str, list[int]] = {}
    for cls, grp in df.groupby("class"):
        secs = sorted({int(t) for t in grp["t"]})
        presence[str(cls)] = secs[:60]
    return {
        "available": True,
        "artifact": ws.rel(out),
        "n_detections": len(df),
        "classes": {k: int(v) for k, v in counts.items()},
        "class_present_at_seconds": presence,
        "frame_stride": stride,
    }


# ---------------------------------------------------------------------------
# Open-vocabulary detection (YOLO-World): text-prompted find
# ---------------------------------------------------------------------------

_WORLD_CACHE: dict = {}


def find_objects(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig,
                 queries: list[str], stride_s: float = 1.0) -> dict:
    """Text-prompted open-vocabulary detection (YOLO-World v2, CPU ~0.3 s/frame).
    Lets the agent verify VLM sightings that COCO classes cannot express
    ('blue nitrile glove', 'pipette'). Honest about weakness: small/unusual
    objects score low; treat conf>=0.2 as present, 0.05-0.2 as weak evidence."""
    queries = [str(q)[:60] for q in queries[:8] if str(q).strip()]
    if not queries:
        return {"available": False, "reason": "no queries given"}
    try:
        from ultralytics import YOLO
    except ImportError as e:
        return {"available": False, "reason": f"ultralytics not importable: {e}"}
    weights = cfg.root / "yolov8s-worldv2.pt"
    if not weights.is_file():
        weights = Path("yolov8s-worldv2.pt")  # ultralytics auto-download location
    try:
        if "model" not in _WORLD_CACHE:
            _WORLD_CACHE["model"] = YOLO(str(weights))
        model = _WORLD_CACHE["model"]
        model.set_classes(queries)
    except Exception as e:  # noqa: BLE001 - clip text encoder may be missing
        return {"available": False, "reason": f"YOLO-World unavailable: {e}"}

    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, round(fps * stride_s))
    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            res = model.predict(frame, verbose=False, conf=0.05)[0]
            for b in res.boxes:
                rows.append({"t": idx / fps, "query": res.names[int(b.cls[0])],
                             "conf": float(b.conf[0])})
        idx += 1
    cap.release()
    summary = {}
    for q in queries:
        hits = [r for r in rows if r["query"] == q]
        strong = [r for r in hits if r["conf"] >= 0.2]
        summary[q] = {
            "n_hits": len(hits),
            "max_conf": round(max((r["conf"] for r in hits), default=0.0), 3),
            "present_seconds": sorted({int(r["t"]) for r in strong})[:30],
            "verdict": ("present" if strong else
                        "weak_evidence" if hits else "not_found"),
        }
    import json as _json

    path = ws.objects_dir / "open_vocab_find.json"
    path.write_text(_json.dumps({"queries": queries, "summary": summary},
                                indent=1), encoding="utf-8")
    return {
        "available": True,
        "artifact": ws.rel(path),
        "summary": summary,
        "note": "open-vocabulary detector; 'weak_evidence' (conf 0.05-0.2) "
                "corroborates but does not prove; 'not_found' does NOT prove "
                "absence for small or unusual objects",
    }


# ---------------------------------------------------------------------------
# AI-generation forensics (weak heuristics, honestly labeled)
# ---------------------------------------------------------------------------


def aigen_forensics(video: Path) -> dict:
    """Cheap frame-statistics heuristics for AI-generated video. WEAK signal:
    modern generators (Veo-class) routinely pass. Measures (a) temporal noise
    autocorrelation - camera sensor noise is ~white in time, generated video
    tends to have smoothed/correlated residuals; (b) high-frequency spatial
    energy stability - texture shimmer shows as HF variance spikes."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"available": False, "reason": "cannot open video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, round(fps / 12))
    grays = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            grays.append(cv2.cvtColor(cv2.resize(frame, (192, 108)),
                                      cv2.COLOR_BGR2GRAY).astype(np.float32))
        idx += 1
        if len(grays) >= 240:
            break
    cap.release()
    if len(grays) < 8:
        return {"available": False, "reason": "too few frames"}
    g = np.stack(grays)
    # residual after temporal smoothing ~= noise + fine motion
    resid = g[1:-1] - 0.5 * (g[:-2] + g[2:])
    r = resid.reshape(len(resid), -1)
    # lag-1 autocorrelation of the residual field over time
    num = float(np.mean(np.sum(r[1:] * r[:-1], axis=1)))
    den = float(np.mean(np.sum(r ** 2, axis=1))) + 1e-9
    noise_autocorr = num / den
    # high-frequency spatial energy per frame (Laplacian variance), stability
    hf = np.array([cv2.Laplacian(f, cv2.CV_32F).var() for f in grays])
    hf_cv = float(np.std(hf) / (np.mean(hf) + 1e-9))
    return {
        "available": True,
        "noise_lag1_autocorr": round(noise_autocorr, 4),
        "hf_energy_cv": round(hf_cv, 4),
        "verdict": "inconclusive",
        "reliability": "NOT DISCRIMINATIVE on calibration: measured on 6 real "
                       "+ 5 generated clips these statistics fully overlap "
                       "(real autocorr -0.26..-0.58, generated -0.41..-0.53). "
                       "Use ONLY visual artifact inspection (morphing, "
                       "shimmer, impossible physics) for AI-generation "
                       "assessment; never cite these numbers as evidence.",
    }


# ---------------------------------------------------------------------------
# Object tracking over time (ultralytics ByteTrack) + trajectory descriptors
# ---------------------------------------------------------------------------


def track_objects(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig,
                  stride: int = 3) -> dict:
    """YOLO + ByteTrack: persistent object identities over time. Emits one
    trajectory per track (normalized center positions) with movement
    descriptors - the difference between 'a cup exists' and 'the cup moves
    upward at t=6.2s'."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        return {"available": False, "reason": f"ultralytics not importable: {e}"}

    weights = cfg.root / "yolo11n.pt"
    if not weights.is_file():
        return {"available": False, "reason": f"missing {weights}"}

    import cv2

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              verbose=False, conf=0.3)[0]
            for b in res.boxes:
                if b.id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                rows.append({
                    "t": idx / fps, "frame": idx,
                    "track_id": int(b.id[0]),
                    "class": res.names[int(b.cls[0])],
                    "conf": float(b.conf[0]),
                    "cx_norm": (x1 + x2) / 2 / w, "cy_norm": (y1 + y2) / 2 / h,
                    "w_norm": (x2 - x1) / w, "h_norm": (y2 - y1) / h,
                    "valid": True, "source": SourceTag.estimated.value,
                })
        idx += 1
    cap.release()
    if not rows:
        return {"available": True, "n_tracks": 0,
                "note": "no objects tracked above conf 0.3"}
    df = pd.DataFrame(rows)
    out = ws.objects_dir / "tracks_2d.parquet"
    write_table(df, out)

    tracks = []
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("t")
        if len(grp) < 3:
            continue
        char = characterize_trajectory(grp["t"].to_numpy(),
                                       grp[["cx_norm", "cy_norm"]].to_numpy())
        # size change signals approach/recede or pick-up (box grows/shrinks)
        area = (grp["w_norm"] * grp["h_norm"]).to_numpy()
        tracks.append({
            "track_id": int(tid),
            "class": str(grp["class"].mode()[0]),
            "t_first_s": round(float(grp["t"].min()), 2),
            "t_last_s": round(float(grp["t"].max()), 2),
            "moving": char.get("path_length", 0) > 0.05,  # >5% of frame
            "trajectory": char,
            "size_change_ratio": round(float(area[-3:].mean() /
                                             max(area[:3].mean(), 1e-9)), 2),
        })
    tracks.sort(key=lambda x: -x["trajectory"].get("path_length", 0.0))
    return {
        "available": True,
        "artifact": ws.rel(out),
        "n_tracks": len(tracks),
        "tracks": tracks[:12],
        "units": "positions normalized to frame size; speeds in frame-fractions/s",
        "frame_stride": stride,
        "note": "tracks sorted by movement; 'moving' objects with rising "
                "size_change_ratio may be approaching / being picked up",
    }


# ---------------------------------------------------------------------------
# Quadruped (robot-dog) pose: DeepLabCut SuperAnimal-Quadruped, 39 keypoints
# ---------------------------------------------------------------------------


def _autocorr_period(signal: np.ndarray, sample_dt: float,
                     min_corr: float = 0.3) -> tuple[float | None, float]:
    """First autocorrelation peak of a 1-D signal -> (period_s, strength)."""
    s = np.asarray(signal, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 8 or float(np.std(s)) < 1e-9:
        return None, 0.0
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac /= ac[0] + 1e-12
    for lag in range(2, len(ac) - 1):
        if ac[lag] > min_corr and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            return round(lag * sample_dt, 2), round(float(ac[lag]), 2)
    return None, 0.0


def animal_pose(video: Path, ws: EpisodeWorkspace, cfg: V2RConfig,
                stride: int = 3, conf_thresh: float = 0.5) -> dict:
    """Quadruped keypoint tracking with DeepLabCut's SuperAnimal-Quadruped model
    (39 body parts: nose, ears, spine, four limbs with thigh/knee/paw, tail).

    Use this INSTEAD of the human `motion` (MediaPipe) tool when the subject is
    a four-legged animal (dog, horse, cow, cat, tiger, etc.). It reports, per
    second: keypoint presence/confidence, limb-tip (paw) speed timelines,
    stride periodicity (autocorrelation of paw motion), body displacement, and
    a body-pose read (spine angle + standing/recumbent heuristic).

    Downstream consumer: Unitree Go2 quadruped retargeting (assets/robots/go2).
    The gait signals here are what a Go2 controller needs to imitate locomotion.
    """
    from . import superanimal as SA

    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError as e:
        return {"available": False, "reason": f"torch/timm/opencv not importable: {e}"}

    if not SA.models_available(cfg.root / "assets"):
        det, pose = SA.model_paths(cfg.root / "assets")
        return {"available": False,
                "reason": f"missing SuperAnimal weights ({det.name}, {pose.name}); "
                          "fetch via dlclibrary.download_huggingface_model"}

    import cv2

    assets = cfg.root / "assets"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"available": False, "reason": "cannot open video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

    rows: list[dict] = []
    per_frame: list[dict] = []  # summary geometry per sampled frame
    total = 0
    present = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            total += 1
            bbox, det_score = SA.detect_animal(frame, assets, min_score=0.2)
            if bbox is not None:
                pts = SA.pose_from_crop(frame, bbox, assets)
                good = pts[:, 2] >= conf_thresh
                if int(good.sum()) >= 4:  # need a few reliable joints
                    present += 1
                    t = idx / fps
                    for c, name in enumerate(SA.QUADRUPED_KEYPOINTS):
                        rows.append({
                            "t": t, "frame": idx,
                            "keypoint_idx": c, "keypoint_name": name,
                            "u": float(pts[c, 0]) / w, "v": float(pts[c, 1]) / h,
                            "conf": float(np.clip(pts[c, 2], 0.0, 1.0)),
                            "valid": bool(pts[c, 2] >= conf_thresh),
                            "source": SourceTag.estimated.value,
                        })
                    per_frame.append({"t": t, "pts": pts, "det_score": det_score,
                                      "n_good": int(good.sum())})
        idx += 1
    cap.release()

    duration_s = (idx / max(fps, 1e-6))
    if not rows:
        return {"available": True, "animal_present_ratio": 0.0,
                "n_frames_sampled": total, "duration_s": round(duration_s, 2),
                "note": "SuperAnimal detector found no quadruped above conf 0.2 "
                        "in any sampled frame - subject is likely not a "
                        "four-legged animal (or too small/occluded to track)"}

    df = pd.DataFrame(rows)
    out = ws.root / "animal" / "keypoints_superanimal.parquet"
    write_table(df, out)

    # -- body geometry & gait signals -------------------------------------
    stamps = np.array([f["t"] for f in per_frame])
    idx_kp = SA.KP_INDEX

    def _kp(f, name):
        p = f["pts"][idx_kp[name]]
        return (p[0], p[1]) if p[2] >= conf_thresh else None

    body_len_px = []
    centroids = []
    spine_angles = []
    height_ratio = []  # body vertical extent / body length (recumbent vs upright)
    for f in per_frame:
        nb, tb = _kp(f, "neck_base"), _kp(f, "tail_base")
        blen = None
        if nb and tb:
            blen = float(np.hypot(nb[0] - tb[0], nb[1] - tb[1]))
            body_len_px.append(blen)
            spine_angles.append(abs(float(np.degrees(
                np.arctan2(tb[1] - nb[1], tb[0] - nb[0])))))
        good = f["pts"][f["pts"][:, 2] >= conf_thresh]
        if len(good):
            centroids.append((f["t"], float(good[:, 0].mean()), float(good[:, 1].mean())))
            if blen and blen > 1e-6:
                vext = float(good[:, 1].max() - good[:, 1].min())
                height_ratio.append(vext / blen)
    median_body_len = float(np.median(body_len_px)) if body_len_px else float(max(w, h) * 0.3)

    # paw speed timelines, normalized to body-lengths / second (scale invariant)
    paw_speed_series: dict[str, list[float]] = {}
    all_paw_speed_t, all_paw_speed_v = [], []
    for paw in SA.PAW_KEYPOINTS:
        sub = df[(df["keypoint_name"] == paw) & (df["valid"])].sort_values("t")
        if len(sub) < 3:
            paw_speed_series[paw] = []
            continue
        pt = sub["t"].to_numpy()
        pu = sub["u"].to_numpy() * w
        pv = sub["v"].to_numpy() * h
        dt = np.diff(pt)
        dt = np.where(dt <= 0, 1e-6, dt)
        sp = np.hypot(np.diff(pu), np.diff(pv)) / dt / median_body_len
        paw_speed_series[paw] = _per_second_bins(pt[1:], sp, duration_s)
        all_paw_speed_t.extend(pt[1:].tolist())
        all_paw_speed_v.extend(sp.tolist())

    # stride periodicity: autocorrelation of the aggregate paw-speed signal
    sample_dt = stride / max(fps, 1e-6)
    stride_period_s, stride_strength = None, 0.0
    if all_paw_speed_t:
        order = np.argsort(all_paw_speed_t)
        agg = np.array(all_paw_speed_v)[order]
        # resample onto the uniform sampling grid via the per-second bins signal
        binned = _per_second_bins(np.array(all_paw_speed_t)[order], agg, duration_s)
        stride_period_s, stride_strength = _autocorr_period(np.array(binned), 1.0)
        if stride_period_s is None:  # finer grid if per-second too coarse
            stride_period_s, stride_strength = _autocorr_period(agg, sample_dt)

    # body displacement (locomotion) from centroid path, in body-lengths
    locomotion_bl = 0.0
    body_speed_1s: list[float] = []
    if len(centroids) >= 2:
        ct = np.array([c[0] for c in centroids])
        cxy = np.array([[c[1], c[2]] for c in centroids])
        seg = np.hypot(np.diff(cxy[:, 0]), np.diff(cxy[:, 1])) / median_body_len
        locomotion_bl = float(seg.sum())
        dtc = np.diff(ct); dtc = np.where(dtc <= 0, 1e-6, dtc)
        body_speed_1s = _per_second_bins(ct[1:], seg / dtc, duration_s)

    # per-second keypoint presence and mean confidence
    present_1s = _per_second_bins(
        df["t"].to_numpy(), df["valid"].to_numpy().astype(float),
        duration_s, agg=lambda a: float(np.mean(a)) if len(a) else 0.0)
    conf_1s = _per_second_bins(df[df["valid"]]["t"].to_numpy(),
                               df[df["valid"]]["conf"].to_numpy(), duration_s)

    mean_spine_angle = round(float(np.mean(spine_angles)), 1) if spine_angles else None
    med_height_ratio = float(np.median(height_ratio)) if height_ratio else None
    posture = "unknown"
    if med_height_ratio is not None:
        # upright quadruped: legs stack under a roughly horizontal back, so the
        # vertical extent is a large fraction of body length; lying flattens it
        posture = "standing_or_walking" if med_height_ratio > 0.55 else "recumbent_or_low"

    locomoting = locomotion_bl > 0.5  # moved >half a body length over the clip

    return {
        "available": True,
        "model": "DeepLabCut SuperAnimal-Quadruped (fasterrcnn + resnet50_gn)",
        "artifact": ws.rel(out),
        "animal_present_ratio": round(present / max(total, 1), 3),
        "n_frames_sampled": total,
        "n_keypoints": len(SA.QUADRUPED_KEYPOINTS),
        "duration_s": round(duration_s, 2),
        "mean_keypoint_conf": round(float(df[df["valid"]]["conf"].mean()), 3),
        "keypoint_presence_per_second": present_1s,
        "mean_conf_per_second": conf_1s,
        "paw_speed_bodylengths_s_per_second": paw_speed_series,
        "body_speed_bodylengths_s_per_second": body_speed_1s,
        "locomotion_body_lengths_total": round(locomotion_bl, 2),
        "locomoting": bool(locomoting),
        "stride_period_s": stride_period_s,
        "stride_periodicity_strength": stride_strength,
        "spine_angle_deg_from_horizontal": mean_spine_angle,
        "posture_heuristic": posture,
        "frame_stride": stride,
        "units": "positions normalized to frame; speeds in body-lengths/s "
                 "(scale invariant); spine angle 0=horizontal back",
        "downstream": "Go2 quadruped retargeting (assets/robots/go2)",
        "note": "gait verb (walk/trot/gallop) follows from stride_period_s + "
                "paw-speed amplitude + body_speed; sit/lie_down/stand follow "
                "from posture_heuristic + low locomotion. Cross-check against "
                "what you SEE - posture from 2D keypoints is approximate.",
    }
