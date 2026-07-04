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
    return {
        "available": True,
        "artifact": ws.rel(out),
        "person_present_ratio": round(present / max(total, 1), 3),
        "mean_conf": round(float(df["conf"].mean()), 3),
        "n_frames": total,
        "right_wrist_path_length_m": round(motion, 3),
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
    return {
        "available": True,
        "artifact": ws.rel(out),
        "hands_present_ratio": round(frames_with_hands / max(total, 1), 3),
        "n_hand_observations": int(len(df) / 25),
        "aperture_m": {"min": round(float(ap.min()), 4), "max": round(float(ap.max()), 4),
                       "mean": round(float(ap.mean()), 4)},
        "note": "positions are hand-centered metric estimates (MediaPipe world "
                "landmarks); orientations identity (not observable), conf reduced",
    }


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
    return {
        "available": True,
        "artifact": ws.rel(out),
        "n_detections": len(df),
        "classes": {k: int(v) for k, v in counts.items()},
        "frame_stride": stride,
    }
