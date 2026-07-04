"""Pose and motion timeseries extraction from video."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import cv2
import numpy as np
import pandas as pd

from ..import_.catalog import load_catalog
from ..schema.io import write_table
from ..schema.timeline import canonical_timestamps, resample_linear
from .schema import ANIMAL_TRACK_COLUMNS, HUMAN_JOINT_NAMES, TIMESERIES_COLUMNS


@dataclass
class ExtractResult:
    video_path: Path
    subject: Literal["human", "animal"]
    parquet_path: Path
    n_frames: int
    n_entities: int
    errors: list[str] = field(default_factory=list)


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def extract_human_mediapipe(
    video_path: Path,
    out_path: Path,
    canonical_hz: float = 30.0,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    log: Callable[[str], None] = print,
) -> ExtractResult:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    fps, n_frames, w, h = _probe_video(video_path)
    _ensure_mediapipe_model()

    model_path = _mediapipe_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    rows: list[dict] = []
    cap = cv2.VideoCapture(str(video_path))
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_i / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(t * 1000))
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            for j, name in enumerate(HUMAN_JOINT_NAMES):
                if j >= len(lm):
                    break
                p = lm[j]
                vis = float(getattr(p, "visibility", 1.0) or 1.0)
                rows.append({
                    "t": t, "frame": frame_i, "subject": "human", "entity_id": 0,
                    "joint_name": name, "joint_idx": j,
                    "x": p.x, "y": p.y, "z": getattr(p, "z", 0.0),
                    "conf": vis, "valid": vis >= min_tracking_confidence,
                    "source": "estimated",
                })
        else:
            for j, name in enumerate(HUMAN_JOINT_NAMES):
                rows.append({
                    "t": t, "frame": frame_i, "subject": "human", "entity_id": 0,
                    "joint_name": name, "joint_idx": j,
                    "x": np.nan, "y": np.nan, "z": np.nan,
                    "conf": 0.0, "valid": False, "source": "estimated",
                })
        frame_i += 1
    cap.release()
    landmarker.close()

    df = pd.DataFrame(rows)
    df = _resample_pose_table(df, canonical_hz, fps, n_frames / fps if fps else 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(df, out_path, required_columns=TIMESERIES_COLUMNS)
    log(f"  human pose: {video_path.name} -> {out_path.name} ({frame_i} frames)")
    return ExtractResult(video_path, "human", out_path, frame_i, 1)


def extract_animal_yolo_track(
    video_path: Path,
    out_path: Path,
    canonical_hz: float = 30.0,
    yolo_weights: str = "yolo11n.pt",
    min_confidence: float = 0.3,
    log: Callable[[str], None] = print,
) -> ExtractResult:
    from ultralytics import YOLO

    # COCO animal class ids: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    COCO_NAMES = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
        19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    }

    fps, n_frames, w, h = _probe_video(video_path)
    model = YOLO(yolo_weights)

    rows: list[dict] = []
    prev: dict[int, tuple[float, float]] = {}

    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        conf=min_confidence,
        verbose=False,
        classes=list(ANIMAL_IDS),
    )

    frame_i = 0
    entity_ids: set[int] = set()
    for r in results:
        t = frame_i / fps
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            frame_i += 1
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in ANIMAL_IDS:
                continue
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            track_id = int(box.id[0]) if box.id is not None else frame_i
            entity_ids.add(track_id)
            vx, vy = 0.0, 0.0
            if track_id in prev:
                pcx, pcy = prev[track_id]
                vx = (cx - pcx) * fps
                vy = (cy - pcy) * fps
            prev[track_id] = (cx, cy)
            rows.append({
                "t": t, "frame": frame_i, "subject": "animal", "entity_id": track_id,
                "class_name": COCO_NAMES.get(cls_id, str(cls_id)), "class_id": cls_id,
                "cx": cx, "cy": cy, "w": bw, "h": bh, "vx": vx, "vy": vy,
                "conf": conf, "valid": conf >= min_confidence, "source": "estimated",
            })
        frame_i += 1

    if not rows:
        log(f"  warning: no animals detected in {video_path.name}; writing empty track")
        rows.append({
            "t": 0.0, "frame": 0, "subject": "animal", "entity_id": -1,
            "class_name": "none", "class_id": -1,
            "cx": np.nan, "cy": np.nan, "w": np.nan, "h": np.nan,
            "vx": np.nan, "vy": np.nan, "conf": 0.0, "valid": False, "source": "estimated",
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(df, out_path, required_columns=ANIMAL_TRACK_COLUMNS)
    log(f"  animal track: {video_path.name} -> {out_path.name} ({frame_i} frames, {len(entity_ids)} tracks)")
    return ExtractResult(video_path, "animal", out_path, frame_i, len(entity_ids))


def _resample_pose_table(df: pd.DataFrame, hz: float, src_fps: float, duration_s: float) -> pd.DataFrame:
    if df.empty or duration_s <= 0:
        return df
    t_dst = canonical_timestamps(duration_s, hz)
    parts = []
    for (entity_id, joint_idx), grp in df.groupby(["entity_id", "joint_idx"], sort=False):
        grp = grp.sort_values("t").drop_duplicates(subset=["t"], keep="last")
        t_src = grp["t"].to_numpy()
        x, _, v_x = resample_linear(t_src, grp["x"].to_numpy(), t_dst)
        y, _, v_y = resample_linear(t_src, grp["y"].to_numpy(), t_dst)
        z, _, v_z = resample_linear(t_src, grp["z"].to_numpy(), t_dst)
        conf, _, v_c = resample_linear(t_src, grp["conf"].to_numpy(), t_dst)
        valid = v_x & v_y & v_z & v_c & (conf >= 0.5)
        parts.append(pd.DataFrame({
            "t": t_dst, "frame": np.arange(len(t_dst)),
            "subject": "human", "entity_id": entity_id,
            "joint_name": grp["joint_name"].iloc[0], "joint_idx": joint_idx,
            "x": x, "y": y, "z": z, "conf": conf, "valid": valid, "source": "estimated",
        }))
    return pd.concat(parts, ignore_index=True) if parts else df


def _mediapipe_model_path() -> Path:
    p = Path(__file__).resolve().parents[3] / "assets" / "models" / "pose_landmarker_lite.task"
    if not p.is_file():
        raise FileNotFoundError(f"MediaPipe model missing: {p}")
    return p


def _ensure_mediapipe_model() -> Path:
    dest = Path(__file__).resolve().parents[3] / "assets" / "models" / "pose_landmarker_lite.task"
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    return dest


def extract_video(
    video_path: Path,
    subject: Literal["human", "animal"],
    out_dir: Path,
    extraction_cfg: dict,
    log: Callable[[str], None] = print,
) -> ExtractResult:
    stem = video_path.stem
    if subject == "human":
        return extract_human_mediapipe(
            video_path,
            out_dir / "human" / f"{stem}.parquet",
            canonical_hz=extraction_cfg.get("canonical_hz", 30.0),
            min_detection_confidence=extraction_cfg.get("human", {}).get("min_detection_confidence", 0.5),
            min_tracking_confidence=extraction_cfg.get("human", {}).get("min_tracking_confidence", 0.5),
            log=log,
        )
    return extract_animal_yolo_track(
        video_path,
        out_dir / "animal" / f"{stem}.parquet",
        canonical_hz=extraction_cfg.get("canonical_hz", 30.0),
        yolo_weights=extraction_cfg.get("animal", {}).get("yolo_weights", "yolo11n.pt"),
        min_confidence=extraction_cfg.get("animal", {}).get("min_confidence", 0.3),
        log=log,
    )


def extract_all(
    root: Path,
    data_root: Optional[Path] = None,
    subject: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> tuple[list[ExtractResult], Path]:
    catalog = load_catalog(root)
    if data_root is None:
        data_root = Path(catalog.data_root)
        if not data_root.is_absolute():
            data_root = root / data_root

    manifest_path = data_root / "import_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run import first: {manifest_path} not found")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = root / "data" / "timeseries"
    extraction_cfg = catalog.extraction.model_dump()

    results: list[ExtractResult] = []
    for entry in manifest["sources"]:
        subj = entry["subject"]
        if subject and subj != subject:
            continue
        for vpath in entry["videos"]:
            vp = Path(vpath)
            if not vp.is_file():
                log(f"  skip missing {vp}")
                continue
            try:
                results.append(extract_video(vp, subj, out_root, extraction_cfg, log=log))
            except Exception as e:
                log(f"  ERROR {vp.name}: {e}")
                results.append(ExtractResult(vp, subj, out_root / subj / f"{vp.stem}.parquet", 0, 0, [str(e)]))

    training_manifest = {
        "n_videos": len(results),
        "n_human": sum(1 for r in results if r.subject == "human"),
        "n_animal": sum(1 for r in results if r.subject == "animal"),
        "episodes": [
            {
                "video": str(r.video_path),
                "subject": r.subject,
                "timeseries": str(r.parquet_path),
                "n_frames": r.n_frames,
                "n_entities": r.n_entities,
                "errors": r.errors,
            }
            for r in results
        ],
    }
    train_dir = root / "data" / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "manifest.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    log(f"Training manifest: {train_dir / 'manifest.json'} ({len(results)} episodes)")
    return results, out_root
