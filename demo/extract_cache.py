"""Demo-side pose/track extraction fallback.

Writes parquet files into demo/.cache/ts/ (never touches v2r/ outputs) so the
frontend can show a real skeleton overlay even while the main pipeline's
timeseries stage is still being finished. Run: python demo/extract_cache.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import pandas as pd

DEMO = Path(__file__).resolve().parent
V2R = DEMO.parent / "v2r"
RAW = V2R / "data" / "raw"
OUT = DEMO / ".cache" / "ts"

sys.path.insert(0, str(V2R / "src"))
from v2r.timeseries.schema import HUMAN_JOINT_NAMES  # noqa: E402

MODEL = DEMO / ".cache" / "pose_landmarker_lite.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


def extract_human(video: Path, out: Path) -> int:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL.is_file():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, frame_i, detected = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_i / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        if result.pose_landmarks:
            detected += 1
            for j, name in enumerate(HUMAN_JOINT_NAMES):
                p = result.pose_landmarks[0][j]
                vis = float(getattr(p, "visibility", 1.0) or 1.0)
                rows.append(dict(t=t, frame=frame_i, subject="human", entity_id=0,
                                 joint_name=name, joint_idx=j,
                                 x=p.x, y=p.y, z=getattr(p, "z", 0.0),
                                 conf=vis, valid=vis >= 0.4, source="estimated"))
        frame_i += 1
    cap.release()
    landmarker.close()
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
    print(f"  {video.name}: {detected}/{frame_i} frames with pose -> {out if rows else 'skipped'}")
    return detected


def extract_animal(video: Path, out: Path) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"  {video.name}: ultralytics not installed; skipping animal track")
        return 0
    ANIMALS = {14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
               19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    cap.release()
    model = YOLO("yolo11n.pt")
    rows, prev = [], {}
    frame_i = 0
    for r in model.track(source=str(video), stream=True, persist=True,
                         conf=0.3, verbose=False, classes=list(ANIMALS)):
        t = frame_i / fps
        for box in (r.boxes or []):
            cls = int(box.cls[0])
            if cls not in ANIMALS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            tid = int(box.id[0]) if box.id is not None else 0
            vx, vy = 0.0, 0.0
            if tid in prev:
                vx, vy = (cx - prev[tid][0]) * fps, (cy - prev[tid][1]) * fps
            prev[tid] = (cx, cy)
            rows.append(dict(t=t, frame=frame_i, subject="animal", entity_id=tid,
                             class_name=ANIMALS[cls], class_id=cls,
                             cx=cx, cy=cy, w=(x2 - x1) / w, h=(y2 - y1) / h,
                             vx=vx, vy=vy, conf=float(box.conf[0]),
                             valid=True, source="estimated"))
        frame_i += 1
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
    print(f"  {video.name}: {len(rows)} detections -> {out if rows else 'skipped'}")
    return len(rows)


def main() -> None:
    import json
    manifest_path = RAW / "import_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for src in manifest["sources"]:
            for vp in src["videos"]:
                video = Path(vp)
                if not video.is_file():
                    continue
                if src["subject"] == "human":
                    extract_human(video, OUT / "human" / f"{video.stem}.parquet")
                else:
                    extract_animal(video, OUT / "animal" / f"{video.stem}.parquet")
    # bundled CC-licensed sample clips (demo/.cache/demo_media)
    dm = DEMO / ".cache" / "demo_media"
    for mp4 in sorted(dm.glob("*.mp4")) if dm.is_dir() else []:
        subject = "human" if mp4.stem.startswith("human") else "animal"
        out = OUT / subject / f"{mp4.stem}.parquet"
        if out.is_file():
            continue
        if subject == "human":
            extract_human(mp4, out)
        else:
            extract_animal(mp4, out)


if __name__ == "__main__":
    main()
