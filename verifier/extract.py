"""Video -> Evidence: pose estimation + object detection with tracking.

Uses two Ultralytics YOLO11 models per frame:
- yolo11n-pose  -> person keypoints + person tracks
- yolo11n       -> non-person object tracks (box, chair, bottle, ...)

Both run in streaming track mode with persistent IDs (ByteTrack).
Model weights are downloaded automatically on first run.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Thresholds
from .tracks import Evidence, Track


def extract_evidence(video_path: str,
                     th: Thresholds,
                     pose_weights: str = "yolo11n-pose.pt",
                     det_weights: str = "yolo11n.pt",
                     device: str | None = None,
                     progress: bool = True) -> Evidence:
    from ultralytics import YOLO  # heavy import, keep local

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_model = YOLO(pose_weights)
    det_model = YOLO(det_weights)
    det_names = det_model.names

    persons: dict[int, Track] = {}
    objects: dict[int, Track] = {}
    frame_idx = 0

    while frame_idx < th.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # --- persons with keypoints ---
        res = pose_model.track(frame, persist=True, conf=th.det_conf,
                               verbose=False, device=device)[0]
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            kpts = (res.keypoints.data.cpu().numpy()
                    if res.keypoints is not None else None)
            for j, tid in enumerate(ids):
                tr = persons.setdefault(
                    tid, Track(track_id=tid, label="person", is_person=True))
                tr.add(frame_idx, boxes[j],
                       kpts[j] if kpts is not None else np.zeros((17, 3)))

        # --- non-person objects ---
        res = det_model.track(frame, persist=True, conf=th.det_conf,
                              verbose=False, device=device)[0]
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.int().tolist()
            for j, tid in enumerate(ids):
                label = det_names[classes[j]]
                if label == "person":
                    continue
                tr = objects.setdefault(
                    tid, Track(track_id=tid, label=label, is_person=False))
                tr.add(frame_idx, boxes[j])

        frame_idx += 1
        if progress and frame_idx % 30 == 0:
            print(f"  processed {frame_idx} frames...")

    cap.release()

    # Drop one-frame flickers — they are detector noise, not evidence.
    person_tracks = [t for t in persons.values() if len(t) >= 3]
    object_tracks = [t for t in objects.values() if len(t) >= 3]

    return Evidence(
        video_path=video_path, fps=float(fps), width=width, height=height,
        n_frames=frame_idx,
        person_tracks=person_tracks, object_tracks=object_tracks,
    )
