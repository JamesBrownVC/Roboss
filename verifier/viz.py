"""Annotated demo video: skeletons, object boxes, violation banner + timeline."""

from __future__ import annotations

import cv2
import numpy as np

from .tracks import Evidence, Violation

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),      # torso
]

GREEN = (80, 200, 80)
RED = (60, 60, 230)
YELLOW = (60, 200, 230)
WHITE = (240, 240, 240)


def render_annotated_video(video_path: str, evidence: Evidence,
                           violations: list[Violation], out_path: str) -> None:
    bad_frames: dict[int, list[str]] = {}
    for v in violations:
        for f in v.frames:
            bad_frames.setdefault(f, []).append(v.type)

    # frame -> list of (track, index_in_track)
    per_frame: dict[int, list[tuple]] = {}
    for tr in evidence.all_tracks:
        for i, f in enumerate(tr.frames):
            per_frame.setdefault(f, []).append((tr, i))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or evidence.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    total = max(evidence.n_frames, 1)

    frame_idx = 0
    while frame_idx < evidence.n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        for tr, i in per_frame.get(frame_idx, []):
            x1, y1, x2, y2 = (int(v) for v in tr.boxes[i])
            color = GREEN if tr.is_person else YELLOW
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{tr.label}#{tr.track_id}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if tr.is_person and i < len(tr.keypoints):
                k = tr.keypoints[i]
                for a, b in SKELETON:
                    if k[a, 2] > 0.5 and k[b, 2] > 0.5:
                        cv2.line(frame, (int(k[a, 0]), int(k[a, 1])),
                                 (int(k[b, 0]), int(k[b, 1])), GREEN, 2)

        if frame_idx in bad_frames:
            types = ", ".join(sorted(set(bad_frames[frame_idx])))
            cv2.rectangle(frame, (0, 0), (w, 34), RED, -1)
            cv2.putText(frame, f"VIOLATION: {types}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

        # timeline bar: red ticks at suspicious frames
        bar_y = h - 14
        cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 8), (90, 90, 90), -1)
        for f in bad_frames:
            x = 10 + int((w - 20) * f / total)
            cv2.rectangle(frame, (x, bar_y), (x + 2, bar_y + 8), RED, -1)
        x_now = 10 + int((w - 20) * frame_idx / total)
        cv2.rectangle(frame, (x_now, bar_y - 3), (x_now + 2, bar_y + 11), WHITE, -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
