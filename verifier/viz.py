"""Annotated demo video: skeletons, object boxes, violation banner + timeline.

Detector tracks are sparse (a box only exists on frames where detection fired)
and jittery. To make the overlay uniform throughout the clip we densify each
track: gaps up to ``max_gap`` frames are linearly interpolated and the anchor
coordinates are temporally smoothed, so boxes/skeletons persist smoothly on
every frame instead of blinking and jumping.
"""

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


def _smooth(values: np.ndarray, win: int) -> np.ndarray:
    """Centered moving-average along axis 0 (edge-padded), per column."""
    v = np.asarray(values, dtype=np.float32)
    if win <= 1 or v.shape[0] < 3:
        return v
    # keep window odd and no larger than the series
    win = min(win, v.shape[0] if v.shape[0] % 2 else v.shape[0] - 1)
    if win < 3:
        return v
    pad = win // 2
    kernel = np.ones(win, dtype=np.float32) / win
    padded = np.pad(v, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(v)
    for c in range(v.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def _densify(frames: np.ndarray, values: np.ndarray, max_gap: int) -> dict[int, np.ndarray]:
    """Map every frame in each track's active range to an (interpolated) value.

    Anchors (real detections) are kept as-is; gaps of ``<= max_gap`` frames
    between consecutive anchors are linearly interpolated. Larger gaps (the
    object genuinely left) are left empty.
    """
    out: dict[int, np.ndarray] = {}
    n = len(frames)
    for i in range(n):
        out[int(frames[i])] = values[i]
    for i in range(n - 1):
        f0, f1 = int(frames[i]), int(frames[i + 1])
        gap = f1 - f0
        if gap <= 1 or gap > max_gap:
            continue
        v0, v1 = values[i], values[i + 1]
        for f in range(f0 + 1, f1):
            t = (f - f0) / gap
            out[f] = v0 * (1.0 - t) + v1 * t
    return out


def render_annotated_video(video_path: str, evidence: Evidence,
                           violations: list[Violation], out_path: str,
                           *, box_smooth_win: int = 5, kpt_smooth_win: int = 5) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or evidence.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    total = max(evidence.n_frames, 1)

    # Bridge detection gaps up to ~0.4s; hold the violation banner ~0.3s so it
    # does not strobe on single-frame violations.
    max_gap = max(2, int(round(fps * 0.4)))
    hold = max(1, int(round(fps * 0.3)))

    # frame -> list of per-track draw records (densified + smoothed)
    per_frame: dict[int, list[dict]] = {}
    for tr in evidence.all_tracks:
        if len(tr) == 0:
            continue
        fr = np.asarray(tr.frames, dtype=np.int64)
        boxes = _smooth(np.asarray(tr.boxes, dtype=np.float32).reshape(-1, 4), box_smooth_win)
        box_map = _densify(fr, boxes, max_gap)

        kpt_map: dict[int, np.ndarray] = {}
        if tr.is_person and tr.keypoints:
            flat = np.stack(tr.keypoints, axis=0).reshape(len(tr), -1)  # (N, 17*3)
            flat = _smooth(flat, kpt_smooth_win)
            kmap = _densify(fr, flat, max_gap)
            kpt_map = {f: v.reshape(-1, 3) for f, v in kmap.items()}

        for f, box in box_map.items():
            per_frame.setdefault(f, []).append({
                "box": box,
                "kpts": kpt_map.get(f),
                "is_person": tr.is_person,
                "label": tr.label,
                "track_id": tr.track_id,
            })

    # true violation frames (for timeline ticks) + a held banner window
    bad_frames: dict[int, list[str]] = {}
    for v in violations:
        for f in v.frames:
            bad_frames.setdefault(f, []).append(v.type)
    banner: dict[int, set[str]] = {}
    for f, types in bad_frames.items():
        for df in range(hold + 1):
            banner.setdefault(f + df, set()).update(types)

    frame_idx = 0
    while frame_idx < evidence.n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        for d in per_frame.get(frame_idx, []):
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            color = GREEN if d["is_person"] else YELLOW
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d['label']}#{d['track_id']}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            k = d["kpts"]
            if d["is_person"] and k is not None:
                for a, b in SKELETON:
                    if k[a, 2] > 0.5 and k[b, 2] > 0.5:
                        cv2.line(frame, (int(k[a, 0]), int(k[a, 1])),
                                 (int(k[b, 0]), int(k[b, 1])), GREEN, 2)

        if frame_idx in banner:
            types = ", ".join(sorted(banner[frame_idx]))
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
