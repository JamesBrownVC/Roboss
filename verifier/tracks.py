"""Data structures shared between extraction and checks.

Everything downstream of extraction works on these plain
numpy containers, so the physics checks are testable without
any model or video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 keypoint indices used by the checks
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

BONES: list[tuple[int, int, str]] = [
    (L_SHOULDER, L_ELBOW, "left upper arm"),
    (L_ELBOW, L_WRIST, "left forearm"),
    (R_SHOULDER, R_ELBOW, "right upper arm"),
    (R_ELBOW, R_WRIST, "right forearm"),
    (L_HIP, L_KNEE, "left thigh"),
    (L_KNEE, L_ANKLE, "left shin"),
    (R_HIP, R_KNEE, "right thigh"),
    (R_KNEE, R_ANKLE, "right shin"),
    (L_SHOULDER, R_SHOULDER, "shoulder girdle"),
    (L_HIP, R_HIP, "pelvis"),
]


@dataclass
class Track:
    """One tracked entity (person or object) across the video."""

    track_id: int
    label: str                      # class name, e.g. "person", "box"
    is_person: bool
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # xyxy, px
    keypoints: list[np.ndarray] = field(default_factory=list)  # (17,3) x,y,conf per frame

    def add(self, frame: int, box, kpts: np.ndarray | None = None) -> None:
        self.frames.append(frame)
        self.boxes.append(tuple(float(v) for v in box))
        if kpts is not None:
            self.keypoints.append(np.asarray(kpts, dtype=np.float32))

    @property
    def frames_arr(self) -> np.ndarray:
        return np.asarray(self.frames, dtype=np.int64)

    @property
    def boxes_arr(self) -> np.ndarray:
        return np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)

    @property
    def centers(self) -> np.ndarray:
        b = self.boxes_arr
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)

    @property
    def kpts_arr(self) -> np.ndarray | None:
        if not self.keypoints:
            return None
        return np.stack(self.keypoints, axis=0)  # (N,17,3)

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class Evidence:
    """Everything extracted from one video."""

    video_path: str
    fps: float
    width: int
    height: int
    n_frames: int
    person_tracks: list[Track] = field(default_factory=list)
    object_tracks: list[Track] = field(default_factory=list)

    @property
    def diag(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def all_tracks(self) -> list[Track]:
        return self.person_tracks + self.object_tracks


@dataclass
class Violation:
    """One detected physical inconsistency."""

    type: str
    severity: float                 # 0..1
    frames: list[int]
    reason: str
    track_id: int | None = None
    label: str | None = None
    gate: str = "formal"            # "formal" (rule engine) | "semantic" (VLM)

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "severity": round(float(self.severity), 2),
            "frames": [int(f) for f in self.frames],
            "reason": self.reason,
            "gate": self.gate,
        }
        if self.label is not None:
            d["entity"] = f"{self.label}#{self.track_id}"
        return d
