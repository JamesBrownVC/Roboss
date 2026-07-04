"""Convert raw videos into per-frame timeseries for AI training."""

from __future__ import annotations

HUMAN_JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

TIMESERIES_COLUMNS = [
    "t", "frame", "subject", "entity_id", "joint_name", "joint_idx",
    "x", "y", "z", "conf", "valid", "source",
]

ANIMAL_TRACK_COLUMNS = [
    "t", "frame", "subject", "entity_id", "class_name", "class_id",
    "cx", "cy", "w", "h", "vx", "vy", "conf", "valid", "source",
]
