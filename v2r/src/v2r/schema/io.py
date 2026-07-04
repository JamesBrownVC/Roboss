"""Parquet / NPZ / JSON / depth-PNG IO with contract enforcement.

Every kinematic Parquet table carries, per logical quantity: value columns,
``conf`` (float 0-1), ``valid`` (bool), ``source`` (SourceTag enum value).
Writers here refuse to emit tables that violate that contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Type, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .models import SourceTag
from .rotations import mat44_flatten_rowmajor, mat44_unflatten_rowmajor

SOURCE_VALUES = {s.value for s in SourceTag}

# ---------------------------------------------------------------------------
# canonical table layouts (column names are part of the interchange contract)
# ---------------------------------------------------------------------------

MAT44_COLS = [f"T_wc_{i}{j}" for i in range(4) for j in range(4)]

POSES_COLUMNS = ["t", "frame", *MAT44_COLS, "conf", "valid", "source"]

# EgoDex convention: 25 SE(3) joints per hand (ARKit skeleton:
# wrist + 4 thumb + 5 per finger x 4 fingers), long format.
EGODEX_HAND_JOINTS = [
    "wrist",
    "thumbKnuckle", "thumbIntermediateBase", "thumbIntermediateTip", "thumbTip",
    "indexFingerMetacarpal", "indexFingerKnuckle", "indexFingerIntermediateBase",
    "indexFingerIntermediateTip", "indexFingerTip",
    "middleFingerMetacarpal", "middleFingerKnuckle", "middleFingerIntermediateBase",
    "middleFingerIntermediateTip", "middleFingerTip",
    "ringFingerMetacarpal", "ringFingerKnuckle", "ringFingerIntermediateBase",
    "ringFingerIntermediateTip", "ringFingerTip",
    "littleFingerMetacarpal", "littleFingerKnuckle", "littleFingerIntermediateBase",
    "littleFingerIntermediateTip", "littleFingerTip",
]
assert len(EGODEX_HAND_JOINTS) == 25

# Main SMPL-X body joints; smplx.npz always carries joints_world (T, 22, 3)
# for these (real mode: body-model forward pass; synthetic mode: synthesized).
SMPLX_MAIN_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
assert len(SMPLX_MAIN_JOINTS) == 22

HANDS_COLUMNS = [
    "t", "frame", "hand", "joint_idx", "joint_name",
    "px", "py", "pz", "qw", "qx", "qy", "qz",
    "conf", "valid", "source", "interpolated",
]

TRACKS_COLUMNS = [
    "t", "frame", "object_id",
    "px", "py", "pz", "qw", "qx", "qy", "qz",
    "conf", "valid", "source",
]

CONTACTS_COLUMNS = [
    "t", "frame", "hand", "object_id",
    "contact", "min_dist_m", "penetration_m",
    "conf", "valid", "source",
]

EE_COLUMNS = [
    "t", "frame", "hand",
    "px", "py", "pz", "qw", "qx", "qy", "qz",
    "gripper_aperture_m",
    "conf", "valid", "source",
]


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------------------
# tabular
# ---------------------------------------------------------------------------


def validate_kinematic_table(df: pd.DataFrame, required_columns: Iterable[str] | None = None) -> None:
    """Enforce the conf/valid/source contract; raise SchemaError on violation."""
    if required_columns is not None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise SchemaError(f"missing columns: {missing}")
    for col in df.columns:
        if col == "conf" or col.endswith("_conf"):
            vals = df[col].to_numpy(dtype=np.float64)
            finite = np.isfinite(vals)
            if np.any((vals[finite] < 0.0) | (vals[finite] > 1.0)):
                raise SchemaError(f"column {col!r} outside [0,1]")
            if not finite.all():
                raise SchemaError(f"column {col!r} contains non-finite values")
        if col == "source" or col.endswith("_source"):
            bad = set(df[col].astype(str).unique()) - SOURCE_VALUES
            if bad:
                raise SchemaError(f"column {col!r} has invalid source tags: {bad}")
    if "t" in df.columns and len(df) > 1:
        # per-group monotonicity is checked by stages; global t must be sorted
        # within each (hand/object/joint) group, so only check when unique keys
        if df["t"].is_monotonic_increasing or _grouped_monotonic(df):
            pass
        else:
            raise SchemaError("timestamps not monotonic (globally or per group)")


def _grouped_monotonic(df: pd.DataFrame) -> bool:
    keys = [k for k in ("hand", "object_id", "joint_idx", "entity_id") if k in df.columns]
    if not keys:
        return False
    return bool(df.groupby(keys, sort=False)["t"].apply(lambda s: s.is_monotonic_increasing).all())


def write_table(df: pd.DataFrame, path: Path | str, required_columns: Iterable[str] | None = None) -> Path:
    path = Path(path)
    validate_kinematic_table(df, required_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_table(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def poses_df(t: np.ndarray, frames: np.ndarray, T_world_cam: np.ndarray,
             conf: np.ndarray, valid: np.ndarray, source: str | np.ndarray) -> pd.DataFrame:
    """Build the canonical geometry/poses.parquet frame from arrays."""
    flat = mat44_flatten_rowmajor(np.asarray(T_world_cam, dtype=np.float64))
    data = {"t": np.asarray(t, dtype=np.float64), "frame": np.asarray(frames, dtype=np.int64)}
    for k, col in enumerate(MAT44_COLS):
        data[col] = flat[:, k]
    data["conf"] = np.asarray(conf, dtype=np.float64)
    data["valid"] = np.asarray(valid, dtype=bool)
    data["source"] = source if not isinstance(source, str) else np.full(len(flat), source)
    return pd.DataFrame(data, columns=POSES_COLUMNS)


def poses_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Inverse of poses_df: returns t, frame, T_world_cam (N,4,4), conf, valid, source."""
    flat = df[MAT44_COLS].to_numpy(dtype=np.float64)
    return {
        "t": df["t"].to_numpy(dtype=np.float64),
        "frame": df["frame"].to_numpy(dtype=np.int64),
        "T_world_cam": mat44_unflatten_rowmajor(flat),
        "conf": df["conf"].to_numpy(dtype=np.float64),
        "valid": df["valid"].to_numpy(dtype=bool),
        "source": df["source"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# npz / json
# ---------------------------------------------------------------------------


def write_npz(path: Path | str, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


M = TypeVar("M", bound=BaseModel)


def write_json_model(path: Path | str, model: BaseModel) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    return path


def read_json_model(path: Path | str, cls: Type[M]) -> M:
    return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# depth PNG (16-bit, value = depth_m * depth_scale; default mm)
# ---------------------------------------------------------------------------


def write_depth_png(path: Path | str, depth_m: np.ndarray, depth_scale: float = 1000.0) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_m, dtype=np.float64) * depth_scale
    depth = np.clip(np.nan_to_num(depth, nan=0.0), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", depth)
    if not ok:
        raise IOError(f"failed to encode depth png: {path}")
    buf.tofile(str(path))  # tofile handles unicode paths on Windows
    return path


def read_depth_png(path: Path | str, depth_scale: float = 1000.0) -> np.ndarray:
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"failed to decode depth png: {path}")
    return img.astype(np.float64) / depth_scale
