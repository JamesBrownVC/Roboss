"""Build unified NPZ training bundles from extracted timeseries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .schema import HUMAN_JOINT_NAMES


def build_training_npz(
    root: Path,
    log: Callable[[str], None] = print,
) -> Path:
    """Flatten parquet timeseries into per-episode NPZ arrays for ML training."""
    manifest_path = root / "data" / "training" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run extract-timeseries first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = root / "data" / "training" / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    for ep in manifest["episodes"]:
        pq = Path(ep["timeseries"])
        if not pq.is_file():
            continue
        df = pd.read_parquet(pq)
        subject = ep["subject"]
        stem = pq.stem
        npz_path = out_dir / f"{subject}_{stem}.npz"

        if subject == "human":
            # (T, J, 3) positions + (T, J) conf
            joints = sorted(df["joint_idx"].unique())
            t_vals = np.sort(df["t"].unique())
            t_index = {float(t): i for i, t in enumerate(t_vals)}
            T, J = len(t_vals), len(joints)
            pos = np.full((T, J, 3), np.nan, dtype=np.float32)
            conf = np.zeros((T, J), dtype=np.float32)
            valid = np.zeros((T, J), dtype=bool)
            jmap = {j: i for i, j in enumerate(joints)}
            for _, row in df.iterrows():
                ti = t_index.get(float(row["t"]))
                if ti is None:
                    ti = int(row["frame"])
                if ti < 0 or ti >= T:
                    continue
                ji = jmap[row["joint_idx"]]
                pos[ti, ji] = [row["x"], row["y"], row["z"]]
                conf[ti, ji] = row["conf"]
                valid[ti, ji] = row["valid"]
            np.savez_compressed(
                npz_path,
                subject=subject,
                video_path=str(ep["video"]),
                t=np.array(t_vals, dtype=np.float32),
                joint_names=np.array([HUMAN_JOINT_NAMES[j] if j < len(HUMAN_JOINT_NAMES) else str(j) for j in joints]),
                positions=pos,
                confidence=conf,
                valid=valid,
            )
        else:
            # animal: (T, E, 6) bbox+velocity per entity
            entities = sorted(df["entity_id"].unique())
            t_vals = np.sort(df["t"].unique())
            t_index = {float(t): i for i, t in enumerate(t_vals)}
            T, E = len(t_vals), max(len(entities), 1)
            bbox = np.full((T, E, 6), np.nan, dtype=np.float32)
            conf = np.zeros((T, E), dtype=np.float32)
            emap = {e: i for i, e in enumerate(entities)}
            for _, row in df.iterrows():
                if row["entity_id"] not in emap:
                    continue
                ti = t_index.get(float(row["t"]))
                if ti is None:
                    ti = int(row["frame"])
                if ti < 0 or ti >= T:
                    continue
                ei = emap[row["entity_id"]]
                bbox[ti, ei] = [row["cx"], row["cy"], row["w"], row["h"], row["vx"], row["vy"]]
                conf[ti, ei] = row["conf"]
            np.savez_compressed(
                npz_path,
                subject=subject,
                video_path=str(ep["video"]),
                t=np.array(t_vals, dtype=np.float32),
                entity_ids=np.array(entities, dtype=np.int32),
                bbox_motion=bbox,
                confidence=conf,
            )

        index.append({"npz": str(npz_path), "subject": subject, "video": ep["video"]})
        log(f"  NPZ {npz_path.name}")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    log(f"Built {len(index)} training NPZ files in {out_dir}")
    return out_dir
