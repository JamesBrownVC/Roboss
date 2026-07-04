"""Rerun snapshot per episode (master prompt 6.J validation step).

Writes export/episode.rrd with the camera trajectory, body joints, hand
joints and object tracks so the episode can be inspected in the rerun viewer
(`rerun export/episode.rrd`). Guarded: returns None when rerun-sdk is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..schema.io import SMPLX_MAIN_JOINTS, read_npz, read_table
from ..schema.workspace import EpisodeWorkspace


def _set_time(rr, frame: int, t: float) -> None:
    """Version-tolerant time setter (rerun API changed across 0.x)."""
    try:
        rr.set_time("frame", sequence=frame)
    except (AttributeError, TypeError):
        try:
            rr.set_time_sequence("frame", frame)
        except AttributeError:
            pass


def write_rerun_snapshot(ws: EpisodeWorkspace, max_frames: int = 300) -> Optional[Path]:
    try:
        import rerun as rr
    except ImportError:
        return None

    out = ws.export_dir / "episode.rrd"
    rr.init(f"v2r/{ws.episode_id}", spawn=False)
    rr.save(str(out))
    try:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    except Exception:
        pass

    # camera trajectory ---------------------------------------------------
    if ws.poses_parquet.is_file():
        poses = read_table(ws.poses_parquet)
        mat_cols = [c for c in poses.columns if c.startswith("T_wc_")]
        T = poses[mat_cols].to_numpy(dtype=np.float64).reshape(-1, 4, 4)
        stride = max(1, len(T) // max_frames)
        rr.log("world/camera_path", rr.LineStrips3D([T[::stride, :3, 3]]), static=True)
        for i in range(0, len(T), stride):
            _set_time(rr, int(poses["frame"].iloc[i]), float(poses["t"].iloc[i]))
            rr.log("world/camera", rr.Transform3D(translation=T[i, :3, 3], mat3x3=T[i, :3, :3]))

    # body joints -----------------------------------------------------------
    if ws.smplx_npz.is_file():
        z = read_npz(ws.smplx_npz)
        jw = z["joints_world"]
        stride = max(1, len(jw) // max_frames)
        for i in range(0, len(jw), stride):
            _set_time(rr, i, float(z["t"][i]))
            rr.log("world/body", rr.Points3D(jw[i], labels=SMPLX_MAIN_JOINTS, radii=0.015))

    # hands -------------------------------------------------------------------
    if ws.hands_parquet.is_file():
        hands = read_table(ws.hands_parquet)
        for side in ("left", "right"):
            h = hands[hands["hand"] == side]
            frames = np.sort(h["frame"].unique())
            stride = max(1, len(frames) // max_frames)
            for f in frames[::stride]:
                rows = h[h["frame"] == f]
                _set_time(rr, int(f), float(rows["t"].iloc[0]))
                rr.log(f"world/hands/{side}",
                       rr.Points3D(rows[["px", "py", "pz"]].to_numpy(), radii=0.006))

    # objects ---------------------------------------------------------------
    if ws.tracks_parquet.is_file():
        tracks = read_table(ws.tracks_parquet)
        for oid, grp in tracks.groupby("object_id"):
            grp = grp.sort_values("frame")
            stride = max(1, len(grp) // max_frames)
            for _, row in grp.iloc[::stride].iterrows():
                _set_time(rr, int(row["frame"]), float(row["t"]))
                rr.log(f"world/objects/{oid}",
                       rr.Points3D([[row["px"], row["py"], row["pz"]]], radii=0.02))

    return out
