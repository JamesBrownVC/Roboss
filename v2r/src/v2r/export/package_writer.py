"""LeRobot v3 fragment + EgoDex mirror stub writer."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import V2RConfig
from ..schema.workspace import EpisodeWorkspace


def write_exports(
    ws: EpisodeWorkspace,
    cfg: V2RConfig,
    robots: list[str],
    synthetic: bool = True,
    tier: str = "monocular",
) -> list[str]:
    outputs: list[str] = []
    lerobot = ws.lerobot_dir
    lerobot.mkdir(parents=True, exist_ok=True)

    meta = {
        "episode_id": ws.episode_id,
        "format": "lerobot_v3_fragment",
        "synthetic": synthetic,
        "tier": tier,
        "tier_description": (
            "multiview_gt: triangulated joints with measured reprojection error bars"
            if tier == "multiview_gt"
            else "monocular: confidence-masked estimated kinematics"
        ),
        "robots": robots,
        "consent_required": True,
    }
    (lerobot / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs.append(ws.rel(lerobot / "meta.json"))

    # Minimal feature manifest (full writer uses lerobot package on CUDA host)
    features = {
        "observation.images.ego": {"dtype": "video", "shape": [720, 1280, 3]},
        "observation.depth.ego": {"dtype": "float32", "shape": [360, 640]},
        "retarget.qpos": {"dtype": "float32", "robots": robots},
    }
    (lerobot / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    outputs.append(ws.rel(lerobot / "features.json"))

    egodex = ws.egodex_mirror_dir
    egodex.mkdir(parents=True, exist_ok=True)
    idx = 0
    stub_h5 = egodex / f"{idx}.hdf5"
    stub_h5.write_bytes(b"HDF5_STUB")
    outputs.append(ws.rel(stub_h5))
    if ws.video_path.is_file():
        import shutil
        dst = egodex / f"{idx}.mp4"
        if not dst.exists():
            shutil.copy2(ws.video_path, dst)
        outputs.append(ws.rel(dst))

    return outputs
