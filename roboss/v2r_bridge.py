"""Bridge from verified videos to James' V2R robot-data pipeline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from .settings import get_settings
from .storage import get_storage


def _ensure_v2r_importable() -> Path:
    root = get_settings().root_dir / "v2r"
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"v2r source tree not found: {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root


def export_video_to_robot_data(video_path: str | Path,
                               outdir: str | Path | None = None,
                               robots: list[str] | None = None,
                               mode: str = "synthetic",
                               stages: str = "all",
                               progress=print) -> dict[str, Any]:
    """Run V2R on one accepted video and expose the result through /assets."""
    v2r_root = _ensure_v2r_importable()
    from v2r.config import V2RConfig
    from v2r.orchestrator.runner import resolve_stages, run_episode

    source_video = Path(video_path).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)

    robot_list = robots or ["g1"]
    cfg = V2RConfig.load(v2r_root)
    stage_set = resolve_stages(stages)

    progress(f"[V2R] Running robot-data export for {source_video.name} ...")
    result = run_episode(
        cfg,
        source_video,
        robots=robot_list,
        stages=stage_set,
        mode_override=mode,
        log=progress,
    )

    storage = get_storage()
    if outdir is None:
        dest = storage.root / "robot_data" / result.episode_id
    else:
        dest = Path(outdir)
    dest = dest.resolve()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(result.workspace, dest)

    stages_out = {
        name: status.value if hasattr(status, "value") else str(status)
        for name, status in result.stages.items()
    }
    manifest = storage.write_manifest(dest, {
        "kind": "robot_dataset_export",
        "episode_id": result.episode_id,
        "accepted": result.accepted,
        "robots": robot_list,
        "mode": mode,
        "stages": stages_out,
        "errors": result.errors,
        "source_video": str(source_video),
        "v2r_workspace": str(result.workspace),
        "lerobot_url": storage.url_for(dest / "export" / "lerobot")
        if (dest / "export" / "lerobot").exists() else None,
        "yield_report_url": storage.url_for(dest / "qa" / "yield_report.md")
        if (dest / "qa" / "yield_report.md").exists() else None,
    })

    return {
        "episode_id": result.episode_id,
        "accepted": result.accepted,
        "robots": robot_list,
        "mode": mode,
        "stages": stages_out,
        "errors": result.errors,
        "source_workspace": str(result.workspace),
        "outdir": str(dest),
        "manifest": manifest.url,
        "files": storage.collect_files(dest),
    }
