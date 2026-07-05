"""Central environment-backed settings for CLI and API entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from env_loader import load_dotenv


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    runs_dir: Path
    gemini_api_key: str | None
    video_model: str
    label_model: str
    gate2_enabled: bool
    label_on_accept: bool
    annotate_enabled: bool
    deterministic_agents: bool
    start_frame_workers: int
    video_workers: int
    # per-stage concurrency caps for the async pipeline
    gen_workers: int       # concurrent video generations (network-bound)
    verify_workers: int    # concurrent verifications (CPU-bound: YOLO)
    label_workers: int     # concurrent labeling jobs (network-bound)


def get_settings() -> Settings:
    load_dotenv()
    root = Path(__file__).resolve().parent.parent
    runs_dir = Path(os.environ.get("ROBOSS_RUNS_DIR", root / "runs"))
    return Settings(
        root_dir=root,
        runs_dir=runs_dir,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        video_model=os.environ.get("ROBOSS_VIDEO_MODEL",
                                   "gemini-omni-flash-preview"),
        label_model=os.environ.get("ROBOSS_LABEL_MODEL", "gemini-3.5-flash"),
        gate2_enabled=_bool_env("ROBOSS_GATE2_ENABLED", True),
        label_on_accept=_bool_env("ROBOSS_LABEL_ON_ACCEPT", True),
        annotate_enabled=_bool_env("ROBOSS_ANNOTATE_ENABLED", False),
        deterministic_agents=_bool_env("ROBOSS_DETERMINISTIC_AGENTS", True),
        start_frame_workers=_int_env("ROBOSS_START_FRAME_WORKERS", 2),
        video_workers=_int_env("ROBOSS_VIDEO_WORKERS", 8),
        gen_workers=_int_env("ROBOSS_GEN_WORKERS", 4),
        verify_workers=_int_env("ROBOSS_VERIFY_WORKERS", 2),
        label_workers=_int_env("ROBOSS_LABEL_WORKERS", 4),
    )

