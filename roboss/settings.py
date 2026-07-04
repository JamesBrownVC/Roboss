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


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    runs_dir: Path
    gemini_api_key: str | None
    video_model: str
    label_model: str
    gate2_enabled: bool
    label_on_accept: bool


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
    )

