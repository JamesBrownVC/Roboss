"""Load and validate the repo configuration (config/*.yaml) into one object."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from .schema.models import RobotSpec


class StageToggle(BaseModel):
    enabled: bool = True
    mode: str = "synthetic"  # synthetic | real
    env: Optional[str] = None


class PipelineSettings(BaseModel):
    workspaces_root: str = "workspaces"
    canonical_hz: float = 30.0
    max_interp_gap_s: float = 0.34
    default_mode: str = "synthetic"
    stages: dict[str, StageToggle] = Field(default_factory=dict)
    retries: int = 2
    retry_backoff_s: float = 5.0
    gpu_slots: int = 1


def _load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class V2RConfig:
    """All repo configuration, resolved against the repo root directory."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        cfg_dir = self.root / "config"
        self.pipeline = PipelineSettings.model_validate(_load_yaml(cfg_dir / "pipeline.yaml"))
        self.qa: dict[str, Any] = _load_yaml(cfg_dir / "qa.yaml")
        self.verbs: list[str] = list(_load_yaml(cfg_dir / "verbs.yaml").get("verbs", []))
        self.licensing: dict[str, Any] = _load_yaml(cfg_dir / "licensing.yaml")
        multiview_path = cfg_dir / "multiview.yaml"
        self.multiview: dict[str, Any] = _load_yaml(multiview_path) if multiview_path.is_file() else {}
        robots_raw = _load_yaml(cfg_dir / "robots.yaml").get("robots", {})
        self.robots: dict[str, RobotSpec] = {}
        for name, spec in robots_raw.items():
            rs = RobotSpec.model_validate(spec)
            rs.name = name
            self.robots[name] = rs

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Optional[Path | str] = None) -> "V2RConfig":
        """Load from `root`, or search upward from cwd for config/pipeline.yaml."""
        if root is not None:
            return cls(Path(root))
        cur = Path.cwd().resolve()
        for cand in (cur, *cur.parents):
            if (cand / "config" / "pipeline.yaml").is_file():
                return cls(cand)
        raise FileNotFoundError(
            "config/pipeline.yaml not found here or in any parent; "
            "run from the v2r repo or pass --root"
        )

    def stage(self, name: str) -> StageToggle:
        return self.pipeline.stages.get(name, StageToggle(mode=self.pipeline.default_mode))

    @property
    def workspaces_root(self) -> Path:
        p = Path(self.pipeline.workspaces_root)
        return p if p.is_absolute() else self.root / p

    def robot(self, name: str) -> RobotSpec:
        if name not in self.robots:
            raise KeyError(f"robot {name!r} not in config/robots.yaml (have: {list(self.robots)})")
        return self.robots[name]
