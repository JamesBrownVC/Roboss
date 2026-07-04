"""Stage contract. A stage is a thin wrapper: it translates the interchange
contract (episode workspace artifacts) to a tool invocation and back. Nothing
else. The orchestrator owns manifests, retries, hashing, and skipping.

Stage DAG (names and dependencies are the contract):

    ingest -> feasibility_judge -> geometry -> human_body -> hands -> contact -> semantics
                              \\-> objects ----/                       |
    [hands, semantics] -> retarget -> physics_validate -> qa -> package

Modes:
    real       shell out to the pinned third-party tool in its isolated env
               (micromamba/pixi/docker). CUDA Linux host required.
    synthetic  produce schema-valid artifacts tagged source=synthesized, with
               deterministic per-episode randomness. Exercises every contract,
               gate, QA and export path on any host. Never call this data
               ground truth.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Type

import numpy as np

from ..config import V2RConfig
from ..schema.models import GateOutcome, StageStatus
from ..schema.workspace import EpisodeWorkspace

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

STAGE_REGISTRY: dict[str, Type["Stage"]] = {}


def register_stage(cls: Type["Stage"]) -> Type["Stage"]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls} has no stage name")
    STAGE_REGISTRY[cls.name] = cls
    return cls


# Canonical stage order + dependencies (orchestrator builds the DAG from this).
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "feasibility_judge": ("ingest",),
    "geometry": ("feasibility_judge",),
    "human_body": ("geometry",),
    "hands": ("human_body",),
    "objects": ("geometry",),
    "contact": ("hands", "objects"),
    "semantics": ("hands", "contact"),
    "retarget": ("human_body", "hands"),
    "physics_validate": ("retarget",),
    "qa": ("physics_validate", "semantics"),
    "package": ("qa",),
}

STAGE_ORDER: tuple[str, ...] = (
    "ingest", "feasibility_judge", "geometry", "human_body", "hands", "objects", "contact",
    "semantics", "retarget", "physics_validate", "qa", "package",
)


# ---------------------------------------------------------------------------
# context & result
# ---------------------------------------------------------------------------


@dataclass
class StageContext:
    ws: EpisodeWorkspace
    cfg: V2RConfig
    robots: list[str]
    mode: str                      # "synthetic" | "real"
    source_video: Optional[Path] = None   # only meaningful for ingest
    log: Callable[[str], None] = print


@dataclass
class StageResult:
    status: StageStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    outputs: list[str] = field(default_factory=list)   # workspace-relative
    gate: Optional[GateOutcome] = None
    tool: str = ""
    repo: str = ""
    commit: str = ""
    weights_sha256: dict[str, str] = field(default_factory=dict)
    runtime_s: float = 0.0


class Stage(ABC):
    """One pipeline stage. Subclasses set `name` and implement `run`."""

    name: ClassVar[str]

    @property
    def deps(self) -> tuple[str, ...]:
        return STAGE_DEPS[self.name]

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        """Read inputs from ctx.ws, write outputs to ctx.ws, return result.

        Gate failures return status=rejected with gate populated — never raise
        for a quality rejection. Raise only for genuine tool/system errors.
        """


# ---------------------------------------------------------------------------
# helpers shared by all stages
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_config(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def rng_for(episode_id: str, stage: str) -> np.random.Generator:
    """Deterministic per-(episode, stage) RNG for synthetic mode."""
    seed = int.from_bytes(
        hashlib.blake2b(f"{episode_id}:{stage}".encode(), digest_size=8).digest(), "little"
    )
    return np.random.default_rng(seed)


def run_tool(
    cmd: list[str],
    env_name: Optional[str] = None,
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Invoke a stage tool inside its isolated env.

    env_name -> prefixed with `micromamba run -n {env}`; docker images are
    invoked by passing the full `docker run ...` cmd with env_name=None.
    """
    full = (["micromamba", "run", "-n", env_name] + cmd) if env_name else cmd
    return subprocess.run(
        full, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout
    )


def gate_from_thresholds(metrics: dict[str, float], checks: list[tuple[str, str, float, bool]]) -> GateOutcome:
    """Build a GateOutcome from (metric_key, op, threshold, required) checks.

    op: 'ge' (metric >= threshold passes) or 'le' (metric <= threshold passes).
    """
    reasons: list[str] = []
    passed = True
    for key, op, thr, required in checks:
        val = metrics.get(key)
        if val is None:
            if required:
                passed = False
                reasons.append(f"{key}: missing")
            continue
        ok = (val >= thr) if op == "ge" else (val <= thr)
        if not ok:
            passed = False
            reasons.append(f"{key}={val:.4g} fails {op} {thr:.4g}")
    return GateOutcome(passed=passed, reasons=reasons, metrics=metrics)
