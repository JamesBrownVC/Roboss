"""DAG runner: manifests, idempotence, retries, GPU semaphore."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ..config import V2RConfig
from ..schema.models import StageStatus
from ..schema.workspace import EpisodeWorkspace
from ..stages import load_all_stages
from ..stages.base import STAGE_DEPS, STAGE_ORDER, STAGE_REGISTRY, StageContext, StageResult
from .gpu import GPUSemaphore
from .manifest import should_skip, write_manifest

GPU_STAGES = frozenset({"geometry", "human_body", "hands", "objects", "semantics", "retarget"})


@dataclass
class RunResult:
    episode_id: str
    workspace: Path
    stages: dict[str, StageStatus] = field(default_factory=dict)
    accepted: bool = False
    errors: list[str] = field(default_factory=list)


def _topo_sort(requested: set[str]) -> list[str]:
    """Run stages in canonical order, including transitive dependencies."""
    expanded = set(requested)
    changed = True
    while changed:
        changed = False
        for s in list(expanded):
            for dep in STAGE_DEPS.get(s, ()):
                if dep not in expanded:
                    expanded.add(dep)
                    changed = True
    return [s for s in STAGE_ORDER if s in expanded]


def resolve_stages(stages_arg: str) -> set[str]:
    if stages_arg in ("all", "*"):
        return set(STAGE_ORDER)
    return {s.strip() for s in stages_arg.split(",") if s.strip()}


def episode_id_from_path(video: Path) -> str:
    stem = video.stem.replace(" ", "_")
    return EpisodeWorkspace.make_episode_id(stem, 0)


def run_episode(
    cfg: V2RConfig,
    source_video: Path,
    robots: list[str],
    stages: set[str],
    mode_override: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> RunResult:
    import_errors = load_all_stages()
    if import_errors:
        raise RuntimeError(f"stage import errors: {import_errors}")

    source_video = Path(source_video).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)

    eid = episode_id_from_path(source_video)
    ws = EpisodeWorkspace(cfg.workspaces_root, eid).create()
    gpu = GPUSemaphore(cfg.pipeline.gpu_slots, lock_dir=cfg.root / ".gpu_locks")
    log_path = ws.root / "run.jsonl"

    requested = set(stages)
    plan = _topo_sort(requested)
    result = RunResult(episode_id=eid, workspace=ws.root)
    halted = False

    for stage_name in plan:
        if halted and stage_name not in ("qa", "package"):
            result.stages[stage_name] = StageStatus.skipped
            continue

        toggle = cfg.stage(stage_name)
        if not toggle.enabled:
            result.stages[stage_name] = StageStatus.skipped
            continue

        mode = mode_override or toggle.mode or cfg.pipeline.default_mode
        if should_skip(ws, cfg, stage_name, robots):
            log(f"[skip] {stage_name} (manifest hash match)")
            result.stages[stage_name] = StageStatus.skipped
            continue

        stage_cls = STAGE_REGISTRY.get(stage_name)
        if stage_cls is None:
            result.errors.append(f"unknown stage {stage_name}")
            result.stages[stage_name] = StageStatus.failed
            halted = True
            continue

        ctx = StageContext(
            ws=ws,
            cfg=cfg,
            robots=robots,
            mode=mode,
            source_video=source_video if stage_name == "ingest" else None,
            log=log,
        )

        started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        t0 = time.time()
        sr: StageResult | None = None
        attempts = cfg.pipeline.retries + 1

        for attempt in range(attempts):
            try:
                use_gpu = stage_name in GPU_STAGES and mode == "real"
                if use_gpu:
                    with gpu.acquire():
                        sr = stage_cls().run(ctx)
                else:
                    sr = stage_cls().run(ctx)
                if sr.status != StageStatus.failed:
                    break
                if attempt < attempts - 1:
                    log(f"[retry] {stage_name} attempt {attempt + 1}/{attempts}")
                    time.sleep(cfg.pipeline.retry_backoff_s)
            except Exception as e:
                sr = StageResult(status=StageStatus.failed, failure_reason=f"{type(e).__name__}: {e}")
                if attempt < attempts - 1:
                    time.sleep(cfg.pipeline.retry_backoff_s)

        assert sr is not None
        sr.runtime_s = time.time() - t0
        write_manifest(ws, stage_name, cfg, robots, sr, started, sr.outputs)
        result.stages[stage_name] = sr.status

        _log_jsonl(log_path, {
            "ts": started,
            "stage": stage_name,
            "status": sr.status.value,
            "mode": mode,
            "metrics": sr.metrics,
            "failure": sr.failure_reason,
        })

        log(f"[{sr.status.value}] {stage_name} ({mode}) {sr.metrics}")

        if sr.status in (StageStatus.failed, StageStatus.rejected):
            halted = True
            if sr.failure_reason:
                result.errors.append(f"{stage_name}: {sr.failure_reason}")

    dec_path = ws.decision_json
    if dec_path.is_file():
        from ..schema.io import read_json_model
        from ..schema.models import Decision
        result.accepted = read_json_model(dec_path, Decision).accepted
    else:
        result.accepted = all(
            result.stages.get(s) in (StageStatus.success, StageStatus.skipped)
            for s in plan
        )

    return result


def _log_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
