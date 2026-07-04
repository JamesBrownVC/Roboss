"""Stage manifest I/O and content hashing for idempotent re-runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import V2RConfig
from ..schema.models import StageManifest, StageStatus
from ..schema.io import read_json_model, write_json_model
from ..schema.workspace import EpisodeWorkspace
from ..stages.base import STAGE_DEPS, sha256_bytes, sha256_config, sha256_file


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def config_hash_for_stage(cfg: V2RConfig, stage: str) -> str:
    payload = {
        "stage": stage,
        "toggle": cfg.stage(stage).model_dump(),
        "qa": cfg.qa.get(stage, cfg.qa.get(stage.replace("_validate", ""), {})),
        "licensing_permissive": cfg.licensing.get("permissive_only", False),
    }
    return sha256_config(payload)


def input_hash_for_stage(ws: EpisodeWorkspace, cfg: V2RConfig, stage: str, robots: list[str]) -> str:
    parts: list[str] = []
    for dep in STAGE_DEPS.get(stage, ()):
        mpath = ws.manifest_path(dep)
        if mpath.is_file():
            m = read_json_model(mpath, StageManifest)
            parts.append(f"{dep}:{m.output_hash}")
        else:
            parts.append(f"{dep}:missing")
    parts.append(f"robots:{','.join(sorted(robots))}")
    parts.append(f"cfg:{config_hash_for_stage(cfg, stage)}")
    if stage == "ingest" and ws.video_path.is_file():
        parts.append(f"video:{sha256_file(ws.video_path)}")
    return sha256_bytes("\n".join(parts).encode("utf-8"))


def output_hash_for_paths(root: Path, rel_paths: list[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        p = root / rel
        h.update(rel.encode("utf-8"))
        if p.is_file():
            h.update(sha256_file(p).encode("utf-8"))
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file():
                    h.update(str(fp.relative_to(root)).encode("utf-8"))
                    h.update(sha256_file(fp).encode("utf-8"))
    return h.hexdigest()


def read_manifest(ws: EpisodeWorkspace, stage: str) -> Optional[StageManifest]:
    p = ws.manifest_path(stage)
    if not p.is_file():
        return None
    return read_json_model(p, StageManifest)


def should_skip(
    ws: EpisodeWorkspace,
    cfg: V2RConfig,
    stage: str,
    robots: list[str],
) -> bool:
    m = read_manifest(ws, stage)
    if m is None or m.status != StageStatus.success:
        return False
    expected_in = input_hash_for_stage(ws, cfg, stage, robots)
    expected_cfg = config_hash_for_stage(cfg, stage)
    return m.input_hash == expected_in and m.config_hash == expected_cfg


def write_manifest(
    ws: EpisodeWorkspace,
    stage: str,
    cfg: V2RConfig,
    robots: list[str],
    result: Any,
    started_at: str,
    outputs: list[str],
) -> StageManifest:
    manifest = StageManifest(
        stage=stage,
        tool=getattr(result, "tool", ""),
        repo=getattr(result, "repo", ""),
        commit=getattr(result, "commit", ""),
        weights_sha256=getattr(result, "weights_sha256", {}),
        config_hash=config_hash_for_stage(cfg, stage),
        input_hash=input_hash_for_stage(ws, cfg, stage, robots),
        output_hash=output_hash_for_paths(ws.root, outputs),
        mode=getattr(result, "mode", cfg.stage(stage).mode),
        started_at=started_at,
        finished_at=_utcnow(),
        runtime_s=getattr(result, "runtime_s", 0.0),
        status=result.status,
        metrics=getattr(result, "metrics", {}),
        failure_reason=getattr(result, "failure_reason", None),
        outputs=outputs,
    )
    write_json_model(ws.manifest_path(stage), manifest)
    return manifest
