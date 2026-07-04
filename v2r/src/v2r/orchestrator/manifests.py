"""StageManifest construction, hashing, and IO.

The ORCHESTRATOR owns manifests; stage wrappers never write them.

Hash rules (master prompt section 8 + module spec):

  config_hash   base.sha256_config over {stage toggle (with the effective
                mode), the qa.yaml section relevant to the stage, the robots
                list}.
  input_hash    base.sha256_config over the dependency stages' manifests'
                output_hash values, plus (ingest only) the sha256 of the
                source video file (base.sha256_file).
  output_hash   base.sha256_config over sorted (relpath, size, token)
                entries of the stage's declared outputs, where token is the
                content sha256 for files < 8 MB, else "size:{bytes}".
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..config import V2RConfig
from ..schema.io import read_json_model, write_json_model
from ..schema.models import StageManifest
from ..schema.workspace import EpisodeWorkspace
from ..stages.base import STAGE_DEPS, StageResult, sha256_config, sha256_file

# Files at or above this size are hashed as "size:{bytes}" instead of content.
SMALL_FILE_HASH_LIMIT = 8 * 1024 * 1024

# qa.yaml section relevant to each stage (default: the stage's own name).
QA_SECTION_FOR = {
    "physics_validate": "physics",
    "qa": "crosschecks",
    "package": "export",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(x: Any) -> Any:
    """Best-effort conversion to plain JSON types (numpy scalars -> python)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if hasattr(x, "item"):  # numpy scalar
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


# ---------------------------------------------------------------------------
# hashes
# ---------------------------------------------------------------------------


def compute_config_hash(
    cfg: V2RConfig,
    stage_name: str,
    robots: Sequence[str],
    effective_mode: Optional[str] = None,
) -> str:
    """Hash of everything configuration-side that should trigger a re-run."""
    toggle = cfg.stage(stage_name).model_dump()
    if effective_mode is not None:
        toggle["mode"] = effective_mode
    qa_key = QA_SECTION_FOR.get(stage_name, stage_name)
    qa_section = cfg.qa.get(qa_key, {})
    return sha256_config(
        {"stage": stage_name, "toggle": toggle, "qa": qa_section, "robots": list(robots)}
    )


def compute_input_hash(
    ws: EpisodeWorkspace,
    stage_name: str,
    source_video: Optional[Path] = None,
) -> str:
    """Hash of upstream outputs: dep manifests' output_hash values, plus the
    source video sha256 for ingest (falls back to raw/video.mp4 on resume)."""
    deps: dict[str, Optional[str]] = {}
    for dep in STAGE_DEPS.get(stage_name, ()):
        m = read_manifest(ws, dep)
        deps[dep] = m.output_hash if m is not None else None
    payload: dict[str, Any] = {"stage": stage_name, "deps": deps}
    if stage_name == "ingest":
        video: Optional[Path] = None
        if source_video is not None and Path(source_video).is_file():
            video = Path(source_video)
        elif ws.video_path.is_file():
            video = ws.video_path
        payload["source_video_sha256"] = sha256_file(video) if video is not None else None
    return sha256_config(payload)


def _entry(root: Path, rel: str) -> tuple[str, int, str]:
    p = root / rel
    if not p.is_file():
        return (rel, -1, "missing")
    size = p.stat().st_size
    token = sha256_file(p) if size < SMALL_FILE_HASH_LIMIT else f"size:{size}"
    return (rel, size, token)


def compute_output_hash(ws: EpisodeWorkspace, outputs: Iterable[str]) -> str:
    """Hash the stage's declared outputs (workspace-relative paths).

    Directories are expanded recursively. Content sha256 for small files,
    "size:{bytes}" token for large ones, "missing" for declared-but-absent.
    """
    entries: list[tuple[str, int, str]] = []
    for out in outputs:
        rel = Path(out).as_posix()
        p = ws.root / rel
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    entries.append(_entry(ws.root, f.relative_to(ws.root).as_posix()))
        else:
            entries.append(_entry(ws.root, rel))
    entries.sort()
    return sha256_config(entries)


# ---------------------------------------------------------------------------
# manifest build / IO
# ---------------------------------------------------------------------------


def build_manifest(
    stage_name: str,
    result: StageResult,
    *,
    mode: str,
    config_hash: str,
    input_hash: str,
    output_hash: str,
    started_at: str,
    finished_at: str,
    runtime_s: float,
) -> StageManifest:
    return StageManifest(
        stage=stage_name,
        tool=result.tool,
        repo=result.repo,
        commit=result.commit,
        weights_sha256=dict(result.weights_sha256),
        config_hash=config_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        mode=mode,
        started_at=started_at,
        finished_at=finished_at,
        runtime_s=float(runtime_s),
        status=result.status,
        metrics=jsonable(result.metrics),
        failure_reason=result.failure_reason,
        outputs=[Path(o).as_posix() for o in result.outputs],
    )


def write_manifest(ws: EpisodeWorkspace, manifest: StageManifest) -> Path:
    return write_json_model(ws.manifest_path(manifest.stage), manifest)


def read_manifest(ws: EpisodeWorkspace, stage_name: str) -> Optional[StageManifest]:
    path = ws.manifest_path(stage_name)
    if not path.is_file():
        return None
    try:
        return read_json_model(path, StageManifest)
    except Exception:
        # unreadable/corrupt manifest == no manifest (forces a re-run)
        return None
