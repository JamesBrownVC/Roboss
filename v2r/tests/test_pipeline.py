"""End-to-end pipeline test (synthetic mode)."""

from pathlib import Path

import pytest

from v2r.config import V2RConfig
from v2r.orchestrator.runner import run_episode, resolve_stages


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_video(repo_root: Path) -> Path:
    p = repo_root / "tests" / "data" / "sample.mp4"
    if not p.is_file():
        import subprocess
        subprocess.run(["python", str(repo_root / "tests" / "data" / "make_sample.py")], check=True)
    assert p.is_file()
    return p


def test_resolve_stages_all():
    stages = resolve_stages("all")
    assert "geometry" in stages
    assert "feasibility_judge" in stages


def test_pipeline_synthetic(sample_video, repo_root, tmp_path):
    cfg = V2RConfig.load(repo_root)
    cfg.pipeline.workspaces_root = str(tmp_path / "workspaces")
    result = run_episode(
        cfg, sample_video, robots=["g1"], stages=resolve_stages("all"), mode_override="synthetic"
    )
    assert result.workspace.is_dir()
    assert (result.workspace / "manifests" / "ingest.manifest.json").is_file()
    lerobot = result.workspace / "export" / "lerobot"
    assert (lerobot / "meta" / "info.json").is_file()
    assert (lerobot / "meta" / "tasks.parquet").is_file()
    assert (lerobot / "data" / "chunk-000" / "file-000.parquet").is_file()
    assert (lerobot / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4").is_file()
    import json

    info = json.loads((lerobot / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["validation"]["passed"]
    assert (result.workspace / "export" / "egodex_mirror" / "0.hdf5").is_file()
    assert (result.workspace / "retargets" / "g1" / "qpos.csv").is_file()
    assert (repo_root / "LICENSE_AUDIT.md").is_file()
    assert result.accepted
