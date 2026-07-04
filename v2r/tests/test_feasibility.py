"""Feasibility judge stage tests."""

from pathlib import Path

import pandas as pd
import pytest

from v2r.config import V2RConfig
from v2r.orchestrator.runner import run_episode, resolve_stages
from v2r.schema.io import read_json_model
from v2r.schema.models import FeasibilityReport
from v2r.schema.workspace import EpisodeWorkspace


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_video(repo_root: Path) -> Path:
    p = repo_root / "tests" / "data" / "sample.mp4"
    if not p.is_file():
        import subprocess
        subprocess.run(["python", str(repo_root / "tests" / "data" / "make_sample.py")], check=True)
    return p


def test_feasibility_judge_synthetic(sample_video, repo_root, tmp_path):
    cfg = V2RConfig.load(repo_root)
    cfg.pipeline.workspaces_root = str(tmp_path / "workspaces")

    result = run_episode(
        cfg, sample_video, robots=["g1"],
        stages={"ingest", "feasibility_judge"},
        mode_override="synthetic",
    )
    ws = EpisodeWorkspace(cfg.workspaces_root, result.episode_id)
    assert ws.feasibility_report_json.is_file()
    assert ws.feasibility_mask_parquet.is_file()

    report = read_json_model(ws.feasibility_report_json, FeasibilityReport)
    assert report.recommendation.value in ("proceed", "reject", "human_review")
    assert 0 <= report.confidence <= 1

    mask = pd.read_parquet(ws.feasibility_mask_parquet)
    assert set(mask.columns) >= {"t", "frame", "valid", "conf", "source", "physics_violation"}


def test_feasibility_in_full_pipeline(sample_video, repo_root, tmp_path):
    cfg = V2RConfig.load(repo_root)
    cfg.pipeline.workspaces_root = str(tmp_path / "workspaces")
    result = run_episode(
        cfg, sample_video, robots=["g1"],
        stages=resolve_stages("all"),
        mode_override="synthetic",
    )
    ws = EpisodeWorkspace(cfg.workspaces_root, result.episode_id)
    assert (ws.root / "manifests" / "feasibility_judge.manifest.json").is_file()
    yield_md = ws.yield_report_md.read_text(encoding="utf-8")
    assert "feasibility_judge" in yield_md or "Feasibility gate" in yield_md
