"""Multi-view session tests (synthetic mode, Windows-safe)."""

from pathlib import Path

import pandas as pd
import pytest

from v2r.config import V2RConfig
from v2r.schema.io import read_json_model
from v2r.schema.models import CrossViewReprojReport, SessionCalibration, SessionSync
from v2r.schema.session import SessionWorkspace
from v2r.session.runner import (
    parse_cam_spec,
    run_session,
    session_create,
    session_triangulate,
)


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


def test_parse_cam_spec():
    specs = parse_cam_spec(["cam0:a.mp4", "cam1:b.mp4"])
    assert "cam0" in specs and "cam1" in specs


def test_synthetic_session_triangulation(sample_video, repo_root, tmp_path):
    cfg = V2RConfig.load(repo_root)
    cfg.pipeline.workspaces_root = str(tmp_path / "workspaces")

    sw = session_create(
        cfg,
        "test_mv",
        {"cam0": sample_video, "cam1": sample_video, "cam2": sample_video},
    )
    assert sw.cam_video("cam0").is_file()
    assert len(sw.list_cameras()) == 3

    df = session_triangulate(sw, cfg, mode="synthetic")
    assert sw.joints_parquet.is_file()
    assert set(df.columns) >= {"t", "frame", "joint", "px", "py", "pz", "conf", "valid", "source", "reproj_error_px"}
    assert (df["source"] == "triangulated").all()
    assert df["conf"].between(0, 1).all()


def test_session_run_multiview(sample_video, repo_root, tmp_path):
    cfg = V2RConfig.load(repo_root)
    cfg.pipeline.workspaces_root = str(tmp_path / "workspaces")

    session_create(cfg, "test_session", {"cam0": sample_video, "cam1": sample_video})
    result = run_session(cfg, "test_session", tier="multiview", robots=["g1"], mode="synthetic")

    sw = SessionWorkspace(cfg.workspaces_root / "sessions", "test_session")
    assert result.steps.get("sync") == "success"
    assert result.steps.get("calibrate") == "success"
    assert result.steps.get("triangulate") == "success"
    assert result.steps.get("fuse") == "success"
    assert sw.sync_json.is_file()
    assert sw.calibration_json.is_file()
    assert sw.joints_parquet.is_file()
    assert sw.cross_view_reproj_json.is_file()

    sync = read_json_model(sw.sync_json, SessionSync)
    assert len(sync.cameras) >= 2
    cal = read_json_model(sw.calibration_json, SessionCalibration)
    assert len(cal.cameras) >= 2
    reproj = read_json_model(sw.cross_view_reproj_json, CrossViewReprojReport)
    assert reproj.mean_reproj_error_px >= 0
    assert reproj.triangulation_wins is not None

    df = pd.read_parquet(sw.joints_parquet)
    assert len(df) > 0
