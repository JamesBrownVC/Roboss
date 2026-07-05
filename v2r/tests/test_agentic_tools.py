"""Unit tests for the agentic perception tools (time-series/motion layer).

Only exercises what runs deterministically on CPU without network: trajectory
characterization, changepoint segmentation, optical flow, scene detection,
and object tracking on the synthetic sample clip.
"""

from pathlib import Path

import numpy as np
import pytest

from v2r.agentic import tools as T
from v2r.config import V2RConfig
from v2r.schema.workspace import EpisodeWorkspace


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_video(repo_root: Path) -> Path:
    p = repo_root / "tests" / "data" / "sample.mp4"
    if not p.is_file():
        import subprocess

        subprocess.run(["python", str(repo_root / "tests" / "data" / "make_sample.py")],
                       check=True)
    return p


@pytest.fixture()
def ws(tmp_path: Path, sample_video: Path) -> EpisodeWorkspace:
    import shutil

    w = EpisodeWorkspace(tmp_path / "workspaces", "tools_test_000000").create()
    shutil.copy2(sample_video, w.video_path)
    return w


# ---------------------------------------------------------------------------
# characterize_trajectory
# ---------------------------------------------------------------------------


def test_trajectory_straight_line():
    t = np.linspace(0, 2, 60)
    pos = np.column_stack([t * 0.5, np.zeros_like(t), np.zeros_like(t)])
    out = T.characterize_trajectory(t, pos)
    assert out["straightness"] > 0.95
    assert out["n_direction_changes"] == 0
    assert abs(out["mean_speed"] - 0.5) < 0.05
    assert not out["periodic"]


def test_trajectory_periodic_wave():
    t = np.linspace(0, 4, 120)
    # 1 Hz oscillation: a waving hand
    pos = np.column_stack([0.2 * np.sin(2 * np.pi * t),
                           np.zeros_like(t), np.zeros_like(t)])
    out = T.characterize_trajectory(t, pos)
    assert out["periodic"], out
    assert 0.3 < out["period_s"] < 0.7  # speed period = half the position period
    assert out["straightness"] < 0.2
    assert out["n_direction_changes"] >= 3


def test_trajectory_too_short():
    out = T.characterize_trajectory(np.array([0.0, 0.1]), np.zeros((2, 3)))
    assert out["n_samples"] == 2
    assert "note" in out


# ---------------------------------------------------------------------------
# segment_motion_channels (ruptures)
# ---------------------------------------------------------------------------


def test_motion_channel_segmentation_finds_regime_change():
    grid_t = np.arange(0.0, 10.0, 0.1)
    # still for 4s, then vigorous motion, then still again
    sig = np.where((grid_t >= 4.0) & (grid_t < 7.0), 5.0, 0.1)
    rng = np.random.default_rng(0)
    channels = {"speed": sig + rng.normal(0, 0.15, len(grid_t))}
    segs = T.segment_motion_channels(grid_t, channels)
    assert 2 <= len(segs) <= 5
    boundaries = [s["start_s"] for s in segs[1:]]
    assert any(abs(b - 4.0) < 1.0 for b in boundaries), boundaries
    assert any(abs(b - 7.0) < 1.0 for b in boundaries), boundaries


def test_motion_channel_segmentation_constant_signal():
    grid_t = np.arange(0.0, 5.0, 0.1)
    segs = T.segment_motion_channels(grid_t, {"flat": np.full(len(grid_t), 2.0)})
    assert len(segs) == 1
    assert segs[0]["start_s"] == 0.0


# ---------------------------------------------------------------------------
# video-level tools on the synthetic sample clip
# ---------------------------------------------------------------------------


def test_optical_flow_timeline(ws: EpisodeWorkspace):
    out = T.optical_flow_timeline(ws.video_path, ws)
    assert out["available"]
    assert len(out["camera_motion_px_s_per_second"]) >= 1
    assert len(out["subject_motion_px_s_per_second"]) == \
        len(out["camera_motion_px_s_per_second"])
    assert (ws.qa_dir / "flow_timeline.json").is_file()


def test_detect_scenes(ws: EpisodeWorkspace):
    out = T.detect_scenes(ws.video_path)
    assert out["available"]
    assert out["n_shots"] >= 1


def test_track_objects(ws: EpisodeWorkspace, repo_root: Path):
    cfg = V2RConfig.load(repo_root)
    out = T.track_objects(ws.video_path, ws, cfg)
    assert out["available"]
    # synthetic counter-card clip: zero tracks is a legitimate honest result
    if out["n_tracks"] > 0:
        tr = out["tracks"][0]
        assert {"track_id", "class", "trajectory", "size_change_ratio"} <= set(tr)


def test_motion_primitives(ws: EpisodeWorkspace, repo_root: Path):
    cfg = V2RConfig.load(repo_root)
    out = T.motion_primitives(ws.video_path, ws, cfg)
    assert out["available"]
    assert out["n_primitives"] >= 1
    seg = out["primitives"][0]
    assert seg["start_s"] == 0.0
    assert "channels" in seg and "activity" in seg
    assert (ws.semantics_dir / "motion_primitives.json").is_file()


def test_aigen_forensics_reports_inconclusive(ws: EpisodeWorkspace):
    out = T.aigen_forensics(ws.video_path)
    assert out["available"]
    # calibration showed no real/generated separation: verdict must never
    # assert anything stronger than inconclusive
    assert out["verdict"] == "inconclusive"
    assert "NOT DISCRIMINATIVE" in out["reliability"]


# ---------------------------------------------------------------------------
# bench scorer (pure functions)
# ---------------------------------------------------------------------------


def test_bench_scorer_positive_clip():
    from v2r.agentic.bench import score_clip

    gt = {"acceptable_verdicts": ["proceed"], "acceptable_human": ["full_body"],
          "expected_skills": ["grasp", "lift"],
          "allowed_skills": ["reach", "grasp", "lift", "hold"],
          "forbidden_skills": ["pour"],
          "key_segments": [{"skills": ["grasp"], "start_s": 5.0, "end_s": 7.0}]}
    pred = {"recommendation": "proceed", "human_present": "full_body",
            "segments": [
                {"start_s": 0.0, "end_s": 5.0, "skill": "idle", "text": ""},
                {"start_s": 5.0, "end_s": 7.0, "skill": "grasp", "text": "",
                 "evidence": "primitives changepoint 5.0s"},
            ]}
    s = score_clip(gt, pred)
    assert s["verdict_ok"] and s["human_ok"] and s["skill_recall"]
    assert s["skill_precision"] == 1.0
    assert s["hallucinations"] == []
    assert s["boundary_iou"] == 1.0
    assert s["evidence_coverage"] == 1.0


def test_bench_scorer_catches_fabrication():
    from v2r.agentic.bench import score_clip

    gt = {"acceptable_verdicts": ["reject"], "acceptable_human": ["none"],
          "expected_skills": [], "allowed_skills": ["idle"],
          "forbidden_skills": ["wave", "walk"]}
    pred = {"recommendation": "proceed", "human_present": "full_body",
            "segments": [{"start_s": 0.0, "end_s": 3.0, "skill": "wave",
                          "text": "person waving"}]}
    s = score_clip(gt, pred)
    assert not s["verdict_ok"] and not s["human_ok"]
    assert not s["skill_recall"]  # negative clip: action predicted = miss
    assert s["hallucinations"] == ["wave"]


def test_bench_aggregate():
    from v2r.agentic.bench import aggregate_scores

    agg = aggregate_scores({
        "a": {"verdict_ok": True, "human_ok": True, "skill_recall": True,
              "skill_precision": 1.0, "hallucinations": [],
              "boundary_iou": 0.8, "boundary_mae_s": 0.5,
              "evidence_coverage": 1.0},
        "b": {"verdict_ok": False, "human_ok": True, "skill_recall": False,
              "skill_precision": 0.5, "hallucinations": ["wave"],
              "boundary_iou": None, "boundary_mae_s": None,
              "evidence_coverage": None},
    })
    assert agg["n_clips"] == 2
    assert agg["verdict_accuracy"] == 0.5
    assert agg["clips_with_hallucinations"] == 1
    assert agg["boundary_iou_mean"] == 0.8


# ---------------------------------------------------------------------------
# animal_pose gait math (SuperAnimal-Quadruped)
# ---------------------------------------------------------------------------


def test_autocorr_period_finds_stride():
    # synthetic paw-tip speed: a 0.5 s gait cycle sampled at 20 Hz
    dt = 0.05
    t = np.arange(0, 6, dt)
    signal = 1.0 + np.sin(2 * np.pi * t / 0.5)  # one full cycle every 0.5 s
    period, strength = T._autocorr_period(signal, dt)
    assert period is not None
    assert abs(period - 0.5) < 0.12, period
    assert strength > 0.3


def test_autocorr_period_constant_has_no_stride():
    period, strength = T._autocorr_period(np.full(60, 3.0), 0.05)
    assert period is None
    assert strength == 0.0


def test_autocorr_period_too_short():
    period, strength = T._autocorr_period(np.array([1.0, 2.0, 1.0]), 0.05)
    assert period is None


def test_quadruped_keypoint_schema():
    from v2r.agentic import superanimal as SA

    assert len(SA.QUADRUPED_KEYPOINTS) == 39
    assert len(SA.KP_INDEX) == 39
    # the four paws and spine anchors used for gait/posture must exist
    for name in SA.PAW_KEYPOINTS + SA.SPINE_KEYPOINTS:
        assert name in SA.KP_INDEX, name
    assert SA.KP_INDEX["nose"] == 0


def test_animal_pose_missing_weights_degrades(tmp_path: Path, sample_video: Path):
    # with no SuperAnimal weights, the tool must degrade gracefully, not raise
    import shutil

    cfg = V2RConfig.load(Path(__file__).resolve().parents[1])
    cfg.root = tmp_path  # empty assets dir -> weights absent
    w = EpisodeWorkspace(tmp_path / "workspaces", "animal_test_000000").create()
    shutil.copy2(sample_video, w.video_path)
    out = T.animal_pose(w.video_path, w, cfg)
    assert out["available"] is False
    assert "missing" in out["reason"].lower() or "not importable" in out["reason"].lower()
