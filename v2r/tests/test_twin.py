"""Digital-twin motion fitter tests (gait extraction + MuJoCo loss fit)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v2r.config import V2RConfig
from v2r.schema.io import EGODEX_HAND_JOINTS  # noqa: F401 (ensures package import)

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _synth_dog_keypoints(path: Path, n=40, fps=10.0, period=1.0):
    """Fabricate a walking-dog keypoint parquet: legs oscillate out of phase."""
    from v2r.twin.gait import LEG_KEYPOINTS

    rng = np.random.default_rng(0)
    t = np.arange(n) / fps
    phases = {"FL": 0.0, "RR": 0.05, "FR": 0.5, "RL": 0.55}
    rows = []
    body_x0 = 0.3
    for i in range(n):
        march = 0.002 * i
        anchors = {
            "back_base": (0.6 + march, 0.4), "neck_base": (0.62 + march, 0.38),
            "tail_base": (0.4 + march, 0.42), "back_end": (0.42 + march, 0.43),
        }
        for name, (u, v) in anchors.items():
            rows.append(dict(t=t[i], frame=i, keypoint_idx=0, keypoint_name=name,
                             u=u, v=v, conf=0.9, valid=True, source="estimated"))
        for leg, (kt, kk, kp) in LEG_KEYPOINTS.items():
            ph = 2 * np.pi * (t[i] / period + phases[leg])
            hipx = (0.45 if leg in ("FL", "FR") else 0.55) + march
            hipy = 0.42
            paw_u = hipx + 0.05 * np.sin(ph)
            paw_v = hipy + 0.10 + 0.04 * np.maximum(0, np.cos(ph))
            knee_u = hipx + 0.025 * np.sin(ph)
            knee_v = hipy + 0.05
            for kp_name, (u, v) in ((kt, (hipx, hipy)), (kk, (knee_u, knee_v)),
                                    (kp, (paw_u, paw_v))):
                rows.append(dict(t=t[i], frame=i, keypoint_idx=0, keypoint_name=kp_name,
                                 u=u, v=v, conf=0.85, valid=True, source="estimated"))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_gait_extraction(tmp_path):
    from v2r.twin.gait import extract_gait

    kp = tmp_path / "kp.parquet"
    _synth_dog_keypoints(kp, period=1.0)
    gait = extract_gait(kp)
    assert gait.body_frame_ok
    assert set(gait.paw) == {"FL", "FR", "RL", "RR"}
    assert 0.5 < gait.stride_period_s < 2.0        # ~1 s synthesized
    assert gait.gait_label in ("walk", "trot", "stand", "gallop")
    # FL and FR are half a stride out of phase
    assert gait.paw["FL"].shape == (40, 2)


def test_twin_fit_reduces_loss(repo_root, tmp_path):
    from v2r.twin.gait import extract_gait
    from v2r.twin.fitter import fit_twin

    cfg = V2RConfig.load(repo_root)
    model = cfg.root / "assets" / "robots" / "go2" / "scene.xml"
    if not model.is_file():
        pytest.skip("go2 model not present")
    kp = tmp_path / "kp.parquet"
    _synth_dog_keypoints(kp)
    gait = extract_gait(kp)
    fit = fit_twin(gait, model, iters=30, log=lambda *a: None)
    # loss must drop and end well below the anti-correlation wall (2.0)
    assert fit.final_loss < fit.loss_curve[0]
    assert fit.final_loss < 0.5
    # fore-aft foot pattern tracks the paw pattern
    for leg in ("FL", "FR", "RL", "RR"):
        dog = gait.paw[leg][:, 0]
        sim = fit.foot_actual[leg][:, 0]
        c = abs(np.corrcoef(dog, sim)[0, 1])
        assert c > 0.9, f"{leg} fore-aft corr {c:.3f}"
    # qpos is finite and full-length
    assert fit.qpos.shape[0] == len(gait.t)
    assert np.isfinite(fit.qpos).all()
