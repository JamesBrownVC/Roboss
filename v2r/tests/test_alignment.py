"""Umeyama alignment tests."""

import numpy as np

from v2r.schema.alignment import align_trajectories, apply_sim3


def test_umeyama_identity():
    pts = np.random.randn(20, 3)
    s, R, t, report = align_trajectories(pts, pts, with_scale=True)
    aligned = apply_sim3(s, R, t, pts)
    assert np.allclose(aligned, pts, atol=1e-5)
    assert report.rms_residual_m < 1e-4
