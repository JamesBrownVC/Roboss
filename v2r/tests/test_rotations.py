"""Rotation convention tests (wxyz storage, scipy conversions)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2r.schema.rotations import (
    axis_angle_to_quat,
    matrix_to_quat,
    quat_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


def test_wxyz_xyzw_roundtrip():
    q_w = np.array([0.6, 0.1, 0.2, 0.3])
    q_w = q_w / np.linalg.norm(q_w)
    q_x = quat_wxyz_to_xyzw(q_w)
    back = quat_xyzw_to_wxyz(q_x)
    assert np.allclose(q_w, back, atol=1e-8)


def test_matrix_quat_roundtrip():
    R = Rotation.from_euler("xyz", [0.1, -0.2, 0.5]).as_matrix()
    q = matrix_to_quat(R)
    R2 = quat_to_matrix(q)
    assert np.allclose(R, R2, atol=1e-6)


def test_axis_angle_quat():
    aa = np.array([0.0, 0.0, 0.5])
    q = axis_angle_to_quat(aa)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6
    assert q[0] >= 0  # canonical w >= 0
