"""Rotation and rigid-transform utilities.

Storage convention (conventions.md): quaternions are scalar-first ``wxyz``
(MuJoCo convention) in every stored artifact. SciPy is ``xyzw``; that
conversion happens in THIS module and nowhere else. No exceptions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# quaternion order conversion (the ONLY place wxyz <-> xyzw happens)
# ---------------------------------------------------------------------------


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first -> scalar-last."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-last -> scalar-first."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-norm quaternion")
    return q / n


def canonicalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Flip sign so the scalar part is non-negative (deterministic storage)."""
    q = np.asarray(q, dtype=np.float64)
    flip = q[..., 0:1] < 0.0
    return np.where(flip, -q, q)


def enforce_quat_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip signs along axis 0 so consecutive quaternions have positive dot."""
    q = np.array(quats, dtype=np.float64, copy=True)
    for i in range(1, q.shape[0]):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


# ---------------------------------------------------------------------------
# representation conversions (all public interfaces are wxyz / matrix / rotvec)
# ---------------------------------------------------------------------------


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    flat = q.reshape(-1, 4)
    m = Rotation.from_quat(quat_wxyz_to_xyzw(flat)).as_matrix()
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz, canonicalized (w >= 0)."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    q = Rotation.from_matrix(flat).as_quat()  # xyzw
    q = quat_xyzw_to_wxyz(q)
    return canonicalize_quat_wxyz(q).reshape(R.shape[:-2] + (4,))


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector (axis * angle, radians) -> (..., 3, 3)."""
    aa = np.asarray(aa, dtype=np.float64)
    flat = aa.reshape(-1, 3)
    m = Rotation.from_rotvec(flat).as_matrix()
    return m.reshape(aa.shape[:-1] + (3, 3))


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec()
    return aa.reshape(R.shape[:-2] + (3,))


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) rotvec -> (..., 4) wxyz, canonicalized."""
    aa = np.asarray(aa, dtype=np.float64)
    flat = aa.reshape(-1, 3)
    q = Rotation.from_rotvec(flat).as_quat()
    q = canonicalize_quat_wxyz(quat_xyzw_to_wxyz(q))
    return q.reshape(aa.shape[:-1] + (4,))


def quat_to_axis_angle(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    flat = q.reshape(-1, 4)
    aa = Rotation.from_quat(quat_wxyz_to_xyzw(flat)).as_rotvec()
    return aa.reshape(q.shape[:-1] + (3,))


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both wxyz: result rotates by q2 then q1."""
    r1 = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q1, dtype=np.float64).reshape(-1, 4)))
    r2 = Rotation.from_quat(quat_wxyz_to_xyzw(np.asarray(q2, dtype=np.float64).reshape(-1, 4)))
    out = (r1 * r2).as_quat()
    out = canonicalize_quat_wxyz(quat_xyzw_to_wxyz(out))
    shape = np.broadcast_shapes(np.shape(q1), np.shape(q2))
    return out.reshape(shape)


def quat_angle_between_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between two wxyz quaternion arrays."""
    q1 = normalize_quat(np.asarray(q1, dtype=np.float64))
    q2 = normalize_quat(np.asarray(q2, dtype=np.float64))
    d = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha) -> np.ndarray:
    """Vectorized slerp between paired wxyz quats. alpha in [0,1], broadcastable."""
    q0 = normalize_quat(np.asarray(q0, dtype=np.float64))
    q1 = normalize_quat(np.asarray(q1, dtype=np.float64))
    alpha = np.asarray(alpha, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    small = theta < 1e-6
    sin_theta = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / sin_theta)
    w1 = np.where(small, alpha, np.sin(alpha * theta) / sin_theta)
    return normalize_quat(w0 * q0 + w1 * q1)


# ---------------------------------------------------------------------------
# SE(3)
# ---------------------------------------------------------------------------


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(..., 3, 3), (..., 3) -> (..., 4, 4)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    batch = R.shape[:-2]
    T = np.zeros(batch + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_from_quat_pos(q_wxyz: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return make_se3(quat_to_matrix(q_wxyz), pos)


def se3_to_quat_pos(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64)
    return matrix_to_quat(T[..., :3, :3]), T[..., :3, 3].copy()


def se3_inverse(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_se3(Rt, -np.einsum("...ij,...j->...i", Rt, t))


def se3_compose(*Ts: np.ndarray) -> np.ndarray:
    out = np.asarray(Ts[0], dtype=np.float64)
    for T in Ts[1:]:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply (4,4) or batched (...,4,4) transform to (...,N,3) points."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    return np.einsum("...ij,...nj->...ni", T[..., :3, :3], pts) + T[..., None, :3, 3]


def mat44_flatten_rowmajor(T: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 16) row-major (storage layout for parquet)."""
    T = np.asarray(T, dtype=np.float64)
    return T.reshape(T.shape[:-2] + (16,))


def mat44_unflatten_rowmajor(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.float64)
    return flat.reshape(flat.shape[:-1] + (4, 4))


def is_valid_rotation(R: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Boolean check for orthonormal, det=+1 (batched)."""
    R = np.asarray(R, dtype=np.float64)
    eye = np.eye(3)
    orth = np.abs(np.swapaxes(R, -1, -2) @ R - eye).max(axis=(-1, -2)) < tol
    det = np.abs(np.linalg.det(R) - 1.0) < tol
    return orth & det
