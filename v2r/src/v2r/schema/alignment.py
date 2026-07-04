"""Similarity-transform alignment (Umeyama) for world-frame harmonization.

Used to align GVHMR's world frame onto ViPE's world frame via their camera
trajectories (master prompt 6.C) and for multi-view extrinsic checks.
"""

from __future__ import annotations

import numpy as np

from .models import FusionReport, Sim3
from .rotations import matrix_to_quat


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform: dst ~= s * R @ src + t.

    src, dst: (N, 3). Returns (s, R (3,3), t (3,)).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching (N,3) arrays, got {src.shape} vs {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("umeyama needs >= 3 point pairs")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d

    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_s = (xs ** 2).sum() / n
        if var_s < 1e-12:
            raise ValueError("degenerate source points (zero variance)")
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0

    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """x' = s * R @ x + t for (N, 3) points."""
    return s * (np.asarray(pts, dtype=np.float64) @ np.asarray(R).T) + np.asarray(t)


def apply_sim3_to_se3(s: float, R: np.ndarray, t: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a Sim(3) frame change to (..., 4, 4) rigid transforms.

    Rotation part is rotated (no scale: rotations stay orthonormal);
    translation part gets the full similarity transform.
    """
    T = np.asarray(T, dtype=np.float64)
    out = T.copy()
    out[..., :3, :3] = np.asarray(R) @ T[..., :3, :3]
    out[..., :3, 3] = s * (T[..., :3, 3] @ np.asarray(R).T) + np.asarray(t)
    return out


def align_trajectories(
    src_positions: np.ndarray, dst_positions: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray, FusionReport]:
    """Umeyama + residual report (rms, p95) for fusion_report.json."""
    s, R, t = umeyama(src_positions, dst_positions, with_scale=with_scale)
    aligned = apply_sim3(s, R, t, src_positions)
    residuals = np.linalg.norm(aligned - np.asarray(dst_positions, dtype=np.float64), axis=1)
    report = FusionReport(
        sim3=Sim3(
            scale=float(s),
            quat_wxyz=tuple(matrix_to_quat(R).tolist()),
            translation=tuple(np.asarray(t, dtype=float).tolist()),
        ),
        rms_residual_m=float(np.sqrt(np.mean(residuals ** 2))),
        p95_residual_m=float(np.percentile(residuals, 95)),
        n_frames=int(len(residuals)),
    )
    return s, R, t, report
