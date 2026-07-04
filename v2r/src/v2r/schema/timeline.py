"""Canonical 30 Hz timeline and resampling with interpolation flags.

Rules (conventions.md + master prompt section 11):
- Original timestamps are kept; resampled values carry an ``interpolated`` flag.
- No interpolation across gaps longer than ``max_gap_s``: those target samples
  are marked invalid instead of fabricated.
"""

from __future__ import annotations

import numpy as np

from .rotations import normalize_quat, quat_slerp_wxyz

EXACT_TOL_S = 1e-6


def canonical_timestamps(duration_s: float, hz: float = 30.0) -> np.ndarray:
    """Timestamps [0, 1/hz, 2/hz, ...] covering [0, duration_s]."""
    n = int(np.floor(duration_s * hz + 1e-9)) + 1
    return np.arange(n, dtype=np.float64) / hz


def check_monotonic(t: np.ndarray, strict: bool = True) -> bool:
    t = np.asarray(t, dtype=np.float64)
    d = np.diff(t)
    return bool(np.all(d > 0) if strict else np.all(d >= 0))


def _bracket(t_src: np.ndarray, t_dst: np.ndarray):
    """For each target time: indices (i0, i1), alpha, exact flag, in-range flag."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    if len(t_src) < 1:
        raise ValueError("empty source timeline")
    if not check_monotonic(t_src, strict=True):
        raise ValueError("source timestamps must be strictly increasing")

    idx = np.searchsorted(t_src, t_dst, side="left")
    in_range = (t_dst >= t_src[0] - EXACT_TOL_S) & (t_dst <= t_src[-1] + EXACT_TOL_S)
    idx = np.clip(idx, 0, len(t_src) - 1)
    # exact hits (within tolerance) at idx or idx-1
    exact_hi = np.abs(t_src[idx] - t_dst) <= EXACT_TOL_S
    idx_lo = np.clip(idx - 1, 0, len(t_src) - 1)
    exact_lo = np.abs(t_src[idx_lo] - t_dst) <= EXACT_TOL_S
    exact = exact_hi | exact_lo
    nearest = np.where(exact_hi, idx, idx_lo)

    i1 = np.clip(idx, 1, len(t_src) - 1) if len(t_src) > 1 else np.zeros_like(idx)
    i0 = i1 - 1 if len(t_src) > 1 else np.zeros_like(idx)
    denom = np.where(len(t_src) > 1, t_src[i1] - t_src[i0], 1.0)
    denom = np.where(denom <= 0, 1.0, denom)
    alpha = np.clip((t_dst - t_src[i0]) / denom, 0.0, 1.0)
    gap = t_src[i1] - t_src[i0] if len(t_src) > 1 else np.zeros_like(alpha)
    return i0, i1, alpha, exact, nearest, in_range, gap


def resample_linear(
    t_src: np.ndarray,
    values: np.ndarray,
    t_dst: np.ndarray,
    max_gap_s: float = 0.34,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear resampling. Returns (values_dst, interpolated, valid).

    values: (N, ...) float array aligned with t_src.
    Targets outside [t_src[0], t_src[-1]] or bridging a gap > max_gap_s
    are valid=False (values copied from nearest sample, do not use).
    """
    values = np.asarray(values, dtype=np.float64)
    i0, i1, alpha, exact, nearest, in_range, gap = _bracket(t_src, t_dst)

    a = alpha.reshape(alpha.shape + (1,) * (values.ndim - 1))
    out = (1.0 - a) * values[i0] + a * values[i1]
    out[exact] = values[nearest[exact]]

    interpolated = ~exact & in_range
    gap_ok = gap <= max_gap_s + EXACT_TOL_S
    valid = in_range & (exact | gap_ok)
    # values for invalid targets: nearest sample (flagged invalid)
    bad = ~valid
    out[bad] = values[nearest[bad]]
    interpolated = interpolated & valid
    return out, interpolated, valid


def resample_quat_wxyz(
    t_src: np.ndarray,
    quats_wxyz: np.ndarray,
    t_dst: np.ndarray,
    max_gap_s: float = 0.34,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slerp resampling for (N, 4) wxyz quaternions. Same contract as resample_linear."""
    q = normalize_quat(np.asarray(quats_wxyz, dtype=np.float64))
    i0, i1, alpha, exact, nearest, in_range, gap = _bracket(t_src, t_dst)

    out = quat_slerp_wxyz(q[i0], q[i1], alpha)
    out[exact] = q[nearest[exact]]

    interpolated = ~exact & in_range
    gap_ok = gap <= max_gap_s + EXACT_TOL_S
    valid = in_range & (exact | gap_ok)
    bad = ~valid
    out[bad] = q[nearest[bad]]
    interpolated = interpolated & valid
    return out, interpolated, valid
