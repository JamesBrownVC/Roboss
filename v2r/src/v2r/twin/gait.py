"""Extract a sagittal-plane gait signal from SuperAnimal dog keypoints.

Output (DogGait) is expressed in a body-centered, body-length-normalized 2D
frame so it can be compared against the Go2 foot trajectories regardless of
image scale, camera distance, or where the dog is in frame:

  x axis = fore-aft, from tail_base -> back_base (dog's forward)
  y axis = vertical (image up)
  origin = mid-hip (mean of the four 'thai' keypoints)
  unit   = body length |back_base - tail_base|

Per leg we recover:
  paw_xy(t)   normalized paw position (fore-aft, vertical)
  thigh_ang(t) image-plane angle of the thigh segment (thai->knee)
  knee_ang(t)  interior knee angle (thai->knee->paw)
and globally: stride period, per-leg swing phase, body forward speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# SuperAnimal-Quadruped leg keypoints -> Go2 leg id
LEG_KEYPOINTS = {
    "FL": ("front_left_thai", "front_left_knee", "front_left_paw"),
    "FR": ("front_right_thai", "front_right_knee", "front_right_paw"),
    "RL": ("back_left_thai", "back_left_knee", "back_left_paw"),
    "RR": ("back_right_thai", "back_right_knee", "back_right_paw"),
}
LEGS = ("FL", "FR", "RL", "RR")
_FORE = ("back_base", "neck_base")      # forward-of-body anchors (fallbacks)
_HIND = ("tail_base", "back_end")       # rear-of-body anchors


@dataclass
class DogGait:
    t: np.ndarray                              # (T,) seconds, canonical
    fps: float
    body_frame_ok: bool
    paw: dict[str, np.ndarray]                 # leg -> (T, 2) normalized fore-aft, vertical
    paw_conf: dict[str, np.ndarray]            # leg -> (T,)
    thigh_ang: dict[str, np.ndarray]           # leg -> (T,) radians (image plane)
    knee_ang: dict[str, np.ndarray]            # leg -> (T,) radians interior
    stride_period_s: float
    swing_phase: dict[str, float]              # leg -> phase in [0,1) of stride
    body_speed_bl_s: float                     # forward speed, body-lengths/s
    duty_factor: dict[str, float]              # leg -> stance fraction
    gait_label: str
    meta: dict = field(default_factory=dict)


def _wide(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (t, {kp: (T,2) uv}, {kp: (T,) conf}) resampled to frame order."""
    frames = np.sort(df["frame"].unique())
    t_by_frame = df.groupby("frame")["t"].first().reindex(frames).to_numpy()
    uv: dict[str, np.ndarray] = {}
    conf: dict[str, np.ndarray] = {}
    fidx = {f: i for i, f in enumerate(frames)}
    for kp, g in df.groupby("keypoint_name"):
        arr = np.full((len(frames), 2), np.nan)
        cf = np.zeros(len(frames))
        for row in g.itertuples(index=False):
            i = fidx[row.frame]
            arr[i] = (row.u, row.v)
            cf[i] = row.conf
        uv[kp] = arr
        conf[kp] = cf
    return t_by_frame, uv, conf


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k or k < 2:
        return x
    kern = np.ones(k) / k
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kern, mode="valid")[: len(x)]


def _dominant_period(sig: np.ndarray, fps: float) -> float:
    """Dominant period (s) of a 1D signal via FFT, ignoring DC."""
    s = sig - np.nanmean(sig)
    s = np.nan_to_num(s)
    if len(s) < 6 or np.allclose(s, 0):
        return 0.0
    freqs = np.fft.rfftfreq(len(s), d=1.0 / fps)
    mag = np.abs(np.fft.rfft(s))
    mag[0] = 0.0
    lo = freqs > 0.3  # ignore < 0.3 Hz (drift)
    if not lo.any():
        return 0.0
    k = np.argmax(mag * lo)
    f = freqs[k]
    return float(1.0 / f) if f > 1e-6 else 0.0


def extract_gait(keypoints_parquet: Path, min_conf: float = 0.3) -> DogGait:
    df = pd.read_parquet(keypoints_parquet)
    t, uv, conf = _wide(df)
    T = len(t)
    fps = float(1.0 / np.median(np.diff(t))) if T > 1 else 30.0

    # body frame axes per frame -----------------------------------------
    def anchor(names):
        for n in names:
            if n in uv and np.isfinite(uv[n]).all(axis=1).mean() > 0.5:
                return uv[n]
        return None

    fore = anchor(_FORE)
    hind = anchor(_HIND)
    thai = [uv[LEG_KEYPOINTS[l][0]] for l in LEGS if LEG_KEYPOINTS[l][0] in uv]
    hips = np.nanmean(np.stack(thai, axis=0), axis=0) if thai else None

    body_ok = fore is not None and hind is not None and hips is not None
    if body_ok:
        axis = fore - hind                                   # (T,2) forward
        blen = np.linalg.norm(axis, axis=1, keepdims=True)
        blen = np.where(blen < 1e-6, np.nan, blen)
        x_hat = axis / blen                                  # forward unit
        y_hat = np.stack([-x_hat[:, 1], x_hat[:, 0]], axis=1)  # +90deg (image up-ish)
        body_len = np.nanmedian(blen)
    else:
        x_hat = np.tile([1.0, 0.0], (T, 1))
        y_hat = np.tile([0.0, -1.0], (T, 1))   # image y grows downward
        hips = np.zeros((T, 2))
        body_len = 1.0

    def to_body(p: np.ndarray) -> np.ndarray:
        rel = p - hips
        fx = np.einsum("ti,ti->t", rel, x_hat)
        fy = np.einsum("ti,ti->t", rel, y_hat)
        return np.stack([fx, fy], axis=1) / (body_len if body_ok else 1.0)

    paw, paw_conf, thigh_ang, knee_ang = {}, {}, {}, {}
    for leg in LEGS:
        kt, kk, kp = LEG_KEYPOINTS[leg]
        if not all(k in uv for k in (kt, kk, kp)):
            continue
        pw = to_body(uv[kp])
        pw[:, 0] = _smooth(pw[:, 0]); pw[:, 1] = _smooth(pw[:, 1])
        paw[leg] = pw
        paw_conf[leg] = np.minimum.reduce([conf[kt], conf[kk], conf[kp]])
        thigh_vec = uv[kk] - uv[kt]
        shank_vec = uv[kp] - uv[kk]
        thigh_ang[leg] = _smooth(np.arctan2(thigh_vec[:, 1], thigh_vec[:, 0]))
        cos = np.einsum("ti,ti->t", -thigh_vec, shank_vec) / (
            np.linalg.norm(thigh_vec, axis=1) * np.linalg.norm(shank_vec, axis=1) + 1e-9)
        knee_ang[leg] = _smooth(np.arccos(np.clip(cos, -1, 1)))

    # stride timing from the strongest paw vertical signal --------------
    period = 0.0
    ref_leg = None
    best = -1.0
    for leg, pw in paw.items():
        amp = np.nanstd(pw[:, 1])
        if amp > best:
            best, ref_leg = amp, leg
    if ref_leg is not None:
        period = _dominant_period(paw[ref_leg][:, 1], fps)

    swing_phase, duty = {}, {}
    for leg, pw in paw.items():
        v = pw[:, 1]
        thr = np.nanmedian(v)
        swing = v > thr           # paw raised = swing (image up)
        duty[leg] = float(1.0 - np.nanmean(swing))
        if ref_leg is not None and period > 0:
            lag = _phase_lag(paw[ref_leg][:, 1], v)
            swing_phase[leg] = float((lag / (period * fps)) % 1.0)
        else:
            swing_phase[leg] = 0.0

    # forward body speed (body-lengths/sec) from hip translation --------
    if body_ok and T > 2:
        disp = np.linalg.norm(np.diff(hips, axis=0), axis=1) / body_len
        body_speed = float(np.nanmedian(disp) * fps)
    else:
        body_speed = 0.0

    label = _classify_gait(period, body_speed, paw)
    return DogGait(
        t=t, fps=fps, body_frame_ok=body_ok,
        paw=paw, paw_conf=paw_conf, thigh_ang=thigh_ang, knee_ang=knee_ang,
        stride_period_s=period, swing_phase=swing_phase,
        body_speed_bl_s=body_speed, duty_factor=duty, gait_label=label,
        meta={"n_frames": T, "fps": round(fps, 2), "body_len_norm": float(body_len),
              "ref_leg": ref_leg, "legs_tracked": sorted(paw)},
    )


def _phase_lag(ref: np.ndarray, sig: np.ndarray) -> float:
    a = np.nan_to_num(ref - np.nanmean(ref))
    b = np.nan_to_num(sig - np.nanmean(sig))
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    corr = np.correlate(b, np.concatenate([a, a]), mode="valid")[: len(a)]
    return float(np.argmax(corr))


def _classify_gait(period: float, speed_bl_s: float, paw: dict) -> str:
    if not paw or period <= 0:
        return "stand" if speed_bl_s < 0.05 else "walk"
    if speed_bl_s < 0.05:
        return "stand"
    if period > 0.7 and speed_bl_s < 0.9:
        return "walk"
    if speed_bl_s > 2.2:
        return "gallop"
    return "trot"
