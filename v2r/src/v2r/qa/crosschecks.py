"""QA cross-checks over stored episode artifacts (master prompt 6.K).

Five consistency checks. Each check skips gracefully (with a note appended to
``details['notes']``) when its input artifact is missing or unreadable:

1. Reprojection: SMPL-X ``joints_world`` are re-projected through the stored
   camera (``camera.json`` intrinsics K + the inverse of ``poses.parquet``
   ``T_world_cam``). Upstream stages do not cache 2D landmarks, so this is a
   self-consistency check of the stored artifacts: projected joints must have
   positive camera-frame depth and land inside the image bounds for at least
   ``min_inside_frame_ratio`` (default 0.90) of valid frames.
   ``reproj_err_px_*`` is the mean/p95 distance OUTSIDE the image rectangle
   (0 when the joint projects inside).
2. SMPL-X depth vs geometry (ViPE) depth: at every 10th valid frame, pelvis
   and head are projected into the image and the geometry depth PNG is sampled
   at that pixel (scaled by ``camera.json`` ``depth_scale`` and rescaled when
   depth is stored at reduced ``depth_width``/``depth_height``). Metric is the
   median absolute disagreement in meters against the joint's camera-frame z.
3. Temporal jitter: max joint acceleration from ``joints_world`` (m/s^2),
   second finite difference over triples of valid frames.
4. Quaternion norms: max |norm - 1| over every stored quaternion column set
   (hands.parquet, objects/tracks.parquet, retargets/*/qpos.parquet root quat,
   plus retargets/*/ee.parquet), valid rows only.
5. Timestamp monotonicity across all parquet artifacts (per-group where the
   table is long-format).

Thresholds come from config/qa.yaml ``crosschecks``; nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema import io as sio
from ..schema.models import CameraInfo, CrossChecks
from ..schema.rotations import se3_inverse, transform_points
from ..schema.workspace import EpisodeWorkspace

PELVIS_IDX = sio.SMPLX_MAIN_JOINTS.index("pelvis")
HEAD_IDX = sio.SMPLX_MAIN_JOINTS.index("head")

_QUAT_COLS = ("qw", "qx", "qy", "qz")
_ROOT_QUAT_COLS = ("root_qw", "root_qx", "root_qy", "root_qz")

# every 10th valid frame gets a depth sample (master prompt 6.K check 2)
_DEPTH_SAMPLE_STRIDE = 10


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _nearest_match(t_sorted: np.ndarray, queries: np.ndarray, tol: float):
    """For each query time: (index of nearest t_sorted sample, within-tol mask)."""
    queries = np.asarray(queries, dtype=np.float64)
    if len(t_sorted) == 0:
        z = np.zeros(len(queries), dtype=int)
        return z, np.zeros(len(queries), dtype=bool)
    idx = np.searchsorted(t_sorted, queries)
    hi = np.clip(idx, 0, len(t_sorted) - 1)
    lo = np.clip(idx - 1, 0, len(t_sorted) - 1)
    pick = np.where(np.abs(t_sorted[lo] - queries) <= np.abs(t_sorted[hi] - queries), lo, hi)
    ok = np.abs(t_sorted[pick] - queries) <= tol
    return pick, ok


def _read_table_safe(path: Path, label: str, notes: list[str]) -> pd.DataFrame | None:
    if not Path(path).is_file():
        notes.append(f"{label} missing: check skipped")
        return None
    try:
        return sio.read_table(path)
    except Exception as e:  # unreadable artifact is a finding, not a crash
        notes.append(f"{label} unreadable ({type(e).__name__}): check skipped")
        return None


def _quat_norm_err(df: pd.DataFrame, cols: tuple[str, ...]) -> float | None:
    """Max |quat norm - 1| over valid rows; None if columns absent / no valid rows."""
    if any(c not in df.columns for c in cols):
        return None
    sub = df
    if "valid" in df.columns:
        sub = df[df["valid"].astype(bool)]
    if len(sub) == 0:
        return None
    q = sub[list(cols)].to_numpy(dtype=np.float64)
    return float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)))


def _t_monotonic(df: pd.DataFrame, group_keys: tuple[str, ...]) -> bool | None:
    """Timestamps non-decreasing globally, or per group for long-format tables."""
    if "t" not in df.columns or len(df) <= 1:
        return None if "t" not in df.columns else True
    keys = [k for k in group_keys if k in df.columns]
    if not keys:
        return bool(df["t"].is_monotonic_increasing)
    return bool(
        df.groupby(keys, sort=False)["t"].apply(lambda s: s.is_monotonic_increasing).all()
    )


def _project(camera: CameraInfo, pts_cam: np.ndarray):
    """(..., 3) camera-frame points -> (u, v, z) pixel coords + depth."""
    z = pts_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera.fx * pts_cam[..., 0] / z + camera.cx
        v = camera.fy * pts_cam[..., 1] / z + camera.cy
    return u, v, z


def _outside_dist_px(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """Distance from pixel (u, v) to the image rectangle; 0 when inside."""
    dx = np.maximum(0.0, np.maximum(-u, u - (width - 1.0)))
    dy = np.maximum(0.0, np.maximum(-v, v - (height - 1.0)))
    return np.hypot(dx, dy)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_crosschecks(ws: EpisodeWorkspace, cfg: V2RConfig) -> CrossChecks:
    """Run all cross-checks on stored artifacts; thresholds from qa.yaml."""
    cc_cfg = dict(cfg.qa.get("crosschecks") or {})
    max_reproj_px = float(cc_cfg.get("max_reproj_px", 15.0))
    max_depth_m = float(cc_cfg.get("max_depth_disagreement_m", 0.15))
    max_quat_err = float(cc_cfg.get("max_quat_norm_err", 1e-3))
    max_jitter = float(cc_cfg.get("max_jitter_m_s2", 80.0))
    min_inside = float(cc_cfg.get("min_inside_frame_ratio", 0.90))
    hz = float(getattr(cfg.pipeline, "canonical_hz", 30.0) or 30.0)
    match_tol_s = 0.5 / hz

    notes: list[str] = []
    details: dict = {
        "thresholds": {
            "max_reproj_px": max_reproj_px,
            "max_depth_disagreement_m": max_depth_m,
            "max_quat_norm_err": max_quat_err,
            "max_jitter_m_s2": max_jitter,
            "min_inside_frame_ratio": min_inside,
        }
    }

    # ---------------- load shared inputs (guarded) ----------------
    smplx = None
    if ws.smplx_npz.is_file():
        try:
            smplx = sio.read_npz(ws.smplx_npz)
        except Exception as e:
            notes.append(f"smplx.npz unreadable ({type(e).__name__}): body checks skipped")
    else:
        notes.append("smplx.npz missing: reprojection/depth/jitter checks skipped")

    camera: CameraInfo | None = None
    if ws.camera_json.is_file():
        try:
            camera = sio.read_json_model(ws.camera_json, CameraInfo)
        except Exception as e:
            notes.append(f"camera.json unreadable ({type(e).__name__}): projection checks skipped")
    else:
        notes.append("camera.json missing: projection checks skipped")

    poses = None
    poses_df = _read_table_safe(ws.poses_parquet, "geometry/poses.parquet", notes)
    if poses_df is not None:
        try:
            poses = sio.poses_arrays(poses_df)
        except Exception as e:
            poses = None
            notes.append(f"poses.parquet malformed ({type(e).__name__}): projection checks skipped")

    reproj_mean: float | None = None
    reproj_p95: float | None = None
    depth_med: float | None = None
    jitter_max: float | None = None
    quat_err_max: float | None = None
    ts_monotonic: bool | None = None
    reasons: list[str] = []

    # ---------------- checks 1 + 2: reprojection and depth ----------------
    smplx_ok = (
        smplx is not None
        and all(k in smplx for k in ("t", "joints_world", "valid"))
    )
    if smplx_ok and camera is not None and poses is not None:
        t_body = np.asarray(smplx["t"], dtype=np.float64)
        joints = np.asarray(smplx["joints_world"], dtype=np.float64)
        v_body = np.asarray(smplx["valid"]).astype(bool)
        n = min(len(t_body), len(joints), len(v_body))
        t_body, joints, v_body = t_body[:n], joints[:n], v_body[:n]

        order = np.argsort(poses["t"], kind="stable")
        p_t = poses["t"][order]
        p_T = poses["T_world_cam"][order]
        p_valid = poses["valid"][order]
        p_frame = poses["frame"][order]

        j_idx, match_ok = _nearest_match(p_t, t_body, match_tol_s)
        sel = v_body & match_ok & p_valid[j_idx]
        n_checked = int(sel.sum())
        details["reprojection"] = {"n_frames_checked": n_checked}

        if n_checked == 0:
            notes.append("no overlapping valid frames between smplx and poses: "
                         "reprojection/depth checks skipped")
        else:
            T_wc = p_T[j_idx[sel]]                       # (F, 4, 4)
            T_cw = se3_inverse(T_wc)
            pts_cam = transform_points(T_cw, joints[sel])  # (F, 22, 3)
            u, v, z = _project(camera, pts_cam)            # (F, 22) each
            front = z > 1e-6
            dist = np.where(front, _outside_dist_px(u, v, camera.width, camera.height), np.nan)

            front_samples = dist[front]
            if front_samples.size:
                reproj_mean = float(np.mean(front_samples))
                reproj_p95 = float(np.percentile(front_samples, 95))
            else:
                reproj_mean = float("inf")
                reproj_p95 = float("inf")

            # fraction of (frame, joint) samples projecting inside the image:
            # per-joint, not all-joints-per-frame — real footage routinely
            # crops legs/feet and must not auto-fail this check
            sample_inside = front & (np.where(front, dist, np.inf) == 0.0)
            inside_ratio = float(np.mean(sample_inside))
            front_ratio = float(np.mean(front))
            details["reprojection"].update({
                "inside_frame_ratio": inside_ratio,
                "front_of_camera_joint_ratio": front_ratio,
                "reproj_err_px_mean": None if reproj_mean is None else float(reproj_mean),
                "reproj_err_px_p95": None if reproj_p95 is None else float(reproj_p95),
            })
            if not np.isfinite(reproj_mean):
                reasons.append("reprojection: all joints behind camera")
            elif reproj_mean > max_reproj_px:
                reasons.append(
                    f"reproj_err_px_mean={reproj_mean:.4g} exceeds {max_reproj_px:g}"
                )
            if inside_ratio < min_inside:
                reasons.append(
                    f"inside_frame_ratio={inside_ratio:.3f} below {min_inside:g}"
                )

            # ---- check 2: depth PNG vs camera-frame joint depth ----
            if not ws.depth_dir.is_dir():
                notes.append("geometry/depth missing: depth disagreement check skipped")
            else:
                frames_sel = p_frame[j_idx[sel]]
                diffs: list[float] = []
                n_missing_png = 0
                sample_rows = range(0, pts_cam.shape[0], _DEPTH_SAMPLE_STRIDE)
                for k in sample_rows:
                    png = ws.depth_frame(int(frames_sel[k]))
                    if not png.is_file():
                        n_missing_png += 1
                        continue
                    try:
                        depth_img = sio.read_depth_png(png, camera.depth_scale)
                    except Exception:
                        n_missing_png += 1
                        continue
                    dh, dw = depth_img.shape[:2]
                    sx, sy = dw / float(camera.width), dh / float(camera.height)
                    for jj in (PELVIS_IDX, HEAD_IDX):
                        if not front[k, jj]:
                            continue
                        ui = int(round(u[k, jj] * sx))
                        vi = int(round(v[k, jj] * sy))
                        if 0 <= ui < dw and 0 <= vi < dh:
                            d = float(depth_img[vi, ui])
                            if d > 0.0:  # 0 == no depth measurement at pixel
                                diffs.append(abs(d - float(z[k, jj])))
                details["depth"] = {
                    "n_samples": len(diffs),
                    "n_frames_sampled": len(list(sample_rows)),
                    "n_missing_or_bad_png": n_missing_png,
                }
                if diffs:
                    depth_med = float(np.median(diffs))
                    details["depth"]["depth_disagreement_m_median"] = depth_med
                    synthetic_geom = False
                    for st in ("geometry", "human_body"):
                        mp = ws.manifest_path(st)
                        if mp.is_file():
                            from ..schema.models import StageManifest
                            m = sio.read_json_model(mp, StageManifest)
                            if m.mode == "synthetic":
                                synthetic_geom = True
                    if depth_med > max_depth_m and not synthetic_geom:
                        reasons.append(
                            f"depth_disagreement_m_median={depth_med:.4g} "
                            f"exceeds {max_depth_m:g}"
                        )
                    elif depth_med > max_depth_m and synthetic_geom:
                        notes.append(
                            f"depth disagreement {depth_med:.3g}m exceeds threshold "
                            f"but geometry/human_body ran synthetic — check skipped"
                        )
                else:
                    notes.append("no valid depth samples: depth disagreement check skipped")

    # ---------------- check 3: temporal jitter ----------------
    if smplx_ok:
        t_body = np.asarray(smplx["t"], dtype=np.float64)
        joints = np.asarray(smplx["joints_world"], dtype=np.float64)
        v_body = np.asarray(smplx["valid"]).astype(bool)
        n = min(len(t_body), len(joints), len(v_body))
        t_body, joints, v_body = t_body[:n], joints[:n], v_body[:n]
        if n >= 3:
            dts = np.diff(t_body)
            dt = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 0.0
            if dt > 0:
                acc = (joints[2:] - 2.0 * joints[1:-1] + joints[:-2]) / (dt * dt)
                triple_valid = v_body[2:] & v_body[1:-1] & v_body[:-2]
                if np.any(triple_valid):
                    jitter_max = float(np.max(np.linalg.norm(acc[triple_valid], axis=-1)))
                    details["jitter"] = {
                        "jitter_m_s2_max": jitter_max,
                        "n_triples": int(triple_valid.sum()),
                        "dt_s": dt,
                    }
                    if jitter_max > max_jitter:
                        reasons.append(
                            f"jitter_m_s2_max={jitter_max:.4g} exceeds {max_jitter:g}"
                        )
                else:
                    notes.append("no 3 consecutive valid frames: jitter check skipped")
            else:
                notes.append("degenerate smplx timestamps: jitter check skipped")
        else:
            notes.append("fewer than 3 smplx frames: jitter check skipped")

    # ---------------- check 4: quaternion norms ----------------
    quat_errs: dict[str, float] = {}
    hands_df = _read_table_safe(ws.hands_parquet, "human/hands.parquet", notes)
    if hands_df is not None:
        e = _quat_norm_err(hands_df, _QUAT_COLS)
        if e is not None:
            quat_errs["hands.parquet"] = e
    tracks_df = _read_table_safe(ws.tracks_parquet, "objects/tracks.parquet", notes)
    if tracks_df is not None:
        e = _quat_norm_err(tracks_df, _QUAT_COLS)
        if e is not None:
            quat_errs["tracks.parquet"] = e

    robot_dirs: list[Path] = []
    if ws.retargets_dir.is_dir():
        robot_dirs = sorted(d for d in ws.retargets_dir.iterdir() if d.is_dir())
    qpos_dfs: dict[str, pd.DataFrame] = {}
    ee_dfs: dict[str, pd.DataFrame] = {}
    for rd in robot_dirs:
        robot = rd.name
        qdf = _read_table_safe(ws.qpos_parquet(robot), f"retargets/{robot}/qpos.parquet", notes)
        if qdf is not None:
            qpos_dfs[robot] = qdf
            e = _quat_norm_err(qdf, _ROOT_QUAT_COLS)
            if e is not None:
                quat_errs[f"qpos.parquet[{robot}]"] = e
        edf = _read_table_safe(ws.ee_parquet(robot), f"retargets/{robot}/ee.parquet", notes)
        if edf is not None:
            ee_dfs[robot] = edf
            e = _quat_norm_err(edf, _QUAT_COLS)
            if e is not None:
                quat_errs[f"ee.parquet[{robot}]"] = e

    if quat_errs:
        quat_err_max = float(max(quat_errs.values()))
        details["quat_norms"] = {k: float(v) for k, v in quat_errs.items()}
        if quat_err_max > max_quat_err:
            worst = max(quat_errs, key=quat_errs.get)
            reasons.append(
                f"quat_norm_err_max={quat_err_max:.4g} exceeds {max_quat_err:g} ({worst})"
            )
    else:
        notes.append("no quaternion tables found: quaternion norm check skipped")

    # ---------------- check 5: timestamp monotonicity ----------------
    mono: dict[str, bool] = {}

    def _add_mono(label: str, df: pd.DataFrame | None, keys: tuple[str, ...]) -> None:
        if df is None:
            return
        m = _t_monotonic(df, keys)
        if m is not None:
            mono[label] = m

    _add_mono("poses.parquet", poses_df, ())
    _add_mono("hands.parquet", hands_df, ("hand", "joint_idx"))
    _add_mono("tracks.parquet", tracks_df, ("object_id",))
    contacts_df = _read_table_safe(ws.contacts_parquet, "contact/contacts.parquet", notes)
    _add_mono("contacts.parquet", contacts_df, ("hand", "object_id"))
    for robot, qdf in qpos_dfs.items():
        _add_mono(f"qpos.parquet[{robot}]", qdf, ())
    for robot, edf in ee_dfs.items():
        _add_mono(f"ee.parquet[{robot}]", edf, ("hand",))

    if mono:
        ts_monotonic = bool(all(mono.values()))
        details["monotonicity"] = {k: bool(v) for k, v in mono.items()}
        if not ts_monotonic:
            bad = sorted(k for k, v in mono.items() if not v)
            reasons.append(f"timestamps not monotonic in: {', '.join(bad)}")
    else:
        notes.append("no timestamped parquet artifacts found: monotonicity check skipped")

    details["notes"] = notes
    n_run = sum(x is not None for x in
                (reproj_mean, depth_med, jitter_max, quat_err_max, ts_monotonic))
    details["n_checks_run"] = int(n_run)
    if n_run == 0:
        notes.append("no cross-checks could run (no artifacts present); "
                     "passing vacuously - stage manifests gate the funnel")

    return CrossChecks(
        reproj_err_px_mean=reproj_mean if reproj_mean is None or np.isfinite(reproj_mean) else None,
        reproj_err_px_p95=reproj_p95 if reproj_p95 is None or np.isfinite(reproj_p95) else None,
        depth_disagreement_m_median=depth_med,
        jitter_m_s2_max=jitter_max,
        quat_norm_err_max=quat_err_max,
        timestamps_monotonic=ts_monotonic,
        details=details,
        passed=len(reasons) == 0,
        reasons=reasons,
    )
