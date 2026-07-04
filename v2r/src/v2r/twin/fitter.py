"""Fit a Go2 joint trajectory to a dog gait by minimizing foot-path tracking
loss inside the MuJoCo twin.

Method:
  1. Turn the dog's normalized paw trajectories into Go2 foot TARGETS in the
     robot's body frame (nominal stance foot + scaled fore-aft / vertical
     excursion, timed by the dog's stride phase).
  2. For each frame, analytic 2-link sagittal IK gives (thigh, calf) joint
     angles hitting each foot target; hip-abduction is left at nominal (2D
     monocular gait has no reliable lateral signal).
  3. Roll the candidate qpos out in the twin (MuJoCo FK), read the ACTUAL foot
     positions, and score tracking loss vs the dog pattern.
  4. scipy optimizes a small parameter vector (fore-aft gain, lift gain, stance
     height, base pitch) to MINIMIZE that rollout loss under joint limits and a
     smoothness penalty. This is the "twin finds the path that minimizes loss."

Output plugs into the existing contract: qpos.parquet (+ csv), a base twist
command channel, and a fit_report the physics_validate / export stages consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .gait import DogGait, LEGS

# Go2 sagittal leg link lengths (m): hip->thigh joint to knee, knee to foot.
L_THIGH = 0.213
L_CALF = 0.213
# joint qpos indices are read from the model; these are the leg dof order
LEG_JOINTS = {leg: (f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint")
              for leg in LEGS}


@dataclass
class TwinFitResult:
    qpos: np.ndarray                    # (T, nq) full model qpos
    t: np.ndarray
    dof_names: list[str]
    foot_target: dict[str, np.ndarray]  # leg -> (T,3) body-frame targets
    foot_actual: dict[str, np.ndarray]  # leg -> (T,3) simulated
    loss_curve: list[float]
    final_loss: float
    params: dict
    base_twist: np.ndarray              # (T,3) vx, vy, yaw_rate (body)
    report: dict = field(default_factory=dict)


def _load_go2(model_path: Path):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = np.array(m.key_qpos[kid]) if kid >= 0 else np.array(m.qpos0)
    return m, d, home


def _foot_bodies(m):
    """calf body id per leg (foot tip = calf origin + calf link down)."""
    import mujoco

    out = {}
    for leg in LEGS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        out[leg] = bid
    return out


def _hip_positions(m, d, foot_calf):
    """Nominal hip (thigh-joint) world positions at home pose."""
    import mujoco

    hips = {}
    for leg in LEGS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_thigh")
        hips[leg] = np.array(d.xpos[bid])
    return hips


def _qadr(m):
    import mujoco

    adr = {}
    for leg in LEGS:
        for j in LEG_JOINTS[leg]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr[j] = int(m.jnt_qposadr[jid])
    return adr


def _sagittal_ik(dx: float, dz: float, thigh_sign: float = 1.0) -> tuple[float, float]:
    """2-link IK in the leg sagittal plane. (dx fore-aft, dz DOWN from hip>0).

    Returns (thigh_joint, calf_joint) in Go2 convention: thigh 0 = straight
    down, positive = swing forward; calf negative = knee folded back.
    """
    r = np.hypot(dx, dz)
    r = min(r, L_THIGH + L_CALF - 1e-3)
    r = max(r, abs(L_THIGH - L_CALF) + 1e-3)
    # knee interior angle via law of cosines
    cos_knee = (L_THIGH ** 2 + L_CALF ** 2 - r ** 2) / (2 * L_THIGH * L_CALF)
    knee = np.arccos(np.clip(cos_knee, -1, 1))          # 0=folded, pi=straight
    calf_joint = -(np.pi - knee)                         # Go2: negative folded
    # thigh angle: point toward foot, minus the knee offset
    alpha = np.arctan2(dx, dz)                           # from straight-down
    cos_beta = (L_THIGH ** 2 + r ** 2 - L_CALF ** 2) / (2 * L_THIGH * r)
    beta = np.arccos(np.clip(cos_beta, -1, 1))
    thigh_joint = alpha + beta
    return float(thigh_joint), float(calf_joint)


def _build_targets(gait: DogGait, hips: dict, stance_h: float,
                   fore_gain: float, lift_gain: float, body_scale: float) -> dict:
    """Go2 foot targets (T,3) in body frame from the dog paw pattern."""
    T = len(gait.t)
    targets = {}
    for leg in LEGS:
        hip = hips[leg]
        base = np.tile([hip[0], hip[1], hip[2] - stance_h], (T, 1)).astype(float)
        if leg in gait.paw:
            pw = gait.paw[leg]
            fore = pw[:, 0] - np.nanmean(pw[:, 0])       # zero-mean fore-aft
            lift = pw[:, 1] - np.nanmedian(pw[:, 1])      # vertical (image up)
            base[:, 0] += np.nan_to_num(fore) * fore_gain * body_scale
            base[:, 2] += np.clip(np.nan_to_num(lift) * lift_gain * body_scale, 0, None)
        targets[leg] = base
    return targets


def _rollout(m, d, home, qadr, targets, hips, foot_calf, base_pitch: float):
    """Set per-frame joint angles from IK to hit targets; return (qpos, foot_actual)."""
    import mujoco

    legs = [l for l in LEGS]
    T = len(next(iter(targets.values())))
    qpos = np.tile(home, (T, 1)).astype(float)
    # base orientation: small pitch about y (wxyz)
    cp, sp = np.cos(base_pitch / 2), np.sin(base_pitch / 2)
    qpos[:, 3:7] = [cp, 0.0, sp, 0.0]
    foot_actual = {leg: np.zeros((T, 3)) for leg in legs}

    lim = {}
    for leg in legs:
        for j in LEG_JOINTS[leg]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            lim[j] = (m.jnt_range[jid][0], m.jnt_range[jid][1])

    for i in range(T):
        for leg in legs:
            hip = hips[leg]
            tgt = targets[leg][i]
            dx = tgt[0] - hip[0]
            dz = hip[2] - tgt[2]                          # downward positive
            thigh, calf = _sagittal_ik(dx, dz)
            _, jt, jc = LEG_JOINTS[leg]
            qpos[i, qadr[jt]] = np.clip(thigh, *lim[jt])
            qpos[i, qadr[jc]] = np.clip(calf, *lim[jc])
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        for leg in legs:
            calf_bid = foot_calf[leg]
            # foot tip = calf body + calf link (0,0,-L_CALF) in calf frame
            R = d.xmat[calf_bid].reshape(3, 3)
            foot_actual[leg][i] = d.xpos[calf_bid] + R @ np.array([0, 0, -L_CALF])
    return qpos, foot_actual


def _wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation in [-1, 1]; 0 if either is flat."""
    w = np.nan_to_num(w)
    if w.sum() < 1e-6:
        return 0.0
    a = np.nan_to_num(a); b = np.nan_to_num(b)
    ma = np.average(a, weights=w); mb = np.average(b, weights=w)
    va = np.average((a - ma) ** 2, weights=w); vb = np.average((b - mb) ** 2, weights=w)
    if va < 1e-9 or vb < 1e-9:
        return 0.0
    cov = np.average((a - ma) * (b - mb), weights=w)
    return float(cov / np.sqrt(va * vb))


def _tracking_loss(gait: DogGait, foot_actual: dict, hips: dict, body_scale: float) -> float:
    """Shape-matching loss in [0, 2]: mean of (1 - correlation) over each leg's
    fore-aft and vertical foot pattern (dog paw vs simulated Go2 foot).
    Sign-sensitive by construction, so a flipped pattern is correctly penalized
    and the optimizer is driven to correct it."""
    corrs, n = 0.0, 0
    for leg in LEGS:
        if leg not in gait.paw or leg not in foot_actual:
            continue
        fa = foot_actual[leg]
        hip = hips[leg]
        sim_fore = fa[:, 0] - hip[0]
        sim_lift = fa[:, 2]
        dog = gait.paw[leg]
        w = np.nan_to_num(gait.paw_conf.get(leg, np.ones(len(fa))))
        corrs += (1.0 - _wcorr(sim_fore, dog[:, 0], w))
        corrs += (1.0 - _wcorr(sim_lift, dog[:, 1], w))
        n += 2
    return corrs / max(n, 1)


def fit_twin(gait: DogGait, model_path: Path, iters: int = 40,
             log=print) -> TwinFitResult:
    import mujoco
    from scipy.optimize import minimize

    m, d, home = _load_go2(Path(model_path))
    mujoco.mj_forward(m, d)
    foot_calf = _foot_bodies(m)
    hips = _hip_positions(m, d, foot_calf)
    qadr = _qadr(m)
    dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(m.njnt)
                 if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) != "floating_base_joint"]
    body_scale = 0.40  # Go2 hip-to-hip ~0.38 m; maps normalized dog units -> metres

    loss_curve: list[float] = []

    # params: fore_gain, lift_gain, stance_h, base_pitch
    def unpack(p):
        return dict(fore_gain=p[0], lift_gain=p[1], stance_h=p[2], base_pitch=p[3])

    def objective(p):
        pr = unpack(p)
        if not (0.05 <= pr["stance_h"] <= 0.34):
            return 1e3
        tgt = _build_targets(gait, hips, pr["stance_h"], pr["fore_gain"],
                             pr["lift_gain"], body_scale)
        qpos, fa = _rollout(m, d, home, qadr, tgt, hips, foot_calf, pr["base_pitch"])
        loss = _tracking_loss(gait, fa, hips, body_scale)
        # smoothness regularizer on joint trajectories
        smooth = float(np.mean(np.diff(qpos[:, 7:], axis=0) ** 2)) * 5.0
        val = loss + smooth
        loss_curve.append(round(val, 5))
        return val

    # multi-seed: fore-aft and lift sign conventions are ambiguous from 2D
    # (image axes vs robot body axes), so seed all four sign combinations and
    # keep the optimum. Correlation loss makes the right signs win.
    seeds = [np.array([fs * 0.9, ls * 0.6, 0.27, 0.0])
             for fs in (1.0, -1.0) for ls in (1.0, -1.0)]
    log(f"[twin] optimizing foot-path tracking loss ({gait.gait_label}, "
        f"stride {gait.stride_period_s:.2f}s, {len(gait.t)} frames, "
        f"{len(seeds)} seeds)...")
    best_res, best_val = None, np.inf
    per_seed = max(8, iters // len(seeds))
    for x0 in seeds:
        r = minimize(objective, x0, method="Nelder-Mead",
                     options={"maxiter": per_seed, "xatol": 1e-3, "fatol": 1e-4})
        if r.fun < best_val:
            best_val, best_res = r.fun, r
    # polish the winning seed with the full budget
    res = minimize(objective, best_res.x, method="Nelder-Mead",
                   options={"maxiter": iters, "xatol": 1e-3, "fatol": 1e-4})
    if best_res.fun < res.fun:
        res = best_res
    best = unpack(res.x)
    tgt = _build_targets(gait, hips, best["stance_h"], best["fore_gain"],
                         best["lift_gain"], body_scale)
    qpos, fa = _rollout(m, d, home, qadr, tgt, hips, foot_calf, best["base_pitch"])
    final_loss = _tracking_loss(gait, fa, hips, body_scale)

    # base twist command channel from the dog's forward speed + stride
    T = len(gait.t)
    vx = np.full(T, gait.body_speed_bl_s * body_scale)   # m/s forward (approx)
    base_twist = np.stack([vx, np.zeros(T), np.zeros(T)], axis=1)

    log(f"[twin] final tracking loss {final_loss:.4f} "
        f"(from {loss_curve[0]:.4f}); {res.nit} iters")
    return TwinFitResult(
        qpos=qpos, t=gait.t, dof_names=dof_names,
        foot_target=tgt, foot_actual=fa,
        loss_curve=loss_curve, final_loss=float(final_loss),
        params=best, base_twist=base_twist,
        report={
            "gait_label": gait.gait_label,
            "stride_period_s": gait.stride_period_s,
            "body_speed_bl_s": gait.body_speed_bl_s,
            "swing_phase": gait.swing_phase,
            "duty_factor": gait.duty_factor,
            "legs_tracked": gait.meta.get("legs_tracked", []),
            "loss_initial": loss_curve[0] if loss_curve else None,
            "loss_final": float(final_loss),
            "loss_reduction_pct": round(100 * (1 - final_loss / loss_curve[0]), 1)
            if loss_curve and loss_curve[0] > 0 else 0.0,
            "optimizer_iters": int(res.nit),
            "params": {k: round(float(v), 4) for k, v in best.items()},
            "body_scale_m": body_scale,
            "caveats": ["2D monocular gait: sagittal pattern only, no true 3D / "
                        "hip abduction", "foot targets are pattern-matched, not "
                        "metric ground truth", "open-loop kinematic fit; a "
                        "tracking controller is required on hardware"],
        },
    )
