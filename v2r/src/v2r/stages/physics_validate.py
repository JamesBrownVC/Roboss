"""Physics validate (Stage I): real Tier-1 checks.

Tier 1 (master prompt 6.I): replay qpos and check joint limits,
self-collision, ground penetration, foot slide, velocity/acceleration limits.
With MuJoCo + the robot MJCF available the replay is a real mj_forward pass;
otherwise an honest numpy kinematic fallback runs limit/vel/acc checks and
marks the contact-based checks as skipped. Nothing is hard-coded valid.

Tier 2 (BeyondMimic-style tracking policy) is a documented stub: it requires
Isaac Lab / GPU training and runs on the CUDA host only.
"""

from __future__ import annotations

import numpy as np

from ..schema.io import read_table, write_json_model, write_npz
from ..schema.models import PhysicsCheck, PhysicsReport, StageStatus
from .base import Stage, StageContext, StageResult, register_stage
from .robot_models import resolve_joint, try_load_model

TOOL = {
    "tool": "mujoco",
    "repo": "github.com/google-deepmind/mujoco",
    "commit": "3.10.0",
}

_SELF_COLLISION_TOL = 1e-3  # m of penetration before we call it a violation


@register_stage
class PhysicsValidateStage(Stage):
    name = "physics_validate"

    def run(self, ctx: StageContext) -> StageResult:
        if not ctx.robots:
            return StageResult(status=StageStatus.failed, failure_reason="no robots specified", **TOOL)

        thresholds = ctx.cfg.qa.get("physics", {})
        outputs: list[str] = []
        metrics: dict = {}
        any_valid = False
        for robot in ctx.robots:
            qp = ctx.ws.qpos_parquet(robot)
            if not qp.is_file():
                return StageResult(
                    status=StageStatus.failed,
                    failure_reason=f"missing retargets/{robot}/qpos.parquet",
                    **TOOL,
                )
            df = read_table(qp).sort_values("frame")
            model = try_load_model(ctx.cfg, robot)
            if model is not None:
                report, masks = self._mujoco_tier1(ctx, robot, model, df, thresholds)
            else:
                report, masks = self._kinematic_tier1(ctx, robot, df, thresholds)

            write_json_model(ctx.ws.physics_report_json(robot), report)
            mask_path = ctx.ws.retarget_dir(robot) / "violation_mask.npz"
            write_npz(mask_path, **masks)
            outputs += [ctx.ws.rel(ctx.ws.physics_report_json(robot)), ctx.ws.rel(mask_path)]
            metrics[robot] = {
                "engine": report.engine,
                "physics_valid": report.physics_valid,
                "violation_frame_ratio": round(report.violation_frame_ratio, 4),
            }
            any_valid = any_valid or report.physics_valid

        status = StageStatus.success if any_valid else StageStatus.rejected
        return StageResult(
            status=status,
            metrics={"physics_valid": all(m["physics_valid"] for m in metrics.values()),
                     "per_robot": metrics},
            failure_reason=None if any_valid else "no robot passed Tier-1 physics",
            outputs=outputs,
            **TOOL,
        )

    # ------------------------------------------------------------------
    # shared: velocity / acceleration finite-difference checks
    # ------------------------------------------------------------------

    @staticmethod
    def _vel_acc_checks(t: np.ndarray, q: np.ndarray, thresholds: dict):
        dt = np.diff(t)
        dt[dt <= 0] = np.inf  # non-monotonic timestamps never pass as motion
        vel = np.abs(np.diff(q, axis=0)) / dt[:, None]
        acc = np.abs(np.diff(vel, axis=0)) / dt[1:, None]
        vmax = float(thresholds.get("max_joint_vel_rad_s", 20.0))
        amax = float(thresholds.get("max_joint_acc_rad_s2", 400.0))
        vel_mask = np.zeros(len(t), dtype=bool)
        acc_mask = np.zeros(len(t), dtype=bool)
        if len(vel):
            vel_mask[1:] = (vel > vmax).any(axis=1)
        if len(acc):
            acc_mask[2:] = (acc > amax).any(axis=1)
        checks = {
            "velocity_limits": PhysicsCheck(
                violations=int(vel_mask.sum()),
                max_value=float(vel.max()) if len(vel) else 0.0,
                threshold=vmax,
            ),
            "acceleration_limits": PhysicsCheck(
                violations=int(acc_mask.sum()),
                max_value=float(acc.max()) if len(acc) else 0.0,
                threshold=amax,
            ),
        }
        return checks, {"velocity_limits": vel_mask, "acceleration_limits": acc_mask}

    # ------------------------------------------------------------------
    # MuJoCo replay
    # ------------------------------------------------------------------

    def _mujoco_tier1(self, ctx: StageContext, robot: str, model, df, thresholds):
        import mujoco

        spec = ctx.cfg.robot(robot)
        data = mujoco.MjData(model)
        n = len(df)
        t = df["t"].to_numpy(dtype=np.float64)

        # base pose: 'home' keyframe when present, else qpos0 clipped to range
        base = np.array(model.qpos0, dtype=np.float64)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            base = np.array(model.key_qpos[key_id], dtype=np.float64)

        # dof column -> qpos address + range
        col_map: list[tuple[str, int, float, float, bool]] = []
        unmapped: list[str] = []
        for name in spec.dof:
            jid = resolve_joint(model, name)
            if jid is None:
                unmapped.append(name)
                continue
            adr = int(model.jnt_qposadr[jid])
            limited = bool(model.jnt_limited[jid])
            lo, hi = (model.jnt_range[jid] if limited else (-np.inf, np.inf))
            col_map.append((name, adr, float(lo), float(hi), limited))
        q_cols = df[[c for c, *_ in col_map]].to_numpy(dtype=np.float64)

        floor_geoms = {
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == 0  # geoms attached to <worldbody>
        }
        feet_body_ids = {}
        for link in spec.feet_links:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            if bid >= 0:
                feet_body_ids[link] = bid

        max_pen = float(thresholds.get("max_ground_penetration_m", 0.02))
        max_slide = float(thresholds.get("max_foot_slide_m_per_frame", 0.01))

        limit_mask = np.zeros(n, dtype=bool)
        selfcol_mask = np.zeros(n, dtype=bool)
        ground_mask = np.zeros(n, dtype=bool)
        slide_mask = np.zeros(n, dtype=bool)
        limit_excess_max = 0.0
        selfcol_pen_max = 0.0
        ground_pen_max = 0.0
        slide_max = 0.0
        prev_feet: dict[str, tuple[np.ndarray, bool]] = {}

        for i in range(n):
            data.qpos[:] = base
            for k, (_, adr, _, _, _) in enumerate(col_map):
                data.qpos[adr] = q_cols[i, k]
            mujoco.mj_forward(model, data)

            # joint limits
            for k, (_, adr, lo, hi, limited) in enumerate(col_map):
                if not limited:
                    continue
                v = q_cols[i, k]
                excess = max(lo - v, v - hi)
                if excess > 1e-9:
                    limit_mask[i] = True
                    limit_excess_max = max(limit_excess_max, float(excess))

            # contacts: ground penetration + self collision
            feet_in_contact: set[int] = set()
            for c in range(data.ncon):
                con = data.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                pen = max(0.0, -float(con.dist))
                is_floor = (g1 in floor_geoms) != (g2 in floor_geoms)
                if is_floor:
                    robot_geom = g2 if g1 in floor_geoms else g1
                    body = int(model.geom_bodyid[robot_geom])
                    # walk up to a named foot ancestor
                    b = body
                    while b > 0:
                        if b in feet_body_ids.values():
                            feet_in_contact.add(b)
                            break
                        b = int(model.body_parentid[b])
                    if pen > max_pen:
                        ground_mask[i] = True
                    ground_pen_max = max(ground_pen_max, pen)
                elif g1 not in floor_geoms and g2 not in floor_geoms:
                    if pen > _SELF_COLLISION_TOL:
                        selfcol_mask[i] = True
                    selfcol_pen_max = max(selfcol_pen_max, pen)

            # foot slide while in contact
            for link, bid in feet_body_ids.items():
                pos = data.xpos[bid].copy()
                in_contact = bid in feet_in_contact
                if link in prev_feet:
                    ppos, pcontact = prev_feet[link]
                    if in_contact and pcontact:
                        slide = float(np.linalg.norm(pos[:2] - ppos[:2]))
                        slide_max = max(slide_max, slide)
                        if slide > max_slide:
                            slide_mask[i] = True
                prev_feet[link] = (pos, in_contact)

        vel_acc_checks, vel_acc_masks = self._vel_acc_checks(t, q_cols, thresholds)
        checks = {
            "joint_limits": PhysicsCheck(
                violations=int(limit_mask.sum()), max_value=limit_excess_max, threshold=0.0),
            "self_collision": PhysicsCheck(
                violations=int(selfcol_mask.sum()), max_value=selfcol_pen_max,
                threshold=_SELF_COLLISION_TOL),
            "ground_penetration": PhysicsCheck(
                violations=int(ground_mask.sum()), max_value=ground_pen_max, threshold=max_pen,
                skipped=not feet_body_ids and not floor_geoms,
                note="root held at nominal stance (qpos has no root trajectory)"),
            "foot_slide": PhysicsCheck(
                violations=int(slide_mask.sum()), max_value=slide_max, threshold=max_slide,
                skipped=not feet_body_ids,
                note="" if feet_body_ids else f"feet_links not found in model: {spec.feet_links}"),
            **vel_acc_checks,
        }
        if unmapped:
            checks["joint_mapping"] = PhysicsCheck(
                violations=len(unmapped), skipped=True,
                note=f"dof columns not in model: {unmapped}")

        masks = {"joint_limits": limit_mask, "self_collision": selfcol_mask,
                 "ground_penetration": ground_mask, "foot_slide": slide_mask,
                 **vel_acc_masks}
        any_mask = np.zeros(n, dtype=bool)
        for m in masks.values():
            any_mask |= m
        hard = [c for name, c in checks.items() if not c.skipped]
        report = PhysicsReport(
            robot=robot, tier=1, n_frames=n, engine="mujoco",
            checks=checks,
            physics_valid=all(c.violations == 0 for c in hard),
            violation_frame_ratio=float(any_mask.mean()) if n else 0.0,
            tracking_error=None, premium=False,
        )
        return report, masks

    # ------------------------------------------------------------------
    # numpy kinematic fallback (no mujoco / no MJCF)
    # ------------------------------------------------------------------

    def _kinematic_tier1(self, ctx: StageContext, robot: str, df, thresholds):
        spec = ctx.cfg.robot(robot)
        dof_cols = [c for c in spec.dof if c in df.columns]
        q = df[dof_cols].to_numpy(dtype=np.float64)
        t = df["t"].to_numpy(dtype=np.float64)
        n = len(df)

        limit_mask = np.zeros(n, dtype=bool)
        excess_max = 0.0
        for k, name in enumerate(dof_cols):
            lo, hi = spec.limits_for(name)
            bad = (q[:, k] < lo - 1e-9) | (q[:, k] > hi + 1e-9)
            limit_mask |= bad
            if bad.any():
                excess_max = max(excess_max, float(np.max(np.maximum(lo - q[:, k], q[:, k] - hi))))

        vel_acc_checks, vel_acc_masks = self._vel_acc_checks(t, q, thresholds)
        checks = {
            "joint_limits": PhysicsCheck(
                violations=int(limit_mask.sum()), max_value=excess_max, threshold=0.0,
                note="limits from config/robots.yaml (MJCF unavailable)"),
            "self_collision": PhysicsCheck(skipped=True, note="requires mujoco + MJCF"),
            "ground_penetration": PhysicsCheck(skipped=True, note="requires mujoco + MJCF"),
            "foot_slide": PhysicsCheck(skipped=True, note="requires mujoco + MJCF"),
            **vel_acc_checks,
        }
        masks = {"joint_limits": limit_mask, **vel_acc_masks}
        any_mask = np.zeros(n, dtype=bool)
        for m in masks.values():
            any_mask |= m
        hard = [c for c in checks.values() if not c.skipped]
        report = PhysicsReport(
            robot=robot, tier=1, n_frames=n, engine="kinematic",
            checks=checks,
            physics_valid=all(c.violations == 0 for c in hard),
            violation_frame_ratio=float(any_mask.mean()) if n else 0.0,
            tracking_error=None, premium=False,
        )
        return report, masks


def tier2_tracking_validation(*_args, **_kwargs):
    """Tier-2 (premium flag): train a BeyondMimic-style whole-body tracking
    policy in Isaac Lab / MuJoCo and record tracking error (master prompt 6.I).
    GPU-host only; not available in the Windows dev harness."""
    raise NotImplementedError("Tier-2 tracking validation runs on the CUDA host (Isaac Lab)")
