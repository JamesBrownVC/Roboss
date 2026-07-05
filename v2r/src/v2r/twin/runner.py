"""Drive the twin-fit end to end for an episode and write contract artifacts.

Consumes: workspaces/{id}/animal/keypoints_superanimal.parquet (from the
`animal_pose` tool). Produces the standard retarget contract for go2 so
physics_validate and the LeRobot export run unchanged:
  retargets/go2/qpos.parquet + qpos.csv
  retargets/go2/cmd_twist.parquet   (base velocity command channel)
  retargets/go2/mapping.json        (provenance = command+kinematic twin-fit)
  retargets/go2/twin_fit_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema.io import write_json_model, write_table
from ..schema.models import RetargetMapping, RobotClass, SourceTag
from ..schema.workspace import EpisodeWorkspace
from .fitter import fit_twin
from .gait import extract_gait


def _physics_status(cfg: V2RConfig, ws: EpisodeWorkspace, robot: str, log) -> dict:
    """Run Tier-1 MuJoCo replay and summarize; failures are EXPECTED for an
    open-loop kinematic gait fit (foot slide during stance) and are reported
    transparently rather than hidden."""
    try:
        from ..stages.base import StageContext
        from ..stages.physics_validate import PhysicsValidateStage

        res = PhysicsValidateStage().run(
            StageContext(ws=ws, cfg=cfg, robots=[robot], mode="synthetic"))
        per = res.metrics.get("per_robot", {}).get(robot, {})
        return {"engine": per.get("engine"), "physics_valid": per.get("physics_valid"),
                "violation_frame_ratio": per.get("violation_frame_ratio"),
                "note": "open-loop kinematic fit; foot-slide/penetration during "
                        "stance is expected without a contact schedule + tracking "
                        "controller. The base-twist channel is the on-robot product."}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def run_twin_fit(cfg: V2RConfig, episode_id: str, robot: str = "go2",
                 iters: int = 40, log=print) -> dict:
    ws = EpisodeWorkspace(cfg.workspaces_root, episode_id)
    kp = ws.root / "animal" / "keypoints_superanimal.parquet"
    if not kp.is_file():
        raise FileNotFoundError(
            f"no animal keypoints at {kp}; run `v2r label --video <dog> --agent loop` "
            "(or the animal_pose tool) first to produce them")

    spec = cfg.robot(robot)
    if spec.robot_class != RobotClass.quadruped:
        raise ValueError(f"twin-fit targets a quadruped; {robot} is {spec.robot_class.value}")
    model_path = spec.model_path
    model_path = Path(model_path) if Path(model_path).is_absolute() else cfg.root / model_path

    log(f"[twin] extracting gait from {ws.rel(kp)}")
    gait = extract_gait(kp)
    log(f"[twin] gait={gait.gait_label} stride={gait.stride_period_s:.2f}s "
        f"speed={gait.body_speed_bl_s:.2f} bl/s legs={gait.meta.get('legs_tracked')}")

    fit = fit_twin(gait, model_path, iters=iters, log=log)

    # ---- write qpos.parquet (contract columns) ----------------------------
    rd = ws.retarget_dir(robot)
    rd.mkdir(parents=True, exist_ok=True)
    T = len(fit.t)
    frames = np.arange(T, dtype=np.int64)
    root = fit.qpos[:, :7]
    conf = np.full(T, 0.6)  # 2D monocular -> moderate confidence
    q = {"t": fit.t, "frame": frames,
         "root_px": root[:, 0], "root_py": root[:, 1], "root_pz": root[:, 2],
         "root_qw": root[:, 3], "root_qx": root[:, 4], "root_qy": root[:, 5], "root_qz": root[:, 6]}
    for k, name in enumerate(fit.dof_names):
        q[name] = fit.qpos[:, 7 + k]
    q["conf"] = conf
    q["valid"] = np.ones(T, dtype=bool)
    q["source"] = np.full(T, SourceTag.estimated.value)
    q["retarget_method"] = np.full(T, "twin_fit_mujoco_v1")
    q["retarget_version"] = np.full(T, "0.1.0")
    q["provenance"] = np.full(T, "command-abstraction")
    qdf = pd.DataFrame(q)
    write_table(qdf, ws.qpos_parquet(robot))
    qdf.to_csv(ws.qpos_csv(robot), index=False)

    # ---- foot trajectories (inspection: dog target vs simulated) ----------
    foot_rows = []
    for leg, fa in fit.foot_actual.items():
        tg = fit.foot_target[leg]
        for i in range(T):
            foot_rows.append({"t": fit.t[i], "frame": int(frames[i]), "leg": leg,
                              "target_x": tg[i, 0], "target_z": tg[i, 2],
                              "actual_x": fa[i, 0], "actual_z": fa[i, 2]})
    pd.DataFrame(foot_rows).to_parquet(rd / "twin_foot_trajectories.parquet", index=False)

    # ---- base twist command channel ---------------------------------------
    twist_conf = (np.clip(gait.path_conf, 0.05, 1.0)
                  if gait.path_conf is not None and len(gait.path_conf) == T else conf)
    tw = pd.DataFrame({
        "t": fit.t, "frame": frames,
        "vx": fit.base_twist[:, 0], "vy": fit.base_twist[:, 1],
        "yaw_rate": fit.base_twist[:, 2],
        "conf": twist_conf, "valid": np.ones(T, dtype=bool),
        "source": np.full(T, SourceTag.estimated.value)})
    write_table(tw, rd / "cmd_twist.parquet")

    # ---- mapping + report --------------------------------------------------
    write_json_model(ws.mapping_json(robot), RetargetMapping(
        robot=robot, robot_class=RobotClass.quadruped,
        retarget_method="twin_fit_mujoco_v1", retarget_version="0.1.0",
        provenance="command-abstraction", key_body_map=dict(spec.key_body_map),
        notes="Digital-twin fit: dog gait -> Go2 joint trajectory minimizing "
              "foot-path tracking loss in MuJoCo. Gait+base-twist are the "
              "transferable product; a locomotion controller owns execution."))
    # ---- honest physics status (Tier-1 twin replay) -----------------------
    physics = _physics_status(cfg, ws, robot, log)
    report = {**fit.report, "physics_tier1": physics,
              "executable_product": "base twist command channel (cmd_twist.parquet) "
              "+ joint gait as a tracking-controller reference; the raw qpos is "
              "NOT dynamically balanced (see physics_tier1)"}
    (rd / "twin_fit_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    fit.report = report

    log(f"[twin] wrote {ws.rel(ws.qpos_parquet(robot))} "
        f"({T} frames) | loss {fit.report['loss_initial']:.4f} -> "
        f"{fit.report['loss_final']:.4f} ({fit.report['loss_reduction_pct']}% better)")
    return {"episode_id": episode_id, "robot": robot,
            "gait": gait.gait_label, "n_frames": T, **fit.report}
