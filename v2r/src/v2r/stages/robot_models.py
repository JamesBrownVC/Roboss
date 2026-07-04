"""Shared robot-model helpers: MuJoCo model loading, joint mapping, limits.

Used by retarget synthesis (limit-aware motion) and physics_validate (real
Tier-1 checks). MuJoCo is optional on the dev host; everything degrades to the
config/robots.yaml limits when the package or the MJCF is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import V2RConfig

_MODEL_CACHE: dict[str, object] = {}


def model_xml_path(cfg: V2RConfig, robot: str) -> Path:
    p = Path(cfg.robot(robot).model_path)
    return p if p.is_absolute() else cfg.root / p


def try_load_model(cfg: V2RConfig, robot: str):
    """Return a mujoco.MjModel or None (mujoco or MJCF unavailable)."""
    path = model_xml_path(cfg, robot)
    key = str(path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = None
    try:
        import mujoco

        if path.is_file():
            model = mujoco.MjModel.from_xml_path(str(path))
    except ImportError:
        model = None
    except Exception:
        # MJCF present but unloadable (e.g. mesh assets not downloaded —
        # see assets/robots/README.md): fall back to kinematic checks
        model = None
    _MODEL_CACHE[key] = model
    return model


def resolve_joint(model, dof_name: str) -> Optional[int]:
    """Map a config dof column to a model joint id.

    Exact name first, then with the leading vendor token stripped
    (``panda_joint1`` -> ``joint1``).
    """
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof_name)
    if jid >= 0:
        return jid
    if "_" in dof_name:
        stripped = dof_name.split("_", 1)[1]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, stripped)
        if jid >= 0:
            return jid
    return None


def home_qpos(cfg: V2RConfig, robot: str) -> dict[str, float]:
    """Per-dof nominal pose: the MJCF 'home' keyframe when available, else
    mid-range. Used for command-abstraction embodiments (quadrupeds) whose
    qpos channel is a standing stance, not a gait."""
    limits = joint_limits(cfg, robot)
    nominal = {name: 0.5 * (lo + hi) for name, (lo, hi) in limits.items()}
    model = try_load_model(cfg, robot)
    if model is None:
        return nominal
    import mujoco

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        return nominal
    key = model.key_qpos[key_id]
    for name in cfg.robot(robot).dof:
        jid = resolve_joint(model, name)
        if jid is not None:
            nominal[name] = float(key[int(model.jnt_qposadr[jid])])
    return nominal


def joint_limits(cfg: V2RConfig, robot: str) -> dict[str, tuple[float, float]]:
    """Per-dof (lo, hi): model jnt_range when available, else robots.yaml."""
    spec = cfg.robot(robot)
    limits = {name: spec.limits_for(name) for name in spec.dof}
    model = try_load_model(cfg, robot)
    if model is None:
        return limits
    for name in spec.dof:
        jid = resolve_joint(model, name)
        if jid is None:
            continue
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            limits[name] = (float(lo), float(hi))
    return limits
