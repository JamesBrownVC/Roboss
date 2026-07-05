#!/usr/bin/env python3
"""Replay a twin-fit qpos.parquet on the official Go2 model in the MuJoCo viewer.

Usage (macOS needs mjpython for the passive viewer):
    ./.venv/bin/mjpython view_qpos.py /tmp/qpos_dog.parquet \
        --model /tmp/menagerie/unitree_go2/scene.xml [--speed 1.0]

Kinematic replay (mj_forward, no dynamics): the twin-fit qpos is an open-loop
kinematic fit — stepping it dynamically would make the robot fall (documented
in the fit report). The base is pinned at a nominal height for display.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

JOINTS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
          for part in ("hip", "thigh", "calf")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qpos", type=Path)
    p.add_argument("--model", type=Path,
                   default=Path("/tmp/menagerie/unitree_go2/scene.xml"))
    p.add_argument("--speed", type=float, default=1.0, help="playback speed")
    p.add_argument("--amplify", type=float, default=1.0,
                   help="scale joint motion around its mean (the twin-fit "
                        "gait can be tiny, e.g. std ~1.4° — try 8)")
    p.add_argument("--twist", type=Path, default=None,
                   help="cmd_twist.parquet: integrate vx/yaw_rate to move "
                        "the base along the dog's trajectory")
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(str(args.model))
    d = mujoco.MjData(m)
    q = pd.read_parquet(args.qpos)
    t = q["t"].to_numpy()
    print(f"{len(q)} frames over {t[-1] - t[0]:.1f}s — looping "
          f"(speed x{args.speed}, amplify x{args.amplify}, "
          f"Esc/close window to quit)")

    # qpos addresses for each named joint (robust to model ordering)
    jadr = {}
    for name in JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"joint {name} not found in {args.model}")
        jadr[name] = m.jnt_qposadr[jid]
    # floating base (first free joint)
    base_adr = m.jnt_qposadr[0]

    root = q[["root_px", "root_py", "root_pz",
              "root_qw", "root_qx", "root_qy", "root_qz"]].to_numpy().copy()
    joints = {name: q[name].to_numpy().copy() for name in JOINTS}

    # amplify gait around each joint's mean so tiny fits become visible
    if args.amplify != 1.0:
        for name in JOINTS:
            v = joints[name]
            joints[name] = v.mean() + (v - v.mean()) * args.amplify

    # integrate the base twist so the robot walks its real trajectory
    if args.twist is not None:
        tw = pd.read_parquet(args.twist).sort_values("t")
        vx = np.interp(t, tw["t"], tw["vx"])
        wz = np.interp(t, tw["t"], tw["yaw_rate"])
        heading, x, y = 0.0, 0.0, 0.0
        for i in range(len(t)):
            dt = (t[i] - t[i - 1]) if i else 0.0
            heading += wz[i] * dt
            x += vx[i] * np.cos(heading) * dt
            y += vx[i] * np.sin(heading) * dt
            root[i, 0], root[i, 1] = x, y
            root[i, 3:7] = [np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)]
        print(f"twist integrated: net displacement "
              f"({x:+.2f}, {y:+.2f}) m, heading {np.degrees(heading):+.0f}\u00b0")

    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            start = time.monotonic()
            for i in range(len(q)):
                if not v.is_running():
                    break
                # wait for this frame's timestamp
                target = (t[i] - t[0]) / args.speed
                lag = target - (time.monotonic() - start)
                if lag > 0:
                    time.sleep(lag)
                d.qpos[base_adr:base_adr + 7] = root[i]
                for name, adr in jadr.items():
                    d.qpos[adr] = joints[name][i]
                mujoco.mj_forward(m, d)   # kinematics only
                v.sync()
            time.sleep(0.5)  # pause before looping


if __name__ == "__main__":
    main()
