"""LeRobot v3 fragment + EgoDex mirror writers (master prompt 6.J).

Writes a structurally valid LeRobot v3 dataset fragment without requiring the
`lerobot` package (not installable on the dev host): meta/info.json,
meta/tasks.parquet, meta/episodes + data chunk parquets, videos/. If `lerobot`
IS importable, a round-trip load is attempted and recorded; otherwise a
structural self-validation runs and the limitation is recorded in info.json.

EgoDex mirror: paired {n}.hdf5 + {n}.mp4 with camera intrinsics and SE(3) pose
tables for camera, head, and 25 joints per hand (EgoDex convention).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema.io import (
    EGODEX_HAND_JOINTS,
    SMPLX_MAIN_JOINTS,
    read_json,
    read_npz,
    read_table,
)
from ..schema.rotations import se3_from_quat_pos
from ..schema.workspace import EpisodeWorkspace

CHUNK = "chunk-000"
FILE = "file-000"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _f32list(arr: np.ndarray) -> list:
    """Row-wise float32 lists for parquet list<float> columns."""
    a = np.asarray(arr, dtype=np.float32)
    return [row for row in a.reshape(len(a), -1)]


def _hand_frames(hands: pd.DataFrame, side: str) -> dict[str, np.ndarray]:
    """Per-frame (T, 25, 8) pos+quat+conf arrays for one hand, EgoDex order."""
    h = hands[hands["hand"] == side]
    pivot: dict[str, np.ndarray] = {}
    frames = np.sort(h["frame"].unique())
    n, j = len(frames), len(EGODEX_HAND_JOINTS)
    pos = np.zeros((n, j, 3), dtype=np.float64)
    quat = np.tile(np.array([1.0, 0, 0, 0]), (n, j, 1))
    conf = np.zeros((n, j), dtype=np.float64)
    order = {name: k for k, name in enumerate(EGODEX_HAND_JOINTS)}
    fidx = {f: i for i, f in enumerate(frames)}
    for row in h.itertuples(index=False):
        i, k = fidx[row.frame], order.get(row.joint_name)
        if k is None:
            continue
        pos[i, k] = (row.px, row.py, row.pz)
        quat[i, k] = (row.qw, row.qx, row.qy, row.qz)
        conf[i, k] = row.conf
    pivot["pos"], pivot["quat"], pivot["conf"], pivot["frames"] = pos, quat, conf, frames
    return pivot


# ---------------------------------------------------------------------------
# LeRobot v3 fragment
# ---------------------------------------------------------------------------


def write_lerobot_fragment(
    ws: EpisodeWorkspace, cfg: V2RConfig, robots: list[str], synthetic: bool, tier: str
) -> list[str]:
    out_root = ws.lerobot_dir
    outputs: list[str] = []
    features: dict[str, dict] = {}
    omitted: list[str] = []

    poses = read_table(ws.poses_parquet)
    n_frames = len(poses)
    t = poses["t"].to_numpy(dtype=np.float64)
    data: dict[str, object] = {
        "timestamp": t.astype(np.float32),
        "frame_index": poses["frame"].to_numpy(dtype=np.int64),
        "episode_index": np.zeros(n_frames, dtype=np.int64),
        "index": np.arange(n_frames, dtype=np.int64),
        "task_index": np.zeros(n_frames, dtype=np.int64),
    }
    # LeRobotDataset casts the data parquet against the declared features:
    # every column must be declared, defaults included.
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        features[k] = {"dtype": "int64", "shape": [1]}
    features["subtask_index"] = {"dtype": "int64", "shape": [1]}

    # camera pose ------------------------------------------------------
    mat_cols = [c for c in poses.columns if c.startswith("T_wc_")]
    data["observation.camera_pose"] = _f32list(poses[mat_cols].to_numpy())
    data["observation.camera_pose_conf"] = poses["conf"].to_numpy(dtype=np.float32)
    features["observation.camera_pose"] = {"dtype": "float32", "shape": [16], "names": mat_cols}
    features["observation.camera_pose_conf"] = {"dtype": "float32", "shape": [1]}

    # video + depth references -----------------------------------------
    cam = read_json(ws.camera_json)
    features["observation.images.ego"] = {
        "dtype": "video",
        "shape": [cam["height"], cam["width"], 3],
        "info": {"video.fps": 30, "video.codec": "mp4v" if synthetic else "h264"},
    }
    # Depth is a per-frame PNG sidecar, not a data column: referenced from a
    # dedicated info.json key so LeRobotDataset's feature parser ignores it.
    dw = cam.get("depth_width") or cam["width"]
    dh = cam.get("depth_height") or cam["height"]
    depth_reference = {
        "shape": [dh, dw],
        "encoding": "uint16_png_millimeters",
        "depth_scale": cam.get("depth_scale", 1000.0),
        "path_template": "../../geometry/depth/{frame:06d}.png",
    }

    # human body ---------------------------------------------------------
    if ws.smplx_npz.is_file():
        z = read_npz(ws.smplx_npz)
        pose = z["pose"]  # (T, 55, 3) axis-angle
        body_pose = pose[:, 1:22, :].reshape(len(pose), -1)
        data["human.smplx.body_pose"] = _f32list(body_pose)
        features["human.smplx.body_pose"] = {"dtype": "float32", "shape": [63],
                                             "names": SMPLX_MAIN_JOINTS[1:] }
        data["human.smplx.conf"] = z["conf"].astype(np.float32)
        features["human.smplx.conf"] = {"dtype": "float32", "shape": [1]}
        jw = z["joints_world"]
        head = jw[:, SMPLX_MAIN_JOINTS.index("head"), :]
        data["human.head_pos"] = _f32list(head)
        features["human.head_pos"] = {"dtype": "float32", "shape": [3]}
    else:
        omitted.append("human.smplx.*")

    # hands ------------------------------------------------------------
    if ws.hands_parquet.is_file():
        hands = read_table(ws.hands_parquet)
        for side, key in (("left", "l"), ("right", "r")):
            hf = _hand_frames(hands, side)
            if len(hf["frames"]) != n_frames:
                omitted.append(f"human.hand_joints.{key} (frame mismatch)")
                continue
            joined = np.concatenate([hf["pos"], hf["quat"]], axis=2)  # (T,25,7)
            data[f"human.hand_joints.{key}"] = _f32list(joined)
            data[f"human.hand_joints.{key}_conf"] = hf["conf"].mean(axis=1).astype(np.float32)
            wrist = EGODEX_HAND_JOINTS.index("wrist")
            data[f"human.wrist_pose.{key}"] = _f32list(joined[:, wrist, :])
            # flat [175]: multi-dim shapes become Array2D in LeRobotDataset,
            # which cannot cast from a flat list<float> parquet column
            features[f"human.hand_joints.{key}"] = {
                "dtype": "float32", "shape": [25 * 7],
                "names": {"joints": EGODEX_HAND_JOINTS, "layout": "px py pz qw qx qy qz"},
            }
            features[f"human.hand_joints.{key}_conf"] = {"dtype": "float32", "shape": [1]}
            features[f"human.wrist_pose.{key}"] = {"dtype": "float32", "shape": [7]}
    else:
        omitted.append("human.hand_joints.*")

    # objects ------------------------------------------------------------
    object_ids: list[str] = []
    if ws.tracks_parquet.is_file():
        tracks = read_table(ws.tracks_parquet)
        for oid, grp in tracks.groupby("object_id", sort=True):
            grp = grp.sort_values("frame")
            if len(grp) != n_frames:
                omitted.append(f"object.{oid}.pose (frame mismatch)")
                continue
            object_ids.append(str(oid))
            arr = grp[["px", "py", "pz", "qw", "qx", "qy", "qz"]].to_numpy()
            data[f"object.{oid}.pose"] = _f32list(arr)
            data[f"object.{oid}.pose_conf"] = grp["conf"].to_numpy(dtype=np.float32)
            features[f"object.{oid}.pose"] = {"dtype": "float32", "shape": [7],
                                              "names": ["px","py","pz","qw","qx","qy","qz"]}
            features[f"object.{oid}.pose_conf"] = {"dtype": "float32", "shape": [1]}
    else:
        omitted.append("object.*.pose")

    # contact ------------------------------------------------------------
    if ws.contacts_parquet.is_file():
        contacts = read_table(ws.contacts_parquet)
        for side, key in (("left", "l"), ("right", "r")):
            c = contacts[contacts["hand"] == side]
            per_frame = c.groupby("frame")["contact"].any() if len(c) else pd.Series(dtype=bool)
            flags = np.zeros(n_frames, dtype=bool)
            idx = per_frame.index.to_numpy()
            idx = idx[idx < n_frames]
            flags[idx] = per_frame.loc[idx].to_numpy()
            data[f"contact.{key}"] = flags
            features[f"contact.{key}"] = {"dtype": "bool", "shape": [1]}
    else:
        omitted.append("contact.*")

    # retarget qpos + action ------------------------------------------------
    primary = robots[0] if robots else None
    for robot in robots:
        qp = ws.qpos_parquet(robot)
        if not qp.is_file():
            omitted.append(f"retarget.{robot}.qpos")
            continue
        qdf = read_table(qp)
        spec = cfg.robot(robot)
        dof_cols = [c for c in spec.dof if c in qdf.columns]
        if len(qdf) != n_frames:
            omitted.append(f"retarget.{robot}.qpos (frame mismatch)")
            continue
        arr = qdf[dof_cols].to_numpy()
        data[f"retarget.{robot}.qpos"] = _f32list(arr)
        data[f"retarget.{robot}.qpos_conf"] = qdf["conf"].to_numpy(dtype=np.float32)
        features[f"retarget.{robot}.qpos"] = {"dtype": "float32", "shape": [len(dof_cols)],
                                              "names": dof_cols}
        features[f"retarget.{robot}.qpos_conf"] = {"dtype": "float32", "shape": [1]}
        if robot == primary:
            data["action"] = data[f"retarget.{robot}.qpos"]
            features["action"] = {"dtype": "float32", "shape": [len(dof_cols)],
                                  "names": dof_cols, "alias_of": f"retarget.{robot}.qpos"}

    # subtask index -------------------------------------------------------
    subtask = np.full(n_frames, -1, dtype=np.int64)
    captions = {"short": "", "medium": "", "long": ""}
    if ws.segments_json.is_file():
        segs = read_json(ws.segments_json).get("segments", [])
        for si, seg in enumerate(segs):
            subtask[(t >= seg["start_s"]) & (t <= seg["end_s"])] = si
    if ws.captions_json.is_file():
        captions.update({k: v for k, v in read_json(ws.captions_json).items()
                         if k in captions})
    data["subtask_index"] = subtask

    # ---- write data chunk ------------------------------------------------
    df = pd.DataFrame(data)
    data_path = out_root / "data" / CHUNK / f"{FILE}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    outputs.append(ws.rel(data_path))

    # ---- tasks -------------------------------------------------------------
    tasks_path = out_root / "meta" / "tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"task_index": [0], "task": [captions["short"] or f"episode {ws.episode_id}"],
         "task_medium": [captions["medium"]], "task_long": [captions["long"]]}
    ).to_parquet(tasks_path, index=False)
    outputs.append(ws.rel(tasks_path))

    # ---- episodes metadata ---------------------------------------------------
    consent = read_json(ws.consent_path) if ws.consent_path.is_file() else {}
    scene = read_json(ws.scene_tags_json) if ws.scene_tags_json.is_file() else {}
    physics = {}
    for robot in robots:
        pj = ws.physics_report_json(robot)
        if pj.is_file():
            pr = read_json(pj)
            physics[robot] = {"physics_valid": pr.get("physics_valid"),
                              "tracking_error": pr.get("tracking_error")}
    betas = read_npz(ws.smplx_npz)["betas"].tolist() if ws.smplx_npz.is_file() else []
    duration_s = float(t[-1]) + (1.0 / 30.0) if n_frames else 0.0
    ep_row = {
        "episode_index": 0,
        "episode_id": ws.episode_id,
        "length": n_frames,
        # LeRobotDataset per-episode location columns (v3 layout)
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "videos/observation.images.ego/chunk_index": 0,
        "videos/observation.images.ego/file_index": 0,
        "videos/observation.images.ego/from_timestamp": 0.0,
        "videos/observation.images.ego/to_timestamp": duration_s,
        "dataset_from_index": 0,
        "dataset_to_index": n_frames,
        "tasks": json.dumps([captions["short"], captions["medium"], captions["long"]]),
        "subtasks": json.dumps(read_json(ws.segments_json).get("segments", [])
                               if ws.segments_json.is_file() else []),
        "smplx_betas": json.dumps(betas),
        "intrinsics": json.dumps({k: cam[k] for k in ("fx", "fy", "cx", "cy")}),
        "pose_quality": float(poses["conf"].mean()),
        "physics": json.dumps(physics),
        "provenance_mode": "synthetic" if synthetic else "real",
        "provenance_tier": tier,
        "scene_tags": json.dumps(scene),
        "consent_id": consent.get("consent_id", ""),
        "license": consent.get("license", ""),
    }
    ep_path = out_root / "meta" / "episodes" / CHUNK / f"{FILE}.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([ep_row]).to_parquet(ep_path, index=False)
    outputs.append(ws.rel(ep_path))

    # ---- video --------------------------------------------------------------
    if ws.video_path.is_file():
        vid_path = out_root / "videos" / "observation.images.ego" / CHUNK / f"{FILE}.mp4"
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ws.video_path, vid_path)
        outputs.append(ws.rel(vid_path))

    # ---- info.json + validation ---------------------------------------------
    # info.json must exist on disk BEFORE the round-trip runs: LeRobotDataset
    # reads it to locate data/video files; validated result is patched in after.
    info = {
        "codebase_version": "v3.0",
        "robot_type": primary or "none",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:1"},
        "data_path": f"data/{CHUNK}/{FILE}.parquet",
        "video_path": f"videos/{{video_key}}/{CHUNK}/{FILE}.mp4",
        "features": features,
        "features_omitted": omitted,
        "depth_reference": depth_reference,
        "synthetic": synthetic,
        "tier": tier,
        "validation": {"method": "pending", "passed": False},
    }
    info_path = out_root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    info["validation"] = _validate_fragment(out_root, n_frames)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    outputs.append(ws.rel(info_path))
    return outputs


def _validate_fragment(root: Path, n_frames: int) -> dict:
    """Round-trip with LeRobotDataset when available (in-process or via the
    repo's .venv), else structural checks."""
    df = pd.read_parquet(root / "data" / CHUNK / f"{FILE}.parquet")
    eps = pd.read_parquet(root / "meta" / "episodes" / CHUNK / f"{FILE}.parquet")
    structural_ok = len(df) == n_frames and int(eps.iloc[0]["length"]) == n_frames

    runner = None  # python executable that has lerobot importable
    try:
        import lerobot  # noqa: F401
        import sys

        runner = sys.executable
    except ImportError:
        for parent in Path(__file__).resolve().parents:
            venv_py = parent / ".venv" / "Scripts" / "python.exe"
            if venv_py.is_file():
                runner = str(venv_py)
                break
    if runner is None:
        return {"method": "structural (lerobot not installed on this host)",
                "passed": bool(structural_ok), "rows": int(len(df))}

    import subprocess

    code = (
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset;"
        f"ds = LeRobotDataset(repo_id='local/v2r', root=r'{root}');"
        "item = ds[0];"
        "print(ds.num_frames)"
    )
    proc = subprocess.run([runner, "-c", code], capture_output=True, text=True, timeout=600)
    loaded = proc.returncode == 0 and proc.stdout.strip().splitlines()[-1:] == [str(n_frames)]
    return {
        "method": "lerobot-roundtrip",
        "passed": bool(structural_ok and loaded),
        "rows": int(len(df)),
        "detail": "" if loaded else (proc.stderr or proc.stdout)[-500:],
    }


# ---------------------------------------------------------------------------
# EgoDex mirror
# ---------------------------------------------------------------------------


def write_egodex_mirror(ws: EpisodeWorkspace, cfg: V2RConfig) -> list[str]:
    import h5py

    outputs: list[str] = []
    egodex = ws.egodex_mirror_dir
    egodex.mkdir(parents=True, exist_ok=True)
    h5_path = egodex / "0.hdf5"

    cam = read_json(ws.camera_json) if ws.camera_json.is_file() else None
    poses = read_table(ws.poses_parquet) if ws.poses_parquet.is_file() else None
    captions = read_json(ws.captions_json) if ws.captions_json.is_file() else {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["episode_id"] = ws.episode_id
        f.attrs["llm_description"] = captions.get("short", "")
        f.attrs["convention"] = "EgoDex mirror: SE(3) 25 joints/hand, wxyz-derived matrices"
        if cam:
            k = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]],
                         dtype=np.float32)
            f.create_dataset("camera/intrinsic", data=k)
        if poses is not None:
            mat_cols = [c for c in poses.columns if c.startswith("T_wc_")]
            T = poses[mat_cols].to_numpy(dtype=np.float32).reshape(-1, 4, 4)
            f.create_dataset("transforms/camera", data=T)
            f.create_dataset("confidences/camera", data=poses["conf"].to_numpy(np.float32))
        if ws.smplx_npz.is_file():
            z = read_npz(ws.smplx_npz)
            head = z["joints_world"][:, SMPLX_MAIN_JOINTS.index("head"), :]
            Th = np.tile(np.eye(4, dtype=np.float32), (len(head), 1, 1))
            Th[:, :3, 3] = head
            f.create_dataset("transforms/head", data=Th)
            f.create_dataset("confidences/head", data=z["conf"].astype(np.float32))
        if ws.hands_parquet.is_file():
            hands = read_table(ws.hands_parquet)
            for side, label in (("left", "leftHand"), ("right", "rightHand")):
                hf = _hand_frames(hands, side)
                if not len(hf["frames"]):
                    continue
                se3 = se3_from_quat_pos(hf["quat"], hf["pos"]).astype(np.float32)  # (T,25,4,4)
                wrist = EGODEX_HAND_JOINTS.index("wrist")
                f.create_dataset(f"transforms/{label}", data=se3[:, wrist])
                f.create_dataset(f"confidences/{label}", data=hf["conf"][:, wrist].astype(np.float32))
                for j, joint in enumerate(EGODEX_HAND_JOINTS):
                    f.create_dataset(f"transforms/{side}_{joint}", data=se3[:, j])
                    f.create_dataset(f"confidences/{side}_{joint}",
                                     data=hf["conf"][:, j].astype(np.float32))
    outputs.append(ws.rel(h5_path))

    if ws.video_path.is_file():
        dst = egodex / "0.mp4"
        shutil.copy2(ws.video_path, dst)
        outputs.append(ws.rel(dst))
    return outputs


# ---------------------------------------------------------------------------
# entry point used by the package stage
# ---------------------------------------------------------------------------


def write_exports(
    ws: EpisodeWorkspace,
    cfg: V2RConfig,
    robots: list[str],
    synthetic: bool = True,
    tier: str = "monocular",
) -> list[str]:
    outputs = write_lerobot_fragment(ws, cfg, robots, synthetic, tier)
    outputs += write_egodex_mirror(ws, cfg)
    return outputs
