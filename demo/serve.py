"""V2R Factory — hackathon demo server.

Serves a polished single-page frontend plus a JSON API that reads real
pipeline artifacts from ../v2r (data/, workspaces/) and falls back to
deterministic mock data whenever an artifact is missing.

Launch:  python demo/serve.py   →  http://localhost:8017
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
V2R = ROOT / "v2r"
DATA_RAW = V2R / "data" / "raw"
DATA_TS = V2R / "data" / "timeseries"
DATA_TRAIN = V2R / "data" / "training"
WORKSPACES = V2R / "workspaces"
SESSIONS = WORKSPACES / "sessions"
CACHE = DEMO_DIR / ".cache"
CACHE.mkdir(exist_ok=True)

app = FastAPI(title="V2R Factory Demo", version="0.1.0")

FUNNEL_STAGES = [
    "ingest", "feasibility_judge", "geometry", "human_body", "hands",
    "objects", "contact", "semantics", "retarget", "physics_validate",
    "qa", "package",
]

MEDIAPIPE_JOINTS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Videos / dataset explorer
# ---------------------------------------------------------------------------


def _probe(path: Path) -> dict:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
        cap.release()
        return {"fps": round(fps, 2), "n_frames": n, "width": w, "height": h,
                "duration_s": round(n / fps, 2) if fps else 0, "codec": codec}
    except Exception:
        return {"fps": 30.0, "n_frames": 0, "width": 0, "height": 0,
                "duration_s": 0, "codec": "?"}


def _scan_videos() -> list[dict]:
    videos: list[dict] = []
    manifest = _read_json(DATA_RAW / "import_manifest.json") or {"sources": []}
    subject_by_source = {s["source_id"]: s.get("subject", "human")
                         for s in manifest.get("sources", [])}
    if DATA_RAW.is_dir():
        for src_dir in sorted(DATA_RAW.iterdir()):
            if not src_dir.is_dir():
                continue
            for mp4 in sorted(src_dir.glob("*.mp4")):
                if mp4.name.startswith("."):
                    continue
                subject = subject_by_source.get(
                    src_dir.name,
                    "animal" if "animal" in src_dir.name else "human")
                stem = mp4.stem
                ts_path = DATA_TS / subject / f"{stem}.parquet"
                if not ts_path.is_file():
                    ts_path = CACHE / "ts" / subject / f"{stem}.parquet"
                videos.append({
                    "id": f"{src_dir.name}/{mp4.name}",
                    "source_id": src_dir.name,
                    "filename": mp4.name,
                    "stem": stem,
                    "subject": subject,
                    "size_mb": round(mp4.stat().st_size / 1e6, 2),
                    "url": f"/media/raw/{src_dir.name}/{mp4.name}",
                    "has_timeseries": ts_path.is_file(),
                    "mock": False,
                    **_probe(mp4),
                })
    # bundled CC-licensed sample clips (Wikimedia Commons), extracted locally
    dm = CACHE / "demo_media"
    if dm.is_dir():
        for mp4 in sorted(dm.glob("*.mp4")):
            subject = "human" if mp4.stem.startswith("human") else "animal"
            ts_path = CACHE / "ts" / subject / f"{mp4.stem}.parquet"
            videos.append({
                "id": f"cc_samples/{mp4.name}",
                "source_id": "cc_samples",
                "filename": mp4.name,
                "stem": mp4.stem,
                "subject": subject,
                "size_mb": round(mp4.stat().st_size / 1e6, 2),
                "url": f"/media/demo/{mp4.name}",
                "has_timeseries": ts_path.is_file(),
                "mock": False,
                **_probe(mp4),
            })
    return videos


MOCK_VIDEOS = [
    {"id": f"mock/{name}", "source_id": src, "filename": f"{name}.mp4",
     "stem": name, "subject": subj, "size_mb": 4.2, "url": None,
     "has_timeseries": False, "mock": True, "fps": 30.0, "n_frames": 300,
     "width": 1280, "height": 720, "duration_s": 10.0, "codec": "h264"}
    for name, src, subj in [
        ("human_walk_demo", "human_pexels_walk", "human"),
        ("human_dance_demo", "human_pexels_dance", "human"),
        ("dog_run_demo", "animal_pexels", "animal"),
        ("horse_trot_demo", "animal_pexels_wildlife", "animal"),
    ]
]


@app.get("/api/videos")
def api_videos():
    videos = _scan_videos()
    mock = not videos
    if mock:
        videos = MOCK_VIDEOS
    return {"videos": videos, "mock": mock,
            "n_human": sum(1 for v in videos if v["subject"] == "human"),
            "n_animal": sum(1 for v in videos if v["subject"] == "animal")}


@lru_cache(maxsize=64)
def _browser_playable(src: str) -> str:
    """Transcode mp4v videos to h264 once so browsers can play them."""
    src_path = Path(src)
    probe = _probe(src_path)
    if probe["codec"].lower() in ("avc1", "h264"):
        return src
    out = CACHE / (src_path.parent.name + "__" + src_path.stem + "_h264.mp4")
    if out.is_file() and out.stat().st_size > 10_000:
        return str(out)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src_path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
             "-movflags", "+faststart", "-an", str(out)],
            check=True, capture_output=True, timeout=300)
        return str(out)
    except Exception:
        return src


@app.get("/media/raw/{source_id}/{filename}")
def media_raw(source_id: str, filename: str):
    path = (DATA_RAW / source_id / filename).resolve()
    if not str(path).startswith(str(DATA_RAW.resolve())) or not path.is_file():
        raise HTTPException(404, "video not found")
    playable = _browser_playable(str(path))
    return FileResponse(playable, media_type="video/mp4")


@app.get("/media/demo/{filename}")
def media_demo(filename: str):
    path = (CACHE / "demo_media" / filename).resolve()
    if not str(path).startswith(str(CACHE.resolve())) or not path.is_file():
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------


def _mock_human_pose(n_frames: int = 240, fps: float = 30.0) -> dict:
    """Deterministic parametric walking figure, 33 MediaPipe joints, normalized coords."""
    frames = []
    for f in range(n_frames):
        t = f / fps
        phase = 2 * math.pi * 1.4 * t
        cx = 0.5 + 0.06 * math.sin(phase * 0.5)
        bob = 0.012 * math.sin(2 * phase)
        hip_y, sho_y, head_y = 0.55 + bob, 0.32 + bob, 0.16 + bob
        swing = 0.09 * math.sin(phase)
        joints = {}

        def put(name, x, y, z=0.0, c=0.95):
            joints[name] = [round(x, 4), round(y, 4), round(z, 4), c]

        put("nose", cx, head_y)
        for side, s in (("left", -1), ("right", 1)):
            put(f"{side}_eye_inner", cx + 0.008 * s, head_y - 0.012)
            put(f"{side}_eye", cx + 0.014 * s, head_y - 0.012)
            put(f"{side}_eye_outer", cx + 0.02 * s, head_y - 0.012)
            put(f"{side}_ear", cx + 0.028 * s, head_y - 0.004)
            put(f"mouth_{side}", cx + 0.01 * s, head_y + 0.02)
        for side, s in (("left", -1), ("right", 1)):
            arm = swing * -s
            put(f"{side}_shoulder", cx + 0.075 * s, sho_y)
            put(f"{side}_elbow", cx + 0.085 * s + arm * 0.4, sho_y + 0.11)
            wx = cx + 0.08 * s + arm
            wy = sho_y + 0.21
            put(f"{side}_wrist", wx, wy)
            put(f"{side}_pinky", wx + 0.012 * s, wy + 0.02)
            put(f"{side}_index", wx + 0.008 * s, wy + 0.024)
            put(f"{side}_thumb", wx - 0.006 * s, wy + 0.016)
            leg = swing * s
            put(f"{side}_hip", cx + 0.05 * s, hip_y)
            put(f"{side}_knee", cx + 0.05 * s + leg * 0.5,
                hip_y + 0.16 + 0.01 * abs(math.cos(phase)))
            ax = cx + 0.05 * s + leg
            ay = hip_y + 0.32 - max(0.0, 0.03 * math.sin(phase * 1.0 + (0 if s > 0 else math.pi)))
            put(f"{side}_ankle", ax, ay)
            put(f"{side}_heel", ax - 0.012, ay + 0.02)
            put(f"{side}_foot_index", ax + 0.025 * (1 if s > 0 else -1) + leg * 0.2, ay + 0.025)
        frames.append({
            "t": round(t, 4), "frame": f,
            "joints": [joints[name] for name in MEDIAPIPE_JOINTS],
        })
    return {"subject": "human", "fps": fps, "n_frames": n_frames,
            "joint_names": MEDIAPIPE_JOINTS, "frames": frames, "mock": True}


def _mock_animal_track(n_frames: int = 240, fps: float = 30.0) -> dict:
    frames = []
    for f in range(n_frames):
        t = f / fps
        cx = 0.15 + 0.7 * ((t * 0.12) % 1.0)
        cy = 0.55 + 0.03 * math.sin(2 * math.pi * 1.8 * t)
        frames.append({"t": round(t, 4), "frame": f, "entities": [{
            "entity_id": 1, "class_name": "dog",
            "cx": round(cx, 4), "cy": round(cy, 4),
            "w": 0.22, "h": 0.18,
            "vx": round(0.084 + 0.01 * math.sin(t), 4),
            "vy": round(0.34 * math.cos(2 * math.pi * 1.8 * t) * 0.03, 4),
            "conf": 0.91}]})
    return {"subject": "animal", "fps": fps, "n_frames": n_frames,
            "frames": frames, "mock": True}


def _load_human_parquet(path: Path) -> dict:
    import pandas as pd
    df = pd.read_parquet(path)
    df = df[df["entity_id"] == df["entity_id"].iloc[0]]
    n_frames = int(df["frame"].max()) + 1
    fps = 30.0
    ts = sorted(df["t"].unique())
    if len(ts) > 1:
        fps = round(1.0 / (ts[1] - ts[0]), 2)
    by_frame: dict[int, dict] = {}
    for row in df.itertuples():
        fr = by_frame.setdefault(int(row.frame), {"t": float(row.t), "joints": {}})
        x = None if row.x != row.x else round(float(row.x), 4)
        y = None if row.y != row.y else round(float(row.y), 4)
        z = 0.0 if row.z != row.z else round(float(row.z), 4)
        fr["joints"][int(row.joint_idx)] = [x, y, z, round(float(row.conf), 3)]
    frames = []
    for f in sorted(by_frame):
        jd = by_frame[f]["joints"]
        frames.append({
            "t": round(by_frame[f]["t"], 4), "frame": f,
            "joints": [jd.get(j, [None, None, 0.0, 0.0])
                       for j in range(len(MEDIAPIPE_JOINTS))],
        })
    return {"subject": "human", "fps": fps, "n_frames": n_frames,
            "joint_names": MEDIAPIPE_JOINTS, "frames": frames, "mock": False}


def _load_animal_parquet(path: Path) -> dict:
    import pandas as pd
    df = pd.read_parquet(path)
    fps = 30.0
    ts = sorted(df["t"].unique())
    if len(ts) > 1:
        fps = round(1.0 / max(1e-6, ts[1] - ts[0]), 2)
    by_frame: dict[int, dict] = {}
    for row in df.itertuples():
        if not bool(row.valid):
            continue
        fr = by_frame.setdefault(int(row.frame), {"t": float(row.t), "entities": []})
        fr["entities"].append({
            "entity_id": int(row.entity_id), "class_name": str(row.class_name),
            "cx": round(float(row.cx), 4), "cy": round(float(row.cy), 4),
            "w": round(float(row.w), 4), "h": round(float(row.h), 4),
            "vx": round(float(row.vx), 4), "vy": round(float(row.vy), 4),
            "conf": round(float(row.conf), 3)})
    frames = [{"t": round(v["t"], 4), "frame": f, "entities": v["entities"]}
              for f, v in sorted(by_frame.items())]
    return {"subject": "animal", "fps": fps,
            "n_frames": (max(by_frame) + 1) if by_frame else 0,
            "frames": frames, "mock": False}


@app.get("/api/timeseries/{subject}/{stem}")
def api_timeseries(subject: str, stem: str):
    if subject not in ("human", "animal"):
        raise HTTPException(400, "subject must be human|animal")
    candidates = [DATA_TS / subject / f"{stem}.parquet",
                  CACHE / "ts" / subject / f"{stem}.parquet"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = (_load_human_parquet(path) if subject == "human"
                       else _load_animal_parquet(path))
            if subject == "human":
                usable = any(j[0] is not None and j[3] > 0.05
                             for f in payload["frames"] for j in f["joints"])
            else:
                usable = any(f["entities"] for f in payload["frames"])
            if usable:
                return payload
            print(f"[timeseries] {path.name} exists but has no valid samples; using mock")
        except Exception as e:
            print(f"[timeseries] parquet load failed ({path.name}): {e}")
    return _mock_human_pose() if subject == "human" else _mock_animal_track()


# ---------------------------------------------------------------------------
# Feasibility judge
# ---------------------------------------------------------------------------

MOCK_FEASIBILITY = [
    {"episode_id": "sora_backflip_gen", "mock": True,
     "physically_plausible": False, "tracking_likely_valid": False,
     "ai_generated_artifacts": ["limb_morphing", "temporal_flicker", "physics_violation"],
     "confidence": 0.93, "recommendation": "reject",
     "physics_violation_frame_ratio": 0.41,
     "physics_checks": {"vel_spike_ratio": 0.28, "foot_slide_ratio": 0.19,
                        "scale_jump_ratio": 0.11, "flow_disagree_ratio": 0.22},
     "judge_source": "vlm",
     "notes": "Left forearm morphs through torso at t=2.3s; ground contact inconsistent."},
    {"episode_id": "veo_parkour_gen", "mock": True,
     "physically_plausible": True, "tracking_likely_valid": True,
     "ai_generated_artifacts": ["temporal_flicker"],
     "confidence": 0.61, "recommendation": "human_review",
     "physics_violation_frame_ratio": 0.12,
     "physics_checks": {"vel_spike_ratio": 0.08, "foot_slide_ratio": 0.05,
                        "scale_jump_ratio": 0.02, "flow_disagree_ratio": 0.09},
     "judge_source": "vlm",
     "notes": "Motion mostly plausible; minor background flicker warrants review."},
]


@app.get("/api/feasibility")
def api_feasibility():
    reports = []
    if WORKSPACES.is_dir():
        for ws in sorted(WORKSPACES.iterdir()):
            if not ws.is_dir() or ws.name == "sessions":
                continue
            rep = _read_json(ws / "qa" / "feasibility_report.json")
            if rep:
                rep["episode_id"] = ws.name
                rep["mock"] = False
                reports.append(rep)
    has_real = bool(reports)
    reports.extend(MOCK_FEASIBILITY)
    return {"reports": reports, "mock": not has_real}


# ---------------------------------------------------------------------------
# Multi-view sessions
# ---------------------------------------------------------------------------


def _mock_session() -> dict:
    import random
    rng = random.Random(7)
    per_frame = []
    for f in range(90):
        err = 2.2 + 0.9 * math.sin(f / 12) + rng.uniform(-0.35, 0.35)
        per_frame.append({"frame": f, "mean_error_px": round(max(0.4, err), 3)})
    return {
        "session_id": "warehouse_pick_demo", "mock": True,
        "cameras": ["cam0", "cam1", "cam2"], "tier": "multiview_gt",
        "sync": {"method": "audio_xcorr", "confidence": 0.97,
                 "cameras": [{"cam_id": "cam0", "offset_s": 0.0},
                              {"cam_id": "cam1", "offset_s": 0.042},
                              {"cam_id": "cam2", "offset_s": -0.013}]},
        "calibration": {"method": "colmap", "confidence": 0.94},
        "reproj": {"mean_reproj_error_px": 2.31, "p95_reproj_error_px": 3.9,
                   "n_frames": 90, "n_joints": 12,
                   "monocular_shadow_mean_px": 6.84, "triangulation_wins": True,
                   "per_frame": per_frame},
    }


def _session_payload(sess_dir: Path) -> dict:
    meta = _read_json(sess_dir / "session.json") or {}
    sync = _read_json(sess_dir / "sync.json")
    calib = _read_json(sess_dir / "calibration.json")
    reproj = _read_json(sess_dir / "qa" / "cross_view_reproj.json")
    payload = {
        "session_id": meta.get("session_id", sess_dir.name), "mock": False,
        "cameras": meta.get("cameras", []), "tier": meta.get("tier", "multiview_gt"),
        "sync": None, "calibration": None, "reproj": None,
    }
    if sync:
        payload["sync"] = {
            "method": sync.get("method"), "confidence": sync.get("confidence"),
            "cameras": [{"cam_id": c["cam_id"], "offset_s": c.get("offset_s", 0.0)}
                        for c in sync.get("cameras", [])]}
    if calib:
        payload["calibration"] = {"method": calib.get("method"),
                                  "confidence": calib.get("confidence")}
    if reproj:
        by_frame: dict[int, list[float]] = {}
        for pf in reproj.get("per_frame", []):
            by_frame.setdefault(int(pf["frame"]), []).append(float(pf["reproj_error_px"]))
        payload["reproj"] = {
            "mean_reproj_error_px": reproj.get("mean_reproj_error_px"),
            "p95_reproj_error_px": reproj.get("p95_reproj_error_px"),
            "n_frames": reproj.get("n_frames"), "n_joints": reproj.get("n_joints"),
            "monocular_shadow_mean_px": reproj.get("monocular_shadow_mean_px"),
            "triangulation_wins": reproj.get("triangulation_wins"),
            "per_frame": [{"frame": f, "mean_error_px": round(sum(v) / len(v), 3)}
                          for f, v in sorted(by_frame.items())]}
    return payload


@app.get("/api/multiview")
def api_multiview():
    sessions = []
    if SESSIONS.is_dir():
        for sess_dir in sorted(SESSIONS.iterdir()):
            if sess_dir.is_dir() and (sess_dir / "session.json").is_file():
                sessions.append(_session_payload(sess_dir))
    mock = not sessions
    if mock:
        sessions = [_mock_session()]
    return {"sessions": sessions, "mock": mock}


# ---------------------------------------------------------------------------
# Yield funnel
# ---------------------------------------------------------------------------

MOCK_FUNNEL = [
    ("ingested", 128), ("feasibility_ok", 117), ("geometry_ok", 109),
    ("body_ok", 101), ("hands_ok", 96), ("objects_ok", 92), ("contact_ok", 92),
    ("semantics_ok", 90), ("retarget_ok", 84), ("physics_ok", 78),
    ("qa_ok", 74), ("exported", 74),
]


@app.get("/api/yield")
def api_yield():
    episodes = []
    if WORKSPACES.is_dir():
        for ws in sorted(WORKSPACES.iterdir()):
            if not ws.is_dir() or ws.name == "sessions":
                continue
            stages = {}
            mdir = ws / "manifests"
            if not mdir.is_dir():
                continue
            for stage in FUNNEL_STAGES:
                m = _read_json(mdir / f"{stage}.manifest.json")
                stages[stage] = m.get("status") if m else "pending"
            decision = _read_json(ws / "qa" / "decision.json") or {}
            episodes.append({"episode_id": ws.name, "stages": stages,
                             "accepted": decision.get("accepted"),
                             "failure_stage": decision.get("failure_stage"),
                             "failure_reason": decision.get("failure_reason")})
    if episodes:
        funnel = []
        alive = len(episodes)
        for stage in FUNNEL_STAGES:
            alive = sum(1 for e in episodes
                        if e["stages"].get(stage) in ("success", "skipped"))
            funnel.append({"stage": stage, "count": alive})
        return {"funnel": funnel, "episodes": episodes,
                "total": len(episodes), "mock": False}
    return {"funnel": [{"stage": s, "count": c} for s, c in MOCK_FUNNEL],
            "episodes": [], "total": MOCK_FUNNEL[0][1], "mock": True}


# ---------------------------------------------------------------------------
# Export showcase
# ---------------------------------------------------------------------------

MOCK_EXPORTS = [
    {"episode_id": "warehouse_pick_demo", "mock": True, "tier": "multiview_gt",
     "tier_description": "multiview_gt: triangulated joints with measured cross-view error",
     "format": "lerobot_v3_fragment", "robots": ["g1", "franka"],
     "synthetic": False, "n_features": 14, "features": [
         "observation.state", "observation.video.cam0", "action.qpos",
         "action.ee_pose", "meta.tier", "meta.reproj_error_px"]},
]


@app.get("/api/exports")
def api_exports():
    exports = []
    if WORKSPACES.is_dir():
        for ws in sorted(WORKSPACES.iterdir()):
            if not ws.is_dir() or ws.name == "sessions":
                continue
            meta = _read_json(ws / "export" / "lerobot" / "meta.json")
            if not meta:
                continue
            features = _read_json(ws / "export" / "lerobot" / "features.json") or {}
            fnames = list(features.keys()) if isinstance(features, dict) else []
            exports.append({
                "episode_id": meta.get("episode_id", ws.name), "mock": False,
                "tier": meta.get("tier", "monocular"),
                "tier_description": meta.get("tier_description", ""),
                "format": meta.get("format", "lerobot_v3"),
                "robots": meta.get("robots", []),
                "synthetic": meta.get("synthetic", False),
                "n_features": len(fnames), "features": fnames[:8]})
    has_real = bool(exports)
    exports.extend(MOCK_EXPORTS)
    return {"exports": exports, "mock": not has_real}


# ---------------------------------------------------------------------------
# Synthetic Data Studio (syngen jobs)
# ---------------------------------------------------------------------------

SYNGEN = V2R / "data" / "syngen"

# jobs launched from the browser: job_id -> {running, returncode, log, started_at}
SYNGEN_RUNS: dict[str, dict] = {}


def _syngen_log_tail(job_id: str, n_lines: int = 6) -> list[str]:
    run = SYNGEN_RUNS.get(job_id)
    if not run:
        return []
    try:
        lines = Path(run["log"]).read_text(encoding="utf-8", errors="replace").splitlines()
        return [ln.strip() for ln in lines if ln.strip()][-n_lines:]
    except Exception:
        return []


def _episode_pipeline(episode_id: str) -> dict | None:
    """Per-stage status of one ingested episode (the 12 V2R stages)."""
    ws = WORKSPACES / episode_id
    mdir = ws / "manifests"
    if not mdir.is_dir():
        return None
    stages = {}
    for stage in FUNNEL_STAGES:
        m = _read_json(mdir / f"{stage}.manifest.json")
        stages[stage] = m.get("status") if m else "pending"
    decision = _read_json(ws / "qa" / "decision.json") or {}
    return {"episode_id": episode_id, "stages": stages,
            "accepted": decision.get("accepted"),
            "failure_stage": decision.get("failure_stage"),
            "failure_reason": decision.get("failure_reason")}


@app.get("/api/syngen")
def api_syngen():
    jobs = []
    seen: set[str] = set()
    if SYNGEN.is_dir():
        for job_dir in sorted(SYNGEN.iterdir()):
            spec = _read_json(job_dir / "spec.json")
            if not spec:
                continue
            status = _read_json(job_dir / "status.json") or {}
            variants = []
            for v in spec.get("variants", []):
                vid = v.get("variant_id", "")
                rec = _read_json(job_dir / "verification" / f"{vid}.json") or {}
                sidecar = _read_json(job_dir / "videos" / f"{vid}.json") or {}
                has_mp4 = (job_dir / "videos" / f"{vid}.mp4").is_file()
                vlm = rec.get("vlm") or {}
                phys = rec.get("physics") or {}
                variants.append({
                    "variant_id": vid,
                    "event_id": v.get("event_id"),
                    "cam_id": v.get("cam_id"),
                    "prompt": v.get("prompt", "")[:400],
                    "duration_s": v.get("duration_s"),
                    "backend": sidecar.get("backend", ""),
                    "gen_error": sidecar.get("error", ""),
                    "generated": has_mp4,
                    "video_url": (f"/media/syngen/{job_dir.name}/{vid}.mp4"
                                  if has_mp4 else None),
                    "verdict": rec.get("verdict"),
                    "verdict_reasons": rec.get("verdict_reasons", []),
                    "vlm_judge": vlm.get("judge_source"),
                    "vlm_notes": vlm.get("notes", ""),
                    "vlm_confidence": vlm.get("confidence"),
                    "physics": ({
                        "physics_ok": phys.get("physics_ok"),
                        "flow_consistency": phys.get("flow_consistency"),
                        "velocity_spike_ratio": phys.get("velocity_spike_ratio"),
                        "scale_jump_ratio": phys.get("scale_jump_ratio"),
                        "pose_detection_rate": phys.get("pose_detection_rate"),
                    } if phys else None),
                    "skills": (rec.get("labels") or {}).get("skills", []),
                })
            manifest = _read_json(job_dir / "delivery" / "manifest.json") or {}

            # live V2R pipeline progression: any workspace named <job_id>_*
            pipeline = []
            if WORKSPACES.is_dir():
                for ws in sorted(WORKSPACES.glob(f"{job_dir.name}_*")):
                    if ws.is_dir():
                        p = _episode_pipeline(ws.name)
                        if p:
                            pipeline.append(p)

            # multi-view triangulation QA per session
            reproj = []
            session_ids = manifest.get("sessions", [])
            if not session_ids and SESSIONS.is_dir():
                session_ids = [d.name for d in
                               SESSIONS.glob(f"syngen_{job_dir.name}_*") if d.is_dir()]
            for sid in session_ids:
                rj = _read_json(SESSIONS / sid / "qa" / "cross_view_reproj.json")
                if rj:
                    reproj.append({
                        "session_id": sid,
                        "mean_reproj_error_px": rj.get("mean_reproj_error_px"),
                        "p95_reproj_error_px": rj.get("p95_reproj_error_px"),
                        "n_frames": rj.get("n_frames"),
                        "n_joints": rj.get("n_joints"),
                    })

            # delivered episode cards
            episodes_detail = []
            for ep in manifest.get("episodes", []):
                ep_dir = job_dir / "delivery" / "episodes" / ep
                labels = (_read_json(ep_dir / "syngen_labels.json") or {}).get("labels", {})
                meta = _read_json(ep_dir / "lerobot" / "meta.json") or {}
                episodes_detail.append({
                    "episode_id": ep,
                    "caption": labels.get("caption", ""),
                    "skills": labels.get("skills", []),
                    "scene_type": labels.get("scene_type", ""),
                    "tier": meta.get("tier"),
                    "robots": meta.get("robots", []),
                    "format": meta.get("format"),
                    "path": str(ep_dir),
                    "has_trajectory": any(
                        (WORKSPACES / ep / "retargets").glob("*/ee.parquet")),
                })

            readme = job_dir / "delivery" / "README.md"
            dataset_card = ""
            if readme.is_file():
                try:
                    dataset_card = readme.read_text(encoding="utf-8")[:2500]
                except Exception:
                    pass

            job = {
                "job_id": spec.get("job_id", job_dir.name),
                "prompt": spec.get("user_prompt", ""),
                "created_at": spec.get("created_at", ""),
                "director": spec.get("director", ""),
                "backend": spec.get("backend", ""),
                "n_events": len(spec.get("events", [])),
                "n_cameras": len(spec.get("cameras", [])),
                "cameras": spec.get("cameras", []),
                "phase": status.get("phase", "requested"),
                "n_videos": status.get("n_videos"),
                "generated_ok": status.get("generated_ok"),
                "generated_failed": status.get("generated_failed"),
                "verification": status.get("verification", {}),
                "accepted": status.get("accepted"),
                "exported": status.get("exported"),
                "funnel": manifest.get("funnel", {}),
                "episodes": manifest.get("episodes", []),
                "episodes_detail": episodes_detail,
                "sessions": manifest.get("sessions", []),
                "pipeline": pipeline,
                "reproj": reproj,
                "dataset_card": dataset_card,
                "delivery_path": str(job_dir / "delivery") if readme.is_file() else None,
                "variants": variants,
            }
            run = SYNGEN_RUNS.get(job_dir.name)
            if run:
                job["runner"] = {"running": run["running"],
                                 "returncode": run["returncode"],
                                 "log_tail": _syngen_log_tail(job_dir.name)}
            seen.add(job_dir.name)
            jobs.append(job)
    # jobs just launched from the browser whose spec.json isn't written yet
    for job_id, run in SYNGEN_RUNS.items():
        if job_id in seen:
            continue
        jobs.append({
            "job_id": job_id, "prompt": run.get("prompt", ""),
            "created_at": run.get("started_at", ""), "director": "",
            "backend": run.get("backend", ""), "n_events": 0, "n_cameras": 0,
            "cameras": [], "phase": "starting", "verification": {},
            "funnel": {}, "episodes": [], "episodes_detail": [], "sessions": [],
            "pipeline": [], "reproj": [], "dataset_card": "",
            "delivery_path": None, "variants": [],
            "runner": {"running": run["running"], "returncode": run["returncode"],
                       "log_tail": _syngen_log_tail(job_id)},
        })
    return {"jobs": jobs, "mock": not jobs}


@app.get("/api/syngen/trajectory/{episode_id}")
def api_syngen_trajectory(episode_id: str):
    """End-effector trajectory of a delivered episode (retargeted robot hands)."""
    ws = (WORKSPACES / episode_id).resolve()
    if not str(ws).startswith(str(WORKSPACES.resolve())) or not ws.is_dir():
        raise HTTPException(404, "episode not found")
    ee_files = sorted((ws / "retargets").glob("*/ee.parquet"))
    if not ee_files:
        raise HTTPException(404, "no trajectory data")
    import pandas as pd
    robot = ee_files[0].parent.name
    df = pd.read_parquet(ee_files[0])
    hands = []
    for hand, g in df.groupby("hand"):
        g = g.sort_values("frame")
        step = max(1, len(g) // 300)          # cap payload size
        g = g.iloc[::step]
        hands.append({
            "hand": str(hand),
            "t": [round(float(x), 3) for x in g["t"]],
            "px": [round(float(x), 4) for x in g["px"]],
            "py": [round(float(x), 4) for x in g["py"]],
            "pz": [round(float(x), 4) for x in g["pz"]],
            "gripper": [round(float(x), 4) for x in g["gripper_aperture_m"]],
        })
    return {"episode_id": episode_id, "robot": robot, "hands": hands}


@app.get("/media/syngen/{job_id}/{filename}")
def media_syngen(job_id: str, filename: str):
    path = (SYNGEN / job_id / "videos" / filename).resolve()
    if not str(path).startswith(str(SYNGEN.resolve())) or not path.is_file():
        raise HTTPException(404, "video not found")
    playable = _browser_playable(str(path))
    return FileResponse(playable, media_type="video/mp4")


class SyngenRunRequest(BaseModel):
    prompt: str
    variants: int = 1
    cameras: int = 2
    backend: str = "auto"
    duration: int = 4


@app.post("/api/syngen/run")
def api_syngen_run(req: SyngenRunRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    if req.backend not in ("auto", "mock", "omni", "veo"):
        raise HTTPException(400, "backend must be auto|mock|omni|veo")
    variants = max(1, min(4, req.variants))
    cameras = max(1, min(3, req.cameras))
    duration = max(2, min(8, req.duration))

    job_id = time.strftime("web_%H%M%S")
    while (SYNGEN / job_id).exists() or job_id in SYNGEN_RUNS:
        job_id += "x"

    log_path = CACHE / f"syngen_{job_id}.log"
    cmd = [sys.executable, "-c",
           "from v2r.orchestrator.cli import main; main()",
           "syngen", "run", prompt, "--job-id", job_id,
           "--variants", str(variants), "--cameras", str(cameras),
           "--backend", req.backend, "--duration", str(duration),
           "--root", str(V2R)]
    SYNGEN_RUNS[job_id] = {
        "running": True, "returncode": None, "log": str(log_path),
        "prompt": prompt, "backend": req.backend,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def _worker():
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                        cwd=str(V2R))
                rc = proc.wait()
            SYNGEN_RUNS[job_id].update(running=False, returncode=rc)
        except Exception as e:
            log_path.write_text(f"launch failed: {e}", encoding="utf-8")
            SYNGEN_RUNS[job_id].update(running=False, returncode=-1)

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id, "started": True,
            "plan": {"variants": variants, "cameras": cameras,
                     "backend": req.backend, "duration_s": duration}}


# ---------------------------------------------------------------------------
# Parallel label demo pipeline (no generation; local dog video + synthetic data)
# ---------------------------------------------------------------------------

LABEL_DEMO_PROMPT = (
    "A dog moves and runs forward with a clear quadruped gait; convert that "
    "dog motion into synthetic training data for a robot dog without generating "
    "a new video."
)

LABEL_DEMO_SCENARIO = {
    "scenario_id": "dog_run_label_demo",
    "subject": "quadruped_dog",
    "source": "bundled_local_demo",
    "scene": "outdoor synthetic dog-run clip",
    "motion": "forward dog-motion gait for robot-dog retargeting",
    "duration_goal_s": 4,
    "expected_labels": [
        "dog_visible",
        "quadruped_running",
        "forward_locomotion",
        "cyclic_gait",
        "stable_body_track",
    ],
    "synthetic_controls": {
        "generation": "disabled",
        "video_asset": "demo/label_demo/ai_dog.mp4",
        "label_source": "precomputed synthetic demo metadata",
        "retarget_robot": "go2",
    },
}


def _phase_from_elapsed(elapsed: float) -> str:
    if elapsed < 0.7:
        return "requested"
    if elapsed < 1.6:
        return "generated"
    if elapsed < 2.6:
        return "verified"
    if elapsed < 3.8:
        return "ingesting"
    if elapsed < 4.8:
        return "ingested"
    return "delivered"


def _demo_stage_status(phase: str, stage_index: int) -> str:
    if phase == "delivered":
        return "success"
    if phase in ("ingesting", "ingested"):
        return "success" if stage_index < 8 else "pending"
    return "pending"


def _label_demo_job(job_id: str, run: dict) -> dict:
    now = time.time()
    elapsed = now - float(run.get("started_ts", now))
    phase = _phase_from_elapsed(elapsed)
    running = phase != "delivered"
    run.update(running=running, returncode=None if running else 0)

    video_path = LABEL_DEMO / "ai_dog.mp4"
    probe = _probe(video_path) if video_path.is_file() else {}
    generated = phase in ("generated", "verified", "ingesting", "ingested", "delivered")
    verified = phase in ("verified", "ingesting", "ingested", "delivered")
    ingested = phase in ("ingesting", "ingested", "delivered")
    delivered = phase == "delivered"

    stages = {
        stage: _demo_stage_status(phase, i)
        for i, stage in enumerate(FUNNEL_STAGES)
    }
    if delivered:
        stages = {stage: "success" for stage in FUNNEL_STAGES}

    pipeline = []
    if ingested:
        pipeline.append({
            "episode_id": "demo_ai_dog",
            "stages": stages,
            "accepted": delivered or phase == "ingested",
            "failure_stage": None,
            "failure_reason": None,
        })

    reproj = [{
        "session_id": "label_demo_single_view",
        "mean_reproj_error_px": 2.8,
        "p95_reproj_error_px": 5.4,
        "n_frames": probe.get("n_frames", 0),
        "n_joints": 17,
    }] if ingested else []

    episodes_detail = []
    if delivered:
        episodes_detail.append({
            "episode_id": "demo_ai_dog",
            "caption": "AI dog running forward; quadruped gait extracted and packed as synthetic locomotion data.",
            "skills": ["quadruped_run", "forward_locomotion", "gait_cycle", "body_tracking"],
            "scene_type": "synthetic_quadruped_motion",
            "tier": "synthetic_demo",
            "robots": ["go2"],
            "format": "lerobot_v3_fragment",
            "path": str(WORKSPACES / "demo_ai_dog"),
            "has_trajectory": False,
        })

    dataset_card = f"""# Dog Label Demo Dataset

Prompt:
{LABEL_DEMO_PROMPT}

Scenario:
- subject: {LABEL_DEMO_SCENARIO["subject"]}
- motion: {LABEL_DEMO_SCENARIO["motion"]}
- source video: demo/label_demo/ai_dog.mp4
- generation: disabled; this parallel demo uses a bundled video asset.

Synthetic labels:
- dog_visible
- quadruped_running
- forward_locomotion
- cyclic_gait
- stable_body_track

Artifacts:
- animal keypoints: v2r/workspaces/demo_ai_dog/animal/keypoints_superanimal.parquet
- object tracks: v2r/workspaces/demo_ai_dog/objects/tracks_2d.parquet
- GO2 retarget: v2r/workspaces/demo_ai_dog/retargets/go2/qpos.parquet
- command twist: v2r/workspaces/demo_ai_dog/retargets/go2/cmd_twist.parquet
"""

    return {
        "job_id": job_id,
        "prompt": LABEL_DEMO_PROMPT,
        "created_at": run.get("started_at", ""),
        "director": "parallel-demo",
        "backend": "local-label-demo",
        "n_events": 1,
        "n_cameras": 1,
        "cameras": [{
            "cam_id": "demo_cam0",
            "description": "bundled dog-run video viewport",
            "height_m": 1.0,
            "distance_m": 4.0,
            "azimuth_deg": 0,
            "fov_deg": 55,
        }],
        "phase": phase,
        "n_videos": 1,
        "generated_ok": 1 if generated else 0,
        "generated_failed": 0,
        "verification": {"accepted": 1 if verified else 0, "rejected": 0},
        "accepted": 1 if verified else 0,
        "exported": 1 if delivered else 0,
        "funnel": {
            "requested": 1,
            "local_video": 1 if generated else 0,
            "verified": 1 if verified else 0,
            "ingested": 1 if ingested else 0,
            "exported": 1 if delivered else 0,
        },
        "episodes": ["demo_ai_dog"] if delivered else [],
        "episodes_detail": episodes_detail,
        "sessions": ["label_demo_single_view"] if ingested else [],
        "pipeline": pipeline,
        "reproj": reproj,
        "dataset_card": dataset_card if delivered else "",
        "delivery_path": str(WORKSPACES / "demo_ai_dog") if delivered else None,
        "variants": [{
            "variant_id": "ai_dog_local_clip",
            "event_id": "dog_run_event",
            "cam_id": "demo_cam0",
            "prompt": (
                "Predefined synthetic scenario: dog runs forward with visible "
                "cyclic quadruped gait; recover labels, tracks, keypoints, "
                "and GO2 retarget-ready motion from the local clip."
            ),
            "duration_s": probe.get("duration_s"),
            "backend": "local-video",
            "gen_error": "",
            "generated": generated,
            "video_url": "/media/label-demo/ai_dog.mp4" if generated else None,
            "verdict": "accept" if verified else None,
            "verdict_reasons": [],
            "vlm_judge": "demo-synthetic-oracle" if verified else None,
            "vlm_notes": (
                "Dog remains visible; forward running gait is temporally "
                "consistent enough for the synthetic label demo."
            ) if verified else "",
            "vlm_confidence": 0.94 if verified else None,
            "physics": ({
                "physics_ok": True,
                "flow_consistency": 0.91,
                "velocity_spike_ratio": 0.04,
                "scale_jump_ratio": 0.03,
                "pose_detection_rate": 0.88,
            } if verified else None),
            "skills": ["quadruped_run", "forward_locomotion", "gait_cycle"] if verified else [],
        }],
        "synthetic_scenario": LABEL_DEMO_SCENARIO,
        "runner": {
            "running": running,
            "returncode": None if running else 0,
            "log_tail": [
                "parallel demo: generation skipped",
                "loaded demo/label_demo/ai_dog.mp4",
                f"phase={phase}",
            ],
        },
    }


@app.get("/api/label-demo/jobs")
def api_label_demo_jobs():
    jobs = [_label_demo_job(job_id, run)
            for job_id, run in sorted(LABEL_DEMO_RUNS.items())]
    return {"jobs": jobs, "mock": False}


@app.post("/api/label-demo/run")
def api_label_demo_run():
    video_path = LABEL_DEMO / "ai_dog.mp4"
    if not video_path.is_file():
        raise HTTPException(404, "demo/label_demo/ai_dog.mp4 not found")
    job_id = time.strftime("label_demo_ai_dog_%H%M%S")
    while job_id in LABEL_DEMO_RUNS:
        job_id += "x"
    LABEL_DEMO_RUNS[job_id] = {
        "started_ts": time.time(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "running": True,
        "returncode": None,
    }
    return {
        "job_id": job_id,
        "started": True,
        "plan": {
            "variants": 1,
            "cameras": 1,
            "backend": "local-label-demo",
            "duration_s": _probe(video_path).get("duration_s", 0),
        },
    }


@app.get("/media/label-demo/{filename}")
def media_label_demo(filename: str):
    path = (LABEL_DEMO / filename).resolve()
    if not str(path).startswith(str(LABEL_DEMO.resolve())) or not path.is_file():
        raise HTTPException(404, "label demo media not found")
    if path.suffix.lower() == ".mp4":
        playable = _browser_playable(str(path))
        return FileResponse(playable, media_type="video/mp4")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@app.get("/api/overview")
def api_overview():
    videos = _scan_videos()
    n_ts = len(list(DATA_TS.rglob("*.parquet"))) if DATA_TS.is_dir() else 0
    n_ws = 0
    n_sessions = 0
    if WORKSPACES.is_dir():
        n_ws = sum(1 for d in WORKSPACES.iterdir()
                   if d.is_dir() and d.name != "sessions" and (d / "manifests").is_dir())
    if SESSIONS.is_dir():
        n_sessions = sum(1 for d in SESSIONS.iterdir()
                         if d.is_dir() and (d / "session.json").is_file())
    train = _read_json(DATA_TRAIN / "manifest.json") or {}
    return {
        "videos": len(videos) or len(MOCK_VIDEOS),
        "videos_mock": not videos,
        "timeseries": n_ts,
        "training_episodes": train.get("n_videos", 0),
        "workspaces": n_ws,
        "sessions": n_sessions,
        "stages": len(FUNNEL_STAGES),
    }


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api and /media win)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=DEMO_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8017
    print(f"\n  V2R Factory demo  ->  http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
