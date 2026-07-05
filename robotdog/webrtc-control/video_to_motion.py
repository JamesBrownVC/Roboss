#!/usr/bin/env python3
"""Regenerate a robot-dog motion sequence from a generated video.

Pipeline:
    1. analyze — extract frames (ffmpeg), detect the robot in each frame
       (bounding box + posture + facing) to build per-frame TIME SERIES,
       then deterministically match those series onto bridge commands:
         posture transitions      -> sport commands (StandUp, Sit, ...)
         facing-angle changes     -> yaw `move` steps
         bbox displacement/scale  -> forward/backward `move` steps
       Only the per-frame detection uses a model (one image call per frame);
       the video->command matching is pure signal processing.
    2. replay  — connect to the bridge websocket (ws://.../ws) and execute the
       plan with the original timing on the real Go2.

Usage:
    # Put GEMINI_API_KEY in the environment (or a .env next to this script)
    python video_to_motion.py analyze output.mp4 -o plan.json
    python video_to_motion.py replay plan.json [--url ws://localhost:8080/ws]
    python video_to_motion.py run output.mp4          # analyze + replay
    python video_to_motion.py replay plan.json --dry-run

The plan is plain JSON so it can be reviewed/edited before touching the robot
(it also embeds the raw "track" time series for inspection):
    {"steps": [
        {"t": 0.0, "action": "sport", "cmd": "StandUp"},
        {"t": 2.0, "action": "move", "x": 0.4, "y": 0.0, "z": 0.0, "duration": 3.0}
    ]}
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- safety
ACRO_BLOCKLIST = {"FrontFlip", "BackFlip", "LeftFlip", "RightFlip", "FrontJump",
                  "FrontPounce", "Handstand", "HandStand", "BackStand", "StandOut"}

# Commands the model may use (mirrors the UI groups in static/app.js)
ALLOWED_SPORT = [
    # posture
    "StandUp", "StandDown", "Sit", "RiseSit", "BalanceStand", "RecoveryStand",
    "Damp", "StopMove", "Pose",
    # tricks
    "Hello", "Stretch", "Dance1", "Dance2", "WiggleHips", "FingerHeart",
    "Heart", "Scrape", "Content", "Wallow",
    # gaits (mode switches)
    "StaticWalk", "TrotRun", "EconomicGait", "ClassicWalk", "FreeWalk",
    "FreeBound", "FreeAvoid", "ContinuousGait", "CrossStep", "CrossWalk",
    "MoonWalk", "OnesidedStep", "Bound",
]
BOOL_PARAM = {"FreeWalk", "FreeBound", "FreeAvoid", "ClassicWalk", "CrossStep",
              "CrossWalk", "MoonWalk", "OnesidedStep", "Bound", "WiggleHips"}

MAX_VX, MAX_VY, MAX_VYAW = 0.6, 0.4, 1.0   # conservative velocity clamps
MAX_MOVE_DURATION = 8.0                     # seconds per continuous move step

# ------------------------------------------------------- tracking tunables
DETECT_MODEL = "gemini-3.5-flash"
TRACK_FPS = 4            # detection sampling rate (frames/second)
BODY_LEN = 0.70          # Go2 body length (m) — pixel->metre scale in side view
ASSUMED_RANGE = 2.0      # assumed camera distance (m) for toward/away speed
DEADBAND = 0.08          # m/s below which apparent motion is treated as noise
MIN_MOVE = 0.75          # s — ignore shorter translation bursts
NOM_SPEED = 0.4          # m/s fallback speed when the camera tracks the robot
NOM_YAW = 0.8            # rad/s used to replay detected turns
SPORT_SETTLE = 2.0       # s — min gap after a posture command (animations
                         #     swallow commands that arrive too early)

DETECT_PROMPT = """This image is a frame from a video showing a quadruped robot dog (Unitree Go2).
Report its state as a JSON object:
{"visible": true or false,
 "box_2d": [ymin, xmin, ymax, xmax],
 "posture": "standing" | "sitting" | "lying" | "other",
 "facing": "left" | "right" | "toward" | "away",
 "walking": true or false}
box_2d is normalized to 0-1000. "facing" is the direction the robot's head
points in the image ("toward" = at the camera). "walking" is true if the robot
is mid-stride (legs scissored or a foot lifted off the ground), false if it
stands still with all four feet planted. Return ONLY the JSON object."""


# ----------------------------------------------------------------- analyze

def _extract_frames(video_path: Path, fps: int, width: int = 640) -> list[tuple[float, bytes]]:
    """Extract (timestamp, jpeg_bytes) frames with ffmpeg."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path),
             "-vf", f"fps={fps},scale={width}:-2", "-q:v", "4",
             f"{tmp}/f%05d.jpg"],
            check=True,
        )
        files = sorted(Path(tmp).glob("f*.jpg"))
        return [(i / fps, f.read_bytes()) for i, f in enumerate(files)]


def _detect_frame(client, model: str, jpeg: bytes) -> dict:
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                  DETECT_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0),
    )
    d = json.loads(resp.text)
    if isinstance(d, list):
        d = d[0] if d else {}
    return d


def _track(client, model: str, frames: list[tuple[float, bytes]]) -> list[dict]:
    """One detection call per frame (parallel) -> per-frame state samples."""
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        t, jpeg = item
        try:
            d = _detect_frame(client, model, jpeg)
        except Exception as e:
            print(f"  detection failed @ t={t:.2f}s: {e}", file=sys.stderr)
            d = {}
        box = d.get("box_2d")
        if d.get("visible") and box and len(box) == 4:
            ymin, xmin, ymax, xmax = map(float, box)
            return {"t": t, "visible": True,
                    "cx": (xmin + xmax) / 2, "cy": (ymin + ymax) / 2,
                    "w": max(xmax - xmin, 1.0), "h": max(ymax - ymin, 1.0),
                    "posture": d.get("posture", "other"),
                    "facing": d.get("facing", "toward"),
                    "walking": bool(d.get("walking"))}
        return {"t": t, "visible": False}

    with ThreadPoolExecutor(max_workers=8) as ex:
        samples = list(ex.map(one, frames))
    ok = sum(s["visible"] for s in samples)
    print(f"Tracked robot in {ok}/{len(samples)} frames", file=sys.stderr)
    return samples


# --------------------------------------- time series -> command matching

# facing direction -> heading angle as seen from above (deg, CCW positive;
# camera sits "south" looking north, so image-right = east = 0)
FACING_ANGLE = {"right": 0.0, "away": 90.0, "left": 180.0, "toward": 270.0}

POSTURE_CMD = {("lying", "standing"): "StandUp",
               ("sitting", "standing"): "RiseSit",
               ("lying", "sitting"): "RiseSit",
               ("standing", "lying"): "StandDown",
               ("sitting", "lying"): "StandDown",
               ("standing", "sitting"): "Sit"}


def _stable_series(samples: list[dict], key: str, n: int = 2) -> list[tuple[float, str]]:
    """[(t_start, value)] whenever `key` holds the same value n frames in a row."""
    out, run_val, run_len, run_start = [], None, 0, 0.0
    for s in samples:
        v = s.get(key) if s["visible"] else None
        if v is None:
            continue
        if v == run_val:
            run_len += 1
        else:
            run_val, run_len, run_start = v, 1, s["t"]
        if run_len == n:
            out.append((run_start, v))
    return out


def _series_to_steps(samples: list[dict], dt: float) -> list[dict]:
    if sum(s["visible"] for s in samples) < 3:
        raise RuntimeError("robot not detected in enough frames to build a plan")
    steps: list[dict] = []

    # 1. posture transitions -> sport commands
    postures = _stable_series(samples, "posture")
    # mirror the video's initial pose (the real dog usually starts standing)
    if postures and postures[0][1] == "lying":
        steps.append({"t": 0.0, "action": "sport", "cmd": "StandDown"})
    elif postures and postures[0][1] == "sitting":
        steps.append({"t": 0.0, "action": "sport", "cmd": "Sit"})
    prev = None
    for t, posture in postures:
        if prev is not None and (prev, posture) in POSTURE_CMD:
            cmd = POSTURE_CMD[(prev, posture)]
            steps.append({"t": t, "action": "sport", "cmd": cmd})
            if cmd in ("StandUp", "RiseSit"):
                # StandUp leaves the Go2 in a stiff locked stand where Move
                # is ignored — switch to dynamic balance right after
                steps.append({"t": t + 1.0, "action": "sport",
                              "cmd": "BalanceStand"})
        prev = posture

    # 2. facing changes -> in-place yaw turns (only meaningful while standing)
    standing = [s if s["visible"] and s.get("posture") == "standing"
                else {"t": s["t"], "visible": False}
                for s in samples]
    prev_a = None
    for t, facing in _stable_series(standing, "facing"):
        a = FACING_ANGLE.get(facing)
        if a is None:
            continue
        if prev_a is not None and a != prev_a:
            delta = ((a - prev_a + 180.0) % 360.0) - 180.0  # deg, + = CCW
            dur = min(abs(math.radians(delta)) / NOM_YAW, MAX_MOVE_DURATION)
            steps.append({"t": t, "action": "move", "x": 0.0, "y": 0.0,
                          "z": math.copysign(NOM_YAW, delta),
                          "duration": round(dur, 2)})
        prev_a = a

    # 3. locomotion: gait detection says WHEN it walks (robust to a tracking
    #    camera); bbox displacement/scale only hints direction and speed.
    def apparent_v(a: dict, b: dict) -> float:
        f = a["facing"]
        if f in ("left", "right"):
            v = (b["cx"] - a["cx"]) / dt * (BODY_LEN / a["w"])
            return -v if f == "left" else v
        v = math.log(b["h"] / a["h"]) / dt * ASSUMED_RANGE
        return -v if f == "away" else v

    series: list[tuple[float, bool, float]] = []   # (t, walking, apparent v)
    for a, b in zip(samples, samples[1:]):
        ok = (a["visible"] and b["visible"]
              and a.get("posture") == "standing"
              and a.get("facing") == b.get("facing"))
        walking = ok and bool(a.get("walking"))
        series.append((a["t"], walking, apparent_v(a, b) if ok else 0.0))

    # close short gaps (<= 2 frames) in the walking signal — single-frame
    # gait classification flickers mid-trot when legs look planted
    walk = [w for _, w, _ in series]
    i = 0
    while i < len(walk):
        if not walk[i]:
            j = i
            while j < len(walk) and not walk[j]:
                j += 1
            if 0 < i and j < len(walk) and (j - i) <= 2:
                walk[i:j] = [True] * (j - i)
            i = j
        else:
            i += 1
    series = [(t, walk[k], v) for k, (t, _, v) in enumerate(series)]

    # 4. run-length encode the walking state into move steps
    runs, cur = [], None
    for t, walking, v in series:
        if cur is not None and cur["walking"] == walking:
            cur["vs"].append(v)
            cur["end"] = t + dt
        else:
            if cur is not None:
                runs.append(cur)
            cur = {"walking": walking, "start": t, "end": t + dt, "vs": [v]}
    if cur is not None:
        runs.append(cur)
    for r in runs:
        if not r["walking"] or (r["end"] - r["start"]) < MIN_MOVE:
            continue
        med = statistics.median(r["vs"])
        if med < -DEADBAND:                     # clearly moving backward
            speed = -min(max(abs(med), 0.15), MAX_VX)
        elif med > DEADBAND:                    # camera static: trust magnitude
            speed = min(max(med, NOM_SPEED), MAX_VX)
        else:                                   # tracking camera: nominal pace
            speed = NOM_SPEED
        steps.append({"t": r["start"], "action": "move",
                      "x": round(speed, 2), "y": 0.0, "z": 0.0,
                      "duration": round(r["end"] - r["start"], 2)})

    steps.sort(key=lambda s: s["t"])
    return steps


# -------------------------------- twin-fit cmd_twist.parquet -> plan
# The v2r twin-fit contract's on-robot executable product is the base-twist
# channel (its qpos is joint-level and needs a Go2 EDU). Columns:
#   t, frame, vx, vy, yaw_rate, conf, valid, source

TWIST_STEP = 0.25          # s — plan resolution (matches the 4 Hz move stream)
TWIST_MIN_CONF = 0.3       # drop low-confidence twin samples


def twist_to_plan(twist_path: Path) -> dict:
    import pandas as pd

    df = pd.read_parquet(twist_path)
    required = {"t", "vx", "vy", "yaw_rate"}
    if not required.issubset(df.columns):
        raise ValueError(f"not a cmd_twist contract (need {sorted(required)}, "
                         f"got {sorted(df.columns)})")
    if "valid" in df.columns:
        df = df[df["valid"]]
    if "conf" in df.columns:
        df = df[df["conf"].fillna(1.0) >= TWIST_MIN_CONF]
    df = df.sort_values("t")
    if df.empty:
        raise ValueError("no valid twist samples")

    # resample on a regular grid, then run-length encode near-constant
    # velocity into move steps the bridge can stream
    t0, t1 = float(df["t"].iloc[0]), float(df["t"].iloc[-1])
    import numpy as np
    grid = np.arange(0.0, t1 - t0 + TWIST_STEP, TWIST_STEP)
    src_t = df["t"].to_numpy() - t0
    vx = np.interp(grid, src_t, df["vx"].to_numpy())
    vy = np.interp(grid, src_t, df["vy"].to_numpy())
    wz = np.interp(grid, src_t, df["yaw_rate"].to_numpy())

    steps, cur = [], None
    for i, tg in enumerate(grid):
        v = (round(float(vx[i]), 2), round(float(vy[i]), 2),
             round(float(wz[i]), 2))
        moving = any(abs(c) > 0.03 for c in v)
        if cur is not None and cur["v"] == v:
            cur["end"] = tg + TWIST_STEP
        else:
            if cur is not None and any(abs(c) > 0.03 for c in cur["v"]):
                steps.append(cur)
            cur = {"v": v, "start": float(tg), "end": float(tg + TWIST_STEP)} \
                if moving else None
    if cur is not None:
        steps.append(cur)

    plan_steps = []
    for s in steps:
        # split runs longer than MAX_MOVE_DURATION into chained steps
        # (the replay streams them back-to-back seamlessly)
        start, end = s["start"], s["end"]
        while start < end - 1e-6:
            chunk = min(MAX_MOVE_DURATION, end - start)
            plan_steps.append({"t": round(start, 2), "action": "move",
                               "x": s["v"][0], "y": s["v"][1], "z": s["v"][2],
                               "duration": round(chunk, 2)})
            start += chunk
    print(f"twist: {len(df)} samples over {t1 - t0:.1f}s -> "
          f"{len(plan_steps)} move step(s)", file=sys.stderr)
    return {"steps": _sanitize(plan_steps), "source": str(twist_path)}


def analyze(video_path: Path, model: str, tricks: bool = False) -> dict:
    from dotenv import load_dotenv
    from google import genai

    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv()
    client = genai.Client()

    print(f"Extracting frames from {video_path.name} @ {TRACK_FPS} fps ...",
          file=sys.stderr)
    frames = _extract_frames(video_path, TRACK_FPS)
    samples = _track(client, model, frames)
    steps = _series_to_steps(samples, 1.0 / TRACK_FPS)
    plan = {"steps": _sanitize(steps), "track": samples}

    if tricks:
        raw = _detect_tricks(client, model, video_path)
        plan["tricks_raw"] = raw
        merged = _merge_tricks(plan["steps"], raw, samples)
        plan["steps"] = _sanitize(merged)
    return plan


# ------------------------------------------- hybrid tricks pass (VLM)
# The bbox tracker cannot see gestures (Hello, Dance, ...). A VLM pass with a
# CLOSED vocabulary proposes (cmd, t) pairs; each proposal is only accepted if
# it falls in an "idle window" of the deterministic track (no locomotion, no
# posture transition) — the tracker stays the source of truth for timing.

TRICKS = {
    "Hello": "sits back and waves one front leg in the air",
    "Stretch": "play-bow: front legs extended forward, chest lowered, rear up",
    "WiggleHips": "shakes its hindquarters rapidly side to side",
    "Dance1": "rhythmic dance, bouncing and swaying to a beat",
    "Dance2": "rhythmic dance, bouncing and swaying to a beat (variant)",
    "Scrape": "paws/scrapes the ground repeatedly with one front foot",
    "Wallow": "rolls onto its back and wiggles its legs in the air",
    "FingerHeart": "sits up on hind legs gesturing with front paws",
}

TRICKS_PROMPT = (
    "Watch this video of a Unitree Go2 quadruped robot dog. Identify ONLY the\n"
    "moments where it performs one of these named gestures:\n"
    + "\n".join(f"- {k}: {v}" for k, v in TRICKS.items())
    + "\n\nReturn JSON: {\"tricks\": [{\"t\": <start seconds>, \"cmd\": \"<name>\","
      " \"confidence\": <0..1>}]}\n"
      "Rules: only gestures you clearly see — plain walking, turning, standing,\n"
      "sitting or lying down are NOT tricks. Empty list if none.")


def _detect_tricks(client, model: str, video_path: Path) -> list[dict]:
    import io
    from google.genai import types

    print("Tricks pass: uploading video ...", file=sys.stderr)
    uploaded = client.files.upload(file=io.BytesIO(video_path.read_bytes()),
                                   config={"mime_type": "video/mp4"})
    deadline = time.monotonic() + 120
    while uploaded.state and uploaded.state.name == "PROCESSING":
        if time.monotonic() > deadline:
            raise RuntimeError("video processing timed out")
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    resp = client.models.generate_content(
        model=model,
        contents=[uploaded, TRICKS_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0),
    )
    data = json.loads(resp.text)
    out = data.get("tricks", data if isinstance(data, list) else [])
    print(f"Tricks pass: {len(out)} proposal(s)", file=sys.stderr)
    return out


def _merge_tricks(steps: list[dict], proposals: list[dict],
                  samples: list[dict], tol: float = 0.75,
                  min_conf: float = 0.5) -> list[dict]:
    """Accept a VLM trick only if it lands in an idle window of the track."""
    t_end = samples[-1]["t"] if samples else 0.0
    # busy intervals from the deterministic steps
    busy = []
    for s in steps:
        if s["action"] == "move":
            busy.append((s["t"], s["t"] + s["duration"]))
        else:
            busy.append((s["t"] - 0.5, s["t"] + 2.0))

    def is_idle(t: float) -> bool:
        return 0.0 <= t <= t_end + tol and \
            not any(a - tol < t < b + tol for a, b in busy)

    merged = list(steps)
    for p in proposals:
        cmd, t = p.get("cmd"), float(p.get("t", -1))
        conf = float(p.get("confidence", 0))
        if cmd not in TRICKS:
            print(f"  trick rejected (unknown): {cmd}", file=sys.stderr)
        elif conf < min_conf:
            print(f"  trick rejected (confidence {conf:.2f}): {cmd} @ {t:.1f}s",
                  file=sys.stderr)
        elif not is_idle(t):
            print(f"  trick rejected (conflicts with tracked motion): "
                  f"{cmd} @ {t:.1f}s", file=sys.stderr)
        else:
            print(f"  trick accepted: {cmd} @ {t:.1f}s (conf {conf:.2f})",
                  file=sys.stderr)
            merged.append({"t": round(t, 2), "action": "sport", "cmd": cmd})
    merged.sort(key=lambda s: s["t"])
    return merged


# -------------------------------------------- scenario timeline -> plan

SCENARIO_PROMPT = f"""You translate a robot action timeline into commands for a Unitree Go2
quadruped robot. For each timeline entry below, emit zero or more commands:

1. Named sport commands: {{{{"t": <s>, "action": "sport", "cmd": "<name>"}}}}
   Allowed: {", ".join(ALLOWED_SPORT)}
2. Locomotion: {{{{"t": <s>, "action": "move", "x": <fwd m/s>, "y": <left m/s>,
   "z": <ccw rad/s>, "duration": <s>}}}}
   Walking pace 0.3-0.5 m/s. "Rapid deceleration"/"stops" -> end the move
   step there (StopMove is implicit); use "StopMove" only for an emergency
   halt mid-motion.

Rules:
- Use the entry's t_start/t_end for timing. Keep actions in order.
- Standing/monitoring/waiting -> no command (or BalanceStand if it must
  transition from motion to active standstill).
- Manipulation (carrying/holding objects) has no Go2 equivalent — ignore it,
  keep only the locomotion component.
- Never use flips, jumps or handstands.

Return JSON: {{{{"steps": [ ... ]}}}}"""


def scenario_to_plan(scenario_path: Path, model: str, actor: str | None) -> dict:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    data = json.loads(scenario_path.read_text())
    scenarios = data.get("scenarios", [data])   # accept bundle or single
    sc = scenarios[0]
    timeline = sc.get("action_timeline") or []
    actors = sorted({e.get("actor") for e in timeline})
    if actor is None:  # default: the robot actor if unambiguous
        robots = [a for a in actors if a and "robot" in a]
        actor = robots[0] if len(robots) == 1 else None
    entries = [e for e in timeline if actor is None or e.get("actor") == actor]
    if not entries:
        raise RuntimeError(f"no timeline entries (actors: {actors})")
    print(f"scenario {sc.get('scenario_id')} — actor {actor}: "
          f"{len(entries)} timeline entries", file=sys.stderr)

    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv()
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[SCENARIO_PROMPT + "\n\nTimeline:\n" +
                  json.dumps(entries, indent=1)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0),
    )
    plan = json.loads(response.text)
    if isinstance(plan, list):
        plan = {"steps": plan}
    return {"steps": _sanitize(plan.get("steps", [])),
            "scenario_id": sc.get("scenario_id"), "actor": actor,
            "timeline": entries}


def _sanitize(steps: list[dict]) -> list[dict]:
    """Clamp velocities/durations and drop anything unsafe or unknown."""
    clamp = lambda v, lim: max(-lim, min(lim, float(v or 0)))
    out = []
    for s in steps:
        action = s.get("action")
        if action == "sport":
            cmd = s.get("cmd")
            if cmd in ACRO_BLOCKLIST or cmd not in ALLOWED_SPORT:
                print(f"  dropped unsafe/unknown command: {cmd}", file=sys.stderr)
                continue
            step = {"t": float(s.get("t", 0)), "action": "sport", "cmd": cmd}
            if cmd in BOOL_PARAM:
                step["parameter"] = {"data": True}
        elif action == "move":
            step = {
                "t": float(s.get("t", 0)), "action": "move",
                "x": clamp(s.get("x"), MAX_VX),
                "y": clamp(s.get("y"), MAX_VY),
                "z": clamp(s.get("z"), MAX_VYAW),
                "duration": min(max(float(s.get("duration", 1.0)), 0.2),
                                MAX_MOVE_DURATION),
            }
        else:
            print(f"  dropped unknown action: {action}", file=sys.stderr)
            continue
        out.append(step)
    out.sort(key=lambda s: s["t"])
    return out


# ------------------------------------------------------------------ replay

async def replay(plan: dict, url: str, dry_run: bool = False,
                 no_avoid: bool = False):
    steps = _sanitize(plan.get("steps", []))
    if not steps:
        print("Plan has no steps.", file=sys.stderr)
        return

    if dry_run:
        for s in steps:
            print(f"  t={s['t']:6.2f}s  {_describe(s)}")
        return

    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            rid = 0
            pending: dict[int, asyncio.Future] = {}
            yaw = {"value": None}   # latest IMU yaw (rad) from sportmodestate

            async def reader():
                async for raw in ws:
                    if raw.type != aiohttp.WSMsgType.TEXT:
                        break
                    m = json.loads(raw.data)
                    if m.get("type") == "response":
                        fut = pending.pop(m.get("id"), None)
                        if fut is not None and not fut.done():
                            fut.set_result(m)
                    elif m.get("type") == "state" and \
                            str(m.get("topic", "")).endswith("sportmodestate"):
                        rpy = (m.get("data") or {}).get("imu_state", {}).get("rpy")
                        if rpy and len(rpy) == 3:
                            yaw["value"] = float(rpy[2])
                for fut in pending.values():
                    if not fut.done():
                        fut.set_exception(RuntimeError("websocket closed"))

            reader_task = asyncio.create_task(reader())

            async def send_nowait(msg: dict):
                nonlocal rid
                rid += 1
                msg["id"] = rid
                await ws.send_str(json.dumps(msg))

            async def call(msg: dict):
                nonlocal rid
                rid += 1
                msg["id"] = rid
                fut = asyncio.get_running_loop().create_future()
                pending[rid] = fut
                await ws.send_str(json.dumps(msg))
                m = await fut
                if not m.get("ok"):
                    print(f"  !! {m.get('error')}", file=sys.stderr)
                return m

            async def turn_closed_loop(s: dict):
                """Turn until the IMU yaw delta is reached (open-loop timing
                under-rotates: ramp-up/down eats a big part of the angle)."""
                target = abs(s["z"]) * s["duration"]        # rad to rotate
                start = yaw["value"]
                deadline = time.monotonic() + s["duration"] * 2.5 + 2.0
                turned = 0.0
                prev = start
                while time.monotonic() < deadline:
                    await send_nowait({"action": "move", "x": s["x"],
                                       "y": s["y"], "z": s["z"]})
                    await asyncio.sleep(0.25)
                    cur = yaw["value"]
                    if cur is None or prev is None:
                        continue
                    d = (cur - prev + math.pi) % (2 * math.pi) - math.pi
                    turned += abs(d)
                    prev = cur
                    if turned >= target - 0.1:              # ~6° tolerance
                        break
                print(f"          turned {math.degrees(turned):.0f}°"
                      f" (target {math.degrees(target):.0f}°)")

            t0 = time.monotonic()
            restore_avoid = False
            last_sport = 0.0
            moving = False   # a velocity stream may still be in effect
            try:
                # MCF firmware filters Move commands while obstacle avoidance
                # is enabled -> the dog stands but never walks
                avoid = (await call({"action": "avoid_get"})).get("result") or {}
                if avoid.get("enable"):
                    if no_avoid:
                        print("obstacle avoidance ON -> disabling for replay")
                        await call({"action": "avoid_set", "enable": False})
                        restore_avoid = True
                        t0 = time.monotonic()
                    else:
                        print("WARNING: obstacle avoidance is ON — Move commands"
                              " will be filtered in mcf mode and the robot will"
                              " not walk. Re-run with --no-avoid to disable it"
                              " during the replay (make sure the area is clear).",
                              file=sys.stderr)
                for i, s in enumerate(steps):
                    # a move chains into the next one if that move starts
                    # exactly when this one ends -> no StopMove, no settle
                    nxt = steps[i + 1] if i + 1 < len(steps) else None
                    chained = (s["action"] == "move" and nxt is not None
                               and nxt["action"] == "move"
                               and abs(nxt["t"] - (s["t"] + s["duration"])) < 0.3)
                    # wait for the step's start time
                    delay = s["t"] - (time.monotonic() - t0)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # let the previous posture animation settle — the Go2
                    # drops sport/move commands that arrive mid-animation
                    settle = last_sport + SPORT_SETTLE - time.monotonic()
                    if settle > 0:
                        await asyncio.sleep(settle)
                    print(f"t={s['t']:6.2f}s  {_describe(s)}")
                    if s["action"] == "sport":
                        msg = {"action": "sport", "cmd": s["cmd"]}
                        if "parameter" in s:
                            msg["parameter"] = s["parameter"]
                        await call(msg)
                        last_sport = time.monotonic()
                        moving = False
                    elif s["x"] == 0 and s["y"] == 0 and s["z"] != 0 \
                            and yaw["value"] is not None and not chained:
                        moving = True
                        await turn_closed_loop(s)
                        await call({"action": "stop"})
                        last_sport = time.monotonic()
                        moving = False
                    else:  # move: stream velocity at 4 Hz like the web UI D-pad
                        moving = True
                        end = time.monotonic() + s["duration"]
                        while time.monotonic() < end:
                            await send_nowait({"action": "move", "x": s["x"],
                                               "y": s["y"], "z": s["z"]})
                            await asyncio.sleep(0.25)
                        if not chained:
                            await call({"action": "stop"})
                            # dog decelerates after StopMove — commands sent
                            # while it settles are swallowed, same as postures
                            last_sport = time.monotonic()
                            moving = False
            finally:
                # halt locomotion if interrupted mid-move — but do NOT send
                # StopMove after a final posture command: it cancels a fresh
                # Sit/StandDown and pops the dog straight back up
                if moving:
                    await call({"action": "stop"})
                    print("StopMove sent.")
                if restore_avoid:
                    print("re-enabling obstacle avoidance")
                    await call({"action": "avoid_set", "enable": True})
                print("done.")
                reader_task.cancel()


def _describe(s: dict) -> str:
    if s["action"] == "sport":
        return f"sport {s['cmd']}"
    return (f"move x={s['x']:+.2f} y={s['y']:+.2f} z={s['z']:+.2f} "
            f"for {s['duration']:.1f}s")


# -------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)

    pa = sub.add_parser("analyze", help="video -> motion plan JSON")
    pa.add_argument("video", type=Path)
    pa.add_argument("-o", "--output", type=Path, default=Path("plan.json"))
    pa.add_argument("--model", default=DETECT_MODEL)
    pa.add_argument("--tricks", action="store_true",
                    help="extra VLM pass for gestures (Hello, Dance, ...), "
                         "anchored to the deterministic track")

    ps = sub.add_parser("scenario",
                        help="scenario timeline JSON -> motion plan (no video)")
    ps.add_argument("scenario", type=Path,
                    help="scenarios.json (or single scenario) with action_timeline")
    ps.add_argument("-o", "--output", type=Path, default=Path("plan.json"))
    ps.add_argument("--model", default=DETECT_MODEL)
    ps.add_argument("--actor", default=None,
                    help="timeline actor to translate (default: the robot)")

    pt = sub.add_parser("twist",
                        help="twin-fit cmd_twist.parquet -> motion plan")
    pt.add_argument("twist", type=Path,
                    help="cmd_twist.parquet from the v2r twin-fit contract")
    pt.add_argument("-o", "--output", type=Path, default=Path("plan.json"))

    pr = sub.add_parser("replay", help="motion plan JSON -> robot")
    pr.add_argument("plan", type=Path)
    pr.add_argument("--url", default="ws://localhost:8080/ws")
    pr.add_argument("--dry-run", action="store_true",
                    help="print the timeline without sending commands")
    pr.add_argument("--no-avoid", action="store_true",
                    help="disable obstacle avoidance during replay, restore after")

    pu = sub.add_parser("run", help="analyze then replay")
    pu.add_argument("video", type=Path)
    pu.add_argument("-o", "--output", type=Path, default=Path("plan.json"))
    pu.add_argument("--model", default=DETECT_MODEL)
    pu.add_argument("--tricks", action="store_true")
    pu.add_argument("--url", default="ws://localhost:8080/ws")
    pu.add_argument("--dry-run", action="store_true")
    pu.add_argument("--no-avoid", action="store_true",
                    help="disable obstacle avoidance during replay, restore after")

    args = p.parse_args()

    if args.mode == "twist":
        plan = twist_to_plan(args.twist)
        args.output.write_text(json.dumps(plan, indent=2))
        print(f"Plan with {len(plan['steps'])} step(s) -> {args.output}")
        for s in plan["steps"]:
            print(f"  t={s['t']:6.2f}s  {_describe(s)}")
        return 0

    if args.mode == "scenario":
        plan = scenario_to_plan(args.scenario, args.model, args.actor)
        args.output.write_text(json.dumps(plan, indent=2))
        print(f"Plan with {len(plan['steps'])} step(s) -> {args.output}")
        for s in plan["steps"]:
            print(f"  t={s['t']:6.2f}s  {_describe(s)}")
        return 0

    if args.mode in ("analyze", "run"):
        plan = analyze(args.video, args.model,
                       tricks=getattr(args, "tricks", False))
        args.output.write_text(json.dumps(plan, indent=2))
        print(f"Plan with {len(plan['steps'])} step(s) -> {args.output}")
        if args.mode == "analyze":
            return 0
    else:
        plan = json.loads(args.plan.read_text())

    asyncio.run(replay(plan, args.url, dry_run=args.dry_run,
                       no_avoid=getattr(args, "no_avoid", False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
