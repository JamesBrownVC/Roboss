"""Extract evaluation frames from a downloaded take: 1fps sampling + first/last frame.

Usage:
    <ml python> extract_frames.py <clip_id> <take_number>

Writes into clips/{clip_id}/frames/:
    f_%02d.png   (1 fps sampling, ffmpeg -vf fps=1)
    f_first.png  (frame at t=0)
    f_last.png   (frame at t=duration - 0.05s)

Prints a JSON summary: {"ok": true, "clip_id":..., "take":..., "duration_s":...,
"frame_count": N, "frames_dir": "..."}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATHS = json.loads((ROOT / "env_paths.json").read_text(encoding="utf-8"))
FFMPEG = ENV_PATHS["ffmpeg_exe"]
FFPROBE = ENV_PATHS["ffprobe_exe"]


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def main() -> int:
    if len(sys.argv) != 3:
        print(json.dumps({"ok": False, "error": "usage: extract_frames.py <clip_id> <take_number>"}))
        return 1
    clip_id, take = sys.argv[1], int(sys.argv[2])
    take_tag = f"{take:02d}"
    clip_dir = ROOT / "clips" / clip_id
    src = clip_dir / f"take_{take_tag}.mp4"
    frames_dir = clip_dir / "frames" / f"take_{take_tag}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not src.is_file():
        print(json.dumps({"ok": False, "clip_id": clip_id, "take": take,
                           "error": f"source not found: {src}"}))
        return 1

    try:
        duration = probe_duration(src)

        # 1fps sampling
        subprocess.run(
            [FFMPEG, "-y", "-i", str(src), "-vf", "fps=1",
             str(frames_dir / "f_%02d.png")],
            capture_output=True, text=True, check=True,
        )

        # first frame
        subprocess.run(
            [FFMPEG, "-y", "-i", str(src), "-vframes", "1",
             str(frames_dir / "f_first.png")],
            capture_output=True, text=True, check=True,
        )

        # last frame
        last_ts = max(0.0, duration - 0.05)
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{last_ts:.3f}", "-i", str(src), "-vframes", "1",
             str(frames_dir / "f_last.png")],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(json.dumps({"ok": False, "clip_id": clip_id, "take": take,
                           "error": f"ffmpeg/ffprobe failed: {e.stderr[-500:] if e.stderr else str(e)}"}))
        return 1

    frame_count = len(list(frames_dir.glob("f_[0-9]*.png")))
    print(json.dumps({
        "ok": True, "clip_id": clip_id, "take": take,
        "duration_s": round(duration, 2),
        "frame_count": frame_count,
        "frames_dir": str(frames_dir),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
