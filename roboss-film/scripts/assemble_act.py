"""Assemble one act's rough cut + contact sheet from passing selects.

Usage:
    <ml python> assemble_act.py <act>        # act in {I, II, III}

For each clip belonging to the act (in manifest order):
  - generated + passed  -> trim select to use_seconds (from optional trim_start),
                           re-encode to 1280x720@24, silent (picture cut).
  - slate / live_action -> a black card with a drawtext LABEL (editorial
                           placeholder, NOT an AI generation; text is allowed on
                           slates, never inside generated plates).
Concatenates in order into roughcut/act{N}.mp4 and tiles the first frame of each
generated select into roughcut/act{N}_contactsheet.png.

Prints a JSON summary line.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = json.loads((ROOT / "env_paths.json").read_text(encoding="utf-8"))
FFMPEG = ENV["ffmpeg_exe"]

ACT_MAP = {
    "I": {"acts": {"I", "I/II"}, "n": 1},
    "II": {"acts": {"II"}, "n": 2},
    "III": {"acts": {"III"}, "n": 3},
}
W, H, FPS = 1280, 720, 24
SLATE_SECONDS = 2.0


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def esc_drawtext(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def make_generated_segment(clip: dict, out: Path) -> None:
    src = ROOT / "selects" / f"{clip['id']}.mp4"
    if not src.exists():
        raise FileNotFoundError(f"missing select: {src}")
    start = float(clip.get("trim_start", 0.0))
    dur = float(clip["use_seconds"])
    run([
        FFMPEG, "-y", "-ss", f"{start}", "-i", str(src), "-t", f"{dur}",
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},setsar=1",
        "-af", "aresample=async=1:first_pts=0,aformat=sample_rates=48000:channel_layouts=stereo",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", str(out),
    ])


def make_slate_segment(clip: dict, out: Path) -> None:
    label = f"{clip['id']}  [{clip.get('type', 'slate').upper()}]"
    sub = clip.get("notes", "") or ""
    sub = sub.split(".")[0][:70]
    dur = float(clip.get("use_seconds") or SLATE_SECONDS)
    fontfile = "C\\:/Windows/Fonts/arial.ttf"
    dt_main = (f"drawtext=fontfile='{fontfile}':text='{esc_drawtext(label)}':"
               f"fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2-30")
    dt_sub = (f"drawtext=fontfile='{fontfile}':text='{esc_drawtext(sub)}':"
              f"fontcolor=0x9aa0a6:fontsize=22:x=(w-text_w)/2:y=(h-text_h)/2+40")
    run([
        FFMPEG, "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={dur}",
        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={dur}",
        "-vf", f"{dt_main},{dt_sub}", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", str(out),
    ])


def build_contact_sheet(gen_clips: list[dict], out: Path) -> None:
    tiles = []
    for clip in gen_clips:
        take = clip.get("selected_take")
        ff = ROOT / "clips" / clip["id"] / "frames" / f"take_{take:02d}" / "f_first.png"
        if ff.exists():
            tiles.append((clip["id"], ff))
    if not tiles:
        return
    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        inputs = []
        filters = []
        for i, (cid, ff) in enumerate(tiles):
            inputs += ["-i", str(ff)]
            filters.append(
                f"[{i}:v]scale=480:270:force_original_aspect_ratio=decrease,"
                f"pad=480:270:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{cid}':"
                f"fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5:x=8:y=8[t{i}]"
            )
        # pad with black tiles to fill the grid
        total = cols * rows
        for j in range(len(tiles), total):
            filters.append(f"color=c=black:s=480x270:d=1[t{j}]")
        lay = "".join(f"[t{k}]" for k in range(total))
        filters.append(f"{lay}xstack=inputs={total}:layout=" +
                       "|".join(f"{(k % cols)*480}_{(k // cols)*270}" for k in range(total)) +
                       ":fill=black[out]")
        fc = ";".join(filters)
        run([FFMPEG, "-y", *inputs, "-filter_complex", fc, "-map", "[out]",
             "-frames:v", "1", str(out)])


def main() -> int:
    act = sys.argv[1] if len(sys.argv) > 1 else "I"
    if act not in ACT_MAP:
        print(json.dumps({"ok": False, "error": f"unknown act {act}"}))
        return 1
    cfg = ACT_MAP[act]
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    clips = [c for c in manifest["clips"] if c.get("act") in cfg["acts"]]
    clips.sort(key=lambda c: c["order"])

    roughcut = ROOT / "roughcut"
    roughcut.mkdir(exist_ok=True)

    segments = []
    gen_clips = []
    skipped = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, clip in enumerate(clips):
            seg = tdp / f"seg_{i:02d}.mp4"
            if clip.get("type") == "generated":
                if clip.get("state") != "passed":
                    skipped.append(f"{clip['id']} (state={clip.get('state')})")
                    continue
                make_generated_segment(clip, seg)
                gen_clips.append(clip)
            else:
                make_slate_segment(clip, seg)
            segments.append(seg)

        if not segments:
            print(json.dumps({"ok": False, "error": "no segments", "skipped": skipped}))
            return 1

        listfile = tdp / "concat.txt"
        listfile.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments),
                            encoding="utf-8")
        out_mp4 = roughcut / f"act{cfg['n']}.mp4"
        run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
             "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
             str(out_mp4)])

        contact = roughcut / f"act{cfg['n']}_contactsheet.png"
        build_contact_sheet(gen_clips, contact)

    print(json.dumps({
        "ok": True, "act": act, "roughcut": str(out_mp4),
        "contact_sheet": str(contact),
        "segments": len(segments), "generated": [c["id"] for c in gen_clips],
        "skipped": skipped,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
