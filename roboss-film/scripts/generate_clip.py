"""Generate one take for one clip via the Gemini Omni video path (not Veo).

Usage:
    <ml python> generate_clip.py <clip_id> <take_number> [--duration-hint N]

Reads clips/manifest prompt + standing negative prompt from ../manifest.json,
calls v2r.syngen.gemini.omni_generate_video (POST /v1beta/interactions,
model gemini-omni-flash-preview), polls the resulting file until ACTIVE, and
downloads the mp4 into clips/{clip_id}/take_{NN}.mp4. Also writes
clips/{clip_id}/prompt_take{NN}.txt with the exact prompt string sent.

Prints a single JSON line to stdout on success:
    {"ok": true, "clip_id": ..., "take": N, "path": "...", "uri": "...",
     "model": "...", "elapsed_s": ...}
or on failure:
    {"ok": false, "clip_id": ..., "take": N, "error": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # roboss-film/
ENV_PATHS = json.loads((ROOT / "env_paths.json").read_text(encoding="utf-8"))
sys.path.insert(0, ENV_PATHS["v2r_src"])

from v2r.syngen import gemini  # noqa: E402

import urllib.error
import urllib.request


def build_prompt(clip_prompt: str, negative_prompt: str) -> str:
    return f"{clip_prompt} Avoid: {negative_prompt}"


def download_file_safe(uri: str, dest: Path, api_key: str | None = None) -> Path:
    """Same as gemini.download_file but passes the certifi SSL context.

    v2r's gemini.download_file() calls urlopen() without context=_ssl_context(),
    unlike every other request in that module, so it hits the default (broken,
    pip_system_certs-corrupted) trust store on this machine. Reimplemented here
    rather than patching the shared v2r repo (co-edited by parallel sessions).
    """
    key = api_key or gemini.get_api_key()
    if not key:
        raise gemini.GeminiError("GEMINI_API_KEY not set")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(uri, headers={"X-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=300, context=gemini._ssl_context()) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.URLError as e:
        raise gemini.GeminiError(f"download failed: {e}") from e
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id")
    ap.add_argument("take", type=int)
    args = ap.parse_args()

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    clip = next((c for c in manifest["clips"] if c["id"] == args.clip_id), None)
    if clip is None:
        print(json.dumps({"ok": False, "clip_id": args.clip_id, "take": args.take,
                           "error": f"clip id not found in manifest"}))
        return 1
    if clip.get("type") != "generated":
        print(json.dumps({"ok": False, "clip_id": args.clip_id, "take": args.take,
                           "error": f"clip type={clip.get('type')} is not generated (slate/live_action)"}))
        return 1

    full_prompt = build_prompt(clip["prompt"], manifest["standing_negative_prompt"])

    clip_dir = ROOT / "clips" / args.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    take_tag = f"{args.take:02d}"
    prompt_path = clip_dir / f"prompt_take{take_tag}.txt"
    prompt_path.write_text(full_prompt, encoding="utf-8")

    dest = clip_dir / f"take_{take_tag}.mp4"

    t0 = time.time()
    try:
        uri = gemini.omni_generate_video(
            full_prompt,
            model=gemini.DEFAULT_OMNI_MODEL,
            aspect_ratio="16:9",
            timeout=600.0,
        )
        gemini.poll_file_active(uri, timeout_s=300.0)
        download_file_safe(uri, dest)
    except gemini.GeminiError as e:
        print(json.dumps({"ok": False, "clip_id": args.clip_id, "take": args.take,
                           "error": str(e), "elapsed_s": round(time.time() - t0, 1)}))
        return 1
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "clip_id": args.clip_id, "take": args.take,
                           "error": f"{type(e).__name__}: {e}",
                           "elapsed_s": round(time.time() - t0, 1)}))
        return 1

    print(json.dumps({
        "ok": True,
        "clip_id": args.clip_id,
        "take": args.take,
        "path": str(dest),
        "uri": uri,
        "model": gemini.DEFAULT_OMNI_MODEL,
        "elapsed_s": round(time.time() - t0, 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
