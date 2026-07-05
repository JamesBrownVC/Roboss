"""Record a clip take evaluation and optionally promote it to selects.

Usage:
    python record_take.py CLIP_ID TAKE verdict mean_soft main_defect repair_applied summary [--select]

This intentionally stores a compact structured eval JSON. Detailed visual notes
stay in the chat/checkpoint summaries; the manifest remains the source of truth
for state, selected take, attempts, and spend accounting.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id")
    ap.add_argument("take", type=int)
    ap.add_argument("verdict", choices=["pass", "fail"])
    ap.add_argument("mean_soft", type=float)
    ap.add_argument("main_defect")
    ap.add_argument("repair_applied")
    ap.add_argument("summary")
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()

    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = next(c for c in manifest["clips"] if c["id"] == args.clip_id)

    repair = None if args.repair_applied in {"-", "null", "None"} else args.repair_applied
    attempt = {
        "take": args.take,
        "verdict": args.verdict,
        "main_defect": args.main_defect,
        "repair_applied": repair,
        "mean_soft_score": args.mean_soft,
        "generation_model": "gemini-omni-flash-preview",
    }
    clip.setdefault("attempts", []).append(attempt)
    if args.select:
        clip["state"] = "passed"
        clip["selected_take"] = args.take
        src = ROOT / "clips" / args.clip_id / f"take_{args.take:02d}.mp4"
        dst = ROOT / "selects" / f"{args.clip_id}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    elif args.verdict == "fail":
        clip["state"] = "pending"

    manifest.setdefault("log", []).append({
        "clip": args.clip_id,
        "take": args.take,
        "verdict": args.verdict,
        "main_defect": args.main_defect,
        "repair_applied": repair,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    eval_path = ROOT / "clips" / args.clip_id / f"eval_take{args.take:02d}.json"
    eval_doc = {
        "clip_id": args.clip_id,
        "take": args.take,
        "evaluator": "GPT-5.5 direct vision review of sampled frames",
        "verdict": args.verdict,
        "mean_soft_score": args.mean_soft,
        "main_defect": args.main_defect,
        "repair_applied": repair,
        "summary": args.summary,
        "hard_criteria_result": "pass" if args.verdict == "pass" else "fail",
    }
    eval_path.write_text(json.dumps(eval_doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "clip_id": args.clip_id, "take": args.take, "selected": args.select}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
