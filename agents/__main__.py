"""CLI: python -m agents "user intention" [--count N] [--out DIR] [--canvas]"""

from __future__ import annotations

import argparse
import sys

from .config import AgentConfig
from .llm import AgentError
from .pipeline import run_pipeline

_DEFAULTS = AgentConfig()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="agents",
        description="Scenario Contract Agent: user intention -> world "
                    "contract + validated scenario variations for video "
                    "generation (Gemini).")
    p.add_argument("intention", help="free-form user intention text")
    p.add_argument("--count", type=int, default=None,
                   help="number of scenario variations (default: from "
                        "intent, usually 5)")
    p.add_argument("--out", default="scenario_bundle",
                   help="output directory (default: scenario_bundle)")
    p.add_argument("--canvas", action="store_true",
                   help="also generate the canonical canvas anchor image")
    p.add_argument("--start-frames", action="store_true",
                   help="also derive a start frame per scenario from the "
                        "canvas (implies --canvas)")
    p.add_argument("--workers", type=int, default=None,
                   help=f"parallel start-frame image calls "
                        f"(default: {_DEFAULTS.start_frame_workers})")
    p.add_argument("--model", default=None,
                   help=f"text model (default: {_DEFAULTS.text_model})")
    p.add_argument("--image-model", default=None,
                   help=f"image model (default: {_DEFAULTS.image_model})")
    args = p.parse_args(argv)

    cfg = AgentConfig()  # fresh instance: never mutate the shared default
    if args.model:
        cfg.text_model = args.model
    if args.image_model:
        cfg.image_model = args.image_model
    if args.workers:
        cfg.start_frame_workers = args.workers

    try:
        result = run_pipeline(
            args.intention, args.out, cfg, count=args.count,
            make_canvas=args.canvas or args.start_frames,
            make_start_frames=args.start_frames,
        )
    except AgentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print()
    print(f"  world      : {result.contract['world_contract']['world_id']}")
    print(f"  models     : text {cfg.text_model}, image {cfg.image_model}, "
          f"video hint {cfg.video_model}")
    print(f"  scenarios  : {len(result.scenarios)} valid"
          + (f", {len(result.dropped)} dropped" if result.dropped else ""))
    for sc in result.scenarios:
        ev = sc["event"]
        print(f"   - {sc['scenario_id']}: {ev['type']} "
              f"({ev['risk_severity']}) @ {ev['trigger_time_seconds']:g}s, "
              f"camera {sc['camera']['angle']}")
    if result.canvas_path:
        print(f"  canvas     : {result.canvas_path}")
    if result.start_frame_paths:
        print(f"  frames     : {len(result.start_frame_paths)} start "
              f"frame(s) in {result.out_dir}\\frames")
    print(f"  bundle     : {result.out_dir}\\bundle.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
