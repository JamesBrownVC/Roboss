"""CLI: python -m verifier <video.mp4> [--scenario scenario.json] [--out report.json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checks import run_all_checks
from .config import DEFAULT_THRESHOLDS
from .extract import extract_evidence
from .report import build_report, save_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="verifier",
        description="Physics-plausibility verifier for generated action videos.")
    p.add_argument("video", help="path to the generated video (mp4/avi/mov)")
    p.add_argument("--scenario", help="metadata packet JSON from the generation side")
    p.add_argument("--out", default=None, help="report JSON path "
                   "(default: <video>_report.json)")
    p.add_argument("--annotated", default=None,
                   help="write annotated demo video to this path")
    p.add_argument("--device", default=None,
                   help="inference device, e.g. cpu / 0 (default: auto)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap on processed frames")
    p.add_argument("--gate2", action="store_true",
                   help="also run the semantic reviewer (needs Gemini API "
                        "credentials)")
    p.add_argument("--gate2-model", default=None,
                   help="reviewer model (default: gemini-3.5-flash)")
    p.add_argument("--gate2-frames", type=int, default=None,
                   help="uniformly sampled frames to send (default: 10)")
    args = p.parse_args(argv)

    th = DEFAULT_THRESHOLDS
    if args.max_frames:
        th.max_frames = args.max_frames
    if args.gate2_model:
        th.gate2_model = args.gate2_model
    if args.gate2_frames:
        th.gate2_frames = args.gate2_frames

    scenario = None
    if args.scenario:
        scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))

    print(f"[1/3] Extracting pose and object tracks from {args.video} ...")
    evidence = extract_evidence(args.video, th, device=args.device)
    print(f"      {len(evidence.person_tracks)} person track(s), "
          f"{len(evidence.object_tracks)} object track(s), "
          f"{evidence.n_frames} frames")

    print("[2/3] Running gate 1 (physics rule engine) ...")
    violations = run_all_checks(evidence, th)

    gate2_meta = None
    if args.gate2:
        from .gate2 import run_gate2
        print(f"      Running gate 2 (semantic reviewer, {th.gate2_model}) ...")
        semantic, gate2_meta = run_gate2(args.video, evidence, violations,
                                         scenario, th)
        if gate2_meta.get("status") != "ok":
            print(f"      gate 2 {gate2_meta['status']}: "
                  f"{gate2_meta.get('error', '')}")
        violations = sorted(violations + semantic,
                            key=lambda v: v.severity, reverse=True)

    report = build_report(evidence, violations, th, scenario, gate2_meta)
    out_path = args.out or str(Path(args.video).with_suffix("")) + "_report.json"
    save_report(report, out_path)

    print(f"[3/3] Report written to {out_path}")
    if args.annotated:
        from .viz import render_annotated_video
        render_annotated_video(args.video, evidence, violations, args.annotated)
        print(f"      Annotated video written to {args.annotated}")

    print()
    print(f"  decision : {report['decision'].upper()}")
    print(f"  score    : {report['plausibility_score']}")
    print(f"  reason   : {report['main_reason']}")
    for v in report["violations"]:
        print(f"   - [{v['severity']:.2f}] ({v['gate']}) {v['type']} @ frames "
              f"{v['frames'][0]}..{v['frames'][-1]}: {v['reason']}")

    return 0 if report["plausible"] else 2


if __name__ == "__main__":
    sys.exit(main())
