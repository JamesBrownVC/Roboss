"""CLI wrapper for one video: generate -> verify -> optional label."""

from __future__ import annotations

import argparse

from roboss.pipeline import exit_with_report_decision, run_video_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end generation, verification and labeling.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt for video generation.")
    parser.add_argument("--scenario", type=str, default="scenario.example.json",
                        help="Scenario JSON path, or 'none'.")
    parser.add_argument("--outdir", type=str, default="runs/latest",
                        help="Output directory for video and reports.")
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device, e.g. cpu / 0 (default: auto).")
    parser.add_argument("--no-gate2", action="store_true",
                        help="Skip the semantic reviewer.")
    parser.add_argument("--no-label", action="store_true",
                        help="Skip auto-labeling even if the video is accepted.")
    args = parser.parse_args()

    scenario_path = None if args.scenario == "none" else args.scenario
    result = run_video_pipeline(
        prompt=args.prompt,
        outdir=args.outdir,
        scenario_path=scenario_path,
        gate2=not args.no_gate2,
        label=not args.no_label,
        device=args.device,
    )

    print()
    print(f"  video    : {result.video_path}")
    print(f"  report   : {result.report_path}")
    print(f"  labels   : {result.labels_path or 'skipped'}")
    print(f"  decision : {result.report['decision'].upper()}")
    print(f"  score    : {result.report['plausibility_score']:.2f}")
    exit_with_report_decision(result.report)


if __name__ == "__main__":
    main()
