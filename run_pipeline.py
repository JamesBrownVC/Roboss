import json
import argparse
import sys
from pathlib import Path

from gemini_service import generate_video, label_video

# Local verifier components (real API — see verifier/__main__.py)
from verifier.config import DEFAULT_THRESHOLDS
from verifier.extract import extract_evidence
from verifier.checks import run_all_checks
from verifier.gate2 import run_gate2
from verifier.report import build_report, save_report


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end generation, verification, and labeling.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt for video generation.")
    parser.add_argument("--scenario", type=str, default="scenario.example.json",
                        help="Path to scenario JSON file for gate 2 + matching.")
    parser.add_argument("--outdir", type=str, default="runs/latest",
                        help="Output directory for video and reports.")
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device, e.g. cpu / 0 (default: auto).")
    parser.add_argument("--no-gate2", action="store_true",
                        help="Skip the semantic reviewer (gate 2).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    video_path = outdir / "generated.mp4"
    report_path = outdir / "report.json"
    labels_path = outdir / "labels.json"

    th = DEFAULT_THRESHOLDS

    # Scenario metadata (dict) drives gate 2 and scenario-mismatch reporting.
    scenario = None
    scenario_file = Path(args.scenario)
    if scenario_file.exists():
        scenario = json.loads(scenario_file.read_text(encoding="utf-8"))

    # ---- Step 1: generate ----
    print("\n=== STEP 1: Video Generation ===")
    video_bytes = generate_video(args.prompt)
    video_path.write_bytes(video_bytes)
    print(f"[Pipeline] Saved generated video to {video_path}")

    # ---- Step 2: verify (gates 1 & 2) ----
    print("\n=== STEP 2: Verification (Gates 1 & 2) ===")

    print("[Verifier] Extracting pose and object tracks (YOLO/ByteTrack)...")
    evidence = extract_evidence(str(video_path), th, device=args.device)

    print("[Verifier] Running Gate 1 (physics rule engine)...")
    violations = run_all_checks(evidence, th)

    gate2_meta = None
    if not args.no_gate2:
        print(f"[Verifier] Running Gate 2 (semantic reviewer, {th.gate2_model})...")
        semantic, gate2_meta = run_gate2(str(video_path), evidence, violations,
                                         scenario, th)
        if gate2_meta.get("status") != "ok":
            print(f"[Verifier] gate 2 {gate2_meta['status']}: "
                  f"{gate2_meta.get('error', '')}")
        violations = sorted(violations + semantic,
                            key=lambda v: v.severity, reverse=True)

    report = build_report(evidence, violations, th, scenario, gate2_meta)
    save_report(report, str(report_path))
    print(f"[Pipeline] Saved verification report to {report_path}")
    print(f"[Pipeline] Decision: {report['decision'].upper()} "
          f"(Score: {report['plausibility_score']:.2f})")

    # ---- Step 3: label (only if accepted) ----
    if report["decision"] == "accept":
        print("\n=== STEP 3: Auto-Labeling ===")
        labels_data = label_video(video_bytes)
        labels_path.write_text(json.dumps(labels_data, indent=2))
        print(f"[Pipeline] Saved labels to {labels_path}")
    else:
        print("\n=== STEP 3: SKIPPED (video was rejected) ===")
        sys.exit(2)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
