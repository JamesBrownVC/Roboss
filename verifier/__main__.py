"""CLI: python -m verifier <video.mp4> [more videos ...] [options]

Single video:  extraction -> gate 1 -> (gate 2 || semantic annotator, in
parallel threads) -> report. Multiple videos: each video is processed in
its own worker (--parallel), scenario packets matched per video via
--scenario-dir or shared via --scenario.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .checks import run_all_checks
from .config import Thresholds
from .extract import extract_evidence
from .report import build_report, save_report


def _load_scenario(args, video: str) -> dict | None:
    """Per-video packet from --scenario-dir (matched by stem), else the
    shared --scenario file."""
    if args.scenario_dir:
        candidate = Path(args.scenario_dir) / (Path(video).stem + ".json")
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        print(f"      no scenario packet for {Path(video).name} "
              f"(expected {candidate.name})")
    if args.scenario:
        return json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    return None


def _make_thresholds(args) -> Thresholds:
    th = Thresholds()  # fresh instance: never mutate the shared default
    if args.max_frames:
        th.max_frames = args.max_frames
    if args.gate2_model:
        th.gate2_model = args.gate2_model
    if args.gate2_frames:
        th.gate2_frames = args.gate2_frames
    return th


def process_video(video: str, args, tag: str = "") -> dict:
    """Full pipeline for one video; returns the report dict."""
    th = _make_thresholds(args)
    scenario = _load_scenario(args, video)

    def log(msg: str) -> None:
        print(f"{tag}{msg}")

    log(f"[1/3] Extracting pose and object tracks from {video} ...")
    evidence = extract_evidence(video, th, device=args.device,
                                progress=not tag)
    log(f"      {len(evidence.person_tracks)} person track(s), "
        f"{len(evidence.object_tracks)} object track(s), "
        f"{evidence.n_frames} frames")

    log("[2/3] Running gate 1 (physics rule engine) ...")
    violations = run_all_checks(evidence, th)

    # gate 2 (judge) and semantic annotator (dataset text) are independent
    # Gemini calls over the same evidence -> run them in parallel threads
    gate2_meta = None
    semantics = None
    jobs = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if args.gate2:
            from .gate2 import run_gate2
            log(f"      gate 2 (semantic reviewer, {th.gate2_model}) "
                f"+ ...")
            jobs["gate2"] = pool.submit(run_gate2, video, evidence,
                                        violations, scenario, th)
        if args.annotate:
            from .annotate import run_annotator
            log(f"      semantic annotator ({th.annotate_model}) "
                f"running in parallel ...")
            jobs["annotate"] = pool.submit(run_annotator, video, evidence,
                                           violations, scenario, th)

    if "gate2" in jobs:
        semantic_violations, gate2_meta = jobs["gate2"].result()
        if gate2_meta.get("status") != "ok":
            log(f"      gate 2 {gate2_meta['status']}: "
                f"{gate2_meta.get('error', '')}")
        violations = sorted(violations + semantic_violations,
                            key=lambda v: v.severity, reverse=True)
    if "annotate" in jobs:
        annotation, ann_meta = jobs["annotate"].result()
        if ann_meta.get("status") != "ok":
            log(f"      annotator {ann_meta['status']}: "
                f"{ann_meta.get('error', '')}")
        semantics = {**ann_meta, "annotation": annotation}

    report = build_report(evidence, violations, th, scenario, gate2_meta,
                          semantics)
    out_path = (args.out if args.out and len(args.videos) == 1
                else str(Path(video).with_suffix("")) + "_report.json")
    save_report(report, out_path)
    log(f"[3/3] Report written to {out_path}")

    if semantics and semantics.get("annotation"):
        sem_path = str(Path(video).with_suffix("")) + "_semantics.json"
        Path(sem_path).write_text(
            json.dumps(semantics["annotation"], indent=2, ensure_ascii=False),
            encoding="utf-8")
        log(f"      Semantics written to {sem_path}")

    if args.annotated and len(args.videos) == 1:
        from .viz import render_annotated_video
        render_annotated_video(video, evidence, violations, args.annotated)
        log(f"      Annotated video written to {args.annotated}")

    return report


def _print_report(report: dict) -> None:
    print(f"  decision : {report['decision'].upper()}")
    print(f"  score    : {report['plausibility_score']}")
    print(f"  reason   : {report['main_reason']}")
    for v in report["violations"]:
        print(f"   - [{v['severity']:.2f}] ({v['gate']}) {v['type']} @ frames "
              f"{v['frames'][0]}..{v['frames'][-1]}: {v['reason']}")
    annotation = (report.get("semantics") or {}).get("annotation")
    if annotation:
        print(f"  caption  : {annotation.get('global_caption', '')[:120]}")
        phases = annotation.get("action_phases", [])
        if phases:
            print("  phases   : " + " | ".join(
                f"{p['t_start']:g}-{p['t_end']:g}s {p['label']}"
                for p in phases))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="verifier",
        description="Physics-plausibility verifier for generated action "
                    "videos.")
    p.add_argument("videos", nargs="+",
                   help="one or more generated videos (mp4/avi/mov)")
    p.add_argument("--scenario",
                   help="shared metadata packet JSON (all videos)")
    p.add_argument("--scenario-dir",
                   help="directory of per-video packets named <video stem>"
                        ".json (e.g. the agents bundle's verifier_packets/)")
    p.add_argument("--out", default=None,
                   help="report path (single video only; batch always "
                        "writes <video>_report.json)")
    p.add_argument("--annotated", default=None,
                   help="write annotated demo video (single video only)")
    p.add_argument("--device", default=None,
                   help="inference device, e.g. cpu / 0 (default: auto)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap on processed frames")
    p.add_argument("--gate2", action="store_true",
                   help="also run the semantic reviewer (needs Gemini API "
                        "credentials)")
    p.add_argument("--annotate", action="store_true",
                   help="also extract semantic text data (captions, action "
                        "phases, risks, QA pairs) in parallel with labeling")
    p.add_argument("--gate2-model", default=None,
                   help="reviewer model (default: gemini-3.5-flash)")
    p.add_argument("--gate2-frames", type=int, default=None,
                   help="uniformly sampled frames to send (default: 10)")
    p.add_argument("--parallel", type=int, default=2,
                   help="videos processed concurrently in batch mode "
                        "(default: 2)")
    args = p.parse_args(argv)

    if len(args.videos) == 1:
        report = process_video(args.videos[0], args)
        print()
        _print_report(report)
        return 0 if report["plausible"] else 2

    # batch mode: several scenarios in parallel
    workers = max(1, min(args.parallel, len(args.videos)))
    print(f"Batch: {len(args.videos)} video(s), {workers} worker(s)")
    reports: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_video, video, args,
                        tag=f"[{Path(video).stem}] "): video
            for video in args.videos
        }
        for future, video in futures.items():
            try:
                reports[video] = future.result()
            except Exception as e:  # noqa: BLE001 - one video must not kill the batch
                print(f"[{Path(video).stem}] FAILED: "
                      f"{type(e).__name__}: {e}")

    print()
    accepted = 0
    for video in args.videos:
        report = reports.get(video)
        if report is None:
            print(f"  {Path(video).name}: ERROR")
            continue
        accepted += report["plausible"]
        print(f"  {Path(video).name}: {report['decision'].upper()} "
              f"(score {report['plausibility_score']})")
    print(f"\n  {accepted}/{len(args.videos)} accepted")
    return 0 if accepted == len(reports) == len(args.videos) else 2


if __name__ == "__main__":
    sys.exit(main())
