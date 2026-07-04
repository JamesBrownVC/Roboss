"""Reusable orchestration for compile, generate, verify and label steps."""

from __future__ import annotations

import json
import re
import sys
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.config import DEFAULT_CONFIG
from agents.pipeline import run_pipeline as run_agents_pipeline

from .settings import get_settings
from .storage import get_storage


@dataclass
class VideoPipelineResult:
    outdir: Path
    video_path: Path
    report_path: Path
    labels_path: Path | None
    report: dict[str, Any]
    robot_data: dict[str, Any] | None = None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return slug[:80] or "run"


def _default_run_dir(prefix: str) -> Path:
    settings = get_settings()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return settings.runs_dir / f"{prefix}_{stamp}"


def scenario_generation_prompt(scenario: dict[str, Any] | None,
                               fallback_prompt: str) -> str:
    """Use the compiled prompt when available, not the short title."""
    if not scenario:
        return fallback_prompt
    return str(
        scenario.get("video_prompt")
        or scenario.get("generation_prompt")
        or scenario.get("scenario_prompt")
        or fallback_prompt
    )


def _agent_config(deterministic: bool):
    if deterministic:
        return replace(DEFAULT_CONFIG, plan_temperature=0.0,
                       variation_temperature=0.0)
    return DEFAULT_CONFIG


def compile_scenarios(intention: str,
                      outdir: str | Path | None = None,
                      count: int | None = None,
                      start_frames: bool = True,
                      deterministic: bool | None = None,
                      start_frame_workers: int | None = None,
                      progress=print) -> dict[str, Any]:
    storage = get_storage()
    settings = get_settings()
    out = Path(outdir) if outdir else _default_run_dir("bundle")
    workers = (settings.start_frame_workers if start_frame_workers is None
               else start_frame_workers)
    deterministic_agents = (settings.deterministic_agents
                            if deterministic is None else deterministic)
    result = run_agents_pipeline(
        intention=intention,
        out_dir=str(out),
        cfg=_agent_config(deterministic_agents),
        count=count,
        make_canvas=start_frames,
        make_start_frames=start_frames,
        start_frame_workers=workers,
        progress=progress,
    )
    scenario_fingerprints = {
        s["scenario_id"]: _fingerprint(s)
        for s in result.scenarios
    }
    data = {
        "outdir": str(out),
        "world_id": result.contract["world_contract"]["world_id"],
        "scene_id": result.contract.get("scene_registry", {}).get("scene_id"),
        "scenarios": [s["scenario_id"] for s in result.scenarios],
        "scenario_fingerprints": scenario_fingerprints,
        "dropped": result.dropped,
        "canvas": result.canvas_path,
        "start_frames": result.start_frame_paths,
        "bundle": str(out / "bundle.json"),
        "deterministic": deterministic_agents,
        "parallelism": {
            "start_frame_workers": workers if start_frames else 0,
        },
    }
    manifest = storage.write_manifest(out, {
        "kind": "scenario_bundle",
        "world_id": data["world_id"],
        "scene_id": data["scene_id"],
        "scenarios": data["scenarios"],
        "scenario_fingerprints": scenario_fingerprints,
        "deterministic": deterministic_agents,
        "parallelism": data["parallelism"],
    })
    data["manifest"] = manifest.url
    data["files"] = storage.collect_files(out)
    return data


def verify_video(video_path: Path,
                 scenario: dict[str, Any] | None,
                 report_path: Path,
                 gate2: bool = True,
                 device: str | None = None,
                 progress=print) -> dict[str, Any]:
    from verifier.checks import run_all_checks
    from verifier.config import DEFAULT_THRESHOLDS
    from verifier.extract import extract_evidence
    from verifier.gate2 import run_gate2
    from verifier.report import build_report, save_report

    th = DEFAULT_THRESHOLDS
    progress("[Verifier] Extracting pose and object tracks ...")
    evidence = extract_evidence(str(video_path), th, device=device)

    progress("[Verifier] Running Gate 1 ...")
    violations = run_all_checks(evidence, th)

    gate2_meta = None
    if gate2:
        progress(f"[Verifier] Running Gate 2 ({th.gate2_model}) ...")
        semantic, gate2_meta = run_gate2(str(video_path), evidence, violations,
                                         scenario, th)
        if gate2_meta.get("status") != "ok":
            progress(f"[Verifier] Gate 2 {gate2_meta['status']}: "
                     f"{gate2_meta.get('error', '')}")
        violations = sorted(violations + semantic,
                            key=lambda v: v.severity, reverse=True)

    report = build_report(evidence, violations, th, scenario, gate2_meta)
    save_report(report, str(report_path))
    return report


def run_video_pipeline(prompt: str,
                       outdir: str | Path | None = None,
                       scenario: dict[str, Any] | None = None,
                       scenario_path: str | Path | None = None,
                       gate2: bool | None = None,
                       label: bool | None = None,
                       device: str | None = None,
                       export_robot_data: bool = False,
                       robots: list[str] | None = None,
                       robot_data_mode: str = "synthetic",
                       robot_data_stages: str = "all",
                       progress=print) -> VideoPipelineResult:
    from gemini_service import generate_video, label_video

    storage = get_storage()
    settings = get_settings()
    out = Path(outdir) if outdir else _default_run_dir("pipeline")
    out.mkdir(parents=True, exist_ok=True)

    if scenario_path and scenario is None:
        scenario = _read_json(Path(scenario_path))

    generation_prompt = scenario_generation_prompt(scenario, prompt)
    scenario_id = (
        str(scenario.get("scenario_id"))
        if scenario and scenario.get("scenario_id") else None
    )
    generation_fingerprint = _fingerprint({
        "prompt": generation_prompt,
        "scenario": scenario,
    })
    storage.save_json(Path(storage.relative(out)) / "generation_request.json", {
        "input_prompt": prompt,
        "generation_prompt": generation_prompt,
        "generation_fingerprint": generation_fingerprint,
        "scenario": scenario,
    })

    video_path = out / "generated.mp4"
    report_path = out / "report.json"
    labels_path = out / "labels.json"

    progress("[Pipeline] Generating video ...")
    video_bytes = generate_video(generation_prompt)
    storage.save_bytes(Path(storage.relative(video_path)), video_bytes)

    use_gate2 = settings.gate2_enabled if gate2 is None else gate2
    report = verify_video(video_path, scenario, report_path,
                          gate2=use_gate2, device=device, progress=progress)

    should_label = settings.label_on_accept if label is None else label
    final_labels_path = None
    if should_label and report["decision"] == "accept":
        progress("[Pipeline] Labeling accepted video ...")
        labels = label_video(video_bytes)
        storage.save_json(Path(storage.relative(labels_path)), labels)
        final_labels_path = labels_path
    else:
        progress("[Pipeline] Labeling skipped.")

    robot_data = None
    if export_robot_data and report["decision"] == "accept":
        from .v2r_bridge import export_video_to_robot_data
        v2r_video_path = video_path
        if scenario_id:
            v2r_video_path = out / f"{_slug(scenario_id)}_v2r_input.mp4"
            shutil.copy2(video_path, v2r_video_path)
        robot_data = export_video_to_robot_data(
            video_path=v2r_video_path,
            outdir=out / "robot_data",
            robots=robots or ["g1"],
            mode=robot_data_mode,
            stages=robot_data_stages,
            progress=progress,
        )
    elif export_robot_data:
        progress("[Pipeline] Robot-data export skipped because video was rejected.")

    storage.write_manifest(out, {
        "kind": "verified_video",
        "decision": report["decision"],
        "plausibility_score": report["plausibility_score"],
        "scenario_id": scenario_id,
        "generation_fingerprint": generation_fingerprint,
        "video_url": storage.url_for(video_path),
        "report_url": storage.url_for(report_path),
        "labels_url": storage.url_for(final_labels_path)
        if final_labels_path else None,
        "robot_data": robot_data,
    })

    return VideoPipelineResult(
        outdir=out,
        video_path=video_path,
        report_path=report_path,
        labels_path=final_labels_path,
        report=report,
        robot_data=robot_data,
    )


def run_e2e_pipeline(intention: str,
                     count: int | None = 3,
                     run_name: str | None = None,
                     start_frames: bool = True,
                     deterministic: bool | None = None,
                     start_frame_workers: int | None = None,
                     video_workers: int | None = None,
                     require_acceptance: bool = True,
                     gate2: bool = True,
                     label: bool = True,
                     device: str | None = None,
                     export_robot_data: bool = False,
                     robots: list[str] | None = None,
                     robot_data_mode: str = "synthetic",
                     robot_data_stages: str = "all",
                     progress=print) -> dict[str, Any]:
    settings = get_settings()
    name = _slug(run_name or f"e2e_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir = settings.runs_dir / name
    bundle_dir = run_dir / "scenarios"

    compile_data = compile_scenarios(
        intention=intention,
        outdir=bundle_dir,
        count=count,
        start_frames=start_frames,
        deterministic=deterministic,
        start_frame_workers=start_frame_workers,
        progress=progress,
    )

    scenarios_file = bundle_dir / "scenarios.json"
    scenarios = _read_json(scenarios_file).get("scenarios", [])
    workers = settings.video_workers if video_workers is None else video_workers
    workers = max(1, min(int(workers or 1), max(1, len(scenarios))))
    if scenarios:
        progress(f"[Batch] Generating/verifying {len(scenarios)} scenario(s) "
                 f"with {workers} worker(s) ...")

    def _run_one(index: int, scenario: dict[str, Any]) -> dict[str, Any]:
        sid = scenario["scenario_id"]
        packet = scenario.get("verifier_packet") or {}
        packet.setdefault("video_prompt", scenario.get("video_prompt"))
        packet.setdefault("scenario_id", sid)
        scenario_out = run_dir / sid
        try:
            result = run_video_pipeline(
                prompt=scenario.get("title", sid),
                outdir=scenario_out,
                scenario=packet,
                gate2=gate2,
                label=label,
                device=device,
                export_robot_data=export_robot_data,
                robots=robots,
                robot_data_mode=robot_data_mode,
                robot_data_stages=robot_data_stages,
                progress=progress,
            )
            status = result.report["decision"]
            code = exit_code_for_report(result.report)
        except Exception as exc:
            status = "error"
            code = 1
            _write_json(scenario_out / "error.json", {
                "type": type(exc).__name__,
                "message": str(exc),
            })
        return {
            "index": index,
            "scenario_id": sid,
            "status": status,
            "exit_code": code,
            "outdir": str(scenario_out),
        }

    if workers == 1 or len(scenarios) <= 1:
        results = [_run_one(i, scenario) for i, scenario in enumerate(scenarios)]
    else:
        unordered = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_one, i, scenario)
                for i, scenario in enumerate(scenarios)
            ]
            for future in as_completed(futures):
                unordered.append(future.result())
        results = sorted(unordered, key=lambda item: item["index"])

    for item in results:
        item.pop("index", None)

    accepted = [r for r in results if r["status"] == "accept"]
    rejected = [r for r in results if r["status"] not in {"accept"}]
    batch_decision = (
        "accept"
        if (not require_acceptance or not rejected) and results
        else "reject"
    )

    summary = {
        "run_dir": str(run_dir),
        "batch_decision": batch_decision,
        "require_acceptance": require_acceptance,
        "deterministic": compile_data.get("deterministic"),
        "parallelism": {
            "start_frame_workers": compile_data.get("parallelism", {}).get(
                "start_frame_workers", 0),
            "video_workers": workers,
        },
        "compile": compile_data,
        "results": results,
        "accepted": accepted,
        "rejected": rejected,
    }
    storage = get_storage()
    storage.save_json(Path(storage.relative(run_dir)) / "summary.json",
                      summary)
    manifest = storage.write_manifest(run_dir, {
        "kind": "verified_video_batch",
        "batch_decision": batch_decision,
        "require_acceptance": require_acceptance,
        "deterministic": summary["deterministic"],
        "parallelism": summary["parallelism"],
        "summary_url": storage.url_for(run_dir / "summary.json"),
    })
    summary["manifest"] = manifest.url
    summary["files"] = storage.collect_files(run_dir)
    return summary


def exit_code_for_report(report: dict[str, Any]) -> int:
    return 0 if report.get("decision") == "accept" else 2


def exit_with_report_decision(report: dict[str, Any]) -> None:
    sys.exit(exit_code_for_report(report))
