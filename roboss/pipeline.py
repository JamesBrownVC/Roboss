"""Reusable orchestration for compile, generate, verify and label steps.

The core is async: every scenario runs its own generate -> verify -> label
chain concurrently, throttled by per-stage semaphores so different resource
types overlap instead of queueing behind each other:

    gen_workers     network-bound video generation calls
    verify_workers  CPU-bound YOLO extraction (+ gate2/annotator threads)
    label_workers   network-bound labeling calls

While scenario A occupies the CPU verifying, scenarios B and C are
generating on the network and D is labeling — the pipeline stays saturated
across stages. Sync wrappers (`run_video_pipeline`, `run_e2e_pipeline`)
keep the CLI/API surface unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor
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


# --------------------------------------------------------------------------
# stage wrappers — module-level indirection so tests can monkeypatch them
# --------------------------------------------------------------------------

def _generate_video(prompt: str) -> bytes:
    from gemini_service import generate_video
    return generate_video(prompt)


def _label_video(video_bytes: bytes) -> dict[str, Any]:
    from gemini_service import label_video
    return label_video(video_bytes)


def _agent_config(deterministic: bool, start_frame_workers: int | None):
    # Gemini 3.5 removed sampling parameters (temperature et al.), so
    # "deterministic" is recorded in the manifest for reproducibility
    # bookkeeping but no longer changes the request shape.
    cfg = DEFAULT_CONFIG
    if start_frame_workers:
        cfg = replace(cfg, start_frame_workers=int(start_frame_workers))
    return cfg


# --------------------------------------------------------------------------
# scenario compilation (agents package)
# --------------------------------------------------------------------------

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
        cfg=_agent_config(deterministic_agents, workers),
        count=count,
        make_canvas=start_frames,
        make_start_frames=start_frames,
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


# --------------------------------------------------------------------------
# verification (gate 1 + gate 2 || semantic annotator)
# --------------------------------------------------------------------------

def verify_video(video_path: Path,
                 scenario: dict[str, Any] | None,
                 report_path: Path,
                 gate2: bool = True,
                 annotate: bool | None = None,
                 device: str | None = None,
                 progress=print) -> dict[str, Any]:
    from verifier.checks import run_all_checks
    from verifier.config import Thresholds
    from verifier.extract import extract_evidence
    from verifier.report import build_report, save_report

    settings = get_settings()
    do_annotate = settings.annotate_enabled if annotate is None else annotate

    th = Thresholds()  # fresh instance: never mutate the shared default
    progress("[Verifier] Extracting pose and object tracks ...")
    evidence = extract_evidence(str(video_path), th, device=device)

    progress("[Verifier] Running Gate 1 ...")
    violations = run_all_checks(evidence, th)

    # gate 2 (judge) and the semantic annotator (dataset text) are
    # independent Gemini calls -> run them in parallel threads
    gate2_meta = None
    semantics = None
    jobs = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if gate2:
            from verifier.gate2 import run_gate2
            progress(f"[Verifier] Gate 2 ({th.gate2_model}) ...")
            jobs["gate2"] = pool.submit(run_gate2, str(video_path), evidence,
                                        violations, scenario, th)
        if do_annotate:
            from verifier.annotate import run_annotator
            progress(f"[Verifier] Semantic annotator ({th.annotate_model}) "
                     f"in parallel ...")
            jobs["annotate"] = pool.submit(run_annotator, str(video_path),
                                           evidence, violations, scenario, th)

    if "gate2" in jobs:
        semantic_violations, gate2_meta = jobs["gate2"].result()
        if gate2_meta.get("status") != "ok":
            progress(f"[Verifier] Gate 2 {gate2_meta['status']}: "
                     f"{gate2_meta.get('error', '')}")
        violations = sorted(violations + semantic_violations,
                            key=lambda v: v.severity, reverse=True)
    if "annotate" in jobs:
        annotation, ann_meta = jobs["annotate"].result()
        if ann_meta.get("status") != "ok":
            progress(f"[Verifier] Annotator {ann_meta['status']}: "
                     f"{ann_meta.get('error', '')}")
        semantics = {**ann_meta, "annotation": annotation}
        if annotation:
            _write_json(report_path.parent / "semantics.json", annotation)

    report = build_report(evidence, violations, th, scenario, gate2_meta,
                          semantics)
    save_report(report, str(report_path))
    return report


# --------------------------------------------------------------------------
# async core
# --------------------------------------------------------------------------

def _make_semaphores(settings) -> dict[str, asyncio.Semaphore]:
    return {
        "generate": asyncio.Semaphore(settings.gen_workers),
        "verify": asyncio.Semaphore(settings.verify_workers),
        "label": asyncio.Semaphore(settings.label_workers),
    }


async def run_video_pipeline_async(
        prompt: str,
        outdir: str | Path | None = None,
        scenario: dict[str, Any] | None = None,
        scenario_path: str | Path | None = None,
        gate2: bool | None = None,
        label: bool | None = None,
        annotate: bool | None = None,
        device: str | None = None,
        export_robot_data: bool = False,
        robots: list[str] | None = None,
        robot_data_mode: str = "synthetic",
        robot_data_stages: str = "all",
        semaphores: dict[str, asyncio.Semaphore] | None = None,
        progress=print) -> VideoPipelineResult:
    """One scenario chain: generate -> verify (-> label, -> robot data).

    Each stage acquires its own semaphore, so many chains interleave:
    generation saturates the network while verification saturates the CPU.
    """
    from gemini_service import enforce_video_continuity_prompt

    storage = get_storage()
    settings = get_settings()
    sems = semaphores or _make_semaphores(settings)
    out = Path(outdir) if outdir else _default_run_dir("pipeline")
    out.mkdir(parents=True, exist_ok=True)

    if scenario_path and scenario is None:
        scenario = _read_json(Path(scenario_path))

    generation_prompt = enforce_video_continuity_prompt(
        scenario_generation_prompt(scenario, prompt)
    )
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

    async with sems["generate"]:
        progress("[Pipeline] Generating video ...")
        video_bytes = await asyncio.to_thread(_generate_video,
                                              generation_prompt)
    storage.save_bytes(Path(storage.relative(video_path)), video_bytes)

    use_gate2 = settings.gate2_enabled if gate2 is None else gate2
    async with sems["verify"]:
        report = await asyncio.to_thread(
            verify_video, video_path, scenario, report_path,
            use_gate2, annotate, device, progress)

    should_label = settings.label_on_accept if label is None else label
    final_labels_path = None
    if should_label and report["decision"] == "accept":
        async with sems["label"]:
            progress("[Pipeline] Labeling accepted video ...")
            labels = await asyncio.to_thread(_label_video, video_bytes)
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
        async with sems["verify"]:   # CPU-bound like verification
            robot_data = await asyncio.to_thread(
                export_video_to_robot_data,
                v2r_video_path,
                out / "robot_data",
                robots or ["g1"],
                robot_data_mode,
                robot_data_stages,
                progress,
            )
    elif export_robot_data:
        progress("[Pipeline] Robot-data export skipped because video was "
                 "rejected.")

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


async def run_e2e_pipeline_async(
        intention: str,
        count: int | None = 3,
        run_name: str | None = None,
        start_frames: bool = True,
        deterministic: bool | None = None,
        start_frame_workers: int | None = None,
        video_workers: int | None = None,
        require_acceptance: bool = True,
        gate2: bool = True,
        label: bool = True,
        annotate: bool | None = None,
        device: str | None = None,
        export_robot_data: bool = False,
        robots: list[str] | None = None,
        robot_data_mode: str = "synthetic",
        robot_data_stages: str = "all",
        progress=print) -> dict[str, Any]:
    """Idea -> N scenarios -> N concurrent generate/verify/label chains."""
    settings = get_settings()
    name = _slug(run_name or f"e2e_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir = settings.runs_dir / name
    bundle_dir = run_dir / "scenarios"

    compile_data = await asyncio.to_thread(
        lambda: compile_scenarios(
            intention=intention,
            outdir=bundle_dir,
            count=count,
            start_frames=start_frames,
            deterministic=deterministic,
            start_frame_workers=start_frame_workers,
            progress=progress,
        ))

    scenarios_file = bundle_dir / "scenarios.json"
    scenarios = _read_json(scenarios_file).get("scenarios", [])
    chain_cap = settings.video_workers if video_workers is None \
        else int(video_workers)
    chain_cap = max(1, min(chain_cap, max(1, len(scenarios))))
    sems = _make_semaphores(settings)
    chain_sem = asyncio.Semaphore(chain_cap)
    if scenarios:
        progress(f"[Batch] {len(scenarios)} scenario chain(s), caps: "
                 f"{chain_cap} chains / {settings.gen_workers} gen / "
                 f"{settings.verify_workers} verify / "
                 f"{settings.label_workers} label")

    async def _run_one(index: int, scenario: dict[str, Any]) -> dict[str, Any]:
        sid = scenario["scenario_id"]
        packet = scenario.get("verifier_packet") or {}
        packet.setdefault("video_prompt", scenario.get("video_prompt"))
        packet.setdefault("scenario_id", sid)
        scenario_out = run_dir / sid

        def tagged(msg: str) -> None:
            progress(f"[{sid}] {msg}")

        async with chain_sem:
            try:
                result = await run_video_pipeline_async(
                    prompt=scenario.get("title", sid),
                    outdir=scenario_out,
                    scenario=packet,
                    gate2=gate2,
                    label=label,
                    annotate=annotate,
                    device=device,
                    export_robot_data=export_robot_data,
                    robots=robots,
                    robot_data_mode=robot_data_mode,
                    robot_data_stages=robot_data_stages,
                    semaphores=sems,
                    progress=tagged,
                )
                status = result.report["decision"]
                code = exit_code_for_report(result.report)
            except Exception as exc:  # noqa: BLE001 - one chain must not kill the batch
                status = "error"
                code = 1
                _write_json(scenario_out / "error.json", {
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                tagged(f"FAILED: {type(exc).__name__}: {exc}")
        return {
            "index": index,
            "scenario_id": sid,
            "status": status,
            "exit_code": code,
            "outdir": str(scenario_out),
        }

    results = list(await asyncio.gather(
        *(_run_one(i, sc) for i, sc in enumerate(scenarios))))
    results.sort(key=lambda item: item["index"])
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
            "video_workers": chain_cap,       # concurrent scenario chains
            "gen_workers": settings.gen_workers,
            "verify_workers": settings.verify_workers,
            "label_workers": settings.label_workers,
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


# --------------------------------------------------------------------------
# sync wrappers — unchanged public surface for run.sh / run_pipeline.py
# --------------------------------------------------------------------------

def run_video_pipeline(*args, **kwargs) -> VideoPipelineResult:
    return asyncio.run(run_video_pipeline_async(*args, **kwargs))


def run_e2e_pipeline(*args, **kwargs) -> dict[str, Any]:
    return asyncio.run(run_e2e_pipeline_async(*args, **kwargs))


def exit_code_for_report(report: dict[str, Any]) -> int:
    return 0 if report.get("decision") == "accept" else 2


def exit_with_report_decision(report: dict[str, Any]) -> None:
    sys.exit(exit_code_for_report(report))
