"""Batch orchestrator: agents pipeline -> Gemini Omni -> verifier per job."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.config import DEFAULT_CONFIG
from agents.llm import AgentError

from .logs import LOG_STORE, batch_logger

GENERATED_DIR = Path("generated")
RUNS_FILE = GENERATED_DIR / "runs.json"

_batches: dict[str, "BatchState"] = {}
_lock = threading.Lock()


def _configure_runtime_dirs() -> None:
    ultralytics_dir = Path.cwd() / ".ultralytics"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_runs() -> list[dict[str, Any]]:
    if not RUNS_FILE.is_file():
        return []
    try:
        data = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_runs(runs: list[dict[str, Any]]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(
        json.dumps(runs[-500:], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _annotation_count(label: dict | None) -> int:
    if not label:
        return 0
    frames = label.get("frames") or []
    return sum(len(frame.get("annotations") or []) for frame in frames)


def _build_label(evidence, scenario: dict, report: dict) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    step = max(1, evidence.n_frames // 20)
    for frame_idx in range(0, evidence.n_frames, step):
        annotations: list[dict[str, Any]] = []
        for track in evidence.all_tracks:
            if frame_idx >= len(track.boxes):
                continue
            x1, y1, x2, y2 = track.boxes[frame_idx]
            annotations.append(
                {
                    "label": track.label,
                    "track_id": track.track_id,
                    "x": round(float(x1), 1),
                    "y": round(float(y1), 1),
                    "w": round(float(x2 - x1), 1),
                    "h": round(float(y2 - y1), 1),
                }
            )
        if annotations:
            frames.append({"frame": frame_idx, "annotations": annotations})
    return {
        "video_summary": scenario.get("title") or report.get("scenario") or "",
        "summary": report.get("main_reason") or "",
        "labels": scenario.get("expected_labels") or [],
        "frames": frames,
    }


@dataclass
class JobState:
    id: str
    index: int
    status: str = "queued"
    error: str | None = None
    videoUrl: str | None = None
    labeledVideoUrl: str | None = None
    reviewStatus: str = "pending"
    review: dict[str, Any] | None = None
    labelStatus: str = "pending"
    label: dict[str, Any] | None = None
    labelError: str | None = None
    renderError: str | None = None
    cameraVariant: dict[str, str] | None = None
    scenario_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "status": self.status,
            "error": self.error,
            "videoUrl": self.videoUrl,
            "labeledVideoUrl": self.labeledVideoUrl,
            "reviewStatus": self.reviewStatus,
            "review": self.review,
            "labelStatus": self.labelStatus,
            "label": self.label,
            "labelError": self.labelError,
            "renderError": self.renderError,
            "cameraVariant": self.cameraVariant,
        }


@dataclass
class BatchState:
    id: str
    status: str
    count: int
    aspect_ratio: str
    prompt: str
    jobs: list[JobState] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    created_at: float = field(default_factory=time.time)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "count": self.count,
            "completed": self.completed,
            "failed": self.failed,
            "aspect_ratio": self.aspect_ratio,
            "error": self.error,
            "jobs": [job.to_dict() for job in self.jobs],
        }


def get_batch(batch_id: str) -> BatchState | None:
    with _lock:
        return _batches.get(batch_id)


def list_stats_runs() -> list[dict[str, Any]]:
    return _load_runs()


def create_batch(*, prompt: str, aspect_ratio: str, count: int) -> BatchState:
    batch_id = uuid.uuid4().hex[:12]
    safe_count = max(1, min(5000, int(count)))
    batch = BatchState(
        id=batch_id,
        status="queued",
        count=safe_count,
        aspect_ratio=aspect_ratio if aspect_ratio in ("16:9", "9:16") else "16:9",
        prompt=prompt.strip(),
        jobs=[
            JobState(id=f"{batch_id}-job-{index + 1}", index=index + 1)
            for index in range(safe_count)
        ],
    )
    with _lock:
        _batches[batch_id] = batch
    LOG_STORE.append(
        f"Batch {batch_id} queued ({safe_count} video(s), {batch.aspect_ratio})",
        agent="api",
        batch_id=batch_id,
    )
    worker = threading.Thread(
        target=_run_batch,
        args=(batch_id,),
        name=f"batch-{batch_id}",
        daemon=True,
    )
    worker.start()
    return batch


def _fail_batch(batch_id: str, message: str, *, agent: str = "api") -> None:
    LOG_STORE.append(message, level="error", agent=agent, batch_id=batch_id)
    with _lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return
        batch.status = "failed"
        batch.error = message
        batch.completed = 0
        batch.failed = batch.count
        for job in batch.jobs:
            job.status = "failed"
            job.error = message
            if job.reviewStatus == "pending":
                job.reviewStatus = "failed"
            if job.labelStatus == "pending":
                job.labelStatus = "failed"


def _update_counters(batch: BatchState) -> None:
    batch.completed = sum(1 for job in batch.jobs if job.status == "completed")
    batch.failed = sum(1 for job in batch.jobs if job.status == "failed")
    if batch.completed + batch.failed >= batch.count:
        if batch.failed == batch.count:
            batch.status = "failed"
        elif batch.failed:
            batch.status = "partial"
        else:
            batch.status = "completed"
    elif batch.status not in ("failed",):
        batch.status = "running"


def _persist_job_run(batch: BatchState, job: JobState, started_at: float) -> None:
    runs = _load_runs()
    known = {run["id"] for run in runs}
    if job.id in known:
        return
    total_seconds = max(1, round(time.time() - started_at))
    runs.append(
        {
            "id": job.id,
            "createdAt": _utc_now(),
            "status": job.status,
            "labelStatus": job.labelStatus,
            "reviewStatus": job.reviewStatus,
            "cameraVariant": (job.cameraVariant or {}).get("name"),
            "aspectRatio": batch.aspect_ratio,
            "zoneCount": _annotation_count(job.label),
            "totalSeconds": total_seconds,
        }
    )
    _save_runs(runs)


def _run_batch(batch_id: str) -> None:
    log = batch_logger(batch_id)
    batch_started = time.time()

    try:
        _configure_runtime_dirs()
        from agents.pipeline import run_pipeline
        from verifier.checks import run_all_checks
        from verifier.config import DEFAULT_THRESHOLDS
        from verifier.extract import extract_evidence
        from verifier.report import build_report
        from verifier.viz import render_annotated_video

        from .video_gen import VideoGenError, generate_video
    except Exception as exc:
        _fail_batch(
            batch_id,
            f"Batch worker failed before pipeline start: {type(exc).__name__}: {exc}",
            agent="api",
        )
        return

    with _lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return
        batch.status = "running"

    batch_dir = GENERATED_DIR / batch_id
    bundle_dir = batch_dir / "bundle"

    try:
        log("Starting agent pipeline (intent -> contract -> scenarios -> compile)")
        result = run_pipeline(
            batch.prompt,
            str(bundle_dir),
            DEFAULT_CONFIG,
            count=batch.count,
            make_canvas=True,
            make_start_frames=True,
            progress=log,
        )
    except AgentError as exc:
        message = f"Agent pipeline failed: {exc}"
        _fail_batch(batch_id, message, agent="pipeline")
        return
    except Exception as exc:
        message = f"Agent pipeline crashed: {type(exc).__name__}: {exc}"
        _fail_batch(batch_id, message, agent="pipeline")
        return

    scenarios = result.scenarios or []
    if not scenarios:
        message = "Agent pipeline produced no valid scenarios"
        _fail_batch(batch_id, message, agent="validator")
        return

    log(
        f"Pipeline complete: {len(scenarios)} scenario(s), "
        f"{len(result.dropped)} dropped",
        agent="pipeline",
    )

    for job in batch.jobs:
        scenario = scenarios[(job.index - 1) % len(scenarios)]
        job.scenario_id = scenario.get("scenario_id")
        angle = scenario.get("camera", {}).get("angle", "front_view")
        job.cameraVariant = {
            "name": angle,
            "title": scenario.get("title") or f"Video {job.index}",
        }

    for job in batch.jobs:
        scenario = scenarios[(job.index - 1) % len(scenarios)]
        job_log = batch_logger(batch_id, job.id)
        job_started = time.time()
        scenario_id = scenario.get("scenario_id") or f"sc_{job.index}"
        job_dir = batch_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)

        raw_path = job_dir / f"{scenario_id}.mp4"
        labeled_path = job_dir / f"{scenario_id}_labeled.mp4"
        report_path = job_dir / f"{scenario_id}_report.json"

        with _lock:
            job.status = "generating"
            batch.status = "running"

        start_frame_path = result.start_frame_paths.get(scenario_id)
        start_frame = None
        if start_frame_path and Path(start_frame_path).is_file():
            start_frame = Path(start_frame_path).read_bytes()
        elif result.canvas_path and Path(result.canvas_path).is_file():
            start_frame = Path(result.canvas_path).read_bytes()
            job_log("Start frame missing; using canvas anchor instead", agent="canvas")

        try:
            generate_video(
                prompt=scenario["video_prompt"],
                output_path=raw_path,
                aspect_ratio=batch.aspect_ratio,
                start_frame=start_frame,
                duration_seconds=float(
                    scenario.get("camera", {}).get("duration_seconds", 8.0)
                ),
                log=job_log,
            )
        except Exception as exc:
            message = str(exc) if isinstance(exc, VideoGenError) else f"{type(exc).__name__}: {exc}"
            job_log(message, level="error", agent="omni")
            with _lock:
                job.status = "failed"
                job.error = message
                _update_counters(batch)
            _persist_job_run(batch, job, job_started)
            continue

        rel_raw = f"/generated/{batch_id}/{job.id}/{raw_path.name}"
        with _lock:
            job.videoUrl = rel_raw
            job.status = "reviewing"
            job.reviewStatus = "running"

        job_log("Verifier: extracting pose and object tracks", agent="verifier")
        try:
            evidence = extract_evidence(str(raw_path), DEFAULT_THRESHOLDS)
            job_log(
                f"Verifier: {len(evidence.person_tracks)} person(s), "
                f"{len(evidence.object_tracks)} object(s), "
                f"{evidence.n_frames} frame(s)",
                agent="verifier",
            )
            violations = run_all_checks(evidence, DEFAULT_THRESHOLDS)
            packet = scenario.get("verifier_packet") or {}
            report = build_report(evidence, violations, DEFAULT_THRESHOLDS, packet)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            issues = [
                f"[{v['severity']:.2f}] {v['type']}: {v['reason']}"
                for v in report.get("violations", [])[:8]
            ]
            passed = report.get("plausible", False)
            job_log(
                f"Verifier decision: {report.get('decision', 'unknown')} "
                f"(score {report.get('plausibility_score')})",
                agent="verifier",
                level="info" if passed else "warn",
            )

            with _lock:
                job.reviewStatus = "passed" if passed else "failed"
                job.review = {
                    "feedback": report.get("main_reason"),
                    "summary": report.get("main_reason"),
                    "issues": issues,
                    "score": report.get("plausibility_score"),
                    "decision": report.get("decision"),
                }
                if not passed:
                    job.error = report.get("main_reason")

            with _lock:
                job.status = "labeling"
                job.labelStatus = "running"
            job_log("Building annotation zones from extracted tracks", agent="verifier")
            label = _build_label(evidence, scenario, report)
            with _lock:
                job.label = label
                job.labelStatus = "completed"

            with _lock:
                job.status = "rendering"
            job_log("Rendering labeled preview video", agent="verifier")
            try:
                render_annotated_video(
                    str(raw_path),
                    evidence,
                    violations,
                    str(labeled_path),
                )
                with _lock:
                    job.labeledVideoUrl = (
                        f"/generated/{batch_id}/{job.id}/{labeled_path.name}"
                    )
            except Exception as exc:
                job_log(f"Annotated render failed: {exc}", level="warn", agent="verifier")
                with _lock:
                    job.renderError = str(exc)

            with _lock:
                job.status = "completed" if passed else "failed"
                if passed:
                    job.error = None
                _update_counters(batch)
        except Exception as exc:
            job_log(f"Verifier failed: {exc}", level="error", agent="verifier")
            with _lock:
                job.status = "failed"
                job.error = str(exc)
                job.reviewStatus = "failed"
                job.labelStatus = "failed"
                job.labelError = str(exc)
                _update_counters(batch)

        _persist_job_run(batch, job, job_started)

    elapsed = round(time.time() - batch_started)
    with _lock:
        _update_counters(batch)
        final = batch.status
    LOG_STORE.append(
        f"Batch {batch_id} finished: {final} in {elapsed}s "
        f"({batch.completed} ok, {batch.failed} failed)",
        agent="api",
        batch_id=batch_id,
    )

