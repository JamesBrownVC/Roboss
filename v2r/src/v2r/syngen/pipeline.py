"""Syngen job orchestration: request -> generate -> verify -> ingest -> deliver.

Every step is idempotent against the data/syngen/{job_id}/ directory so the
CLI can resume a job at any point (`request` then later `deliver`, or the
one-shot `run`).
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import V2RConfig
from ..orchestrator.runner import resolve_stages, run_episode
from ..schema.io import write_json_model
from ..schema.models import (
    CameraCalibration,
    CameraInfo,
    SessionCalibration,
    VideoProbe,
)
from ..schema.workspace import EpisodeWorkspace
from ..session.runner import (
    session_create,
    session_fuse,
    session_sync,
    session_triangulate,
)
from .backends import GenResult, generate_all, get_backend
from .spec import CameraSpec, JobDirs, JobSpec, expand_request, make_job_id
from .verify import VerificationRecord, verify_all


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _update_status(dirs: JobDirs, phase: str, **extra) -> None:
    status = {}
    if dirs.status_json.is_file():
        try:
            status = json.loads(dirs.status_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    status["job_id"] = dirs.job_id
    status["phase"] = phase
    status["updated_at"] = _now()
    status.setdefault("history", []).append({"phase": phase, "at": _now()})
    status.update(extra)
    dirs.status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")


def load_status(dirs: JobDirs) -> dict:
    if dirs.status_json.is_file():
        return json.loads(dirs.status_json.read_text(encoding="utf-8"))
    return {"job_id": dirs.job_id, "phase": "missing"}


# ---------------------------------------------------------------------------
# Step 1+2: request
# ---------------------------------------------------------------------------


def create_request(
    cfg: V2RConfig,
    user_prompt: str,
    n_variants: int = 4,
    n_cameras: int = 2,
    job_id: Optional[str] = None,
    backend: str = "auto",
    duration_s: int = 4,
    extra_params: Optional[dict] = None,
    use_llm: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[JobSpec, JobDirs]:
    """Expand the user request into a parametrized job spec and persist it.

    n_variants = number of motion/lighting event variations; each event is
    rendered from n_cameras viewpoints, so total videos = n_variants * n_cameras.
    """
    job_id = job_id or make_job_id(user_prompt)
    dirs = JobDirs(cfg.root, job_id).create()
    if dirs.spec_json.is_file():
        log(f"[request] job {job_id!r} already has spec.json; reusing")
        return dirs.load_spec(), dirs
    resolved = get_backend(backend)
    spec = expand_request(
        user_prompt, n_events=n_variants, n_cameras=n_cameras, job_id=job_id,
        backend=resolved.name, duration_s=duration_s,
        extra_params=extra_params, use_llm=use_llm, log=log,
    )
    dirs.save_spec(spec)
    _update_status(dirs, "requested",
                   n_events=len(spec.events), n_cameras=len(spec.cameras),
                   n_videos=len(spec.variants), director=spec.director,
                   backend=spec.backend, prompt=user_prompt)
    log(f"[request] spec saved: {dirs.spec_json} "
        f"({len(spec.events)} events x {len(spec.cameras)} cams = {len(spec.variants)} videos)")
    return spec, dirs


# ---------------------------------------------------------------------------
# Step 3: generate
# ---------------------------------------------------------------------------


def run_generation(
    spec: JobSpec,
    dirs: JobDirs,
    backend_name: Optional[str] = None,
    max_workers: int = 4,
    log: Callable[[str], None] = print,
) -> list[GenResult]:
    pending = [v for v in spec.variants if not dirs.video_mp4(v.variant_id).is_file()]
    if not pending:
        log("[generate] all videos already present; skipping")
        return [GenResult(v.variant_id, True, dirs.video_mp4(v.variant_id), "cached")
                for v in spec.variants]
    backend = get_backend(backend_name or spec.backend)
    _update_status(dirs, "generating", backend=backend.name)
    sub_spec = spec.model_copy(update={"variants": pending})
    results = generate_all(sub_spec, dirs, backend, max_workers=max_workers, log=log)
    n_ok = sum(1 for r in results if r.ok)
    _update_status(dirs, "generated", generated_ok=n_ok,
                   generated_failed=len(results) - n_ok)
    return results


# ---------------------------------------------------------------------------
# Step 4: verify
# ---------------------------------------------------------------------------


def run_verification(
    cfg: V2RConfig,
    spec: JobSpec,
    dirs: JobDirs,
    max_workers: int = 3,
    log: Callable[[str], None] = print,
) -> list[VerificationRecord]:
    done = all(dirs.verification_json(v.variant_id).is_file() for v in spec.variants)
    if done:
        log("[verify] verification already complete; loading cached records")
        return [VerificationRecord.model_validate_json(
                    dirs.verification_json(v.variant_id).read_text(encoding="utf-8"))
                for v in spec.variants]
    _update_status(dirs, "verifying")
    records = verify_all(spec, dirs, cfg.verbs, max_workers=max_workers, log=log)
    counts = {k: sum(1 for r in records if r.verdict == k)
              for k in ("accept", "reject", "review")}
    _update_status(dirs, "verified", verification=counts)
    log(f"[verify] accept={counts['accept']} review={counts['review']} reject={counts['reject']}")
    return records


# ---------------------------------------------------------------------------
# Step 5: ingest into V2R (episodes + multi-view sessions)
# ---------------------------------------------------------------------------


def _calibration_from_priors(spec: JobSpec, dirs: JobDirs,
                             cam_ids: list[str]) -> SessionCalibration:
    """Build a SessionCalibration from the director's camera parameters
    (spec.json) — these seed the multi-view tier instead of the default
    synthetic semicircle."""
    from ..stages._synthetic import probe_from_video

    by_id = {c.cam_id: c for c in spec.cameras}
    cams: list[CameraCalibration] = []
    for cam_id in cam_ids:
        prior: CameraSpec = by_id.get(cam_id, CameraSpec(cam_id=cam_id))
        # any variant video from this cam gives width/height
        probe: Optional[VideoProbe] = None
        for v in spec.variants:
            if v.cam_id == cam_id and dirs.video_mp4(v.variant_id).is_file():
                try:
                    probe = probe_from_video(dirs.video_mp4(v.variant_id))
                    break
                except Exception:
                    continue
        if probe is None:
            probe = VideoProbe(width=1280, height=720, fps=30.0, n_frames=96, duration_s=4.0)
        # fx from the director's FOV prior: fx = W / (2 tan(fov/2))
        fov = math.radians(max(20.0, min(120.0, prior.fov_deg)))
        fx = probe.width / (2.0 * math.tan(fov / 2.0))
        intr = CameraInfo(
            width=probe.width, height=probe.height,
            fx=fx, fy=fx, cx=probe.width / 2.0, cy=probe.height / 2.0,
            scale_source="synthetic",
        )
        az = math.radians(prior.azimuth_deg)
        eye = np.array([prior.distance_m * math.sin(az),
                        prior.distance_m * math.cos(az),
                        prior.height_m])
        target = np.array([0.0, 0.0, 0.9])
        fwd = target - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up2 = np.cross(right, fwd)
        T = np.eye(4)
        T[:3, :3] = np.stack([right, -up2, fwd], axis=1)
        T[:3, 3] = eye
        cams.append(CameraCalibration(cam_id=cam_id, intrinsics=intr,
                                      T_world_cam=T.tolist()))
    return SessionCalibration(
        method="synthetic", reference_cam=cam_ids[0], cameras=cams, confidence=0.8)


def _attach_labels(ws: EpisodeWorkspace, record: VerificationRecord,
                   spec: JobSpec) -> None:
    """Write syngen labels + provenance into the episode's semantics/."""
    labels = record.labels.model_dump() if record.labels else {}
    payload = {
        "syngen_job_id": spec.job_id,
        "variant_id": record.variant_id,
        "event_id": record.event_id,
        "cam_id": record.cam_id,
        "user_prompt": spec.user_prompt,
        "labels": labels,
        "verification_verdict": record.verdict,
        "source": "synthesized",
    }
    ws.semantics_dir.mkdir(parents=True, exist_ok=True)
    (ws.semantics_dir / "syngen_labels.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def run_ingest(
    cfg: V2RConfig,
    spec: JobSpec,
    dirs: JobDirs,
    records: list[VerificationRecord],
    robots: Optional[list[str]] = None,
    mode: str = "synthetic",
    log: Callable[[str], None] = print,
) -> dict:
    """Feed accepted variants into V2R: single-view episodes always; a
    multi-view session per event that has >= 2 accepted cameras."""
    robots = robots or ["g1"]
    accepted = [r for r in records if r.verdict == "accept"]
    _update_status(dirs, "ingesting", accepted=len(accepted))

    ingest_report: dict = {"episodes": [], "sessions": [], "errors": []}

    # single-view episodes. run_episode names the workspace after the video
    # stem, so stage a job-prefixed copy to keep episode ids unique per job.
    staging = dirs.root / "_ingest_src"
    staging.mkdir(parents=True, exist_ok=True)
    for rec in accepted:
        video = staging / f"{spec.job_id}_{rec.variant_id}.mp4"
        if not video.is_file():
            shutil.copy2(dirs.video_mp4(rec.variant_id), video)
        try:
            result = run_episode(cfg, video, robots=robots,
                                 stages=resolve_stages("all"),
                                 mode_override=mode, log=lambda m: None)
            ws = EpisodeWorkspace(cfg.workspaces_root, result.episode_id)
            _attach_labels(ws, rec, spec)
            ingest_report["episodes"].append({
                "variant_id": rec.variant_id,
                "episode_id": result.episode_id,
                "workspace": str(result.workspace),
                "accepted": result.accepted,
                "errors": result.errors,
            })
            log(f"[ingest] {rec.variant_id} -> episode {result.episode_id} "
                f"(pipeline accepted={result.accepted})")
        except Exception as e:
            msg = f"{rec.variant_id}: {type(e).__name__}: {e}"
            ingest_report["errors"].append(msg)
            log(f"[ingest] ERROR {msg}")

    # multi-view sessions: one per event with >=2 accepted cameras
    by_event: dict[str, list[VerificationRecord]] = {}
    for rec in accepted:
        by_event.setdefault(rec.event_id, []).append(rec)
    for event_id, recs in sorted(by_event.items()):
        cams = {r.cam_id: dirs.video_mp4(r.variant_id) for r in recs}
        if len(cams) < 2:
            continue
        session_id = f"syngen_{spec.job_id}_{event_id}"
        try:
            sw = session_create(cfg, session_id, cams, log=lambda m: None)
            session_sync(sw, cfg, mode=mode)
            # seed calibration from director priors BEFORE triangulation
            cal = _calibration_from_priors(spec, dirs, sorted(cams))
            write_json_model(sw.calibration_json, cal)
            session_triangulate(sw, cfg, mode=mode)
            eid = session_fuse(sw, cfg, robots, mode=mode, log=lambda m: None)
            ingest_report["sessions"].append({
                "event_id": event_id,
                "session_id": session_id,
                "cameras": sorted(cams),
                "fused_episode_id": eid,
                "workspace": str(sw.root),
                "calibration_from_spec_priors": True,
            })
            log(f"[ingest] event {event_id} -> session {session_id} "
                f"({len(cams)} cams, calibration seeded from spec.json)")
        except Exception as e:
            msg = f"session {session_id}: {type(e).__name__}: {e}"
            ingest_report["errors"].append(msg)
            log(f"[ingest] ERROR {msg}")

    dirs.ingest_json.write_text(json.dumps(ingest_report, indent=2), encoding="utf-8")
    _update_status(dirs, "ingested",
                   episodes=len(ingest_report["episodes"]),
                   sessions=len(ingest_report["sessions"]))
    return ingest_report


# ---------------------------------------------------------------------------
# Step 6: deliver
# ---------------------------------------------------------------------------


def run_delivery(
    cfg: V2RConfig,
    spec: JobSpec,
    dirs: JobDirs,
    records: list[VerificationRecord],
    ingest_report: dict,
    log: Callable[[str], None] = print,
) -> Path:
    delivery = dirs.delivery_dir
    (delivery / "episodes").mkdir(parents=True, exist_ok=True)

    accepted = [r for r in records if r.verdict == "accept"]
    rejected = [r for r in records if r.verdict == "reject"]
    review = [r for r in records if r.verdict == "review"]

    # copy LeRobot exports of pipeline-accepted episodes
    n_exported = 0
    for ep in ingest_report.get("episodes", []):
        ws = EpisodeWorkspace(cfg.workspaces_root, ep["episode_id"])
        if not ep.get("accepted") or not ws.lerobot_dir.is_dir():
            continue
        dst = delivery / "episodes" / ep["episode_id"]
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ws.lerobot_dir, dst / "lerobot")
        # carry semantics labels alongside the export
        lbl = ws.semantics_dir / "syngen_labels.json"
        if lbl.is_file():
            shutil.copy2(lbl, dst / "syngen_labels.json")
        n_exported += 1

    (delivery / "rejected.json").write_text(json.dumps([
        {"variant_id": r.variant_id, "verdict": r.verdict,
         "reasons": r.verdict_reasons}
        for r in rejected + review
    ], indent=2), encoding="utf-8")

    pipeline_accepted = sum(1 for e in ingest_report.get("episodes", []) if e.get("accepted"))
    funnel = [
        ("requested (events x cams)", len(spec.variants)),
        ("generated", sum(1 for v in spec.variants if dirs.video_mp4(v.variant_id).is_file())),
        ("verification: accept", len(accepted)),
        ("verification: review", len(review)),
        ("verification: reject", len(rejected)),
        ("v2r pipeline accepted (feasibility+QA)", pipeline_accepted),
        ("exported (LeRobot)", n_exported),
        ("multi-view sessions", len(ingest_report.get("sessions", []))),
    ]

    card = _dataset_card(spec, records, funnel, ingest_report)
    (delivery / "README.md").write_text(card, encoding="utf-8")

    manifest = {
        "job_id": spec.job_id,
        "delivered_at": _now(),
        "funnel": {k: v for k, v in funnel},
        "episodes": [e["episode_id"] for e in ingest_report.get("episodes", [])
                     if e.get("accepted")],
        "sessions": [s["session_id"] for s in ingest_report.get("sessions", [])],
    }
    (delivery / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _update_status(dirs, "delivered", exported=n_exported,
                   delivery=str(delivery))
    log(f"[deliver] {delivery} (exported {n_exported} episodes)")
    return delivery


def _dataset_card(spec: JobSpec, records: list[VerificationRecord],
                  funnel: list[tuple[str, int]], ingest_report: dict) -> str:
    lines = [
        f"# Synthetic dataset: {spec.job_id}",
        "",
        f"Generated {spec.created_at} by the V2R syngen loop.",
        "",
        f"**User prompt:** {spec.user_prompt}",
        "",
        f"- Director (prompt expansion): `{spec.director}` "
        f"(Gemini generateContent, temperature 0.9)" if spec.director == "gemini"
        else f"- Director (prompt expansion): `{spec.director}` (deterministic offline)",
        f"- Video backend: `{spec.backend}`",
        f"- Events (motion/lighting variations): {len(spec.events)}",
        f"- Cameras per event: {len(spec.cameras)}",
        "",
        "## World notes",
        "",
        spec.world_notes or "(none)",
        "",
        "## Cameras (also used as multi-view calibration priors)",
        "",
        "| cam | height m | distance m | azimuth deg | fov deg |",
        "|-----|----------|------------|-------------|---------|",
    ]
    for c in spec.cameras:
        lines.append(f"| {c.cam_id} | {c.height_m:.1f} | {c.distance_m:.1f} "
                     f"| {c.azimuth_deg:.0f} | {c.fov_deg:.0f} |")
    lines += ["", "## Yield funnel", "", "| stage | count |", "|-------|-------|"]
    for name, count in funnel:
        lines.append(f"| {name} | {count} |")
    lines += ["", "## Verification results", "",
              "| variant | verdict | vlm judge | physics ok | skills |",
              "|---------|---------|-----------|------------|--------|"]
    for r in records:
        judge = r.vlm.judge_source if r.vlm else "-"
        phys = "yes" if (r.physics and r.physics.physics_ok) else "no"
        skills = ", ".join(r.labels.skills) if r.labels else "-"
        lines.append(f"| {r.variant_id} | {r.verdict} | {judge} | {phys} | {skills} |")
    sessions = ingest_report.get("sessions", [])
    if sessions:
        lines += ["", "## Multi-view sessions", ""]
        for s in sessions:
            lines.append(f"- `{s['session_id']}`: cams {', '.join(s['cameras'])} "
                         f"-> fused episode `{s['fused_episode_id']}`")
    lines += ["", "---",
              "Rejected/review variants and reasons: `rejected.json`. ",
              "Episodes are LeRobot-v3 fragment exports under `episodes/`.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One-shot
# ---------------------------------------------------------------------------


def run_job(
    cfg: V2RConfig,
    user_prompt: Optional[str] = None,
    job_id: Optional[str] = None,
    n_variants: int = 4,
    n_cameras: int = 2,
    backend: str = "auto",
    duration_s: int = 4,
    robots: Optional[list[str]] = None,
    use_llm: bool = True,
    max_workers: int = 4,
    log: Callable[[str], None] = print,
) -> Path:
    """Steps 1-6 end to end. Resumes from existing artifacts when present."""
    if job_id and JobDirs(cfg.root, job_id).spec_json.is_file():
        dirs = JobDirs(cfg.root, job_id)
        spec = dirs.load_spec()
        log(f"[run] resuming job {job_id!r}")
    else:
        if not user_prompt:
            raise ValueError("a prompt is required for a new job")
        spec, dirs = create_request(
            cfg, user_prompt, n_variants=n_variants, n_cameras=n_cameras,
            job_id=job_id, backend=backend, duration_s=duration_s,
            use_llm=use_llm, log=log)
    run_generation(spec, dirs, max_workers=max_workers, log=log)
    records = run_verification(cfg, spec, dirs, log=log)
    ingest_report = run_ingest(cfg, spec, dirs, records, robots=robots, log=log)
    return run_delivery(cfg, spec, dirs, records, ingest_report, log=log)
