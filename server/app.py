"""FastAPI application exposing the Roboss video pipeline to the frontend."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from env_loader import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, Response, StreamingResponse

from .batches import GENERATED_DIR, ReferenceAsset, create_batch, create_demo_dog_batch, get_batch, list_stats_runs
from .logs import LOG_STORE

app = FastAPI(title="Roboss API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")

ROOT_DIR = Path(__file__).resolve().parent.parent
DOG_DEMO_VIDEO = ROOT_DIR / "demo" / "label_demo" / "ai_dog.mp4"
CACHE_DEMO_BATCH_ID = "d79e3fa0e9ce"


def read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def gemini_api_key_configured() -> bool:
    load_dotenv()
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def generated_url_to_path(url: str | None) -> Path | None:
    if url == "/api/demo/dog/video":
        return DOG_DEMO_VIDEO if DOG_DEMO_VIDEO.is_file() else None
    if not url or not url.startswith("/generated/"):
        return None
    relative = Path(url.removeprefix("/generated/"))
    if any(part in ("", ".", "..") for part in relative.parts):
        return None
    root = GENERATED_DIR.resolve()
    path = (GENERATED_DIR / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_file() else None


def generated_batch_dir(batch_id: str) -> Path | None:
    if not batch_id or any(char in batch_id for char in "\\/"):
        return None
    root = GENERATED_DIR.resolve()
    path = (GENERATED_DIR / batch_id).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_dir() else None


def generated_batch_payload(batch_id: str) -> dict[str, Any] | None:
    batch_dir = generated_batch_dir(batch_id)
    if batch_dir is None:
        return None

    intent = read_json_file(batch_dir / "bundle" / "intent.json") or {}
    scenarios_payload = read_json_file(batch_dir / "bundle" / "scenarios.json") or {}
    scenarios = {
        scenario.get("scenario_id"): scenario
        for scenario in scenarios_payload.get("scenarios", [])
        if isinstance(scenario, dict) and scenario.get("scenario_id")
    }
    jobs: list[dict[str, Any]] = []
    for job_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        videos = sorted(job_dir.glob("*.mp4"))
        raw_path = next((path for path in videos if not path.stem.endswith("_labeled")), None)
        labeled_path = next((path for path in videos if path.stem.endswith("_labeled")), None)
        if raw_path is None and labeled_path is None:
            continue
        report_path = next(job_dir.glob("*_report.json"), None)
        report = read_json_file(report_path) if report_path else None
        scenario_id = (report or {}).get("video_id") or (raw_path or labeled_path).stem.replace("_labeled", "")
        scenario = scenarios.get(scenario_id, {})
        decision = str((report or {}).get("decision") or "accept").lower()
        plausible = bool((report or {}).get("plausible", decision != "reject"))
        accepted = plausible and decision not in {"reject", "rejected"}
        review_status = "passed" if accepted else "rejected"
        label = {
            "video_summary": scenario.get("title") or (report or {}).get("scenario") or scenario_id,
            "summary": "Recovered cache demo data from generated/d79e3fa0e9ce.",
            "labels": scenario.get("expected_labels") or [],
            "synthetic_scenario": scenario,
            "verifier_report": report,
        }
        violation_messages = [
            f"[{violation.get('severity')}] {violation.get('type')}: {violation.get('reason')}"
            for violation in (report or {}).get("violations", [])[:8]
            if isinstance(violation, dict)
        ]
        index = len(jobs) + 1
        jobs.append(
            {
                "id": job_dir.name,
                "index": index,
                "status": "completed" if accepted else "failed",
                "error": None if accepted else (report or {}).get("main_reason", "Rejected by verifier."),
                "videoUrl": f"/generated/{batch_id}/{job_dir.name}/{raw_path.name}" if raw_path else None,
                "labeledVideoUrl": (
                    f"/generated/{batch_id}/{job_dir.name}/{labeled_path.name}"
                    if labeled_path
                    else None
                ),
                "reviewStatus": review_status,
                "review": {
                    "feedback": (report or {}).get("main_reason"),
                    "summary": (report or {}).get("main_reason"),
                    "issues": violation_messages,
                    "violations": violation_messages,
                    "score": (report or {}).get("plausibility_score"),
                    "plausibility_score": (report or {}).get("plausibility_score"),
                    "decision": (report or {}).get("decision"),
                } if report else None,
                "labelStatus": "completed",
                "label": label,
                "labelError": None,
                "renderError": None,
                "cameraVariant": {
                    "name": scenario_id,
                    "title": scenario.get("title") or f"Recovered cache video {index}",
                },
            }
        )

    if not jobs:
        return None
    failed = sum(1 for job in jobs if job["status"] == "failed" or job["reviewStatus"] == "rejected")
    return {
        "id": batch_id,
        "status": "completed",
        "count": len(jobs),
        "completed": len(jobs) - failed,
        "failed": failed,
        "aspect_ratio": "16:9",
        "error": None,
        "prompt": intent.get("raw_intention") or "",
        "jobs": jobs,
    }


class ReferencePayload(BaseModel):
    data: str
    mimeType: str = Field(alias="mimeType")

    model_config = {"populate_by_name": True}


class CreateBatchRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"
    count: int = 1
    reference_image: ReferencePayload | None = None
    reference_video: ReferencePayload | None = None


def decode_reference(body: CreateBatchRequest) -> ReferenceAsset | None:
    payload = body.reference_image or body.reference_video
    if payload is None:
        return None

    kind = "image" if body.reference_image else "video"
    mime_type = payload.mimeType.strip().lower()
    if kind == "image" and not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Reference image must have an image MIME type.")
    if kind == "video" and not mime_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Reference video must have a video MIME type.")

    try:
        data = base64.b64decode(payload.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Reference asset is not valid base64.") from exc

    if not data:
        raise HTTPException(status_code=400, detail="Reference asset is empty.")
    return ReferenceAsset(kind=kind, mime_type=mime_type, data=data)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "roboss",
        "version": "0.1.0",
        "geminiApiKeyConfigured": gemini_api_key_configured(),
        "requiredEnv": ["GEMINI_API_KEY"],
    }


@app.post("/api/videos")
def post_videos(body: CreateBatchRequest) -> dict[str, Any]:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if not gemini_api_key_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is missing. Add it to the project root .env file "
                "as GEMINI_API_KEY=your_key_here, then restart the backend."
            ),
        )
    reference = decode_reference(body)
    batch = create_batch(
        prompt=prompt,
        aspect_ratio=body.aspect_ratio,
        count=body.count,
        reference=reference,
    )
    return batch.to_dict()


@app.post("/api/demo/dog")
def post_demo_dog() -> dict[str, Any]:
    if not DOG_DEMO_VIDEO.is_file():
        raise HTTPException(status_code=404, detail="Dog demo video not found.")
    batch = create_demo_dog_batch(video_url="/api/demo/dog/video")
    return batch.to_dict()


@app.post("/api/demo/cache")
def post_demo_cache() -> dict[str, Any]:
    payload = generated_batch_payload(CACHE_DEMO_BATCH_ID)
    if payload is None:
        raise HTTPException(status_code=404, detail="Cache demo batch not found.")
    LOG_STORE.append(
        f"Loaded cached demo batch {CACHE_DEMO_BATCH_ID} from generated/{CACHE_DEMO_BATCH_ID}",
        agent="api",
        batch_id=CACHE_DEMO_BATCH_ID,
    )
    return payload


@app.get("/api/demo/dog/video")
def get_demo_dog_video() -> FileResponse:
    if not DOG_DEMO_VIDEO.is_file():
        raise HTTPException(status_code=404, detail="Dog demo video not found.")
    return FileResponse(DOG_DEMO_VIDEO, media_type="video/mp4")


@app.get("/api/batches/{batch_id}")
def get_batch_route(batch_id: str) -> dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is not None:
        return batch.to_dict()

    restored = generated_batch_payload(batch_id)
    if restored is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return restored


@app.get("/api/batches/{batch_id}/download")
def download_batch_videos(batch_id: str) -> Response:
    batch = get_batch(batch_id)
    batch_dir = generated_batch_dir(batch_id)
    if batch is None and batch_dir is None:
        raise HTTPException(status_code=404, detail="Batch not found.")

    files: list[tuple[str, Path]] = []
    data_files: list[tuple[str, bytes]] = []
    if batch is not None:
        data_files.append(
            (
                "manifest.json",
                json.dumps(batch.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"),
            )
        )
        for job in batch.jobs:
            for kind, url in (("raw", job.videoUrl), ("labeled", job.labeledVideoUrl)):
                path = generated_url_to_path(url)
                if path is not None:
                    files.append((f"{kind}/{job.index:03d}_{path.name}", path))
            if job.label is not None:
                data_files.append(
                    (
                        f"labels/{job.index:03d}_{job.id}_labels.json",
                        json.dumps(job.label, indent=2, ensure_ascii=False).encode("utf-8"),
                    )
                )
            if job.review is not None:
                data_files.append(
                    (
                        f"reviews/{job.index:03d}_{job.id}_review.json",
                        json.dumps(job.review, indent=2, ensure_ascii=False).encode("utf-8"),
                    )
                )
    elif batch_dir is not None:
        for path in sorted(item for item in batch_dir.rglob("*") if item.is_file()):
            relative_path = path.relative_to(batch_dir).as_posix()
            files.append((f"generated/{batch_id}/{relative_path}", path))

    if not files and not data_files:
        raise HTTPException(status_code=404, detail="No dataset files are available for this batch yet.")

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w", compression=zipfile.ZIP_STORED) as archive:
        used_names: set[str] = set()
        def unique_archive_name(archive_name: str) -> str:
            suffix = 2
            while archive_name in used_names:
                base = Path(archive_name)
                archive_name = f"{base.parent}/{base.stem}_{suffix}{base.suffix}"
                suffix += 1
            used_names.add(archive_name)
            return archive_name

        for archive_name, path in files:
            archive.write(path, unique_archive_name(archive_name))
        for archive_name, data in data_files:
            archive.writestr(unique_archive_name(archive_name), data)

    filename = f"roboss-{batch_id}-dataset.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(
        content=archive_bytes.getvalue(),
        media_type="application/zip",
        headers=headers,
    )


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return {"runs": list_stats_runs()}


@app.get("/api/logs")
def get_logs(since: int | None = None) -> dict[str, Any]:
    entries = LOG_STORE.get_since(since)
    return {"entries": entries}


@app.get("/api/logs/stream")
async def stream_logs() -> StreamingResponse:
    async def event_generator():
        queue = LOG_STORE.subscribe()
        try:
            for entry in LOG_STORE.get_since():
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            LOG_STORE.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
