"""FastAPI backend — generate then agentically label a video (image + sound).

The labeling step is agentic: one focused Gemini pass per concern, streamed
to the UI as each pass completes:

    1. inventory — ALL visible elements (objects) + actions (video call)
    2. tracking  — frames extracted at 2 fps (ffmpeg), one image-detection
                   call per frame, box keyframes assembled per object
    3. audio     — soundtrack analysis (speech transcript, music, SFX, ambient)

Endpoints:
    POST /api/generate    NDJSON stream -> creates version v1, v2...
    POST /api/label       NDJSON stream -> agentic labeling passes
    POST /api/transcribe  Audio blob -> text (mic input)
    GET  /api/videos/{id} Serves the stored MP4
    GET  /api/versions    Version strip data

Run:
    source .venv/bin/activate
    uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEN_MODEL = "gemini-omni-flash-preview"
LABEL_MODEL = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# Versioned in-memory store (session-scoped)
# ---------------------------------------------------------------------------


@dataclass
class Version:
    id: str
    prompt: str
    interaction_id: Optional[str]
    video: bytes
    labels: list = field(default_factory=list)
    summary: Optional[dict] = None


STORE: dict[str, Version] = {}
ORDER: list[str] = []


def _new_version(prompt: str, interaction_id: Optional[str], video: bytes) -> Version:
    vid = f"v{len(ORDER) + 1}"
    version = Version(id=vid, prompt=prompt, interaction_id=interaction_id, video=video)
    STORE[vid] = version
    ORDER.append(vid)
    return version


# ---------------------------------------------------------------------------
# Label schema — the front/back contract
# ---------------------------------------------------------------------------

Track = Literal["object", "action", "audio"]


class BoxKeyframe(BaseModel):
    # timestamp in seconds, box [ymin, xmin, ymax, xmax] normalized to 0-1000
    t: float
    box_2d: list[int]


class Label(BaseModel):
    t_start: float
    t_end: float
    track: Track
    label: str
    detail: str
    confidence: float
    # box keyframes sampled over time (objects only)
    boxes: Optional[list[BoxKeyframe]] = None


class LabelSet(BaseModel):
    labels: list[Label]


def _summary(labels: list[dict]) -> dict:
    return {
        "objects": sum(1 for l in labels if l["track"] == "object"),
        "actions": sum(1 for l in labels if l["track"] == "action"),
        "audio": sum(1 for l in labels if l["track"] == "audio"),
    }


# ---------------------------------------------------------------------------
# Agentic labeling pipeline
# ---------------------------------------------------------------------------

COMMON_RULES = """
Rules:
- Every label MUST cite a time range (t_start/t_end in seconds) you can defend.
- confidence is 0.0-1.0. Be honest: do not assert what you cannot see or hear clearly.
- label names in English snake_case, detail is one short sentence.
- Return ONLY labels of the requested track type(s).
"""

VISUAL_PROMPT = f"""You are a meticulous video annotator. Watch this video and produce a COMPLETE
inventory of the IMAGE track:

- track "object": every distinct visible element — people, vehicles, equipment,
  clothing items, signage / on-screen text, notable scenery elements (max 12,
  most salient first). No boxes in this pass.
  IMPORTANT: if the video contains shot changes / scene cuts, or an object
  disappears and reappears, split it into SEPARATE labels (one per continuous
  appearance) — a time range must never span across a cut.
- track "action": everything that happens — movements, interactions, camera moves
  if significant (max 8).
{COMMON_RULES}"""

TRACKING_FPS = 2  # frames per second sampled for box detection

DETECT_PROMPT_TEMPLATE = """Detect the following objects in this image (they may or may not be visible):

{objects}

Return one detection per object that is clearly visible, using EXACTLY the label
names above. box_2d = [ymin, xmin, ymax, xmax] normalized to 0-1000, TIGHT
around the visible extent of the object — hug the pixels, do not pad.
Skip objects that are not visible in this image. confidence is 0.0-1.0.
"""


class Detection(BaseModel):
    label: str
    box_2d: list[int]
    confidence: float


class FrameDetections(BaseModel):
    detections: list[Detection]


def _extract_frames(video: bytes, fps: int = TRACKING_FPS, width: int = 640) -> list[tuple[float, bytes]]:
    """Extract (timestamp, jpeg_bytes) frames from an mp4 using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.mp4"
        src.write_bytes(video)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-vf", f"fps={fps},scale={width}:-2",
                "-q:v", "3",
                str(Path(tmp) / "f%04d.jpg"),
            ],
            check=True,
            timeout=60,
        )
        frames = []
        for f in sorted(Path(tmp).glob("f*.jpg")):
            idx = int(f.stem[1:]) - 1
            frames.append((idx / fps, f.read_bytes()))
        return frames


def _detect_frame(client: genai.Client, jpeg: bytes, object_names: list[str]) -> list[dict]:
    prompt = DETECT_PROMPT_TEMPLATE.format(objects="\n".join(f"- {n}" for n in object_names))
    response = client.models.generate_content(
        model=LABEL_MODEL,
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FrameDetections,
            temperature=0.0,
        ),
    )
    if response.parsed is None:
        return []
    return [d.model_dump() for d in response.parsed.detections]

AUDIO_PROMPT = f"""You are a meticulous audio annotator. LISTEN to this video's soundtrack and
produce a COMPLETE inventory of the AUDIO track only (track "audio"):

- speech: one label per utterance, detail = verbatim transcript
- music: style and mood
- sound effects: engine noise, beeps, impacts, footsteps...
- ambient/background sound
No boxes for audio labels.
{COMMON_RULES}"""


def _upload_video(client: genai.Client, video: bytes):
    uploaded = client.files.upload(file=io.BytesIO(video), config={"mime_type": "video/mp4"})
    deadline = time.monotonic() + 120
    while uploaded.state and uploaded.state.name == "PROCESSING":
        if time.monotonic() > deadline:
            raise RuntimeError("Video processing timed out")
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name == "FAILED":
        raise RuntimeError("Video processing failed")
    return uploaded


def _run_pass(client: genai.Client, uploaded, prompt: str, retries: int = 1) -> list[dict]:
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=LABEL_MODEL,
                contents=[uploaded, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LabelSet,
                    temperature=0.0,
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                ),
            )
            if response.parsed is None:
                raise RuntimeError(f"Unparseable labeling response: {(response.text or '')[:200]}")
            return [l.model_dump() for l in response.parsed.labels]
        except Exception as err:
            last_err = err
    raise last_err  # type: ignore[misc]


async def _label_stream(version: Version):
    def event(payload: dict) -> str:
        return json.dumps(payload) + "\n"

    # explicit timeout: without it a stuck API call hangs the pass forever
    client = genai.Client(http_options={"timeout": 180_000})
    all_labels: list[dict] = []

    async def run_with_heartbeat(prompt: str, pass_name: str):
        """Run a pass in a thread, yielding progress events while it runs."""
        task = asyncio.create_task(asyncio.to_thread(_run_pass, client, uploaded, prompt))
        started = time.monotonic()
        while not task.done():
            await asyncio.wait({task}, timeout=2)
            if not task.done():
                yield {"type": "progress", "pass": pass_name, "elapsed": round(time.monotonic() - started)}
        yield {"type": "_result", "labels": task.result()}

    try:
        yield event({"type": "pass_start", "pass": "upload", "message": "Uploading video to Gemini..."})
        uploaded = await asyncio.to_thread(_upload_video, client, version.video)

        # Pass 1 — inventory (objects + actions, no boxes yet)
        yield event({"type": "pass_start", "pass": "inventory", "message": "Pass 1/3 — visual inventory (objects, actions)..."})
        visual: list[dict] = []
        async for ev in run_with_heartbeat(VISUAL_PROMPT, "inventory"):
            if ev["type"] == "_result":
                visual = [l for l in ev["labels"] if l["track"] in ("object", "action")]
            else:
                yield event(ev)
        all_labels += visual
        yield event({"type": "labels", "pass": "inventory", "labels": visual})

        # Pass 2 — per-frame detection: extract frames, detect all objects in
        # each frame in parallel, then assemble box keyframes per object
        objects = [l for l in visual if l["track"] == "object"][:10]
        if objects:
            yield event({"type": "pass_start", "pass": "tracking", "message": "Pass 2/3 — extracting frames..."})
            frames = await asyncio.to_thread(_extract_frames, version.video)
            object_names = [o["label"] for o in objects]
            total = len(frames)
            yield event({"type": "pass_start", "pass": "tracking", "message": f"Pass 2/3 — detecting {len(object_names)} object(s) in {total} frames..."})
            started = time.monotonic()

            semaphore = asyncio.Semaphore(8)

            async def detect_one(t: float, jpeg: bytes):
                async with semaphore:
                    detections = await asyncio.wait_for(
                        asyncio.to_thread(_detect_frame, client, jpeg, object_names), timeout=45
                    )
                    return t, detections

            pending = {asyncio.ensure_future(detect_one(t, jpeg)) for t, jpeg in frames}
            keyframes: dict[str, list[dict]] = {}
            done_count = 0
            while pending:
                done, pending = await asyncio.wait(pending, timeout=2, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    done_count += 1
                    try:
                        t, detections = task.result()
                    except Exception:
                        continue  # failed/timed-out frame: skip
                    for d in detections:
                        if d["label"] in object_names and len(d["box_2d"]) == 4:
                            keyframes.setdefault(d["label"], []).append({"t": t, "box_2d": d["box_2d"]})
                if pending:
                    yield event(
                        {
                            "type": "progress",
                            "pass": "tracking",
                            "elapsed": round(time.monotonic() - started),
                            "message": f"Pass 2/3 — detecting objects... {done_count}/{total} frames",
                        }
                    )

            # assemble tracked labels: detected time range + dense keyframes
            tracked_all: list[dict] = []
            for o in objects:
                kfs = sorted(keyframes.get(o["label"], []), key=lambda k: k["t"])
                if kfs:
                    tracked_all.append(
                        {
                            **o,
                            "t_start": kfs[0]["t"],
                            "t_end": kfs[-1]["t"] + 1 / TRACKING_FPS,
                            "boxes": kfs,
                        }
                    )
            if tracked_all:
                tracked_names = {t["label"] for t in tracked_all}
                all_labels = [
                    l for l in all_labels if not (l["track"] == "object" and l["label"] in tracked_names)
                ] + tracked_all
                yield event({"type": "labels", "pass": "tracking", "labels": tracked_all})

        # Pass 3 — audio analysis
        yield event({"type": "pass_start", "pass": "audio", "message": "Pass 3/3 — soundtrack analysis (speech, music, SFX)..."})
        audio: list[dict] = []
        async for ev in run_with_heartbeat(AUDIO_PROMPT, "audio"):
            if ev["type"] == "_result":
                audio = [{**l, "track": "audio", "boxes": None} for l in ev["labels"]]
            else:
                yield event(ev)
        all_labels += audio
        yield event({"type": "labels", "pass": "audio", "labels": audio})

        version.labels = all_labels
        version.summary = _summary(all_labels)
        yield event(
            {
                "type": "done",
                "video_id": version.id,
                "labels": all_labels,
                "summary": version.summary,
            }
        )
    except Exception as err:
        yield event({"type": "error", "error": str(err)})


# ---------------------------------------------------------------------------
# Generation (Omni Flash)
# ---------------------------------------------------------------------------


def _extract_video(interaction) -> dict:
    video = getattr(interaction, "output_video", None)
    if video is not None and (getattr(video, "data", None) or getattr(video, "uri", None)):
        return {"data": getattr(video, "data", None), "uri": getattr(video, "uri", None)}
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) == "model_output":
            for part in getattr(step, "content", None) or []:
                if getattr(part, "type", None) == "video":
                    return {"data": getattr(part, "data", None), "uri": getattr(part, "uri", None)}
    raise RuntimeError("No video returned by the model")


def _generate_sync(prompt: str, aspect_ratio: str) -> tuple[dict, Optional[str]]:
    client = genai.Client()
    interaction = client.interactions.create(
        model=GEN_MODEL,
        input=prompt,
        response_format={"type": "video", "aspect_ratio": aspect_ratio},
    )
    return _extract_video(interaction), getattr(interaction, "id", None)


async def _generation_stream(prompt: str, aspect_ratio: str):
    def event(payload: dict) -> str:
        return json.dumps(payload) + "\n"

    if not os.environ.get("GEMINI_API_KEY"):
        yield event({"type": "error", "error": "GEMINI_API_KEY missing in .env"})
        return

    started = time.monotonic()
    yield event({"type": "status", "message": "Generating video..."})

    task = asyncio.create_task(asyncio.to_thread(_generate_sync, prompt, aspect_ratio))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=2)
            if not task.done():
                yield event({"type": "progress", "elapsed": round(time.monotonic() - started)})

        video, interaction_id = task.result()

        data = video.get("data")
        if not data and video.get("uri"):
            yield event({"type": "status", "message": "Downloading video..."})
            async with httpx.AsyncClient(timeout=120) as http:
                res = await http.get(
                    video["uri"], headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
                )
                res.raise_for_status()
                video_bytes = res.content
        else:
            video_bytes = base64.b64decode(data)

        version = _new_version(prompt, interaction_id, video_bytes)
        yield event(
            {
                "type": "done",
                "video_id": version.id,
                "elapsed": round(time.monotonic() - started),
            }
        )
    except Exception as err:
        yield event({"type": "error", "error": str(err)})


def _transcribe_sync(data: bytes, mime_type: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model=LABEL_MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime_type),
            "Transcribe this audio verbatim. Return only the transcription text, nothing else.",
        ],
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Roboss — generate & label")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    aspectRatio: str = "16:9"


class LabelRequest(BaseModel):
    video_id: str


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is required")
    return StreamingResponse(
        _generation_stream(req.prompt.strip(), req.aspectRatio),
        media_type="application/x-ndjson",
    )


@app.post("/api/label")
async def label(req: LabelRequest):
    version = STORE.get(req.video_id)
    if version is None:
        raise HTTPException(404, f"Unknown version {req.video_id}")
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY missing in .env")
    return StreamingResponse(_label_stream(version), media_type="application/x-ndjson")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    try:
        text = await asyncio.to_thread(
            _transcribe_sync, data, audio.content_type or "audio/webm"
        )
    except Exception as err:
        raise HTTPException(502, f"Transcription failed: {err}")
    return {"text": text}


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    version = STORE.get(video_id)
    if version is None:
        raise HTTPException(404, f"Unknown version {video_id}")
    return Response(content=version.video, media_type="video/mp4")


@app.get("/api/versions")
async def versions():
    return [
        {"id": v.id, "prompt": v.prompt, "summary": v.summary}
        for v in (STORE[vid] for vid in ORDER)
    ]


@app.get("/api/versions/{video_id}/labels")
async def version_labels(video_id: str):
    version = STORE.get(video_id)
    if version is None:
        raise HTTPException(404, f"Unknown version {video_id}")
    return {"video_id": version.id, "labels": version.labels, "summary": version.summary}
