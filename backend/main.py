"""FastAPI backend — generate then agentically label a video (image + sound).

The labeling step is agentic: one focused Gemini pass per concern, streamed
to the UI as each pass completes:

    1. inventory — ALL visible elements (objects) + actions, no boxes
    2. tracking  — tight bounding-box keyframes for each inventoried object
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

TRACKING_PROMPT_TEMPLATE = f"""You are a precision object-detection system. An annotator inventoried these
objects in this video (label, time range, description):

{{objects}}

For EACH object above, return one label (track "object", keep the exact same
label name, t_start, t_end, detail and confidence) with boxes: bounding-box
keyframes sampled every ~1 second across its time range (at least 2 keyframes;
more if the object or camera moves fast). Each keyframe:
- t: timestamp in seconds
- box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000, TIGHT around the
  visible extent of the object AT THAT EXACT MOMENT — hug the pixels, do not
  pad, do not reuse a previous box if the object moved.
Accuracy of box placement is the ONLY goal of this pass.
{COMMON_RULES}"""

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


def _run_pass(client: genai.Client, uploaded, prompt: str) -> list[dict]:
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
    label_set: LabelSet = response.parsed
    return [l.model_dump() for l in label_set.labels]


async def _label_stream(version: Version):
    def event(payload: dict) -> str:
        return json.dumps(payload) + "\n"

    client = genai.Client()
    all_labels: list[dict] = []

    try:
        yield event({"type": "pass_start", "pass": "upload", "message": "Uploading video to Gemini..."})
        uploaded = await asyncio.to_thread(_upload_video, client, version.video)

        # Pass 1 — inventory (objects + actions, no boxes yet)
        yield event({"type": "pass_start", "pass": "inventory", "message": "Pass 1/3 — visual inventory (objects, actions)..."})
        visual = await asyncio.to_thread(_run_pass, client, uploaded, VISUAL_PROMPT)
        visual = [l for l in visual if l["track"] in ("object", "action")]
        all_labels += visual
        yield event({"type": "labels", "pass": "inventory", "labels": visual})

        # Pass 2 — dedicated box tracking for the inventoried objects
        objects = [l for l in visual if l["track"] == "object"]
        if objects:
            yield event({"type": "pass_start", "pass": "tracking", "message": f"Pass 2/3 — tracking {len(objects)} object(s) (tight boxes)..."})
            tracking_prompt = TRACKING_PROMPT_TEMPLATE.format(
                objects="\n".join(
                    f"- {o['label']} [{o['t_start']}s–{o['t_end']}s]: {o['detail']}" for o in objects
                )
            )
            tracked = await asyncio.to_thread(_run_pass, client, uploaded, tracking_prompt)
            tracked = [{**l, "track": "object"} for l in tracked if l.get("boxes")]
            if tracked:
                # replace inventory objects with their tracked versions
                all_labels = [l for l in all_labels if l["track"] != "object"] + tracked
            yield event({"type": "labels", "pass": "tracking", "labels": tracked})

        # Pass 3 — audio analysis
        yield event({"type": "pass_start", "pass": "audio", "message": "Pass 3/3 — soundtrack analysis (speech, music, SFX)..."})
        audio = await asyncio.to_thread(_run_pass, client, uploaded, AUDIO_PROMPT)
        audio = [{**l, "track": "audio", "boxes": None} for l in audio]
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
