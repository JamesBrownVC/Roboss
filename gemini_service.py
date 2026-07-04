import os
import io
import time
import base64
import httpx
from typing import Literal, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GEN_MODEL = "gemini-omni-flash-preview"
LABEL_MODEL = "gemini-3.5-flash"
TRACKING_CONCURRENCY = int(os.environ.get("TRACKING_CONCURRENCY", "4"))

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

Track = Literal["object", "action", "audio"]

class BoxKeyframe(BaseModel):
    t: float
    box_2d: list[int]

class Label(BaseModel):
    t_start: float
    t_end: float
    track: Track
    label: str
    detail: str
    confidence: float
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
# Prompts
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

# ---------------------------------------------------------------------------
# Generation Logic
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

def _resolve_video_bytes(video: dict) -> bytes:
    data = video.get("data")
    if data:
        return base64.b64decode(data) if isinstance(data, str) else data
    uri = video.get("uri")
    if uri:
        res = httpx.get(uri, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=120)
        res.raise_for_status()
        return res.content
    raise RuntimeError("No video data or uri returned by the model")

def generate_video(prompt: str, aspect_ratio: str = "16:9") -> bytes:
    """Generate a video and return its raw MP4 bytes."""
    print(f"[Generator] Generating video for prompt: '{prompt}'")
    client = genai.Client()
    input_parts = [{"type": "text", "text": prompt}]
    
    interaction = client.interactions.create(
        model=GEN_MODEL,
        input=input_parts,
        response_format={"type": "video", "aspect_ratio": aspect_ratio},
    )
    
    video_dict = _extract_video(interaction)
    video_bytes = _resolve_video_bytes(video_dict)
    print("[Generator] Video generation successful.")
    return video_bytes

# ---------------------------------------------------------------------------
# Labeling Logic
# ---------------------------------------------------------------------------

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

def label_video(video_bytes: bytes) -> dict:
    """Label a video (inventory, tracking, audio) and return the full label set and summary."""
    print("[Labeler] Starting labeling process...")
    client = genai.Client()
    all_labels = []

    print("[Labeler] Uploading video...")
    uploaded = _upload_video(client, video_bytes)

    # Pass 1 — inventory
    print("[Labeler] Pass 1/3 — visual inventory...")
    visual = _run_pass(client, uploaded, VISUAL_PROMPT)
    visual = [l for l in visual if l["track"] in ("object", "action")]
    all_labels += visual

    # Pass 2 — dedicated box tracking
    objects = [l for l in visual if l["track"] == "object"]
    tracked = []
    if objects:
        print(f"[Labeler] Pass 2/3 — tracking {len(objects)} object(s)...")
        for i, obj in enumerate(objects):
            print(f"[Labeler] Tracking object {i+1}/{len(objects)}: {obj['label']}...")
            tracking_prompt = TRACKING_PROMPT_TEMPLATE.format(
                objects=f"- {obj['label']} [{obj['t_start']}s–{obj['t_end']}s]: {obj['detail']}"
            )
            try:
                res = _run_pass(client, uploaded, tracking_prompt)
                tracked.extend(res)
            except Exception as e:
                print(f"[Labeler] Warning: Tracking {obj['label']} failed: {e}")

        tracked = [{**l, "track": "object"} for l in tracked if l.get("boxes")]
        if tracked:
            all_labels = [l for l in all_labels if l["track"] != "object"] + tracked

    # Pass 3 — audio analysis
    print("[Labeler] Pass 3/3 — audio analysis...")
    audio = _run_pass(client, uploaded, AUDIO_PROMPT)
    audio = [{**l, "track": "audio", "boxes": None} for l in audio]
    all_labels += audio

    summary = _summary(all_labels)
    print("[Labeler] Labeling complete.")
    
    return {
        "labels": all_labels,
        "summary": summary
    }
