import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import requests
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gemini-omni-flash-preview"
CRUSOE_MODEL_NAME = "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B"
CRUSOE_DEFAULT_BASE_URL = "https://api.inference.crusoecloud.com/v1"
ANNOTATION_PIPELINE_VERSION = "vlm-zones-framewise-v2"
REVIEW_PIPELINE_VERSION = "gemini-vlm-review-v1"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 15 * 60
DEFAULT_VIEW_BATCH_SIZE = 4
BATCH_WORKERS = DEFAULT_VIEW_BATCH_SIZE
LABEL_FRAME_COUNT = 1
LABEL_FRAME_WIDTH = 512
MAX_LABEL_FRAME_PAYLOAD_BYTES = 2 * 1024 * 1024
DEFAULT_GEMINI_REVIEW_MODEL = "gemini-3.5-flash"
MAX_REVIEW_REVISIONS = 0

JobStatus = Literal[
    "queued",
    "generating",
    "reviewing",
    "correcting",
    "labeling",
    "rendering",
    "completed",
    "failed",
]
LabelStatus = Literal["pending", "running", "completed", "failed"]
ReviewStatus = Literal["pending", "running", "passed", "failed"]
RenderStatus = Literal["pending", "running", "completed", "failed", "skipped"]
BatchStatus = Literal["queued", "running", "completed", "failed", "partial"]

CAMERA_VARIANTS = [
    {
        "name": "front_view",
        "title": "Front view",
        "prompt": (
            "Video 1: Front view. The camera is positioned directly in front of the hazard, facing it straight on. "
            "The cause of the hazard must be clearly visible from the beginning."
        ),
    },
    {
        "name": "rear_view",
        "title": "Rear view",
        "prompt": (
            "Video 2: Rear view. The camera is positioned behind the hazard or behind the main object, showing "
            "the incident from the opposite side. The cause must still be visible through angle, movement, or camera push-in."
        ),
    },
    {
        "name": "side_view",
        "title": "Side view",
        "prompt": (
            "Video 3: Side view. The camera is positioned at a 90-degree angle from the hazard, showing depth, "
            "instability, and movement direction clearly."
        ),
    },
    {
        "name": "high_angle_inspection",
        "title": "High-angle inspection view",
        "prompt": (
            "Video 4: High-angle inspection view. The camera is slightly above the scene, looking down at the hazard "
            "to show layout, spacing, floor marks, surrounding equipment, and the final damage pattern."
        ),
    },
]


def _build_industrial_safety_prompt(user_prompt: str, camera_variant: dict[str, str], aspect_ratio: str) -> str:
    return (
        "Create a continuous cinematic industrial safety inspection video, single unbroken shot, "
        f"{aspect_ratio}, realistic documentary training style.\n\n"
        "Use the user prompt below as the incident brief. If the brief is vague, incomplete, contradictory, "
        "or badly written, normalize it into one coherent industrial safety incident. Infer realistic missing "
        "details, but keep the scene plausible and easy to inspect visually. Across the multi-angle set, keep "
        "the same environment, same hazard cause, same warning signs, same failure sequence, and same after state. "
        "Only the camera angle changes between videos.\n\n"
        f"User incident brief:\n{user_prompt}\n\n"
        "Scene requirements: clearly establish the industrial environment, such as a warehouse, factory, food "
        "production area, chemical plant, construction site, or logistics hub. Include realistic background "
        "elements: racks, machines, pipes, pallets, warning signs, concrete floor, tools, lights, labels, dust, "
        "scratches, stains, and other domain-appropriate details.\n\n"
        "The camera starts with a clean before view of the risky situation. The visible cause of the hazard must "
        "be clearly shown before anything happens. Make the cause visually obvious and believable, not hidden. "
        "Examples include a broken wooden pallet, corrosion, loose cable, missing brace, overloaded shelf, leaking "
        "oil, damaged tire, torn safety net, cracked pipe, unstable stack, or any cause implied by the user brief.\n\n"
        "Show early warning signs before the incident: leaning boxes, loose wrapping, dripping liquid, cracked wood "
        "fibers, rust marks, vibration, moisture stains, bent metal, missing bolts, stretched straps, unstable "
        "balance, floor marks, warning labels, or other subtle stress details.\n\n"
        "The camera slowly pushes in and inspects close-up textures: damaged material, dust, labels, scratches, "
        "stains, shadows, natural lighting, and signs of stress or failure.\n\n"
        "At exactly 4 seconds, the hazard develops into a realistic incident caused directly by the visible defect. "
        "Show the failure moment clearly with physically believable motion. No explosion unless explicitly required "
        "by the user brief. No injuries, no gore, and no people harmed.\n\n"
        "After the incident, the camera continues moving around the same scene to show the after state: collapsed "
        "objects, damaged packaging, torn wrap, spilled material, broken parts, puddles, scrape marks, empty rack "
        "space, scattered tools, contamination zone, stopped machine, or other consequences implied by the incident.\n\n"
        "Camera-angle requirement for this specific output:\n"
        f"{camera_variant['prompt']}\n\n"
        "All videos in the set must follow the same structure: clean before view, visible hazard cause, warning "
        "signs, failure moment at 4 seconds, and after-state inspection. Keep the video realistic, cinematic, "
        "documentary-style, with natural industrial lighting, detailed textures, physically accurate motion, no "
        "dialogue, no text overlays, no dramatic music, ambient industrial sound only, no injuries, and no gore."
    )


jobs_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}
batches: dict[str, dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=BATCH_WORKERS)

app = FastAPI(title="Roboss Gemini Video API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


class VideoCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    count: int = Field(default=DEFAULT_VIEW_BATCH_SIZE, ge=1, le=DEFAULT_VIEW_BATCH_SIZE)


class VideoJobResponse(BaseModel):
    id: str
    batchId: str
    index: int
    status: JobStatus
    prompt: str
    basePrompt: str
    cameraVariant: dict[str, str]
    aspect_ratio: Literal["16:9", "9:16"]
    draftVideoUrl: str | None = None
    videoUrl: str | None = None
    error: str | None = None
    reviewStatus: ReviewStatus = "pending"
    review: dict[str, Any] | None = None
    reviewHistory: list[dict[str, Any]] = Field(default_factory=list)
    revisionCount: int = 0
    correctionPrompt: str | None = None
    labelStatus: LabelStatus = "pending"
    label: dict[str, Any] | None = None
    labelError: str | None = None
    labeledVideoUrl: str | None = None
    renderStatus: RenderStatus = "pending"
    renderError: str | None = None


class VideoBatchResponse(BaseModel):
    id: str
    status: BatchStatus
    prompt: str
    aspect_ratio: Literal["16:9", "9:16"]
    count: int
    completed: int
    failed: int
    jobs: list[VideoJobResponse]


def _read_env_value(env_path: Path, key_names: tuple[str, ...]) -> str | None:
    if not env_path.exists():
        return None
    values = dotenv_values(env_path)
    for key_name in key_names:
        value = values.get(key_name)
        if value:
            return value
    return None


def _get_env_value(key_names: tuple[str, ...]) -> str | None:
    for env_path in (ROOT_DIR / ".env", ROOT_DIR.parent / ".env"):
        value = _read_env_value(env_path, key_names)
        if value:
            return value

    for key_name in key_names:
        value = os.getenv(key_name)
        if value:
            return value

    return None


def get_gemini_api_key() -> str | None:
    return _get_env_value(("API_KEY_GEMINI", "GEMINI_API_KEY"))


def get_gemini_review_model() -> str:
    return _get_env_value(("GEMINI_REVIEW_MODEL",)) or DEFAULT_GEMINI_REVIEW_MODEL


def get_crusoe_api_key() -> str | None:
    return _get_env_value(("API_KEY_CRUSOE", "CRUSOE_API_KEY"))


def _normalize_api_key(api_key: str) -> str:
    value = api_key.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def get_crusoe_base_url() -> str:
    return _get_env_value(("CRUSOE_BASE_URL",)) or CRUSOE_DEFAULT_BASE_URL


def _crusoe_chat_completions_url() -> str:
    base_url = get_crusoe_base_url().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _state_name(state: Any) -> str:
    name = _get_field(state, "name")
    value = str(name or state or "").upper()
    if "." in value:
        value = value.split(".")[-1]
    if value.startswith("FILE_STATE_"):
        value = value.removeprefix("FILE_STATE_")
    return value


def _extract_output_video(interaction: Any) -> Any:
    output_video = _get_field(interaction, "output_video")
    if output_video:
        return output_video

    for step in _get_field(interaction, "steps", []) or []:
        for item in _get_field(step, "content", []) or []:
            if _get_field(item, "type") == "video":
                return item

    return None


def _extract_output_text(interaction: Any) -> str:
    output_text = _get_field(interaction, "output_text")
    if output_text:
        return str(output_text).strip()

    parts: list[str] = []
    for step in _get_field(interaction, "steps", []) or []:
        for item in _get_field(step, "content", []) or []:
            text = _get_field(item, "text")
            if text:
                parts.append(str(text).strip())
    return "\n".join(part for part in parts if part).strip()


def _extract_file_name(uri: str) -> str:
    match = re.search(r"files/([^/:?]+)", uri)
    if match:
        return f"files/{match.group(1)}"
    if uri.startswith("files/"):
        return uri.split(":")[0].split("?")[0]
    raise RuntimeError("Gemini returned a video URI that could not be parsed.")


def _download_uri_with_http(uri: str, api_key: str) -> bytes:
    request = Request(uri, headers={"x-goog-api-key": api_key})
    try:
        with urlopen(request, timeout=120) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Could not download generated video: {exc}") from exc


def _download_video_bytes(client: Any, output_video: Any, api_key: str) -> bytes:
    inline_data = _get_field(output_video, "data")
    if inline_data:
        return base64.b64decode(inline_data)

    uri = _get_field(output_video, "uri")
    if not uri:
        raise RuntimeError("Gemini did not return a video payload.")

    file_name = _extract_file_name(uri)
    started_at = time.monotonic()

    while True:
        file_info = client.files.get(name=file_name)
        state = _state_name(_get_field(file_info, "state"))

        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError("Gemini video generation failed while processing the file.")
        if time.monotonic() - started_at > POLL_TIMEOUT_SECONDS:
            raise TimeoutError("Gemini video generation timed out.")

        time.sleep(POLL_INTERVAL_SECONDS)

    for candidate in (output_video, uri, file_name):
        try:
            downloaded = client.files.download(file=candidate)
            if isinstance(downloaded, bytes):
                return downloaded
        except Exception:
            if candidate == file_name and not str(uri).startswith("http"):
                raise

    if str(uri).startswith("http"):
        return _download_uri_with_http(uri, api_key)

    raise RuntimeError("Could not download generated video from Gemini.")


GEMINI_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "missing_requirements": {"type": "array", "items": {"type": "string"}},
        "visual_qa_notes": {"type": "array", "items": {"type": "string"}},
        "correction_prompt": {"type": "string"},
        "summary": {"type": "string"},
        "feedback": {"type": "string"},
    },
    "required": ["passed", "score", "issues", "missing_requirements", "correction_prompt"],
}


def _wait_for_gemini_file(client: Any, file_info: Any) -> Any:
    started_at = time.monotonic()
    current = file_info
    while True:
        state = _state_name(_get_field(current, "state"))
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError("Gemini file processing failed.")
        if time.monotonic() - started_at > POLL_TIMEOUT_SECONDS:
            raise TimeoutError("Gemini file processing timed out.")

        time.sleep(POLL_INTERVAL_SECONDS)
        name = _get_field(current, "name")
        if not name:
            raise RuntimeError("Gemini file upload did not return a file name.")
        current = client.files.get(name=name)


def _upload_video_for_gemini(client: Any, video_path: Path) -> Any:
    file_info = client.files.upload(file=str(video_path))
    return _wait_for_gemini_file(client, file_info)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _review_fallback_correction_prompt(job: dict[str, Any], review: dict[str, Any]) -> str:
    problems = review.get("missing_requirements") or review.get("issues") or []
    problem_text = "; ".join(str(problem) for problem in problems if str(problem).strip())
    if not problem_text:
        problem_text = "The first draft does not fully satisfy the warehouse safety training prompt."
    return (
        "Revise the previous video while preserving the same warehouse incident and camera variant. "
        f"Fix these quality issues: {problem_text}. "
        "Make the pallet fall clearly visible, include a before state and an after state, and keep pallets, "
        "boxes, shrink wrap, racks, floor markings, debris, damaged boxes, and hazard zones easy to annotate. "
        f"Camera variant requirement: {job.get('cameraVariant', {}).get('prompt', '')}"
    )


def _normalise_review(raw_review: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    raw_passed = raw_review.get("passed", False)
    if isinstance(raw_passed, str):
        passed = raw_passed.strip().lower() in {"true", "yes", "pass", "passed", "ok"}
    else:
        passed = bool(raw_passed)

    score = _float_or_none(raw_review.get("score"))
    score = min(1.0, max(0.0, score)) if score is not None else (1.0 if passed else 0.0)
    issues = _string_list(raw_review.get("issues"))
    missing_requirements = _string_list(raw_review.get("missing_requirements"))
    visual_qa_notes = _string_list(raw_review.get("visual_qa_notes"))
    correction_prompt = str(raw_review.get("correction_prompt") or "").strip()

    review = {
        "schema_version": "gemini-video-review-v1",
        "pipeline": REVIEW_PIPELINE_VERSION,
        "model": get_gemini_review_model(),
        "passed": passed,
        "score": score,
        "issues": issues,
        "missing_requirements": missing_requirements,
        "visual_qa_notes": visual_qa_notes,
        "correction_prompt": correction_prompt,
        "summary": str(raw_review.get("summary") or "").strip(),
        "feedback": str(raw_review.get("feedback") or "").strip(),
    }
    if not review["passed"] and not review["correction_prompt"]:
        review["correction_prompt"] = _review_fallback_correction_prompt(job, review)
    return review


def _review_video_with_gemini(client: Any, video_path: Path, job: dict[str, Any]) -> dict[str, Any]:
    uploaded_video = _upload_video_for_gemini(client, video_path)
    video_uri = _get_field(uploaded_video, "uri")
    mime_type = _get_field(uploaded_video, "mime_type", _get_field(uploaded_video, "mimeType", "video/mp4"))
    if not video_uri:
        raise RuntimeError("Gemini file upload did not return a video URI.")

    review_prompt = (
        "You are a strict VLM quality reviewer for synthetic warehouse safety training videos. "
        "Return only JSON matching the provided schema. Evaluate whether the video can be released for "
        "object-detection training and whether it follows the prompt.\n\n"
        "Visual QA discipline:\n"
        "- Assume there are visual problems. Your job is to find them, not to confirm the draft is good.\n"
        "- Inspect the video like a bug hunt across the beginning, middle, fall moment, and aftermath.\n"
        "- Compare expected content against actual pixels: do not pass a video because the prompt says something "
        "should be present; pass only if it is visibly present.\n"
        "- Report all concerns, including minor ones, in visual_qa_notes.\n"
        "- Look for cropped or cut-off important objects, confusing camera framing, occluded pallet/fall action, "
        "missing before/after states, low contrast, motion blur that hides objects, inconsistent physics, objects "
        "too small for annotation, or details that are too dark/unclear for VLM labels.\n"
        "- If you find zero issues, look again critically before passing.\n\n"
        "Required checks:\n"
        "- visible warehouse scene with racks or storage aisle\n"
        "- one pallet falls or has clearly fallen\n"
        "- before and after states are visible\n"
        "- annotatable details are visible: pallet, cardboard boxes, shrink wrap, racks, floor markings, debris, "
        "damaged boxes, and hazard zones\n"
        "- the requested camera angle variant is respected\n"
        "- video is coherent enough for VLM training\n\n"
        f"Original user prompt:\n{job['basePrompt']}\n\n"
        f"Full generation prompt:\n{job['prompt']}\n\n"
        f"Camera variant:\n{job.get('cameraVariant', {}).get('prompt', '')}\n\n"
        "If the video fails, write a concise correction_prompt for Gemini Omni Flash to edit the previous video. "
        "The correction prompt must be in English, preserve good parts, and only request the missing or unclear elements. "
        "Write feedback in English for the UI, explaining briefly what is wrong or why the video passed."
    )

    interaction = client.interactions.create(
        model=get_gemini_review_model(),
        input=[
            {"type": "video", "uri": video_uri, "mime_type": mime_type or "video/mp4"},
            {"type": "text", "text": review_prompt},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": GEMINI_REVIEW_SCHEMA,
        },
    )
    raw_review = _extract_json_from_text(_extract_output_text(interaction), "Gemini review")
    review = _normalise_review(raw_review, job)
    review["video"] = video_path.name
    return review


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _extract_json_from_text(text: str, source: str = "Crusoe") -> dict[str, Any]:
    text = text.strip()
    if not text or text.lower() in {"none", "null"}:
        raise RuntimeError(f"{source} returned an empty JSON response.")

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"{source} returned text that could not be parsed as JSON.")


def _message_content_to_text(message: Any) -> str:
    content = _get_field(message, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(_get_field(item, "text", "")).strip()
            for item in content
            if _get_field(item, "text")
        ]
        return " ".join(part for part in parts if part)
    reasoning = _get_field(message, "reasoning")
    if isinstance(reasoning, str):
        return reasoning.strip()
    return ""


def _video_duration_seconds(video_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def _image_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(image_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None, None
    match = re.search(r"(\d+)x(\d+)", result.stdout)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _video_dimensions(video_path: Path) -> tuple[int | None, int | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None, None
    match = re.search(r"(\d+)x(\d+)", result.stdout)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _sample_video_frames(video_path: Path) -> list[dict[str, Any]]:
    duration = _video_duration_seconds(video_path)
    frame_dir = GENERATED_DIR / f"{video_path.stem}_frames"
    frame_dir.mkdir(exist_ok=True)

    for existing_frame in frame_dir.glob("frame_*.jpg"):
        existing_frame.unlink(missing_ok=True)

    sampled_frames: list[dict[str, Any]] = []
    timestamps = [
        ((index + 1) / (LABEL_FRAME_COUNT + 1)) * duration
        if duration
        else float(index)
        for index in range(LABEL_FRAME_COUNT)
    ]

    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = frame_dir / f"frame_{index:02d}.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-vf",
            f"scale={LABEL_FRAME_WIDTH}:-1:force_original_aspect_ratio=decrease",
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(frame_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            continue
        if not frame_path.exists():
            continue
        width, height = _image_dimensions(frame_path)
        sampled_frames.append(
            {
                "frame_index": index,
                "timestamp_seconds": round(timestamp, 3),
                "imageUrl": f"/generated/{frame_dir.name}/{frame_path.name}",
                "width": width,
                "height": height,
                "data": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
            }
        )

    if not sampled_frames:
        raise RuntimeError("Could not sample any frame from the generated video.")

    payload_size = sum(len(frame["data"]) for frame in sampled_frames)
    if payload_size > MAX_LABEL_FRAME_PAYLOAD_BYTES:
        raise RuntimeError(
            f"Sampled frame payload is too large ({payload_size} bytes). "
            "Reduce LABEL_FRAME_COUNT or LABEL_FRAME_WIDTH."
        )

    return sampled_frames


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _clamp_unit(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def _coordinate_scale(values: list[float], frame_width: Any = None, frame_height: Any = None) -> tuple[float, float]:
    x_values = values[0::2]
    y_values = values[1::2]
    max_x = max([abs(value) for value in x_values] + [1.0])
    max_y = max([abs(value) for value in y_values] + [1.0])
    width = _float_or_none(frame_width)
    height = _float_or_none(frame_height)

    if max_x <= 1 and max_y <= 1:
        return 1.0, 1.0
    if width and height and max_x <= width * 1.25 and max_y <= height * 1.25:
        return width, height
    if max(max_x, max_y) <= 1000:
        return 1000.0, 1000.0
    return max_x, max_y


def _normalise_box(raw_box: Any, frame_width: Any = None, frame_height: Any = None) -> dict[str, float] | None:
    if isinstance(raw_box, list) and len(raw_box) >= 4:
        values = [_float_or_none(value) for value in raw_box[:4]]
        if any(value is None for value in values):
            return None
        first, second, third, fourth = values
        if max(values) > 1 and third > first and fourth > second:
            x, y, width, height = first, second, third - first, fourth - second
        else:
            x, y, width, height = first, second, third, fourth
        if max(abs(x), abs(y), abs(width), abs(height)) > 1:
            scale_x, scale_y = _coordinate_scale([x, y, x + width, y + height], frame_width, frame_height)
            x, y, width, height = x / scale_x, y / scale_y, width / scale_x, height / scale_y
    elif isinstance(raw_box, dict):
        x = raw_box.get("x", raw_box.get("left"))
        y = raw_box.get("y", raw_box.get("top"))
        width = raw_box.get("width", raw_box.get("w"))
        height = raw_box.get("height", raw_box.get("h"))
        if width is None and raw_box.get("x2") is not None:
            width = _float_or_none(raw_box.get("x2")) - (_float_or_none(x) or 0)
        if height is None and raw_box.get("y2") is not None:
            height = _float_or_none(raw_box.get("y2")) - (_float_or_none(y) or 0)
        values = [_float_or_none(value) for value in (x, y, width, height)]
        if any(value is None for value in values):
            return None
        x, y, width, height = values
        if max(abs(x), abs(y), abs(width), abs(height)) > 1:
            scale_x, scale_y = _coordinate_scale([x, y, x + width, y + height], frame_width, frame_height)
            x, y, width, height = x / scale_x, y / scale_y, width / scale_x, height / scale_y
    else:
        return None

    normalised = {
        "x": _clamp_unit(x),
        "y": _clamp_unit(y),
        "width": _clamp_unit(width),
        "height": _clamp_unit(height),
    }
    if any(value is None for value in normalised.values()):
        return None
    if normalised["width"] <= 0 or normalised["height"] <= 0:
        return None
    if normalised["x"] + normalised["width"] > 1:
        normalised["width"] = max(0.0, 1.0 - normalised["x"])
    if normalised["y"] + normalised["height"] > 1:
        normalised["height"] = max(0.0, 1.0 - normalised["y"])
    return normalised


def _normalise_detection_label(raw_label: dict[str, Any], sampled_frames: list[dict[str, Any]]) -> dict[str, Any]:
    raw_frames = raw_label.get("frames", [])
    if not isinstance(raw_frames, list):
        raw_frames = []

    if not raw_frames:
        if isinstance(raw_label.get("annotations"), list):
            raw_frames = [
                {
                    "frame_index": sampled_frames[0]["frame_index"],
                    "annotations": raw_label["annotations"],
                }
            ]

    if not raw_frames:
        top_level_annotations: list[dict[str, Any]] = []
        meta_keys = {
            "schema_version",
            "task",
            "video_summary",
            "summary",
            "labels",
            "training_notes",
            "notes",
            "annotations",
        }
        for key, value in raw_label.items():
            if key in meta_keys:
                continue
            if isinstance(value, list) and value and all(isinstance(item, list) for item in value):
                for item in value:
                    top_level_annotations.append({"label": key, "box": item})
            elif isinstance(value, (list, dict)):
                top_level_annotations.append({"label": key, "box": value})
        if top_level_annotations and sampled_frames:
            raw_frames = [
                {
                    "frame_index": sampled_frames[0]["frame_index"],
                    "annotations": top_level_annotations,
                }
            ]

    raw_by_index: dict[int, dict[str, Any]] = {}
    for frame in raw_frames:
        if not isinstance(frame, dict):
            continue
        frame_index = frame.get("frame_index", frame.get("index"))
        try:
            raw_by_index[int(frame_index)] = frame
        except (TypeError, ValueError):
            continue

    output_frames: list[dict[str, Any]] = []
    all_labels: set[str] = set()

    for sampled in sampled_frames:
        frame_index = int(sampled["frame_index"])
        raw_frame = raw_by_index.get(frame_index, {})
        raw_annotations = raw_frame.get("annotations", raw_frame.get("objects", raw_frame.get("zones", [])))
        if not isinstance(raw_annotations, list):
            raw_annotations = []

        annotations: list[dict[str, Any]] = []
        for annotation in raw_annotations:
            if not isinstance(annotation, dict):
                continue
            raw_box = annotation.get("box", annotation.get("bbox", annotation.get("bounding_box")))
            box = _normalise_box(raw_box, sampled.get("width"), sampled.get("height"))
            label = str(annotation.get("label", annotation.get("class", "object"))).strip().lower()
            label = re.sub(r"[^a-z0-9_]+", "_", label).strip("_") or "object"
            if not box:
                continue
            confidence = _float_or_none(annotation.get("confidence", annotation.get("score")))
            annotations.append(
                {
                    "label": label,
                    "box": box,
                    "confidence": min(1.0, max(0.0, confidence)) if confidence is not None else None,
                    "notes": str(annotation.get("notes", annotation.get("description", ""))).strip(),
                }
            )
            all_labels.add(label)

        output_frames.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": sampled["timestamp_seconds"],
                "imageUrl": sampled["imageUrl"],
                "annotations": annotations,
            }
        )

    raw_labels = raw_label.get("labels")
    labels = sorted(all_labels)
    if not labels and isinstance(raw_labels, list):
        labels = [str(label).strip() for label in raw_labels if str(label).strip()]

    return {
        "schema_version": "vlm-zones-v1",
        "task": "object_detection",
        "video_summary": str(raw_label.get("video_summary", raw_label.get("summary", ""))).strip(),
        "labels": labels,
        "frames": output_frames,
        "training_notes": str(raw_label.get("training_notes", raw_label.get("notes", ""))).strip(),
    }


def _post_crusoe_chat(payload: dict[str, Any], api_key: str) -> requests.Response:
    retry_statuses = {429, 500, 502, 503, 504}
    last_response: requests.Response | None = None

    for attempt in range(3):
        response = requests.post(
            _crusoe_chat_completions_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        if response.status_code not in retry_statuses:
            return response
        last_response = response
        time.sleep(2 * (attempt + 1))

    return last_response if last_response is not None else response


def _check_crusoe_auth() -> dict[str, Any]:
    api_key = get_crusoe_api_key()
    if not api_key:
        return {
            "ok": False,
            "status": "missing_key",
            "message": "API_KEY_CRUSOE is missing.",
            "baseUrl": get_crusoe_base_url(),
            "model": CRUSOE_MODEL_NAME,
        }

    api_key = _normalize_api_key(api_key)
    models_url = f"{get_crusoe_base_url().rstrip('/')}/models"
    try:
        response = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status": "network_error",
            "message": str(exc),
            "baseUrl": get_crusoe_base_url(),
            "model": CRUSOE_MODEL_NAME,
        }

    if response.status_code in (401, 403):
        return {
            "ok": False,
            "status": "authentication_failed",
            "message": (
                "API_KEY_CRUSOE is present, but Crusoe rejected it. "
                "Use an Inference API key for this base URL."
            ),
            "httpStatus": response.status_code,
            "baseUrl": get_crusoe_base_url(),
            "model": CRUSOE_MODEL_NAME,
        }

    if response.status_code >= 400:
        return {
            "ok": False,
            "status": "request_failed",
            "message": response.text[:500],
            "httpStatus": response.status_code,
            "baseUrl": get_crusoe_base_url(),
            "model": CRUSOE_MODEL_NAME,
        }

    data = response.json()
    model_ids = [
        str(item.get("id", ""))
        for item in data.get("data", [])
        if isinstance(item, dict)
    ]
    return {
        "ok": True,
        "status": "authenticated",
        "httpStatus": response.status_code,
        "baseUrl": get_crusoe_base_url(),
        "model": CRUSOE_MODEL_NAME,
        "modelAvailable": CRUSOE_MODEL_NAME in model_ids,
        "modelCount": len(model_ids),
    }


def _label_video_from_frames(video_path: Path, prompt: str, api_key: str) -> dict[str, Any]:
    sampled_frames = _sample_video_frames(video_path)
    output_frames: list[dict[str, Any]] = []
    all_labels: set[str] = set()
    frame_errors: list[str] = []

    for frame in sampled_frames:
        payload = {
            "model": CRUSOE_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an annotation assistant for VLM training data. Return only valid JSON, no markdown. "
                        "Create approximate object-detection zones with normalized top-left x/y and width/height, "
                        "all between 0 and 1. Prefer stable snake_case labels."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Annotate frame {frame['frame_index']} at {frame['timestamp_seconds']} seconds. "
                                "Return JSON with this shape: "
                                "{\"annotations\":[{\"label\":\"pallet\",\"box\":{\"x\":0.1,\"y\":0.2,"
                                "\"width\":0.3,\"height\":0.4},\"confidence\":0.8,\"notes\":\"\"}],"
                                "\"labels\":[\"pallet\"]}. "
                                "Important labels when visible: pallet, unstable_pallet, falling_pallet, box, "
                                "damaged_box, rack, shrink_wrap, debris, floor_marking, hazard_zone, empty_rack_space. "
                                "Do not return empty annotations when obvious objects are visible. "
                                "Do not use pixel coordinates. The original generation prompt was: "
                                f"{prompt}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame['data']}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_completion_tokens": 6000,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }

        response = _post_crusoe_chat(payload, api_key)
        if response.status_code >= 400:
            if response.status_code in (401, 403):
                raise RuntimeError(
                    "API_KEY_CRUSOE is present, but Crusoe rejected it. Check that it is an Inference API key "
                    "for https://api.inference.crusoecloud.com/v1 and that it has access to "
                    f"{CRUSOE_MODEL_NAME}."
                )
            if response.status_code in (502, 503, 504):
                frame_errors.append(f"frame {frame['frame_index']}: Crusoe gateway failed")
                continue
            raise RuntimeError(f"Crusoe label request failed ({response.status_code}): {response.text[:500]}")

        data = response.json()
        message = data["choices"][0]["message"]
        text = _message_content_to_text(message)
        if not text:
            frame_errors.append(f"frame {frame['frame_index']}: empty response")
            output_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "imageUrl": frame["imageUrl"],
                    "annotations": [],
                }
            )
            continue

        raw_label = _extract_json_from_text(text)
        frame_label = _normalise_detection_label(raw_label, [frame])
        frame_output = frame_label["frames"][0]
        output_frames.append(frame_output)
        all_labels.update(frame_label.get("labels", []))

    return {
        "schema_version": "vlm-zones-v1",
        "task": "object_detection",
        "video_summary": "Sampled-frame object detection annotations for VLM training.",
        "labels": sorted(all_labels),
        "frames": output_frames,
        "training_notes": "; ".join(frame_errors),
        "label_source": "sampled_frames",
        "sampled_frames": len(sampled_frames),
    }


def _annotation_count(label: dict[str, Any]) -> int:
    frames = label.get("frames", [])
    if not isinstance(frames, list):
        return 0
    return sum(
        len(frame.get("annotations", []))
        for frame in frames
        if isinstance(frame, dict) and isinstance(frame.get("annotations"), list)
    )


def _escape_drawtext_text(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_ .-]+", "_", str(value or "object")).strip() or "object"
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _drawtext_fontfile_option() -> str:
    font_candidates = [
        Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / "segoeui.ttf",
    ]
    for font_path in font_candidates:
        if font_path.exists():
            escaped_path = str(font_path).replace("\\", "/").replace(":", "\\:")
            return f"fontfile='{escaped_path}':"
    return ""


def _annotation_filters(label: dict[str, Any], video_width: int, video_height: int) -> list[str]:
    colors = [
        "lime",
        "cyan",
        "yellow",
        "orange",
        "magenta",
        "deepskyblue",
        "springgreen",
        "red",
    ]
    filters: list[str] = []
    color_index = 0

    for frame in label.get("frames", []):
        if not isinstance(frame, dict):
            continue

        annotations = frame.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            box = annotation.get("box")
            if not isinstance(box, dict):
                continue
            x = _clamp_unit(box.get("x"))
            y = _clamp_unit(box.get("y"))
            width = _clamp_unit(box.get("width"))
            height = _clamp_unit(box.get("height"))
            if x is None or y is None or width is None or height is None:
                continue
            if width <= 0 or height <= 0:
                continue

            px = int(round(x * video_width))
            py = int(round(y * video_height))
            pw = max(2, int(round(width * video_width)))
            ph = max(2, int(round(height * video_height)))
            if px + pw > video_width:
                pw = max(2, video_width - px)
            if py + ph > video_height:
                ph = max(2, video_height - py)

            color = colors[color_index % len(colors)]
            color_index += 1
            label_text = _escape_drawtext_text(annotation.get("label"))
            text_y = max(0, py - 28)
            filters.append(
                f"drawbox=x={px}:y={py}:w={pw}:h={ph}:color={color}@0.85:t=3"
            )
            filters.append(
                "drawtext="
                f"{_drawtext_fontfile_option()}"
                f"text='{label_text}':x={px}:y={text_y}:fontsize=20:"
                f"fontcolor=black:box=1:boxcolor={color}@0.85:boxborderw=4"
            )

    return filters


def _render_labeled_video(video_path: Path, label: dict[str, Any]) -> str:
    video_width, video_height = _video_dimensions(video_path)
    if not video_width or not video_height:
        raise RuntimeError("Could not read generated video dimensions for label rendering.")

    filters = _annotation_filters(label, video_width, video_height)
    if not filters:
        raise RuntimeError("No annotations available to render into the video.")

    output_path = video_path.with_name(f"{video_path.stem}.labeled.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg label render failed: {result.stderr[:500]}")
    if not output_path.exists():
        raise RuntimeError("ffmpeg label render did not create an output video.")
    return f"/generated/{output_path.name}"


def _label_video(video_path: Path, prompt: str) -> dict[str, Any]:
    api_key = get_crusoe_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing Crusoe API key. Add API_KEY_CRUSOE to .env in the repo or parent folder."
        )
    api_key = _normalize_api_key(api_key)

    label = _label_video_from_frames(video_path, prompt, api_key)
    if _annotation_count(label) == 0:
        raise RuntimeError("Nemotron returned 0 structured zones for the sampled video frames.")
    label_path = video_path.with_suffix(".label.json")
    label_path.write_text(json.dumps(label, ensure_ascii=True, indent=2), encoding="utf-8")
    return label


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def _append_review(job_id: str, review: dict[str, Any]) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        history = list(jobs[job_id].get("reviewHistory") or [])
        history.append(review)
        jobs[job_id].update(
            {
                "review": review,
                "reviewHistory": history,
                "reviewStatus": "passed" if review.get("passed") else "failed",
                "correctionPrompt": review.get("correction_prompt") or None,
            }
        )


def _create_omni_video_interaction(
    client: Any,
    input_payload: Any,
    aspect_ratio: str,
    previous_interaction_id: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "input": input_payload,
        "response_format": {
            "type": "video",
            "delivery": "uri",
            "aspect_ratio": aspect_ratio,
        },
    }
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id
    else:
        kwargs["generation_config"] = {
            "video_config": {
                "task": "text_to_video",
            },
        }
    return client.interactions.create(**kwargs)


def _download_interaction_video(client: Any, interaction: Any, api_key: str, output_path: Path) -> None:
    output_video = _extract_output_video(interaction)
    video_bytes = _download_video_bytes(client, output_video, api_key)
    output_path.write_bytes(video_bytes)


def _generate_video(job_id: str) -> None:
    with jobs_lock:
        job = dict(jobs[job_id])

    _set_job(job_id, status="generating", error=None)

    try:
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing Gemini API key. Add API_KEY_GEMINI to .env in the repo or parent folder."
            )

        os.environ.setdefault("GEMINI_API_KEY", api_key)

        from google import genai

        client = genai.Client(api_key=api_key)
        output_path = GENERATED_DIR / f"{job_id}.mp4"

        interaction = _create_omni_video_interaction(client, job["prompt"], job["aspect_ratio"])
        _download_interaction_video(client, interaction, api_key, output_path)
        _set_job(
            job_id,
            status="reviewing",
            draftVideoUrl=None,
            error=None,
            reviewStatus="running",
        )

        current_path = output_path
        current_interaction = interaction
        revision_count = 0
        review = _review_video_with_gemini(client, current_path, job)
        _append_review(job_id, review)
        _write_json(current_path.with_suffix(".review.json"), review)

        while not review.get("passed") and revision_count < MAX_REVIEW_REVISIONS:
            correction_prompt = review.get("correction_prompt") or _review_fallback_correction_prompt(job, review)
            _set_job(
                job_id,
                status="correcting",
                reviewStatus="failed",
                correctionPrompt=correction_prompt,
            )
            previous_id = _get_field(current_interaction, "id")
            if not previous_id:
                raise RuntimeError("Gemini did not return an interaction id for video correction.")

            correction_interaction = _create_omni_video_interaction(
                client,
                correction_prompt,
                job["aspect_ratio"],
                previous_interaction_id=str(previous_id),
            )
            revision_count += 1
            corrected_path = GENERATED_DIR / f"{job_id}.revision{revision_count}.mp4"
            _download_interaction_video(client, correction_interaction, api_key, corrected_path)
            _set_job(
                job_id,
                status="reviewing",
                draftVideoUrl=f"/generated/{corrected_path.name}",
                revisionCount=revision_count,
                reviewStatus="running",
            )

            current_path = corrected_path
            current_interaction = correction_interaction
            review = _review_video_with_gemini(client, current_path, job)
            _append_review(job_id, review)
            _write_json(current_path.with_suffix(".review.json"), review)

        if not review.get("passed"):
            feedback = review.get("feedback") or review.get("summary") or ""
            details = "; ".join(review.get("issues") or review.get("missing_requirements") or [])
            raise RuntimeError(
                "Gemini review failed. Final video was not published. "
                f"{feedback} {details}".strip()
            )

        if current_path != output_path:
            shutil.copyfile(current_path, output_path)

        _set_job(
            job_id,
            status="labeling",
            videoUrl=f"/generated/{output_path.name}",
            error=None,
            reviewStatus="passed",
            labelStatus="running",
            labelError=None,
            renderStatus="pending",
            revisionCount=revision_count,
        )

        try:
            label = _label_video(output_path, job["prompt"])
            _set_job(
                job_id,
                status="rendering",
                labelStatus="completed",
                label=label,
                labelError=None,
                renderStatus="running",
            )
            try:
                labeled_video_url = _render_labeled_video(output_path, label)
                _set_job(
                    job_id,
                    status="completed",
                    labeledVideoUrl=labeled_video_url,
                    renderStatus="completed",
                    renderError=None,
                )
            except Exception as render_exc:
                _set_job(
                    job_id,
                    status="completed",
                    labeledVideoUrl=None,
                    renderStatus="failed",
                    renderError=str(render_exc),
                )
        except Exception as label_exc:
            _set_job(
                job_id,
                status="completed",
                labelStatus="failed",
                label=None,
                labelError=str(label_exc),
                renderStatus="skipped",
            )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            error=str(exc),
            videoUrl=None,
            reviewStatus="failed",
            labelStatus="failed",
            renderStatus="skipped",
        )


def _job_response(job_id: str) -> VideoJobResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Video job not found.")
        return VideoJobResponse(id=job_id, **job)


def _batch_status(job_responses: list[VideoJobResponse]) -> BatchStatus:
    statuses = [job.status for job in job_responses]
    active_statuses = {"queued", "generating", "reviewing", "correcting", "labeling", "rendering"}
    if not statuses:
        return "queued"
    if any(status in active_statuses for status in statuses):
        return "queued" if all(status == "queued" for status in statuses) else "running"
    if all(job.status == "completed" and job.labelStatus == "completed" for job in job_responses):
        return "completed"
    if all(status == "failed" for status in statuses):
        return "failed"
    if all(status in ("completed", "failed") for status in statuses):
        return "partial"
    return "queued"


def _batch_response(batch_id: str) -> VideoBatchResponse:
    with jobs_lock:
        batch = batches.get(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Video batch not found.")

        job_responses = [
            VideoJobResponse(id=job_id, **jobs[job_id])
            for job_id in batch["jobIds"]
            if job_id in jobs
        ]

    completed = sum(1 for job in job_responses if job.status == "completed" and job.labelStatus == "completed")
    failed = sum(1 for job in job_responses if job.status == "failed" or job.labelStatus == "failed")
    return VideoBatchResponse(
        id=batch_id,
        status=_batch_status(job_responses),
        prompt=batch["prompt"],
        aspect_ratio=batch["aspect_ratio"],
        count=batch["count"],
        completed=completed,
        failed=failed,
        jobs=job_responses,
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "hasGeminiKey": bool(get_gemini_api_key()),
        "hasCrusoeKey": bool(get_crusoe_api_key()),
        "model": MODEL_NAME,
        "batchSize": DEFAULT_VIEW_BATCH_SIZE,
        "cameraVariants": [variant["name"] for variant in CAMERA_VARIANTS],
        "reviewModel": get_gemini_review_model(),
        "reviewPipeline": REVIEW_PIPELINE_VERSION,
        "maxReviewRevisions": MAX_REVIEW_REVISIONS,
        "labelModel": CRUSOE_MODEL_NAME,
        "annotationPipeline": ANNOTATION_PIPELINE_VERSION,
    }


@app.get("/api/crusoe/diagnostics")
def crusoe_diagnostics() -> dict[str, Any]:
    return _check_crusoe_auth()


@app.post("/api/annotations/test-latest")
def test_latest_annotation() -> dict[str, Any]:
    videos = sorted(GENERATED_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not videos:
        raise HTTPException(status_code=404, detail="No generated MP4 found.")

    video_path = videos[0]
    label = _label_video(
        video_path,
        "Warehouse safety scene. Detect visible objects and hazard zones for VLM training.",
    )
    return {
        "video": video_path.name,
        "annotationPipeline": ANNOTATION_PIPELINE_VERSION,
        "label": label,
        "zoneCount": _annotation_count(label),
    }


@app.post("/api/videos", response_model=VideoBatchResponse, status_code=202)
def create_video(request: VideoCreateRequest) -> VideoBatchResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")
    if not get_gemini_api_key():
        raise HTTPException(
            status_code=500,
            detail="Missing Gemini API key. Add API_KEY_GEMINI to .env in the repo or parent folder.",
        )
    if not get_crusoe_api_key():
        raise HTTPException(
            status_code=500,
            detail="Missing Crusoe API key. Add API_KEY_CRUSOE to .env in the repo or parent folder.",
        )

    batch_id = uuid.uuid4().hex
    requested_count = min(request.count, DEFAULT_VIEW_BATCH_SIZE)
    selected_variants = CAMERA_VARIANTS[:requested_count]
    job_ids = [uuid.uuid4().hex for _ in selected_variants]

    with jobs_lock:
        batches[batch_id] = {
            "jobIds": job_ids,
            "prompt": prompt,
            "aspect_ratio": request.aspect_ratio,
            "count": requested_count,
        }

        for index, (job_id, camera_variant) in enumerate(zip(job_ids, selected_variants), start=1):
            final_prompt = _build_industrial_safety_prompt(prompt, camera_variant, request.aspect_ratio)
            jobs[job_id] = {
                "batchId": batch_id,
                "index": index,
                "status": "queued",
                "prompt": final_prompt,
                "basePrompt": prompt,
                "cameraVariant": {
                    "name": camera_variant["name"],
                    "title": camera_variant["title"],
                    "prompt": camera_variant["prompt"],
                },
                "aspect_ratio": request.aspect_ratio,
                "draftVideoUrl": None,
                "videoUrl": None,
                "error": None,
                "reviewStatus": "pending",
                "review": None,
                "reviewHistory": [],
                "revisionCount": 0,
                "correctionPrompt": None,
                "labelStatus": "pending",
                "label": None,
                "labelError": None,
                "labeledVideoUrl": None,
                "renderStatus": "pending",
                "renderError": None,
            }

    for job_id in job_ids:
        executor.submit(_generate_video, job_id)

    return _batch_response(batch_id)


@app.get("/api/videos/{job_id}", response_model=VideoJobResponse)
def get_video(job_id: str) -> VideoJobResponse:
    return _job_response(job_id)


@app.get("/api/batches/{batch_id}", response_model=VideoBatchResponse)
def get_batch(batch_id: str) -> VideoBatchResponse:
    return _batch_response(batch_id)
