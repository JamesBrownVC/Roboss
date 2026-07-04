import base64
import json
import os
import re
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
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 15 * 60
DEFAULT_VIEW_BATCH_SIZE = 6
BATCH_WORKERS = DEFAULT_VIEW_BATCH_SIZE
LABEL_FRAME_COUNT = 4
LABEL_FRAME_WIDTH = 512
MAX_LABEL_FRAME_PAYLOAD_BYTES = 2 * 1024 * 1024

JobStatus = Literal["queued", "running", "labeling", "completed", "failed"]
LabelStatus = Literal["pending", "running", "completed", "failed"]
BatchStatus = Literal["queued", "running", "completed", "failed", "partial"]

CAMERA_VARIANTS = [
    {
        "name": "wide_front",
        "title": "Wide front safety view",
        "prompt": (
            "Camera variant 1 of 6: wide frontal safety-training view from the end of the aisle. "
            "Show the full rack bay, pallet, floor markings, surrounding boxes, and the complete before/after context."
        ),
    },
    {
        "name": "close_load_detail",
        "title": "Close load detail",
        "prompt": (
            "Camera variant 2 of 6: close detail shot focused on the unstable pallet load. "
            "Show shrink wrap tension, cardboard labels, box edges, straps, wood grain, cracks, and impact details."
        ),
    },
    {
        "name": "high_overhead",
        "title": "High overhead aisle view",
        "prompt": (
            "Camera variant 3 of 6: high overhead or mezzanine-style view looking down the aisle. "
            "Make the hazard zone, pallet footprint, floor markings, debris spread, and rack positions easy to annotate."
        ),
    },
    {
        "name": "low_floor",
        "title": "Low floor impact view",
        "prompt": (
            "Camera variant 4 of 6: low floor-level view near the pallet impact area. "
            "Emphasize falling boxes, crushed cardboard, torn wrap, scattered debris, and floor contact."
        ),
    },
    {
        "name": "side_aisle",
        "title": "Side aisle profile",
        "prompt": (
            "Camera variant 5 of 6: side profile view along the rack aisle. "
            "Show the pallet tilt, rack uprights, empty rack space, boxes sliding, and depth of the aisle."
        ),
    },
    {
        "name": "after_inspection",
        "title": "Aftermath inspection",
        "prompt": (
            "Camera variant 6 of 6: post-incident inspection view after the fall. "
            "Show collapsed pallet, damaged boxes, torn plastic wrap, debris field, scrape marks, and empty rack location."
        ),
    },
]

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
    videoUrl: str | None = None
    error: str | None = None
    labelStatus: LabelStatus = "pending"
    label: dict[str, Any] | None = None
    labelError: str | None = None


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
    return str(name or state or "").upper()


def _extract_output_video(interaction: Any) -> Any:
    output_video = _get_field(interaction, "output_video")
    if output_video:
        return output_video

    for step in _get_field(interaction, "steps", []) or []:
        for item in _get_field(step, "content", []) or []:
            if _get_field(item, "type") == "video":
                return item

    return None


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


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text or text.lower() in {"none", "null"}:
        raise RuntimeError("Crusoe returned an empty label response.")

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

    raise RuntimeError("Crusoe returned text that could not be parsed as annotation JSON.")


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


def _generate_video(job_id: str) -> None:
    with jobs_lock:
        job = dict(jobs[job_id])

    _set_job(job_id, status="running", error=None)

    try:
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing Gemini API key. Add API_KEY_GEMINI to .env in the repo or parent folder."
            )

        os.environ.setdefault("GEMINI_API_KEY", api_key)

        from google import genai

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=job["prompt"],
            response_format={
                "type": "video",
                "delivery": "uri",
                "aspect_ratio": job["aspect_ratio"],
            },
            generation_config={
                "video_config": {
                    "task": "text_to_video",
                },
            },
        )

        output_video = _extract_output_video(interaction)
        video_bytes = _download_video_bytes(client, output_video, api_key)

        file_name = f"{job_id}.mp4"
        output_path = GENERATED_DIR / file_name
        output_path.write_bytes(video_bytes)

        _set_job(
            job_id,
            status="labeling",
            videoUrl=f"/generated/{file_name}",
            error=None,
            labelStatus="running",
            labelError=None,
        )

        try:
            label = _label_video(output_path, job["prompt"])
            _set_job(
                job_id,
                status="completed",
                labelStatus="completed",
                label=label,
                labelError=None,
            )
        except Exception as label_exc:
            _set_job(
                job_id,
                status="completed",
                labelStatus="failed",
                label=None,
                labelError=str(label_exc),
            )
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc), videoUrl=None, labelStatus="failed")


def _job_response(job_id: str) -> VideoJobResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Video job not found.")
        return VideoJobResponse(id=job_id, **job)


def _batch_status(job_responses: list[VideoJobResponse]) -> BatchStatus:
    statuses = [job.status for job in job_responses]
    if not statuses:
        return "queued"
    if any(status in ("queued", "running", "labeling") for status in statuses):
        return "running" if any(status in ("running", "labeling") for status in statuses) else "queued"
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
            final_prompt = (
                f"{prompt}\n\n"
                f"{camera_variant['prompt']}\n\n"
                "Keep the same core scene and event as the user prompt, but make this camera angle visually distinct "
                "from the other batch variants. Prioritize clear object visibility for VLM annotation: pallets, boxes, "
                "rack posts, shrink wrap, floor markings, debris, damaged boxes, and hazard zones."
            )
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
                "videoUrl": None,
                "error": None,
                "labelStatus": "pending",
                "label": None,
                "labelError": None,
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
