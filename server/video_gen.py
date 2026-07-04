"""Gemini Omni Flash video generation via google-genai interactions API."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from env_loader import load_dotenv

DEFAULT_VIDEO_MODEL = "gemini-omni-flash-preview"
POLL_INTERVAL_S = 5
MAX_POLL_ATTEMPTS = 60


class VideoGenError(RuntimeError):
    pass


def _client():
    from google import genai

    load_dotenv()
    try:
        return genai.Client()
    except (TypeError, ValueError) as exc:
        raise VideoGenError(
            f"Cannot create Gemini client: {exc}. Set GEMINI_API_KEY."
        ) from exc


def _build_input(prompt: str, start_frame: bytes | None):
    if not start_frame:
        return prompt
    return [
        {
            "type": "image",
            "data": base64.b64encode(start_frame).decode("ascii"),
            "mime_type": "image/png",
        },
        {"type": "text", "text": prompt},
    ]


def _extract_video(interaction):
    video = getattr(interaction, "output_video", None)
    if video is not None and (getattr(video, "data", None) or getattr(video, "uri", None)):
        return video

    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for part in getattr(step, "content", None) or []:
            if getattr(part, "type", None) == "video":
                return part

    raise VideoGenError("Gemini Omni returned no video")


def _state_name(file_info) -> str:
    state = getattr(file_info, "state", None)
    return getattr(state, "name", None) or str(state or "")


def _download_uri(client, uri: str) -> bytes:
    file_name = uri.split("/")[-1].split(":")[0]
    if file_name:
        for _ in range(MAX_POLL_ATTEMPTS):
            try:
                file_info = client.files.get(name=f"files/{file_name}")
            except Exception:
                break
            state = _state_name(file_info)
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise VideoGenError("Gemini Omni video processing failed")
            time.sleep(POLL_INTERVAL_S)

    try:
        downloaded = client.files.download(file=uri)
        if isinstance(downloaded, bytes):
            return downloaded
        data = getattr(downloaded, "data", None)
        if isinstance(data, bytes):
            return data
    except Exception as client_exc:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise VideoGenError("Cannot download Gemini Omni output: GEMINI_API_KEY missing") from client_exc
        request = Request(uri, headers={"x-goog-api-key": api_key})
        try:
            with urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as url_exc:
            raise VideoGenError(
                f"Could not download Gemini Omni output: {client_exc}; fallback failed: {url_exc}"
            ) from url_exc

    raise VideoGenError("Could not download Gemini Omni output")


def generate_video(
    *,
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
    start_frame: bytes | None = None,
    duration_seconds: float = 8.0,
    model: str = DEFAULT_VIDEO_MODEL,
    log=None,
) -> Path:
    """Generate an MP4 with Gemini Omni Flash."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = _client()

    if log:
        log(f"Gemini Omni: starting generation ({model}, {aspect_ratio})")

    try:
        interaction = client.interactions.create(
            model=model,
            input=_build_input(prompt, start_frame),
            response_format={"type": "video", "aspect_ratio": aspect_ratio},
        )
    except Exception as exc:
        raise VideoGenError(f"Gemini Omni request failed: {exc}") from exc

    video = _extract_video(interaction)
    data = getattr(video, "data", None)
    uri = getattr(video, "uri", None)

    if data:
        video_bytes = base64.b64decode(data)
    elif uri:
        if log:
            log("Gemini Omni: downloading video output")
        video_bytes = _download_uri(client, uri)
    else:
        raise VideoGenError("Gemini Omni returned an empty video payload")

    output_path.write_bytes(video_bytes)

    if log:
        log(f"Gemini Omni: saved {output_path.name}")
    return output_path
