"""Minimal Gemini REST client (urllib only, no SDK dependency).

The API key is read from the GEMINI_API_KEY environment variable; a `.env`
file at the repo root (or any parent of the v2r root) is loaded as fallback.
The key is never stored in source or in any artifact JSON.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_TEXT_MODEL = "gemini-flash-latest"
DEFAULT_VISION_MODEL = "gemini-flash-latest"
# Both video paths confirmed working with this key:
#   omni: POST /v1beta/interactions (model gemini-omni-flash-preview) — default
#   veo:  models/veo-*:predictLongRunning + operation polling
DEFAULT_OMNI_MODEL = "gemini-omni-flash-preview"
DEFAULT_VEO_MODEL = "veo-3.1-fast-generate-preview"
DEFAULT_VIDEO_MODEL = DEFAULT_OMNI_MODEL


class GeminiError(RuntimeError):
    pass


def _load_dotenv_key(start: Optional[Path] = None) -> Optional[str]:
    """Search `.env` files upward from `start` (or this file) for GEMINI_API_KEY."""
    cur = (start or Path(__file__)).resolve()
    candidates = [cur] if cur.is_dir() else []
    candidates += list(cur.parents)
    for cand in candidates:
        env_file = cand / ".env"
        if not env_file.is_file():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except OSError:
            continue
    return None


def get_api_key(root: Optional[Path] = None) -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    return _load_dotenv_key(root)


def have_api_key(root: Optional[Path] = None) -> bool:
    return bool(get_api_key(root))


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _ssl_context():
    """Explicit certifi-based SSL context.

    The conda env carries pip_system_certs, whose Windows trust-store
    injection fails ([ASN1: NOT_ENOUGH_DATA]) and breaks the DEFAULT ssl
    context for all urllib requests. Building our own context from certifi
    sidesteps the broken store.
    """
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _request(
    method: str,
    url: str,
    api_key: str,
    payload: Optional[dict] = None,
    timeout: float = 120.0,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise GeminiError(f"HTTP {e.code} on {url.split('?')[0]}: {body}") from e
    except urllib.error.URLError as e:
        raise GeminiError(f"network error: {e.reason}") from e


# ---------------------------------------------------------------------------
# generateContent (text / multimodal, optional JSON-schema output)
# ---------------------------------------------------------------------------


def generate_content(
    parts: list[dict],
    model: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.7,
    response_schema: Optional[dict] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> str:
    """Call models/{model}:generateContent; return the text of the first candidate.

    `parts` follow the REST shape, e.g. [{"text": ...}] or
    [{"inline_data": {"mime_type": "image/jpeg", "data": <b64>}}, {"text": ...}].
    When `response_schema` is given, JSON output mode is enabled.
    """
    key = api_key or get_api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set (env var or .env at repo root)")

    generation_config: dict[str, Any] = {"temperature": temperature}
    if response_schema is not None:
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = response_schema

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }
    url = f"{BASE_URL}/models/{model}:generateContent"

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            data = _request("POST", url, key, payload, timeout=timeout)
            candidates = data.get("candidates") or []
            if not candidates:
                raise GeminiError(f"no candidates in response: {json.dumps(data)[:300]}")
            parts_out = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts_out)
            if not text:
                raise GeminiError("empty text in first candidate")
            return text
        except GeminiError as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))
    raise GeminiError(f"generateContent failed after {max_retries + 1} attempts: {last_err}")


def extract_json(text: str) -> Any:
    """Parse model output as JSON, tolerating markdown code fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -3]
    return json.loads(s)


def image_part(jpeg_bytes: bytes) -> dict:
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(jpeg_bytes).decode("ascii"),
        }
    }


# ---------------------------------------------------------------------------
# Gemini Omni video generation (Interactions API)
# ---------------------------------------------------------------------------


def omni_generate_video(
    prompt: str,
    model: str = DEFAULT_OMNI_MODEL,
    aspect_ratio: str = "16:9",
    api_key: Optional[str] = None,
    timeout: float = 600.0,
) -> str:
    """Generate a video via POST /v1beta/interactions (blocking call, ~40s).

    Returns the download URI of the generated video. The backing file may
    still be PROCESSING; poll it with `poll_file_active` before downloading.
    Raises GeminiError on failure (including content-policy blocks, which the
    API surfaces as HTTP 400).
    """
    key = api_key or get_api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")
    payload = {
        "model": model,
        "input": prompt,
        "response_format": {
            "type": "video",
            "delivery": "uri",
            "aspect_ratio": aspect_ratio,
        },
    }
    data = _request("POST", f"{BASE_URL}/interactions", key, payload, timeout=timeout)
    if data.get("status") not in (None, "completed"):
        raise GeminiError(f"interaction status={data.get('status')}: {json.dumps(data)[:300]}")
    for step in data.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content", []):
            if content.get("type") == "video" and content.get("uri"):
                return content["uri"]
    raise GeminiError(f"no video uri in interaction response: {json.dumps(data)[:300]}")


def poll_file_active(
    uri: str,
    api_key: Optional[str] = None,
    poll_interval_s: float = 5.0,
    timeout_s: float = 300.0,
) -> None:
    """Poll files/{id} until state==ACTIVE (omni videos start as PROCESSING)."""
    key = api_key or get_api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")
    try:
        file_id = uri.split("/files/", 1)[1].split(":", 1)[0]
    except IndexError:
        raise GeminiError(f"cannot extract file id from uri: {uri}")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = _request("GET", f"{BASE_URL}/files/{file_id}", key)
        state = data.get("state")
        if state == "ACTIVE":
            return
        if state == "FAILED":
            raise GeminiError(f"file {file_id} entered FAILED state")
        time.sleep(poll_interval_s)
    raise GeminiError(f"file {file_id} not ACTIVE after {timeout_s:.0f}s")


# ---------------------------------------------------------------------------
# Veo video generation (predictLongRunning + operation polling + download)
# ---------------------------------------------------------------------------


def start_video_generation(
    prompt: str,
    model: str = DEFAULT_VEO_MODEL,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 4,
    api_key: Optional[str] = None,
) -> str:
    """Kick off Veo generation; return the operation name to poll."""
    key = api_key or get_api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"aspectRatio": aspect_ratio, "durationSeconds": duration_seconds},
    }
    url = f"{BASE_URL}/models/{model}:predictLongRunning"
    data = _request("POST", url, key, payload)
    name = data.get("name")
    if not name:
        raise GeminiError(f"no operation name returned: {json.dumps(data)[:300]}")
    return name


def poll_video_operation(
    operation_name: str,
    api_key: Optional[str] = None,
    poll_interval_s: float = 8.0,
    timeout_s: float = 600.0,
) -> str:
    """Poll until done; return the download URI of the first generated sample."""
    key = api_key or get_api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")
    url = f"{BASE_URL}/{operation_name}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = _request("GET", url, key)
        if data.get("done"):
            if "error" in data:
                raise GeminiError(f"video operation failed: {json.dumps(data['error'])[:300]}")
            samples = (
                data.get("response", {})
                .get("generateVideoResponse", {})
                .get("generatedSamples", [])
            )
            if not samples:
                raise GeminiError(f"operation done but no samples: {json.dumps(data)[:300]}")
            uri = samples[0].get("video", {}).get("uri")
            if not uri:
                raise GeminiError("no video uri in generated sample")
            return uri
        time.sleep(poll_interval_s)
    raise GeminiError(f"video operation timed out after {timeout_s:.0f}s: {operation_name}")


def download_file(uri: str, dest: Path, api_key: Optional[str] = None) -> Path:
    key = api_key or get_api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(uri, headers={"X-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.URLError as e:
        raise GeminiError(f"download failed: {e}") from e
    return dest
