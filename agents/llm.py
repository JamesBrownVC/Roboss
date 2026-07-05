"""Thin Gemini wrapper: schema-constrained JSON calls and image calls.

Mirrors the client pattern already used by verifier/gate2.py (google-genai,
GEMINI_API_KEY from the environment), plus:
- one cached client per process;
- retry with backoff on retryable API errors (429/5xx);
- thinking_level instead of temperature (Gemini 3.5 deprecates sampling
  parameters in favor of thinking levels).
"""

from __future__ import annotations

import json
import time

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3

_client_instance = None


class AgentError(RuntimeError):
    """Raised when a pipeline stage cannot produce its output."""


def _client():
    global _client_instance
    if _client_instance is None:
        from google import genai

        from env_loader import load_dotenv

        load_dotenv()  # same .env the verifier's gate 2 reads
        try:
            _client_instance = genai.Client()
        # TypeError: the SDK raises it when no credentials can be resolved.
        except (TypeError, ValueError) as e:
            raise AgentError(f"Cannot create Gemini client: {e}. "
                             f"Set GEMINI_API_KEY.") from e
    return _client_instance


def _call_with_retry(make_request):
    """Run an API call, retrying transient failures with backoff."""
    from google.genai import errors

    last_error = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return make_request()
        except errors.APIError as e:
            last_error = e
            if getattr(e, "code", None) in _RETRYABLE_CODES \
                    and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2.0 * (2 ** attempt))  # 2s, 4s
                continue
            raise AgentError(f"{type(e).__name__}: {e}") from e
        except (TypeError, ValueError) as e:
            raise AgentError(f"{type(e).__name__}: {e}") from e
    raise AgentError(f"{type(last_error).__name__}: {last_error}")


def generate_json(model: str, system: str, user_parts: list,
                  schema: dict, max_output_tokens: int,
                  thinking_level: str | None = None) -> dict:
    """One schema-constrained call; returns the parsed JSON dict."""
    from google.genai import types

    config_kwargs = dict(
        system_instruction=system,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
    )
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level)

    response = _call_with_retry(lambda: _client().models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=user_parts),
        config=types.GenerateContentConfig(**config_kwargs),
    ))

    if response.text is None:
        raise AgentError("empty model response")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        raise AgentError(f"model returned invalid JSON: {e}") from e


def generate_image(model: str, parts: list) -> bytes | None:
    """One image-generation call; returns PNG/JPEG bytes or None."""
    from google.genai import types

    response = _call_with_retry(lambda: _client().models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    ))

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data is not None and part.inline_data.data:
                return part.inline_data.data
    return None


def text_part(text: str):
    from google.genai import types
    return types.Part.from_text(text=text)


def image_part(data: bytes, mime_type: str = "image/png"):
    from google.genai import types
    return types.Part.from_bytes(data=data, mime_type=mime_type)
