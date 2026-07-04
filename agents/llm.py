"""Thin Gemini wrapper: schema-constrained JSON calls and image calls.

Mirrors the client pattern already used by verifier/gate2.py (google-genai,
GEMINI_API_KEY from the environment).
"""

from __future__ import annotations

import json
import base64

from env_loader import load_dotenv


class AgentError(RuntimeError):
    """Raised when a pipeline stage cannot produce its output."""


def _client():
    from google import genai
    load_dotenv()
    try:
        return genai.Client()
    # TypeError: the SDK raises it when no credentials can be resolved.
    except (TypeError, ValueError) as e:
        raise AgentError(f"Cannot create Gemini client: {e}. "
                         f"Set GEMINI_API_KEY.") from e


def generate_json(model: str, system: str, user_parts: list,
                  schema: dict, max_output_tokens: int,
                  temperature: float) -> dict:
    """One schema-constrained call; returns the parsed JSON dict."""
    from google.genai import errors, types

    try:
        client = _client()
        response = client.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=user_parts),
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
    except (errors.APIError, RuntimeError, TypeError, ValueError) as e:
        raise AgentError(f"{type(e).__name__}: {e}") from e

    if response.text is None:
        raise AgentError("empty model response")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        raise AgentError(f"model returned invalid JSON: {e}") from e


def generate_image(model: str, parts: list) -> bytes | None:
    """One image-generation call; returns PNG/JPEG bytes or None."""
    from google.genai import errors

    try:
        client = _client()
        response = client.interactions.create(
            model=model,
            input=_interaction_input(parts),
        )
    except (errors.APIError, RuntimeError, TypeError, ValueError) as e:
        raise AgentError(f"{type(e).__name__}: {e}") from e

    image = getattr(response, "output_image", None)
    if image is not None and getattr(image, "data", None):
        return base64.b64decode(image.data)
    return None


def text_part(text: str):
    from google.genai import types
    return types.Part.from_text(text=text)


def image_part(data: bytes, mime_type: str = "image/png"):
    from google.genai import types
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def _interaction_input(parts: list) -> str | list[dict]:
    """Convert genai Part objects to Interactions API input blocks."""
    converted: list[dict] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text is not None:
            converted.append({"type": "text", "text": text})
            continue
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            converted.append({
                "type": "image",
                "data": base64.b64encode(inline.data).decode("ascii"),
                "mime_type": inline.mime_type or "image/png",
            })
    if len(converted) == 1 and converted[0]["type"] == "text":
        return converted[0]["text"]
    return converted
