"""LLM router for the labeling agent.

Roles:
  orchestrator - runs the agent loop; multimodal preferred
                 (Crusoe: nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B)
  fast         - critic / quick text reasoning (Crusoe: moonshotai/Kimi-K2.6)
  vision       - frame Q&A (Crusoe omni, else Gemini flash)

Provider selection (OpenAI-compatible endpoints):
  - Crusoe Managed Inference when CRUSOE_API_KEY is set, or when
    NVIDIA_API_KEY holds a non-"nvapi-" value (the repo .env historically
    stored the Crusoe key under that name).
  - NVIDIA NIM when NVIDIA_API_KEY starts with "nvapi-".
  - Gemini as the fallback chain for every call (auth/rate/server errors).

Crusoe quirks verified live: the endpoint is behind Cloudflare and returns
HTTP 403 (error 1010) for urllib's default User-Agent, so a custom UA is
mandatory; reasoning models spend tokens on a separate `reasoning` field, so
budget max_tokens generously.
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ..syngen import gemini

USER_AGENT = "v2r-labeler/0.1"

CRUSOE_BASE = "https://api.inference.crusoecloud.com/v1"
CRUSOE_MODELS = {
    "orchestrator": "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
    "fast": "moonshotai/Kimi-K2.6",
    "vision": "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
}
CRUSOE_MULTIMODAL = {"nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B"}

NIM_BASE = "https://integrate.api.nvidia.com/v1"
NIM_MODELS = {
    "orchestrator": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "fast": "moonshotai/kimi-k2.6",
    "vision": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}
NIM_MULTIMODAL = {"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                  "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                  "nvidia/nemotron-nano-12b-v2-vl"}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    pass


def _ssl_ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _env_lookup(names: list[str], root: Optional[Path] = None) -> Optional[str]:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    start = Path(root) if root else Path(__file__).resolve()
    for cand in [start, *start.parents]:
        env = cand / ".env"
        if env.is_file():
            for name in names:
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.startswith(f"{name}="):
                        val = line.split("=", 1)[1].strip()
                        if val:
                            return val
    return None


def openai_compat_provider(root: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
    """Return (provider_name, api_key) for the OpenAI-compatible provider.

    CRUSOE_API_KEY wins; otherwise NVIDIA_API_KEY is classified by prefix:
    real NIM keys start with "nvapi-", anything else is a Crusoe key that was
    historically stored under the wrong name in this repo's .env.
    """
    key = _env_lookup(["CRUSOE_API_KEY"], root)
    if key:
        return "crusoe", key
    key = _env_lookup(["NVIDIA_API_KEY"], root)
    if key:
        return ("nim", key) if key.startswith("nvapi-") else ("crusoe", key)
    return None, None


def nvidia_api_key(root: Optional[Path] = None) -> Optional[str]:
    """Backward-compatible accessor (NIM-classified keys only)."""
    provider, key = openai_compat_provider(root)
    return key if provider == "nim" else None


class LLMRouter:
    """chat() with automatic Crusoe/NIM -> Gemini fallback and multimodal handling."""

    def __init__(self, root: Path, log=print):
        self.root = Path(root)
        self.log = log
        self.oa_provider, self.oa_key = openai_compat_provider(self.root)
        if self.oa_provider == "crusoe":
            self.oa_base, self.oa_models, self.oa_multimodal = (
                CRUSOE_BASE, CRUSOE_MODELS, CRUSOE_MULTIMODAL)
        else:
            self.oa_base, self.oa_models, self.oa_multimodal = (
                NIM_BASE, NIM_MODELS, NIM_MULTIMODAL)
        self.gemini_key = gemini.get_api_key(self.root)
        self.oa_ok: Optional[bool] = None  # unknown until first call
        self.stats: list[dict] = []

    # ------------------------------------------------------------------
    def available(self) -> bool:
        return bool(self.oa_key or self.gemini_key)

    def active_provider(self, role: str = "orchestrator") -> str:
        if self.oa_key and self.oa_ok is not False:
            return f"{self.oa_provider}:{self.oa_models[role]}"
        if self.gemini_key:
            return f"gemini:{gemini.DEFAULT_VISION_MODEL}"
        return "none"

    def orchestrator_is_multimodal(self) -> bool:
        if self.oa_key and self.oa_ok is not False:
            return self.oa_models["orchestrator"] in self.oa_multimodal
        return bool(self.gemini_key)  # gemini flash is multimodal

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        role: str = "orchestrator",
        images: Optional[list[bytes]] = None,
        json_schema: Optional[dict] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        force_json: bool = False,
    ) -> str:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}].
        `images` (jpeg bytes) attach to the LAST user message. `force_json`
        requests provider-native JSON mode (no schema, just valid JSON)."""
        errors = []
        if self.oa_key and self.oa_ok is not False:
            try:
                out = self._openai_chat(messages, role, images, json_schema,
                                        max_tokens, temperature, force_json)
                self.oa_ok = True
                return out
            except LLMError as e:
                errors.append(f"{self.oa_provider}: {e}")
                if "401" in str(e):
                    self.oa_ok = False  # bad key: stop trying this run
                self.log(f"[llm] {self.oa_provider} failed ({str(e)[:120]}); "
                         "falling back to Gemini")
        if self.gemini_key:
            try:
                return self._gemini_chat(messages, images, json_schema, temperature)
            except Exception as e:  # noqa: BLE001
                errors.append(f"gemini: {e}")
        raise LLMError(" | ".join(errors) or "no LLM provider configured")

    # ------------------------------------------------------------------
    def _openai_chat(self, messages, role, images, json_schema, max_tokens,
                     temperature, force_json=False) -> str:
        model = self.oa_models[role]
        msgs = [dict(m) for m in messages]
        if images:
            if model not in self.oa_multimodal:
                raise LLMError(f"model {model} is text-only but images were attached")
            # attach to last user message as OpenAI content array
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "user":
                    content = [{"type": "text", "text": msgs[i]["content"]}]
                    for jpg in images:
                        b64 = base64.b64encode(jpg).decode("ascii")
                        content.append({"type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    msgs[i]["content"] = content
                    break
        payload = {"model": model, "messages": msgs,
                   "max_tokens": max_tokens, "temperature": temperature}
        # Crusoe supports OpenAI structured outputs (verified live on both
        # Nemotron omni and Kimi): schema-constrained > json mode > prompt
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema}}
        elif force_json:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.time()
        req = urllib.request.Request(
            f"{self.oa_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.oa_key}",
                     "Content-Type": "application/json",
                     "User-Agent": USER_AGENT})  # Cloudflare 403s urllib's default UA
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180, context=_ssl_ctx()) as r:
                    data = json.loads(r.read().decode("utf-8"))
                content = data["choices"][0]["message"].get("content")
                if isinstance(content, list):  # some models echo content-part arrays
                    content = "".join(p.get("text", "") for p in content
                                      if isinstance(p, dict))
                text = _THINK_RE.sub("", content or "").strip()
                self.stats.append({"provider": self.oa_provider, "model": model,
                                   "role": role, "s": round(time.time() - t0, 1)})
                if not text:
                    raise LLMError("empty response after stripping reasoning "
                                   "(reasoning may have exhausted max_tokens)")
                return text
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                last = LLMError(f"HTTP {e.code}: {body}")
                if e.code in (401, 403):
                    raise last
                if e.code == 429 and attempt < 2:
                    time.sleep(4.0 * (attempt + 1))  # per-IP burst limits
                    continue
                if e.code >= 500 and attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise last
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = LLMError(f"network: {e}")
                if attempt < 2:
                    time.sleep(2.0)
                    continue
                raise last
        raise last or LLMError(f"{self.oa_provider}: retries exhausted")

    # ------------------------------------------------------------------
    def _gemini_chat(self, messages, images, json_schema, temperature) -> str:
        # flatten the transcript into one prompt (gemini REST is stateless here)
        lines = []
        for m in messages:
            tag = m["role"].upper()
            lines.append(f"[{tag}]\n{m['content']}")
        parts: list[dict] = []
        for jpg in images or []:
            parts.append(gemini.image_part(jpg))
        parts.append({"text": "\n\n".join(lines)})
        t0 = time.time()
        out = gemini.generate_content(
            parts, model=gemini.DEFAULT_VISION_MODEL, temperature=temperature,
            response_schema=json_schema, api_key=self.gemini_key)
        self.stats.append({"provider": "gemini", "model": gemini.DEFAULT_VISION_MODEL,
                           "s": round(time.time() - t0, 1)})
        return out


def parse_json_reply(text: str):
    """Tolerant JSON extraction: fences, leading prose, trailing junk,
    python-repr'd content arrays, and unbalanced (truncated) tails."""
    s = text.strip()
    # some models echo the OpenAI content-part array as literal text:
    # [{'type': 'text', 'text': '{"thought": ...}'}]
    if s.startswith("[{") and "'text'" in s[:40] or s.startswith('[{"type"'):
        try:
            import ast

            parts = ast.literal_eval(s)
            if isinstance(parts, list):
                s = "".join(p.get("text", "") for p in parts
                            if isinstance(p, dict)).strip()
        except (ValueError, SyntaxError):
            pass
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        if start < 0:
            raise
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start:i + 1])
        raise
