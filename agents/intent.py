"""Agent 1 — Intent Parser: free-form user intention -> structured intent."""

from __future__ import annotations

from .config import AgentConfig
from .llm import generate_json, text_part
from .schemas import INTENT_SCHEMA

SYSTEM = """\
You parse a user's intention for synthetic robotics-safety video generation
into a structured brief. Extract what is stated; infer conservative defaults
for what is not: variation_count defaults to 5, duration_seconds to 6,
camera_style to "static surveillance-like". actors are the moving agents
(humans, robots); objects are the passive items (boxes, shelves, tools).
Keep names short and generic (e.g. "humanoid robot", "warehouse worker",
"cardboard box")."""


def parse_intent(intention: str, cfg: AgentConfig,
                 count_override: int | None = None) -> dict:
    intent = generate_json(
        model=cfg.text_model,
        system=SYSTEM,
        user_parts=[text_part(f"User intention:\n{intention}")],
        schema=INTENT_SCHEMA,
        max_output_tokens=cfg.max_output_tokens,
        thinking_level=cfg.intent_thinking_level,
    )
    if count_override:
        intent["variation_count"] = count_override
    intent["variation_count"] = max(1, min(int(intent.get(
        "variation_count", 5)), 20))
    intent["duration_seconds"] = float(intent.get("duration_seconds", 6.0))
    intent["raw_intention"] = intention
    return intent
