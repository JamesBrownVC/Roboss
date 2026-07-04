"""Agent 3 — Scenario Variation Planner: contract -> N controlled variations.

Also hosts the repair call used by the validator loop: invalid scenarios are
sent back to the model together with the deterministic error list.
"""

from __future__ import annotations

import json

from .config import AgentConfig
from .llm import generate_json, text_part
from .schemas import SCENARIOS_SCHEMA

SYSTEM = """\
You are a scenario variation planner. You receive a world contract (an
immutable canvas) and produce N scenario variations INSIDE that canvas.

Hard rules:
- Every scenario references ALL locked entity ids in referenced_entities.
- Never invent new main entities, never change appearance, materials,
  layout or style — those are locked.
- Vary only: camera angle (from the allowed list), event type (from the
  allowed list), event timing, risk severity, small position shifts within
  the allowed deltas, and the action timeline.
- If any entity moves from its default position before the action starts,
  include it in position_deltas with dx/dy inside allowed_position_deltas.
  Use zero or omit position_deltas for locked static objects.
- Scenarios must be meaningfully different from each other (different
  events, angles, severities, timings) while staying in the same world.
- action_timeline: 3-6 ordered steps, t_start < t_end, no gaps in logic,
  actors are locked entity ids only, all within the camera duration.
- The event trigger time must lie inside the allowed timing window and the
  timeline must show the event happening at that time.
- expected_labels: 3-6 short snake_case labels a downstream dataset
  annotator should be able to extract from the finished video.
- scenario_id: sc_<number>_<short_event_slug>.
Every scenario must be physically possible for a real robot and human."""


def _contract_text(contract: dict, count: int) -> str:
    return (f"World contract (immutable):\n"
            f"{json.dumps(contract['world_contract'], indent=2)}\n\n"
            f"Variation policy:\n"
            f"{json.dumps(contract['variation_policy'], indent=2)}\n\n"
            f"Generate exactly {count} scenarios.")


def plan_scenarios(contract: dict, count: int, cfg: AgentConfig) -> list[dict]:
    data = generate_json(
        model=cfg.text_model,
        system=SYSTEM,
        user_parts=[text_part(_contract_text(contract, count))],
        schema=SCENARIOS_SCHEMA,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.variation_temperature,
    )
    return data["scenarios"]


def repair_scenarios(contract: dict, invalid: list[dict],
                     errors: dict[str, list[str]],
                     cfg: AgentConfig) -> list[dict]:
    """Ask the model to fix scenarios that failed deterministic validation."""
    error_text = json.dumps(errors, indent=2)
    data = generate_json(
        model=cfg.text_model,
        system=SYSTEM,
        user_parts=[text_part(
            _contract_text(contract, len(invalid))
            + "\n\nThe following scenarios FAILED validation. Fix each one "
              "so it satisfies every rule, keeping its scenario_id and "
              "overall idea:\n"
            + json.dumps(invalid, indent=2)
            + "\n\nValidation errors per scenario_id:\n" + error_text)],
        schema=SCENARIOS_SCHEMA,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.plan_temperature,  # precision mode for repairs
    )
    return data["scenarios"]
