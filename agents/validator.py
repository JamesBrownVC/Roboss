"""Agent 4 — deterministic contract validator (no LLM).

The LLM proposes, this module disposes: every scenario is checked against
the world contract and variation policy with plain Python. Violations are
returned as human-readable strings that the repair loop feeds back to the
model.
"""

from __future__ import annotations

from .config import RISK_SEVERITIES


ROBOT_RESPONSE_KEYWORDS = [
    "react", "respond", "stop", "halt", "slow", "yield", "avoid", "reroute",
    "back", "retreat", "stabilize", "secure", "assist", "wait", "brake",
    "pause", "move_away", "sidestep",
]


def _range_for(policy: dict, entity_id: str) -> dict | None:
    for item in policy.get("allowed_position_deltas", []):
        if item.get("entity_id") == entity_id:
            return item
    return None


def _inside(value: float, bounds: list | None) -> bool:
    if not bounds or len(bounds) != 2:
        return abs(value) <= 1e-9
    lo, hi = sorted(float(v) for v in bounds)
    return lo <= value <= hi


def _robot_entity_ids(entities: list[dict]) -> set[str]:
    return {
        e["id"]
        for e in entities
        if "robot" in str(e.get("id", "")).lower()
        or "robot" in str(e.get("type", "")).lower()
    }


def _norm(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ")


def validate_contract(contract: dict) -> list[str]:
    """Contract-level invariants, checked once per pipeline run (they can
    never be fixed by regenerating scenarios, so they don't belong in the
    per-scenario repair loop)."""
    errors: list[str] = []
    entity_ids = {e["id"] for e in contract["world_contract"]["locked_entities"]}
    registry_ids = {a.get("entity_id") for a in contract.get(
        "object_registry", [])}
    if registry_ids != entity_ids:
        errors.append("object_registry must contain exactly the locked "
                      f"entities: expected {sorted(entity_ids)}, got "
                      f"{sorted(registry_ids)}")
    if not contract.get("scene_registry", {}).get("scene_id"):
        errors.append("scene_registry.scene_id is required")
    if not contract.get("identity_checks"):
        errors.append("identity_checks must not be empty")
    if not _robot_entity_ids(contract["world_contract"]["locked_entities"]):
        errors.append("world_contract.locked_entities must include at least "
                      "one robot entity because every scenario requires a "
                      "robot response")
    return errors


def validate_scenario(contract: dict, scenario: dict) -> list[str]:
    """Returns a list of violation messages; empty list = valid."""
    errors: list[str] = []
    wc = contract["world_contract"]
    policy = contract["variation_policy"]
    entity_ids = {e["id"] for e in wc["locked_entities"]}

    # identity contract: every locked entity must be referenced
    referenced = set(scenario.get("referenced_entities", []))
    missing = entity_ids - referenced
    if missing:
        errors.append(f"referenced_entities is missing locked entities: "
                      f"{sorted(missing)}")
    unknown = referenced - entity_ids
    if unknown:
        errors.append(f"referenced_entities contains unknown ids: "
                      f"{sorted(unknown)} (no new main entities allowed)")

    # camera contract
    camera = scenario.get("camera", {})
    angle = camera.get("angle")
    if angle not in policy["allowed_camera_angles"]:
        errors.append(f"camera.angle '{angle}' is not in allowed_camera_angles "
                      f"{policy['allowed_camera_angles']}")
    duration = float(camera.get("duration_seconds", 0))
    if duration <= 0:
        errors.append("camera.duration_seconds must be positive")

    # event contract
    event = scenario.get("event", {})
    if event.get("type") not in policy["allowed_event_types"]:
        errors.append(f"event.type '{event.get('type')}' is not in "
                      f"allowed_event_types {policy['allowed_event_types']}")
    if event.get("risk_severity") not in RISK_SEVERITIES:
        errors.append(f"event.risk_severity '{event.get('risk_severity')}' "
                      f"must be one of {RISK_SEVERITIES}")
    trigger = float(event.get("trigger_time_seconds", -1))
    if not 0 <= trigger <= duration:
        errors.append(f"event.trigger_time_seconds {trigger} is outside the "
                      f"clip duration [0, {duration}]")
    timing = policy.get("event_timing_seconds")
    if timing and len(timing) == 2 and not timing[0] <= trigger <= timing[1]:
        errors.append(f"event.trigger_time_seconds {trigger} is outside the "
                      f"allowed timing window {timing}")

    # spatial contract: scenario deltas must be explicit and bounded
    for i, delta in enumerate(scenario.get("position_deltas", [])):
        eid = delta.get("entity_id")
        if eid not in entity_ids:
            errors.append(f"position_deltas[{i}]: entity_id '{eid}' is not "
                          "a locked entity")
            continue
        allowed = _range_for(policy, eid)
        dx = float(delta.get("dx", 0.0))
        dy = float(delta.get("dy", 0.0))
        if not allowed:
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                errors.append(f"position_deltas[{i}]: {eid} may not move "
                              "because no allowed delta is defined")
            continue
        if not _inside(dx, allowed.get("dx")):
            errors.append(f"position_deltas[{i}]: dx {dx} is outside "
                          f"{allowed.get('dx')} for {eid}")
        if not _inside(dy, allowed.get("dy")):
            errors.append(f"position_deltas[{i}]: dy {dy} is outside "
                          f"{allowed.get('dy')} for {eid}")

    # action contract: ordered timeline, known actors, inside the clip
    timeline = scenario.get("action_timeline", [])
    if not timeline:
        errors.append("action_timeline is empty")
    prev_start = -1.0
    for i, step in enumerate(timeline):
        t0, t1 = float(step.get("t_start", -1)), float(step.get("t_end", -1))
        if t0 < 0 or t1 <= t0:
            errors.append(f"action_timeline[{i}]: t_start/t_end invalid "
                          f"({t0}..{t1})")
        if t1 > duration + 1e-6:
            errors.append(f"action_timeline[{i}]: ends at {t1}s, beyond the "
                          f"{duration}s clip")
        if t0 < prev_start:
            errors.append(f"action_timeline[{i}]: steps are not ordered by "
                          f"t_start")
        prev_start = max(prev_start, t0)
        if step.get("actor") not in entity_ids:
            errors.append(f"action_timeline[{i}]: actor "
                          f"'{step.get('actor')}' is not a locked entity")

    # robot response contract: the scenario must be a situation that causes
    # an observable robot reaction after the trigger.
    expected_response = str(scenario.get("expected_robot_response", "")).strip()
    robot_ids = _robot_entity_ids(wc["locked_entities"])
    if not expected_response:
        errors.append("expected_robot_response is required")
    if not robot_ids:
        errors.append("no robot entity is available to react to the event")
    response_text = _norm(expected_response)
    response_terms = {w for w in response_text.split() if len(w) >= 3}
    has_robot_response = False
    for step in timeline:
        if step.get("actor") not in robot_ids:
            continue
        if float(step.get("t_start", -1)) + 1e-6 < trigger:
            continue
        action_text = _norm(str(step.get("action", "")))
        if (
            any(k in action_text for k in ROBOT_RESPONSE_KEYWORDS)
            or any(term in action_text for term in response_terms)
        ):
            has_robot_response = True
            break
    if not has_robot_response:
        errors.append("action_timeline must include an observable robot "
                      "response step after the event trigger matching "
                      "expected_robot_response")

    # label contract
    if not scenario.get("expected_labels"):
        errors.append("expected_labels is empty")
    if not str(scenario.get("scenario_id", "")).strip():
        errors.append("scenario_id is empty")

    return errors


def validate_all(contract: dict,
                 scenarios: list[dict]) -> list[tuple[str, list[str]]]:
    """One (scenario_id, errors) pair per input scenario, index-aligned.

    Duplicate ids only invalidate the *later* copies — the first occurrence
    stays valid, so a legitimate scenario is never lost to a duplicate.
    """
    results: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for i, sc in enumerate(scenarios):
        sid = str(sc.get("scenario_id") or f"index_{i}")
        errs = validate_scenario(contract, sc)
        if sid in seen:
            errs.append(f"duplicate scenario_id '{sid}' — keep the first "
                        f"occurrence, give this one a new id")
        seen.add(sid)
        results.append((sid, errs))
    return results
