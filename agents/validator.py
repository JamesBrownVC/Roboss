"""Agent 4 — deterministic contract validator (no LLM).

The LLM proposes, this module disposes: every scenario is checked against
the world contract and variation policy with plain Python. Violations are
returned as human-readable strings that the repair loop feeds back to the
model.
"""

from __future__ import annotations

from .config import RISK_SEVERITIES


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


def validate_scenario(contract: dict, scenario: dict) -> list[str]:
    """Returns a list of violation messages; empty list = valid."""
    errors: list[str] = []
    wc = contract["world_contract"]
    policy = contract["variation_policy"]
    entity_ids = {e["id"] for e in wc["locked_entities"]}
    registry_ids = {a.get("entity_id") for a in contract.get(
        "object_registry", [])}

    if registry_ids and registry_ids != entity_ids:
        errors.append("object_registry must contain exactly the locked "
                      f"entities: expected {sorted(entity_ids)}, got "
                      f"{sorted(registry_ids)}")
    scene = contract.get("scene_registry", {})
    if not scene.get("scene_id"):
        errors.append("scene_registry.scene_id is required")
    if not contract.get("identity_checks"):
        errors.append("identity_checks must not be empty")

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

    # label contract
    if not scenario.get("expected_labels"):
        errors.append("expected_labels is empty")
    if not str(scenario.get("scenario_id", "")).strip():
        errors.append("scenario_id is empty")

    return errors


def validate_all(contract: dict,
                 scenarios: list[dict]) -> dict[str, list[str]]:
    """scenario_id -> error list. Also catches duplicate ids."""
    results: dict[str, list[str]] = {}
    seen: set[str] = set()
    for i, sc in enumerate(scenarios):
        sid = str(sc.get("scenario_id") or f"index_{i}")
        errs = validate_scenario(contract, sc)
        if sid in seen:
            errs.append(f"duplicate scenario_id '{sid}'")
        seen.add(sid)
        results[sid] = errs
    return results
