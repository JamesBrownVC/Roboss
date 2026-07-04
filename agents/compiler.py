"""Agent 5 — deterministic video prompt compiler (no LLM).

The video prompt is *compiled* from the contract, never free-written by a
model: this guarantees every prompt repeats the same appearance locks,
environment and style verbatim across all scenarios — the textual half of
cross-video consistency. It also emits keyframe states (temporal anchors)
and a verifier packet consumable by `python -m verifier --scenario`.
"""

from __future__ import annotations


def _entity_clause(entity: dict) -> str:
    clause = f"the same {entity['appearance']}"
    material = entity.get("material")
    if material and material.lower() not in entity["appearance"].lower():
        clause += f" ({material})"
    return clause


def compile_video_prompt(contract: dict, scenario: dict) -> str:
    wc = contract["world_contract"]
    env = wc["locked_environment"]
    camera = scenario["camera"]
    scene = contract.get("scene_registry", {})
    registry = contract.get("object_registry", [])

    parts = [
        f"{env['style'].rstrip('.')}.",
        f"Setting: {env['layout']}, {env['floor']} floor, {env['lighting']}.",
        "Scene identity anchor: "
        f"{scene.get('layout_signature', env['layout'])}; "
        f"{scene.get('background_signature', env['layout'])}; "
        f"{scene.get('lighting_signature', env['lighting'])}.",
        "Identical across all videos of this series: "
        + "; ".join(_entity_clause(e) for e in wc["locked_entities"]) + ".",
        "Object identity anchors: "
        + "; ".join(
            f"{a['entity_id']} = shape {a['shape_signature']}, "
            f"material {a['material_signature']}, "
            f"color {a['color_signature']}"
            for a in registry
        ) + ".",
        f"Camera: {camera['angle'].replace('_', ' ')}, "
        f"{camera['movement'].replace('_', ' ')}, "
        f"{camera['duration_seconds']:g} seconds.",
    ]

    id_to_appearance = {e["id"]: e["appearance"]
                        for e in wc["locked_entities"]}
    for step in scenario["action_timeline"]:
        actor = id_to_appearance.get(step["actor"], step["actor"])
        parts.append(f"From {step['t_start']:g}s to {step['t_end']:g}s "
                     f"the {actor} {step['action']}.")

    parts.append("Do not change any object's identity, shape, color, "
                 "material or size; do not alter the room layout, lighting "
                 "or visual style; do not add new people or objects.")
    return " ".join(parts)


def compile_keyframes(scenario: dict) -> list[dict]:
    """Three temporal anchors: initial state, event trigger, final state."""
    timeline = scenario["action_timeline"]
    event = scenario["event"]
    duration = scenario["camera"]["duration_seconds"]
    first = [s for s in timeline if s["t_start"] <= timeline[0]["t_start"]]
    return [
        {"time": 0.0,
         "state": "; ".join(f"{s['actor']}: {s['action']}" for s in first)},
        {"time": event["trigger_time_seconds"], "state": event["description"]},
        {"time": duration,
         "state": f"{timeline[-1]['actor']}: {timeline[-1]['action']}"},
    ]


def compile_verifier_packet(contract: dict, scenario: dict) -> dict:
    """The metadata packet the verification gate consumes (--scenario)."""
    wc = contract["world_contract"]
    scene = contract.get("scene_registry", {})
    return {
        "scenario_prompt": scenario["title"],
        "expected_objects": [e["type"].replace("_", " ")
                             for e in wc["locked_entities"]],
        "expected_action": scenario["event"]["type"],
        "expected_outcome": scenario["expected_robot_response"],
        "scenario_id": scenario["scenario_id"],
        "world_id": wc["world_id"],
        "object_registry": contract.get("object_registry", []),
        "scene_registry": scene,
        "reference_assets": contract.get("reference_assets", []),
        "identity_checks": contract.get("identity_checks", []),
        "consistency_policy": {
            "locked_entity_ids": wc.get("entity_ids", [
                e["id"] for e in wc["locked_entities"]
            ]),
            "locked_scene_id": scene.get("scene_id", wc["world_id"]),
            "forbidden_changes": contract["variation_policy"].get(
                "forbidden_changes", []),
            "consistency_requirements": contract["variation_policy"].get(
                "consistency_requirements", []),
        },
    }


def compile_scenario(contract: dict, scenario: dict) -> dict:
    """Returns the scenario enriched with all compiled artifacts."""
    out = dict(scenario)
    out["inherits_world_contract"] = contract["world_contract"]["world_id"]
    out["video_prompt"] = compile_video_prompt(contract, scenario)
    out["keyframes"] = compile_keyframes(scenario)
    out["verifier_packet"] = compile_verifier_packet(contract, scenario)
    return out
