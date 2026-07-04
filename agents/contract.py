"""Agent 2 — World Contract Builder: intent -> immutable canvas + policy.

The contract is the source of truth every scenario inherits. It splits the
world into an immutable part (entities with appearance locks, environment,
style) and a mutable part (the variation policy).
"""

from __future__ import annotations

import json

from .config import AgentConfig, CAMERA_ANGLES
from .llm import generate_json, text_part
from .schemas import CONTRACT_SCHEMA

SYSTEM = f"""\
You build a WORLD CONTRACT for consistent multi-video generation: one
immutable canvas that many scenario variations will share. Video models
drift — they change objects, materials and layouts between generations —
so the contract must pin down everything that has to stay identical.

Rules:
- 2 to 6 locked entities with snake_case ids ending in _01 (_02 for a
  second instance of the same type).
- "appearance" must be specific enough to repeat verbatim in every video
  prompt (colors, size, distinguishing marks), but contain no readable
  text or brand names.
- For each entity, write an identity_signature and locked_attributes that
  can be used later to verify cross-video consistency.
- object_registry must contain one anchor per locked entity: shape,
  material, color, surface details, scale, required reference views, and
  negative drift examples.
- scene_registry must describe the canonical background and spatial layout:
  layout signature, background signature, lighting signature, spatial map,
  required reference views, and negative drift examples.
- reference_assets must define the assets the generation side should create:
  one canonical canvas, object reference views, scene reference views, and
  per-scenario start frames.
- identity_checks must describe how generated videos should be checked after
  rendering: object crop similarity, background/layout similarity, and VLM
  contract compliance.
- default_position uses meters in a scene-local frame (x right, y forward,
  z up); use default_relation instead when an entity is held by or attached
  to another (e.g. "held_by robot_01 at chest level").
- allowed_camera_angles: pick 3-6 from {json.dumps(CAMERA_ANGLES)}.
- allowed_event_types: 4-8 short snake_case event names that fit the user's
  risk focus and are physically possible for these entities.
- allowed_position_deltas: small per-entity ranges in meters ([-0.6, 0.6]
  at most for humans, tighter for robots and static objects).
- forbidden_changes: explicit list of what generation must never alter
  (identity, colors, materials, layout, style, adding main actors).
- consistency_requirements: what every scenario must contain and preserve.
The environment style must read like realistic footage, not cinematic."""


def _fallback_object_anchor(entity: dict) -> dict:
    """Derive a conservative object anchor if the model omits one."""
    eid = entity["id"]
    return {
        "entity_id": eid,
        "shape_signature": entity.get("identity_signature")
        or entity.get("appearance", entity.get("type", eid)),
        "material_signature": entity.get("material", "unspecified material"),
        "color_signature": entity.get("appearance", "unspecified color"),
        "surface_details": entity.get("appearance", ""),
        "scale_signature": entity.get("appearance", ""),
        "reference_views": ["front_view", "side_view", "three_quarter_view"],
        "negative_drift": [
            "do not change silhouette",
            "do not change color or material",
            "do not add readable text or logos",
        ],
    }


def _fallback_scene_anchor(wc: dict) -> dict:
    env = wc["locked_environment"]
    return {
        "scene_id": wc["world_id"],
        "layout_signature": env["layout"],
        "background_signature": f"{env['layout']}; {env['floor']} floor",
        "lighting_signature": env["lighting"],
        "spatial_map": wc.get("description", env["layout"]),
        "reference_views": ["wide_establishing", "left_angle",
                            "right_angle", "empty_background_plate"],
        "negative_drift": [
            "do not move shelves, walls, floor lines or large fixtures",
            "do not change lighting family",
            "do not change visual style",
        ],
    }


def normalize_contract(contract: dict) -> dict:
    """Make the advanced contract complete and internally cross-linked."""
    wc = contract["world_contract"]
    policy = contract["variation_policy"]
    entities = wc["locked_entities"]
    wc["entity_ids"] = [e["id"] for e in entities]

    for e in entities:
        e.setdefault("identity_signature", e.get("appearance", e["type"]))
        e.setdefault("locked_attributes", [
            "shape", "silhouette", "color", "material", "scale",
            "surface_details",
        ])

    anchors = {
        a.get("entity_id"): a
        for a in contract.get("object_registry", [])
        if a.get("entity_id")
    }
    contract["object_registry"] = [
        anchors.get(e["id"], _fallback_object_anchor(e))
        for e in entities
    ]

    contract["scene_registry"] = (
        contract.get("scene_registry") or _fallback_scene_anchor(wc)
    )

    scene_id = contract["scene_registry"]["scene_id"]
    assets = contract.get("reference_assets") or []
    if not assets:
        assets = [{
            "asset_id": f"{scene_id}_canvas",
            "kind": "canvas",
            "target": "canvas.png",
            "description": "Canonical full-scene visual anchor for the series.",
            "source_scene_id": scene_id,
        }]
        for anchor in contract["object_registry"]:
            eid = anchor["entity_id"]
            for view in anchor["reference_views"]:
                assets.append({
                    "asset_id": f"{eid}_{view}",
                    "kind": "object_view",
                    "target": f"references/{eid}_{view}.png",
                    "description": f"Canonical {view} reference for {eid}.",
                    "source_entity_id": eid,
                })
        for view in contract["scene_registry"]["reference_views"]:
            assets.append({
                "asset_id": f"{scene_id}_{view}",
                "kind": "scene_view",
                "target": f"references/{scene_id}_{view}.png",
                "description": f"Canonical {view} reference for {scene_id}.",
                "source_scene_id": scene_id,
            })
    contract["reference_assets"] = assets

    checks = contract.get("identity_checks") or []
    if not checks:
        checks = [
            {
                "check_id": f"{a['entity_id']}_crop_similarity",
                "scope": "object",
                "target_id": a["entity_id"],
                "method": "compare sampled object crops to reference views "
                          "with VLM and embedding similarity",
                "pass_condition": "same silhouette, material, color, scale "
                                  "and surface details across the series",
            }
            for a in contract["object_registry"]
        ]
        checks.append({
            "check_id": f"{scene_id}_layout_similarity",
            "scope": "scene",
            "target_id": scene_id,
            "method": "compare sampled background frames to scene reference "
                      "views with VLM and layout descriptors",
            "pass_condition": "same background identity, spatial layout, "
                              "lighting family and visual style",
        })
    contract["identity_checks"] = checks

    policy.setdefault("allowed_position_deltas", [])
    policy.setdefault("forbidden_changes", [])
    policy.setdefault("consistency_requirements", [])
    return contract


def build_contract(intent: dict, cfg: AgentConfig) -> dict:
    contract = generate_json(
        model=cfg.text_model,
        system=SYSTEM,
        user_parts=[text_part("Structured intent:\n"
                              + json.dumps(intent, indent=2))],
        schema=CONTRACT_SCHEMA,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.plan_temperature,
    )
    contract = normalize_contract(contract)
    policy = contract["variation_policy"]
    # defensive defaults so the validator always has something to check
    policy.setdefault("event_timing_seconds",
                      [0.5, max(1.0, intent["duration_seconds"] - 1.0)])
    policy.setdefault("allowed_position_deltas", [])
    if not policy["allowed_camera_angles"]:
        policy["allowed_camera_angles"] = ["front_view", "side_view"]
    return contract
