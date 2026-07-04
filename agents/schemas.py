"""Response schemas for Gemini structured output.

Rules that keep these schemas Gemini-friendly:
- arrays of objects instead of dicts with dynamic keys
  (e.g. position deltas are [{entity_id, ...}] not {entity_id: ...});
- enums come from fixed vocabularies in config.py so the deterministic
  validator can check membership.
"""

from .config import CAMERA_ANGLES, RISK_SEVERITIES

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "environment": {"type": "string"},
        "main_task": {"type": "string"},
        "risk_focus": {"type": "string"},
        "actors": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "variation_count": {"type": "integer"},
        "duration_seconds": {"type": "number"},
        "camera_style": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["domain", "environment", "main_task", "risk_focus",
                 "actors", "objects", "variation_count", "duration_seconds"],
}

_POSITION = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
    "required": ["x", "y"],
}

_ENTITY = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "appearance": {"type": "string"},
        "identity_signature": {"type": "string"},
        "locked_attributes": {"type": "array", "items": {"type": "string"}},
        "material": {"type": "string"},
        "default_position": _POSITION,
        "default_relation": {"type": "string"},
    },
    "required": ["id", "type", "appearance", "material"],
}

_OBJECT_ANCHOR = {
    "type": "object",
    "properties": {
        "entity_id": {"type": "string"},
        "shape_signature": {"type": "string"},
        "material_signature": {"type": "string"},
        "color_signature": {"type": "string"},
        "surface_details": {"type": "string"},
        "scale_signature": {"type": "string"},
        "reference_views": {
            "type": "array",
            "items": {"type": "string"},
        },
        "negative_drift": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["entity_id", "shape_signature", "material_signature",
                 "color_signature", "reference_views"],
}

_SCENE_ANCHOR = {
    "type": "object",
    "properties": {
        "scene_id": {"type": "string"},
        "layout_signature": {"type": "string"},
        "background_signature": {"type": "string"},
        "lighting_signature": {"type": "string"},
        "spatial_map": {"type": "string"},
        "reference_views": {
            "type": "array",
            "items": {"type": "string"},
        },
        "negative_drift": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["scene_id", "layout_signature", "background_signature",
                 "lighting_signature", "reference_views"],
}

_REFERENCE_ASSET = {
    "type": "object",
    "properties": {
        "asset_id": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": ["canvas", "object_view", "scene_view", "start_frame",
                     "mask", "depth_map"],
        },
        "target": {"type": "string"},
        "description": {"type": "string"},
        "source_entity_id": {"type": "string"},
        "source_scene_id": {"type": "string"},
    },
    "required": ["asset_id", "kind", "target", "description"],
}

_IDENTITY_CHECK = {
    "type": "object",
    "properties": {
        "check_id": {"type": "string"},
        "scope": {"type": "string", "enum": ["object", "scene", "series"]},
        "target_id": {"type": "string"},
        "method": {"type": "string"},
        "pass_condition": {"type": "string"},
    },
    "required": ["check_id", "scope", "target_id", "method",
                 "pass_condition"],
}

_DELTA_RANGE = {
    "type": "object",
    "properties": {
        "entity_id": {"type": "string"},
        "dx": {"type": "array", "items": {"type": "number"}},
        "dy": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["entity_id", "dx", "dy"],
}

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "world_contract": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "description": {"type": "string"},
                "locked_entities": {"type": "array", "items": _ENTITY},
                "locked_environment": {
                    "type": "object",
                    "properties": {
                        "layout": {"type": "string"},
                        "floor": {"type": "string"},
                        "lighting": {"type": "string"},
                        "style": {"type": "string"},
                    },
                    "required": ["layout", "floor", "lighting", "style"],
                },
            },
            "required": ["world_id", "description", "locked_entities",
                         "locked_environment"],
        },
        "object_registry": {
            "type": "array",
            "items": _OBJECT_ANCHOR,
        },
        "scene_registry": _SCENE_ANCHOR,
        "reference_assets": {
            "type": "array",
            "items": _REFERENCE_ASSET,
        },
        "identity_checks": {
            "type": "array",
            "items": _IDENTITY_CHECK,
        },
        "variation_policy": {
            "type": "object",
            "properties": {
                "allowed_camera_angles": {
                    "type": "array",
                    "items": {"type": "string", "enum": CAMERA_ANGLES},
                },
                "allowed_event_types": {
                    "type": "array", "items": {"type": "string"},
                },
                "allowed_position_deltas": {
                    "type": "array", "items": _DELTA_RANGE,
                },
                "event_timing_seconds": {
                    "type": "array", "items": {"type": "number"},
                },
                "forbidden_changes": {
                    "type": "array", "items": {"type": "string"},
                },
                "consistency_requirements": {
                    "type": "array", "items": {"type": "string"},
                },
            },
            "required": ["allowed_camera_angles", "allowed_event_types",
                         "forbidden_changes", "consistency_requirements"],
        },
    },
    "required": ["world_contract", "object_registry", "scene_registry",
                 "reference_assets", "identity_checks", "variation_policy"],
}

_TIMELINE_STEP = {
    "type": "object",
    "properties": {
        "t_start": {"type": "number"},
        "t_end": {"type": "number"},
        "actor": {"type": "string"},
        "action": {"type": "string"},
    },
    "required": ["t_start", "t_end", "actor", "action"],
}

SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {
        "scenario_id": {"type": "string"},
        "title": {"type": "string"},
        "referenced_entities": {"type": "array", "items": {"type": "string"}},
        "camera": {
            "type": "object",
            "properties": {
                "angle": {"type": "string", "enum": CAMERA_ANGLES},
                "movement": {"type": "string"},
                "duration_seconds": {"type": "number"},
            },
            "required": ["angle", "movement", "duration_seconds"],
        },
        "event": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "trigger_time_seconds": {"type": "number"},
                "risk_severity": {"type": "string", "enum": RISK_SEVERITIES},
                "description": {"type": "string"},
            },
            "required": ["type", "trigger_time_seconds", "risk_severity",
                         "description"],
        },
        "position_deltas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "dx": {"type": "number"},
                    "dy": {"type": "number"},
                    "dz": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["entity_id", "dx", "dy"],
            },
        },
        "action_timeline": {"type": "array", "items": _TIMELINE_STEP},
        "expected_labels": {"type": "array", "items": {"type": "string"}},
        "expected_robot_response": {"type": "string"},
    },
    "required": ["scenario_id", "title", "referenced_entities", "camera",
                 "event", "action_timeline", "expected_labels",
                 "expected_robot_response"],
}

SCENARIOS_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {"type": "array", "items": SCENARIO_SCHEMA},
    },
    "required": ["scenarios"],
}
