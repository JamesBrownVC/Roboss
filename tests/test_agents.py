"""Unit tests for the deterministic parts of the agents package.

No Gemini calls: the validator and compiler are pure functions over dicts,
so contract compliance logic is tested with a fabricated contract.
"""

import copy

from agents.compiler import (
    compile_scenario,
    compile_video_prompt,
    compile_verifier_packet,
)
from agents.canvas import canvas_prompt, start_frame_prompt
from agents.validator import validate_all, validate_scenario

CONTRACT = {
    "world_contract": {
        "world_id": "warehouse_canvas_001",
        "description": "Narrow warehouse aisle with a humanoid robot.",
        "locked_entities": [
            {"id": "robot_01", "type": "humanoid_robot",
             "appearance": "white and dark gray slim humanoid robot",
             "material": "matte composite shell",
             "default_position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            {"id": "human_01", "type": "warehouse_worker",
             "appearance": "worker in yellow safety vest and white helmet",
             "material": "fabric",
             "default_position": {"x": 2.0, "y": 0.8, "z": 0.0}},
            {"id": "box_01", "type": "cardboard_box",
             "appearance": "medium brown cardboard box with worn edges",
             "material": "brown cardboard",
             "default_relation": "held_by robot_01 at chest level"},
        ],
        "locked_environment": {
            "layout": "narrow aisle with metal shelves on both sides",
            "floor": "gray concrete",
            "lighting": "cold industrial ceiling lights",
            "style": "realistic surveillance-like industrial video",
        },
    },
    "object_registry": [
        {
            "entity_id": "robot_01",
            "shape_signature": "slim humanoid robot with rounded shoulders",
            "material_signature": "matte composite shell",
            "color_signature": "white and dark gray",
            "surface_details": "smooth unbranded panels",
            "scale_signature": "adult human height",
            "reference_views": ["front_view", "side_view",
                                "three_quarter_view"],
            "negative_drift": ["do not change silhouette"],
        },
        {
            "entity_id": "human_01",
            "shape_signature": "adult warehouse worker silhouette",
            "material_signature": "fabric safety clothing",
            "color_signature": "yellow safety vest and white helmet",
            "surface_details": "plain vest with no readable text",
            "scale_signature": "adult human height",
            "reference_views": ["front_view", "side_view"],
            "negative_drift": ["do not change clothing colors"],
        },
        {
            "entity_id": "box_01",
            "shape_signature": "medium rectangular cardboard box",
            "material_signature": "brown cardboard",
            "color_signature": "medium brown",
            "surface_details": "worn edges",
            "scale_signature": "carried at robot chest level",
            "reference_views": ["front_view", "side_view", "top_view"],
            "negative_drift": ["do not become a bag or crate"],
        },
    ],
    "scene_registry": {
        "scene_id": "warehouse_canvas_001",
        "layout_signature": "narrow aisle with metal shelves on both sides",
        "background_signature": "gray concrete floor and metal warehouse shelves",
        "lighting_signature": "cold industrial ceiling lights",
        "spatial_map": "robot in aisle center, worker near side shelf",
        "reference_views": ["wide_establishing", "left_angle",
                            "empty_background_plate"],
        "negative_drift": ["do not move shelves or change floor material"],
    },
    "reference_assets": [
        {
            "asset_id": "warehouse_canvas_001_canvas",
            "kind": "canvas",
            "target": "canvas.png",
            "description": "Canonical full-scene visual anchor.",
            "source_scene_id": "warehouse_canvas_001",
        },
    ],
    "identity_checks": [
        {
            "check_id": "box_01_crop_similarity",
            "scope": "object",
            "target_id": "box_01",
            "method": "compare object crops to reference views",
            "pass_condition": "same box shape, color and worn edges",
        },
        {
            "check_id": "warehouse_canvas_001_layout_similarity",
            "scope": "scene",
            "target_id": "warehouse_canvas_001",
            "method": "compare background frames to scene references",
            "pass_condition": "same shelves, floor and lighting",
        },
    ],
    "variation_policy": {
        "allowed_camera_angles": ["side_view", "overhead_view", "robot_pov"],
        "allowed_event_types": ["human_slip", "box_slips_from_robot",
                                "human_crosses_path"],
        "allowed_position_deltas": [
            {"entity_id": "human_01", "dx": [-0.6, 0.6], "dy": [-0.6, 0.6]},
        ],
        "event_timing_seconds": [1.0, 4.0],
        "forbidden_changes": ["changing robot_01 appearance"],
        "consistency_requirements": ["all scenarios contain all entities"],
    },
}

GOOD_SCENARIO = {
    "scenario_id": "sc_001_human_slip",
    "title": "Human slips near the robot carrying a box",
    "referenced_entities": ["robot_01", "human_01", "box_01"],
    "camera": {"angle": "side_view", "movement": "static",
               "duration_seconds": 6.0},
    "event": {"type": "human_slip", "trigger_time_seconds": 2.2,
              "risk_severity": "high",
              "description": "human_01 slips on the floor near robot_01"},
    "position_deltas": [
        {"entity_id": "human_01", "dx": 0.2, "dy": -0.1,
         "reason": "worker starts slightly closer to the aisle"},
    ],
    "action_timeline": [
        {"t_start": 0.0, "t_end": 1.5, "actor": "robot_01",
         "action": "walks forward carrying box_01"},
        {"t_start": 1.5, "t_end": 2.2, "actor": "human_01",
         "action": "crosses the aisle near robot_01"},
        {"t_start": 2.2, "t_end": 3.1, "actor": "human_01",
         "action": "slips and loses balance"},
        {"t_start": 3.1, "t_end": 5.0, "actor": "robot_01",
         "action": "stops and stabilizes box_01"},
    ],
    "expected_labels": ["human_slip", "unsafe_proximity", "robot_should_stop"],
    "expected_robot_response": "stop_and_secure_object",
}


def _broken(**overrides):
    sc = copy.deepcopy(GOOD_SCENARIO)
    for path, value in overrides.items():
        keys = path.split(".")
        node = sc
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return sc


# ---------------------------------------------------------------- validator

def test_good_scenario_passes():
    assert validate_scenario(CONTRACT, GOOD_SCENARIO) == []


def test_missing_entity_rejected():
    sc = _broken(referenced_entities=["robot_01", "human_01"])
    assert any("missing locked entities" in e
               for e in validate_scenario(CONTRACT, sc))


def test_invented_entity_rejected():
    sc = _broken(referenced_entities=["robot_01", "human_01", "box_01",
                                      "forklift_01"])
    assert any("unknown ids" in e for e in validate_scenario(CONTRACT, sc))


def test_bad_camera_angle_rejected():
    sc = _broken(**{"camera.angle": "front_view"})  # not in contract subset
    assert any("allowed_camera_angles" in e
               for e in validate_scenario(CONTRACT, sc))


def test_bad_event_type_rejected():
    sc = _broken(**{"event.type": "meteor_strike"})
    assert any("allowed_event_types" in e
               for e in validate_scenario(CONTRACT, sc))


def test_trigger_outside_window_rejected():
    sc = _broken(**{"event.trigger_time_seconds": 5.5})  # window is [1, 4]
    assert any("timing window" in e or "duration" in e
               for e in validate_scenario(CONTRACT, sc))


def test_unordered_timeline_rejected():
    sc = copy.deepcopy(GOOD_SCENARIO)
    sc["action_timeline"][1], sc["action_timeline"][3] = \
        sc["action_timeline"][3], sc["action_timeline"][1]
    assert any("not ordered" in e for e in validate_scenario(CONTRACT, sc))


def test_unknown_actor_rejected():
    sc = copy.deepcopy(GOOD_SCENARIO)
    sc["action_timeline"][0]["actor"] = "ghost_01"
    assert any("not a locked entity" in e
               for e in validate_scenario(CONTRACT, sc))


def test_position_delta_outside_policy_rejected():
    sc = copy.deepcopy(GOOD_SCENARIO)
    sc["position_deltas"][0]["dx"] = 1.2
    assert any("outside" in e and "human_01" in e
               for e in validate_scenario(CONTRACT, sc))


def test_duplicate_ids_rejected():
    results = validate_all(CONTRACT, [GOOD_SCENARIO,
                                      copy.deepcopy(GOOD_SCENARIO)])
    assert any("duplicate scenario_id" in e
               for errs in results.values() for e in errs)


# ----------------------------------------------------------------- compiler

def test_prompt_repeats_all_appearance_locks():
    prompt = compile_video_prompt(CONTRACT, GOOD_SCENARIO)
    for entity in CONTRACT["world_contract"]["locked_entities"]:
        assert entity["appearance"] in prompt
    env = CONTRACT["world_contract"]["locked_environment"]
    assert env["layout"] in prompt and env["lighting"] in prompt
    assert "Object identity anchors" in prompt
    assert "Scene identity anchor" in prompt
    assert "Do not change" in prompt          # consistency clause
    assert "side view" in prompt              # camera plan


def test_compiled_scenario_artifacts():
    sc = compile_scenario(CONTRACT, GOOD_SCENARIO)
    assert sc["inherits_world_contract"] == "warehouse_canvas_001"
    kf = sc["keyframes"]
    assert kf[0]["time"] == 0.0 and kf[1]["time"] == 2.2 and kf[2]["time"] == 6.0
    packet = sc["verifier_packet"]
    assert packet["expected_action"] == "human_slip"
    assert "cardboard box" in packet["expected_objects"]
    assert packet["expected_outcome"] == "stop_and_secure_object"
    assert packet["object_registry"][2]["entity_id"] == "box_01"
    assert packet["scene_registry"]["scene_id"] == "warehouse_canvas_001"
    assert packet["identity_checks"]


def test_verifier_packet_matches_verifier_format():
    packet = compile_verifier_packet(CONTRACT, GOOD_SCENARIO)
    # the fields verifier/report.py reads from --scenario
    assert "scenario_prompt" in packet and "expected_objects" in packet


def test_canvas_prompts_reference_the_contract():
    cp = canvas_prompt(CONTRACT)
    assert "gray concrete" in cp and "cardboard box" in cp
    sc = compile_scenario(CONTRACT, GOOD_SCENARIO)
    sp = start_frame_prompt(sc)
    assert "side view" in sp and "SAME" in sp
