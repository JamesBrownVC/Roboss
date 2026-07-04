"""Visual anchors: canonical canvas image + per-scenario start frames.

Cross-video consistency needs more than repeated prompt text. The canvas
image is the single visual source of truth for the world; each scenario's
start frame is derived FROM that canvas (image + edit instruction), so
every video series can be generated image-to-video from frames that share
one origin. Both stages are optional and degrade gracefully.
"""

from __future__ import annotations

from .compiler import _entity_clause
from .config import AgentConfig
from .llm import generate_image, image_part, text_part


def canvas_prompt(contract: dict) -> str:
    wc = contract["world_contract"]
    env = wc["locked_environment"]
    scene = contract.get("scene_registry", {})
    entities = "; ".join(_entity_clause(e).replace("the same ", "a ")
                         for e in wc["locked_entities"])
    anchors = "; ".join(
        f"{a['entity_id']}: {a['shape_signature']}, "
        f"{a['material_signature']}, {a['color_signature']}"
        for a in contract.get("object_registry", [])
    )
    return (f"{env['style'].rstrip('.')}. Wide establishing shot of "
            f"{env['layout']}, {env['floor']} floor, {env['lighting']}. "
            f"Canonical scene anchor: "
            f"{scene.get('layout_signature', env['layout'])}; "
            f"{scene.get('background_signature', env['layout'])}. "
            f"In the scene: {entities}. Everything in a neutral resting "
            f"state, nobody moving, no text or logos anywhere. This image "
            f"is the canonical reference for a whole video series, so make "
            f"every object clearly and fully visible. Object anchors: "
            f"{anchors}.")


def generate_canvas(contract: dict, cfg: AgentConfig) -> bytes | None:
    return generate_image(cfg.image_model, [text_part(canvas_prompt(contract))])


def start_frame_prompt(scenario: dict) -> str:
    kf0 = scenario["keyframes"][0]["state"]
    angle = scenario["camera"]["angle"].replace("_", " ")
    return (f"Using the provided image as the exact scene reference, render "
            f"the SAME room and the SAME objects — identical shapes, colors, "
            f"materials, layout and lighting — from a {angle}. Initial "
            f"state of the action: {kf0}. Change nothing about any object's "
            f"identity; only the viewpoint and the actors' poses may differ.")


def generate_start_frame(canvas_png: bytes, scenario: dict,
                         cfg: AgentConfig) -> bytes | None:
    return generate_image(cfg.image_model, [
        image_part(canvas_png),
        text_part(start_frame_prompt(scenario)),
    ])
