"""Models and knobs for the scenario contract pipeline."""

from dataclasses import dataclass

# Fixed vocabularies keep the LLM output validatable. The contract picks a
# subset of these; every scenario must stay inside the contract's subset.
CAMERA_ANGLES = [
    "front_view",
    "side_view",
    "three_quarter_view",
    "overhead_view",
    "robot_pov",
    "corner_security_camera",
    "close_up",
]

RISK_SEVERITIES = ["low", "medium", "high"]

# Roboss videos are meant for motion analysis: keep the camera above the
# scene so robot trajectories and human/object interactions stay visible.
REQUIRED_CAMERA_ANGLE = "overhead_view"


@dataclass
class AgentConfig:
    # same Gemini family the verifier's gate 2 already uses
    text_model: str = "gemini-3.5-flash"
    # most capable image model (Nano Banana Pro): reference-image support and
    # character consistency, which is exactly what the anchoring needs
    image_model: str = "gemini-3-pro-image"
    # hint for the video-generation side (not called from this package):
    # top image-to-video model, consumes our start frames
    video_model: str = "veo-3.1-generate-preview"
    max_output_tokens: int = 16000
    # Gemini 3.5 replaces temperature/top_p/top_k with thinking_level
    # (minimal | low | medium | high); sampling params are deprecated there.
    intent_thinking_level: str = "low"       # simple extraction
    plan_thinking_level: str = "high"        # contract + repairs: precision
    variation_thinking_level: str = "high"   # scenarios: complex planning
    max_repair_rounds: int = 2               # validator -> LLM fix loop
    start_frame_workers: int = 4             # parallel image calls


DEFAULT_CONFIG = AgentConfig()
