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


@dataclass
class AgentConfig:
    # same Gemini family the verifier's gate 2 already uses
    text_model: str = "gemini-3.5-flash"
    image_model: str = "gemini-3.1-flash-image"
    max_output_tokens: int = 16000
    plan_temperature: float = 0.3       # intent + contract: precision
    variation_temperature: float = 0.9  # scenarios: diversity
    max_repair_rounds: int = 2          # validator -> LLM fix loop


DEFAULT_CONFIG = AgentConfig()
