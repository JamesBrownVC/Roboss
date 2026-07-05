"""Pydantic v2 models for every JSON artifact in the episode workspace."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceTag(str, Enum):
    captured = "captured"
    estimated = "estimated"
    triangulated = "triangulated"
    fused = "fused"
    synthesized = "synthesized"


class StageStatus(str, Enum):
    success = "success"
    failed = "failed"       # tool error / exception (retryable)
    rejected = "rejected"   # gate failure: labeled funnel outcome, not an exception
    skipped = "skipped"     # manifest hash matched; nothing re-run


class RobotClass(str, Enum):
    humanoid_wholebody = "humanoid_wholebody"
    ee_manipulator = "ee_manipulator"
    quadruped = "quadruped"


# ---------------------------------------------------------------------------
# raw/
# ---------------------------------------------------------------------------


class VideoProbe(BaseModel):
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str = ""
    pix_fmt: str = ""
    corrupt_frames: int = 0
    original_path: str = ""
    shots: list[tuple[float, float]] = Field(default_factory=list)  # (start_s, end_s)


class ConsentRecord(BaseModel):
    consent_id: str
    license: str
    subject_ids: list[str] = Field(default_factory=list)
    blur_applied: bool = False
    allow_blurred_video_redistribution: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# geometry/
# ---------------------------------------------------------------------------


class CameraInfo(BaseModel):
    model: Literal["pinhole", "fisheye", "equirect"] = "pinhole"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float] = Field(default_factory=list)
    # depth PNG units per meter (1000.0 == millimeters)
    depth_scale: float = 1000.0
    # depth maps may be stored at reduced resolution; None == video resolution
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    scale_source: Literal[
        "vipe_near_metric", "smplx_height", "known_object", "triangulated", "synthetic"
    ] = "vipe_near_metric"
    scale_correction: float = 1.0

    def K(self) -> list[list[float]]:
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]


# ---------------------------------------------------------------------------
# human/
# ---------------------------------------------------------------------------


class Sim3(BaseModel):
    """Similarity transform (Umeyama result), applied as x' = s * R @ x + t."""

    scale: float
    quat_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]


class FusionReport(BaseModel):
    """World alignment of body-tracker output into the ViPE world frame."""

    sim3: Sim3
    rms_residual_m: float
    p95_residual_m: float
    n_frames: int
    notes: str = ""


# ---------------------------------------------------------------------------
# semantics/
# ---------------------------------------------------------------------------


class Segment(BaseModel):
    start_s: float
    end_s: float
    skill: str          # must come from config/verbs.yaml (validated at stage level)
    text: str
    # per-label provenance for data buyers: which measurements/tools support
    # this segment (e.g. "boundary from primitives changepoint at 4.9s;
    # skill from VLM + wrist-speed profile"). Optional: legacy labels lack it.
    evidence: Optional[str] = None

    @field_validator("end_s")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        if "start_s" in info.data and v < info.data["start_s"]:
            raise ValueError("end_s < start_s")
        return v


class SegmentsFile(BaseModel):
    segments: list[Segment]
    method: str = ""
    source: SourceTag = SourceTag.estimated


class Captions(BaseModel):
    short: str
    medium: str
    long: str
    source: SourceTag = SourceTag.estimated


class Utterance(BaseModel):
    """One spoken utterance, aligned to the video timeline. The situation ->
    utterance pairing is what teaches a robot WHAT TO SAY in context."""

    t_start_s: float
    t_end_s: float
    speaker: str = "person_0"
    text: str
    language: str = ""            # BCP-47-ish, e.g. "en", "fr"
    intent: str = "other"         # greeting|instruction|comment|response|other
    aligned_segment: Optional[str] = None   # skill of the co-occurring action segment
    conf: float = Field(0.5, ge=0.0, le=1.0)


class UtterancesFile(BaseModel):
    utterances: list[Utterance] = Field(default_factory=list)
    has_speech: bool = False
    audio_notes: str = ""         # e.g. "music only", "no audio track"
    method: str = ""
    source: SourceTag = SourceTag.estimated


class SceneTags(BaseModel):
    scene_type: str = "unknown"
    lighting: str = "unknown"
    clutter: int = Field(3, ge=1, le=5)
    surfaces: list[str] = Field(default_factory=list)
    source: SourceTag = SourceTag.estimated


# ---------------------------------------------------------------------------
# robots / retarget / physics
# ---------------------------------------------------------------------------


class RobotSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str = ""
    robot_class: RobotClass = Field(alias="class")
    model_path: str
    control_rate_hz: float = 50.0
    dof: list[str]
    key_body_map: dict[str, str] = Field(default_factory=dict)
    joint_limits: dict[str, tuple[float, float]] = Field(default_factory=dict)
    joint_limits_default: tuple[float, float] = (-3.14, 3.14)
    feet_links: list[str] = Field(default_factory=list)

    def limits_for(self, joint: str) -> tuple[float, float]:
        return self.joint_limits.get(joint, self.joint_limits_default)


class RetargetMapping(BaseModel):
    robot: str
    robot_class: RobotClass
    retarget_method: str          # e.g. "gmr", "mink_diff_ik", "base_twist_abstraction"
    retarget_version: str = "0.1.0"
    # kinematic-retarget vs command-abstraction (master prompt 6.H)
    provenance: Literal["kinematic-retarget", "command-abstraction"] = "kinematic-retarget"
    key_body_map: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class PhysicsCheck(BaseModel):
    violations: int = 0
    max_value: float = 0.0
    threshold: float = 0.0
    skipped: bool = False
    note: str = ""


class PhysicsReport(BaseModel):
    robot: str
    tier: int = 1
    n_frames: int = 0
    engine: str = "kinematic"     # "mujoco" | "kinematic" (numpy fallback)
    checks: dict[str, PhysicsCheck] = Field(default_factory=dict)
    physics_valid: bool = False
    violation_frame_ratio: float = 0.0
    tracking_error: Optional[float] = None    # Tier 2 only
    premium: bool = False


# ---------------------------------------------------------------------------
# qa/
# ---------------------------------------------------------------------------


class GateOutcome(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    accepted: bool
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    gates: dict[str, GateOutcome] = Field(default_factory=dict)


class CrossChecks(BaseModel):
    reproj_err_px_mean: Optional[float] = None
    reproj_err_px_p95: Optional[float] = None
    depth_disagreement_m_median: Optional[float] = None
    jitter_m_s2_max: Optional[float] = None
    quat_norm_err_max: Optional[float] = None
    timestamps_monotonic: Optional[bool] = None
    details: dict[str, Any] = Field(default_factory=dict)
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)


class FeasibilityRecommendation(str, Enum):
    proceed = "proceed"
    reject = "reject"
    human_review = "human_review"


class FeasibilityReport(BaseModel):
    """Pre-analysis QA: physics heuristics + optional VLM judge."""

    physically_plausible: bool
    tracking_likely_valid: bool
    ai_generated_artifacts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: FeasibilityRecommendation
    physics_violation_frame_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    physics_checks: dict[str, Any] = Field(default_factory=dict)
    judge_source: Literal["synthetic", "rule_based", "vlm", "api"] = "synthetic"
    notes: str = ""


class CameraSync(BaseModel):
    cam_id: str
    offset_s: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class SessionSync(BaseModel):
    method: Literal["timecode", "audio_xcorr", "visual_flash", "synthetic"] = "synthetic"
    reference_cam: str = "cam0"
    cameras: list[CameraSync] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CameraCalibration(BaseModel):
    cam_id: str
    intrinsics: CameraInfo
    T_world_cam: list[list[float]]  # 4x4 row-major at reference frame


class SessionCalibration(BaseModel):
    method: Literal["checkerboard", "colmap", "synthetic"] = "synthetic"
    reference_cam: str = "cam0"
    cameras: list[CameraCalibration] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class CrossViewReprojFrame(BaseModel):
    frame: int
    joint: str
    reproj_error_px: float
    confidence: float = Field(ge=0.0, le=1.0)


class CrossViewReprojReport(BaseModel):
    """Objective cross-view reprojection accuracy (verifiable, not self-asserted)."""

    session_id: str
    n_frames: int = 0
    n_joints: int = 0
    mean_reproj_error_px: float = 0.0
    p95_reproj_error_px: float = 0.0
    per_frame: list[CrossViewReprojFrame] = Field(default_factory=list)
    monocular_shadow_mean_px: Optional[float] = None
    triangulation_wins: Optional[bool] = None
    source: SourceTag = SourceTag.triangulated


# ---------------------------------------------------------------------------
# manifests/
# ---------------------------------------------------------------------------


class StageManifest(BaseModel):
    stage: str
    tool: str = ""
    repo: str = ""
    commit: str = ""                                  # pinned git commit
    weights_sha256: dict[str, str] = Field(default_factory=dict)
    config_hash: str = ""
    input_hash: str = ""
    output_hash: str = ""
    mode: str = "synthetic"                           # synthetic | real
    started_at: str = ""                              # UTC isoformat
    finished_at: str = ""
    runtime_s: float = 0.0
    status: StageStatus = StageStatus.failed
    metrics: dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
    outputs: list[str] = Field(default_factory=list)  # workspace-relative paths
