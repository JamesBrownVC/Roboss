"""Interchange schema library: pydantic models, IO, rotations, timeline,
alignment, and the episode workspace layout. This package IS the contract:
stages talk to each other only through artifacts written/read via here."""

from . import alignment, io, models, rotations, timeline  # noqa: F401
from .models import (  # noqa: F401
    CameraInfo,
    Captions,
    ConsentRecord,
    CrossChecks,
    Decision,
    FusionReport,
    GateOutcome,
    PhysicsCheck,
    PhysicsReport,
    RetargetMapping,
    RobotClass,
    RobotSpec,
    SceneTags,
    Segment,
    SegmentsFile,
    Sim3,
    SourceTag,
    StageManifest,
    StageStatus,
    VideoProbe,
)
from .workspace import EpisodeWorkspace  # noqa: F401
