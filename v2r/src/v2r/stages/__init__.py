"""Stage wrappers. Import via load_all_stages() so a broken module surfaces
as a named import error instead of killing the whole package import."""

from importlib import import_module

from .base import STAGE_DEPS, STAGE_ORDER, STAGE_REGISTRY, Stage, StageContext, StageResult  # noqa: F401

STAGE_MODULES = [
    "ingest",
    "feasibility_judge",
    "geometry",
    "human_body",
    "hands",
    "objects",
    "contact",
    "semantics",
    "retarget",
    "physics_validate",
    "qa_stage",
    "package",
]

IMPORT_ERRORS: dict[str, str] = {}


def load_all_stages() -> dict[str, str]:
    """Import every stage module; return {module: error} for failures."""
    IMPORT_ERRORS.clear()
    for mod in STAGE_MODULES:
        try:
            import_module(f"v2r.stages.{mod}")
        except Exception as e:  # pragma: no cover - surfaced by orchestrator
            IMPORT_ERRORS[mod] = f"{type(e).__name__}: {e}"
    return dict(IMPORT_ERRORS)
