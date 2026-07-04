"""Contact (Stage F): signed distance hand mesh ↔ object mesh."""

from __future__ import annotations

from ..schema.models import StageStatus
from . import _synthetic as syn
from .base import Stage, StageContext, StageResult, register_stage


@register_stage
class ContactStage(Stage):
    name = "contact"

    def run(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        # Geometric inference runs in orchestrator env (no isolated CUDA tool)
        syn.synthesize_contacts(ws, ctx.cfg, syn.episode_rng(ws, self.name))
        outputs = [ws.rel(ws.contacts_parquet)]
        return StageResult(
            status=StageStatus.success,
            metrics={"n_contacts": 1},
            outputs=outputs,
            tool="geometric_contact",
            repo="v2r-internal",
            commit="0.1.0",
        )
