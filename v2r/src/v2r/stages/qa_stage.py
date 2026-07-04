"""QA stage: cross-checks, decision, yield report."""

from __future__ import annotations

from ..qa.crosschecks import run_crosschecks
from ..qa.yield_report import write_yield_report
from ..schema.io import write_json_model
from ..schema.models import Decision, GateOutcome, StageStatus
from .base import Stage, StageContext, StageResult, register_stage


@register_stage
class QAStage(Stage):
    name = "qa"

    def run(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        cross = run_crosschecks(ws, ctx.cfg)
        write_json_model(ws.crosschecks_json, cross)

        gates: dict[str, GateOutcome] = {}
        for stage in ("ingest", "feasibility_judge", "geometry", "human_body", "hands", "objects"):
            mpath = ws.manifest_path(stage)
            if mpath.is_file():
                from ..schema.models import StageManifest
                from ..schema.io import read_json_model
                m = read_json_model(mpath, StageManifest)
                if m.metrics:
                    gates[stage] = GateOutcome(passed=m.status == StageStatus.success, metrics=m.metrics)

        accepted = cross.passed and all(g.passed for g in gates.values())
        decision = Decision(
            accepted=accepted,
            failure_stage=None if accepted else "qa",
            failure_reason=None if accepted else "; ".join(cross.reasons[:3]),
            gates=gates,
        )
        write_json_model(ws.decision_json, decision)
        write_yield_report(ws, ctx.cfg, ctx.robots)

        outputs = [
            ws.rel(ws.crosschecks_json),
            ws.rel(ws.decision_json),
            ws.rel(ws.yield_report_md),
        ]
        status = StageStatus.success if accepted else StageStatus.rejected
        return StageResult(
            status=status,
            metrics={"accepted": accepted},
            outputs=outputs,
            gate=GateOutcome(passed=accepted, reasons=cross.reasons, metrics=cross.model_dump()),
            tool="v2r-qa",
            repo="v2r-internal",
            commit="0.1.0",
        )
