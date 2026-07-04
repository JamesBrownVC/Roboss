"""Feasibility judge: pre-analysis physics gate + optional VLM judge."""

from __future__ import annotations

from ..qa.feasibility import run_feasibility_judge
from ..schema.models import GateOutcome, StageStatus
from .base import Stage, StageContext, StageResult, register_stage, run_tool


@register_stage
class FeasibilityJudgeStage(Stage):
    name = "feasibility_judge"

    def run(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        toggle = ctx.cfg.stage("feasibility_judge")

        if ctx.mode == "real" and toggle.env:
            verbs_path = ctx.cfg.root / "config" / "verbs.yaml"
            cmd = [
                "python",
                str(ctx.cfg.root / "envs" / toggle.env / "tool_entry.py"),
                "--workspace", str(ws.root),
                "--mode", "real",
            ]
            proc = run_tool(cmd, env_name=toggle.env, cwd=ctx.cfg.root)
            if proc.returncode != 0 and proc.returncode != 2:
                return StageResult(
                    status=StageStatus.failed,
                    failure_reason=(proc.stderr or proc.stdout)[-2000:],
                    tool="qwen-vl-feasibility",
                    repo="envs/feasibility_judge",
                    commit="0.1.0",
                )

        report, _mask, gate_passed = run_feasibility_judge(ws, ctx.cfg, ctx.mode)
        outputs = [
            ws.rel(ws.feasibility_report_json),
            ws.rel(ws.feasibility_mask_parquet),
        ]
        metrics = {
            "recommendation": report.recommendation.value,
            "confidence": report.confidence,
            "physics_violation_frame_ratio": report.physics_violation_frame_ratio,
            "physically_plausible": report.physically_plausible,
        }
        gate = GateOutcome(
            passed=gate_passed,
            reasons=[] if gate_passed else [
                f"recommendation={report.recommendation.value}",
                f"physics_violation_frame_ratio={report.physics_violation_frame_ratio:.3f}",
            ],
            metrics=metrics,
        )
        status = StageStatus.success if gate_passed else StageStatus.rejected
        return StageResult(
            status=status,
            metrics=metrics,
            outputs=outputs,
            gate=gate,
            tool="v2r-feasibility-judge",
            repo="v2r-internal",
            commit="0.1.0",
        )
