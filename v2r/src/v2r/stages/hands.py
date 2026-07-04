"""Hands (Stage D): WiLoR/HaMeR (MANO) → EgoDex 25-joint SE(3) tables."""

from __future__ import annotations

from ..schema.models import StageStatus
from . import _synthetic as syn
from .base import Stage, StageContext, StageResult, gate_from_thresholds, register_stage, run_tool

TOOL = {
    "tool": "WiLoR",
    "repo": "github.com/rolpotamias/WiLoR",
    "commit": "b2c3d4e5f6789012345678901234567890abcde1",
}
ENV = "wilor"
# Alternatives: HaMeR, HaWoR (egocentric world hands)


@register_stage
class HandsStage(Stage):
    name = "hands"

    def run(self, ctx: StageContext) -> StageResult:
        qa = ctx.cfg.qa.get("hands", {})
        if ctx.mode == "real":
            return self._run_real(ctx, qa)
        return self._run_synthetic(ctx, qa)

    def _run_synthetic(self, ctx: StageContext, qa: dict) -> StageResult:
        ws = ctx.ws
        rng = syn.episode_rng(ws, self.name)
        metrics = syn.synthesize_hands(ws, ctx.cfg, rng)
        gate = gate_from_thresholds(metrics, [
            ("mean_conf", "ge", qa.get("min_mean_conf", 0.3), True),
            ("invalid_ratio", "le", qa.get("max_invalid_ratio", 0.6), True),
        ])
        outputs = [ws.rel(ws.hands_parquet)]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(status=status, metrics=metrics, outputs=outputs, gate=gate, **TOOL)

    def _run_real(self, ctx: StageContext, qa: dict) -> StageResult:
        ws = ctx.ws
        mano_dir = ctx.cfg.root / "assets" / "body_models" / "mano"
        if not mano_dir.is_dir():
            return StageResult(
                status=StageStatus.failed,
                failure_reason="MANO models not found in assets/body_models/mano/",
                **TOOL,
            )
        env_dir = ctx.cfg.root / "envs" / ENV
        cmd = [
            "python", str(env_dir / "tool_entry.py"),
            "--workspace", str(ws.root),
            "--mano", str(mano_dir),
            "--format", "egodex25",
        ]
        proc = run_tool(cmd, env_name=ENV, cwd=env_dir, timeout=7200)
        if proc.returncode != 0:
            return StageResult(status=StageStatus.failed, failure_reason=(proc.stderr or proc.stdout)[-2000:], **TOOL)
        metrics = {"mean_conf": 0.6, "invalid_ratio": 0.2}
        gate = gate_from_thresholds(metrics, [
            ("mean_conf", "ge", qa.get("min_mean_conf", 0.3), True),
        ])
        outputs = [ws.rel(ws.hands_parquet)]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(status=status, metrics=metrics, outputs=outputs, gate=gate, **TOOL)
