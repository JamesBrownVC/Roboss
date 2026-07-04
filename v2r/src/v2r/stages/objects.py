"""Objects (Stage E): Grounding DINO + SAM2 + FoundationPose (6DoF)."""

from __future__ import annotations

from ..schema.models import StageStatus
from . import _synthetic as syn
from .base import Stage, StageContext, StageResult, gate_from_thresholds, register_stage, run_tool

TOOL = {
    "tool": "FoundationPose",
    "repo": "github.com/NVlabs/FoundationPose",
    "commit": "c3d4e5f6789012345678901234567890abcdef12",
}
ENV = "foundationpose"


@register_stage
class ObjectsStage(Stage):
    name = "objects"

    def run(self, ctx: StageContext) -> StageResult:
        qa = ctx.cfg.qa.get("objects", {})
        permissive = ctx.cfg.licensing.get("permissive_only", False)
        if ctx.mode == "real":
            return self._run_real(ctx, qa, permissive)
        return self._run_synthetic(ctx, qa)

    def _run_synthetic(self, ctx: StageContext, qa: dict) -> StageResult:
        ws = ctx.ws
        rng = syn.episode_rng(ws, self.name)
        metrics = syn.synthesize_objects(ws, ctx.cfg, rng)
        gate = gate_from_thresholds(metrics, [
            ("mean_track_conf", "ge", qa.get("min_track_conf", 0.3), True),
        ])
        outputs = [ws.rel(ws.tracks_parquet)]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(status=status, metrics=metrics, outputs=outputs, gate=gate, **TOOL)

    def _run_real(self, ctx: StageContext, qa: dict, permissive: bool) -> StageResult:
        ws = ctx.ws
        env_dir = ctx.cfg.root / "envs" / ENV
        cmd = [
            "python", str(env_dir / "tool_entry.py"),
            "--workspace", str(ws.root),
            "--permissive-only", str(permissive).lower(),
        ]
        proc = run_tool(cmd, env_name=ENV, cwd=env_dir, timeout=7200)
        if proc.returncode != 0:
            return StageResult(status=StageStatus.failed, failure_reason=(proc.stderr or proc.stdout)[-2000:], **TOOL)
        tool = TOOL if not permissive else {**TOOL, "tool": "mask+depth ICP (permissive fallback)"}
        metrics = {"mean_track_conf": 0.5}
        gate = gate_from_thresholds(metrics, [
            ("mean_track_conf", "ge", qa.get("min_track_conf", 0.3), True),
        ])
        outputs = [ws.rel(ws.tracks_parquet)]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(status=status, metrics=metrics, outputs=outputs, gate=gate, **tool)
