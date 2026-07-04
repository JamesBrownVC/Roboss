"""Retarget (Stage H): GMR (humanoid), mink (manipulator), quadruped adapter."""

from __future__ import annotations

from ..schema.models import StageStatus
from . import _synthetic as syn
from .base import Stage, StageContext, StageResult, register_stage, run_tool

TOOL = {
    "tool": "GMR",
    "repo": "github.com/YanjieZe/GMR",
    "commit": "d4e5f6789012345678901234567890abcdef1234",
}
ENV = "gmr"


@register_stage
class RetargetStage(Stage):
    name = "retarget"

    def run(self, ctx: StageContext) -> StageResult:
        if not ctx.robots:
            return StageResult(
                status=StageStatus.failed,
                failure_reason="no robots specified (--robots)",
                **TOOL,
            )
        outputs: list[str] = []
        metrics: dict = {"robots": ctx.robots}
        for robot in ctx.robots:
            if ctx.mode == "real":
                err = self._run_real_robot(ctx, robot)
                if err:
                    return StageResult(status=StageStatus.failed, failure_reason=err, **TOOL)
            else:
                syn.synthesize_retarget(ctx.ws, ctx.cfg, robot, syn.episode_rng(ctx.ws, f"{self.name}:{robot}"))
            outputs.extend([
                ctx.ws.rel(ctx.ws.qpos_parquet(robot)),
                ctx.ws.rel(ctx.ws.ee_parquet(robot)),
                ctx.ws.rel(ctx.ws.mapping_json(robot)),
            ])
        return StageResult(status=StageStatus.success, metrics=metrics, outputs=outputs, **TOOL)

    def _run_real_robot(self, ctx: StageContext, robot: str) -> str | None:
        ws = ctx.ws
        env_dir = ctx.cfg.root / "envs" / ENV
        spec = ctx.cfg.robot(robot)
        cmd = [
            "python", str(env_dir / "tool_entry.py"),
            "--workspace", str(ws.root),
            "--robot", robot,
            "--robot-class", spec.robot_class.value,
            "--smplx", str(ws.smplx_npz),
        ]
        proc = run_tool(cmd, env_name=ENV, cwd=env_dir, timeout=7200)
        if proc.returncode != 0:
            return (proc.stderr or proc.stdout)[-2000:]
        return None
