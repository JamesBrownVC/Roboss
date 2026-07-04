"""Semantics (Stage G): Qwen-VL + changepoint subtask segmentation."""

from __future__ import annotations

from ..schema.models import StageStatus
from . import _synthetic as syn
from .base import Stage, StageContext, StageResult, register_stage, run_tool

TOOL = {
    "tool": "Qwen2.5-VL",
    "repo": "huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    "commit": "v1.0.0-pinned",
}
ENV = "semantics"


@register_stage
class SemanticsStage(Stage):
    name = "semantics"

    def run(self, ctx: StageContext) -> StageResult:
        if ctx.mode == "real":
            return self._run_real(ctx)
        return self._run_synthetic(ctx)

    def _run_synthetic(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        syn.synthesize_semantics(ws, ctx.cfg)
        outputs = [
            ws.rel(ws.segments_json),
            ws.rel(ws.captions_json),
            ws.rel(ws.scene_tags_json),
        ]
        return StageResult(
            status=StageStatus.success,
            metrics={"n_segments": 1},
            outputs=outputs,
            **TOOL,
        )

    def _run_real(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        env_dir = ctx.cfg.root / "envs" / ENV
        verbs_path = ctx.cfg.root / "config" / "verbs.yaml"
        cmd = [
            "python", str(env_dir / "tool_entry.py"),
            "--workspace", str(ws.root),
            "--verbs", str(verbs_path),
            "--temperature", "0",
        ]
        proc = run_tool(cmd, env_name=ENV, cwd=env_dir, timeout=7200)
        if proc.returncode != 0:
            return StageResult(status=StageStatus.failed, failure_reason=(proc.stderr or proc.stdout)[-2000:], **TOOL)
        outputs = [ws.rel(ws.segments_json), ws.rel(ws.captions_json), ws.rel(ws.scene_tags_json)]
        return StageResult(status=StageStatus.success, metrics={"n_segments": 1}, outputs=outputs, **TOOL)
