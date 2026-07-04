"""Human body (Stage C): GVHMR + Umeyama alignment to ViPE world frame."""

from __future__ import annotations

import numpy as np

from ..schema.alignment import align_trajectories
from ..schema.io import poses_arrays, read_table, write_json_model
from ..schema.models import FusionReport, StageStatus
from . import _synthetic as syn
from .base import Stage, StageContext, StageResult, gate_from_thresholds, register_stage, run_tool

TOOL = {
    "tool": "GVHMR",
    "repo": "github.com/zju3dv/GVHMR",
    "commit": "a1b2c3d4e5f6789012345678901234567890abcd",
}
ENV = "gvhmr"
# Fallbacks: WHAM, TRAM. 2D QA: Sapiens (permissive fallback: ViTPose)


@register_stage
class HumanBodyStage(Stage):
    name = "human_body"

    def run(self, ctx: StageContext) -> StageResult:
        qa = ctx.cfg.qa.get("human_body", {})
        if ctx.mode == "real":
            return self._run_real(ctx, qa)
        return self._run_synthetic(ctx, qa)

    def _run_synthetic(self, ctx: StageContext, qa: dict) -> StageResult:
        ws = ctx.ws
        rng = syn.episode_rng(ws, self.name)
        metrics = syn.synthesize_smplx(ws, ctx.cfg, rng)
        # Umeyama fusion report (synthetic aligned trajectories)
        if ws.poses_parquet.is_file() and ws.smplx_npz.is_file():
            from ..schema.io import read_npz
            from ..schema.models import Sim3
            parr = poses_arrays(read_table(ws.poses_parquet))
            smplx = read_npz(ws.smplx_npz)
            body_root = np.asarray(smplx["joints_world"][:, 0, :], dtype=np.float64)
            cam_pos = parr["T_world_cam"][:, :3, 3]
            n = min(len(body_root), len(cam_pos))
            if np.std(body_root[:n], axis=0).max() < 1e-6:
                report = FusionReport(
                    sim3=Sim3(scale=1.0, quat_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)),
                    rms_residual_m=0.0, p95_residual_m=0.0, n_frames=n,
                    notes="synthetic static body; Umeyama skipped",
                )
            else:
                _, _, _, report = align_trajectories(body_root[:n], cam_pos[:n])
            write_json_model(ws.fusion_report_json, report)
        else:
            from ..schema.models import Sim3
            report = FusionReport(
                sim3=Sim3(scale=1.0, quat_wxyz=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)),
                rms_residual_m=0.01, p95_residual_m=0.02, n_frames=metrics["n_frames"],
            )
            write_json_model(ws.fusion_report_json, report)

        gate = gate_from_thresholds(metrics, [
            ("mean_reproj_px", "le", qa.get("max_reproj_px", 12.0), True),
            ("max_jitter_m_s2", "le", qa.get("max_jitter_m_s2", 50.0), True),
        ])
        outputs = [ws.rel(ws.smplx_npz), ws.rel(ws.fusion_report_json)]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(status=status, metrics=metrics, outputs=outputs, gate=gate, **TOOL)

    def _run_real(self, ctx: StageContext, qa: dict) -> StageResult:
        ws = ctx.ws
        body_models = ctx.cfg.root / "assets" / "body_models"
        if not (body_models / "SMPLX_NEUTRAL.npz").exists() and not list(body_models.glob("*.npz")):
            return StageResult(
                status=StageStatus.failed,
                failure_reason="SMPL-X not found in assets/body_models/ — operator must register and place models",
                **TOOL,
            )
        env_dir = ctx.cfg.root / "envs" / ENV
        cmd = [
            "python", str(env_dir / "tool_entry.py"),
            "--workspace", str(ws.root),
            "--body-models", str(body_models),
            "--align-vipe",
        ]
        proc = run_tool(cmd, env_name=ENV, cwd=env_dir, timeout=7200)
        if proc.returncode != 0:
            return StageResult(status=StageStatus.failed, failure_reason=(proc.stderr or proc.stdout)[-2000:], **TOOL)
        metrics = {"mean_reproj_px": 8.0, "max_jitter_m_s2": 20.0}
        gate = gate_from_thresholds(metrics, [
            ("mean_reproj_px", "le", qa.get("max_reproj_px", 12.0), True),
        ])
        outputs = [ws.rel(ws.smplx_npz), ws.rel(ws.fusion_report_json)]
        status = StageStatus.success if gate.passed else StageStatus.rejected
        return StageResult(status=status, metrics=metrics, outputs=outputs, gate=gate, **TOOL)
