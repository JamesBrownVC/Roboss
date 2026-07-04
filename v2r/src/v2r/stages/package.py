"""Package (Stage J): LeRobot v3 + EgoDex mirror export."""

from __future__ import annotations

from ..export.package_writer import write_exports
from ..qa.license_audit import generate_license_audit
from ..schema.io import read_json_model
from ..schema.models import ConsentRecord, StageStatus
from .base import Stage, StageContext, StageResult, register_stage


@register_stage
class PackageStage(Stage):
    name = "package"

    def run(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        export_cfg = ctx.cfg.qa.get("export", {})
        if export_cfg.get("require_consent", True):
            if not ws.consent_path.is_file():
                return StageResult(
                    status=StageStatus.failed,
                    failure_reason="consent.json missing — export blocked",
                    tool="lerobot",
                    repo="github.com/huggingface/lerobot",
                    commit="0.1.0",
                )
            consent = read_json_model(ws.consent_path, ConsentRecord)
            if not consent.consent_id:
                return StageResult(
                    status=StageStatus.failed,
                    failure_reason="invalid consent record",
                    tool="lerobot",
                    repo="github.com/huggingface/lerobot",
                    commit="0.1.0",
                )

        generate_license_audit(ctx.cfg.root, ctx.cfg)
        outputs = write_exports(ws, ctx.cfg, ctx.robots, synthetic=(ctx.mode == "synthetic"))
        return StageResult(
            status=StageStatus.success,
            metrics={"n_robots": len(ctx.robots)},
            outputs=outputs,
            tool="lerobot",
            repo="github.com/huggingface/lerobot",
            commit="0.1.0",
        )
