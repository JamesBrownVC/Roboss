"""Yield report generator (funnel counts per stage)."""

from __future__ import annotations

from ..config import V2RConfig
from ..schema.io import read_json_model
from ..schema.models import StageManifest, StageStatus
from ..schema.workspace import EpisodeWorkspace


def write_yield_report(ws: EpisodeWorkspace, cfg: V2RConfig, robots: list[str]) -> None:
    lines = [
        f"# Yield Report — {ws.episode_id}",
        "",
        "| Stage | Status | Key metric |",
        "|-------|--------|------------|",
    ]
    funnel_stages = [
        "ingest", "feasibility_judge", "geometry", "human_body", "hands", "objects",
        "contact", "semantics", "retarget", "physics_validate", "qa", "package",
    ]
    for stage in funnel_stages:
        mpath = ws.manifest_path(stage)
        if mpath.is_file():
            m = read_json_model(mpath, StageManifest)
            metric = next(iter(m.metrics.values()), "—") if m.metrics else "—"
            lines.append(f"| {stage} | {m.status.value} | {metric} |")
        else:
            lines.append(f"| {stage} | pending | — |")

    if ws.feasibility_report_json.is_file():
        from ..schema.models import FeasibilityReport
        fr = read_json_model(ws.feasibility_report_json, FeasibilityReport)
        lines.extend([
            "",
            "## Feasibility gate (pre-analysis QA)",
            "",
            f"- recommendation: **{fr.recommendation.value}**",
            f"- confidence: {fr.confidence:.2f}",
            f"- physics_violation_frame_ratio: {fr.physics_violation_frame_ratio:.3f}",
            f"- physically_plausible: {fr.physically_plausible}",
            f"- judge_source: {fr.judge_source}",
        ])
        if fr.ai_generated_artifacts:
            lines.append(f"- ai_artifacts: {', '.join(fr.ai_generated_artifacts)}")

    lines.extend(["", "## Retarget / physics by robot", ""])
    for robot in robots:
        for suffix in ("retarget", "physics_validate"):
            mpath = ws.manifest_path(suffix)
            status = "—"
            if mpath.is_file():
                status = read_json_model(mpath, StageManifest).status.value
            phys = ws.physics_report_json(robot)
            phys_ok = "—"
            if phys.is_file():
                from ..schema.models import PhysicsReport
                pr = read_json_model(phys, PhysicsReport)
                phys_ok = str(pr.physics_valid)
            lines.append(f"- **{robot}**: retarget={status}, physics_valid={phys_ok}")

    ws.yield_report_md.parent.mkdir(parents=True, exist_ok=True)
    ws.yield_report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
