"""Pre-analysis feasibility: physics heuristics + optional LLM/VLM judge."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import V2RConfig
from ..schema.io import read_json_model, write_json_model, write_table
from ..schema.models import (
    FeasibilityRecommendation,
    FeasibilityReport,
    SourceTag,
    VideoProbe,
)
from ..schema.timeline import canonical_timestamps
from ..schema.workspace import EpisodeWorkspace
from ..stages.base import rng_for


def _physics_checks(
    ws: EpisodeWorkspace,
    cfg: V2RConfig,
    rng: np.random.Generator,
    mode: str,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    """Run physics-oriented heuristics; return (metrics, mask_df, violation_ratio)."""
    qa = cfg.qa.get("feasibility", {})
    max_vel = qa.get("max_joint_vel_m_s", 8.0)
    max_acc = qa.get("max_joint_acc_m_s2", 40.0)
    max_slide = qa.get("max_foot_slide_m_per_frame", 0.01)
    max_scale_jump = qa.get("max_scale_jump_ratio", 1.5)

    if ws.probe_path.is_file():
        probe = read_json_model(ws.probe_path, VideoProbe)
    else:
        probe = VideoProbe(width=1280, height=720, fps=30.0, n_frames=90, duration_s=3.0)

    t = canonical_timestamps(probe.duration_s, cfg.pipeline.canonical_hz)
    n = len(t)
    frames = np.arange(n, dtype=np.int64)

    if mode == "synthetic":
        # Deterministic per-episode: mostly plausible, occasional soft flags
        base_violation = float(rng.uniform(0.0, 0.08))
        vel_spikes = rng.random(n) < base_violation * 0.5
        slide_frames = rng.random(n) < base_violation * 0.3
        scale_jumps = rng.random(n) < 0.02
        flow_disagree = rng.random(n) < base_violation * 0.2
    else:
        vel_spikes = np.zeros(n, dtype=bool)
        slide_frames = np.zeros(n, dtype=bool)
        scale_jumps = np.zeros(n, dtype=bool)
        flow_disagree = np.zeros(n, dtype=bool)
        if ws.smplx_npz.is_file():
            from ..schema.io import read_npz
            data = read_npz(ws.smplx_npz)
            joints = data.get("joints_world")
            if joints is not None and len(joints) >= 3:
                dt = np.diff(t[: len(joints)])
                vel = np.linalg.norm(np.diff(joints, axis=0), axis=2) / dt[:, None]
                acc = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[1:, None]
                vel_spikes[: len(vel)] = vel.max(axis=1) > max_vel
                slide_frames[: len(acc)] = acc.max(axis=1) > max_acc

    physics_violation = vel_spikes | slide_frames | scale_jumps | flow_disagree
    violation_ratio = float(physics_violation.mean()) if n else 0.0

    conf = np.clip(1.0 - 0.5 * physics_violation.astype(np.float64), 0.1, 1.0)
    valid = ~physics_violation
    source = np.full(n, SourceTag.estimated.value if mode == "real" else SourceTag.synthesized.value)

    mask_df = pd.DataFrame({
        "t": t,
        "frame": frames,
        "valid": valid,
        "conf": conf,
        "source": source,
        "physics_violation": physics_violation,
        "vel_spike": vel_spikes,
        "foot_slide": slide_frames,
        "scale_jump": scale_jumps,
        "flow_pose_disagree": flow_disagree,
    })

    metrics = {
        "physics_violation_frame_ratio": violation_ratio,
        "vel_spike_ratio": float(vel_spikes.mean()) if n else 0.0,
        "foot_slide_ratio": float(slide_frames.mean()) if n else 0.0,
        "scale_jump_ratio": float(scale_jumps.mean()) if n else 0.0,
        "flow_disagree_ratio": float(flow_disagree.mean()) if n else 0.0,
        "max_joint_vel_threshold_m_s": max_vel,
        "max_foot_slide_threshold_m": max_slide,
        "max_scale_jump_ratio": max_scale_jump,
    }
    return metrics, mask_df, violation_ratio


def _mock_judge(
    physics: dict[str, Any],
    violation_ratio: float,
    rng: np.random.Generator,
) -> FeasibilityReport:
    """Deterministic synthetic judge."""
    artifacts: list[str] = []
    if violation_ratio > 0.05:
        artifacts.append("physics_violation")
    if physics.get("flow_disagree_ratio", 0) > 0.1:
        artifacts.append("temporal_flicker")
    if physics.get("scale_jump_ratio", 0) > 0.03:
        artifacts.append("limb_morphing")

    plausible = violation_ratio < 0.15
    tracking_ok = violation_ratio < 0.20
    confidence = float(np.clip(1.0 - violation_ratio * 2.0 + rng.uniform(-0.05, 0.05), 0.0, 1.0))

    if not plausible or violation_ratio > 0.35:
        rec = FeasibilityRecommendation.reject
    elif violation_ratio > 0.12 or not tracking_ok:
        rec = FeasibilityRecommendation.human_review
    else:
        rec = FeasibilityRecommendation.proceed

    return FeasibilityReport(
        physically_plausible=plausible,
        tracking_likely_valid=tracking_ok,
        ai_generated_artifacts=artifacts,
        confidence=confidence,
        recommendation=rec,
        physics_violation_frame_ratio=violation_ratio,
        physics_checks=physics,
        judge_source="synthetic",
        notes="Deterministic mock judge for synthetic/CI mode.",
    )


def _rule_based_judge(physics: dict[str, Any], violation_ratio: float, qa: dict) -> FeasibilityReport:
    """Fallback when VLM/API unavailable."""
    max_ratio = qa.get("physics_violation_frame_ratio_max", 0.25)
    artifacts = []
    if violation_ratio > 0.1:
        artifacts.append("physics_violation")
    plausible = violation_ratio <= max_ratio
    rec = (
        FeasibilityRecommendation.proceed if plausible
        else FeasibilityRecommendation.reject if violation_ratio > max_ratio * 1.5
        else FeasibilityRecommendation.human_review
    )
    return FeasibilityReport(
        physically_plausible=plausible,
        tracking_likely_valid=violation_ratio < max_ratio,
        ai_generated_artifacts=artifacts,
        confidence=float(max(0.0, 1.0 - violation_ratio * 3)),
        recommendation=rec,
        physics_violation_frame_ratio=violation_ratio,
        physics_checks=physics,
        judge_source="rule_based",
    )


def _api_judge(
    physics: dict[str, Any],
    violation_ratio: float,
    video_path: Optional[Path],
) -> Optional[FeasibilityReport]:
    """OpenAI-compatible API via V2R_JUDGE_API env var."""
    api_url = os.environ.get("V2R_JUDGE_API")
    if not api_url:
        return None
    payload = {
        "physics_checks": physics,
        "physics_violation_frame_ratio": violation_ratio,
        "video_path": str(video_path) if video_path else None,
        "schema": {
            "physically_plausible": "bool",
            "tracking_likely_valid": "bool",
            "ai_generated_artifacts": "list[str]",
            "confidence": "float 0-1",
            "recommendation": "proceed|reject|human_review",
        },
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return FeasibilityReport(
            physically_plausible=bool(data["physically_plausible"]),
            tracking_likely_valid=bool(data["tracking_likely_valid"]),
            ai_generated_artifacts=list(data.get("ai_generated_artifacts", [])),
            confidence=float(data.get("confidence", 0.5)),
            recommendation=FeasibilityRecommendation(data["recommendation"]),
            physics_violation_frame_ratio=violation_ratio,
            physics_checks=physics,
            judge_source="api",
        )
    except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
        return None


def run_feasibility_judge(
    ws: EpisodeWorkspace,
    cfg: V2RConfig,
    mode: str,
) -> tuple[FeasibilityReport, pd.DataFrame, bool]:
    """Run physics checks + judge; return (report, mask_df, gate_passed)."""
    qa = cfg.qa.get("feasibility", {})
    rng = rng_for(ws.episode_id, "feasibility_judge")

    physics, mask_df, violation_ratio = _physics_checks(ws, cfg, rng, mode)

    report: FeasibilityReport
    if mode == "synthetic":
        report = _mock_judge(physics, violation_ratio, rng)
    else:
        api_report = _api_judge(physics, violation_ratio, ws.video_path if ws.video_path.is_file() else None)
        if api_report is not None:
            report = api_report
        else:
            report = _rule_based_judge(physics, violation_ratio, qa)

    write_json_model(ws.feasibility_report_json, report)
    write_table(mask_df, ws.feasibility_mask_parquet, required_columns=["t", "frame", "valid", "conf", "source"])

    max_ratio = qa.get("physics_violation_frame_ratio_max", 0.25)
    gate_passed = (
        report.recommendation != FeasibilityRecommendation.reject
        and violation_ratio <= max_ratio
    )
    return report, mask_df, gate_passed
