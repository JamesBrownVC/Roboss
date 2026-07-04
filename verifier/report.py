"""Assemble the final structured plausibility report."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Thresholds
from .scoring import decide
from .tracks import Evidence, Violation


def _scenario_check(evidence: Evidence, scenario: dict | None) -> dict:
    """Compare detected classes against the metadata packet from the
    video-generation side (expected_objects etc.)."""
    detected = sorted({t.label for t in evidence.object_tracks})
    if evidence.person_tracks:
        detected.insert(0, "person")
    out: dict = {"detected_classes": detected}
    if not scenario:
        return out
    expected = [str(o).lower() for o in scenario.get("expected_objects", [])]
    missing = []
    for exp in expected:
        hit = any(exp in d or d in exp for d in detected)
        # "human"/"person" synonymy — the only one worth hardcoding
        if not hit and exp in ("human", "man", "woman"):
            hit = "person" in detected
        if not hit:
            missing.append(exp)
    out["expected_objects"] = expected
    out["missing_expected_objects"] = missing
    return out


def build_report(evidence: Evidence,
                 violations: list[Violation],
                 th: Thresholds,
                 scenario: dict | None = None,
                 gate2_meta: dict | None = None) -> dict:
    plausible, score, main_reason = decide(violations, th)
    scenario_prompt = (scenario or {}).get("scenario_prompt")

    report = {
        "video_id": Path(evidence.video_path).stem,
        "scenario": scenario_prompt,
        "plausible": plausible,
        "plausibility_score": round(score, 2),
        "decision": "accept" if plausible else "reject",
        "main_reason": main_reason,
        "violations": [v.to_dict() for v in violations],
        "suspicious_frames": sorted({f for v in violations for f in v.frames}),
        "extracted_evidence": {
            "fps": round(evidence.fps, 2),
            "frames_processed": evidence.n_frames,
            "humans_detected": len(evidence.person_tracks),
            "objects_detected": sorted({t.label for t in evidence.object_tracks}),
            "tracks": [
                {"id": t.track_id, "label": t.label, "frames_tracked": len(t)}
                for t in evidence.all_tracks
            ],
            **_scenario_check(evidence, scenario),
        },
        "gates": {
            "formal": {
                "violations": sum(1 for v in violations if v.gate == "formal"),
            },
            "semantic": gate2_meta or {"status": "not_run"},
        },
    }
    return report


def save_report(report: dict, path: str) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
