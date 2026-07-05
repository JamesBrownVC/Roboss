"""Unit tests for the semantic annotator's deterministic layer.

No Gemini calls: normalize_annotation and evidence_summary are pure
functions, so the structure enforcement is tested with fabricated data.
"""

from verifier.annotate import (
    evidence_summary,
    normalize_annotation,
)
from verifier.tracks import Evidence, Track, Violation

DURATION = 6.0


def test_phases_sorted_and_clamped():
    data = {
        "action_phases": [
            {"t_start": 3.0, "t_end": 9.5, "label": "robot_stops",
             "description": "robot halts"},          # t_end beyond clip
            {"t_start": 0.0, "t_end": 1.5, "label": "robot_carrying_box",
             "description": "robot walks"},
            {"t_start": 2.0, "t_end": 1.0, "label": "impossible",
             "description": "inverted span"},        # dropped
            {"t_start": 1.5, "t_end": 3.0, "label": "  ",
             "description": "empty label"},          # dropped
        ],
    }
    out = normalize_annotation(data, DURATION)
    labels = [p["label"] for p in out["action_phases"]]
    assert labels == ["robot_carrying_box", "robot_stops"]
    assert out["action_phases"][-1]["t_end"] == DURATION


def test_interactions_keep_untimed_entries():
    data = {
        "interactions": [
            {"actors": ["robot", "human"], "type": "unsafe_proximity",
             "t_start": 1.0, "t_end": 2.0, "description": "close pass"},
            {"actors": ["robot", "box"], "type": "carrying",
             "description": "no explicit time"},
            {"actors": [], "type": "ghost", "description": "no actors"},
        ],
    }
    out = normalize_annotation(data, DURATION)
    assert len(out["interactions"]) == 2
    assert out["interactions"][0]["t_end"] == 2.0
    assert "t_start" not in out["interactions"][1]


def test_risks_deduped_and_outcome_vocab_enforced():
    data = {
        "risk_assessment": {"risk_states": ["fall_risk", "fall_risk", " "],
                            "hazards": ["wet floor", ""]},
        "outcome": {"result": "kind_of_ok", "description": "?"},
        "qa_pairs": [
            {"question": "What happens?", "answer": "A slip."},
            {"question": "", "answer": "dropped"},
            "not even a dict",
        ],
    }
    out = normalize_annotation(data, DURATION)
    assert out["risk_assessment"]["risk_states"] == ["fall_risk"]
    assert out["risk_assessment"]["hazards"] == ["wet floor"]
    assert out["outcome"]["result"] == "unclear"
    assert len(out["qa_pairs"]) == 1


def test_evidence_summary_grounding():
    person = Track(track_id=1, label="person", is_person=True)
    person.add(0, (0, 0, 10, 10))
    box = Track(track_id=2, label="box", is_person=False)
    box.add(0, (5, 5, 8, 8))
    ev = Evidence(video_path="x.mp4", fps=30.0, width=1280, height=720,
                  n_frames=90, person_tracks=[person], object_tracks=[box])
    v = [Violation(type="foot_skate", severity=0.6, frames=[10, 11, 12],
                   reason="slide")]
    summary = evidence_summary(ev, v)
    assert summary["video_seconds"] == 3.0
    assert summary["humans_tracked"] == 1
    assert summary["objects_tracked"] == ["box"]
    assert summary["physics_violations"] == [
        {"type": "foot_skate", "frames": [10, 12]}]
