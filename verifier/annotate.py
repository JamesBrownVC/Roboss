"""Semantic annotator — dataset enrichment, not a gate.

Runs in parallel with the labeling stages: while gate 2 judges, this module
describes. It receives sampled frames, the planned scenario packet and a
grounding summary of gate-1 evidence, and produces structured semantic text
(captions, action phases, interactions, risk states, outcome, QA pairs)
aligned with the physical labels on the same timeline.

The model writes the text; `normalize_annotation` (pure Python, unit-tested)
enforces the structure: ordered phases, times clamped to the clip, fixed
outcome vocabulary.
"""

from __future__ import annotations

import json

import numpy as np

from env_loader import load_dotenv

from .config import Thresholds
from .gate2 import sample_frames
from .tracks import Evidence, Violation

OUTCOME_VOCAB = ["success", "failure", "near_miss", "unclear"]

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "global_caption": {"type": "string"},
        "scene_context": {
            "type": "object",
            "properties": {
                "environment": {"type": "string"},
                "lighting": {"type": "string"},
                "surface": {"type": "string"},
                "notable_conditions": {"type": "string"},
            },
            "required": ["environment", "lighting", "surface"],
        },
        "actors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "role": {"type": "string"},
                    "appearance": {"type": "string"},
                    "behavior_summary": {"type": "string"},
                },
                "required": ["ref", "role", "behavior_summary"],
            },
        },
        "action_phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["t_start", "t_end", "label", "description"],
            },
        },
        "interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "type": {"type": "string"},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["actors", "type", "description"],
            },
        },
        "risk_assessment": {
            "type": "object",
            "properties": {
                "risk_states": {"type": "array", "items": {"type": "string"}},
                "hazards": {"type": "array", "items": {"type": "string"}},
                "expected_robot_response": {"type": "string"},
            },
            "required": ["risk_states", "hazards"],
        },
        "outcome": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "enum": OUTCOME_VOCAB},
                "description": {"type": "string"},
            },
            "required": ["result", "description"],
        },
        "prompt_alignment": {
            "type": "object",
            "properties": {
                "matches_scenario": {"type": "boolean"},
                "deviations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["matches_scenario", "deviations"],
        },
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        },
    },
    "required": ["global_caption", "scene_context", "actors",
                 "action_phases", "risk_assessment", "outcome",
                 "prompt_alignment"],
}

SYSTEM_PROMPT = """\
You are a semantic annotator for a robotics training dataset. You receive
frames sampled from one AI-generated action video, the scenario it was
generated from, and a summary of tracked physical evidence. Produce rich,
factual, structured annotations that will sit NEXT TO the physical labels
(keypoints, trajectories) on the same timeline.

Rules:
- Describe only what is visible in the frames; use the scenario text to
  name things, not to invent events that are not shown.
- action_phases: 3-7 contiguous phases covering the whole clip, snake_case
  labels (e.g. robot_carrying_box, human_crossing_path), times in seconds
  taken from the frame labels.
- interactions: agent-agent and agent-object interactions (proximity,
  contact, handover, collision_risk ...), with time ranges when visible.
- risk_states / hazards: short snake_case states an annotator should be
  able to defend by pointing at frames.
- expected_robot_response: what a well-behaved robot should do in this
  situation (one short imperative phrase).
- prompt_alignment: does the video actually depict the planned scenario?
  List concrete deviations.
- qa_pairs: 3-5 question-answer pairs a robotics VLM could be evaluated
  on using only this video.
Write in English. Be specific; avoid hedging filler."""


def evidence_summary(evidence: Evidence,
                     violations: list[Violation]) -> dict:
    """Compact gate-1 grounding for the annotator prompt."""
    return {
        "video_seconds": round(evidence.n_frames / evidence.fps, 2),
        "fps": round(evidence.fps, 2),
        "humans_tracked": len(evidence.person_tracks),
        "objects_tracked": sorted({t.label for t in evidence.object_tracks}),
        "physics_violations": [
            {"type": v.type, "frames": [v.frames[0], v.frames[-1]]}
            for v in violations[:8]
        ],
    }


def build_user_content(frames: list[tuple[int, float, bytes]],
                       scenario: dict | None,
                       grounding: dict | None) -> list[tuple[str, object]]:
    """Same (kind, value) convention as gate2.build_user_content."""
    content: list[tuple[str, object]] = []
    intro = ["Annotate this AI-generated video from the sampled frames."]
    if scenario:
        planned = {k: scenario[k] for k in
                   ("scenario_prompt", "expected_action", "expected_objects",
                    "expected_outcome") if k in scenario}
        if planned:
            intro.append("Planned scenario: " + json.dumps(planned) + ".")
    if grounding:
        intro.append("Tracked physical evidence (ground your times and "
                     "actors in this): " + json.dumps(grounding) + ".")
    content.append(("text", " ".join(intro)))

    for frame_idx, ts, jpeg in frames:
        content.append(("text", f"Frame {frame_idx} (t={ts:.2f}s):"))
        content.append(("image/jpeg", jpeg))

    content.append(("text", "Produce the full structured annotation."))
    return content


def _clean_span(item: dict, duration: float) -> dict | None:
    """Clamp a timed item to [0, duration]; None if it has no valid span."""
    try:
        t0 = max(0.0, float(item.get("t_start", 0.0)))
        t1 = min(duration, float(item.get("t_end", duration)))
    except (TypeError, ValueError):
        return None
    if t1 <= t0:
        return None
    out = dict(item)
    out["t_start"], out["t_end"] = round(t0, 2), round(t1, 2)
    return out


def normalize_annotation(data: dict, duration: float) -> dict:
    """Deterministic cleanup: the LLM writes, Python enforces structure."""
    out = dict(data)

    phases = []
    for p in data.get("action_phases", []):
        cleaned = _clean_span(p, duration)
        if cleaned and str(cleaned.get("label", "")).strip():
            cleaned["label"] = str(cleaned["label"]).strip()
            phases.append(cleaned)
    out["action_phases"] = sorted(phases, key=lambda p: p["t_start"])

    interactions = []
    for it in data.get("interactions", []):
        if not it.get("actors") or not str(it.get("type", "")).strip():
            continue
        if "t_start" in it or "t_end" in it:
            span = _clean_span(it, duration)
            interactions.append(span if span else
                                {k: v for k, v in it.items()
                                 if k not in ("t_start", "t_end")})
        else:
            interactions.append(dict(it))  # untimed stays untimed
    out["interactions"] = interactions

    risk = dict(data.get("risk_assessment", {}))
    risk["risk_states"] = sorted({str(s).strip() for s in
                                  risk.get("risk_states", []) if str(s).strip()})
    risk["hazards"] = [str(h).strip() for h in risk.get("hazards", [])
                       if str(h).strip()]
    out["risk_assessment"] = risk

    outcome = dict(data.get("outcome", {}))
    if outcome.get("result") not in OUTCOME_VOCAB:
        outcome["result"] = "unclear"
    out["outcome"] = outcome

    out["qa_pairs"] = [q for q in data.get("qa_pairs", [])
                       if isinstance(q, dict)
                       and str(q.get("question", "")).strip()
                       and str(q.get("answer", "")).strip()]
    return out


def run_annotator(video_path: str,
                  evidence: Evidence,
                  violations: list[Violation],
                  scenario: dict | None,
                  th: Thresholds) -> tuple[dict | None, dict]:
    """Returns (normalized annotation or None, metadata for the report)."""
    from google import genai
    from google.genai import errors, types

    frames = sample_frames(video_path, evidence.n_frames, th.annotate_frames,
                           suspicious=[], max_side=th.gate2_max_side,
                           fps=evidence.fps)
    if not frames:
        return None, {"status": "skipped",
                      "error": "no frames could be sampled"}

    grounding = evidence_summary(evidence, violations)
    parts = []
    for kind, value in build_user_content(frames, scenario, grounding):
        if kind == "text":
            parts.append(types.Part.from_text(text=str(value)))
        else:
            parts.append(types.Part.from_bytes(data=value, mime_type=kind))

    meta = {"status": "ok", "model": th.annotate_model,
            "frames_sent": [f for f, _, _ in frames]}
    try:
        load_dotenv()
        client = genai.Client()
        response = client.models.generate_content(
            model=th.annotate_model,
            contents=types.Content(role="user", parts=parts),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=16000,
                response_mime_type="application/json",
                response_schema=ANNOTATION_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
    # TypeError: the SDK raises it when no credentials can be resolved.
    except (errors.APIError, RuntimeError, TypeError, ValueError) as e:
        return None, {"status": "error", "error": f"{type(e).__name__}: {e}"}

    if response.text is None:
        return None, {"status": "error", "error": "empty model response"}
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        return None, {"status": "error", "error": f"invalid JSON: {e}"}

    duration = evidence.n_frames / evidence.fps
    annotation = normalize_annotation(data, round(duration, 2))
    return annotation, meta
