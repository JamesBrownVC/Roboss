"""Gate 2 — semantic reviewer.

Sends sampled video frames plus the gate-1 findings to a multimodal Gemini
model and asks it to flag impossibilities the rule engine cannot see:
extra limbs, morphing objects, magic effects, impossible gestures, scene
teleports, prompt mismatch. Structured outputs (`response_schema`)
guarantee the reply is valid JSON matching our schema — the model cannot
free-form its verdict.

Gate 2 is advisory on top of gate 1: its findings are merged into the same
violation list with `gate: "semantic"` and scored with the same formula.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from env_loader import load_dotenv

from .config import Thresholds
from .tracks import Evidence, Violation

SEMANTIC_TYPES = [
    "anatomical_anomaly",     # extra/missing/merged limbs, impossible poses
    "object_morphing",        # object changes identity, shape or count
    "magic_effect",           # glow, energy beams, things moved by no force
    "impossible_gesture",     # hands/body doing physically impossible motion
    "scene_inconsistency",    # background/lighting/scale breaks continuity
    "prompt_mismatch",        # video does not show the requested scenario
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": SEMANTIC_TYPES},
                    "severity": {"type": "number"},
                    "frame_numbers": {
                        "type": "array", "items": {"type": "integer"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["type", "severity", "frame_numbers", "reason"],
            },
        },
        "semantic_score": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["violations", "semantic_score", "summary"],
}

SYSTEM_PROMPT = """\
You are the semantic verification gate in a pipeline that filters \
AI-generated action videos before they become robotics training data. \
A deterministic rule engine (gate 1) has already checked trajectories, \
contacts and gravity from tracked keypoints. Your job is to catch what \
rules cannot see, by looking at the actual frames:

- anatomical_anomaly: extra, missing or merged limbs; heads/hands fused \
with objects; anatomically impossible poses
- object_morphing: an object changing identity, shape, size or count \
between frames (a box becoming a bag, one cup becoming two)
- magic_effect: glows, energy effects, objects moved by no visible force
- impossible_gesture: hand or body motion no human could perform
- scene_inconsistency: background, lighting, scale or perspective breaking \
between frames
- prompt_mismatch: the frames clearly do not depict the requested scenario

Be conservative: report only what you can point to in specific frames. \
Ordinary generated-video softness, compression artifacts or motion blur \
are NOT violations. severity is 0.0-1.0 (above 0.85 means definitely \
impossible). semantic_score is your overall plausibility judgement, \
1.0 = fully plausible. Frame numbers must be taken from the labels that \
precede each image."""


def sample_frames(video_path: str, n_frames: int, k: int,
                  suspicious: list[int], max_side: int,
                  fps: float) -> list[tuple[int, float, bytes]]:
    """Uniform K frames plus up to 4 gate-1 suspicious frames.

    Returns (frame_index, timestamp_s, jpeg_bytes) sorted by frame index.
    """
    picks = set(np.linspace(0, max(n_frames - 1, 0), num=min(k, n_frames),
                            dtype=int).tolist())
    for f in sorted(set(suspicious))[:4]:
        if 0 <= f < n_frames:
            picks.add(int(f))

    cap = cv2.VideoCapture(video_path)
    out = []
    for f in sorted(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append((f, f / fps, buf.tobytes()))
    cap.release()
    return out


def build_user_content(frames: list[tuple[int, float, bytes]],
                       scenario: dict | None,
                       gate1_violations: list[Violation]) -> list[tuple[str, object]]:
    content: list[tuple[str, object]] = []
    prompt_text = (scenario or {}).get("scenario_prompt")
    intro = ["Review these frames sampled from one AI-generated video."]
    if prompt_text:
        intro.append(f"The video was generated from the prompt: \"{prompt_text}\".")
    if gate1_violations:
        findings = "; ".join(
            f"{v.type} at frames {v.frames[0]}..{v.frames[-1]}"
            for v in gate1_violations[:6])
        intro.append(f"Gate 1 (rule engine) already flagged: {findings}. "
                     f"Confirm or refute what you can see, and look for "
                     f"anything it missed.")
    content.append(("text", " ".join(intro)))

    for frame_idx, ts, jpeg in frames:
        content.append(("text", f"Frame {frame_idx} (t={ts:.2f}s):"))
        content.append(("image/jpeg", jpeg))

    content.append(("text", "Report all semantic/physical impossibilities "
                            "you can point to in these frames."))
    return content


def build_gemini_parts(frames: list[tuple[int, float, bytes]],
                       scenario: dict | None,
                       gate1_violations: list[Violation]) -> list[object]:
    """Gemini content parts for sampled frames and review instructions."""
    from google.genai import types

    parts = []
    for kind, value in build_user_content(frames, scenario, gate1_violations):
        if kind == "text":
            parts.append(types.Part.from_text(text=str(value)))
        else:
            parts.append(types.Part.from_bytes(data=value, mime_type=kind))
    return parts


def parse_gate2_response(data: dict) -> list[Violation]:
    """Pure and unit-testable: model JSON -> Violation list."""
    violations = []
    for v in data.get("violations", []):
        vtype = v.get("type")
        if vtype not in SEMANTIC_TYPES:
            continue
        sev = float(np.clip(float(v.get("severity", 0.5)), 0.0, 1.0))
        frames = sorted({int(f) for f in v.get("frame_numbers", []) if int(f) >= 0})
        violations.append(Violation(
            type=vtype,
            severity=sev,
            frames=frames or [0],
            reason=str(v.get("reason", "")).strip(),
            gate="semantic",
        ))
    violations.sort(key=lambda x: x.severity, reverse=True)
    return violations


def run_gate2(video_path: str,
              evidence: Evidence,
              gate1_violations: list[Violation],
              scenario: dict | None,
              th: Thresholds) -> tuple[list[Violation], dict]:
    """Returns (semantic violations, gate2 metadata for the report)."""
    from google import genai
    from google.genai import errors, types

    suspicious = sorted({f for v in gate1_violations for f in v.frames})
    frames = sample_frames(video_path, evidence.n_frames, th.gate2_frames,
                           suspicious, th.gate2_max_side, evidence.fps)
    if not frames:
        return [], {"status": "skipped", "error": "no frames could be sampled"}

    meta = {"status": "ok", "model": th.gate2_model,
            "frames_sent": [f for f, _, _ in frames]}
    try:
        load_dotenv()
        client = genai.Client()
        response = client.models.generate_content(
            model=th.gate2_model,
            contents=types.Content(
                role="user",
                parts=build_gemini_parts(frames, scenario, gate1_violations),
            ),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=8000,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
    # TypeError: the SDK raises it when no credentials can be resolved.
    except (errors.APIError, RuntimeError, TypeError, ValueError) as e:
        return [], {"status": "error", "error": f"{type(e).__name__}: {e}"}

    text = response.text
    if text is None:
        return [], {"status": "error", "error": "empty model response"}
    data = json.loads(text)

    violations = parse_gate2_response(data)
    meta["semantic_score"] = round(
        float(np.clip(float(data.get("semantic_score", 1.0)), 0.0, 1.0)), 2)
    meta["summary"] = data.get("summary", "")
    return violations, meta
