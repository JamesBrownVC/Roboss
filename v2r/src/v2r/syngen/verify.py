"""Step 4 — agentic verification: two parallel tracks per generated video.

Track A  VLM judge: sampled frames -> Gemini multimodal generateContent with a
         structured verification schema (extends the qa/feasibility.py
         FeasibilityReport contract). Falls back to a deterministic
         heuristic judge when the API is unavailable.

Track B  Tool-based physics/math: optical-flow consistency, velocity spikes,
         scale jumps computed directly from decoded frames, plus MediaPipe
         pose-extraction sanity when the `timeseries` extra is installed
         (same model as the dev harness in timeseries/extract.py).

A third structured Gemini call labels the video (skill verbs from
config/verbs.yaml, scene tags, subject type); a keyword fallback keeps
labeling offline-safe. Both tracks merge into verification/{variant_id}.json
with a final verdict: accept | reject | review.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..schema.models import FeasibilityRecommendation, FeasibilityReport
from .gemini import GeminiError, extract_json, generate_content, have_api_key, image_part
from .spec import JobDirs, JobSpec, VariantSpec

VLM_TEMPERATURE = 0.2
N_JUDGE_FRAMES = 6


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SyngenLabels(BaseModel):
    skills: list[str] = Field(default_factory=list)   # from config/verbs.yaml
    scene_type: str = "unknown"
    lighting: str = "unknown"
    subject_type: str = "human"
    caption: str = ""
    source: str = "mock"                              # "gemini" | "mock"


class TrackBReport(BaseModel):
    n_frames: int = 0
    fps: float = 30.0
    flow_mean: float = 0.0
    velocity_spike_ratio: float = 0.0
    scale_jump_ratio: float = 0.0
    flow_consistency: float = 1.0        # 1 = smooth, 0 = chaotic
    pose_detection_rate: Optional[float] = None   # MediaPipe, when available
    pose_tool: str = "none"
    physics_ok: bool = True
    reasons: list[str] = Field(default_factory=list)


class VerificationRecord(BaseModel):
    variant_id: str
    event_id: str = ""
    cam_id: str = ""
    verdict: str = "review"              # accept | reject | review
    verdict_reasons: list[str] = Field(default_factory=list)
    vlm: Optional[FeasibilityReport] = None
    physics: Optional[TrackBReport] = None
    labels: Optional[SyngenLabels] = None


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def sample_frames(video: Path, n: int = N_JUDGE_FRAMES,
                  max_side: int = 640) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.unique(np.linspace(0, max(total - 1, 0), n).astype(int))
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        s = max_side / max(h, w)
        if s < 1.0:
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        frames.append(frame)
    cap.release()
    return frames


def _jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Track A: VLM judge
# ---------------------------------------------------------------------------

VLM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "physically_plausible": {"type": "BOOLEAN"},
        "subject_visible": {"type": "BOOLEAN"},
        "camera_consistent": {"type": "BOOLEAN",
                              "description": "Static/consistent camera, no cuts or jumps"},
        "artifacts": {"type": "ARRAY", "items": {"type": "STRING"},
                      "description": "e.g. limb_morphing, temporal_flicker, "
                                     "extra_limbs, object_teleport, texture_crawl"},
        "confidence": {"type": "NUMBER"},
        "recommendation": {"type": "STRING", "enum": ["proceed", "reject", "human_review"]},
        "notes": {"type": "STRING"},
    },
    "required": ["physically_plausible", "subject_visible", "camera_consistent",
                 "confidence", "recommendation"],
}

VLM_PROMPT = """These are {n} frames sampled uniformly from a single AI-generated video
intended as robot-training data for the task: "{prompt}"

Judge the video for training-data feasibility:
- physically_plausible: does the subject's motion obey physics across frames
  (no teleporting, limbs attached, consistent scale)?
- subject_visible: is the subject clearly visible in (nearly) all frames?
- camera_consistent: same static viewpoint throughout, no cuts?
- artifacts: list generation artifacts you observe.
- recommendation: proceed (usable), reject (unusable), human_review (borderline).
Be strict: pose trackers will run on this video."""


def vlm_judge(
    video: Path,
    variant: VariantSpec,
    log: Callable[[str], None] = print,
    offline: bool = False,
) -> FeasibilityReport:
    """Track A. Gemini multimodal when a key is present, else heuristic fallback.

    `offline=True` forces the deterministic judge — used for mock-backend
    videos, which are schematic silhouettes that a real VLM would (correctly)
    reject as non-photorealistic.
    """
    frames = sample_frames(video)
    if not offline and have_api_key() and frames:
        try:
            parts: list[dict] = [image_part(_jpeg(f)) for f in frames]
            parts.append({"text": VLM_PROMPT.format(n=len(frames), prompt=variant.prompt)})
            raw = generate_content(parts, temperature=VLM_TEMPERATURE,
                                   response_schema=VLM_SCHEMA)
            data = extract_json(raw)
            rec = str(data.get("recommendation", "human_review"))
            if rec not in ("proceed", "reject", "human_review"):
                rec = "human_review"
            return FeasibilityReport(
                physically_plausible=bool(data.get("physically_plausible", False)),
                tracking_likely_valid=bool(data.get("subject_visible", False))
                and bool(data.get("camera_consistent", False)),
                ai_generated_artifacts=[str(a) for a in data.get("artifacts", [])],
                confidence=float(np.clip(float(data.get("confidence", 0.5)), 0.0, 1.0)),
                recommendation=FeasibilityRecommendation(rec),
                judge_source="vlm",
                notes=str(data.get("notes", "")),
            )
        except (GeminiError, ValueError, json.JSONDecodeError) as e:
            log(f"[vlm] {variant.variant_id}: Gemini judge failed ({e}); heuristic fallback")
    return _heuristic_vlm_fallback(frames, variant)


def _heuristic_vlm_fallback(frames: list[np.ndarray],
                            variant: VariantSpec) -> FeasibilityReport:
    """Offline judge: brightness/variance sanity + inter-frame difference."""
    if not frames:
        return FeasibilityReport(
            physically_plausible=False, tracking_likely_valid=False,
            ai_generated_artifacts=["undecodable_video"], confidence=0.9,
            recommendation=FeasibilityRecommendation.reject,
            judge_source="rule_based", notes="no frames could be decoded",
        )
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in frames]
    has_content = all(g.std() > 5.0 for g in grays)
    diffs = [float(np.abs(a - b).mean()) for a, b in zip(grays, grays[1:])]
    has_motion = bool(diffs) and max(diffs) > 0.5
    not_chaotic = not diffs or max(diffs) < 60.0
    plausible = has_content and not_chaotic
    ok = plausible and has_motion
    return FeasibilityReport(
        physically_plausible=plausible,
        tracking_likely_valid=ok,
        ai_generated_artifacts=[] if ok else ["low_content_or_motion"],
        confidence=0.5,
        recommendation=(FeasibilityRecommendation.proceed if ok
                        else FeasibilityRecommendation.human_review),
        judge_source="rule_based",
        notes="offline heuristic judge (no GEMINI_API_KEY or VLM call failed)",
    )


# ---------------------------------------------------------------------------
# Track B: physics / math tools
# ---------------------------------------------------------------------------


def physics_check(
    video: Path,
    max_frames: int = 150,
    log: Callable[[str], None] = print,
    pose_gate: bool = True,
) -> TrackBReport:
    """Optical-flow consistency + velocity spikes + scale jumps on real frames,
    then MediaPipe pose sanity when available (mirrors qa/feasibility.py
    heuristics but computed from pixels instead of pose tracks)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return TrackBReport(physics_ok=False, reasons=["cannot open video"])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    flows: list[float] = []
    areas: list[float] = []
    prev_small: Optional[np.ndarray] = None
    idx = 0
    read = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step:
            idx += 1
            continue
        idx += 1
        read += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        # moving-foreground area as a scale proxy
        _, fg = cv2.threshold(small, int(small.mean()) + 30, 255, cv2.THRESH_BINARY)
        areas.append(float((fg > 0).mean()))
        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            flows.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev_small = small
    cap.release()

    rep = TrackBReport(n_frames=read, fps=fps)
    if len(flows) >= 3:
        fl = np.asarray(flows)
        rep.flow_mean = float(fl.mean())
        med = float(np.median(fl)) + 1e-6
        rep.velocity_spike_ratio = float((fl > 4.0 * med).mean())
        dfl = np.abs(np.diff(fl))
        rep.flow_consistency = float(np.clip(1.0 - dfl.mean() / (med * 3.0), 0.0, 1.0))
    if len(areas) >= 3:
        ar = np.asarray(areas) + 1e-4
        ratio = ar[1:] / ar[:-1]
        rep.scale_jump_ratio = float(((ratio > 1.5) | (ratio < 1 / 1.5)).mean())

    # MediaPipe pose sanity (optional dependency, guarded)
    try:
        rep.pose_detection_rate, rep.pose_tool = _pose_sanity(video)
    except Exception as e:
        log(f"[physics] pose sanity unavailable: {type(e).__name__}: {e}")

    reasons = []
    if read < 10:
        reasons.append(f"too few decodable frames ({read})")
    if rep.velocity_spike_ratio > 0.15:
        reasons.append(f"velocity_spike_ratio={rep.velocity_spike_ratio:.2f} > 0.15")
    if rep.scale_jump_ratio > 0.20:
        reasons.append(f"scale_jump_ratio={rep.scale_jump_ratio:.2f} > 0.20")
    if rep.flow_consistency < 0.25:
        reasons.append(f"flow_consistency={rep.flow_consistency:.2f} < 0.25")
    if pose_gate and rep.pose_detection_rate is not None and rep.pose_detection_rate < 0.30:
        reasons.append(f"pose_detection_rate={rep.pose_detection_rate:.2f} < 0.30")
    rep.reasons = reasons
    rep.physics_ok = not reasons
    return rep


def _pose_sanity(video: Path, n_probe: int = 10) -> tuple[Optional[float], str]:
    """Fraction of probed frames where MediaPipe detects a pose."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    from ..timeseries.extract import _ensure_mediapipe_model

    model_path = _ensure_mediapipe_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    frames = sample_frames(video, n=n_probe)
    if not frames:
        landmarker.close()
        return None, "mediapipe"
    hits = 0
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.pose_landmarks:
            hits += 1
    landmarker.close()
    return hits / len(frames), "mediapipe"


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


def _label_schema(verbs: list[str]) -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "skills": {"type": "ARRAY", "items": {"type": "STRING", "enum": verbs}},
            "scene_type": {"type": "STRING"},
            "lighting": {"type": "STRING"},
            "subject_type": {"type": "STRING", "enum": ["human", "animal", "other"]},
            "caption": {"type": "STRING"},
        },
        "required": ["skills", "subject_type", "caption"],
    }


def label_video(
    video: Path,
    variant: VariantSpec,
    verbs: list[str],
    log: Callable[[str], None] = print,
) -> SyngenLabels:
    """Structured Gemini labeling; keyword fallback stays inside the verb vocab."""
    if have_api_key():
        try:
            frames = sample_frames(video, n=4)
            parts: list[dict] = [image_part(_jpeg(f)) for f in frames]
            parts.append({"text": (
                f"Frames from a video generated for: \"{variant.prompt}\".\n"
                f"Label it: pick 1-3 skill verbs strictly from this vocabulary: "
                f"{', '.join(verbs)}. Also give scene_type, lighting, subject_type, "
                f"and a one-sentence caption.")})
            raw = generate_content(parts, temperature=0.3,
                                   response_schema=_label_schema(verbs))
            data = extract_json(raw)
            skills = [s for s in data.get("skills", []) if s in verbs] or _keyword_skills(variant.prompt, verbs)
            return SyngenLabels(
                skills=skills,
                scene_type=str(data.get("scene_type", "unknown")),
                lighting=str(data.get("lighting", "unknown")),
                subject_type=str(data.get("subject_type", "human")),
                caption=str(data.get("caption", "")),
                source="gemini",
            )
        except (GeminiError, ValueError, json.JSONDecodeError) as e:
            log(f"[label] {variant.variant_id}: Gemini labeling failed ({e}); keyword fallback")
    return SyngenLabels(
        skills=_keyword_skills(variant.prompt, verbs),
        scene_type="indoor", lighting="unknown", subject_type="human",
        caption=variant.prompt[:140], source="mock",
    )


_KEYWORD_MAP = {
    "pick": "grasp", "picking": "grasp", "grab": "grasp", "grasp": "grasp",
    "lift": "lift", "lower": "lower", "place": "place", "put": "place",
    "push": "push", "pull": "pull", "open": "open", "close": "close",
    "pour": "pour", "wave": "reach", "waves": "reach", "reach": "reach",
    "walk": "walk", "walking": "walk", "turn": "turn", "rotate": "rotate",
    "press": "press", "wipe": "wipe", "cut": "cut", "fold": "fold",
    "hold": "hold", "insert": "insert", "remove": "remove", "release": "release",
}


def _keyword_skills(prompt: str, verbs: list[str]) -> list[str]:
    found = []
    for word in prompt.lower().replace(",", " ").replace(".", " ").split():
        skill = _KEYWORD_MAP.get(word)
        if skill and skill in verbs and skill not in found:
            found.append(skill)
    return found or ([verbs[-1]] if verbs else ["idle"])


# ---------------------------------------------------------------------------
# Verdict merging + per-job driver
# ---------------------------------------------------------------------------


def merge_verdict(vlm: FeasibilityReport, physics: TrackBReport) -> tuple[str, list[str]]:
    """Merge Track A + Track B into accept | reject | review.

    reject  if either track hard-fails (VLM says reject, or physics finds
            2+ violations / undecodable video)
    review  if exactly one track is borderline
    accept  only when both tracks pass
    """
    reasons: list[str] = []
    vlm_reject = vlm.recommendation == FeasibilityRecommendation.reject
    vlm_review = vlm.recommendation == FeasibilityRecommendation.human_review
    if vlm_reject:
        reasons.append("vlm: reject (" + (", ".join(vlm.ai_generated_artifacts) or vlm.notes or "unspecified") + ")")
    elif vlm_review:
        reasons.append("vlm: human_review")

    n_phys = len(physics.reasons)
    if n_phys:
        reasons.extend(f"physics: {r}" for r in physics.reasons)

    hard_physics = n_phys >= 2 or any("decodable" in r or "open video" in r for r in physics.reasons)
    if vlm_reject or hard_physics:
        return "reject", reasons
    if vlm_review or n_phys == 1:
        return "review", reasons
    return "accept", reasons or ["both tracks passed"]


def _variant_backend(dirs: JobDirs, variant_id: str) -> str:
    sidecar = dirs.video_sidecar(variant_id)
    if sidecar.is_file():
        try:
            return str(json.loads(sidecar.read_text(encoding="utf-8")).get("backend", ""))
        except json.JSONDecodeError:
            pass
    return ""


def verify_variant(
    spec: JobSpec,
    dirs: JobDirs,
    variant: VariantSpec,
    verbs: list[str],
    log: Callable[[str], None] = print,
) -> VerificationRecord:
    """Run Track A and Track B in parallel for one video, label, merge, save.

    Mock-backend videos (schematic silhouettes, or "cached" re-verifications of
    them) are verified offline: a real VLM would correctly reject them as
    non-photorealistic and MediaPipe cannot pose-track a stick figure, so the
    deterministic judge + physics-without-pose-gate keep the pipeline demoable.
    Veo (or any real) backend videos get the full strict verification.
    """
    video = dirs.video_mp4(variant.variant_id)
    offline = _variant_backend(dirs, variant.variant_id) in ("mock", "cached")
    if not video.is_file():
        rec = VerificationRecord(
            variant_id=variant.variant_id, event_id=variant.event_id,
            cam_id=variant.cam_id, verdict="reject",
            verdict_reasons=["video file missing (generation failed)"])
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(vlm_judge, video, variant, log, offline)
            fut_b = pool.submit(physics_check, video, 150, log, not offline)
            vlm, physics = fut_a.result(), fut_b.result()
        if offline:
            labels = SyngenLabels(
                skills=_keyword_skills(variant.prompt, verbs),
                scene_type="synthetic_mock", lighting="synthetic",
                subject_type="human", caption=variant.prompt[:140], source="mock")
        else:
            labels = label_video(video, variant, verbs, log=log)
        verdict, reasons = merge_verdict(vlm, physics)
        rec = VerificationRecord(
            variant_id=variant.variant_id, event_id=variant.event_id,
            cam_id=variant.cam_id, verdict=verdict, verdict_reasons=reasons,
            vlm=vlm, physics=physics, labels=labels)
    dirs.verification_json(variant.variant_id).write_text(
        rec.model_dump_json(indent=2), encoding="utf-8")
    log(f"[verify] {variant.variant_id}: {rec.verdict}")
    return rec


def verify_all(
    spec: JobSpec,
    dirs: JobDirs,
    verbs: list[str],
    max_workers: int = 3,
    log: Callable[[str], None] = print,
) -> list[VerificationRecord]:
    records: list[VerificationRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(verify_variant, spec, dirs, v, verbs, log)
                   for v in spec.variants]
        for fut in futures:
            records.append(fut.result())
    records.sort(key=lambda r: r.variant_id)
    return records
