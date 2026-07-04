"""Agentic labeler: VLM-driven video labeling with real perception tools.

Flow (agent = Gemini, tools = local perception):
  1. PLAN   - the agent sees the probe + sampled frames and decides which
              tools apply (pose? hands? objects? motion?) with reasons.
  2. SENSE  - chosen tools run locally (MediaPipe pose/hands, YOLO, motion
              energy); each writes source='estimated' artifacts.
  3. LABEL  - the agent sees frames + tool evidence + the verb vocabulary and
              emits segments/captions/scene-tags/feasibility as constrained
              JSON, which is validated and written to the workspace.

Without GEMINI_API_KEY the labeler still runs: tools execute and a heuristic
composes conservative labels (marked judge_source='heuristic', low conf).
Nothing is ever labeled beyond what the evidence supports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from ..config import V2RConfig
from ..schema.io import write_json_model
from ..schema.models import (
    Captions,
    ConsentRecord,
    SceneTags,
    Segment,
    SegmentsFile,
    SourceTag,
)
from ..schema.workspace import EpisodeWorkspace
from ..syngen import gemini
from . import tools as T

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "skill": {"type": "string"},
                    "text": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["start_s", "end_s", "skill", "text"],
            },
        },
        "captions": {
            "type": "object",
            "properties": {
                "short": {"type": "string"},
                "medium": {"type": "string"},
                "long": {"type": "string"},
            },
            "required": ["short", "medium", "long"],
        },
        "scene_tags": {
            "type": "object",
            "properties": {
                "scene_type": {"type": "string"},
                "lighting": {"type": "string"},
                "clutter": {"type": "integer"},
                "surfaces": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scene_type", "lighting", "clutter", "surfaces"],
        },
        "feasibility": {
            "type": "object",
            "properties": {
                "human_present": {"type": "string"},
                "physically_plausible": {"type": "boolean"},
                "tracking_likely_valid": {"type": "boolean"},
                "ai_generated_suspected": {"type": "boolean"},
                "ai_generated_artifacts": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "recommendation": {"type": "string", "enum": ["proceed", "reject", "human_review"]},
            },
            "required": ["human_present", "physically_plausible", "tracking_likely_valid",
                         "ai_generated_suspected", "ai_generated_artifacts",
                         "confidence", "recommendation"],
        },
    },
    "required": ["segments", "captions", "scene_tags", "feasibility"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "human_present": {"type": "string",
                          "enum": ["full_body", "partial", "hands_only", "none"]},
        "animal_present": {"type": "boolean"},
        "camera": {"type": "string"},
        "ai_generated_suspected": {"type": "boolean"},
        "run_pose": {"type": "boolean"},
        "run_hands": {"type": "boolean"},
        "run_objects": {"type": "boolean"},
        "run_motion": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["scene_description", "human_present", "animal_present", "camera",
                 "ai_generated_suspected", "run_pose", "run_hands", "run_objects",
                 "run_motion", "reasoning"],
}


def _ensure_workspace(cfg: V2RConfig, video: Path, episode_id: Optional[str],
                      log: Callable[[str], None]) -> EpisodeWorkspace:
    import re

    if episode_id is None:
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", video.stem)
        episode_id = EpisodeWorkspace.make_episode_id(stem, 0)
    ws = EpisodeWorkspace(cfg.workspaces_root, episode_id).create()
    if not ws.video_path.is_file():
        import shutil

        shutil.copy2(video, ws.video_path)
        log(f"[label] copied source video into {ws.rel(ws.video_path)}")
    if not ws.consent_path.is_file():
        write_json_model(ws.consent_path, ConsentRecord(
            consent_id=f"agentic_{episode_id}",
            license="source-license-unverified",
            subject_ids=[], blur_applied=False,
            notes="Auto-created by agentic labeler; operator must replace with a real consent record before export.",
        ))
    return ws


def _validate_labels(raw: dict, verbs: list[str], duration_s: float) -> dict:
    """Clamp/repair the agent's labels; never trust free-form output blindly.

    Raises ValueError on structurally incomplete labels (missing captions or
    feasibility verdict) so the agent loop can reject the finalize and ask
    for a complete labels object instead of writing a broken workspace.
    """
    caps = raw.get("captions") or {}
    missing = [k for k in ("short", "medium", "long") if not str(caps.get(k, "")).strip()]
    if missing:
        raise ValueError(f"captions incomplete (missing {missing})")
    feas = raw.get("feasibility") or {}
    missing = [k for k in ("human_present", "recommendation") if not feas.get(k)]
    if missing:
        raise ValueError(f"feasibility incomplete (missing {missing})")
    if feas["recommendation"] not in ("proceed", "reject", "human_review"):
        raise ValueError(f"invalid recommendation {feas['recommendation']!r}")
    if feas["human_present"] == "none":
        # quadruped gaits/locomotion are legitimate with no human (animal
        # subject -> Go2 retargeting); only human-manipulation verbs contradict
        non_human_verbs = {"idle", "walk", "trot", "gallop", "jump", "sit",
                           "lie_down", "stand", "turn"}
        human_verbs = {s.get("skill") for s in raw.get("segments", [])} - non_human_verbs
        if human_verbs:
            raise ValueError(
                f"contradiction: human_present='none' but segments use human "
                f"action verbs {sorted(human_verbs)}; either revise "
                "human_present or use only locomotion/idle verbs "
                "(walk/trot/gallop/jump/sit/lie_down/stand/turn/idle)")

    fixed_segments: list[Segment] = []
    fallback = "idle" if "idle" in verbs else verbs[0]
    prev_end = 0.0
    for seg in sorted(raw.get("segments", []), key=lambda s: s.get("start_s", 0.0)):
        s0 = max(0.0, min(float(seg["start_s"]), duration_s))
        s1 = max(s0, min(float(seg["end_s"]), duration_s))
        if s1 - s0 < 1e-3:
            continue
        s0 = max(s0, prev_end)  # enforce non-overlap
        if s1 <= s0:
            continue
        skill = seg["skill"] if seg["skill"] in verbs else fallback
        ev = seg.get("evidence")
        fixed_segments.append(Segment(start_s=s0, end_s=s1, skill=skill,
                                      text=str(seg.get("text", ""))[:200],
                                      evidence=str(ev)[:300] if ev else None))
        prev_end = s1
    if not fixed_segments:
        fixed_segments = [Segment(start_s=0.0, end_s=duration_s, skill=fallback,
                                  text="no confident segmentation")]
    raw["segments"] = fixed_segments
    tags = raw.get("scene_tags") or {}
    tags["clutter"] = int(min(5, max(1, tags.get("clutter", 3))))
    tags.setdefault("scene_type", "unknown")
    tags.setdefault("lighting", "unknown")
    tags.setdefault("surfaces", [])
    raw["scene_tags"] = tags
    feas["confidence"] = float(min(1.0, max(0.0, feas.get("confidence", 0.3))))
    feas.setdefault("physically_plausible", True)
    feas.setdefault("tracking_likely_valid", False)
    feas.setdefault("ai_generated_suspected", False)
    feas.setdefault("ai_generated_artifacts", [])
    raw["feasibility"] = feas
    return raw


def _heuristic_labels(evidence: dict, verbs: list[str], duration_s: float) -> dict:
    """No-VLM fallback: conservative labels from tool evidence only."""
    pose = evidence.get("pose", {})
    hands = evidence.get("hands", {})
    objects = evidence.get("objects", {})
    human = "none"
    if pose.get("person_present_ratio", 0) > 0.5:
        human = "full_body"
    elif hands.get("hands_present_ratio", 0) > 0.3:
        human = "hands_only"
    classes = ", ".join(list(objects.get("classes", {}))[:5]) or "none detected"
    seg_skill = "hold" if human == "hands_only" and "hold" in verbs else (
        "idle" if "idle" in verbs else verbs[0])
    return {
        "segments": [{"start_s": 0.0, "end_s": duration_s, "skill": seg_skill,
                      "text": f"heuristic: human={human}, objects={classes}"}],
        "captions": {
            "short": f"Unlabeled clip (human: {human}).",
            "medium": f"Heuristic labeling only. Human presence: {human}. "
                      f"Detected object classes: {classes}.",
            "long": "GEMINI_API_KEY unavailable or VLM call failed; labels are "
                    "tool-evidence heuristics only. Evidence: "
                    + json.dumps(evidence, default=str)[:800],
        },
        "scene_tags": {"scene_type": "unknown", "lighting": "unknown",
                       "clutter": 3, "surfaces": []},
        "feasibility": {
            "human_present": human,
            "physically_plausible": True,
            "tracking_likely_valid": pose.get("person_present_ratio", 0) > 0.5,
            "ai_generated_suspected": False,
            "ai_generated_artifacts": [],
            "confidence": 0.3,
            "recommendation": "human_review",
        },
    }


def run_agentic_labeler(
    cfg: V2RConfig,
    video: Path,
    episode_id: Optional[str] = None,
    model: Optional[str] = None,
    agent: str = "auto",
    log: Callable[[str], None] = print,
) -> dict:
    """agent: 'loop' = iterative investigate-verify agent (NIM/Gemini);
    'simple' = one-pass plan/sense/label; 'auto' = loop when an LLM is up."""
    video = Path(video)
    ws = _ensure_workspace(cfg, video, episode_id, log)
    model = model or gemini.DEFAULT_VISION_MODEL
    have_vlm = gemini.have_api_key(cfg.root)

    probe = T.probe_video(ws.video_path)
    duration_s = float(probe.get("duration_s", 0.0))

    # ---- iterative agent path ----------------------------------------------
    if agent in ("auto", "loop"):
        from .agent_loop import AgentLoop
        from .llm import LLMRouter

        router = LLMRouter(cfg.root, log=log)
        if router.available():
            log(f"[label] agent loop | orchestrator={router.active_provider('orchestrator')} "
                f"critic={router.active_provider('fast')}")
            loop = AgentLoop(cfg, ws, router, log=log)
            result = loop.run()
            judge = f"agent-loop:{router.active_provider('orchestrator')}"
            return _write_outputs(
                ws, video, result["labels"],
                plan={"mode": "agent-loop", "steps": result["steps"],
                      "revisions": result["revisions"],
                      "critic": result["evidence"].get("critic")},
                evidence=result["evidence"], judge_source=judge, log=log)
        if agent == "loop":
            log("[label] no LLM provider available; falling back to simple mode")

    jpegs, stamps, sample_info = T.sample_frames(
        ws.video_path, n=6, save_dir=ws.frames_review_dir)
    log(f"[label] probe: {probe.get('width')}x{probe.get('height')} "
        f"{probe.get('fps'):.0f} fps {duration_s:.1f}s | frames sampled: {sample_info['n_sampled']}")

    # ---- 1. PLAN ----------------------------------------------------------
    plan = None
    if have_vlm:
        try:
            parts = [gemini.image_part(j) for j in jpegs]
            parts.append({"text": (
                "You are the planning step of a video-labeling agent for robot "
                "learning data. The frames above are evenly sampled from one "
                f"video ({json.dumps({k: probe[k] for k in ('width', 'height', 'fps', 'duration_s') if k in probe})}, "
                f"timestamps {['%.1f' % s for s in stamps]}). Decide which local "
                "perception tools to run: run_pose (MediaPipe full-body pose - "
                "only useful if a person's body is visible), run_hands (MediaPipe "
                "hand tracking - useful if hands are visible close-up), "
                "run_objects (YOLO COCO object detection), run_motion (motion-"
                "energy timeline - cheap, useful for segmentation). Also assess "
                "whether the video looks AI-generated (artifacts: impossible "
                "physics, morphing, texture shimmer, inconsistent shadows). "
                "Be strict about human_present: 'hands_only' when only hands/"
                "forearms are visible."
            )})
            plan = gemini.extract_json(gemini.generate_content(
                parts, model=model, temperature=0.0, response_schema=PLAN_SCHEMA,
                api_key=gemini.get_api_key(cfg.root)))
            log(f"[label] plan: human={plan['human_present']} pose={plan['run_pose']} "
                f"hands={plan['run_hands']} objects={plan['run_objects']} "
                f"motion={plan['run_motion']} | {plan['scene_description'][:90]}")
        except Exception as e:  # noqa: BLE001
            log(f"[label] VLM plan failed ({e}); running all tools")
    if plan is None:
        plan = {"scene_description": "", "human_present": "unknown", "animal_present": False,
                "camera": "unknown", "ai_generated_suspected": False,
                "run_pose": True, "run_hands": True, "run_objects": True,
                "run_motion": True, "reasoning": "no VLM; run everything"}

    # ---- 2. SENSE ----------------------------------------------------------
    evidence: dict = {"probe": probe, "plan": plan}
    if plan["run_motion"]:
        evidence["motion"] = T.motion_timeline(ws.video_path)
        log(f"[label] motion: mean_energy={evidence['motion'].get('mean_energy')} "
            f"changepoints={evidence['motion'].get('changepoint_candidates_s')}")
    if plan["run_pose"]:
        evidence["pose"] = T.track_human_pose(ws.video_path, ws, cfg)
        log(f"[label] pose: present_ratio={evidence['pose'].get('person_present_ratio')} "
            f"conf={evidence['pose'].get('mean_conf')}")
    if plan["run_hands"]:
        evidence["hands"] = T.track_hands(ws.video_path, ws, cfg)
        log(f"[label] hands: present_ratio={evidence['hands'].get('hands_present_ratio')} "
            f"aperture={evidence['hands'].get('aperture_m')}")
    if plan["run_objects"]:
        evidence["objects"] = T.detect_objects(ws.video_path, ws, cfg)
        log(f"[label] objects: {evidence['objects'].get('classes')}")

    # ---- 3. LABEL ----------------------------------------------------------
    labels = None
    judge_source = "heuristic"
    if have_vlm:
        try:
            ev_for_llm = {k: v for k, v in evidence.items() if k != "probe"}
            parts = [gemini.image_part(j) for j in jpegs]
            parts.append({"text": (
                "You are the labeling step of a video-labeling agent for robot "
                "learning data. Produce final labels for this video from the "
                "frames above plus the tool evidence below.\n\n"
                f"TOOL EVIDENCE:\n{json.dumps(ev_for_llm, default=str)[:6000]}\n\n"
                f"VIDEO: duration {duration_s:.2f}s.\n"
                f"SEGMENT SKILLS: every segments[].skill MUST be one of "
                f"{json.dumps(cfg.verbs)} - segment the video by activity over "
                "time (use motion changepoints and pose/hand evidence; segments "
                "must be non-overlapping, within [0, duration], covering the "
                "salient activity; use 'idle' for uneventful spans).\n"
                "CAPTIONS: short (one clause), medium (1-2 sentences), long "
                "(detailed paragraph incl. camera, scene, activity timeline).\n"
                "HONESTY RULES: label only what is visible; if no human is "
                "present say so in feasibility.human_present='none' and do NOT "
                "invent manipulation skills; if only hands are visible use "
                "'hands_only'. Note AI-generation artifacts if present. "
                "confidence in [0,1] reflects overall label reliability. "
                "recommendation: 'proceed' only if this clip is usable for "
                "robot-learning labeling; 'reject' if unusable (no subject, "
                "abstract content); 'human_review' if borderline."
            )})
            labels = gemini.extract_json(gemini.generate_content(
                parts, model=model, temperature=0.0, response_schema=LABEL_SCHEMA,
                api_key=gemini.get_api_key(cfg.root)))
            judge_source = f"gemini:{model}"
        except Exception as e:  # noqa: BLE001
            log(f"[label] VLM labeling failed ({e}); heuristic fallback")
    if labels is not None:
        try:
            labels = _validate_labels(labels, cfg.verbs, duration_s)
        except ValueError as e:
            log(f"[label] VLM labels incomplete ({e}); heuristic fallback")
            labels = None
            judge_source = "heuristic"
    if labels is None:
        labels = _validate_labels(_heuristic_labels(evidence, cfg.verbs, duration_s),
                                  cfg.verbs, duration_s)
    return _write_outputs(ws, video, labels, plan=plan, evidence=evidence,
                          judge_source=judge_source, log=log)


def _write_outputs(ws: EpisodeWorkspace, video: Path, labels: dict, plan: dict,
                   evidence: dict, judge_source: str,
                   log: Callable[[str], None]) -> dict:
    src = SourceTag.estimated
    write_json_model(ws.segments_json, SegmentsFile(
        segments=labels["segments"], method=f"agentic:{judge_source}", source=src))
    cap = labels["captions"]
    write_json_model(ws.captions_json, Captions(
        short=cap["short"], medium=cap["medium"], long=cap["long"], source=src))
    tg = labels["scene_tags"]
    write_json_model(ws.scene_tags_json, SceneTags(
        scene_type=tg["scene_type"], lighting=tg["lighting"], clutter=tg["clutter"],
        surfaces=tg["surfaces"], source=src))

    feas = labels["feasibility"]
    report = {
        "judge_source": judge_source,
        "video": str(video),
        "episode_id": ws.episode_id,
        "plan": plan,
        "evidence": {k: v for k, v in evidence.items() if k not in ("probe", "plan")},
        "feasibility": feas,
        "labels_written": [ws.rel(ws.segments_json), ws.rel(ws.captions_json),
                           ws.rel(ws.scene_tags_json)],
    }
    (ws.qa_dir / "agentic_label_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    md = [
        f"# Agentic label report - {ws.episode_id}",
        "",
        f"- judge: {judge_source}",
        f"- human present: {feas.get('human_present')}",
        f"- AI-generated suspected: {feas.get('ai_generated_suspected')} "
        f"{feas.get('ai_generated_artifacts')}",
        f"- recommendation: **{feas.get('recommendation')}** "
        f"(confidence {feas.get('confidence'):.2f})",
    ]
    if plan.get("mode") == "agent-loop":
        md.append(f"- agent: {plan['steps']} steps, {plan['revisions']} revision(s), "
                  f"critic={plan.get('critic', {}).get('verdict')}")
        if evidence.get("notes"):
            md += ["", "## Agent notes", ""]
            md += [f"- {n}" for n in evidence["notes"]]
    md += ["", "## Segments", ""]
    for s in labels["segments"]:
        md.append(f"- `{s.start_s:6.2f} - {s.end_s:6.2f}s` **{s.skill}** - {s.text}")
    md += ["", "## Captions", "", f"- short: {cap['short']}",
           f"- medium: {cap['medium']}", "", f"{cap['long']}", ""]
    (ws.qa_dir / "agentic_label_report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[label] report: {ws.rel(ws.qa_dir / 'agentic_label_report.md')} "
        f"| recommendation={feas.get('recommendation')}")
    return report
