"""True labeling agent: an iterative investigate-verify loop over a video.

Each turn the agent (Nemotron omni / Kimi via NIM, or Gemini) emits ONE JSON
action; python executes it and returns the observation. The agent can look at
frames (anywhere, any zoom), run perception tools, request whole-video
analysis, take notes, and finally submit labels — which a separate critic
checks against the accumulated evidence before they are accepted. One
revision round is allowed. Everything is transcripted to the workspace.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..config import V2RConfig
from ..schema.workspace import EpisodeWorkspace
from ..syngen import gemini
from . import tools as T
from .llm import LLMError, LLMRouter, parse_json_reply

MAX_STEPS = 12
MAX_LOOKS = 3
MAX_REVISIONS = 1
MAX_FRAMES_PER_LOOK = 8

FINALIZE_TEMPLATE = """{"thought": "...", "action": "finalize", "args": {"labels": {
  "segments": [{"start_s": 0.0, "end_s": 0.0, "skill": "<verb>", "text": "...",
    "evidence": "<which measurement supports this segment>"}],
  "captions": {"short": "...", "medium": "...", "long": "..."},
  "scene_tags": {"scene_type": "...", "lighting": "...", "clutter": 3, "surfaces": []},
  "feasibility": {"human_present": "full_body|partial|hands_only|none",
    "physically_plausible": true, "tracking_likely_valid": true,
    "ai_generated_suspected": false, "ai_generated_artifacts": [],
    "confidence": 0.5, "recommendation": "proceed|reject|human_review"}}}}"""

VIDEO_EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {"type": "array", "items": {"type": "object", "properties": {
            "t_start_s": {"type": "number"}, "t_end_s": {"type": "number"},
            "description": {"type": "string"},
        }, "required": ["t_start_s", "t_end_s", "description"]}},
        "camera_motion": {"type": "string"},
        "ai_generated_suspected": {"type": "boolean"},
        "ai_generated_artifacts": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["events", "camera_motion", "ai_generated_suspected",
                 "ai_generated_artifacts", "summary"],
}


def _system_prompt(cfg: V2RConfig, probe: dict, multimodal: bool) -> str:
    frames_note = (
        "Frames you request with `look` are shown to you directly as images."
        if multimodal else
        "You cannot see images directly; `look` returns descriptions from a vision model."
    )
    return f"""You are a rigorous video-labeling agent for robot-learning data.
GOAL: produce the best possible labels for ONE video by iteratively gathering
evidence with tools, reconciling conflicts, and only then finalizing.

VIDEO: {probe.get('width')}x{probe.get('height')}, {probe.get('fps', 0):.0f} fps, \
{probe.get('duration_s', 0):.2f} s.
{frames_note}

Respond with EXACTLY ONE JSON object per turn, no prose outside it:
  {{"thought": "<brief reasoning>", "action": "<name>", "args": {{...}}}}

ACTIONS:
- look {{"timestamps": [seconds,...]}} or {{"n": k}} (max {MAX_FRAMES_PER_LOOK} frames/call,
  max {MAX_LOOKS} calls). Optional "region": [x0,y0,x1,y1] normalized crop to zoom in.
- run_tool {{"name": "pose"|"animal_pose"|"hands"|"objects"|"track"|"flow"|
  "primitives"|"scenes"|"action"|"motion"}} - local perception:
  pose = MediaPipe full-body per-second presence + wrist speed timeline (HUMANS);
  animal_pose = SuperAnimal-Quadruped 39-keypoint tracking for FOUR-LEGGED
    ANIMALS (dog, horse, cow, cat, tiger...). Returns per-second keypoint
    presence, paw-speed timelines, stride periodicity, body displacement,
    spine angle + standing/recumbent posture. Use INSTEAD of `pose` when the
    subject is an animal. Downstream: Unitree Go2 quadruped retargeting;
  hands = MediaPipe hand tracking + aperture timeline (fails on gloves);
  objects = YOLO per-frame detection (what exists, when);
  track = YOLO+ByteTrack object TRAJECTORIES over time: per-object movement,
    speed, direction changes, periodicity, size change (approach/pick-up);
  flow = dense optical flow per second: camera motion vs subject motion
    separated - tells you if the camera or the subject moves;
  primitives = MOTION-PRIMITIVE segmentation: statistical changepoints over
    all motion channels (flow, wrist speed, aperture, object speeds) ->
    time boundaries for your segments. Run AFTER pose/hands/track so their
    signals are included; strongest evidence for segment start/end times;
  scenes = shot-cut detection (is this edited footage?);
  action = Kinetics-400 action classifier per window (weak prior, 400
    everyday classes - corroborates but never overrules what you see);
  find = OPEN-VOCABULARY detection with your own text queries, e.g.
    {{"name": "find", "queries": ["blue nitrile glove", "pipette"]}} -
    use when MediaPipe/YOLO fail but you SEE something (gloves, tools);
    'not_found' does not prove absence for small/unusual objects;
  transcribe = SPEECH transcription from the audio track (timestamps,
    speaker, verbatim text, language, intent). Run it whenever people speak
    or video_analysis mentions speech/talking: aligned (situation, utterance)
    pairs teach the robot WHAT TO SAY in context. Quote key utterances in
    captions.long. No speech -> it honestly reports has_speech=false;
  forensics = AI-generation statistics (WEAK heuristic; only a hint);
  motion = cheap per-bin motion energy (superseded by flow/primitives).
  Each returns a JSON summary and writes artifacts.
- video_analysis {{}} - a vision model watches the ENTIRE video and returns
  dense timestamped events + AI-generation assessment. Expensive; use once,
  and prefer it when temporal detail or AI-artifact detection matters.
- note {{"text": "..."}} - record a working hypothesis or conflict.
- finalize {{"labels": {{"segments": [{{"start_s","end_s","skill","text",
  "evidence"}}...],
  "captions": {{"short","medium","long"}},
  "scene_tags": {{"scene_type","lighting","clutter"(1-5),"surfaces":[]}},
  "feasibility": {{"human_present": "full_body"|"partial"|"hands_only"|"none",
  "physically_plausible", "tracking_likely_valid", "ai_generated_suspected",
  "ai_generated_artifacts": [], "confidence"(0-1),
  "recommendation": "proceed"|"reject"|"human_review"}}}}}}

RULES:
- Every segments[].skill MUST come from: {json.dumps(cfg.verbs)}.
- Segments non-overlapping, within [0, duration], ordered.
- INVESTIGATE CONFLICTS: if a tool contradicts what you see (e.g. you see
  hands but the hand tracker finds none), say so in a note, explain it
  (gloves? occlusion? tracker limitation?) and adjust confidence - never
  silently ignore a conflict.
- YOUR EYES OUTRANK THE TOOLS: the pose detector fires on animals and
  person-like textures. If frames show no human, a positive pose ratio is a
  false positive - do not label a human. If you see an animal, run the
  `objects` tool to identify it and set human_present='none'.
- ANIMAL SUBJECTS: if the subject is a four-legged animal, set
  human_present='none', run `animal_pose` (not `pose`), and label GAIT/behavior
  segments with quadruped verbs (walk/trot/gallop/jump/sit/lie_down/stand/turn/
  idle). A well-tracked animal clip is 'proceed' (it retargets to a Go2 robot
  dog); note the animal species/behavior in captions. Derive gait from
  animal_pose: stride_period_s + paw-speed amplitude + body_speed distinguish
  walk (slow, long period) / trot / gallop (fast, high paw speed); low
  locomotion + recumbent posture => sit/lie_down/stand.
- Label ONLY what is evidenced. No human -> human_present='none' and no
  manipulation verbs. Fabricating labels is the one unforgivable failure.
- recommendation: 'proceed' only if usable for robot-learning labeling.
- CALIBRATION: 'confidence' is your confidence in the LABELS THEMSELVES, not
  in the clip's usefulness - a certain "no human, reject" deserves HIGH
  confidence. Set ai_generated_suspected=true ONLY with concrete artifacts
  observed (morphing, shimmer, impossible physics) and list them.
- Be economical: every action costs money; finalize as soon as evidence is
  sufficient, but not before checking temporal structure (best: run
  pose/hands/track first, then `primitives` for changepoint-based segment
  boundaries; or video_analysis) and subject presence (look + pose/hands).
- Segment start/end times MUST come from measured evidence (primitives
  boundaries, flow timeline, video_analysis events) - never guessed from
  still frames alone.
- PROVENANCE: every segment's "evidence" field must state which measurement
  supports it, e.g. "boundary from primitives changepoint at 4.9s; skill
  from frames at 5-6s + wrist speed peak". Buyers audit this field.
- THE TASK ENDS ONLY WHEN YOU ISSUE `finalize` WITH THE FULL LABELS OBJECT.
  Notes never complete the task. Never claim you are done without finalize."""


class AgentLoop:
    def __init__(self, cfg: V2RConfig, ws: EpisodeWorkspace, router: LLMRouter,
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.ws = ws
        self.router = router
        self.log = log
        self.evidence: dict = {}
        self.transcript: list[dict] = []
        self.looks = 0
        self.video_analysis_done = False
        self.finalize_failures = 0
        self.temporal_gate_fires = 0

    # ------------------------------------------------------------------
    def run(self) -> dict:
        from .labeler import _validate_labels  # shared validator

        probe = T.probe_video(self.ws.video_path)
        self.evidence["probe"] = probe
        duration = float(probe.get("duration_s", 0.0))
        multimodal = self.router.orchestrator_is_multimodal()

        messages = [
            {"role": "system", "content": _system_prompt(self.cfg, probe, multimodal)},
            {"role": "user", "content": "Begin. Investigate the video and produce labels."},
        ]
        pending_images: list[bytes] = []
        revisions = 0
        labels = None
        last_finalize_attempt = None
        consecutive_notes = 0

        for step in range(1, MAX_STEPS + 1):
            try:
                # generous budget: reasoning models spend tokens on a hidden
                # reasoning channel before emitting the JSON action;
                # force_json = provider-native JSON mode (kills parse errors)
                reply = self.router.chat(messages, role="orchestrator",
                                         images=pending_images or None,
                                         max_tokens=16384, force_json=True)
            except LLMError as e:
                self.log(f"[agent] LLM unavailable at step {step}: {e}")
                break
            pending_images = []
            try:
                act = parse_json_reply(reply)
                action = act.get("action", "")
                args = act.get("args", {}) or {}
                thought = str(act.get("thought", ""))[:300]
                if not action:
                    # models sometimes drop the wrapper and emit labels directly
                    if "labels" in act or "segments" in act:
                        action, args = "finalize", ({"labels": act.get("labels", act)})
                    elif "labels" in args or "segments" in args:
                        action = "finalize"
            except Exception:
                hint = ("Invalid response. Reply with exactly one JSON object: "
                        '{"thought": ..., "action": ..., "args": {...}}')
                if '"finalize"' in reply and not reply.rstrip().endswith("}"):
                    hint = ("Your reply was TRUNCATED mid-JSON. Re-issue finalize "
                            "with SHORTER text: captions <=20 words each, segment "
                            "texts <=8 words, thought <=10 words.")
                messages += [{"role": "assistant", "content": reply[:2000]},
                             {"role": "user", "content": hint}]
                self._record(step, "parse_error", {}, reply[:600], 0.0)
                continue

            t0 = time.time()
            messages.append({"role": "assistant", "content": reply[:12000]})
            self.log(f"[agent] step {step}: {action} {json.dumps(args)[:100]} | {thought[:90]}")

            if action == "finalize":
                raw_labels = args.get("labels", args)
                gate_msg = self._temporal_gate(raw_labels, step)
                if gate_msg:
                    messages.append({"role": "user", "content": gate_msg})
                    self._record(step, "finalize", {}, "temporal gate: measured "
                                 "evidence required", time.time() - t0)
                    continue
                try:
                    labels = _validate_labels(raw_labels, self.cfg.verbs, duration)
                except Exception as e:  # noqa: BLE001
                    self.finalize_failures += 1
                    # give the model one shot at fixing its own labels; after
                    # that, repair the missing fields from the evidence ledger
                    # (small reasoning models often never emit the full object)
                    repaired = (self._repair_labels(raw_labels, duration)
                                if self.finalize_failures >= 2 else None)
                    if repaired is not None:
                        self.log(f"[agent] finalize repaired from evidence ({e})")
                        labels = repaired
                    else:
                        messages.append({"role": "user", "content":
                                         f"finalize rejected, malformed labels ({e}). Re-issue "
                                         "finalize with ALL fields of the labels object filled "
                                         "in, exactly this shape:\n" + FINALIZE_TEMPLATE})
                        self._record(step, "finalize", {}, f"malformed: {e}", time.time() - t0)
                        continue
                last_finalize_attempt = labels
                verdict = self._critic(labels)
                self._record(step, "finalize", {}, f"critic: {verdict['verdict']}", time.time() - t0)
                if verdict["verdict"] == "accept" or revisions >= MAX_REVISIONS:
                    self.evidence["critic"] = verdict
                    break
                revisions += 1
                self.log(f"[agent] critic requested revision: {verdict['problems']}")
                messages.append({"role": "user", "content":
                                 "A verification critic reviewed your labels against the "
                                 "evidence and requests a revision. Problems: "
                                 + json.dumps(verdict["problems"])
                                 + " Address each problem (gather more evidence if needed) "
                                   "and finalize again."})
                labels = None
                continue

            obs, images = self._execute(action, args)
            self._record(step, action, args, obs, time.time() - t0)
            obs_msg = f"OBSERVATION ({action}): {json.dumps(obs, default=str)[:3500]}"
            if images and multimodal:
                pending_images = images
                obs_msg += f" [{len(images)} frames attached]"

            # anti-spin guards: notes never end the task; force finalize near budget
            consecutive_notes = consecutive_notes + 1 if action == "note" else 0
            if consecutive_notes >= 2:
                obs_msg += ("\nREMINDER: notes do NOT complete the task. Issue the "
                            "`finalize` action with the full labels object now.")
            if step >= MAX_STEPS - 2 and labels is None:
                obs_msg += (f"\nBUDGET WARNING: step {step}/{MAX_STEPS}. You MUST issue "
                            "`finalize` with your best labels on the next turn.")
            messages.append({"role": "user", "content": obs_msg})

        if labels is None and last_finalize_attempt is not None:
            self.log("[agent] using last finalize attempt (critic problems unresolved)")
            labels = last_finalize_attempt
            self.evidence.setdefault("critic", {"verdict": "revise",
                                                "problems": ["revision never re-finalized"]})
        # unresolved critic problems must not ship as 'proceed': downgrade
        crit = self.evidence.get("critic", {})
        if (labels is not None and crit.get("verdict") == "revise"
                and labels["feasibility"].get("recommendation") == "proceed"):
            labels["feasibility"]["recommendation"] = "human_review"
            labels["feasibility"]["confidence"] = min(
                float(labels["feasibility"].get("confidence", 0.5)), 0.6)
            self.evidence.setdefault("notes", []).append(
                "recommendation downgraded proceed -> human_review: critic "
                "problems unresolved at revision budget: "
                + "; ".join(crit.get("problems", [])[:2]))
        if labels is None:
            self.log("[agent] no finalize issued; composing from evidence (heuristic)")
            from .labeler import _heuristic_labels, _validate_labels as _v

            labels = _v(_heuristic_labels(self.evidence, self.cfg.verbs, duration),
                        self.cfg.verbs, duration)
            self.evidence["critic"] = {"verdict": "not_run", "problems": ["agent loop did not finalize"]}

        (self.ws.qa_dir / "agentic_transcript.json").write_text(
            json.dumps({"transcript": self.transcript, "llm_stats": self.router.stats},
                       indent=2, default=str), encoding="utf-8")
        return {"labels": labels, "evidence": self.evidence,
                "steps": len(self.transcript), "revisions": revisions}

    # ------------------------------------------------------------------
    def _execute(self, action: str, args: dict) -> tuple[dict, list[bytes]]:
        try:
            if action == "look":
                return self._look(args)
            if action == "run_tool":
                return self._run_tool(str(args.get("name", "")), args), []
            if action == "video_analysis":
                return self._video_analysis(), []
            if action == "note":
                self.evidence.setdefault("notes", []).append(str(args.get("text", ""))[:500])
                return {"noted": True}, []
            return {"error": f"unknown action {action!r}; valid: look, run_tool, "
                             "video_analysis, note, finalize"}, []
        except Exception as e:  # noqa: BLE001 - agent gets the error as observation
            return {"error": f"{type(e).__name__}: {e}"}, []

    def _look(self, args: dict) -> tuple[dict, list[bytes]]:
        if self.looks >= MAX_LOOKS:
            return {"error": f"look budget exhausted ({MAX_LOOKS})"}, []
        self.looks += 1
        import cv2
        import numpy as np

        probe = self.evidence["probe"]
        fps = probe.get("fps", 30.0)
        duration = probe.get("duration_s", 0.0)
        stamps = args.get("timestamps")
        if not stamps:
            n = int(args.get("n", 6))
            stamps = list(np.linspace(0, max(duration - 0.05, 0), min(n, MAX_FRAMES_PER_LOOK)))
        # clamp to the last decodable frame (exactly `duration` seeks past EOF)
        last_ok = max(duration - 1.5 / max(fps, 1.0), 0.0)
        stamps = [float(min(max(s, 0.0), last_ok)) for s in stamps][:MAX_FRAMES_PER_LOOK]
        region = args.get("region")

        cap = cv2.VideoCapture(str(self.ws.video_path))
        jpegs: list[bytes] = []
        brightened = False
        for s in stamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if region and len(region) == 4:
                x0, y0, x1, y1 = [float(v) for v in region]
                frame = frame[int(y0 * h):int(y1 * h) or h, int(x0 * w):int(x1 * w) or w]
                h, w = frame.shape[:2]
            # night scenes hide subjects from the VLM; brighten honestly
            luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            if luma < 45:
                frame = cv2.convertScaleAbs(frame, alpha=min(3.0, 90.0 / max(luma, 1.0)),
                                            beta=25)
                brightened = True
            if w > 512:
                frame = cv2.resize(frame, (512, int(512 * h / w)))
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                jpegs.append(buf.tobytes())
        cap.release()

        obs: dict = {"timestamps_s": [round(s, 2) for s in stamps],
                     "n_frames": len(jpegs), "region": region}
        if brightened:
            obs["note"] = ("dark video: frames auto-brightened for visibility "
                           "(exposure boost, not scene content change)")
        if not self.router.orchestrator_is_multimodal():
            desc = self.router.chat(
                [{"role": "user", "content":
                  f"Describe precisely what is visible in these {len(jpegs)} video frames "
                  f"(timestamps {obs['timestamps_s']}): subjects, hands, objects, actions, "
                  "camera. One short paragraph per frame."}],
                role="vision", images=jpegs)
            obs["descriptions"] = desc[:2500]
            jpegs = []
        self.evidence.setdefault("looks", []).append(obs)
        return obs, jpegs

    def _run_tool(self, name: str, args: dict | None = None) -> dict:
        args = args or {}
        if name in self.evidence and name != "find":  # find: new queries allowed
            return {"error": f"tool {name!r} already ran; see earlier observation"}
        if name == "pose":
            out = T.track_human_pose(self.ws.video_path, self.ws, self.cfg)
            if out.get("presence_flicker", 0) > 0.3 and "objects" not in self.evidence:
                out["warning"] = ("presence flickers heavily - likely a FALSE "
                                  "POSITIVE (animal/texture); run 'objects' to "
                                  "identify what is actually in frame")
        elif name == "animal_pose":
            out = T.animal_pose(self.ws.video_path, self.ws, self.cfg)
        elif name == "hands":
            out = T.track_hands(self.ws.video_path, self.ws, self.cfg)
        elif name == "objects":
            out = T.detect_objects(self.ws.video_path, self.ws, self.cfg)
        elif name == "track":
            out = T.track_objects(self.ws.video_path, self.ws, self.cfg)
        elif name == "flow":
            out = T.optical_flow_timeline(self.ws.video_path, self.ws)
        elif name == "primitives":
            out = T.motion_primitives(self.ws.video_path, self.ws, self.cfg)
        elif name == "scenes":
            out = T.detect_scenes(self.ws.video_path)
        elif name == "action":
            out = T.recognize_action(self.ws.video_path)
        elif name == "find":
            out = T.find_objects(self.ws.video_path, self.ws, self.cfg,
                                 queries=list(args.get("queries", [])))
        elif name == "forensics":
            out = T.aigen_forensics(self.ws.video_path)
        elif name == "motion":
            out = T.motion_timeline(self.ws.video_path)
        elif name == "transcribe":
            from ..labeling.transcribe import transcribe_to_workspace

            out = transcribe_to_workspace(self.ws, self.cfg,
                                          api_key=self.router.gemini_key)
        else:
            return {"error": f"unknown tool {name!r}; valid: pose, animal_pose, "
                             "hands, objects, track, flow, primitives, scenes, "
                             "action, find, forensics, motion, transcribe"}
        self.evidence[name] = out
        return out

    def _video_analysis(self) -> dict:
        if self.video_analysis_done:
            return {"error": "video_analysis already run; see earlier observation"}
        self.video_analysis_done = True
        if self.router.gemini_key:
            raw = gemini.analyze_video(
                self.ws.video_path,
                prompt=("Watch this entire video. Return dense timestamped events "
                        "(every distinct action/phase), camera motion, and an "
                        "AI-generation assessment (look for temporal artifacts: "
                        "morphing, texture shimmer, impossible physics, flickering)."),
                response_schema=VIDEO_EVENTS_SCHEMA,
                api_key=self.router.gemini_key)
            out = gemini.extract_json(raw)
        else:
            out = self._video_analysis_from_frames()
        self.evidence["video_analysis"] = out
        return out

    def _video_analysis_from_frames(self) -> dict:
        """Gemini-free fallback: dense frame strip through the multimodal
        orchestrator (coarser than native video, but no single provider
        dependency)."""
        if not self.router.orchestrator_is_multimodal():
            return {"error": "video_analysis unavailable (no video-capable "
                             "provider); use look + tools"}
        jpegs, stamps, _ = T.sample_frames(self.ws.video_path, n=12)
        prompt = (
            f"These {len(jpegs)} frames are sampled at timestamps "
            f"{[round(s, 1) for s in stamps]} (seconds) from one video. "
            "Reconstruct the event timeline. Reply with ONLY JSON matching: "
            + json.dumps(VIDEO_EVENTS_SCHEMA))
        reply = self.router.chat([{"role": "user", "content": prompt}],
                                 role="vision", images=jpegs, temperature=0.0,
                                 max_tokens=8192)
        out = parse_json_reply(reply)
        out["note"] = "frame-sampled approximation (native video analysis unavailable)"
        return out

    # ------------------------------------------------------------------
    # only these produce explicit time BOUNDARIES; track/flow measure motion
    # but do not segment it
    BOUNDARY_TOOLS = ("primitives", "video_analysis", "motion")

    def _temporal_gate(self, raw: dict, step: int) -> str | None:
        """Multi-segment 'proceed' labels need measured time boundaries, not
        eyeballed ones. Fires at most twice, never near the step budget."""
        if self.temporal_gate_fires >= 2 or step >= MAX_STEPS - 2:
            return None
        feas = raw.get("feasibility") or {}
        segments = raw.get("segments") or []
        action_segs = [s for s in segments
                       if isinstance(s, dict) and s.get("skill") != "idle"]
        needs_timing = feas.get("recommendation") == "proceed" and action_segs
        if not needs_timing:
            return None
        has_boundaries = any(k in self.evidence for k in self.BOUNDARY_TOOLS)
        if not has_boundaries:
            self.temporal_gate_fires += 1
            return ("finalize deferred: your segments claim action timings but no "
                    "measured segment BOUNDARIES exist. Run `primitives` (best - "
                    "run pose/hands/track first so their channels are included) "
                    "or `video_analysis`, align your segment boundaries to the "
                    "measured changepoints/events, then finalize again.")
        # boundaries exist but only motion tools ran (no primitives): nudge
        # once toward the changepoint segmentation if pose/track channels exist
        if ("primitives" not in self.evidence and "video_analysis" not in self.evidence
                and self.temporal_gate_fires == 0
                and any(k in self.evidence for k in ("pose", "hands", "track"))):
            self.temporal_gate_fires += 1
            return ("before finalizing: motion channels are available from the "
                    "tools you ran - run `primitives` to get statistically "
                    "measured segment boundaries, then align your segments to "
                    "the changepoints (or justify deviations in a note).")
        return None

    # ------------------------------------------------------------------
    def _repair_labels(self, raw: dict, duration: float) -> dict | None:
        """Fill missing captions/feasibility/scene_tags from the evidence
        ledger, keeping the model's segments (its actual judgment). Returns
        validated labels or None if the segments themselves are unusable."""
        from .labeler import _validate_labels

        segments = raw.get("segments") or []
        if not segments or not all(
                isinstance(s, dict) and "skill" in s for s in segments):
            return None
        va = self.evidence.get("video_analysis", {})
        pose = self.evidence.get("pose", {})
        hands = self.evidence.get("hands", {})
        animal = self.evidence.get("animal_pose", {})
        seg_desc = "; ".join(f"{s.get('text', s.get('skill', ''))}" for s in segments[:4])
        summary = va.get("summary") or seg_desc or "unlabeled activity"

        caps = raw.get("captions") or {}
        caps.setdefault("short", summary[:120])
        caps.setdefault("medium", summary[:300])
        caps.setdefault("long", (summary + " Segments: " + seg_desc)[:800])
        raw["captions"] = caps

        feas = raw.get("feasibility") or {}
        animal_subject = animal.get("animal_present_ratio", 0) > 0.3
        if not feas.get("human_present"):
            if animal_subject and pose.get("person_present_ratio", 0) <= 0.5:
                feas["human_present"] = "none"  # quadruped clip, no human
            elif pose.get("person_present_ratio", 0) > 0.5:
                feas["human_present"] = "full_body"
            elif hands.get("hands_present_ratio", 0) > 0.3:
                feas["human_present"] = "hands_only"
            else:
                feas["human_present"] = "none"
        feas.setdefault("ai_generated_suspected",
                        bool(va.get("ai_generated_suspected", False)))
        feas.setdefault("ai_generated_artifacts",
                        list(va.get("ai_generated_artifacts", [])))
        # fields the agent never asserted -> conservative verdict
        feas.setdefault("recommendation", "human_review")
        feas.setdefault("confidence", 0.4)
        raw["feasibility"] = feas
        # never let the repair create a none/human-verb contradiction. For an
        # animal subject, the truthful fix is to coerce manipulation verbs to
        # locomotion (a dog can't 'grasp') and KEEP human_present='none' -
        # inventing a human to satisfy the verb would be a fabrication.
        _LOCO = {"idle", "walk", "trot", "gallop", "jump", "sit",
                 "lie_down", "stand", "turn"}
        if feas.get("human_present") == "none" and any(
                s.get("skill") not in _LOCO for s in segments):
            if animal_subject:
                loco = "walk" if animal.get("locomoting") else "stand"
                for s in segments:
                    if s.get("skill") not in _LOCO:
                        s["skill"] = loco
            else:
                feas["human_present"] = "partial"  # agent asserted human activity
            feas["recommendation"] = "human_review"
        try:
            labels = _validate_labels(raw, self.cfg.verbs, duration)
        except Exception:  # noqa: BLE001
            return None
        self.evidence.setdefault("notes", []).append(
            "labels partially auto-repaired: agent's segments kept; missing "
            "captions/feasibility fields composed from tool evidence")
        return labels

    # ------------------------------------------------------------------
    def _critic_frames(self) -> list[bytes]:
        """Fresh evenly-spaced frames so the critic can check labels against
        pixels, not just the text evidence dump."""
        try:
            jpegs, _, _ = T.sample_frames(self.ws.video_path, n=4, max_width=384)
            return jpegs
        except Exception:  # noqa: BLE001
            return []

    def _critic(self, labels: dict) -> dict:
        """Independent verification of labels vs the evidence ledger."""
        ev = {k: v for k, v in self.evidence.items() if k != "probe"}
        probe = self.evidence.get("probe", {})
        seg_view = [{"start_s": s.start_s, "end_s": s.end_s, "skill": s.skill,
                     "text": s.text, "evidence": s.evidence}
                    for s in labels["segments"]]
        prompt = (
            "You are a strict verification critic for video labels used in robot "
            f"learning. The video is {probe.get('duration_s', 0):.2f} s long at "
            f"{probe.get('fps', 0):.0f} fps ({probe.get('n_frames', 0)} frames) - "
            "use these ground-truth values, do not infer duration from frame "
            "counts at an assumed fps. "
            "Compare the LABELS against the EVIDENCE and list concrete "
            "problems: skills not supported by evidence; human_present inconsistent "
            "with pose/hands ratios or frame observations; segment times outside "
            "the video or contradicting motion/video-analysis events; overclaimed "
            "confidence; ai_generated_suspected=true with an EMPTY artifacts list "
            "(assertion without evidence); confidence<=0.1 despite decisive "
            "evidence (miscalibration works both ways); "
            "ignored conflicts between tools and visual observations. "
            "Minor wording issues are NOT problems. A tool-vs-visual conflict "
            "that the agent EXPLAINED in a note (e.g. gloves defeat the hand "
            "tracker, occlusion, tracker limitation) and that the frames "
            "support is NOT a problem - trust frames over failed detectors. "
            "If the labels are honest and evidence-grounded, accept.\n\n"
            f"EVIDENCE: {json.dumps(ev, default=str)[:9000]}\n\n"
            f"AGENT NOTES: {json.dumps(self.evidence.get('notes', []))[:1500]}\n\n"
            f"LABELS: segments={json.dumps(seg_view)} "
            f"feasibility={json.dumps(labels.get('feasibility', {}))}\n\n"
            'Reply with ONLY JSON: {"verdict": "accept"|"revise", "problems": ["..."]}'
        )
        try:
            # multimodal critic when the vision role can see: frames catch
            # fabrications that a text-only evidence dump would miss
            frames = self._critic_frames() if self.router.orchestrator_is_multimodal() else []
            if frames:
                prompt = ("The attached frames are evenly sampled from the video "
                          "being labeled - check the labels against them too.\n\n"
                          + prompt)
                reply = self.router.chat([{"role": "user", "content": prompt}],
                                         role="vision", temperature=0.0,
                                         images=frames, force_json=True)
            else:
                reply = self.router.chat([{"role": "user", "content": prompt}],
                                         role="fast", temperature=0.0,
                                         force_json=True)
            out = parse_json_reply(reply)
            verdict = out.get("verdict", "accept")
            problems = [str(p)[:300] for p in out.get("problems", [])][:6]
            return {"verdict": verdict if verdict in ("accept", "revise") else "accept",
                    "problems": problems}
        except Exception as e:  # noqa: BLE001
            self.log(f"[agent] critic unavailable ({e}); accepting labels")
            return {"verdict": "accept", "problems": [f"critic unavailable: {e}"]}

    def _record(self, step: int, action: str, args: dict, obs, dt: float) -> None:
        self.transcript.append({
            "step": step, "action": action, "args": args,
            "observation": json.loads(json.dumps(obs, default=str)) if isinstance(obs, dict)
            else str(obs)[:500],
            "seconds": round(dt, 2),
        })
