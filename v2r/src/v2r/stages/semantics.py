"""Semantics (Stage G): changepoint subtask segmentation + captions.

Segmentation is a REAL algorithm (aperture + contact transitions -> labeled
subtasks with verbs from config/verbs.yaml), run identically on synthetic and
estimated kinematics. Real mode additionally uses Qwen2.5-VL (isolated env)
for caption/scene-tag quality; the local path uses templated captions.
"""

from __future__ import annotations

from ..labeling.kinematics import (
    captions_from_segments,
    default_scene_tags,
    segment_episode,
    segments_file,
)
from ..schema.io import read_table, write_json_model
from ..schema.models import SourceTag, StageStatus
from .base import Stage, StageContext, StageResult, register_stage, run_tool

TOOL = {
    "tool": "aperture+contact changepoints (local); Qwen2.5-VL (real env)",
    "repo": "huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    "commit": "v1.0.0-pinned",
}
ENV = "semantics"


@register_stage
class SemanticsStage(Stage):
    name = "semantics"

    def run(self, ctx: StageContext) -> StageResult:
        if ctx.mode == "real":
            return self._run_real(ctx)
        return self._run_local(ctx)

    def _run_local(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        missing = [p.name for p in (ws.hands_parquet, ws.contacts_parquet, ws.tracks_parquet)
                   if not p.is_file()]
        if missing:
            return StageResult(status=StageStatus.failed,
                               failure_reason=f"missing inputs: {missing}", **TOOL)
        hands = read_table(ws.hands_parquet)
        contacts = read_table(ws.contacts_parquet)
        tracks = read_table(ws.tracks_parquet)
        t = hands[(hands["joint_name"] == "wrist") & (hands["hand"] == "right")].sort_values("frame")["t"].to_numpy()
        if len(t) == 0:
            t = tracks.sort_values("frame")["t"].unique()

        segments = segment_episode(t, contacts, tracks, hands, ctx.cfg.verbs)

        bad = [s.skill for s in segments if s.skill not in ctx.cfg.verbs]
        if bad:
            return StageResult(status=StageStatus.failed,
                               failure_reason=f"skills outside vocabulary: {bad}", **TOOL)
        object_ids = sorted(tracks["object_id"].astype(str).unique())
        write_json_model(ws.segments_json,
                         segments_file(segments, method="aperture+contact changepoints",
                                       source=SourceTag.estimated))
        write_json_model(ws.captions_json, captions_from_segments(segments, object_ids))
        write_json_model(ws.scene_tags_json, default_scene_tags(
            SourceTag.synthesized if ctx.mode == "synthetic" else SourceTag.estimated))

        skills = [s.skill for s in segments]
        metrics = {
            "n_segments": len(segments),
            "skills": list(dict.fromkeys(skills)),
            "coverage_s": round(segments[-1].end_s - segments[0].start_s, 2) if segments else 0.0,
        }
        outputs = [ws.rel(ws.segments_json), ws.rel(ws.captions_json), ws.rel(ws.scene_tags_json)]

        # speech channel: (situation, utterance) pairs teach the robot what to
        # SAY in context. Best effort: needs a Gemini key + an audio track
        # (ingest preserves audio when it can copy the source through).
        try:
            from ..labeling.transcribe import transcribe_to_workspace

            tr = transcribe_to_workspace(ws, ctx.cfg)
            metrics["transcription"] = {k: tr[k] for k in
                                        ("available", "has_speech", "n_utterances",
                                         "audio_notes", "reason") if k in tr}
            if tr.get("available") and ws.utterances_json.is_file():
                outputs.append(ws.rel(ws.utterances_json))
        except Exception as e:  # noqa: BLE001 - transcription is optional
            metrics["transcription"] = {"available": False, "reason": str(e)[:150]}

        return StageResult(status=StageStatus.success, metrics=metrics, outputs=outputs, **TOOL)

    def _run_real(self, ctx: StageContext) -> StageResult:
        ws = ctx.ws
        env_dir = ctx.cfg.root / "envs" / ENV
        verbs_path = ctx.cfg.root / "config" / "verbs.yaml"
        cmd = [
            "python", str(env_dir / "tool_entry.py"),
            "--workspace", str(ws.root),
            "--verbs", str(verbs_path),
            "--temperature", "0",
        ]
        proc = run_tool(cmd, env_name=ENV, cwd=env_dir, timeout=7200)
        if proc.returncode != 0:
            return StageResult(status=StageStatus.failed,
                               failure_reason=(proc.stderr or proc.stdout)[-2000:], **TOOL)
        outputs = [ws.rel(ws.segments_json), ws.rel(ws.captions_json), ws.rel(ws.scene_tags_json)]
        return StageResult(status=StageStatus.success, metrics={"n_segments": -1}, outputs=outputs, **TOOL)
