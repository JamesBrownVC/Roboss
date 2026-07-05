"""Bridge: run the V2R agentic labeler + demo renderer on a generated video.

This is the DIRECT visual pipeline (prompt -> Omni video -> Nemotron+Kimi
agent looks at the pixels -> annotated render), detached from the
deterministic v2r stage DAG which remains a research/CI harness.

Runs as subprocesses of the v2r virtualenv so the server process stays free
of heavy perception deps (mediapipe, ultralytics, matplotlib).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
V2R_ROOT = ROOT / "v2r"
V2R_PY = V2R_ROOT / ".venv" / "Scripts" / "python.exe"
LABEL_TIMEOUT_S = 1200
RENDER_TIMEOUT_S = 600


class AgenticBridgeError(RuntimeError):
    pass


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(V2R_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd: list[str], timeout: float, log: Callable[[str], None]) -> None:
    proc = subprocess.run(
        cmd, cwd=str(V2R_ROOT), env=_env(),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-600:]
        raise AgenticBridgeError(f"{Path(cmd[0]).name} exited {proc.returncode}: {tail}")


def agentic_label_and_render(
    video_path: Path,
    job_id: str,
    out_dir: Path,
    log: Callable[..., None],
) -> dict[str, Any]:
    """Label `video_path` with the agent loop, render the annotated video.

    Returns {"report": <agentic report dict>, "labeled_path": Path|None}.
    Raises AgenticBridgeError when the labeler itself fails.
    """
    if not V2R_PY.is_file():
        raise AgenticBridgeError(f"v2r venv python not found at {V2R_PY}")
    episode = "studio_" + re.sub(r"[^A-Za-z0-9_-]", "_", job_id)[:60]

    log("Agentic labeler: Nemotron orchestrator + Kimi critic investigating video",
        agent="labeler")
    _run([str(V2R_PY), "-m", "v2r.orchestrator.cli", "label",
          "--video", str(Path(video_path).resolve()),
          "--episode-id", episode, "--agent", "loop",
          "--root", str(V2R_ROOT)],
         LABEL_TIMEOUT_S, log)

    ws = V2R_ROOT / "workspaces" / episode
    report_path = ws / "qa" / "agentic_label_report.json"
    if not report_path.is_file():
        raise AgenticBridgeError("labeler finished but wrote no report")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    feas = report.get("feasibility", {})
    log(f"Agent verdict: {feas.get('recommendation')} "
        f"(conf {feas.get('confidence', 0):.2f}, "
        f"AI-artifacts: {feas.get('ai_generated_suspected')})",
        agent="labeler")

    labeled_path = None
    try:
        _run([str(V2R_PY), str(V2R_ROOT / "scripts" / "render_label_demo.py"),
              episode, episode], RENDER_TIMEOUT_S, log)
        rendered = ROOT / "demo" / "label_demo" / f"{episode}.mp4"
        if rendered.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            labeled_path = out_dir / f"{episode}_labeled.mp4"
            shutil.copy2(rendered, labeled_path)
    except Exception as exc:  # noqa: BLE001 - render is cosmetic, labels are not
        log(f"Annotated render failed (labels still valid): {exc}",
            level="warn", agent="labeler")

    return {"report": report, "labeled_path": labeled_path, "episode": episode}


def label_fields_from_report(report: dict[str, Any], ws_root: Path | None = None) -> dict[str, Any]:
    """Convert the agentic report into the label dict fields the UI shows."""
    feas = report.get("feasibility", {})
    episode = report.get("episode_id", "")
    segments: list[dict[str, Any]] = []
    utterances: list[dict[str, Any]] = []
    captions: dict[str, Any] = {}
    ws = (ws_root or (V2R_ROOT / "workspaces")) / episode if episode else None
    if ws is not None:
        seg_p = ws / "semantics" / "segments.json"
        cap_p = ws / "semantics" / "captions.json"
        utt_p = ws / "semantics" / "utterances.json"
        if seg_p.is_file():
            segments = json.loads(seg_p.read_text(encoding="utf-8")).get("segments", [])
        if cap_p.is_file():
            captions = json.loads(cap_p.read_text(encoding="utf-8"))
        if utt_p.is_file():
            utterances = json.loads(utt_p.read_text(encoding="utf-8")).get("utterances", [])
    return {
        "video_summary": captions.get("short", ""),
        "summary": captions.get("medium", ""),
        "labels": sorted({s.get("skill", "") for s in segments} - {""}),
        "segments": segments,
        "utterances": utterances,
        "agent": {
            "judge": report.get("judge_source"),
            "recommendation": feas.get("recommendation"),
            "confidence": feas.get("confidence"),
            "ai_generated_suspected": feas.get("ai_generated_suspected"),
            "ai_generated_artifacts": feas.get("ai_generated_artifacts", []),
            "human_present": feas.get("human_present"),
            "steps": (report.get("plan") or {}).get("steps"),
            "critic": ((report.get("plan") or {}).get("critic") or {}).get("verdict"),
        },
    }
