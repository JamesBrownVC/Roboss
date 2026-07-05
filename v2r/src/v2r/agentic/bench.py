"""Label-accuracy regression bench: runs the agent loop over a hand-labeled
ground-truth fixture and reports quantified per-axis accuracy - the numbers a
data buyer sees instead of letter grades.

Scoring axes:
  verdict accuracy    - recommendation within the acceptable set
  human accuracy      - human_present within the acceptable set
  skill recall        - >=1 expected skill appears in predicted non-idle segments
  skill precision     - predicted non-idle skills within the allowed set
  hallucinations      - occurrences of forbidden skills (fabrication; hard fail)
  boundary IoU / MAE  - temporal overlap of predicted vs GT key segments
  evidence coverage   - non-idle segments carrying a provenance string
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from ..config import V2RConfig


# ---------------------------------------------------------------------------
# pure scoring (unit-testable)
# ---------------------------------------------------------------------------


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 1e-9 else 0.0


def score_clip(gt: dict, predicted: dict) -> dict:
    """Score one clip's predicted labels against its ground truth.

    `predicted`: {"recommendation", "human_present",
                  "segments": [{"start_s","end_s","skill","evidence"}...]}
    """
    segs = predicted.get("segments", [])
    pred_skills = {s["skill"] for s in segs}
    action_segs = [s for s in segs if s["skill"] != "idle"]
    action_skills = {s["skill"] for s in action_segs}

    verdict_ok = predicted.get("recommendation") in gt["acceptable_verdicts"]
    human_ok = predicted.get("human_present") in gt["acceptable_human"]

    expected = set(gt.get("expected_skills", []))
    allowed = set(gt.get("allowed_skills", [])) | {"idle"}
    forbidden = set(gt.get("forbidden_skills", []))

    if expected:
        recall_hits = expected & pred_skills
        skill_recall = len(recall_hits) > 0
    else:
        # negative clip: recall = correctly predicting no action
        skill_recall = len(action_skills) == 0
    if action_skills:
        skill_precision = len(action_skills & allowed) / len(action_skills)
    else:
        skill_precision = 1.0 if not expected else 0.0
    hallucinations = sorted(pred_skills & forbidden)

    # boundary scoring against key segments
    ious, maes = [], []
    for key in gt.get("key_segments", []):
        want = set(key["skills"])
        matches = [s for s in segs if s["skill"] in want]
        if not matches:
            ious.append(0.0)
            continue
        # union interval of matching predictions vs GT interval
        m0 = min(s["start_s"] for s in matches)
        m1 = max(s["end_s"] for s in matches)
        ious.append(_interval_iou(key["start_s"], key["end_s"], m0, m1))
        maes.append((abs(m0 - key["start_s"]) + abs(m1 - key["end_s"])) / 2)

    evidence_cov = (sum(1 for s in action_segs if (s.get("evidence") or "").strip())
                    / len(action_segs)) if action_segs else None

    return {
        "verdict_ok": bool(verdict_ok),
        "human_ok": bool(human_ok),
        "skill_recall": bool(skill_recall),
        "skill_precision": round(float(skill_precision), 3),
        "hallucinations": hallucinations,
        "boundary_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "boundary_mae_s": round(sum(maes) / len(maes), 2) if maes else None,
        "evidence_coverage": (round(evidence_cov, 2)
                              if evidence_cov is not None else None),
        "predicted": {
            "recommendation": predicted.get("recommendation"),
            "human_present": predicted.get("human_present"),
            "skills": sorted(pred_skills),
        },
    }


def aggregate_scores(per_clip: dict[str, dict]) -> dict:
    rows = list(per_clip.values())
    n = len(rows)
    if not n:
        return {}
    ious = [r["boundary_iou"] for r in rows if r["boundary_iou"] is not None]
    maes = [r["boundary_mae_s"] for r in rows if r["boundary_mae_s"] is not None]
    evs = [r["evidence_coverage"] for r in rows
           if r["evidence_coverage"] is not None]
    return {
        "n_clips": n,
        "verdict_accuracy": round(sum(r["verdict_ok"] for r in rows) / n, 3),
        "human_present_accuracy": round(sum(r["human_ok"] for r in rows) / n, 3),
        "skill_recall": round(sum(r["skill_recall"] for r in rows) / n, 3),
        "skill_precision_mean": round(
            sum(r["skill_precision"] for r in rows) / n, 3),
        "clips_with_hallucinations": sum(1 for r in rows if r["hallucinations"]),
        "boundary_iou_mean": round(sum(ious) / len(ious), 3) if ious else None,
        "boundary_mae_s_mean": round(sum(maes) / len(maes), 2) if maes else None,
        "evidence_coverage_mean": round(sum(evs) / len(evs), 2) if evs else None,
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def load_fixture(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["clips"]


def read_predictions(cfg: V2RConfig, episode_id: str) -> Optional[dict]:
    ws_root = Path(cfg.workspaces_root) / episode_id
    seg_path = ws_root / "semantics" / "segments.json"
    rep_path = ws_root / "qa" / "agentic_label_report.json"
    if not seg_path.is_file() or not rep_path.is_file():
        return None
    segs = json.loads(seg_path.read_text(encoding="utf-8"))
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    feas = rep.get("feasibility", {})
    return {
        "recommendation": feas.get("recommendation"),
        "human_present": feas.get("human_present"),
        "segments": segs.get("segments", []),
    }


def run_bench(
    cfg: V2RConfig,
    fixture: Path,
    prefix: str = "bench",
    agent: str = "loop",
    only: Optional[list[str]] = None,
    reuse: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    from .labeler import run_agentic_labeler

    clips = load_fixture(fixture)
    if only:
        clips = {k: v for k, v in clips.items() if k in only}
    per_clip: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name, gt in clips.items():
        episode_id = f"{prefix}_{name}_000000"
        video = (cfg.root / gt["video"]).resolve()
        if not video.is_file():
            log(f"[bench] {name}: MISSING video {video}")
            per_clip[name] = {"error": f"missing video {video}"}
            continue
        ws_root = Path(cfg.workspaces_root) / episode_id
        if ws_root.exists() and not reuse:
            shutil.rmtree(ws_root, ignore_errors=True)
        t0 = time.time()
        if not (reuse and ws_root.exists()):
            try:
                run_agentic_labeler(cfg, video, episode_id=episode_id,
                                    agent=agent, log=log)
            except Exception as e:  # noqa: BLE001 - score what we can
                log(f"[bench] {name}: run failed: {e}")
        timings[name] = round(time.time() - t0, 1)
        pred = read_predictions(cfg, episode_id)
        if pred is None:
            per_clip[name] = {"error": "no predictions written"}
            continue
        per_clip[name] = score_clip(gt, pred)
        per_clip[name]["seconds"] = timings[name]
        log(f"[bench] {name}: verdict_ok={per_clip[name]['verdict_ok']} "
            f"recall={per_clip[name]['skill_recall']} "
            f"precision={per_clip[name]['skill_precision']} "
            f"halluc={per_clip[name]['hallucinations']} "
            f"iou={per_clip[name]['boundary_iou']}")

    scored = {k: v for k, v in per_clip.items() if "error" not in v}
    report = {
        "fixture": str(fixture),
        "agent": agent,
        "aggregate": aggregate_scores(scored),
        "per_clip": per_clip,
    }
    out = cfg.root / "qa_bench_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"[bench] report written: {out}")
    return report
