"""Empirical baseline: VLM-only labeling (single Gemini whole-video call, no
tools, no critic, no gates) scored against the same ground truth as the agent
loop. Writes qa_vlm_baseline.json."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2r.agentic.bench import load_fixture, score_clip  # noqa: E402
from v2r.config import V2RConfig  # noqa: E402
from v2r.syngen import gemini  # noqa: E402

LABEL_PROMPT = """You label videos for robot-learning training data.
Watch this entire video and return labels as JSON:
{"segments": [{"start_s": <num>, "end_s": <num>, "skill": "<verb>",
   "text": "<desc>"}...],
 "human_present": "full_body"|"partial"|"hands_only"|"none",
 "recommendation": "proceed"|"reject"|"human_review",
 "ai_generated_suspected": true|false,
 "confidence": 0..1}
Every skill MUST be one of: %s.
Segment the video by activity over time; use 'idle' for uneventful spans.
'proceed' only if usable for robot-learning; 'reject' if no usable subject."""

SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {"type": "array", "items": {"type": "object", "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "skill": {"type": "string"}, "text": {"type": "string"}},
            "required": ["start_s", "end_s", "skill", "text"]}},
        "human_present": {"type": "string"},
        "recommendation": {"type": "string"},
        "ai_generated_suspected": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["segments", "human_present", "recommendation",
                 "ai_generated_suspected", "confidence"],
}

cfg = V2RConfig.load(ROOT)
clips = load_fixture(ROOT / "tests/data/label_bench.yaml")
targets = sys.argv[1:] or ["aigen_cup", "pipette", "waves"]
out = {}
for name in targets:
    gt = clips[name]
    video = (ROOT / gt["video"]).resolve()
    t0 = time.time()
    try:
        raw = gemini.analyze_video(
            video, prompt=LABEL_PROMPT % json.dumps(cfg.verbs),
            response_schema=SCHEMA, api_key=gemini.get_api_key(ROOT))
        labels = gemini.extract_json(raw)
    except Exception as e:  # noqa: BLE001
        out[name] = {"error": str(e)[:300]}
        print(name, "FAILED:", str(e)[:200])
        continue
    dt = round(time.time() - t0, 1)
    pred = {"recommendation": labels.get("recommendation"),
            "human_present": labels.get("human_present"),
            "segments": labels.get("segments", [])}
    s = score_clip(gt, pred)
    s["seconds"] = dt
    s["ai_generated_suspected"] = labels.get("ai_generated_suspected")
    s["raw_segments"] = labels.get("segments", [])[:10]
    out[name] = s
    print(f"{name}: verdict_ok={s['verdict_ok']} recall={s['skill_recall']} "
          f"precision={s['skill_precision']} halluc={s['hallucinations']} "
          f"iou={s['boundary_iou']} ({dt}s)")

(ROOT / "qa_vlm_baseline.json").write_text(json.dumps(out, indent=2),
                                           encoding="utf-8")
print("written qa_vlm_baseline.json")
