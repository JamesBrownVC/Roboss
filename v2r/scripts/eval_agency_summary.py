"""Summarize agent-loop transcripts: tool sequences, providers, verdicts."""
import json
import sys
from collections import Counter
from pathlib import Path

WS = Path(__file__).resolve().parents[1] / "workspaces"

runs = sys.argv[1:] or [
    "agentloop_lab_000000", "agentloop_veo_000000", "agentloop_crusoe_000000",
    "eval_dance_000000", "eval_dance_000001", "eval_dance_000002",
    "eval_nohuman_000000", "eval_nohuman_000001", "eval_nohuman_000002",
    "eval_timelapse_000000", "eval_novel_000000", "eval_aigen_000001",
]
for rid in runs:
    qa = WS / rid / "qa"
    try:
        t = json.loads((qa / "agentic_transcript.json").read_text(encoding="utf-8"))
        r = json.loads((qa / "agentic_label_report.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"{rid:32} MISSING")
        continue
    seq = []
    for s in t["transcript"]:
        a = s["action"]
        if a == "run_tool":
            seq.append(s["args"].get("name", "?"))
        elif a in ("look", "video_analysis", "note"):
            seq.append(a)
        elif a == "finalize":
            seq.append("FIN")
        elif a == "parse_error":
            seq.append("(perr)")
    provs = Counter(st["provider"] for st in t["llm_stats"])
    feas = r["feasibility"]
    print(f"{rid:32} {len(t['transcript']):2} steps  {dict(provs)}")
    print(f"  {' > '.join(seq)}")
    print(f"  -> {feas.get('human_present'):10} {feas.get('recommendation'):13} "
          f"conf={feas.get('confidence')} "
          f"critic={r['plan'].get('critic', {}).get('verdict')} "
          f"rev={r['plan'].get('revisions')}")
