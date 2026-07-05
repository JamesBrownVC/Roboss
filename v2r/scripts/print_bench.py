"""Pretty-print qa_bench_report.json (and optional VLM baseline comparison)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "qa_bench_report.json").read_text(encoding="utf-8"))
print("AGGREGATE:", json.dumps(r["aggregate"], indent=1))
print()
for name, s in r["per_clip"].items():
    if "error" in s:
        print(f"{name:18} ERROR {s['error'][:90]}")
        continue
    print(f"{name:18} v={int(s['verdict_ok'])} h={int(s['human_ok'])} "
          f"rec={int(s['skill_recall'])} prec={s['skill_precision']:.2f} "
          f"halluc={s['hallucinations']} iou={s['boundary_iou']} "
          f"ev={s['evidence_coverage']} ({s.get('seconds', 0):.0f}s)")
    p = s["predicted"]
    print(f"{'':18} pred: {p['recommendation']} {p['human_present']} {p['skills']}")

vb = ROOT / "qa_vlm_baseline.json"
if vb.is_file():
    d = json.loads(vb.read_text(encoding="utf-8"))
    ok = [k for k, v in d.items() if "error" not in v]
    n = len(ok)
    if n:
        print("\nVLM-ONLY BASELINE aggregate:")
        print(" verdict_accuracy:", round(sum(d[k]["verdict_ok"] for k in ok) / n, 3))
        print(" skill_recall:", round(sum(d[k]["skill_recall"] for k in ok) / n, 3))
        print(" precision_mean:", round(sum(d[k]["skill_precision"] for k in ok) / n, 3))
        print(" clips_with_halluc:", sum(1 for k in ok if d[k]["hallucinations"]))
        ious = [d[k]["boundary_iou"] for k in ok if d[k]["boundary_iou"] is not None]
        print(" boundary_iou_mean:", round(sum(ious) / len(ious), 3) if ious else None)
        secs = [d[k]["seconds"] for k in ok]
        print(" seconds_mean:", round(sum(secs) / len(secs), 1))
