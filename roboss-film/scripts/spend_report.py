import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

total_attempts = 0
total_generated_clips = 0
total_use_seconds = 0
per_act = {}
rows = []
for c in m["clips"]:
    if c.get("type") == "generated":
        total_generated_clips += 1
        n = len(c.get("attempts", []))
        total_attempts += n
        total_use_seconds += c.get("use_seconds") or 0
        act = c.get("act")
        d = per_act.setdefault(act, {"clips": 0, "attempts": 0, "seconds": 0})
        d["clips"] += 1
        d["attempts"] += n
        d["seconds"] += c.get("use_seconds") or 0
        rows.append((c["id"], act, n, c.get("state"), c.get("selected_take")))

print("TOTAL generated clips:", total_generated_clips)
print("TOTAL generation attempts (API calls):", total_attempts)
print("TOTAL used seconds (sum of use_seconds, generated only):", total_use_seconds)
print()
for act, d in sorted(per_act.items()):
    print(f"Act {act}: {d['clips']} clips, {d['attempts']} attempts, {d['seconds']}s used")
print()
overbudget = [r for r in rows if r[2] > 4]
print("Clips that exceeded the 4-attempt guardrail:")
for r in overbudget:
    print(" ", r)
not_gen = [c["id"] for c in m["clips"] if c.get("type") != "generated"]
print()
print("Non-generated (slate/live_action) placeholders:", not_gen)
print()
print("All clip states:")
for c in m["clips"]:
    print(f"  {c['id']:12s} act={str(c.get('act')):5s} type={c.get('type'):12s} state={c.get('state')}")
