"""Mark a clip passed: set state/selected_take, append attempt + log entry,
and copy the winning take into selects/.

Usage:
    <ml python> mark_pass.py <clip_id> <take> <mean_soft> "<main_defect>" ["<repair_applied_or_empty>"]

Also supports recording a prior failed attempt (without changing final state):
    <ml python> mark_pass.py --fail <clip_id> <take> <mean_soft_or_null> "<main_defect>" "<repair_applied>"
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "manifest.json"


def load():
    return json.loads(MAN.read_text(encoding="utf-8"))


def save(m):
    MAN.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def find(m, cid):
    c = next((c for c in m["clips"] if c["id"] == cid), None)
    if not c:
        raise SystemExit(f"clip {cid} not found")
    return c


def main() -> int:
    args = sys.argv[1:]
    is_fail = False
    if args and args[0] == "--fail":
        is_fail = True
        args = args[1:]
    cid, take = args[0], int(args[1])
    mean = args[2]
    mean_val = None if mean in ("null", "None", "") else float(mean)
    main_defect = args[3] if len(args) > 3 else ""
    repair = args[4] if len(args) > 4 and args[4] else None

    m = load()
    c = find(m, cid)
    c.setdefault("attempts", [])
    c["attempts"].append({
        "take": take, "verdict": "fail" if is_fail else "pass",
        "main_defect": main_defect, "repair_applied": repair,
        "mean_soft_score": mean_val, "generation_model": "gemini-omni-flash-preview",
    })
    m.setdefault("log", []).append({
        "clip": cid, "take": take, "verdict": "fail" if is_fail else "pass",
        "main_defect": main_defect, "repair_applied": repair,
    })
    if not is_fail:
        c["state"] = "passed"
        c["selected_take"] = take
        src = ROOT / "clips" / cid / f"take_{take:02d}.mp4"
        dst = ROOT / "selects" / f"{cid}.mp4"
        shutil.copyfile(src, dst)
    else:
        c["state"] = "failed"
    save(m)
    print(json.dumps({"ok": True, "clip": cid, "take": take,
                      "verdict": "fail" if is_fail else "pass",
                      "select": None if is_fail else str(ROOT / "selects" / f"{cid}.mp4")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
