#!/usr/bin/env python3
"""Qwen-VL feasibility judge tool entry (real mode)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--mode", default="real")
    args = p.parse_args()
    ws = Path(args.workspace)
    report_path = ws / "qa" / "feasibility_report.json"
    if report_path.is_file():
        print(f"feasibility report exists: {report_path}")
        return 0
    # Placeholder: orchestrator runs rule-based fallback when VLM not installed
    print("Qwen-VL feasibility judge — extend after vLLM install", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
