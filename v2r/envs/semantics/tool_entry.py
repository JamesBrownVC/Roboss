#!/usr/bin/env python3
"""Qwen-VL semantics tool entry."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--verbs", required=True)
    p.add_argument("--temperature", default="0")
    args = p.parse_args()
    if not Path(args.verbs).is_file():
        print(f"verbs file missing: {args.verbs}", file=sys.stderr)
        return 1
    print("Qwen-VL real inference — extend tool_entry.py after vLLM install", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
