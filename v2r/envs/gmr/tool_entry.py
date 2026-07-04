#!/usr/bin/env python3
"""GMR retarget tool entry."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--robot-class", required=True)
    p.add_argument("--smplx", required=True)
    args = p.parse_args()
    if not Path(args.smplx).is_file():
        print(f"smplx.npz missing: {args.smplx}", file=sys.stderr)
        return 1
    print("GMR real retarget — extend tool_entry.py after env install", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
