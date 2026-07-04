#!/usr/bin/env python3
"""WiLoR/HaMeR tool entry — MANO → EgoDex 25-joint SE(3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--mano", required=True)
    p.add_argument("--format", default="egodex25")
    args = p.parse_args()
    if not Path(args.mano).is_dir():
        print(f"MANO dir missing: {args.mano}", file=sys.stderr)
        return 1
    print("WiLoR real inference — extend tool_entry.py after env install", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
