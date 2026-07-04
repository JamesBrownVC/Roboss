#!/usr/bin/env python3
"""GVHMR tool entry — SMPL-X + Umeyama alignment to ViPE."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--body-models", required=True)
    p.add_argument("--align-vipe", action="store_true")
    args = p.parse_args()
    bm = Path(args.body_models)
    if not any(bm.glob("*.npz")) and not (bm / "SMPLX_NEUTRAL.npz").exists():
        print(f"SMPL-X not found under {bm}", file=sys.stderr)
        return 1
    print("GVHMR real inference — extend tool_entry.py after env install", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
