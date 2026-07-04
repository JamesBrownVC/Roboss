#!/usr/bin/env python3
"""ViPE tool entry — invoked by V2R geometry stage in real mode."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="ViPE → V2R geometry contract")
    p.add_argument("--workspace", required=True)
    p.add_argument("--video", required=True)
    args = p.parse_args()
    ws = Path(args.workspace)
    if not Path(args.video).is_file():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1
    try:
        import vipe  # noqa: F401 — installed in vipe env
    except ImportError:
        print(
            "ViPE not installed in this env. See envs/vipe/README.md",
            file=sys.stderr,
        )
        return 2
    print("ViPE real inference not wired in this stub — install ViPE and extend tool_entry.py")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
