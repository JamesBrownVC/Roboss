#!/usr/bin/env python3
"""FoundationPose + detection stack tool entry."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--permissive-only", default="false")
    args = p.parse_args()
    if args.permissive_only.lower() == "true":
        print("Using permissive Open3D ICP fallback — extend tool_entry.py", file=sys.stderr)
    else:
        print("FoundationPose real inference — extend tool_entry.py", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
