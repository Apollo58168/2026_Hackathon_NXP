#!/usr/bin/env python3
"""SmartDrawer entry point.

The current deliverable is the PC simulation vertical slice.  Hardware
capture/TFLite/HDMI adapters are intentionally not faked here.
"""
from __future__ import annotations

import argparse
import json

from .simulator import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartDrawer simulation-first runner")
    parser.add_argument("--simulate", action="store_true", help="run the deterministic vertical slice")
    args = parser.parse_args()
    if not args.simulate:
        parser.error("hardware runtime is not included yet; use --simulate")
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
