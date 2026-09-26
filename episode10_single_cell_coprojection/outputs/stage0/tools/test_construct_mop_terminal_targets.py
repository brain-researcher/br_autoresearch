#!/usr/bin/env python3
"""Dependency-free synthetic regression test for the EP10 constructor."""

from __future__ import annotations

import json

from construct_mop_terminal_targets import run_self_test


def main() -> int:
    result = run_self_test()
    if result["status"] != "pass":
        raise AssertionError("EP10 terminal-target synthetic tests did not pass")
    if result["uses_real_atlas_or_swc"] is not False:
        raise AssertionError("synthetic tests accessed a real atlas or SWC")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
