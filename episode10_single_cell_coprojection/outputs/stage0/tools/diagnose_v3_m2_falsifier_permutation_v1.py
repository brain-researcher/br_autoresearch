#!/usr/bin/env python3
"""Run only the frozen failed timing probe, retaining its scalar failure."""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_v3_m2_falsifier_timing_v1 as timing
import validate_v3_m1_numerical_execution_v3 as execution_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    timing.require_slurm_one_thread()
    root = args.output_root
    if not root.is_dir() or root.is_symlink():
        raise timing.FalsifierTimingError("diagnostic root must be a fresh job directory")
    if any(root.iterdir()):
        raise timing.FalsifierTimingError("diagnostic root must be empty")
    backend = dict(execution_policy.numeric_backend())
    timing._write_json_atomic(root / "numeric_backend.json", backend)
    result = timing._run_probe("permutation", str(root / "falsifier_timing_checkpoints"))
    timing.assert_aggregate_safe(result)
    timing._write_json_atomic(root / "permutation_only_result.json", result)
    print(f"Frozen permutation timing probe: {result['status']}", flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
