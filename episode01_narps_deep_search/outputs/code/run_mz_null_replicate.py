#!/usr/bin/env python3
"""One of the fixed remaining M0/MZ null baseline cohorts, indices1 through199.

Reuse the original representative's simulation, four fits and checkpoints.
Only this process's seed, index, report name and checkpoint metadata differ.
Baseline-wave completion does not select a family, a full menu or a gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_representative as representative


BASE_SEED = 1300037
SEED_STRIDE = 10007
FIXED_COHORTS = 200
REPORT_NAME = "mz_null_replicate.json"
PHASE = "M0_null_MZ_pairwise_replicate"
WAVE_SCOPE = "M0_MZ_baseline_only"
PILOT_DIRECTORY = Path(__file__).resolve().parents[2] / "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-representative-46574266"


def null_seed(index):
    """The fixed schedule includes preserved pilot0; new execution excludes it."""
    if type(index) is not int or not 0 <= index < FIXED_COHORTS:
        raise ValueError("The null cell has exactly the fixed indices0 through199")
    return BASE_SEED + SEED_STRIDE * index


def run(args):
    require_compute()
    seed = null_seed(args.replicate_index)
    if args.replicate_index == 0:
        raise ValueError("Preserve the completed null pilot0; select a remaining index1 through199")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("Each null baseline worker requires exactly one allocated CPU")
    output = args.output_dir.resolve()
    if output == PILOT_DIRECTORY or PILOT_DIRECTORY in output.parents:
        raise FileExistsError("Preserve the completed pilot0 directory")
    original_writer = representative.write_report

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, wave_selected=True, wave_scope=WAVE_SCOPE,
                      fixed_cohort_count=FIXED_COHORTS, selected_replicates=[args.replicate_index],
                      full_menu_wave_selected=False, retry_selected=False,
                      interpretation="One fixed M0-null cohort with both M0/MZ baseline directions. Only the baseline wave is selected. Pairwise qualification is not full-menu false selection, a false-selection rate or an early-kill decision; no family, interval or gate verdict is computed.")
        original_writer(directory, report)

    # Each worker is a separate one-CPU process. The unchanged runner retains
    # exact design/specification, thread, cap, fresh-output and technical guards.
    # Restore every binding on return or exception; no shared files are edited.
    with patch.object(representative, "SEED", seed), \
            patch.object(representative, "REPLICATE_INDEX", args.replicate_index), \
            patch.object(representative, "REPORT_NAME", REPORT_NAME), \
            patch.object(representative, "write_report", write_checkpoint):
        return representative.run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--design-events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in
                     ("phase", "status", "generating_model", "replicate_index", "seed",
                      "pairwise_diagnostic_complete", "wave_scope", "nested_comparison", "elapsed_seconds", "error")
                     if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["pairwise_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
