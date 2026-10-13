#!/usr/bin/env python3
"""Fit one selected MC or MV component on a saved indexed M0-null baseline.

Use the original null runner and indexed baseline validator without changing
the pending MVZ adapter or any scientific helper. Source validation, baseline
score replay and two new fits remain inside the original bounded process.
The coordinator owns component selection and terminal-source qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_replicate as indexed
import run_mz_null_rival as null
import summarize_mz_null_baseline_wave as baseline


FIRST_ARRAY = 46579924
REMAINING_ARRAY = 46580387
MODELS = ("MC", "MV")
PHASE = "M0_null_saved_indexed_simple_rival_diagnostics"
REPORT_NAME = "mz_null_indexed_simple_rival.json"
PRESERVED_DIRECTORIES = (
    indexed.PILOT_DIRECTORY,
    baseline.RUNS / "20261005-M0-null-MC-rival-46577173",
    baseline.RUNS / "20261005-M0-null-MV-rival-46577173",
    baseline.RUNS / "20261005-M0-null-MVZ-rival-46575931",
    baseline.RUNS / "20261005-M0-null-MVC-rival-46576234",
    baseline.RUNS / f"20261005-M0-null-baseline-array-{FIRST_ARRAY}",
    baseline.RUNS / f"20261005-M0-null-baseline-array-{REMAINING_ARRAY}",
)


def source_directory(index):
    indexed.null_seed(index)
    if index == 0:
        raise ValueError("Preserve null pilot0; this adapter accepts only indexed cohorts1 through199")
    return baseline.source_directory(index, FIRST_ARRAY, REMAINING_ARRAY)


def run(args):
    require_compute()
    source = source_directory(args.replicate_index)
    if args.model not in MODELS or args.source_dir.resolve() != source.resolve():
        raise ValueError("Select only MC or MV and this index's exact original null baseline array source")
    output = args.output_dir.resolve()
    if any(output == directory or directory in output.parents for directory in PRESERVED_DIRECTORIES):
        raise FileExistsError("Preserve the baseline arrays and original null0 representatives")
    original_writer = null.write_report
    source_job = None

    def validate_indexed_source(report, truth, people, spec):
        nonlocal source_job
        if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
                or spec["optimizer"]["maximum_sweeps"] != 500
                or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != indexed.FIXED_COHORTS
                or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
                or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
            raise ValueError("Preserve the original500 implementation and fixed200 null baseline cell")
        baseline.completed_diagnostic(report, truth, people, args.replicate_index, spec)
        source_job = str(report["job_id"])

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, source_job_id=source_job, source_baseline_validated=source_job is not None,
                      selected_replicates=[args.replicate_index], selected_models=[args.model],
                      fixed_cohort_count=indexed.FIXED_COHORTS, component_scope=f"{args.model}_only",
                      wave_selected=True, wave_scope=f"{args.model}_component_only",
                      full_menu_wave_selected=False, retry_selected=False,
                      interpretation=f"One {args.model} component on an unchanged indexed M0-null cohort. Saved M0/MZ scores are replayed and retained; no simulation or baseline fit is regenerated. No full-menu selection, family, false-selection rate, interval, gate or early-kill decision is supplied.",
                      process_state_boundary="The coordinator separately establishes terminal source state; runtime source validation is not scheduler evidence.")
        original_writer(directory, report)

    # Bind only the unchanged null runner inside this single-worker process.
    # Source failures retain checkpoints, and the true worker ID is recorded
    # only after validation. No array-parent ID substitutes for that identity.
    with patch.object(null, "SOURCE", source), \
            patch.object(null, "SOURCE_REPORT", indexed.REPORT_NAME), \
            patch.object(null, "SOURCE_JOB_ID", None), \
            patch.object(null, "SEED", indexed.null_seed(args.replicate_index)), \
            patch.object(null, "REPLICATE_INDEX", args.replicate_index), \
            patch.object(null, "REPORT_NAME", REPORT_NAME), \
            patch.object(null, "validate_source", validate_indexed_source), \
            patch.object(null, "write_report", write_checkpoint):
        return null.run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in
                     ("phase", "status", "fitted_model", "replicate_index", "seed", "source_job_id",
                      "source_baseline_validated", "rival_diagnostics_complete", "diagnostics", "elapsed_seconds", "error")
                     if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
