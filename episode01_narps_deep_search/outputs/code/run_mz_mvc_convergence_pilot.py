#!/usr/bin/env python3
"""Fresh MVC fit for saved MZ index42 with a recorded 500-to-1000 sweep repair.

Only this process's references in the tested saved-rival runner are adapted.
The shared fitter, original implementation and all source attempts remain
unchanged. No nonconverged optimizer state is reused and no retry wave or
scientific gate is selected.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_saved_replicate_rival as saved


INDEX = 42
MODEL = "MVC"
ORIGINAL_SWEEPS = 500
RUNTIME_SWEEPS = 1000
FAILED_REPORT = saved.RUNS_ROOT / "20261005-MZ-MVC-saved-array-46572753/replicate42/mz_saved_replicate_rival.json"
REPORT_NAME = "mz_mvc_convergence_pilot.json"
PHASE = "MZ_MVC_fixed_index_convergence_pilot"


def repair_context(original, failed):
    """Accept only the recorded numerical exhaustion, without using its state."""
    if original["optimizer"]["maximum_sweeps"] != ORIGINAL_SWEEPS:
        raise ValueError("The preserved implementation must retain its original 500-sweep ceiling")
    if (failed.get("analysis_id") != "behavioral_mechanism_kill_test"
            or failed.get("phase") != "MZ_SESOI_saved_replicate_rival_diagnostics"
            or failed.get("generating_model") != "MZ" or failed.get("fitted_model") != MODEL
            or type(failed.get("replicate_index")) is not int or failed["replicate_index"] != INDEX
            or type(failed.get("seed")) is not int or failed["seed"] != saved.cell_seed("MZ", INDEX)
            or Path(failed.get("source_dir", "")).resolve() != saved.source_directory(INDEX).resolve()
            or failed.get("implementation") != original
            or failed.get("status") != "technical_rival_failure"
            or failed.get("rival_diagnostics_complete") is not False
            or failed.get("scored_primary_directions") != []
            or failed.get("completed_rivals") != [] or "participant_scores" in failed):
        raise ValueError("Use only the preserved unscored MVC index42 technical attempt")
    fits = failed.get("fits", [])
    if (len(fits) != 1 or fits[0].get("model") != MODEL or fits[0].get("rival") != "fixed"
            or fits[0].get("training_runs") != [1, 4] or fits[0].get("scoring_runs") != [2, 3]
            or fits[0].get("converged") is not False or fits[0].get("subject_optimizer_failures") != []
            or len(fits[0].get("objective_history", [])) != ORIGINAL_SWEEPS):
        raise ValueError("The selected failure must be first-half outer-sweep exhaustion at 500 sweeps")
    for value in (failed.get("elapsed_seconds"), fits[0].get("elapsed_seconds")):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Preserve finite elapsed cost from the failed attempt and fit")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = RUNTIME_SWEEPS
    metadata = {
        "kind": "maximum_sweeps_only", "original_maximum_sweeps": ORIGINAL_SWEEPS,
        "runtime_maximum_sweeps": RUNTIME_SWEEPS, "failed_report": str(FAILED_REPORT),
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": fits[0]["elapsed_seconds"],
        "previous_fit_sweeps": ORIGINAL_SWEEPS,
        "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "nonconverged_checkpoint_state_reused": False,
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
    }
    return runtime, metadata


def run(args):
    require_compute()
    if args.implementation.resolve() != saved.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The whole-process cap must be 1 through 5400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("This pilot requires exactly one allocated CPU")
    output = args.output_dir.resolve()
    if output.exists() or FAILED_REPORT.parent in output.parents or output == FAILED_REPORT.parent:
        raise FileExistsError("Use a fresh output outside the preserved failed attempt")
    original = json.loads(args.implementation.read_text())
    failed = json.loads(FAILED_REPORT.read_text())
    runtime, metadata = repair_context(original, failed)
    original_fitter, original_validator, original_writer = saved.fit_partially_pooled, saved.validate_fit, saved.write_report

    def fit_with_runtime(training, model, rival, spec):
        if model != MODEL or rival != "fixed" or spec != original:
            raise ValueError("The numerical override is limited to this MVC fit under the original specification")
        # The unchanged fitter initializes from its original centers/offsets.
        return original_fitter(training, model, rival, runtime)

    def validate_with_runtime(fit, model, train, held, spec, people):
        if spec != original or model not in ("M0", "MZ", MODEL):
            raise ValueError("Validate original baselines and this MVC repair only")
        return original_validator(fit, model, train, held, runtime if model == MODEL else original, people)

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, implementation=original, runtime_implementation=runtime,
                      numerical_repair=metadata, selected_replicates=[INDEX], retry_wave_selected=False,
                      full_menu_false_selection="not_evaluated", false_selection_rate="not_evaluated")
        if "elapsed_seconds" in report:
            report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        original_writer(directory, report)

    # This standalone process has one worker. Restore all references even on
    # exceptions; other jobs' modules and all on-disk code remain untouched.
    with patch.object(saved, "fit_partially_pooled", fit_with_runtime), \
            patch.object(saved, "validate_fit", validate_with_runtime), \
            patch.object(saved, "write_report", write_checkpoint), \
            patch.object(saved, "REPORT_NAME", REPORT_NAME):
        return saved.run(argparse.Namespace(model=MODEL, replicate_index=INDEX,
                                           source_dir=saved.source_directory(INDEX),
                                           implementation=args.implementation, output_dir=output,
                                           max_seconds=args.max_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in
                     ("phase", "status", "replicate_index", "seed", "rival_diagnostics_complete",
                      "numerical_repair", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error")
                     if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
