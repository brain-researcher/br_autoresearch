#!/usr/bin/env python3
"""Repair only null0 MVC's diagnosed second-half exhaustion from job46576234.

Validate and rescore its preserved converged first fit; freshly fit only the
second half with a 1000-sweep ceiling. All other inference settings and the
saved M0/MZ baseline remain unchanged. This selects no retry wave or gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
from run_single_process_recovery import validate_fit
import run_mz_null_rival as null
from summarize_mz_menu_wave import technical_cause


FAILED_JOB_ID = "46576234"
FAILED_REPORT = null.SOURCE.parent / f"20261005-M0-null-MVC-rival-{FAILED_JOB_ID}" / "mz_null_rival.json"
REPORT_NAME = "mz_null_mvc_convergence_repair.json"
PHASE = "M0_null_MVC_second_half_convergence_repair"
ORIGINAL_SWEEPS = 500
RUNTIME_SWEEPS = 1000


def repair_context(original, failed, source, people):
    """Require the selected failure after the caller validates the actual source."""
    if (not isinstance(failed, dict)
            or failed.get("analysis_id") != "behavioral_mechanism_kill_test"
            or failed.get("phase") != "M0_null_saved_representative_rival_diagnostics"
            or failed.get("generating_model") != "M0" or failed.get("fitted_model") != "MVC"
            or type(failed.get("replicate_index")) is not int or failed["replicate_index"] != 0
            or type(failed.get("seed")) is not int or failed["seed"] != 1300037
            or failed.get("job_id") != FAILED_JOB_ID
            or failed.get("source_job_id") != null.SOURCE_JOB_ID
            or failed.get("source_report") != null.SOURCE_REPORT
            or failed.get("source_dir") is None or Path(failed["source_dir"]).resolve() != null.SOURCE.resolve()
            or failed.get("source_finished_utc") != source["finished_utc"]
            or failed.get("implementation") != original
            or failed.get("status") != "technical_rival_failure"
            or failed.get("rival_diagnostics_complete") is not False):
        raise ValueError("Only the original null0 MVC46576234 failure on this exact source is selected")
    for field in ("started_utc", "finished_utc"):
        value = failed.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("The preserved failed attempt requires dated terminal provenance")
    if (any(failed.get(field) != "not_evaluated" for field in null.CLAIM_FIELDS)
            or failed.get("full_recovery_pass") is not None or failed.get("episode_terminal") is not False
            or any(failed.get(field) is not False for field in
                   ("wave_selected", "simulation_regenerated", "saved_M0_MZ_refitted",
                    "observed_choices_or_RTs_used", "old_synthetic_responses_used"))):
        raise ValueError("The failed attempt must retain its original diagnostic-only source scope")
    cause = technical_cause(failed, original)
    if cause["classification"] != "second_half_500_sweep_exhaustion":
        raise ValueError("Only evidenced second-half500-sweep exhaustion permits this repair")
    # Inventory validation used the failed fit's identities; now bind the reused
    # fit to the actual loaded source cohort and the unchanged original criteria.
    first = failed["fits"][0]
    validate_fit(first, "MVC", null.HALVES[0][0], null.HALVES[0][1], original, people)
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = RUNTIME_SWEEPS
    metadata = {
        "kind": "second_half_maximum_sweeps_only", "failed_report": str(FAILED_REPORT),
        "failed_job_id": FAILED_JOB_ID, "source_finished_utc": source["finished_utc"],
        "original_maximum_sweeps": ORIGINAL_SWEEPS, "runtime_maximum_sweeps": RUNTIME_SWEEPS,
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"] for fit in failed["fits"]],
        "diagnosed_cause": cause,
        "first_half_original_fit_elapsed_seconds": first["elapsed_seconds"],
        "first_half_score_handling": "recomputed_from_preserved_parameters_no_saved_MVC_score_agreement_claim",
        "new_first_half_fit_elapsed_seconds_meaning": "copy_return_overhead_not_original_fit_cost",
        "second_half_initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "nonconverged_checkpoint_state_reused": False,
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
    }
    return runtime, deepcopy(first), metadata


def run(args):
    require_compute()
    output = args.output_dir.resolve()
    for protected in (null.SOURCE.resolve(), FAILED_REPORT.parent.resolve()):
        if output == protected or protected in output.parents:
            raise FileExistsError("Preserve the source pilot and original failed MVC attempt")
    original_source_validator, original_fit_validator = null.validate_source, null.validate_fit
    original_fitter, original_writer = null.fit_partially_pooled, null.write_report
    original = runtime = preserved_first = metadata = None
    fitter_calls = 0

    def validate_source(source, truth, people, spec):
        nonlocal original, runtime, preserved_first, metadata
        if (spec["optimizer"]["maximum_sweeps"] != ORIGINAL_SWEEPS
                or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200):
            raise ValueError("Preserve the original500 implementation and fixed200 recovery plan")
        original_source_validator(source, truth, people, spec)
        original = deepcopy(spec)
        failed = json.loads(FAILED_REPORT.read_text())
        runtime, preserved_first, metadata = repair_context(original, failed, source, people)

    def fitter(training, model, rival, spec):
        nonlocal fitter_calls
        if model != "MVC" or rival != "fixed" or original is None or spec != original or metadata is None:
            raise ValueError("Only the validated null0 MVC repair can fit or reuse a half")
        fitter_calls += 1
        if fitter_calls == 1:
            return deepcopy(preserved_first)
        if fitter_calls == 2:
            return original_fitter(training, model, rival, runtime)
        raise ValueError("Only one preserved first fit and one fresh second fit are selected")

    def fit_validator(fit, model, train, held, spec, people):
        if model in ("M0", "MZ"):
            return original_fit_validator(fit, model, train, held, spec, people)
        if model != "MVC" or original is None or spec != original or metadata is None:
            raise ValueError("Unexpected model or specification in the null MVC repair")
        if (list(train), list(held)) == (list(null.HALVES[0][0]), list(null.HALVES[0][1])):
            selected = original
        elif (list(train), list(held)) == (list(null.HALVES[1][0]), list(null.HALVES[1][1])):
            selected = runtime
        else:
            raise ValueError("Only the original two primary directions are selected")
        return original_fit_validator(fit, model, train, held, selected, people)

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, selected_replicates=[0], selected_models=["MVC"],
                      wave_selected=False, retry_wave_selected=False, full_menu_wave_selected=False,
                      repair_context_validated=metadata is not None,
                      first_half_fit_reused=fitter_calls >= 1, fresh_second_half_fit_called=fitter_calls >= 2,
                      per_half_provenance=[
                          {"training_runs": [1, 4], "scoring_runs": [2, 3],
                           "fit_source": str(FAILED_REPORT), "action": "reuse_validated_fit_and_rescore",
                           "validation_maximum_sweeps": ORIGINAL_SWEEPS},
                          {"training_runs": [2, 3], "scoring_runs": [1, 4],
                           "fit_source": "current_repair", "action": "fresh_fit",
                           "validation_maximum_sweeps": RUNTIME_SWEEPS}],
                      interpretation="One diagnosed null0 MVC second-half numerical repair. Reuse and rescore only its validated original first fit; freshly fit the second half. No family, rate, interval, early-kill, full-menu wave or gate verdict.")
        if original is not None:
            report["implementation"] = original
        if metadata is not None:
            report.update(runtime_implementation=runtime, numerical_repair=metadata)
            if "elapsed_seconds" in report:
                report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        original_writer(directory, report)

    # Source and failure reads, validation, replay and all new work remain inside
    # the original runner's whole-process timer. Each binding is process-local.
    with patch.object(null, "validate_source", validate_source), \
            patch.object(null, "fit_partially_pooled", fitter), \
            patch.object(null, "validate_fit", fit_validator), \
            patch.object(null, "write_report", write_checkpoint), \
            patch.object(null, "REPORT_NAME", REPORT_NAME):
        return null.run(argparse.Namespace(model="MVC", source_dir=null.SOURCE,
            implementation=args.implementation, output_dir=output, max_seconds=args.max_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "rival_diagnostics_complete",
        "first_half_fit_reused", "fresh_second_half_fit_called", "elapsed_seconds",
        "cumulative_process_elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False))
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
