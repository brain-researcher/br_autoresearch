#!/usr/bin/env python3
"""Complete only MZ42 MVC's diagnosed reverse-half 1000-sweep exhaustion.

Rescore the preserved first fit under its original runtime1000 specification;
freshly fit the reverse half with runtime1500. The original500 implementation,
saved cohort, baseline fits, two prior attempts and shared runners stay intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

from qualify_assets import require_compute
from run_single_process_recovery import validate_fit
import run_mz_mvc_convergence_pilot as pilot
import run_mz_saved_replicate_rival as saved
import summarize_mz_menu_wave as wave


INDEX, MODEL = 42, "MVC"
ORIGINAL_SWEEPS, FIRST_SWEEPS, REVERSE_SWEEPS = 500, 1000, 1500
ORIGINAL_REPORT = pilot.FAILED_REPORT
ORIGINAL_JOB_ID = "46572920"
ORIGINAL_FINISHED_UTC = "2026-10-05T05:03:24.901836+00:00"
PILOT_JOB_ID = "46577369"
PILOT_REPORT = saved.RUNS_ROOT / f"20261005-MZ42-MVC-convergence-{PILOT_JOB_ID}" / pilot.REPORT_NAME
PILOT_FINISHED_UTC = "2026-10-05T06:51:35.390940+00:00"
SOURCE_FINISHED_UTC = "2026-10-05T00:51:12.784922+00:00"
REPORT_NAME = "mz_mvc_reverse_completion.json"
PHASE = "MZ42_MVC_reverse_completion"
NONCONVERGENCE = "RuntimeError: Nonconverged fit is technical; it is not a recovery failure"
CLAIM_FIELDS = ("family_selection", "confusion_rate", "group_recovery", "interval_coverage",
                "full_gate", "full_menu_false_selection", "false_selection_rate", "early_kill")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def repair_context(original, failed, previous, source, people):
    """Bind both terminal attempts to the already validated actual source cohort."""
    diagnosis = wave.technical_mvc_unknown(failed, INDEX, original, ORIGINAL_REPORT)["technical_cause"]
    if diagnosis["classification"] != "first_half_500_sweep_exhaustion":
        raise ValueError("The original MZ42 attempt must retain its diagnosed first-half500 exhaustion")
    if failed.get("job_id") != ORIGINAL_JOB_ID or failed.get("finished_utc") != ORIGINAL_FINISHED_UTC:
        raise ValueError("Preserve the exact terminal original MVC worker46572920 from array46572753")
    first_runtime, prior_repair = pilot.repair_context(original, failed)
    for report in (failed, previous):
        wave.saved.completed_support(report)
        if (report.get("source_finished_utc") != source["finished_utc"]
                or source["finished_utc"] != SOURCE_FINISHED_UTC
                or any(fit.get("participants") != sorted(people) for fit in report.get("fits", []))):
            raise ValueError("Both attempts must bind to the exact source completion and actual participants")
        if (any(report.get(field) != "not_evaluated" for field in CLAIM_FIELDS[:5])
                or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
                or any(report.get(field) is not False for field in
                       ("simulation_regenerated", "saved_M0_MZ_refitted", "observed_choices_or_RTs_used"))
                or report.get("diagnostics") != {}):
            raise ValueError("Both prior technical attempts must retain the original diagnostic-only scope")
    if (previous.get("analysis_id") != "behavioral_mechanism_kill_test"
            or previous.get("phase") != pilot.PHASE or previous.get("job_id") != PILOT_JOB_ID
            or previous.get("generating_model") != "MZ" or previous.get("fitted_model") != MODEL
            or type(previous.get("replicate_index")) is not int or previous["replicate_index"] != INDEX
            or type(previous.get("seed")) is not int or previous["seed"] != saved.cell_seed("MZ", INDEX)
            or Path(previous.get("source_dir", "")).resolve() != saved.source_directory(INDEX).resolve()
            or previous.get("finished_utc") != PILOT_FINISHED_UTC
            or previous.get("implementation") != original or previous.get("runtime_implementation") != first_runtime
            or previous.get("numerical_repair") != prior_repair
            or previous.get("status") != "technical_rival_failure"
            or previous.get("rival_diagnostics_complete") is not False or previous.get("error") != NONCONVERGENCE
            or previous.get("internal_seconds") != 5400 or previous.get("selected_replicates") != [INDEX]
            or previous.get("retry_wave_selected") is not False
            or any(previous.get(field) != "not_evaluated" for field in CLAIM_FIELDS[5:7])):
        raise ValueError("Only the exact terminal index42 runtime1000 pilot46577369 is selected")
    discrepancy = previous.get("saved_score_replay_max_absolute_discrepancy")
    if (previous.get("saved_scores_replayed") is not True
            or previous.get("saved_score_replay_absolute_tolerance") != saved.shared.REPLAY_ABSOLUTE_TOLERANCE
            or not finite(discrepancy) or not 0 <= discrepancy <= saved.shared.REPLAY_ABSOLUTE_TOLERANCE
            or previous.get("scored_primary_directions") != [{"training_runs": [1, 4], "scoring_runs": [2, 3]}]
            or previous.get("completed_rivals") != [] or "participant_scores" in previous):
        raise ValueError("The pilot must preserve baseline replay and exactly one completed MVC direction")
    fits = previous.get("fits", [])
    if len(fits) != 2:
        raise ValueError("The terminal pilot must preserve both ordered returned fits")
    first, reverse = fits
    validate_fit(first, MODEL, *saved.HALVES[0], first_runtime, people)
    if len(first["objective_history"]) != 666:
        raise ValueError("Preserve the selected pilot's converged666-sweep first fit")
    history = reverse.get("objective_history")
    if (reverse.get("model") != MODEL or reverse.get("rival") != "fixed"
            or reverse.get("training_runs") != [2, 3] or reverse.get("scoring_runs") != [1, 4]
            or reverse.get("converged") is not False or reverse.get("subject_optimizer_failures") != []
            or not isinstance(history, list) or len(history) != FIRST_SWEEPS
            or not all(finite(value) for value in history)):
        raise ValueError("Only reverse-half1000 exhaustion without final subject failures is repairable")
    change = abs(history[-1] - history[-2])
    threshold = first_runtime["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if not finite(change) or not finite(threshold) or change <= threshold:
        raise ValueError("The reverse fit must exceed the unchanged joint convergence tolerance")
    for value in (previous.get("elapsed_seconds"), previous.get("cumulative_process_elapsed_seconds"),
                  *(fit.get("elapsed_seconds") for fit in fits)):
        if not finite(value) or value < 0:
            raise ValueError("Preserve finite nonnegative prior process and returned-fit costs")
    if previous["cumulative_process_elapsed_seconds"] != failed["elapsed_seconds"] + previous["elapsed_seconds"]:
        raise ValueError("The pilot must include the original attempt once in its cumulative cost")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = REVERSE_SWEEPS
    metadata = {
        "kind": "reverse_half_maximum_sweeps_only", "original_report": str(ORIGINAL_REPORT),
        "pilot_report": str(PILOT_REPORT), "pilot_job_id": PILOT_JOB_ID,
        "original_finished_utc": failed["finished_utc"], "pilot_finished_utc": previous["finished_utc"],
        "source_finished_utc": source["finished_utc"], "original_error": failed["error"],
        "pilot_error": previous["error"], "original_diagnosis": deepcopy(diagnosis),
        "pilot_diagnosis": {"classification": "second_half_1000_sweep_exhaustion", "completed_sweeps": 1000,
                            "final_objective_change": change, "unchanged_convergence_threshold": threshold},
        "original_maximum_sweeps": ORIGINAL_SWEEPS, "first_half_maximum_sweeps": FIRST_SWEEPS,
        "reverse_half_maximum_sweeps": REVERSE_SWEEPS,
        "original_process_elapsed_seconds": failed["elapsed_seconds"],
        "pilot_process_elapsed_seconds": previous["elapsed_seconds"],
        "previous_cumulative_process_elapsed_seconds": previous["cumulative_process_elapsed_seconds"],
        "pilot_fit_elapsed_seconds": [fit["elapsed_seconds"] for fit in fits],
        "first_half_original_fit_elapsed_seconds": first["elapsed_seconds"],
        "first_half_materialization_elapsed_seconds": None, "first_half_rescore_elapsed_seconds": None,
        "first_half_rescored": False,
        "first_half_score_handling": "recomputed_from_preserved_parameters_no_saved_MVC_score_agreement_claim",
        "new_first_half_fit_elapsed_seconds_meaning": "copy_return_overhead_not_original_fit_cost",
        "reverse_initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "nonconverged_checkpoint_state_reused": False,
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "cost_accounting": "pilot_cumulative_plus_new_process_once; first_fit_cost_already_in_pilot_process",
    }
    return first_runtime, runtime, deepcopy(first), metadata


def run(args):
    require_compute()
    if args.implementation.resolve() != saved.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 5400:
        raise ValueError("The whole-process cap must be 1 through5400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or any(output == path.resolve() or path.resolve() in output.parents for path in
                             (saved.source_directory(INDEX), ORIGINAL_REPORT.parent, PILOT_REPORT.parent)):
        raise FileExistsError("Use a fresh output outside the saved cohort and both preserved attempts")
    original_source_validator, original_fit_validator = saved.validate_source, saved.validate_fit
    original_fitter, original_writer, original_scorer = saved.fit_partially_pooled, saved.write_report, saved.shared.score_fit
    original = first_runtime = runtime = preserved_first = metadata = None
    fit_calls = 0

    def source_validator(source, truth, people, spec, index):
        nonlocal original, first_runtime, runtime, preserved_first, metadata
        if (index != INDEX or spec["optimizer"]["maximum_sweeps"] != ORIGINAL_SWEEPS
                or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200):
            raise ValueError("Preserve index42, the original500 implementation and fixed200 plan")
        original_source_validator(source, truth, people, spec, index)
        original = deepcopy(spec)
        failed = json.loads(ORIGINAL_REPORT.read_text())
        previous = json.loads(PILOT_REPORT.read_text())
        first_runtime, runtime, preserved_first, metadata = repair_context(original, failed, previous, source, people)

    def fitter(training, model, rival, spec):
        nonlocal fit_calls
        if model != MODEL or rival != "fixed" or metadata is None or spec != original or fit_calls >= 2:
            raise ValueError("Only the validated first fit and one fresh reverse fit are selected")
        fit_calls += 1
        return deepcopy(preserved_first) if fit_calls == 1 else original_fitter(training, model, rival, runtime)

    def fit_validator(fit, model, train, held, spec, people):
        if model in ("M0", "MZ") and spec["optimizer"]["maximum_sweeps"] == ORIGINAL_SWEEPS:
            return original_fit_validator(fit, model, train, held, spec, people)
        if model != MODEL or metadata is None or spec != original:
            raise ValueError("Only the original baseline and mixed1000/1500 MVC panel may be validated")
        direction = (tuple(train), tuple(held))
        if direction not in saved.HALVES:
            raise ValueError("Preserve the two original primary directions")
        selected = first_runtime if direction == saved.HALVES[0] else runtime
        return original_fit_validator(fit, model, train, held, selected, people)

    def score_fit(fit, scoring, spec, check=lambda: None):
        if fit.get("model") != MODEL or fit.get("training_runs") != [1, 4]:
            return original_scorer(fit, scoring, spec, check)
        started = time.perf_counter()
        try:
            result = original_scorer(fit, scoring, spec, check)
            metadata["first_half_rescored"] = True
            return result
        finally:
            metadata["first_half_rescore_elapsed_seconds"] = time.perf_counter() - started

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, selected_replicates=[INDEX], selected_models=[MODEL],
                      repair_context_validated=metadata is not None, first_half_fit_reused=fit_calls >= 1,
                      fresh_reverse_fit_called=fit_calls >= 2, wave_selected=False, retry_wave_selected=False,
                      automatic_retry=False, full_menu_wave_selected=False,
                      baseline_validation_maximum_sweeps=ORIGINAL_SWEEPS,
                      per_half_provenance=[
                          {"training_runs": [1, 4], "scoring_runs": [2, 3], "fit_source": str(PILOT_REPORT),
                           "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": FIRST_SWEEPS},
                          {"training_runs": [2, 3], "scoring_runs": [1, 4], "fit_source": "current_completion",
                           "action": "fresh_fit", "validation_maximum_sweeps": REVERSE_SWEEPS}],
                      **{field: "not_evaluated" for field in CLAIM_FIELDS})
        if original is not None:
            report["implementation"] = original
        if metadata is not None:
            if report["fits"]:
                metadata["first_half_materialization_elapsed_seconds"] = report["fits"][0]["elapsed_seconds"]
            report.update(first_half_runtime_implementation=first_runtime, runtime_implementation=runtime,
                          numerical_repair=deepcopy(metadata))
            if "elapsed_seconds" in report:
                report["cumulative_process_elapsed_seconds"] = metadata["previous_cumulative_process_elapsed_seconds"] + report["elapsed_seconds"]
        original_writer(directory, report)

    # Both failed-report reads and all source validation, replay and new work
    # occur inside saved.run's existing deadline and technical-failure handler.
    with patch.object(saved, "validate_source", source_validator), \
            patch.object(saved, "fit_partially_pooled", fitter), \
            patch.object(saved, "validate_fit", fit_validator), \
            patch.object(saved.shared, "score_fit", score_fit), \
            patch.object(saved, "write_report", write_checkpoint), \
            patch.object(saved, "REPORT_NAME", REPORT_NAME):
        return saved.run(argparse.Namespace(model=MODEL, replicate_index=INDEX, source_dir=saved.source_directory(INDEX),
            implementation=args.implementation, output_dir=output, max_seconds=args.max_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "rival_diagnostics_complete",
        "first_half_fit_reused", "fresh_reverse_fit_called", "elapsed_seconds", "cumulative_process_elapsed_seconds",
        "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
