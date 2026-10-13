#!/usr/bin/env python3
"""Pool existing nested-menu diagnostics for explicitly selected terminal indices.

The coordinator must establish terminal process state before invocation. Read
only selected reports, retain the fixed 200-cohort denominator, and leave MC1
and every unselected index unknown. The terminal-index mode also retains
identified technical MVC array failures as unknown menus, without comparisons.
No scheduler queries, polling, family selection, confusion rates, intervals,
or full-gate verdicts are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION, cell_seed, validate_fit
from run_mz_saved_replicate_rival import source_directory
import summarize_mz_fixed_menu as fixed
import summarize_mz_saved_replicate_menu as saved


RUNS = Path(__file__).resolve().parents[2] / "outputs/runs/behavioral_mechanism_kill_test"
FIXED_COHORTS = 200
RIVAL_ARRAYS = {"MC": 46550138, "MV": 46550142, "MVZ": 46550145, "MVC": 46572753}


def selected_indices(value):
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token.isdecimal() for token in tokens):
        raise argparse.ArgumentTypeError("Supply explicit comma-separated indices, for example 0,2,3")
    indices = [int(token) for token in tokens]
    if len(indices) != len(set(indices)) or any(not 0 <= index < FIXED_COHORTS for index in indices):
        raise argparse.ArgumentTypeError("Select unique fixed indices 0 through 199")
    if 1 in indices:
        raise argparse.ArgumentTypeError("Index 1 remains unknown: MC1 was interrupted and its retry is held")
    return sorted(indices)


def input_paths(index):
    if index == 0:
        return (
            RUNS / "20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json",
            RUNS / "20261005-MZ0-fixed-rivals-attempt001/single_process_rivals.json",
            RUNS / "20261005-MZ0-MVZ-rival-46545300/mz_complex_rival.json",
            RUNS / "20261005-MZ0-MVC-rival-46545303/mz_complex_rival.json",
        )
    # The unchanged source helper fixes 1–20 to 46538652 and 21–199 to 46539590.
    return (source_directory(index) / "single_process_recovery.json", *(
        RUNS / f"20261005-MZ-{model}-saved-array-{RIVAL_ARRAYS[model]}" /
        f"replicate{index}/mz_saved_replicate_rival.json" for model in saved.RIVALS
    ))


def technical_cause(report, implementation):
    """Describe evidenced 500-sweep exhaustion, without qualifying a retry."""
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def cost(value):
        return value if finite(value) and value >= 0 else None

    fits = report.get("fits")
    diagnosis = {
        "classification": "unresolved",
        "process_elapsed_seconds": cost(report.get("elapsed_seconds")),
        "returned_fit_elapsed_seconds": [cost(fit.get("elapsed_seconds")) if isinstance(fit, dict) else None
                                         for fit in fits] if isinstance(fits, list) else None,
    }

    def unresolved(reason):
        diagnosis["reason"] = reason
        return diagnosis

    optimizer = implementation.get("optimizer", {})
    tolerance = optimizer.get("joint_relative_tolerance")
    if optimizer.get("maximum_sweeps") != 500 or not finite(tolerance) or tolerance <= 0:
        return unresolved("original_500_sweep_limit_or_convergence_tolerance_unverified")
    if report.get("error") != "RuntimeError: Nonconverged fit is technical; it is not a recovery failure":
        return unresolved("recorded_failure_is_not_the_original_nonconvergence_validator_error")
    if (not isinstance(fits, list) or len(fits) not in (1, 2)
            or any(not isinstance(fit, dict) for fit in fits)):
        return unresolved("one_or_two_ordered_returned_fits_required")
    if (diagnosis["process_elapsed_seconds"] is None
            or any(value is None for value in diagnosis["returned_fit_elapsed_seconds"])):
        return unresolved("finite_nonnegative_prior_process_and_fit_costs_required")
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (report.get("saved_scores_replayed") is not True
            or report.get("saved_score_replay_absolute_tolerance") != saved.REPLAY_ABSOLUTE_TOLERANCE
            or not finite(discrepancy) or not 0 <= discrepancy <= saved.REPLAY_ABSOLUTE_TOLERANCE):
        return unresolved("successful_original_baseline_replay_unverified")
    expected_scored = [{"training_runs": list(train), "scoring_runs": list(held)}
                       for train, held in HALVES[:len(fits) - 1]]
    if (report.get("scored_primary_directions") != expected_scored
            or report.get("completed_rivals") != [] or "participant_scores" in report
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
        return unresolved("scored_directions_or_incomplete_full_cohort_state_do_not_match")
    for fit, (train, held) in zip(fits, HALVES):
        if (fit.get("model") != "MVC" or fit.get("rival") != "fixed"
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)):
            return unresolved("returned_fit_model_or_half_order_does_not_match")
    exhausted = fits[-1]
    history = exhausted.get("objective_history")
    if (exhausted.get("converged") is not False or exhausted.get("subject_optimizer_failures") != []
            or not isinstance(history, list) or len(history) != 500 or not all(finite(value) for value in history)):
        return unresolved("finite_500_sweep_history_without_final_subject_failures_unverified")
    change = abs(history[-1] - history[-2])
    threshold = tolerance * (1 + abs(history[-2]))
    diagnosis.update(final_objective_change=change if finite(change) else None,
                     unchanged_convergence_threshold=threshold if finite(threshold) else None)
    if not finite(change) or not finite(threshold) or change <= threshold:
        return unresolved("final_objective_change_does_not_exceed_unchanged_stop_threshold")
    if len(fits) == 2:
        people = exhausted.get("participants")
        if (not isinstance(people, list) or len(people) != 108
                or any(not isinstance(person, str) or not person for person in people)
                or len(set(people)) != 108):
            return unresolved("shared_participant_identity_for_prior_half_unverified")
        try:
            validate_fit(fits[0], "MVC", HALVES[0][0], HALVES[0][1], implementation, people)
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            diagnosis["prior_half_validation_error"] = f"{type(error).__name__}: {error}"
            return unresolved("prior_half_does_not_pass_original_fit_validation")
        diagnosis["prior_half_validation"] = "passed_under_original_implementation"
    else:
        diagnosis["prior_half_validation"] = "not_applicable"
    diagnosis.update(classification="first_half_500_sweep_exhaustion" if len(fits) == 1
                     else "second_half_500_sweep_exhaustion", completed_sweeps=500,
                     training_runs=list(HALVES[len(fits) - 1][0]), scoring_runs=list(HALVES[len(fits) - 1][1]))
    return diagnosis


def technical_mvc_unknown(report, index, implementation, path):
    """Recognize only the selected saved-cohort MVC array's technical failure."""
    if (not isinstance(report, dict)
            or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != "MZ_SESOI_saved_replicate_rival_diagnostics"
            or report.get("status") != "technical_rival_failure"
            or report.get("generating_model") != "MZ" or report.get("fitted_model") != "MVC"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != cell_seed("MZ", index)
            or report.get("implementation") != implementation
            or report.get("source_dir") is None
            or fixed.episode_path(report["source_dir"]) != source_directory(index).resolve()
            or path != input_paths(index)[-1]
            or report.get("rival_diagnostics_complete") is not False
            or not isinstance(report.get("error"), str) or not report["error"]):
        raise ValueError("Technical MVC report must retain its selected cohort and implementation identity")
    return {"replicate_index": index, "status": "unknown", "reason": "MVC_technical_rival_failure",
            "technical_report": str(path), "technical_status": report["status"], "error": report["error"],
            "technical_cause": technical_cause(report, implementation)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--completed-indices", type=selected_indices,
                           help="Strict selection: every selected terminal index must have a valid complete menu")
    selection.add_argument("--terminal-indices", type=selected_indices,
                           help="Selected terminal indices; identified technical MVC array failures remain unknown")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output")
    implementation = json.loads(IMPLEMENTATION.read_text())
    if implementation["recovery_plan"]["replicates_per_qualified_generating_cell"] != FIXED_COHORTS:
        raise ValueError("The fixed 200-cohort recovery denominator changed")

    summaries, cohorts, completed = [], [], []
    terminal_mode = args.terminal_indices is not None
    selected = set(args.terminal_indices if terminal_mode else args.completed_indices)
    for index in range(FIXED_COHORTS):
        if index not in selected:
            cohorts.append({"replicate_index": index, "status": "unknown",
                            "reason": "MC1_interrupted_retry_held" if index == 1
                            else "not_selected_as_terminal_by_coordinator"})
            continue
        paths = input_paths(index)
        if terminal_mode and index != 0:
            # Inspect the exact selected MVC attempt first. A recognized
            # technical failure leaves the whole menu unknown, so other
            # reports cannot contribute a partial-model comparison.
            mvc = json.loads(paths[-1].read_text())
            if isinstance(mvc, dict) and mvc.get("status") == "technical_rival_failure":
                cohorts.append(technical_mvc_unknown(mvc, index, implementation, paths[-1]))
                continue
            reports = [json.loads(path.read_text()) for path in paths[:-1]] + [mvc]
        else:
            reports = [json.loads(path.read_text()) for path in paths]
        # Both existing builders perform their established cohort, fit, score,
        # and implementation validation before computing the frozen diagnostics.
        if index == 0:
            summary = fixed.build_summary(*reports, implementation)
        else:
            summary = saved.build_summary(*reports, implementation, index)
        summaries.append(summary)
        completed.append(index)
        cohorts.append({"replicate_index": index, "status": "nested_diagnostics_completed",
                        "input_reports": [str(path) for path in paths], "summary": summary})

    maximal_sets = Counter(tuple(summary["maximal_eligible_models_diagnostic"]) for summary in summaries)
    diagnostic_counts = {
        "completed_menus": len(summaries), "unknown_menus": FIXED_COHORTS - len(summaries),
        "nested_comparison_qualifications": {
            f"{model}-{baseline}": sum(summary["nested_comparisons"][f"{model}-{baseline}"]
                                       ["qualifies_nested_comparison"] for summary in summaries)
            for model, baseline in fixed.EDGES},
        "nested_comparison_eligibility": {
            model: sum(summary["nested_comparison_eligibility"][model] for summary in summaries)
            for model in fixed.MODELS},
        "maximal_eligible_model_sets_diagnostic": [
            {"models": list(models), "menus": count} for models, count in sorted(maximal_sets.items())],
        "menus_with_multiple_incomparable_eligible_models": sum(
            summary["multiple_incomparable_eligible_models"] for summary in summaries),
        "interpretation": "Raw counts among selected complete menus only. Unknown cohorts are neither failures nor negative comparisons; maximal eligible sets are not selected families.",
    }
    result = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_fixed_200_menu_wave_nested_diagnostics",
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "generating_model": "MZ",
        "fixed_cohort_count": FIXED_COHORTS, "completed_indices": completed,
        "selected_terminal_indices": sorted(selected),
        "selection_mode": "terminal_with_technical_unknowns" if terminal_mode else "strict_completed",
        "unknown_indices": [index for index in range(FIXED_COHORTS) if index not in completed],
        "implementation_path": str(IMPLEMENTATION), "implementation": implementation,
        "model_menu": list(fixed.MODELS), "cohorts": cohorts, "diagnostic_counts": diagnostic_counts,
        "family_selection": "not_evaluated", "confusion_rate": "not_evaluated",
        "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated", "full_gate": "not_evaluated",
        "full_recovery_pass": None, "episode_terminal": False,
        "process_state_boundary": "The coordinator independently established terminal input attempts for the explicit selected indices; report completion fields are not scheduler evidence.",
        "interpretation": "Existing per-cohort nested diagnostics with a fixed denominator of 200 and explicit unknowns. No family is selected, including when MZ is uniquely maximal; no recovery/confusion rate or interval/gate verdict is computed.",
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({"phase": result["phase"], "fixed_cohort_count": FIXED_COHORTS,
                      "diagnostic_counts": diagnostic_counts, "family_selection": "not_evaluated"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
