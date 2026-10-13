#!/usr/bin/env python3
"""Fixed200 necessary-screen counts for explicit terminal twice-SESOI cohorts.

This is the uniform z=.40 M0/MZ baseline cell, not a full-menu family result.
Missing and technical attempts remain unknown; no phase or truth is relabeled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
import run_mz_twice_wave as wave
import summarize_mz_fixed_menu as fixed


levels, core = wave.levels, wave.levels.core
IMPLEMENTATION = core.IMPLEMENTATION
RUNS = core.DESIGN_SOURCE.parent.parent
PLANNED, REQUIRED = 200, 180
REPRESENTATIVE_JOB = 46603942
FIRST_ARRAY, REMAINING_ARRAY = 46606307, 46606413
PHASE = "MZ_twice_SESOI_fixed200_baseline_summary"
EXTRA_DIAGNOSTICS = ("positive_in_both_task_versions", "positive_participant_median", "participant_median_95CI_lower_positive")
FIT_ORDER = tuple((model, train, held) for train, held in core.HALVES for model in ("M0", "MZ"))


def terminal_indices(value):
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token.isdecimal() for token in tokens):
        raise argparse.ArgumentTypeError("Supply explicit comma-separated terminal indices0 through199")
    indices = [int(token) for token in tokens]
    if len(set(indices)) != len(indices) or any(not 0 <= index < PLANNED for index in indices):
        raise argparse.ArgumentTypeError("Terminal indices must be unique within the fixed200 cell")
    return sorted(indices)


def array_id(value):
    if type(value) is bool or not str(value).isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("Supply an explicit positive array job ID")
    return int(value)


def report_path(index, first_array, remaining_array):
    levels.level_seed("twice", index)
    if index == 0:
        directory = RUNS / f"20261005-MZ-effect-level-representatives-{REPRESENTATIVE_JOB}" / "twice" / "replicate0"
    else:
        selected = first_array if index == 1 else remaining_array
        if type(selected) is not int or selected < 1:
            raise ValueError("Declare the selected source array before reading this index")
        directory = RUNS / f"20261005-MZ-twice-SESOI-wave-{selected}" / f"replicate{index}"
    return directory / levels.REPORT_NAME


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_identity(report, index, spec):
    if (not isinstance(report, dict) or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != "MZ_twice_SESOI_baseline_component"
            or report.get("effect_level") != "twice" or report.get("generating_model") != "MZ"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != levels.level_seed("twice", index)
            or type(report.get("SESOI_level")) not in (int, float) or report["SESOI_level"] != 2.
            or not finite(report.get("generating_z")) or report["generating_z"] != .40
            or report.get("focal_generating_truth") != levels.focal_truth("twice")
            or report.get("implementation") != spec or report.get("fitted_models") != ["M0", "MZ"]
            or report.get("selected_replicates") != [index]
            or any(type(report.get(field)) is not int or report[field] != PLANNED
                   for field in ("planned_cell_replicates", "fixed_cohort_count"))):
        raise ValueError("Preserve the exact twice-SESOI phase, z=.40 truth, seed, index and original implementation")
    truth = report["focal_generating_truth"]
    if any(not finite(value) for value in truth.values()):
        raise ValueError("Focal generating truth must be finite numeric values")
    job = str(report.get("job_id", ""))
    if not job.isdecimal() or int(job) < 1 or (index == 0 and job != str(REPRESENTATIVE_JOB)):
        raise ValueError("Preserve representative0 or the explicit indexed worker identity")
    if (report.get("wave_selected") is not (index != 0)
            or (index != 0 and (report.get("dispatch_scope") != wave.DISPATCH_SCOPE
                or report.get("externally_selected_index_range") != [1, 199]
                or report.get("subsequent_wave_selected") is not False))):
        raise ValueError("Representative0 and externally selected indexed-wave scope must remain distinct")
    if (any(report.get(field) is not False for field in ("full_menu_wave_selected", "retry_selected", "automatic_retry",
            "observed_choices_or_RTs_used", "old_synthetic_responses_used", "episode_terminal"))
            or any(report.get(field) != "not_evaluated" for field in levels.CLAIM_FIELDS)
            or report.get("full_behavioral_gate") != "not_evaluated" or report.get("full_recovery_pass") is not None
            or any(report.get(field) != value for field, value in levels.criterion_scope("twice").items())
            or report.get("recovery_90pct_criterion_applies") is not True
            or report.get("focal_effect_between_participant_variance") != 0.
            or report.get("individual_rank_RMSE_qualification") != "not_evaluated_constant_focal_effect"
            or report.get("nuisance_generation") != "recorded_center_and_individual_scale_truncated_normal_as_profile01"
            or report.get("deadline_missingness") != "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel"
            or report.get("design_fields") != list(core.DESIGN_FIELDS)
            or not isinstance(report.get("design_source"), str)
            or fixed.episode_path(report["design_source"]) != core.DESIGN_SOURCE.resolve()
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= levels.MAX_SECONDS):
        raise ValueError("Preserve the fixed design, necessary-only criterion and closed scientific claims")
    try:
        started, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if started.utcoffset() is None or finished.utcoffset() is None or finished < started:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Selected reports must retain terminal timestamps") from error
    if not finite(report.get("elapsed_seconds")) or report["elapsed_seconds"] < 0:
        raise ValueError("Retain finite nonnegative process cost")


def completed_diagnostic(report, spec):
    if (report.get("status") != "necessary_primary_comparison_completed"
            or report.get("valid_recovery_component") is not True or report.get("baseline_diagnostic_complete") is not True
            or [report.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]
            or type(report.get("simulated_deadline_misses")) is not int or not 0 <= report["simulated_deadline_misses"] <= 27648
            or not finite(report.get("simulation_seconds")) or report["simulation_seconds"] < 0):
        raise ValueError("Only a complete108-person four-fit component supplies a necessary-screen result")
    scores = fixed.score_panel(report, ("M0", "MZ"))
    fixed.fit_panel(report, FIT_ORDER, spec, scores)
    diagnostic = fixed.nested_contrast(scores, "MZ", "M0")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("Retain the original held-out comparison")
    for field in ("mean_gain_nats_per_trial", "participant_SE", "version_means", "participant_median"):
        if comparison.get(field) != diagnostic[field]:
            raise ValueError("Recorded core diagnostic differs from the complete paired score panel")
    interval = comparison.get("participant_median_95CI")
    if (not isinstance(interval, list) or len(interval) != 2 or not all(finite(value) for value in interval)
            or interval[0] > interval[1] or comparison.get("bootstrap") != "9999_fixed_seed_participant_draws_within_EI_ER"):
        raise ValueError("Retain finite ordered recorded median-bootstrap diagnostics")
    flags = {**diagnostic["requirements"],
        "positive_in_both_task_versions": all(value > 0 for value in diagnostic["version_means"].values()),
        "positive_participant_median": diagnostic["participant_median"] > 0,
        "participant_median_95CI_lower_positive": interval[0] > 0}
    passed, richer = diagnostic["qualifies_nested_comparison"], all(flags.values())
    if (comparison.get("requirements") != flags or any(type(value) is not bool for value in comparison["requirements"].values())
            or comparison.get("focal_model") != "MZ" or comparison.get("necessary_nested_MZ_vs_M0_pass") is not passed
            or comparison.get("counts_as_curtailment_failure") is not (not passed)
            or report.get("counts_as_curtailment_failure") is not (not passed)
            or comparison.get("necessary_primary_comparison_pass") is not richer
            or comparison.get("richer_primary_diagnostic_pass") is not richer
            or comparison.get("full_recovery_pass") is not None
            or any(comparison.get(field) != "not_evaluated" for field in ("family_selection", "family_recovery_rate"))
            or any(comparison.get(field) != value for field, value in levels.criterion_scope("twice").items())
            or comparison.get("recovery_90pct_criterion_applies") is not True
            or comparison.get("curtailment_rule") != "only_valid_at_or_above_SESOI_mean_0p01_and_strict_2SE_failures; no_automatic_curtailment"
            or comparison.get("diagnostics_not_recovery_rate_gates") != list(EXTRA_DIAGNOSTICS)
            or "core_family_selection_pass" in comparison or "core_family_selection_pass" in report):
        raise ValueError("Core counts and separate richer diagnostics must preserve the original necessary-only rules")
    return diagnostic, {"recorded_richer_diagnostic_pass": richer, "recorded_participant_median_95CI": interval,
                        "recorded_bootstrap": comparison["bootstrap"], "requirements": {key: flags[key] for key in EXTRA_DIAGNOSTICS}}


def technical_metadata(report):
    if (report.get("status") != "technical_recovery_failure" or report.get("valid_recovery_component") is not False
            or report.get("baseline_diagnostic_complete") is not False or report.get("counts_as_curtailment_failure") is not False
            or not isinstance(report.get("error"), str) or not report["error"]
            or "participant_scores" in report or "comparison" in report
            or not isinstance(report.get("fits"), list) or len(report["fits"]) > 4):
        raise ValueError("Technical attempts must retain their error and incomplete prefix without scientific scores")
    fits = []
    for fit, (model, train, held) in zip(report["fits"], FIT_ORDER):
        if (not isinstance(fit, dict) or fit.get("model") != model or fit.get("rival") != "fixed"
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
                or type(fit.get("converged")) is not bool or not finite(fit.get("elapsed_seconds")) or fit["elapsed_seconds"] < 0):
            raise ValueError("Technical returned fits must retain the selected ordered model/direction prefix")
        fits.append({key: fit.get(key) for key in ("model", "training_runs", "scoring_runs", "converged", "elapsed_seconds", "subject_optimizer_failures")})
    return {"error": report["error"], "returned_fits": fits}


def pool(indices, first_array, remaining_array, spec):
    if (len(set(indices)) != len(indices) or any(type(index) is not int or not 0 <= index < PLANNED for index in indices)
            or type(first_array) is not int or first_array != FIRST_ARRAY
            or type(remaining_array) is not int or remaining_array != REMAINING_ARRAY):
        raise ValueError("Supply unique selected terminal indices and exactly the selected first/remaining arrays")
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["optimizer"]["maximum_sweeps"] != 500
            or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != PLANNED
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("Only the unchanged original500 fixed200 primary cell is selected")
    cohorts, completed, successes, failures, richer_successes = [], [], 0, 0, 0
    for index in range(PLANNED):
        row = {"replicate_index": index, "seed": levels.level_seed("twice", index), "status": "unknown"}
        cohorts.append(row)
        if index not in indices:
            row["reason"] = "not_selected_as_terminal_by_coordinator"
            continue
        path = report_path(index, first_array, remaining_array)
        row["report"] = str(path)
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            row["reason"] = "selected_terminal_report_missing"
            continue
        validate_identity(report, index, spec)
        row.update(job_id=str(report["job_id"]), elapsed_seconds=report["elapsed_seconds"], finished_utc=report["finished_utc"])
        if report.get("status") == "technical_recovery_failure":
            row.update(reason="technical_component_failure", **technical_metadata(report))
            continue
        diagnostic, richer = completed_diagnostic(report, spec)
        passed = diagnostic["qualifies_nested_comparison"]
        row.update(status="necessary_nested_diagnostic_completed", nested_comparison=diagnostic, richer_diagnostics=richer,
                   necessary_nested_MZ_vs_M0_pass=passed, counts_as_curtailment_failure=not passed)
        completed.append(index)
        successes += int(passed)
        failures += int(not passed)
        richer_successes += int(richer["recorded_richer_diagnostic_pass"])
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ", "effect_level": "twice",
        "SESOI_level": 2., "generating_z": .40, "focal_generating_truth": levels.focal_truth("twice"), "implementation": spec,
        "fixed_cohort_count": PLANNED, "required_necessary_screen_successes": REQUIRED, **levels.criterion_scope("twice"),
        "source_arrays": {"index1": first_array, "indices2_through199": remaining_array},
        "representative0_report": str(report_path(0, first_array, remaining_array)), "selected_terminal_indices": sorted(indices),
        "completed_indices": completed, "unknown_indices": [row["replicate_index"] for row in cohorts if row["status"] == "unknown"],
        "valid_components": len(completed), "necessary_screen_successes": successes, "valid_core_failures": failures,
        "unknown_cohorts": PLANNED - len(completed), "best_possible_necessary_screen_successes": PLANNED - failures,
        "necessary_90pct_component_count_met": successes >= REQUIRED,
        "necessary_90pct_component_unreachable": PLANNED - failures < REQUIRED,
        "recorded_richer_diagnostic_successes": richer_successes, "recorded_richer_diagnostic_failures": len(completed) - richer_successes,
        "cohorts": cohorts, **{field: "not_evaluated" for field in levels.CLAIM_FIELDS},
        "full_menu_selection": "not_evaluated", "full_behavioral_gate": "not_evaluated", "full_recovery_pass": None,
        "episode_terminal": False, "full_menu_wave_selected": False, "retry_selected": False, "automatic_retry": False,
        "interpretation": "Fixed200 twice-SESOI z=.40 baseline counts only. Unknown cohorts are not failures. At21 validated core failures, even every other index passing cannot reach180;180 observed necessary-screen successes establishes only this component count. Richer bootstrap diagnostics do not enter either bound. Full-menu family recovery, scientific rates, intervals, heterogeneity and the behavioral gate remain unevaluated. No further work is dispatched.",
        "process_state_boundary": "The coordinator explicitly selects terminal indices; report timestamps do not establish scheduler state."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-indices", type=terminal_indices, required=True)
    parser.add_argument("--first-array-job-id", type=array_id, required=True)
    parser.add_argument("--remaining-array-job-id", type=array_id, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if args.output.exists():
        raise FileExistsError("Preserve earlier summaries; select a fresh output")
    result = pool(args.terminal_indices, args.first_array_job_id, args.remaining_array_job_id, json.loads(IMPLEMENTATION.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("phase", "valid_components", "necessary_screen_successes", "valid_core_failures",
        "unknown_cohorts", "necessary_90pct_component_count_met", "necessary_90pct_component_unreachable")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
