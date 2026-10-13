#!/usr/bin/env python3
"""Continuous diagnostics for explicit terminal half-SESOI M0/MZ cohorts.

The fixed200 cell is below SESOI: no recovery pass rate or curtailment applies.
Only selected terminal reports and the unchanged implementation are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from qualify_assets import require_compute
import run_mz_half_wave as wave
import summarize_mz_twice_wave as shared


levels, core = wave.levels, wave.levels.core
fixed = shared.fixed
IMPLEMENTATION = core.IMPLEMENTATION
RUNS = core.DESIGN_SOURCE.parent.parent
PLANNED = 200
REPRESENTATIVE_JOB = 46603942
FIRST_ARRAY, REMAINING_ARRAY = 46607094, 46607282
PHASE = "MZ_half_SESOI_fixed200_baseline_summary"
CONTINUOUS_FIELDS = ("mean_gain_nats_per_trial", "participant_SE", "version_means", "participant_median")
terminal_indices, array_id = shared.terminal_indices, shared.array_id


def report_path(index, first_array, remaining_array):
    levels.level_seed("half", index)
    if index == 0:
        directory = RUNS / f"20261005-MZ-effect-level-representatives-{REPRESENTATIVE_JOB}" / "half" / "replicate0"
    else:
        selected = first_array if index == 1 else remaining_array
        if type(selected) is not int or selected < 1:
            raise ValueError("Declare the selected source array before reading this index")
        directory = RUNS / f"20261005-MZ-half-SESOI-wave-{selected}" / f"replicate{index}"
    return directory / levels.REPORT_NAME


def validate_identity(report, index, spec):
    if (not isinstance(report, dict) or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != "MZ_half_SESOI_baseline_component"
            or report.get("effect_level") != "half" or report.get("generating_model") != "MZ"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != levels.level_seed("half", index)
            or type(report.get("SESOI_level")) not in (int, float) or report["SESOI_level"] != .5
            or not shared.finite(report.get("generating_z")) or report["generating_z"] != .475
            or report.get("focal_generating_truth") != levels.focal_truth("half")
            or report.get("implementation") != spec or report.get("fitted_models") != ["M0", "MZ"]
            or report.get("selected_replicates") != [index]
            or any(type(report.get(field)) is not int or report[field] != PLANNED
                   for field in ("planned_cell_replicates", "fixed_cohort_count"))):
        raise ValueError("Preserve the exact half-SESOI phase, z=.475 truth, seed, index and original implementation")
    if any(not shared.finite(value) for value in report["focal_generating_truth"].values()):
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
            "observed_choices_or_RTs_used", "old_synthetic_responses_used", "episode_terminal",
            "recovery_90pct_criterion_applies", "counts_as_curtailment_failure"))
            or any(report.get(field) != "not_evaluated" for field in levels.CLAIM_FIELDS)
            or report.get("full_behavioral_gate") != "not_evaluated" or report.get("full_recovery_pass") is not None
            or any(report.get(field) != value for field, value in levels.criterion_scope("half").items())
            or report.get("focal_effect_between_participant_variance") != 0.
            or report.get("individual_rank_RMSE_qualification") != "not_evaluated_constant_focal_effect"
            or report.get("nuisance_generation") != "recorded_center_and_individual_scale_truncated_normal_as_profile01"
            or report.get("deadline_missingness") != "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel"
            or report.get("design_fields") != list(core.DESIGN_FIELDS)
            or not isinstance(report.get("design_source"), str)
            or fixed.episode_path(report["design_source"]) != core.DESIGN_SOURCE.resolve()
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= levels.MAX_SECONDS):
        raise ValueError("Preserve the fixed design, below-SESOI exclusion and closed scientific claims")
    try:
        started, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if started.utcoffset() is None or finished.utcoffset() is None or finished < started:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Selected reports must retain terminal timestamps") from error
    if not shared.finite(report.get("elapsed_seconds")) or report["elapsed_seconds"] < 0:
        raise ValueError("Retain finite nonnegative process cost")


def completed_diagnostic(report, spec):
    if (report.get("status") != "necessary_primary_comparison_completed"
            or report.get("valid_recovery_component") is not True or report.get("baseline_diagnostic_complete") is not True
            or [report.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]
            or type(report.get("simulated_deadline_misses")) is not int or not 0 <= report["simulated_deadline_misses"] <= 27648
            or not shared.finite(report.get("simulation_seconds")) or report["simulation_seconds"] < 0):
        raise ValueError("Only a complete108-person four-fit component supplies continuous diagnostics")
    scores = fixed.score_panel(report, ("M0", "MZ"))
    fixed.fit_panel(report, shared.FIT_ORDER, spec, scores)
    diagnostic = fixed.nested_contrast(scores, "MZ", "M0")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("Retain the original held-out comparison")
    if any(comparison.get(field) != diagnostic[field] for field in CONTINUOUS_FIELDS):
        raise ValueError("Recorded diagnostic differs from the complete paired score panel")
    interval = comparison.get("participant_median_95CI")
    if (not isinstance(interval, list) or len(interval) != 2 or not all(shared.finite(value) for value in interval)
            or interval[0] > interval[1] or comparison.get("bootstrap") != "9999_fixed_seed_participant_draws_within_EI_ER"):
        raise ValueError("Retain finite ordered recorded median-bootstrap diagnostics")
    flags = {**diagnostic["requirements"],
        "positive_in_both_task_versions": all(value > 0 for value in diagnostic["version_means"].values()),
        "positive_participant_median": diagnostic["participant_median"] > 0,
        "participant_median_95CI_lower_positive": interval[0] > 0}
    # Validate the producer's diagnostic booleans, without exporting pass/failure counts.
    if (comparison.get("requirements") != flags or any(type(value) is not bool for value in comparison["requirements"].values())
            or comparison.get("focal_model") != "MZ"
            or comparison.get("necessary_nested_MZ_vs_M0_pass") is not diagnostic["qualifies_nested_comparison"]
            or comparison.get("counts_as_curtailment_failure") is not False
            or comparison.get("necessary_primary_comparison_pass") is not all(flags.values())
            or comparison.get("richer_primary_diagnostic_pass") is not all(flags.values())
            or comparison.get("full_recovery_pass") is not None
            or any(comparison.get(field) != "not_evaluated" for field in ("family_selection", "family_recovery_rate"))
            or any(comparison.get(field) != value for field, value in levels.criterion_scope("half").items())
            or comparison.get("recovery_90pct_criterion_applies") is not False
            or comparison.get("curtailment_rule") != "not_applicable_below_SESOI"
            or comparison.get("diagnostics_not_recovery_rate_gates") != list(shared.EXTRA_DIAGNOSTICS)
            or "core_family_selection_pass" in comparison or "core_family_selection_pass" in report):
        raise ValueError("Below-SESOI diagnostics cannot supply a recovery criterion or curtailment failure")
    return {**{field: diagnostic[field] for field in CONTINUOUS_FIELDS},
            "recorded_participant_median_95CI": interval, "recorded_bootstrap": comparison["bootstrap"]}


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
    cohorts, completed = [], []
    for index in range(PLANNED):
        row = {"replicate_index": index, "seed": levels.level_seed("half", index), "status": "unknown"}
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
            row.update(reason="technical_component_failure", **shared.technical_metadata(report))
            continue
        row.update(status="continuous_diagnostic_completed", continuous_diagnostics=completed_diagnostic(report, spec))
        completed.append(index)
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ", "effect_level": "half",
        "SESOI_level": .5, "generating_z": .475, "focal_generating_truth": levels.focal_truth("half"), "implementation": spec,
        "fixed_cohort_count": PLANNED, **levels.criterion_scope("half"), "counts_as_curtailment_failure": False,
        "source_arrays": {"index1": first_array, "indices2_through199": remaining_array},
        "representative0_report": str(report_path(0, first_array, remaining_array)), "selected_terminal_indices": sorted(indices),
        "completed_indices": completed, "unknown_indices": [row["replicate_index"] for row in cohorts if row["status"] == "unknown"],
        "valid_components": len(completed), "unknown_cohorts": PLANNED - len(completed),
        "cohorts": cohorts, **{field: "not_evaluated" for field in levels.CLAIM_FIELDS},
        "full_menu_selection": "not_evaluated", "full_behavioral_gate": "not_evaluated", "full_recovery_pass": None,
        "episode_terminal": False, "full_menu_wave_selected": False, "retry_selected": False, "automatic_retry": False,
        "interpretation": "Fixed200 half-SESOI z=.475 baseline diagnostics only. Valid cohorts retain continuous paired-score and recorded median-bootstrap diagnostics; missing or technical cohorts remain unknown. No90%/180 criterion,21-failure curtailment or pass-rate claim applies below SESOI. Full-menu recovery, interval coverage, heterogeneity and the behavioral gate remain unevaluated. No further work is dispatched.",
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
    print(json.dumps({key: result[key] for key in ("phase", "valid_components", "unknown_cohorts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
