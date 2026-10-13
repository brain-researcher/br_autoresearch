#!/usr/bin/env python3
"""Pool explicit terminal M0/MZ null baseline cohorts, retaining fixed200 support.

The coordinator supplies terminal indices and both selected array identities.
Only those sources are opened. Missing reports and identified technical
failures remain unknown; completed pairwise diagnostics are never a family,
false-selection rate, early-kill decision, interval or full-gate verdict.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
import run_mz_null_replicate as indexed
import run_mz_null_rival as null
import summarize_mz_fixed_menu as fixed


IMPLEMENTATION = null.IMPLEMENTATION
RUNS = indexed.PILOT_DIRECTORY.parent
PHASE = "M0_null_MZ_baseline_wave_diagnostics"


def terminal_indices(value):
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token.isdecimal() for token in tokens):
        raise argparse.ArgumentTypeError("Supply explicit comma-separated terminal indices0 through199")
    indices = [int(token) for token in tokens]
    if len(set(indices)) != len(indices) or any(not 0 <= index < indexed.FIXED_COHORTS for index in indices):
        raise argparse.ArgumentTypeError("Terminal indices must be unique and within the fixed200 cohort cell")
    return sorted(indices)


def array_id(value):
    if not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("Supply an explicit positive array job ID")
    return int(value)


def source_directory(index, first_array, remaining_array):
    indexed.null_seed(index)
    if index == 0:
        return indexed.PILOT_DIRECTORY
    selected = first_array if index <= 4 else remaining_array
    if type(selected) is not int or selected < 1:
        raise ValueError("An explicit selected array ID is required for this index")
    return RUNS / f"20261005-M0-null-baseline-array-{selected}" / f"replicate{index}"


def report_path(index, first_array, remaining_array):
    name = indexed.representative.REPORT_NAME if index == 0 else indexed.REPORT_NAME
    return source_directory(index, first_array, remaining_array) / name


def validate_identity(report, index, implementation):
    expected_phase = "M0_null_MZ_pairwise_representative" if index == 0 else indexed.PHASE
    if (not isinstance(report, dict) or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != expected_phase or report.get("generating_model") != "M0"
            or report.get("fitted_models") != ["M0", "MZ"]
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != indexed.null_seed(index)
            or type(report.get("SESOI_level")) not in (int, float) or report["SESOI_level"] != 0.
            or report.get("focal_generating_truth") != indexed.representative.NULL_TRUTH
            or report.get("implementation") != implementation):
        raise ValueError("Null report index, seed, phase, generator or original implementation differs")
    job = str(report.get("job_id", ""))
    if not job.isdecimal() or int(job) < 1 or (index == 0 and job != null.SOURCE_JOB_ID):
        raise ValueError("Retain the original pilot0 or an explicit indexed worker job identity")
    for field in ("started_utc", "finished_utc"):
        value = report.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("Terminal source reports require dated timestamps with timezones")
    if (report.get("observed_choices_or_RTs_used") is not False
            or report.get("old_synthetic_responses_used") is not False
            or report.get("design_fields") != list(null.DESIGN_FIELDS)
            or report.get("design_source") is None
            or Path(report["design_source"]).resolve() != null.DESIGN_SOURCE.resolve()
            or report.get("nuisance_generation") != "recorded_center_and_individual_scale_truncated_normal_as_profile01"
            or report.get("deadline_missingness") != "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel"
            or report.get("focal_effect_between_participant_variance") != 0.):
        raise ValueError("The fixed null design, nuisance regime or response scope changed")
    if (any(report.get(field) != "not_evaluated" for field in null.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or report.get("selected_replicates") != [index]
            or "counts_as_curtailment_failure" in report or "core_family_selection_pass" in report):
        raise ValueError("Pairwise null baselines cannot supply family, rate, interval or curtailment claims")
    if index == 0:
        if report.get("wave_selected") is not False:
            raise ValueError("Preserve the original single-representative pilot0 scope")
    elif (report.get("wave_selected") is not True or report.get("wave_scope") != indexed.WAVE_SCOPE
          or type(report.get("fixed_cohort_count")) is not int or report["fixed_cohort_count"] != indexed.FIXED_COHORTS
          or report.get("full_menu_wave_selected") is not False or report.get("retry_selected") is not False):
        raise ValueError("Indexed sources must retain the fixed200 baseline-only wave scope")


def completed_diagnostic(report, truth, people, index, implementation):
    validate_identity(report, index, implementation)
    if index == 0:
        # Preserve its real identity and use its original complete-source validator.
        null.validate_source(report, truth, people, implementation)
    else:
        if (report.get("status") != "pairwise_diagnostic_completed"
                or report.get("pairwise_diagnostic_complete") is not True
                or report.get("scored_primary_directions") !=
                    [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in null.HALVES]
                or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
            raise ValueError("Only a completed full-cohort four-fit baseline panel may contribute")
        if not isinstance(truth, dict) or set(truth) != set(people):
            raise ValueError("Saved null truth participant identities differ")
        for params in truth.values():
            if (not isinstance(params, dict)
                    or set(params) != {"kappa", "log_a", "nondecision_logit", "eta", "rho", "criterion"}
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in params.values())
                    or any(params[key] != 0. for key in ("eta", "rho", "criterion"))):
                raise ValueError("Saved truth must retain the finite all-zero M0 focal terms")
            for name in ("kappa", "log_a", "nondecision_logit"):
                low, high = implementation["parameters"][name]["bounds"]
                if not low <= params[name] <= high:
                    raise ValueError("Saved nuisance truth lies outside the frozen support")
        if report.get("simulated_deadline_misses") != sum(int(row["no_response"]) for rows in people.values() for row in rows):
            raise ValueError("Saved simulation deadline-miss accounting differs")
        scores = fixed.score_panel(report, ("M0", "MZ"))
        if set(scores) != set(people) or any(scores[person]["version"] != people[person][0]["task_version"] for person in people):
            raise ValueError("Saved score participant or task-version identities differ")
        fixed.fit_panel(report, [(model, train, held) for train, held in null.HALVES for model in ("M0", "MZ")],
                        implementation, people)
    diagnostic = fixed.nested_contrast(fixed.score_panel(report, ("M0", "MZ")), "MZ", "M0")
    if report.get("nested_comparison") != diagnostic:
        raise ValueError("Recorded pairwise diagnostic differs from the complete saved score panel")
    return diagnostic


def technical_metadata(report):
    if (report.get("pairwise_diagnostic_complete") is not False
            or not isinstance(report.get("error"), str) or not report["error"]
            or "participant_scores" in report or "nested_comparison" in report
            or not isinstance(report.get("fits"), list) or any(not isinstance(fit, dict) for fit in report["fits"])):
        raise ValueError("An identified technical failure must retain its error and incomplete checkpoint state")
    fits = []
    for fit in report["fits"]:
        fits.append({**{key: fit.get(key) for key in
                       ("model", "rival", "training_runs", "scoring_runs", "converged", "elapsed_seconds", "subject_optimizer_failures")},
                     "returned_objective_sweeps": len(fit["objective_history"]) if isinstance(fit.get("objective_history"), list) else None,
                     "support_contact_count": len(fit["support_contacts"]) if isinstance(fit.get("support_contacts"), list) else None})
    return {"technical_status": report["status"], "error": report["error"],
            "started_utc": report["started_utc"], "finished_utc": report["finished_utc"],
            "process_elapsed_seconds": report.get("elapsed_seconds"),
            "simulation_seconds": report.get("simulation_seconds"),
            "scored_primary_directions": report.get("scored_primary_directions"), "returned_fits": fits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-array-job-id", type=array_id, required=True)
    parser.add_argument("--remaining-array-job-id", type=array_id)
    parser.add_argument("--terminal-indices", type=terminal_indices, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if any(index >= 5 for index in args.terminal_indices) and args.remaining_array_job_id is None:
        raise ValueError("Select the remaining array explicitly before opening indices5 through199")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output")
    implementation = json.loads(IMPLEMENTATION.read_text())
    if (implementation.get("policy_version") != 6 or implementation.get("state") != "recorded_before_model_scoring"
            or implementation["optimizer"]["maximum_sweeps"] != 500
            or implementation["recovery_plan"]["replicates_per_qualified_generating_cell"] != indexed.FIXED_COHORTS
            or implementation["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or implementation["contaminant"]["fraction"] != .02 or implementation["contaminant"]["miss_probability"] != .01):
        raise ValueError("Preserve the original500 implementation and fixed200 null cohort plan")
    cohorts, completed = [], []
    for index in range(indexed.FIXED_COHORTS):
        row = {"replicate_index": index, "seed": indexed.null_seed(index), "status": "unknown"}
        cohorts.append(row)
        if index not in args.terminal_indices:
            row["reason"] = "not_selected_as_terminal_by_coordinator"
            continue
        path = report_path(index, args.first_array_job_id, args.remaining_array_job_id)
        row["report"] = str(path)
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            row["reason"] = "selected_terminal_report_missing"
            continue
        validate_identity(report, index, implementation)
        if report.get("status") == "technical_representative_failure":
            row.update(reason="technical_baseline_failure", **technical_metadata(report))
            continue
        if report.get("status") != "pairwise_diagnostic_completed" or report.get("pairwise_diagnostic_complete") is not True:
            raise ValueError("A selected report must be a completed baseline or an identified technical failure")
        truth_path, events_path = path.parent / "generating_parameters.json", path.parent / "simulated_events.csv.gz"
        truth = json.loads(truth_path.read_text())
        people = null.shared.load_cohort(events_path)
        diagnostic = completed_diagnostic(report, truth, people, index, implementation)
        row.update(status="pairwise_diagnostic_completed", nested_comparison=diagnostic,
                   generating_parameters=str(truth_path), simulated_events=str(events_path))
        completed.append(index)
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
              "generating_model": "M0", "focal_generating_truth": dict(indexed.representative.NULL_TRUTH),
              "implementation": implementation, "fixed_cohort_count": indexed.FIXED_COHORTS,
              "wave_scope": indexed.WAVE_SCOPE, "selected_terminal_indices": args.terminal_indices,
              "source_arrays": {"indices1_through4": args.first_array_job_id, "indices5_through199": args.remaining_array_job_id},
              "pilot0_report": str(report_path(0, args.first_array_job_id, args.remaining_array_job_id)),
              "completed_indices": completed, "unknown_indices": [row["replicate_index"] for row in cohorts if row["status"] == "unknown"],
              "cohorts": cohorts,
              "diagnostic_counts": {"completed_pairwise_diagnostics": len(completed),
                                    "unknown_cohorts": indexed.FIXED_COHORTS - len(completed),
                                    "MZ_M0_nested_margin_qualifications": sum(row["nested_comparison"]["qualifies_nested_comparison"]
                                        for row in cohorts if row["status"] == "pairwise_diagnostic_completed")},
              **{field: "not_evaluated" for field in null.CLAIM_FIELDS}, "early_kill": "not_evaluated",
              "full_recovery_pass": None, "episode_terminal": False, "full_menu_wave_selected": False, "retry_selected": False,
              "interpretation": "Counts describe complete M0/MZ pairwise diagnostics only. Unknown cohorts are not scientific failures. Pairwise qualification supplies no full-menu false selection, false-selection rate, family selection or early-kill decision.",
              "process_state_boundary": "The coordinator separately establishes terminal processes; report fields and missing files are not scheduler evidence.",
              "recorded_utc": datetime.now(timezone.utc).isoformat()}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("phase", "fixed_cohort_count", "completed_indices", "diagnostic_counts", "false_selection_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
