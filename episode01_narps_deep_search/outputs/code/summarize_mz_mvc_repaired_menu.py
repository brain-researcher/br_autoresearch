#!/usr/bin/env python3
"""Fixed index42 menu after successful MVC reverse completion46587905 only.

The coordinator establishes terminal processes before invocation. Original500
baseline/rivals, preserved first1000 and fresh reverse1500 stay explicit. Both
failed attempts remain provenance, never complete-menu inputs or wave results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_mvc_reverse_completion as repair
import summarize_mz_saved_replicate_menu as saved


INDEX = 42
COMPLETION_JOB_ID = "46587905"
IMPLEMENTATION = repair.saved.IMPLEMENTATION
BASE_REPORT = repair.saved.source_directory(INDEX) / "single_process_recovery.json"
RIVAL_REPORTS = tuple(repair.saved.RUNS_ROOT /
    f"20261005-MZ-{model}-saved-array-{job}/replicate42/mz_saved_replicate_rival.json"
    for model, job in (("MC", 46550138), ("MV", 46550142), ("MVZ", 46550145)))
COMPLETION_REPORT = repair.saved.RUNS_ROOT / f"20261005-MZ42-MVC-reverse-completion-{COMPLETION_JOB_ID}" / repair.REPORT_NAME
INPUT_REPORTS = (BASE_REPORT, *RIVAL_REPORTS, COMPLETION_REPORT, repair.ORIGINAL_REPORT, repair.PILOT_REPORT)
PHASE = "MZ42_MVC_repaired_menu_nested_diagnostics"
CLAIM_FIELDS = repair.CLAIM_FIELDS


def validate_completion(report, base, implementation, baseline, failed, previous):
    """Reconstruct both prior attempts after the original builder validates base."""
    first_runtime, runtime, preserved_first, expected = repair.repair_context(
        implementation, failed, previous, base, baseline)
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != repair.PHASE or report.get("job_id") != COMPLETION_JOB_ID
            or report.get("generating_model") != "MZ" or report.get("fitted_model") != "MVC"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != INDEX
            or type(report.get("seed")) is not int or report["seed"] != saved.cell_seed("MZ", INDEX)
            or report.get("implementation") != implementation
            or report.get("first_half_runtime_implementation") != first_runtime
            or report.get("runtime_implementation") != runtime
            or saved.fixed.episode_path(report.get("source_dir", "")) != repair.saved.source_directory(INDEX).resolve()
            or report.get("source_finished_utc") != base["finished_utc"]):
        raise ValueError("Only completed MVC reverse repair46587905 with original500/first1000/reverse1500 is allowed")
    saved.completed_support(report)
    if (report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True
            or report.get("saved_scores_replayed") is not True
            or any(report.get(field) is not False for field in
                   ("simulation_regenerated", "saved_M0_MZ_refitted", "observed_choices_or_RTs_used",
                    "wave_selected", "retry_wave_selected", "automatic_retry", "full_menu_wave_selected"))
            or any(report.get(field) is not True for field in
                   ("repair_context_validated", "first_half_fit_reused", "fresh_reverse_fit_called"))
            or report.get("completed_rivals") != ["MVC"]
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in saved.HALVES]):
        raise ValueError("Require both completed MVC directions, preserved-first reuse and original baseline replay")
    provenance = [
        {"training_runs": [1, 4], "scoring_runs": [2, 3], "fit_source": str(repair.PILOT_REPORT),
         "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": 1000},
        {"training_runs": [2, 3], "scoring_runs": [1, 4], "fit_source": "current_completion",
         "action": "fresh_fit", "validation_maximum_sweeps": 1500},
    ]
    if (report.get("per_half_provenance") != provenance or report.get("baseline_validation_maximum_sweeps") != 500
            or any(report.get(field) != "not_evaluated" for field in CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or report.get("selected_replicates") != [INDEX] or report.get("selected_models") != ["MVC"]):
        raise ValueError("Retain mixed fit provenance and single-cohort diagnostic-only claims")
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
            or not 0 <= discrepancy <= saved.REPLAY_ABSOLUTE_TOLERANCE
            or report.get("saved_score_replay_absolute_tolerance") != saved.REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("The fixed successful saved-score replay evidence is required")
    scores = saved.fixed.score_panel(report, ("M0", "MZ", "MVC"))
    if set(scores) != set(baseline) or any(scores[person][key] != baseline[person][key]
            for person in baseline for key in ("M0", "MZ", "version")):
        raise ValueError("Completion participant identities, original baseline last bits or task versions differ")
    fits = report.get("fits")
    if not isinstance(fits, list) or len(fits) != 2:
        raise ValueError("Two completed ordered MVC fits are required")
    for fit, (train, held), spec in zip(fits, saved.HALVES, (first_runtime, runtime)):
        saved.fixed.validate_fit(fit, "MVC", train, held, spec, baseline)
    if ({key: value for key, value in fits[0].items() if key != "elapsed_seconds"}
            != {key: value for key, value in preserved_first.items() if key != "elapsed_seconds"}):
        raise ValueError("Preserve every first-fit parameter, history and diagnostic; only copy timing may differ")
    recorded = report.get("numerical_repair")
    if not isinstance(recorded, dict):
        raise ValueError("The reconstructed original/pilot repair context is required")
    expected = deepcopy(expected)
    for field in ("first_half_materialization_elapsed_seconds", "first_half_rescore_elapsed_seconds"):
        value = recorded.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Preserve finite new copy and rescore costs separately from the original first fit")
        expected[field] = value
    expected["first_half_rescored"] = True
    if recorded != expected or recorded["first_half_materialization_elapsed_seconds"] != fits[0].get("elapsed_seconds"):
        raise ValueError("Repair context or successful preserved-fit rescore evidence differs")
    for value in (report.get("elapsed_seconds"), report.get("cumulative_process_elapsed_seconds"),
                  *(fit.get("elapsed_seconds") for fit in fits)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Require finite nonnegative new process and returned-fit costs")
    if (type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= 5400
            or report["cumulative_process_elapsed_seconds"] !=
                expected["previous_cumulative_process_elapsed_seconds"] + report["elapsed_seconds"]):
        raise ValueError("Count the prior cumulative process cost plus this completion exactly once")
    return scores


def build_summary(base, mc, mv, mvz, mvc, failed, previous, implementation):
    if any(not isinstance(report, dict) for report in (base, mc, mv, mvz, mvc, failed, previous)):
        raise ValueError("Require all five complete model reports and both preserved failed attempts")
    if (implementation["optimizer"]["maximum_sweeps"] != 500
            or implementation["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200):
        raise ValueError("Keep the original500 implementation and fixed200 cohort plan")
    original_validator = saved.validate_rival

    def selected_validator(report, model, source, spec, baseline, index):
        if model != "MVC":
            return original_validator(report, model, source, spec, baseline, index)
        if index != INDEX or report is not mvc or source is not base or spec != implementation:
            raise ValueError("The completion adapter is scoped to this one MVC panel")
        return validate_completion(report, source, spec, baseline, failed, previous)

    # Only the prepared adapter's process-local MVC admission changes. The
    # existing builder owns base/rival validation and all nine nested contrasts.
    with patch.object(saved, "validate_rival", selected_validator):
        result = saved.build_summary(base, mc, mv, mvz, mvc, implementation, INDEX)
    result.update(phase=PHASE, first_half_runtime_implementation=mvc["first_half_runtime_implementation"],
                  runtime_implementation=mvc["runtime_implementation"], numerical_repair=mvc["numerical_repair"],
                  per_half_provenance=mvc["per_half_provenance"], baseline_validation_maximum_sweeps=500,
                  mvc_completion_report=str(COMPLETION_REPORT), original_failed_mvc_report=str(repair.ORIGINAL_REPORT),
                  failed_mvc_pilot_report=str(repair.PILOT_REPORT), selected_replicates=[INDEX],
                  fixed_cohort_count=200, wave_selected=False, retry_wave_selected=False, full_menu_wave_selected=False,
                  completion_process_elapsed_seconds=mvc["elapsed_seconds"],
                  cumulative_process_elapsed_seconds=mvc["cumulative_process_elapsed_seconds"],
                  **{field: "not_evaluated" for field in CLAIM_FIELDS})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output")
    implementation = json.loads(IMPLEMENTATION.read_text())
    reports = [json.loads(path.read_text()) for path in INPUT_REPORTS]
    result = build_summary(*reports, implementation)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(), input_reports=[str(path) for path in INPUT_REPORTS])
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("phase", "replicate_index", "nested_comparison_eligibility",
        "maximal_eligible_models_diagnostic", "family_selection", "retry_wave_selected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
