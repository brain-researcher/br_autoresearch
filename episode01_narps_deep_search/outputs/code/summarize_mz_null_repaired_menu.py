#!/usr/bin/env python3
"""One null0 menu with the exact completed MVC second-half repair46582236.

Keep the original first-half fit and explicit500/1000 validation provenance.
Reuse the existing nine nested contrasts without choosing a family or rate.
The coordinator establishes terminal processes before invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_mvc_convergence_repair as repair
import summarize_mz_null_menu as menu


REPAIR_JOB_ID = "46582236"
SIMPLE_JOB_ID = "46577173"
PHASE = "M0_null_repaired_MVC_menu_nested_diagnostics"
RUNS = menu.null.SOURCE.parent
REPAIR_REPORT = RUNS / f"20261005-M0-null-MVC-convergence-{REPAIR_JOB_ID}" / repair.REPORT_NAME
INPUT_REPORTS = (
    menu.BASE_REPORT,
    RUNS / f"20261005-M0-null-MC-rival-{SIMPLE_JOB_ID}/mz_null_rival.json",
    RUNS / f"20261005-M0-null-MV-rival-{SIMPLE_JOB_ID}/mz_null_rival.json",
    RUNS / "20261005-M0-null-MVZ-rival-46575931/mz_null_rival.json",
    REPAIR_REPORT,
    repair.FAILED_REPORT,
)


def validate_repaired_mvc(report, base, implementation, baseline, failed, people):
    """Called only after the unchanged null-menu builder validates the source."""
    runtime, original_first, metadata = repair.repair_context(implementation, failed, base, people)
    if (not isinstance(report, dict) or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != repair.PHASE or str(report.get("job_id")) != REPAIR_JOB_ID
            or report.get("generating_model") != "M0" or report.get("fitted_model") != "MVC"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != 0
            or type(report.get("seed")) is not int or report["seed"] != 1300037
            or report.get("source_dir") is None
            or menu.fixed.episode_path(report["source_dir"]) != menu.null.SOURCE.resolve()
            or report.get("source_report") != menu.null.SOURCE_REPORT
            or str(report.get("source_job_id")) != menu.null.SOURCE_JOB_ID
            or report.get("source_finished_utc") != base["finished_utc"]
            or report.get("implementation") != implementation
            or report.get("runtime_implementation") != runtime or report.get("numerical_repair") != metadata):
        raise ValueError("Only completed null0 MVC repair46582236 with its exact source and repair context is selected")
    menu.completed_support(report)
    if (report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True
            or report.get("completed_rivals") != ["MVC"]
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in menu.HALVES]
            or report.get("saved_scores_replayed") is not True
            or any(report.get(field) is not False for field in
                   ("simulation_regenerated", "saved_M0_MZ_refitted", "observed_choices_or_RTs_used",
                    "old_synthetic_responses_used", "wave_selected", "retry_wave_selected", "full_menu_wave_selected"))
            or any(report.get(field) is not True for field in
                   ("repair_context_validated", "first_half_fit_reused", "fresh_second_half_fit_called"))):
        raise ValueError("Require both completed repair directions with validated reuse and no regenerated baseline")
    expected_provenance = [
        {"training_runs": [1, 4], "scoring_runs": [2, 3], "fit_source": str(repair.FAILED_REPORT),
         "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": 500},
        {"training_runs": [2, 3], "scoring_runs": [1, 4], "fit_source": "current_repair",
         "action": "fresh_fit", "validation_maximum_sweeps": 1000},
    ]
    if (report.get("per_half_provenance") != expected_provenance
            or report.get("selected_replicates") != [0] or report.get("selected_models") != ["MVC"]
            or any(report.get(field) != "not_evaluated" for field in menu.null.CLAIM_FIELDS)
            or report.get("early_kill", "not_evaluated") != "not_evaluated"
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False):
        raise ValueError("Preserve mixed original500/fresh1000 provenance and single-cohort diagnostic claims")
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
            or not 0 <= discrepancy <= menu.null.shared.REPLAY_ABSOLUTE_TOLERANCE
            or report.get("saved_score_replay_absolute_tolerance") != menu.null.shared.REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("The successful original baseline replay evidence is required")
    scores = menu.fixed.score_panel(report, ("M0", "MZ", "MVC"))
    if set(scores) != set(baseline) or any(scores[person][key] != baseline[person][key]
            for person in baseline for key in ("M0", "MZ", "version")):
        raise ValueError("Preserve the exact original baseline identities, task versions and score last bits")
    fits = report.get("fits")
    if not isinstance(fits, list) or len(fits) != 2 or any(not isinstance(fit, dict) for fit in fits):
        raise ValueError("The complete ordered pair of repaired MVC fits is required")
    for fit, (train, held), spec in zip(fits, menu.HALVES, (implementation, runtime)):
        menu.fixed.validate_fit(fit, "MVC", train, held, spec, people)
    if ({key: value for key, value in fits[0].items() if key != "elapsed_seconds"}
            != {key: value for key, value in original_first.items() if key != "elapsed_seconds"}):
        raise ValueError("The first-half fit must preserve every original parameter, history and diagnostic")
    for value in (report.get("elapsed_seconds"), report.get("cumulative_process_elapsed_seconds"),
                  *(fit.get("elapsed_seconds") for fit in fits)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Preserve finite nonnegative new process and per-fit timing")
    if (type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= 5400
            or report["cumulative_process_elapsed_seconds"] !=
                metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]):
        raise ValueError("Cumulative cost must count the prior process and new process exactly once")
    return scores


def build_summary(base, mc, mv, mvz, mvc, failed, implementation, truth, people):
    if any(not isinstance(report, dict) for report in (base, mc, mv, mvz, mvc, failed)):
        raise ValueError("Require all five complete model reports and the original failed MVC attempt")
    if (implementation["optimizer"]["maximum_sweeps"] != 500
            or implementation["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200):
        raise ValueError("Keep the original500 implementation and fixed200 recovery plan")
    original_validator = menu.validate_rival

    def selected_validator(report, model, source, spec, baseline):
        if model != "MVC":
            if model in ("MC", "MV") and str(report.get("job_id")) != SIMPLE_JOB_ID:
                raise ValueError("Use the selected completed MC/MV components from46577173")
            return original_validator(report, model, source, spec, baseline)
        if report is not mvc or source is not base or spec != implementation:
            raise ValueError("The repaired MVC admission is scoped to this one null menu")
        return validate_repaired_mvc(report, source, spec, baseline, failed, people)

    # The original builder validates the source and owns all nine contrasts and
    # partial-order calculations. Only this process's MVC admission is adapted.
    with patch.object(menu, "validate_rival", selected_validator):
        result = menu.build_summary(base, mc, mv, mvz, mvc, implementation, truth, people)
    result.update(phase=PHASE, runtime_implementation=mvc["runtime_implementation"],
                  numerical_repair=mvc["numerical_repair"], per_half_provenance=mvc["per_half_provenance"],
                  repaired_mvc_report=str(REPAIR_REPORT), original_failed_mvc_report=str(repair.FAILED_REPORT),
                  repair_process_elapsed_seconds=mvc["elapsed_seconds"],
                  cumulative_mvc_process_elapsed_seconds=mvc["cumulative_process_elapsed_seconds"],
                  early_kill="not_evaluated", retry_wave_selected=False, full_menu_wave_selected=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != menu.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier summaries; select a fresh output")
    implementation = json.loads(menu.IMPLEMENTATION.read_text())
    reports = [json.loads(path.read_text()) for path in INPUT_REPORTS]
    truth_path = menu.null.SOURCE / "generating_parameters.json"
    events_path = menu.null.SOURCE / "simulated_events.csv.gz"
    truth = json.loads(truth_path.read_text())
    people = menu.null.shared.load_cohort(events_path)
    result = build_summary(*reports, implementation, truth, people)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  input_reports=[str(path) for path in INPUT_REPORTS],
                  generating_parameters=str(truth_path), simulated_events=str(events_path))
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("phase", "generating_model", "nested_comparison_eligibility",
        "maximal_eligible_models_diagnostic", "family_selection", "full_menu_false_selection")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
