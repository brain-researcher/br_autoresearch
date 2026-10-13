#!/usr/bin/env python3
"""Nine fixed nested diagnostics for the one completed M0-null model menu.

The coordinator establishes terminal input processes before invocation. This
summary retains the null generator and selects no family, false-selection
rate, interval verdict, full gate, or additional null simulation wave.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION
from summarize_mz_saved_replicate_menu import completed_support
import run_mz_null_rival as null
import summarize_mz_fixed_menu as fixed


BASE_REPORT = null.SOURCE / null.SOURCE_REPORT
RIVALS = ("MC", "MV", "MVZ", "MVC")
SELECTED_JOB_IDS = {"MVZ": "46575931", "MVC": "46576234"}


def validate_rival(report, model, base, implementation, baseline):
    if (not isinstance(report, dict) or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != "M0_null_saved_representative_rival_diagnostics"
            or report.get("generating_model") != "M0" or report.get("fitted_model") != model
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != null.REPLICATE_INDEX
            or type(report.get("seed")) is not int or report["seed"] != null.SEED
            or report.get("implementation") != implementation or report.get("source_dir") is None
            or fixed.episode_path(report["source_dir"]) != null.SOURCE.resolve()
            or report.get("source_report") != null.SOURCE_REPORT
            or str(report.get("source_job_id")) != null.SOURCE_JOB_ID
            or report.get("source_finished_utc") != base["finished_utc"]):
        raise ValueError("Rival report has a different null cohort, source, model or implementation")
    job_id = str(report.get("job_id", ""))
    if not job_id.isdecimal() or (model in SELECTED_JOB_IDS and job_id != SELECTED_JOB_IDS[model]):
        raise ValueError("Use the selected MVZ/MVC attempts and an explicit MC/MV job identity")
    completed_support(report)
    if (report.get("status") != "rival_diagnostics_completed"
            or report.get("rival_diagnostics_complete") is not True
            or report.get("saved_scores_replayed") is not True
            or report.get("simulation_regenerated") is not False
            or report.get("saved_M0_MZ_refitted") is not False
            or report.get("observed_choices_or_RTs_used") is not False
            or report.get("old_synthetic_responses_used") is not False
            or report.get("completed_rivals") != [model]
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES]):
        raise ValueError("Require a completed, nonregenerated rival with both scored directions")
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
            or not 0 <= discrepancy <= null.shared.REPLAY_ABSOLUTE_TOLERANCE
            or report.get("saved_score_replay_absolute_tolerance") != null.shared.REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("Rival report lacks successful replay at the unchanged absolute tolerance")
    if (any(report.get(field) != "not_evaluated" for field in null.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or report.get("wave_selected") is not False):
        raise ValueError("Rival report exceeds the single-cohort diagnostic-only claim boundary")
    scores = fixed.score_panel(report, ("M0", "MZ", model))
    if set(scores) != set(baseline):
        raise ValueError("Rival and baseline participant identities differ")
    for person in baseline:
        if any(scores[person][key] != baseline[person][key] for key in ("M0", "MZ", "version")):
            raise ValueError("Exact saved M0/MZ scores or task versions differ")
    fixed.fit_panel(report, [(model, train, held) for train, held in HALVES], implementation, baseline)
    return scores


def build_summary(base, mc, mv, mvz, mvc, implementation, truth, people):
    if implementation.get("joint_z_c_recovery_passed") is not False:
        raise ValueError("Only the current six-model fixed-bound menu is selected")
    null.validate_source(base, truth, people, implementation)
    baseline = fixed.score_panel(base, ("M0", "MZ"))
    panels = [validate_rival(report, model, base, implementation, baseline)
              for model, report in zip(RIVALS, (mc, mv, mvz, mvc))]
    scores = {person: dict(row) for person, row in baseline.items()}
    for model, panel in zip(RIVALS, panels):
        for person in scores:
            scores[person][model] = panel[person][model]
    contrasts = {f"{model}-{nested}": fixed.nested_contrast(scores, model, nested)
                 for model, nested in fixed.EDGES}
    eligibility = {model: all(contrasts[f"{model}-{nested}"]["qualifies_nested_comparison"]
                             for nested in fixed.NESTED[model]) for model in fixed.MODELS}
    maximal = [model for model in fixed.MODELS if eligibility[model] and not any(
        eligibility[other] and fixed.MECHANISMS[model] < fixed.MECHANISMS[other] for other in fixed.MODELS)]
    return {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": "M0_null_fixed_menu_nested_diagnostics",
        "generating_model": "M0", "replicate_index": null.REPLICATE_INDEX, "seed": null.SEED,
        "SESOI_level": 0., "focal_generating_truth": {"eta": 0., "rho": 0., "criterion": 0.},
        "source_dir": str(null.SOURCE), "source_finished_utc": base["finished_utc"],
        "implementation": implementation, "participants": 108, "runs": 432, "trials": 27648,
        "model_menu": list(fixed.MODELS), "participant_scores": scores, "nested_comparisons": contrasts,
        "nested_comparison_eligibility": eligibility, "maximal_eligible_models_diagnostic": maximal,
        "multiple_incomparable_eligible_models": len(maximal) > 1,
        **{field: "not_evaluated" for field in null.CLAIM_FIELDS},
        "full_recovery_pass": None, "episode_terminal": False, "selected_replicates": [0], "wave_selected": False,
        "interpretation": "Nine existing nested contrasts and partial-order eligibility on one M0-null cohort only. No family or false selection is declared even with one maximal model; no nonnested arbitration, rate, interval or gate verdict is computed.",
        "process_state_boundary": "The coordinator separately establishes terminal input processes; report completion fields are not scheduler evidence.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--mc-report", type=Path, required=True)
    parser.add_argument("--mv-report", type=Path, required=True)
    parser.add_argument("--mvz-report", type=Path, required=True)
    parser.add_argument("--mvc-report", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.base_report.resolve() != BASE_REPORT.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the exact completed null pilot report and unchanged implementation")
    paths = (args.base_report, args.mc_report, args.mv_report, args.mvz_report, args.mvc_report)
    if len({path.resolve() for path in paths}) != 5:
        raise ValueError("Supply five distinct explicit input reports")
    if args.output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output")
    implementation = json.loads(args.implementation.read_text())
    reports = [json.loads(path.read_text()) for path in paths]
    truth_path = null.SOURCE / "generating_parameters.json"
    events_path = null.SOURCE / "simulated_events.csv.gz"
    truth = json.loads(truth_path.read_text())
    people = null.shared.load_cohort(events_path)
    result = build_summary(*reports, implementation, truth, people)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  input_reports=[str(path.resolve()) for path in paths],
                  generating_parameters=str(truth_path), simulated_events=str(events_path))
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in
                     ("phase", "generating_model", "nested_comparison_eligibility",
                      "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models",
                      "family_selection", "full_menu_false_selection")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
