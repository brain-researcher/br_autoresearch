#!/usr/bin/env python3
"""Nine fixed nested diagnostics from five terminal reports for one MZ index.

Accept the original M0/MZ report and exactly one completed MC/MV/MVZ/MVC
add-on each. Root must establish terminal process state separately; no scheduler
query, source simulation read, fitting, family arbitration or rate is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
from run_single_process_recovery import DESIGN_FIELDS, DESIGN_SOURCE, HALVES, IMPLEMENTATION, cell_seed
from run_single_process_rivals import REPLAY_ABSOLUTE_TOLERANCE
from run_mz_saved_replicate_rival import source_directory
from summarize_single_process_recovery import summarize_reports
import summarize_mz_fixed_menu as fixed


RIVALS = ("MC", "MV", "MVZ", "MVC")


def completed_support(report):
    fixed.completed_support(report)
    for field in ("started_utc", "finished_utc"):
        if datetime.fromisoformat(report[field]).utcoffset() is None:
            raise ValueError("Terminal report timestamps require explicit timezones")


def validate_base(base, implementation, index):
    source_directory(index)
    summary = summarize_reports([(f"saved_MZ_replicate{index}", base)], "MZ", implementation)
    if (summary["valid_replicates"] != 1 or base["replicate_index"] != index
            or base["seed"] != cell_seed("MZ", index)):
        raise ValueError("Only the completed original MZ report for the selected index/seed is allowed")
    completed_support(base)
    if (base.get("observed_choices_or_RTs_used") is not False
            or base.get("old_synthetic_responses_used") is not False
            or base.get("design_fields") != list(DESIGN_FIELDS)
            or Path(base.get("design_source", "")).resolve() != DESIGN_SOURCE.resolve()):
        raise ValueError("The baseline must retain the original synthetic source scope")
    baseline = fixed.score_panel(base, ("M0", "MZ"))
    fixed.fit_panel(base, [(model, train, held) for train, held in HALVES for model in ("M0", "MZ")],
                    implementation, baseline)
    return baseline


def validate_rival(report, model, base, implementation, baseline, index):
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != "MZ_SESOI_saved_replicate_rival_diagnostics"
            or report.get("generating_model") != "MZ" or report.get("fitted_model") != model
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != cell_seed("MZ", index)
            or report.get("implementation") != implementation or report.get("source_dir") is None
            or fixed.episode_path(report["source_dir"]) != source_directory(index).resolve()
            or report.get("source_finished_utc") != base["finished_utc"]):
        raise ValueError("Rival report has a different index, seed, source, model or implementation")
    completed_support(report)
    if (report.get("status") != "rival_diagnostics_completed"
            or report.get("rival_diagnostics_complete") is not True
            or report.get("saved_scores_replayed") is not True
            or report.get("simulation_regenerated") is not False
            or report.get("saved_M0_MZ_refitted") is not False
            or report.get("observed_choices_or_RTs_used") is not False
            or report.get("completed_rivals") != [model]
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES]):
        raise ValueError("Only complete two-direction, nonregenerated rival reports may enter")
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
            or not 0 <= discrepancy <= REPLAY_ABSOLUTE_TOLERANCE
            or report.get("saved_score_replay_absolute_tolerance") != REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("Rival report lacks the fixed successful numerical replay evidence")
    if (any(report.get(field) != "not_evaluated" for field in
            ("family_selection", "confusion_rate", "group_recovery", "interval_coverage", "full_gate"))
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False):
        raise ValueError("Rival report exceeds the diagnostic-only claim boundary")
    scores = fixed.score_panel(report, ("M0", "MZ", model))
    if set(scores) != set(baseline):
        raise ValueError("Rival and baseline participant identities differ")
    for person in baseline:
        if any(scores[person][key] != baseline[person][key] for key in ("M0", "MZ", "version")):
            raise ValueError("The exact saved M0/MZ scores or task versions differ")
    fixed.fit_panel(report, [(model, train, held) for train, held in HALVES], implementation, baseline)
    return scores


def build_summary(base, mc, mv, mvz, mvc, implementation, index):
    if (implementation.get("policy_version") != 6 or implementation.get("state") != "recorded_before_model_scoring"
            or implementation["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or implementation.get("joint_z_c_recovery_passed") is not False):
        raise ValueError("Only the current six-model fixed-bound menu is selected")
    baseline = validate_base(base, implementation, index)
    panels = [validate_rival(report, model, base, implementation, baseline, index)
              for model, report in zip(RIVALS, (mc, mv, mvz, mvc))]
    scores = {person: dict(row) for person, row in baseline.items()}
    for model, panel in zip(RIVALS, panels):
        for person in scores:
            scores[person][model] = panel[person][model]
    # Reuse the existing numerical contrast function and exact registered
    # edge/mechanism constants. There is no new threshold or nonnested ranking.
    contrasts = {f"{model}-{nested}": fixed.nested_contrast(scores, model, nested)
                 for model, nested in fixed.EDGES}
    eligibility = {model: all(contrasts[f"{model}-{nested}"]["qualifies_nested_comparison"]
                             for nested in fixed.NESTED[model]) for model in fixed.MODELS}
    maximal = [model for model in fixed.MODELS if eligibility[model] and not any(
        eligibility[other] and fixed.MECHANISMS[model] < fixed.MECHANISMS[other] for other in fixed.MODELS)]
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_saved_replicate_menu_nested_diagnostics",
            "generating_model": "MZ", "replicate_index": index, "seed": cell_seed("MZ", index),
            "source_dir": str(source_directory(index)), "source_finished_utc": base["finished_utc"],
            "implementation": implementation, "participants": 108, "runs": 432, "trials": 27648,
            "model_menu": list(fixed.MODELS), "nested_comparisons": contrasts,
            "nested_comparison_eligibility": eligibility, "maximal_eligible_models_diagnostic": maximal,
            "multiple_incomparable_eligible_models": len(maximal) > 1,
            "family_selection": "not_evaluated", "confusion_rate": "not_evaluated",
            "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated", "full_gate": "not_evaluated",
            "full_recovery_pass": None, "episode_terminal": False,
            "interpretation": "Nine fixed nested contrasts and partial-order eligibility only. No family is selected, even if one maximal model remains; no nonnested arbitration or recovery/confusion rate is computed.",
            "process_state_boundary": "Root independently establishes terminal attempts; report completion fields are not scheduler evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--mc-report", type=Path, required=True)
    parser.add_argument("--mv-report", type=Path, required=True)
    parser.add_argument("--mvz-report", type=Path, required=True)
    parser.add_argument("--mvc-report", type=Path, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    expected = source_directory(args.replicate_index) / "single_process_recovery.json"
    if args.base_report.resolve() != expected.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the exact fixed-index original report and unchanged implementation")
    paths = (args.base_report, args.mc_report, args.mv_report, args.mvz_report, args.mvc_report)
    if len({path.resolve() for path in paths}) != 5:
        raise ValueError("Supply five distinct explicit input reports")
    if args.output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output")
    implementation = json.loads(args.implementation.read_text())
    reports = [json.loads(path.read_text()) for path in paths]
    result = build_summary(*reports, implementation, args.replicate_index)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(), input_reports=[str(path) for path in paths])
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("phase", "replicate_index", "nested_comparison_eligibility",
        "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models", "family_selection")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
