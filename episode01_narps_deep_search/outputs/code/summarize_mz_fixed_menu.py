#!/usr/bin/env python3
"""Nine nested comparisons from the completed, explicit MZ0 fixed-model panel.

The coordinator must establish that each supplied attempt is terminal before
calling this command. Completion fields alone do not establish process state.
This command neither selects a family nor calculates a recovery/confusion rate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION, validate_fit
from run_single_process_rivals import SOURCE, SEED
from summarize_single_process_recovery import summarize_reports


EPISODE = Path(__file__).resolve().parents[2]
BASE_REPORT = SOURCE / "single_process_recovery.json"
MODELS = ("M0", "MV", "MZ", "MC", "MVZ", "MVC")
MECHANISMS = {"M0": frozenset(), "MV": frozenset(("eta",)),
              "MZ": frozenset(("z",)), "MC": frozenset(("c",)),
              "MVZ": frozenset(("eta", "z")), "MVC": frozenset(("eta", "c"))}
NESTED = {"M0": (), "MV": ("M0",), "MZ": ("M0",), "MC": ("M0",),
          "MVZ": ("M0", "MV", "MZ"), "MVC": ("M0", "MV", "MC")}
EDGES = tuple((model, baseline) for model in MODELS for baseline in NESTED[model])


def episode_path(value):
    path = Path(value)
    return (path if path.is_absolute() else EPISODE / path).resolve()


def score_panel(report, models):
    scores = report.get("participant_scores")
    if not isinstance(scores, dict) or len(scores) != 108:
        raise ValueError("Every report requires the complete 108-person score panel")
    if any(not isinstance(person, str) or not person for person in scores):
        raise ValueError("Participant identities must be nonempty strings")
    expected = set(models) | {"version"}
    for row in scores.values():
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("A report has missing or extra model/score fields")
        if row["version"] not in ("EI", "ER"):
            raise ValueError("Invalid task version")
        if any(type(row[model]) not in (int, float) or not math.isfinite(row[model]) for model in models):
            raise ValueError("Participant scores must be finite numbers")
    if Counter(row["version"] for row in scores.values()) != {"EI": 54, "ER": 54}:
        raise ValueError("Both fixed 54-person task versions are required")
    return scores


def fit_panel(report, expected, implementation, people):
    fits = report.get("fits")
    if not isinstance(fits, list) or len(fits) != len(expected):
        raise ValueError("The complete ordered model/half fit panel is required")
    for fit, (model, training, scoring) in zip(fits, expected):
        validate_fit(fit, model, training, scoring, implementation, people)


def completed_support(report):
    if (not isinstance(report.get("started_utc"), str) or not report["started_utc"]
            or not isinstance(report.get("finished_utc"), str) or not report["finished_utc"]
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
        raise ValueError("Completed report must retain its full cohort support and timestamps")


def validate_base(report, implementation):
    summary = summarize_reports([("saved_MZ0", report)], "MZ", implementation)
    if summary["valid_replicates"] != 1 or report["replicate_index"] != 0 or report["seed"] != SEED:
        raise ValueError("Only the completed generating MZ replicate zero is selected")
    completed_support(report)
    if report.get("observed_choices_or_RTs_used") is not False or report.get("old_synthetic_responses_used") is not False:
        raise ValueError("The baseline must be the selected synthetic recovery cohort")
    scores = score_panel(report, ("M0", "MZ"))
    expected = [(model, train, held) for train, held in HALVES for model in ("M0", "MZ")]
    fit_panel(report, expected, implementation, scores)
    return scores


def validate_addon(report, models, base, implementation, baseline):
    complex_model = models[0] if len(models) == 1 else None
    phase = "MZ0_complex_rival_diagnostics" if complex_model else "MZ0_MC_MV_rival_diagnostics"
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != phase or report.get("generating_model") != "MZ"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != 0
            or type(report.get("seed")) is not int or report["seed"] != SEED
            or report.get("implementation") != implementation
            or report.get("source_dir") is None or episode_path(report["source_dir"]) != SOURCE.resolve()
            or report.get("source_finished_utc") != base["finished_utc"]):
        raise ValueError("Rival report does not belong to the same MZ0 cohort and implementation")
    completed_support(report)
    if (report.get("status") != "rival_diagnostics_completed"
            or report.get("rival_diagnostics_complete") is not True
            or report.get("saved_scores_replayed") is not True
            or report.get("simulation_regenerated") is not False
            or report.get("saved_M0_MZ_refitted") is not False
            or report.get("completed_rivals") != list(models)):
        raise ValueError("Only complete, nonregenerated rival panels may enter the summary")
    if (any(report.get(key) != "not_evaluated" for key in
            ("family_selection", "confusion_rate", "group_recovery", "full_gate"))
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False):
        raise ValueError("Inputs must retain their diagnostic-only claim boundary")
    if complex_model:
        if (report.get("fitted_model") != complex_model or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES]):
            raise ValueError("Complex-rival model or scored direction identity differs")
    scores = score_panel(report, ("M0", "MZ", *models))
    if set(scores) != set(baseline):
        raise ValueError("Rival and baseline participant identities differ")
    for person in baseline:
        # The add-ons deliberately copy the original recorded baseline after
        # numerical replay; these are shared saved values, not recomputed ones.
        if any(scores[person][key] != baseline[person][key] for key in ("M0", "MZ", "version")):
            raise ValueError("Shared saved M0/MZ scores or task versions differ")
    expected = [(model, train, held) for model in models for train, held in HALVES]
    fit_panel(report, expected, implementation, baseline)
    return scores


def nested_contrast(scores, model, baseline):
    people = sorted(scores)
    differences = np.array([scores[person][model] - scores[person][baseline] for person in people])
    if differences.shape != (108,) or not np.isfinite(differences).all():
        raise ValueError("Nested contrasts require all 108 finite paired differences")
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / np.sqrt(108))
    if not math.isfinite(mean) or not math.isfinite(standard_error):
        raise ValueError("Nonfinite derived contrast statistic")
    versions = np.array([scores[person]["version"] for person in people])
    requirements = {"mean_gain_at_least_0p01": mean >= .01,
                    "mean_gain_greater_than_two_participant_SE": mean > 2 * standard_error}
    return {"model": model, "nested_baseline": baseline, "participants": 108,
            "mean_gain_nats_per_trial": mean, "participant_SE": standard_error,
            "version_means": {version: float(differences[versions == version].mean()) for version in ("EI", "ER")},
            "participant_median": float(np.median(differences)), "requirements": requirements,
            "qualifies_nested_comparison": all(requirements.values())}


def build_summary(base, simple, mvz, mvc, implementation):
    if (implementation.get("policy_version") != 6 or implementation.get("state") != "recorded_before_model_scoring"
            or implementation["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or implementation.get("joint_z_c_recovery_passed") is not False):
        raise ValueError("Only the current six-model fixed-bound menu is selected")
    baseline = validate_base(base, implementation)
    panels = [(validate_addon(simple, ("MC", "MV"), base, implementation, baseline), ("MC", "MV")),
              (validate_addon(mvz, ("MVZ",), base, implementation, baseline), ("MVZ",)),
              (validate_addon(mvc, ("MVC",), base, implementation, baseline), ("MVC",))]
    scores = {person: dict(row) for person, row in baseline.items()}
    for panel, models in panels:
        for person in scores:
            for model in models:
                scores[person][model] = panel[person][model]
    contrasts = {f"{model}-{nested}": nested_contrast(scores, model, nested) for model, nested in EDGES}
    eligibility = {model: all(contrasts[f"{model}-{nested}"]["qualifies_nested_comparison"]
                             for nested in NESTED[model]) for model in MODELS}
    maximal = [model for model in MODELS if eligibility[model] and not any(
        eligibility[other] and MECHANISMS[model] < MECHANISMS[other] for other in MODELS)]
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ0_fixed_menu_nested_diagnostics",
            "generating_model": "MZ", "replicate_index": 0, "seed": SEED,
            "source_dir": str(SOURCE), "source_finished_utc": base["finished_utc"],
            "implementation": implementation, "participants": 108, "runs": 432, "trials": 27648,
            "model_menu": list(MODELS), "participant_scores": scores, "nested_comparisons": contrasts,
            "nested_comparison_eligibility": eligibility,
            "maximal_eligible_models_diagnostic": maximal,
            "multiple_incomparable_eligible_models": len(maximal) > 1,
            "family_selection": "not_evaluated", "confusion_rate": "not_evaluated",
            "group_recovery": "not_evaluated", "full_gate": "not_evaluated",
            "full_recovery_pass": None, "episode_terminal": False,
            "interpretation": "Nine fixed nested contrasts and partial-order eligibility only. No family is selected, even if one maximal eligible model remains. Nonnested arbitration awaits the scientist's choice; other recovery regimes and robustness obligations remain.",
            "process_state_boundary": "The coordinator independently establishes terminal attempts; report fields do not substitute for process or scheduler evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--simple-rivals-report", type=Path, required=True)
    parser.add_argument("--mvz-report", type=Path, required=True)
    parser.add_argument("--mvc-report", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.base_report.resolve() != BASE_REPORT.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the selected completed MZ0 baseline and unchanged implementation")
    paths = (args.base_report, args.simple_rivals_report, args.mvz_report, args.mvc_report)
    if len({path.resolve() for path in paths}) != 4:
        raise ValueError("Supply four distinct explicit input reports")
    if args.output.exists():
        raise FileExistsError("Preserve earlier summaries; select a fresh output")
    implementation = json.loads(args.implementation.read_text())
    reports = [json.loads(path.read_text()) for path in paths]
    result = build_summary(*reports, implementation)
    result.update(recorded_utc=datetime.now(timezone.utc).isoformat(), input_reports=[str(path) for path in paths])
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("phase", "nested_comparison_eligibility",
          "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models", "family_selection")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
