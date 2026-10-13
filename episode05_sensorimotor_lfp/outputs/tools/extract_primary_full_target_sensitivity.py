#!/usr/bin/env python3
"""Extract a complete full-target sensitivity summary without refitting or source reads."""
import argparse
import json
from pathlib import Path

from extract_primary_relevance import ANIMALS, bounds, seed_summaries, thin_panel
from primary_full_target_sensitivity import RECIPES
from primary_reference import SESSIONS
from source_adapter import compute_only


def diagnostics(rows):
    folds = [fold for row in rows for fold in row["selected_fit_diagnostics"]]
    if len(folds) != 8*len(rows) or any(len(f["controls_equal_candidate_by_replicate"]) != 8 for f in folds):
        raise ValueError("Eight held-direction fits and eight controls per fit required")
    return {"fold_count": len(folds),
        **{key+"_range": bounds(fold[key] for fold in folds) for key in ("trace", "numerical_rank", "effective_df")},
        "native_training_mu_X_range": bounds(fold["mu"][0] for fold in folds),
        "native_training_mu_Y_range": bounds(fold["mu"][1] for fold in folds),
        "baseline_equals_mu_fold_count": sum(fold["baseline_equals_mu"] for fold in folds),
        "all_folds_baseline_equals_mu": all(fold["baseline_equals_mu"] for fold in folds),
        "exact_control_cross_equality_by_replicate": [{"replicate": r,
            "equal_candidate_fold_count": sum(fold["controls_equal_candidate_by_replicate"][r] for fold in folds),
            "all_folds_equal_candidate": all(fold["controls_equal_candidate_by_replicate"][r] for fold in folds)} for r in range(8)],
        "interpretation": "Native global target mean; shared candidate/control input Gram and ridge df; repeated folds are not independent animals"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    report = json.loads(args.summary.read_text())
    if (report.get("status") != "complete_full_target_sensitivity_control_panel"
            or report["sensitivity_only"] is not True or report["promotion_authorized"] is not False):
        raise ValueError("Only a complete nonpromotable full-target sensitivity may be extracted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES):
        raise ValueError("Both full-target sensitivity recipes required")
    for panel in [*panels.values(), report["training_target_label_control"], *report["anchor_contrasts"].values()]:
        if len(panel["sessions"]) != 16 or {r["session_id"] for r in panel["sessions"]} != set(SESSIONS):
            raise ValueError("All original sessions required in every panel and contrast")
    candidate, control = (panels[recipe] for recipe in RECIPES)
    keys = ("status", "scientific_trial_count_increment", "control_trial_count_increment",
            "scientific_trial_count_by_recipe", "total_valid_primary_trials_including_prior",
            "post_initial_comparisons", "post_initial_controls", "structural_residual_anchor_duplicate",
            "all_controls_equal_candidate", "training_pairing_directional_screen_pass",
            "sensitivity_only", "promotion_authorized", "anchor_noninferiority_requirement")
    result = {"source_summary": str(args.summary), **{key: report[key] for key in keys}}
    result.update(panels=[thin_panel(candidate), thin_panel(control)],
        candidate_minus_mean_control=thin_panel(report["training_target_label_control"]),
        anchor_contrasts={key: thin_panel(value) for key, value in report["anchor_contrasts"].items()},
        control_seeds=seed_summaries(candidate, control), selected_fit_diagnostics=diagnostics(candidate["sessions"]),
        selected_fit_diagnostics_by_animal={animal: diagnostics(
            [r for r in candidate["sessions"] if r["animal"] == animal]) for animal in ANIMALS})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
