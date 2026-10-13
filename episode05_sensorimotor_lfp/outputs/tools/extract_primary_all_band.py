#!/usr/bin/env python3
"""Compact evidence from a complete all-band panel; no source reads or refits."""
import argparse
import json
from pathlib import Path

from extract_primary_relevance import ANIMALS, bounds, seed_summaries, thin_panel
from primary_all_band import RECIPES, REPLICATES
from primary_reference import SESSIONS
from source_adapter import compute_only


def optional_bounds(values):
    values = list(values)
    return bounds(values) if values else None


def diagnostics(rows):
    folds = [fold for row in rows for fold in row["selected_fit_diagnostics"]]
    if (len(folds) != 8*len(rows)
            or any([fold["held_direction"] for fold in row["selected_fit_diagnostics"]] != list(range(8)) for row in rows)
            or any(len(fold["controls"]) != 8 or len(fold["controls_equal_candidate_by_replicate"]) != 8 for fold in folds)):
        raise ValueError("Eight ordered held directions and eight controls per fit required")
    paired = [fold["paired"] for fold in folds]
    controls = [arm for fold in folds for arm in fold["controls"]]
    result = {
        "fold_count": len(folds), "control_arm_count": len(controls),
        "feature_width_rule": "9E: retained LMP E plus high 2E plus added six-band 6E",
        "added_fitting_energy_range": bounds(fold["added_fitting_energy"] for fold in folds),
        "zero_added_energy_fold_count": sum(fold["added_fitting_energy"] == 0 for fold in folds),
        "structural_J_duplicate_fold_count": sum(fold["structural_J_duplicate"] for fold in folds),
        "structural_J_duplicate_session_ids": [row["session_id"] for row in rows
            if all(fold["structural_J_duplicate"] for fold in row["selected_fit_diagnostics"])],
        "all_controls_equal_candidate_session_ids": [row["session_id"] for row in rows
            if all(all(fold["controls_equal_candidate_by_replicate"]) for fold in row["selected_fit_diagnostics"])],
        "exact_control_design_equality_by_replicate": [{"replicate": replicate,
            "equal_candidate_fold_count": sum(fold["controls_equal_candidate_by_replicate"][replicate] for fold in folds),
            "all_folds_equal_candidate": all(fold["controls_equal_candidate_by_replicate"][replicate] for fold in folds)}
            for replicate in REPLICATES],
        "interpretation": "Own full Grams include changed retained/added cross terms; matching width and penalty does not match spectra or effective df. Repeated folds are not independent animals.",
    }
    for name, arms in (("paired", paired), ("controls", controls)):
        result[name] = {"arm_count": len(arms),
            **{key+"_range": bounds(arm[key] for arm in arms) for key in ("trace", "numerical_rank", "effective_df")},
            "zero_trace_count": sum(arm["trace"] == 0 for arm in arms),
            "zero_df_count": sum(arm["effective_df"] == 0 for arm in arms)}
    for key in ("trace", "effective_df"):
        result["control_"+key+"_over_paired_range"] = optional_bounds(
            arm[key]/fold["paired"][key] for fold in folds
            if fold["paired"][key] > 0 for arm in fold["controls"])
        result["control_minus_paired_"+key+"_range"] = bounds(
            arm[key]-fold["paired"][key] for fold in folds for arm in fold["controls"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    report = json.loads(args.summary.read_text())
    if report.get("status") != "complete_all_band_control_panel" or report.get("promotion_authorized") is not False:
        raise ValueError("Only a complete sixteen-session all-band development panel may be extracted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    expected = {"joint_independently_selected", "joint_all_band_lambda_matched",
                "lmp_independently_selected", "lmp_all_band_lambda_matched"}
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES) or set(report["anchor_contrasts"]) != expected:
        raise ValueError("Both all-band recipes and all four J/LMP anchor contrasts required")
    for panel in [*panels.values(), report["training_added_label_control"], *report["anchor_contrasts"].values()]:
        if len(panel["sessions"]) != 16 or {row["session_id"] for row in panel["sessions"]} != set(SESSIONS):
            raise ValueError("Every original session exactly once required in each panel and contrast")
    candidate, control = (panels[recipe] for recipe in RECIPES)
    keys = ("status", "scientific_trial_count_increment", "control_trial_count_increment",
            "scientific_trial_count_by_recipe", "total_valid_primary_trials_including_prior",
            "post_initial_comparisons", "post_initial_controls", "structural_J_duplicate",
            "all_controls_equal_candidate", "complementarity_screens", "all_complementarity_screens_pass",
            "reference_anchors", "promotion_authorized", "search_complete")
    result = {"source_summary": str(args.summary), **{key: report[key] for key in keys}}
    result.update(panels=[thin_panel(candidate), thin_panel(control)],
        candidate_minus_mean_control=thin_panel(report["training_added_label_control"]),
        anchor_contrasts={key: thin_panel(value) for key, value in report["anchor_contrasts"].items()},
        control_seeds=seed_summaries(candidate, control),
        selected_fit_diagnostics=diagnostics(candidate["sessions"]),
        selected_fit_diagnostics_by_animal={animal: diagnostics(
            [row for row in candidate["sessions"] if row["animal"] == animal]) for animal in ANIMALS})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
