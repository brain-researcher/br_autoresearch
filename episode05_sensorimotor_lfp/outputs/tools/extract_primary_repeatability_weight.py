#!/usr/bin/env python3
"""Compact evidence from a complete repeatability-weight panel; no refitting."""
import argparse
from collections import Counter
import json
from pathlib import Path

from extract_primary_relevance import ANIMALS, bounds, seed_summaries, thin_panel
from primary_reference import SESSIONS
from primary_repeatability_weight import RECIPES, REPLICATES
from source_adapter import compute_only


def optional_bounds(values):
    values = list(values)
    return bounds(values) if values else None


def arm_diagnostics(arms):
    return {
        "arm_count": len(arms),
        **{key+"_range": bounds(arm[key] for arm in arms)
           for key in ("trace", "original_trace", "numerical_rank", "effective_df")},
        "base_weight_range": bounds(value for arm in arms for value in arm["base_weights"]),
        "applied_weight_range": bounds(value for arm in arms for value in arm["weights"]),
        "applied_positive_energy_weight_range": optional_bounds(
            arm["weights"][column] for arm in arms for column in arm["positive_energy_columns"]),
        "formula_weight_range": bounds(value for arm in arms for value in arm["formula_weights"]),
        "maximum_absolute_trace_difference": max(abs(arm["trace"]-arm["original_trace"]) for arm in arms),
        "zero_trace_count": sum(arm["trace"] == 0 for arm in arms),
        "structural_identity_count": sum(arm["structural_identity"] for arm in arms),
        "all_structural_identity": all(arm["structural_identity"] for arm in arms),
    }


def diagnostics(rows):
    folds = [fold for row in rows for fold in row["selected_weight_diagnostics"]]
    if len(folds) != 8*len(rows) or any(len(fold["controls"]) != 8 for fold in folds):
        raise ValueError("Eight original direction fits and eight controls per fit required")
    paired = [fold["paired"] for fold in folds]
    controls = [arm for fold in folds for arm in fold["controls"]]
    statistics = [fold["repeatability"] for fold in folds]
    totals = [value for row in statistics for value in row["T"]]
    products = [value for row in statistics for value in row["P"]]
    ratios = [value for row in statistics for value in row["r"]]
    raw = [value for row in statistics for value in row["raw_ratio"]]
    categories = Counter(value for row in statistics for value in row["clip_category"])
    trace_ratios = [control["trace"]/fold["paired"]["trace"] for fold in folds
                    if fold["paired"]["trace"] > 0 for control in fold["controls"]]
    df_ratios = [control["effective_df"]/fold["paired"]["effective_df"] for fold in folds
                 if fold["paired"]["effective_df"] > 0 for control in fold["controls"]]
    return {
        "fold_count": len(folds), "fold_electrode_count": len(totals),
        "description": "Repeated fold/electrode diagnostics; no new independent animals or model-selection rules",
        "repeatability": {
            "T_range": bounds(totals), "P_range": bounds(products), "r_range": bounds(ratios),
            "a_range": bounds(value for row in statistics for value in row["a"]),
            "raw_ratio_range": optional_bounds(value for value in raw if value is not None),
            "undefined_raw_ratio_count": sum(value is None for value in raw),
            "clip_category_counts": dict(sorted(categories.items())),
            "T_zero_count": sum(value == 0 for value in totals),
            "T_positive_count": sum(value > 0 for value in totals),
            "P_negative_count": sum(value < 0 for value in products),
            "P_zero_count": sum(value == 0 for value in products),
            "P_positive_count": sum(value > 0 for value in products),
            "r_zero_count": sum(value == 0 for value in ratios),
            "r_one_count": sum(value == 1 for value in ratios),
            "r_between_zero_and_one_count": sum(0 < value < 1 for value in ratios),
        },
        "paired": arm_diagnostics(paired), "controls": arm_diagnostics(controls),
        "control_trace_over_paired_range": optional_bounds(trace_ratios),
        "control_df_over_paired_range": optional_bounds(df_ratios),
        "control_structural_identity_by_replicate": [
            {"replicate": replicate,
             "identity_fold_count": sum(fold["controls"][replicate]["structural_identity"] for fold in folds),
             "all_folds_identity": all(fold["controls"][replicate]["structural_identity"] for fold in folds),
             "equal_candidate_fold_count": sum(fold["controls_equal_candidate_by_replicate"][replicate] for fold in folds),
             "all_folds_equal_candidate": all(fold["controls_equal_candidate_by_replicate"][replicate] for fold in folds)}
            for replicate in REPLICATES],
        "candidate_identity_session_ids": [row["session_id"] for row in rows
            if all(fold["paired"]["structural_identity"] for fold in row["selected_weight_diagnostics"])],
        "all_controls_identity_session_ids": [row["session_id"] for row in rows
            if all(arm["structural_identity"] for fold in row["selected_weight_diagnostics"] for arm in fold["controls"])],
        "all_controls_equal_candidate_session_ids": [row["session_id"] for row in rows
            if all(all(fold["controls_equal_candidate_by_replicate"]) for fold in row["selected_weight_diagnostics"])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    report = json.loads(args.summary.read_text())
    if report.get("status") != "complete_repeatability_weight_control_panel":
        raise ValueError("Only a complete sixteen-session repeatability-weight panel may be extracted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if (len(report["panels"]) != 2 or set(panels) != set(RECIPES)
            or any(len(panel["sessions"]) != 16
                   or {row["session_id"] for row in panel["sessions"]} != set(SESSIONS)
                   for panel in panels.values())):
        raise ValueError("Both panels must contain every original session exactly once")
    candidate, control = (panels[recipe] for recipe in RECIPES)
    keys = ("status", "scientific_trial_count_increment", "control_trial_count_increment",
            "scientific_trial_count_by_recipe", "total_valid_primary_trials_including_prior",
            "post_initial_comparisons", "post_initial_controls", "structural_identity_by_recipe",
            "all_controls_equal_candidate", "weighting_screens", "all_weighting_screens_pass")
    result = {"source_summary": str(args.summary), **{key: report[key] for key in keys}}
    result.update(
        panels=[thin_panel(candidate), thin_panel(control)],
        candidate_minus_mean_control=thin_panel(report["weight_label_control"]),
        anchor_contrasts={key: thin_panel(value) for key, value in report["anchor_contrasts"].items()},
        control_seeds=seed_summaries(candidate, control),
        selected_weight_diagnostics=diagnostics(candidate["sessions"]),
        selected_weight_diagnostics_by_animal={animal: diagnostics(
            [row for row in candidate["sessions"] if row["animal"] == animal]) for animal in ANIMALS},
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
