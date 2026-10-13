#!/usr/bin/env python3
"""Compact evidence from a complete HW panel; read its summary only, with no fits."""
import argparse
import json
from pathlib import Path

from extract_primary_relevance import ANIMALS, bounds, seed_summaries, thin_panel
from extract_primary_repeatability_weight import diagnostics
from primary_history_repeatability_weight import LAGS, RECIPES
from primary_reference import SESSIONS
from source_adapter import compute_only


def history_diagnostics(rows):
    folds = [fold for row in rows for fold in row["selected_weight_diagnostics"]]
    result = diagnostics(rows)
    physical = [fold["paired"]["history_energy_by_electrode"] for fold in folds]
    by_lag = [fold["paired"]["history_energy_by_lag_electrode"] for fold in folds]
    if any(len(lag_rows) != 4 or any(len(values) != len(q) for values in lag_rows)
           for q, lag_rows in zip(physical, by_lag)):
        raise ValueError("Complete four-lag energy for every original physical electrode required")
    result["history_support"] = {
        "lag_offsets": list(LAGS),
        "physical_history_energy_range": bounds(value for q in physical for value in q),
        "physical_history_energy_zero_count": sum(value == 0 for q in physical for value in q),
        "physical_history_energy_positive_count": sum(value > 0 for q in physical for value in q),
        "energy_by_lag": [{"lag": lag,
            "energy_range": bounds(value for matrix in by_lag for value in matrix[index]),
            "zero_count": sum(value == 0 for matrix in by_lag for value in matrix[index]),
            "positive_count": sum(value > 0 for matrix in by_lag for value in matrix[index])}
            for index, lag in enumerate(LAGS)],
        "current_prior_T_zero_but_history_active_count": sum(
            t == 0 and q > 0 for fold in folds
            for t, q in zip(fold["repeatability"]["T"], fold["paired"]["history_energy_by_electrode"])),
        "lag0_decoder_energy_zero_but_history_active_count": sum(
            current == 0 and q > 0 for matrix, physical_q in zip(by_lag, physical)
            for current, q in zip(matrix[0], physical_q)),
        "all_past_lags_zero_fold_count": sum(fold["all_past_lags_zero"] for fold in folds),
        "all_past_lags_zero_session_ids": [row["session_id"] for row in rows
            if all(fold["all_past_lags_zero"] for fold in row["selected_weight_diagnostics"])],
        "interpretation": "Repeated fitting fold/electrode diagnostics; total history energy governs structural active support",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    report = json.loads(args.summary.read_text())
    if report.get("status") != "complete_history_repeatability_weight_control_panel":
        raise ValueError("Only a complete sixteen-session HW panel may be extracted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if (len(report["panels"]) != 2 or set(panels) != set(RECIPES)
            or any(len(panel["sessions"]) != 16
                   or {row["session_id"] for row in panel["sessions"]} != set(SESSIONS)
                   for panel in panels.values())):
        raise ValueError("Both panels must contain every original session exactly once")
    expected = {"history_independently_selected", "history_HW_lambda_matched",
                "repeatability_independently_selected", "repeatability_HW_lambda_matched"}
    if set(report["anchor_contrasts"]) != expected:
        raise ValueError("Both H and W independent and candidate-matched penalty contrasts required")
    for contrast in [report["weight_label_control"], *report["anchor_contrasts"].values()]:
        if len(contrast["sessions"]) != 16 or {row["session_id"] for row in contrast["sessions"]} != set(SESSIONS):
            raise ValueError("All original sessions required in every contrast")
    candidate, control = (panels[recipe] for recipe in RECIPES)
    keys = ("status", "scientific_trial_count_increment", "control_trial_count_increment",
            "scientific_trial_count_by_recipe", "total_valid_primary_trials_including_prior",
            "post_initial_comparisons", "post_initial_controls", "structural_identity_by_recipe",
            "structural_current_only_duplicate", "all_controls_equal_candidate",
            "weighting_screens", "all_weighting_screens_pass", "reference_anchors")
    result = {"source_summary": str(args.summary), **{key: report[key] for key in keys}}
    result.update(
        panels=[thin_panel(candidate), thin_panel(control)],
        candidate_minus_mean_control=thin_panel(report["weight_label_control"]),
        anchor_contrasts={key: thin_panel(value) for key, value in report["anchor_contrasts"].items()},
        control_seeds=seed_summaries(candidate, control),
        selected_weight_diagnostics=history_diagnostics(candidate["sessions"]),
        selected_weight_diagnostics_by_animal={animal: history_diagnostics(
            [row for row in candidate["sessions"] if row["animal"] == animal]) for animal in ANIMALS},
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
