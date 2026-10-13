#!/usr/bin/env python3
"""Extract only complete original-support H2W candidate/control evidence."""
import argparse
import json
import math
from pathlib import Path

from extract_primary_capacity_matched_surrogate import brief_panel
from primary_history2_repeatability_weight import RECIPES, REPLICATES
from primary_reference import SESSIONS
from source_adapter import compute_only


def extract(path):
    report = json.loads(Path(path).read_text())
    counts, duplicates = report["scientific_trial_count_by_recipe"], report["structural_duplicate_of"]
    if (report["status"] != "complete_history2_repeatability_panel" or report["recipes"] != list(RECIPES)
            or report["recipe_order"] != list(RECIPES) or set(counts) != set(RECIPES) or set(duplicates) != set(RECIPES)
            or any(type(counts[r]) is not int or counts[r] != int(not duplicates[r]) for r in RECIPES)
            or report["scientific_trial_count_increment"] != sum(counts.values())
            or report["literal_null_control_count_increment"] != counts[RECIPES[1]]
            or report["alternative_explanation_count_increment"] != counts[RECIPES[1]]
            or report["eligible_for_comparable_primary_incumbent"] != bool(counts[RECIPES[0]])
            or any(report[key] is not True for key in ("all_original_replays", "all_new_saved_scores_replayed", "original_selectors_independently_replayed"))
            or any(report[key] is not False for key in ("original_HW_refitted", "original_W_refitted", "new_promotion_gate", "promotion_authorized"))):
        raise ValueError("Only complete replayed H2W pair with exact independent duplicate charges permitted")
    seeds = report["candidate_minus_each_surrogate"]
    if [seed["replicate"] for seed in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed surrogate comparisons required")
    capacity = report["shared_capacity_by_session"]
    if len(capacity) != 16 or {row["session_id"] for row in capacity} != set(SESSIONS):
        raise ValueError("Complete original sixteen-session capacity evidence required")
    folds, widths = [], []
    for session in capacity:
        width = session["history_column_count"]
        widths.append(width)
        if width != 2*session["physical_electrode_count"] or [row["held_direction"] for row in session["direction_fits"]] != list(range(8)):
            raise ValueError("Exactly two lag blocks in every original direction fit required")
        for row in session["direction_fits"]:
            identity = row["capacity_identity"]
            if (identity["shared_Gram_eigendecompositions"] != 1 or identity["surrogate_specific_Gram_eigendecompositions"] != 0
                    or identity["feature_columns"] != width or identity["fitting_design_rows"] != row["fitting_bin_rows"]
                    or not all(math.isfinite(row[key]) for key in ("shared_trace", "shared_numerical_rank", "shared_selected_effective_df"))):
                raise ValueError("Shared finite candidate/control capacity required")
            folds.append(row)
    accounting_keys = ("recipe_order", "scientific_trial_count_increment", "scientific_trial_count_by_recipe",
        "literal_null_control_count_increment", "alternative_explanation_count_increment", "structural_duplicate_of",
        "eligible_for_comparable_primary_incumbent", "reference_margin_screens", "all_reference_margin_screens_pass",
        "blend_reference_available", "original_HW_refitted", "original_W_refitted", "new_promotion_gate", "promotion_authorized")
    return {"status": report["status"], "recipes": report["recipes"], "path": str(path),
        **{key: report[key] for key in ("job_id", "elapsed_seconds", "maximum_RSS_KiB")},
        "accounting": {key: report[key] for key in accounting_keys},
        "exact_replay": {key: report[key] for key in ("all_original_replays", "all_new_saved_scores_replayed", "original_selectors_independently_replayed")},
        "candidate": brief_panel(report["candidate"], sessions=True),
        "mean_surrogate": brief_panel(report["mean_surrogate"], sessions=True),
        "candidate_minus_mean_surrogate": brief_panel(report["candidate_minus_mean_surrogate"], sessions=True),
        "candidate_minus_each_surrogate": [{"replicate": row["replicate"], **brief_panel(row)} for row in seeds],
        "reference_contrasts": {name: brief_panel(panel, sessions=True) for name, panel in report["reference_contrasts"].items()},
        "original_components": {name: brief_panel(panel) for name, panel in report["original_components"].items()},
        "capacity": {"direction_folds": len(folds), "surrogate_direction_folds": len(folds)*8,
            "history_width_range": [min(widths), max(widths)], "shared_Gram_eigendecompositions": len(folds),
            "surrogate_specific_Gram_eigendecompositions": 0,
            **{key+"_range": [min(row[key] for row in folds), max(row[key] for row in folds)]
                for key in ("shared_trace", "shared_numerical_rank", "shared_selected_effective_df")},
            "exact_identity_counts": {key: sum(sum(row["capacity_identity"][key]) for row in folds)
                for key in ("surrogate_cross_product_equals_original_by_replicate", "surrogate_design_equals_original_by_replicate", "whole_fitting_map_identity_by_replicate")},
            "zero_energy_flags": {key: sum(row[key] for row in folds) for key in
                ("retained_past_fitting_energy_exactly_zero", "discarded_past_fitting_energy_exactly_zero")},
            "zero_energy_alone_proves_duplicate": False,
            "interpretation": "Exact conditional capacity matching within H2W only; W/HW trace or penalty matching does not match df"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps(extract(args.summary), indent=2, allow_nan=False))
