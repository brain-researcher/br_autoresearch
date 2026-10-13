#!/usr/bin/env python3
"""Extract concise evidence only from the complete fixed HW surrogate panel."""
import argparse
import json
import math
from pathlib import Path

from extract_primary_elapsed_only import _panel
from primary_capacity_matched_surrogate import RECIPE, REPLICATES
from primary_reference import LAMBDAS, SESSIONS
from source_adapter import compute_only


def brief_panel(item, *, sessions=False):
    validated = _panel(item)
    keep = ("primary_increment", "identity_increment", "primary_by_animal", "identity_by_animal",
            "primary_positive_session_counts", "identity_positive_session_counts", "lambda_counts",
            "within_animal_omission_ranges")
    result = {key: validated[key] for key in keep}
    if sessions:
        result["sessions"] = validated["sessions"]
    return result


def extract(path):
    report = json.loads(Path(path).read_text())
    if (report["status"] != "complete_capacity_matched_surrogate_panel" or report["recipe"] != RECIPE
            or report["promotion_authorized"] is not False or report["new_promotion_gate"] is not False
            or report["updates_primary_incumbent"] is not False or report["original_HW_refitted"] is not False
            or report["all_original_HW_neural_state_replay"] is not True
            or report["all_original_HW_saved_arrays_and_scoring_replay"] is not True
            or report["all_original_HW_weighted_covariance_replay"] is not True
            or report["original_HW_anchor"]["refitted"] is not False
            or report["scientific_trial_count_increment"] not in (0, 1)
            or report["literal_null_control_count_increment"] != report["scientific_trial_count_increment"]
            or report["alternative_explanation_count_increment"] != report["scientific_trial_count_increment"]
            or report["scientific_trial_count_increment"] != int(not report["structural_original_HW_duplicate"])):
        raise ValueError("Only a complete fixed surrogate panel with all original replays is permitted")
    seeds = report["original_HW_minus_each_surrogate"]
    if [seed["replicate"] for seed in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed seed comparisons required")
    capacity = report["shared_capacity_by_session"]
    if len(capacity) != 16 or {row["session_id"] for row in capacity} != set(SESSIONS):
        raise ValueError("All sixteen original shared-capacity records required")
    folds, widths = [], []
    for session in capacity:
        widths.append(session["history_column_count"])
        if ([row["held_direction"] for row in session["direction_fits"]] != list(range(8))
                or session["history_column_count"] != 4*session["physical_electrode_count"]):
            raise ValueError("Eight complete original history-capacity records required")
        for row in session["direction_fits"]:
            identity = row["capacity_identity"]
            if (identity["shared_Gram_eigendecompositions"] != 1
                    or identity["surrogate_specific_Gram_eigendecompositions"] != 0
                    or identity["original_HW_refitted"] is not False
                    or identity["feature_columns"] != session["history_column_count"]
                    or identity["fitting_design_rows"] != row["fitting_bin_rows"]
                    or [r["lambda"] for r in row["shared_ridge_df_by_lambda"]] != list(LAMBDAS)
                    or not all(math.isfinite(row[k]) for k in
                        ("shared_trace", "shared_numerical_rank", "shared_selected_effective_df"))
                    or not all(math.isfinite(r["effective_df"]) for r in row["shared_ridge_df_by_lambda"])):
                raise ValueError("Fixed common Gram and finite shared capacity diagnostics required")
            for key in ("surrogate_cross_product_equals_original_by_replicate",
                        "surrogate_design_equals_original_by_replicate", "whole_fitting_map_identity_by_replicate"):
                if len(identity[key]) != 8 or any(type(value) is not bool for value in identity[key]):
                    raise ValueError("All eight exact capacity identity flags required")
            folds.append(row)
    counts = {key: sum(sum(row["capacity_identity"][key]) for row in folds) for key in
              ("surrogate_cross_product_equals_original_by_replicate",
               "surrogate_design_equals_original_by_replicate", "whole_fitting_map_identity_by_replicate")}
    return {"status": report["status"], "recipe": RECIPE, "path": str(path),
        "job_id": report["job_id"], "elapsed_seconds": report["elapsed_seconds"],
        "maximum_RSS_KiB": report["maximum_RSS_KiB"],
        "accounting": {key: report[key] for key in ("scientific_trial_count_increment",
            "literal_null_control_count_increment", "alternative_explanation_count_increment",
            "structural_original_HW_duplicate", "updates_primary_incumbent", "promotion_authorized", "new_promotion_gate")},
        "exact_replay": {key: report[key] for key in ("all_original_HW_neural_state_replay",
            "all_original_HW_saved_arrays_and_scoring_replay", "all_original_HW_weighted_covariance_replay",
            "original_HW_refitted")},
        "original_HW": brief_panel(report["original_HW"]),
        "mean_surrogate": brief_panel(report["surrogate"], sessions=True),
        "HW_minus_mean_surrogate": brief_panel(report["original_HW_minus_mean_surrogate"], sessions=True),
        "HW_minus_each_seed": [{"replicate": row["replicate"], **brief_panel(row)} for row in seeds],
        "capacity": {"direction_folds": len(folds), "surrogate_direction_folds": len(folds)*8,
            "history_width_range": [min(widths), max(widths)],
            **{key+"_range": [min(row[key] for row in folds), max(row[key] for row in folds)] for key in
               ("shared_trace", "shared_numerical_rank", "shared_selected_effective_df")},
            "shared_Gram_eigendecompositions": sum(row["capacity_identity"]["shared_Gram_eigendecompositions"] for row in folds),
            "surrogate_specific_Gram_eigendecompositions": 0, "exact_identity_counts": counts,
            "interpretation": "Shared Gram matches conditional ridge capacity; retains direction/time structure; not a signal-free or raw-phase null"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps(extract(args.summary), indent=2, allow_nan=False))
