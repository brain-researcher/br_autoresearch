#!/usr/bin/env python3
"""Summarize only a complete C/T/CT report; no source, partial banks or fits."""
import argparse
import json
from pathlib import Path
from statistics import mean

from extract_primary_relevance import ANIMALS, METRICS, animal_values, bounds, thin_panel
from primary_matched_budget import RECIPES, VIEWS
from primary_reference import SESSIONS
from source_adapter import compute_only


def require_sessions(rows):
    if (len(rows) != 16 or {row["session_id"] for row in rows} != set(SESSIONS)
            or any(row["animal"] != ("Mihili" if row["session_id"].startswith("Mihili_") else "Chewie") for row in rows)):
        raise ValueError("Exactly the sixteen original sessions and animal identities required")


def compact_panel(panel):
    result = thin_panel(panel)
    result["identity_positive_session_counts"] = {
        animal: sum(row["identity_increment"] > 0 for row in panel["sessions"] if row["animal"] == animal)
        for animal in ANIMALS}
    for key in ("primary_formula", "role", "sensitivity_only", "scientific_trial_count_increment", "promotion_authorized"):
        if key in panel:
            result[key] = panel[key]
    return result


def view_summaries(rows, key):
    if any([view["view"] for view in row[key]] != list(VIEWS) for row in rows):
        raise ValueError("Every session needs all eight ordered fixed views")
    summaries = []
    for view in VIEWS:
        scores = [{"session_id": row["session_id"], "animal": row["animal"],
                   **{metric: row[key][view][metric] for metric in METRICS}} for row in rows]
        item = {"view": view}
        for metric in METRICS:
            animals = animal_values(scores, metric)
            item[metric] = {"overall": mean(animals.values()), "animals": animals,
                "positive_session_counts": {animal: sum(row[metric] > 0 for row in scores if row["animal"] == animal)
                                            for animal in ANIMALS}}
        summaries.append(item)
    ranges = {metric: {"overall_range": bounds(row[metric]["overall"] for row in summaries),
        "animal_ranges": {animal: bounds(row[metric]["animals"][animal] for row in summaries) for animal in ANIMALS},
        "all_views_both_animals_positive": all(row[metric]["animals"][animal] > 0 for row in summaries for animal in ANIMALS)}
        for metric in METRICS}
    return {"views": summaries, "ranges": ranges,
        "interpretation": "Finite fixed subset sensitivity, not confidence intervals or independent animal replication; each condition's selected lambda is shared across its views"}


def diagnostics(rows):
    views = [view for row in rows for view in row["selected_view_scores"]]
    folds = [fold for view in views for fold in view["selected_fit_diagnostics"]]
    if (len(views) != 8*len(rows) or len(folds) != 8*len(views)
            or any([fold["held_direction"] for fold in view["selected_fit_diagnostics"]] != list(range(8)) for view in views)):
        raise ValueError("Every selected support view needs all eight held-direction fits")
    return {"session_view_count": len(views), "repeated_view_fold_count": len(folds),
        **{key+"_range": bounds(view[key] for view in views)
           for key in ("trial_count", "electrode_count", "denominator", "baseline_R2", "baseline_change", "full_HW_original_baseline_R2")},
        "history_column_count_range": bounds(4*view["electrode_count"] for view in views),
        **{key+"_range": bounds(fold[key] for fold in folds)
           for key in ("fitting_bin_rows", "trace", "numerical_rank", "effective_df")},
        "full_HW_reference_view_reuses": sum(view["fit_key"] == "full_HW" for view in views),
        "unique_new_fit_keys_within_condition": sum(len({view["fit_key"] for view in row["selected_view_scores"]
            if view["fit_key"] != "full_HW"}) for row in rows),
        "interpretation": "Descriptive support and selected-penalty capacity; repeated views/folds can reuse exactly the same fit and are not new observations"}


def global_reuse(panels):
    rows = {recipe: {row["session_id"]: row for row in panels[recipe]["sessions"]} for recipe in RECIPES}
    result = []
    for session in SESSIONS:
        views = [view for recipe in RECIPES for view in rows[recipe][session]["selected_view_scores"]]
        full = sum(view["fit_key"] == "full_HW" for view in views)
        new = len({view["fit_key"] for view in views if view["fit_key"] != "full_HW"})
        result.append({"session_id": session, "total_condition_views": len(views),
            "original_trial_count": rows[RECIPES[0]][session]["selected_view_scores"][0]["trial_count"],
            "original_electrode_count": rows[RECIPES[1]][session]["selected_view_scores"][0]["electrode_count"],
            "full_HW_view_reuses": full, "unique_new_fits": new,
            "within_attempt_duplicate_view_reuses": len(views)-full-new})
    return {"sessions": result,
        **{key: sum(row[key] for row in result) for key in ("total_condition_views", "full_HW_view_reuses", "unique_new_fits", "within_attempt_duplicate_view_reuses")},
        "interpretation": "Fit keys are session-local; exact shared supports are counted once across all three conditions"}


def baseline_summary(rows):
    require_sessions(rows)
    key = "mean_normalized_baseline_SSE_change"
    animals = {animal: mean(row[key] for row in rows if row["animal"] == animal) for animal in ANIMALS}
    return {"overall": mean(animals.values()), "animals": animals,
        "session_range": bounds(row[key] for row in rows), "sessions": rows,
        "formula": "mean_view((SSE_Breduced-SSE_Bfull)/D_retained)",
        "interpretation": "Positive means the reduced fitting support worsened the behavioral baseline. Own-baseline delta differences equal direct prediction gain plus this contribution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    report = json.loads(args.summary.read_text())
    if (report.get("status") != "complete_matched_budget_sensitivity_panel"
            or report.get("sensitivity_only") is not True or report.get("promotion_authorized") is not False
            or report.get("incumbent_reset") is not False or report["recipe_order"] != list(RECIPES)):
        raise ValueError("Only the complete nonpromotable ordered C/T/CT sensitivity may be extracted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if (len(report["panels"]) != 3 or set(panels) != set(RECIPES)
            or set(report["reference_contrasts"]) != set(RECIPES)
            or set(report["baseline_changes"]) != set(RECIPES)):
        raise ValueError("All three diagnostic panels, baseline changes and reference contrasts required")
    count = report["scientific_trial_count_increment"]
    if (type(count) is not int or count not in (0, 1, 2, 3)
            or report["stress_diagnostic_count_increment"] != count or report["null_control_trial_count_increment"] != 0
            or report["post_initial_null_controls"] != 11 or report["post_initial_alternative_checks"] != 11+count
            or report["post_initial_comparisons"] != 21+count
            or report["total_valid_primary_trials_including_prior"] != 27+count):
        raise ValueError("Stress diagnostics and literal null-control accounting must remain distinct")
    for recipe in RECIPES:
        require_sessions(panels[recipe]["sessions"])
        if set(report["reference_contrasts"][recipe]) != {"independently_selected", "diagnostic_lambda_matched"}:
            raise ValueError("Both original and diagnostic-matched full-HW reference selections required")
        for contrast in report["reference_contrasts"][recipe].values():
            require_sessions(contrast["sessions"])
    keys = ("status", "scientific_trial_count_increment", "stress_diagnostic_count_increment",
        "null_control_trial_count_increment", "scientific_trial_count_by_recipe", "structural_duplicate_of",
        "total_valid_primary_trials_including_prior", "post_initial_comparisons", "post_initial_null_controls",
        "post_initial_alternative_checks", "recipe_order", "prospective_patience_transitions", "sensitivity_only",
        "incumbent_reset", "promotion_authorized", "search_complete", "original_HW_summary_path", "reference_refitted")
    result = {"source_summary": str(args.summary), **{key: report[key] for key in keys},
        "panels": [compact_panel(panels[recipe]) for recipe in RECIPES],
        "direct_prediction_reference_contrasts": {recipe: {name: compact_panel(contrast)
            for name, contrast in report["reference_contrasts"][recipe].items()} for recipe in RECIPES},
        "baseline_changes": {recipe: baseline_summary(report["baseline_changes"][recipe]) for recipe in RECIPES},
        "view_sensitivity": {recipe: view_summaries(panels[recipe]["sessions"], "selected_view_scores") for recipe in RECIPES},
        "reference_contrast_view_sensitivity": {recipe: {name: view_summaries(contrast["sessions"], "view_contrasts")
            for name, contrast in report["reference_contrasts"][recipe].items()} for recipe in RECIPES},
        "selected_support_and_capacity": {recipe: diagnostics(panels[recipe]["sessions"]) for recipe in RECIPES},
        "selected_support_and_capacity_by_animal": {recipe: {animal: diagnostics(
            [row for row in panels[recipe]["sessions"] if row["animal"] == animal]) for animal in ANIMALS} for recipe in RECIPES},
        "exact_fit_reuse": global_reuse(panels)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
