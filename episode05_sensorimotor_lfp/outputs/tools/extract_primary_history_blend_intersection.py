#!/usr/bin/env python3
"""Extract only complete I/R-blend evidence, selected supports and exact charges."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from extract_primary_capacity_matched_surrogate import brief_panel
from primary_history_blend import ALPHAS, SCORE_KEYS
from primary_history_blend_intersection import RECIPES, REPLICATES
from primary_reference import LAMBDAS, SESSIONS
from run_primary_history_blend_intersection import PANEL_STATUS, REPLAY_FLAGS
from source_adapter import compute_only


COMPLETE_REPLAY_FLAGS = (*REPLAY_FLAGS, "all_new_saved_W_scores_replayed",
                         "all_selected_mixture_scores_replayed", "all_exact_duplicate_flags_replayed")
REFERENCE_NAMES = ("I_minus_full_blend", "R_minus_full_blend", "I_minus_HW_blend_lambda_matched",
                   "R_minus_HW_blend_lambda_matched", "I_minus_original_I_HW", "R_minus_original_R_HW")
DRAW_REFERENCE_NAMES = ("blend_lambda_matched", "original_R_HW")


def compact_panel(panel):
    result = brief_panel(panel, sessions=True)
    by_id = {row["session_id"]: row for row in panel["sessions"]}
    for row in result["sessions"]:
        original = by_id[row["session_id"]]
        row.update({key: original[key] for key in ("alpha", "alpha_index", "cell_index", "reference_lambda") if key in original})
    for key in ("recipe", "reference_margin_screens", "all_reference_margin_screens_pass", "scientific_trial_count_increment"):
        if key in panel:
            result[key] = panel[key]
    result["alpha_counts"] = dict(Counter(str(row["alpha"]) for row in panel["sessions"]))
    result["joint_lambda_alpha_counts"] = dict(Counter(
        f"lambda={row['lambda']},alpha={row['alpha']}" for row in panel["sessions"]))
    if any("reference_lambda" in row for row in panel["sessions"]):
        result["reference_lambda_counts"] = dict(Counter(str(row["reference_lambda"]) for row in panel["sessions"]))
    return result


def compact_draws(panels):
    if [panel["replicate"] for panel in panels] != list(REPLICATES):
        raise ValueError("Every one of the eight ordered ordinary random-support draws required")
    return [{"replicate": panel["replicate"], **compact_panel(panel)} for panel in panels]


def extract(path):
    report = json.loads(Path(path).read_text())
    if (report["status"] != PANEL_STATUS or report["recipes"] != list(RECIPES)
            or report["recipe_order"] != list(RECIPES) or report["sensitivity_only"] is not True
            or any(report[key] is not False for key in ("incumbent_reset", "promotion_authorized", "original_HW_refitted",
                "full_W_refitted", "eligible_for_comparable_primary_incumbent", "provider_source_values_read"))
            or any(report[key] is not True for key in COMPLETE_REPLAY_FLAGS)
            or set(report["selected_prediction_paths"]) != set(SESSIONS)
            or report["literal_null_control_count_increment"] != 0):
        raise ValueError("Only complete nonpromotable support-blend evidence with every saved-array replay is permitted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES):
        raise ValueError("Both complete independently charged I/R-blend panels required")
    compact = [compact_panel(panels[recipe]) for recipe in RECIPES]
    rows = [{row["session_id"]: row for row in panels[recipe]["sessions"]} for recipe in RECIPES]
    diagnostics, reuse = [], []
    for session in SESSIONS:
        irow, rrow = (values[session] for values in rows)
        li, ai = irow["lambda_index"], irow["alpha_index"]
        if (type(li) is not int or li not in range(5) or type(ai) is not int or ai not in range(5)
                or irow["cell_index"] != 5*li+ai or irow["lambda"] != LAMBDAS[li] or irow["alpha"] != ALPHAS[ai]
                or any(rrow[key] != irow[key] for key in ("lambda_index", "alpha_index", "cell_index", "lambda", "alpha"))
                or irow["outer_training_sessions"] != [other for other in SESSIONS if other != session]
                or rrow["outer_training_sessions"] != irow["outer_training_sessions"]
                or len(irow["selected_supports"]) != 1 or len(rrow["selected_supports"]) != 8):
            raise ValueError("Original other-fifteen I selector and all eight controls at its same lambda/alpha required")
        supports = irow["selected_supports"]+rrow["selected_supports"]
        intersection_ids = set(supports[0]["support"]["electrode_ids"])
        if not intersection_ids:
            raise ValueError("Nonempty fixed physical guide intersection required")
        selected_supports = []
        for index, view in enumerate(supports):
            support, flags = view["support"], view["exact_reuse"]
            if (support["replicate"] != (None if index == 0 else index-1)
                    or len(set(support["electrode_ids"])) != len(intersection_ids)
                    or len(support["electrode_positions"]) != len(intersection_ids)
                    or any(view[key] != irow[key] for key in ("lambda_index", "alpha_index", "lambda", "alpha"))
                    or not all(math.isfinite(view[key]) for key in (*SCORE_KEYS, "W_candidate_SSE", "HW_candidate_SSE", "error_cross_product_over_D"))
                    or set(flags) != {"prior_I_HW", "full_blend", "prior_R_HW_draws", "current_I_blend"}
                    or any(type(flags[key]) is not bool for key in ("prior_I_HW", "full_blend", "current_I_blend"))
                    or len(flags["prior_R_HW_draws"]) != 8 or any(type(flag) is not bool for flag in flags["prior_R_HW_draws"])
                    or index == 0 and flags["current_I_blend"] is not True):
                raise ValueError("Every selected support, component error and exact per-draw identity is required")
            if set(view["selected_fit_diagnostics"]) != {"W", "HW"}:
                raise ValueError("Separate current-W and preserved-HW capacity records required")
            for name, folds in view["selected_fit_diagnostics"].items():
                if ([fold["held_direction"] for fold in folds] != list(range(8))
                        or any(not math.isfinite(fold[key]) for fold in folds for key in
                               ("fitting_bin_rows", "trace", "numerical_rank", "effective_df"))):
                    raise ValueError("Every selected W/HW direction fold needs finite own capacity diagnostics")
            if any(w["fitting_bin_rows"] != hw["fitting_bin_rows"] for w, hw in
                   zip(view["selected_fit_diagnostics"]["W"], view["selected_fit_diagnostics"]["HW"])):
                raise ValueError("W and HW must preserve the same original fitting rows")
            selected_supports.append({"replicate": support["replicate"],
                "support": {key: support[key] for key in ("recipe", "replicate", "electrode_ids", "electrode_positions", "full_support", "seed_literal") if key in support},
                "electrode_count": len(intersection_ids),
                "intersection_overlap_count": len(intersection_ids.intersection(support["electrode_ids"])),
                "lambda": view["lambda"], "alpha": view["alpha"],
                **{key: view[key] for key in (*SCORE_KEYS, "W_candidate_SSE", "HW_candidate_SSE", "error_cross_product_over_D")},
                "selected_fit_diagnostics": view["selected_fit_diagnostics"], "exact_reuse": flags})
        diagnostics.append({"session_id": session, "animal": irow["animal"], "lambda": irow["lambda"], "alpha": irow["alpha"],
                            "intersection": selected_supports[0], "random_supports": selected_supports[1:]})
        reuse.append((supports[0]["exact_reuse"], [view["exact_reuse"] for view in supports[1:]]))

    # These use replayed literal-array equality evidence, not equal mean scores
    # or an alpha endpoint heuristic. Each recipe's global charge is separate.
    i_checks = {
        "original_I_HW": all(i["prior_I_HW"] for i, _ in reuse),
        "full_blend": all(i["full_blend"] for i, _ in reuse),
        "original_R_HW": all(all(i["prior_R_HW_draws"]) for i, _ in reuse),
    }
    r_checks = {
        "original_R_HW": all(all(view["prior_R_HW_draws"][r] for r, view in enumerate(random)) for _, random in reuse),
        RECIPES[0]: all(all(view["current_I_blend"] for view in random) for _, random in reuse),
        "original_I_HW": all(all(view["prior_I_HW"] for view in random) for _, random in reuse),
        "full_blend": all(all(view["full_blend"] for view in random) for _, random in reuse),
    }
    duplicates = {recipe: [name for name, same in checks.items() if same]
                  for recipe, checks in zip(RECIPES, (i_checks, r_checks))}
    charges = {recipe: int(not duplicates[recipe]) for recipe in RECIPES}
    count = sum(charges.values())
    prior = report["prior_valid_comparisons"]
    patience = report["prior_patience"]
    transitions = []
    for recipe in RECIPES:
        prior += charges[recipe]
        patience += charges[recipe]
        transitions.append({"recipe": recipe, "charge": charges[recipe], "completed_count": prior,
                            "patience": patience, "incumbent_reset": False})
    if (type(report["scientific_trial_count_increment"]) is not int
            or any(type(report["scientific_trial_count_by_recipe"][recipe]) is not int for recipe in RECIPES)
            or report["structural_duplicate_of"] != duplicates or report["scientific_trial_count_by_recipe"] != charges
            or any(report[key] != count for key in ("scientific_trial_count_increment", "stress_diagnostic_count_increment", "alternative_explanation_count_increment"))
            or any(panels[recipe]["scientific_trial_count_increment"] != charges[recipe] for recipe in RECIPES)
            or report["total_valid_primary_trials_including_prior"] != prior or prior > 64
            or report["patience_after"] != patience or report["count_transitions"] != transitions
            or report["post_initial_comparisons"] != prior-6
            or report["post_initial_alternative_checks"] != report["prior_alternative_checks"]+count
            or report["post_initial_null_controls"] != report["prior_literal_null_controls"]):
        raise ValueError("Independent literal-array duplicates, ordered stress charges and live counters must agree")
    if set(report["reference_contrasts"]) != set(REFERENCE_NAMES) or set(report["random_support_HW_contrasts"]) != set(DRAW_REFERENCE_NAMES):
        raise ValueError("All full-blend, own-HW and original-HW comparisons required")
    draw_effects = compact_draws(report["intersection_minus_each_random_support"])
    exceptions = [{"replicate": panel["replicate"], "metrics": {metric: {animal: {
        "animal_mean": panel[animal_key][animal],
        "nonpositive_sessions": [row["session_id"] for row in panel["sessions"] if row["animal"] == animal and row[metric] <= 0],
        "nonpositive_omissions": panel["within_animal_omission_ranges"][animal][key]["nonpositive_omissions"],
    } for animal in ("Mihili", "Chewie")} for metric, key, animal_key in
        (("delta_R2", "primary_increment", "primary_by_animal"), ("identity_increment", "identity_increment", "identity_by_animal"))}}
        for panel in draw_effects]
    capacity = {}
    for arm in ("intersection", "random_supports"):
        views = [view for session in diagnostics for view in
                 ([session[arm]] if arm == "intersection" else session[arm])]
        capacity[arm] = {}
        for name in ("W", "HW"):
            folds = [fold for view in views for fold in view["selected_fit_diagnostics"][name]]
            capacity[arm][name] = {"direction_folds": len(folds), **{
                key+"_range": [min(fold[key] for fold in folds), max(fold[key] for fold in folds)]
                for key in ("trace", "numerical_rank", "effective_df")}}
    accounting = ("recipe_order", "scientific_trial_count_by_recipe", "structural_duplicate_of", "scientific_trial_count_increment",
        "stress_diagnostic_count_increment", "literal_null_control_count_increment", "alternative_explanation_count_increment",
        "prior_valid_comparisons", "prior_patience", "prior_alternative_checks", "prior_literal_null_controls",
        "total_valid_primary_trials_including_prior", "post_initial_comparisons", "post_initial_alternative_checks", "post_initial_null_controls",
        "patience_after", "count_transitions", "incumbent_reset", "eligible_for_comparable_primary_incumbent", "sensitivity_only",
        "original_HW_refitted", "full_W_refitted", "provider_source_values_read", "promotion_authorized", "search_complete")
    return {"source_summary": str(path), "status": report["status"], "recipes": list(RECIPES),
        **{key: report[key] for key in ("job_id", "elapsed_seconds", "maximum_RSS_KiB")},
        "accounting": {key: report[key] for key in accounting},
        "exact_replay": {key: report[key] for key in COMPLETE_REPLAY_FLAGS}, "panels": compact,
        "intersection_minus_random_support": compact_panel(report["intersection_minus_random_support"]),
        "random_support_panels": compact_draws(report["random_support_panels"]),
        "intersection_minus_each_random_support": draw_effects, "intersection_minus_draw_exceptions": exceptions,
        "reference_contrasts": {name: compact_panel(report["reference_contrasts"][name]) for name in REFERENCE_NAMES},
        "random_support_HW_contrasts": {name: compact_draws(report["random_support_HW_contrasts"][name]) for name in DRAW_REFERENCE_NAMES},
        "selected_support_diagnostics": diagnostics, "selected_component_capacity": capacity,
        "selected_prediction_paths": report["selected_prediction_paths"],
        **{key: report[key] for key in ("metadata_path", "original_intersection_summary", "original_full_blend_summary")},
        "interpretation": "Mean eight support-specific correct/wrong SSEs, not predictions; finite draw spread is not a confidence interval. Cross-products use full native-XY component errors and are descriptive, not a separate neural increment. Equal support count or lambda does not match W/HW covariance or df. Guide membership does not identify spike contamination. Summary itself reads no provider values; session-stage source reads were authorized for reduced W."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps(extract(args.summary), indent=2, allow_nan=False))
