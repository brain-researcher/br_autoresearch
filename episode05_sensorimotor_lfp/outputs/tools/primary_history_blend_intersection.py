"""Actual constant W/HW blend on the frozen intersection and eight ordinary supports."""
import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_elapsed_only import full_model_contrast, panel
from primary_history_blend import (ALPHAS, COMPONENTS, SCORE_KEYS, blend_prediction, prediction_scores,
    score_session, select_cell, validate_bank as validate_blend_bank, validate_original_selections)
from primary_intersection import RECIPES as OLD_RECIPES, average_banks
from primary_pca import prepare_split, projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_repeatability_weight import repeatability_statistics, ridge_predictions, trace_weights
from primary_temporal import directional_screens

RECIPES = ("lmp_W_HW_blend_unit_intersection_v1", "lmp_W_HW_blend_unit_intersection_random_support8_v1")
SUPPORT_KEYS = ("baseline_R2", "baseline_SSE", "denominator", "direction_counts")
REPLICATES = tuple(range(8))


def fit_current_view(selected_x, y, labels, hw_metadata, check=lambda: None):
    """Only original W's true-pair arm; preserve selected18 -> current15 layout."""
    x, y, labels = np.asarray(selected_x), np.asarray(y), np.asarray(labels)
    if (x.shape[:2] != (len(y), 18) or x.ndim != 3 or x.shape[2] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),)
            or set(labels.tolist()) != set(range(8)) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or [row["held_direction"] for row in hw_metadata] != list(range(8))):
        raise ValueError("Complete qualified reduced history/current view and saved HW fold metadata required")
    current = x[:, 3:18, :]
    baseline = np.full_like(y, np.nan)
    predictions = np.full((5, *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        train, test = labels != held, labels == held
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        true_u = residual-intercept
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        z, state = prepare_split(current, train)
        statistics = repeatability_statistics(z[train], labels[train])
        preserved = {"physical_mean": state["physical_mean"].tolist(), "physical_sd": state["physical_sd"].tolist(),
            "repeatability": {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in statistics.items()}}
        if any(preserved[key] != hw_metadata[held][key] for key in preserved):
            raise ValueError("Exact reduced W current-state replay against saved HW failed")
        flat = z[train].reshape(-1, x.shape[-1])
        gram = flat.T@flat/len(flat)
        weights, info = trace_weights(statistics["a"], gram)
        transformed = z*weights
        covariance = projected_covariance(transformed[train])
        fitted = ridge_predictions(transformed[train], transformed[test], true_u, check)
        for index in range(5):
            predictions[index, test] = base+fitted[index].reshape(base.shape)
        metadata.append({"held_direction": held, "fitting_bin_rows": len(true_u), **preserved,
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "paired_weights": info, "paired_weighted_covariance": covariance,
            "saved_HW_current_state_exact_replay": True})
    return baseline, predictions, metadata


def mean_cells(views):
    if len(views) != 8:
        raise ValueError("All eight ordinary support mixtures required")
    first = views[0]["blend_bank"]
    if any(any(view["blend_bank"][key] != first[key] for key in SUPPORT_KEYS) for view in views):
        raise ValueError("Mean support mixture must preserve original B/D/directions")
    result = []
    for index in range(25):
        li, ai = divmod(index, 5)
        correct = float(np.mean([view["blend_bank"]["cells"][index]["candidate_SSE"] for view in views]))
        wrong = float(np.mean([view["blend_bank"]["cells"][index]["wrong_pair_SSE"] for view in views]))
        result.append({"lambda_index": li, "alpha_index": ai, "lambda": LAMBDAS[li], "alpha": ALPHAS[ai],
            "candidate_R2": 1-correct/first["denominator"], "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return result


def score_views(y, baseline, labels, description, w_predictions, hw_predictions,
                old_hw_index, full_blend_prediction, check=lambda: None):
    supports = description["support_sets"]
    if (len(supports) != 9 or len(w_predictions) != 9 or len(hw_predictions) != 9
            or type(old_hw_index) is not int or old_hw_index not in range(5)
            or np.asarray(labels).tolist() != description["trial_directions"]
            or len(y) != description["complete_trials"] or full_blend_prediction.shape != y.shape):
        raise ValueError("All nine exact frozen supports and original selected HW index required")
    views = []
    for vi, support in enumerate(supports):
        check()
        w, hw = w_predictions[vi], hw_predictions[vi]
        bank = score_session(y, baseline, labels, w, hw, check)
        w_bank = score_bank(y, labels, baseline, w, COMPONENTS["W"])
        hw_bank = score_bank(y, labels, baseline, hw, support["recipe"])
        for cell in bank["cells"]:
            li, alpha = cell["lambda_index"], cell["alpha"]
            prediction = blend_prediction(w[li], hw[li], alpha)
            intersection = blend_prediction(w_predictions[0][li], hw_predictions[0][li], alpha)
            cell["exact_reuse"] = {
                "prior_I_HW": bool(np.array_equal(prediction, hw_predictions[0][old_hw_index])),
                "full_blend": bool(np.array_equal(prediction, full_blend_prediction)),
                "prior_R_HW_draws": [bool(np.array_equal(prediction, values[old_hw_index])) for values in hw_predictions[1:]],
                "current_I_blend": bool(np.array_equal(prediction, intersection))}
        views.append({"support": support, "blend_bank": bank, "W_bank": w_bank, "HW_bank": hw_bank,
            "error_cross_products_by_lambda": [float(np.sum((y-w[li])*(y-hw[li]))) for li in range(5)]})
    return {"recipes": list(RECIPES), "views": views, "mean_random_cells": mean_cells(views[1:]),
        "original_I_HW_lambda_index": old_hw_index,
        "full_blend_scores": prediction_scores(y, baseline, labels, full_blend_prediction),
        **{key: views[0]["blend_bank"][key] for key in SUPPORT_KEYS},
        "native_XY_scaled": False, "original_HW_refitted": False, "full_W_refitted": False,
        "support_aggregation": "mean eight correct/wrong SSEs at I-blend lambda and alpha; never averaged predictions"}


def validate_bank(bank):
    if (bank["recipes"] != list(RECIPES) or len(bank["views"]) != 9
            or bank["native_XY_scaled"] is not False or bank["original_HW_refitted"] is not False
            or bank["full_W_refitted"] is not False or type(bank["original_I_HW_lambda_index"]) is not int
            or bank["original_I_HW_lambda_index"] not in range(5)):
        raise ValueError("Complete nine-support mixture bank and unchanged old neural models required")
    for vi, view in enumerate(bank["views"]):
        blend = view["blend_bank"]
        validate_blend_bank(blend)
        expected_recipe = OLD_RECIPES[0] if vi == 0 else OLD_RECIPES[1]
        if (view["support"]["recipe"] != expected_recipe
                or view["support"]["replicate"] != (None if vi == 0 else vi-1)
                or len(view["error_cross_products_by_lambda"]) != 5
                or not np.isfinite(view["error_cross_products_by_lambda"]).all()):
            raise ValueError("Intersection then eight fixed ordinary support views required")
        for old, recipe in ((view["W_bank"], COMPONENTS["W"]), (view["HW_bank"], expected_recipe)):
            if (old["recipe"] != recipe or old["lambdas"] != list(LAMBDAS)
                    or [row["lambda"] for row in old["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(row[key]) for row in old["lambda_rows"] for key in SCORE_KEYS)
                    or any(old[key] != bank[key] or old[key] != blend[key] for key in SUPPORT_KEYS)):
                raise ValueError("Unchanged original common support and component scores required")
        for li in range(5):
            for ai, component in ((0, "W_bank"), (4, "HW_bank")):
                if any(blend["cells"][5*li+ai][key] != view[component]["lambda_rows"][li][key] for key in SCORE_KEYS):
                    raise ValueError("Literal mixture endpoint score replay failed")
        for cell in blend["cells"]:
            flags = cell["exact_reuse"]
            if (set(flags) != {"prior_I_HW", "full_blend", "prior_R_HW_draws", "current_I_blend"}
                    or any(type(flags[key]) is not bool for key in ("prior_I_HW", "full_blend", "current_I_blend"))
                    or len(flags["prior_R_HW_draws"]) != 8 or any(type(flag) is not bool for flag in flags["prior_R_HW_draws"])
                    or (vi == 0 and flags["current_I_blend"] is not True)):
                raise ValueError("Literal complete-panel and per-draw reuse evidence required")
    if bank["mean_random_cells"] != mean_cells(bank["views"][1:]):
        raise ValueError("Random support mixture must average all eight correct/wrong SSEs")


def validate_prior_hw_selections(sessions, old_hw_panels):
    if set(sessions) != set(SESSIONS) or set(old_hw_panels) != {"I", "R"}:
        raise ValueError("Complete original I/R HW selection required")
    old_i = {s: sessions[s]["views"][0]["HW_bank"] for s in SESSIONS}
    if old_hw_panels["I"]["recipe"] != OLD_RECIPES[0] or old_hw_panels["R"]["recipe"] != OLD_RECIPES[1]:
        raise ValueError("Original reduced HW recipe identities changed")
    original_i = validate_original_selections(old_i, old_hw_panels["I"])
    if len(old_hw_panels["R"]["sessions"]) != 16 or {r["session_id"] for r in old_hw_panels["R"]["sessions"]} != set(SESSIONS):
        raise ValueError("Complete original R-HW mean panel required")
    original_r = {row["session_id"]: row for row in old_hw_panels["R"]["sessions"]}
    for session in SESSIONS:
        index = original_i[session]["lambda_index"]
        bank = average_banks([view["HW_bank"] for view in sessions[session]["views"][1:]])
        row = original_r[session]
        if (sessions[session]["original_I_HW_lambda_index"] != index or row["animal"] != ANIMALS[session]
                or row["lambda_index"] != index or row["outer_training_sessions"] != original_i[session]["outer_training_sessions"]
                or any(row[key] != bank["lambda_rows"][index][key] for key in ("lambda", *SCORE_KEYS))):
            raise ValueError("Old R-HW must replay the original I-HW-selected penalty without retuning")
        if len(row["selected_supports"]) != 8:
            raise ValueError("Every original selected R-HW draw required")
        for vi, saved in enumerate(row["selected_supports"], 1):
            view = sessions[session]["views"][vi]
            if (saved["support"] != view["support"]
                    or any(saved[key] != view["HW_bank"]["lambda_rows"][index][key] for key in ("lambda", *SCORE_KEYS))):
                raise ValueError("Original selected R-HW support and per-draw loss replay failed")
    return {"I": original_i, "R": original_r}


def selected_support(view, index):
    cell = view["blend_bank"]["cells"][index]
    li = cell["lambda_index"]
    return {"support": view["support"], **cell,
        "W_candidate_SSE": view["W_bank"]["lambda_rows"][li]["candidate_SSE"],
        "HW_candidate_SSE": view["HW_bank"]["lambda_rows"][li]["candidate_SSE"],
        "error_cross_product_over_D": view["error_cross_products_by_lambda"][li]/view["blend_bank"]["denominator"],
        "selected_fit_diagnostics": {name: [{"held_direction": row["held_direction"],
            "fitting_bin_rows": row["fitting_bin_rows"], "trace": row["paired_weighted_covariance"]["trace"],
            "numerical_rank": row["paired_weighted_covariance"]["numerical_rank"],
            "effective_df": row["paired_weighted_covariance"]["ridge_df_by_lambda"][li]["effective_df"]}
            for row in view.get(name+"_fit_metadata", [])] for name in ("W", "HW")}}


def summarize_banks(sessions, old_hw_panels, full_blend_panel, *, prior_valid, prior_patience,
                    prior_alternatives, prior_nulls):
    if (set(sessions) != set(SESSIONS) or len(full_blend_panel["sessions"]) != 16
            or {r["session_id"] for r in full_blend_panel["sessions"]} != set(SESSIONS)
            or any(type(v) is not int or v < 0 for v in (prior_valid, prior_patience, prior_alternatives, prior_nulls))
            or not 28 <= prior_valid <= 64):
        raise ValueError("Complete original cohort/anchors and explicit live counters required")
    for bank in sessions.values():
        validate_bank(bank)
    old_rows = validate_prior_hw_selections(sessions, old_hw_panels)
    full_rows = {row["session_id"]: row for row in full_blend_panel["sessions"]}
    choices = {s: sessions[s]["views"][0]["blend_bank"] for s in SESSIONS}
    selected = {recipe: [] for recipe in RECIPES}
    seed_rows, seed_effects = [[] for _ in REPLICATES], [[] for _ in REPLICATES]
    paired = []
    keys = ("I_minus_full_blend", "R_minus_full_blend", "I_minus_HW_blend_lambda_matched",
            "R_minus_HW_blend_lambda_matched", "I_minus_original_I_HW", "R_minus_original_R_HW")
    contrasts = {key: [] for key in keys}
    draw_contrasts = {key: [[] for _ in REPLICATES] for key in ("blend_lambda_matched", "original_R_HW")}
    equal_i = {key: [] for key in ("original_I_HW", "full_blend", "original_R_HW")}
    equal_r = {key: [] for key in ("original_R_HW", RECIPES[0], "original_I_HW", "full_blend")}
    for session in SESSIONS:
        bank = sessions[session]
        if any(bank["full_blend_scores"][key] != full_rows[session][key] for key in SCORE_KEYS):
            raise ValueError("Original full-support blend scores must replay exactly")
        index, training, scores = select_cell(choices, session)
        vi = bank["views"]
        cell, random_cell = vi[0]["blend_bank"]["cells"][index], bank["mean_random_cells"][index]
        li = cell["lambda_index"]
        identity = {"session_id": session, "animal": ANIMALS[session]}
        shared = {**identity, "cell_index": index, "outer_training_sessions": training,
            "intersection_blend_training_primary_scores": scores, "baseline_R2": bank["baseline_R2"]}
        selected[RECIPES[0]].append({**shared, **cell, "selected_supports": [selected_support(vi[0], index)]})
        selected[RECIPES[1]].append({**shared, **random_cell,
            "selected_supports": [selected_support(view, index) for view in vi[1:]]})
        paired.append({**identity, "lambda": cell["lambda"], "alpha": cell["alpha"],
            **full_model_contrast(cell, random_cell, bank["denominator"])})
        matched_r = average_banks([view["HW_bank"] for view in vi[1:]])["lambda_rows"][li]
        comparisons = {
            "I_minus_full_blend": (cell, full_rows[session]), "R_minus_full_blend": (random_cell, full_rows[session]),
            "I_minus_HW_blend_lambda_matched": (cell, vi[0]["HW_bank"]["lambda_rows"][li]),
            "R_minus_HW_blend_lambda_matched": (random_cell, matched_r),
            "I_minus_original_I_HW": (cell, old_rows["I"][session]),
            "R_minus_original_R_HW": (random_cell, old_rows["R"][session])}
        for name, (new, old) in comparisons.items():
            contrasts[name].append({**identity, "lambda": cell["lambda"], "alpha": cell["alpha"],
                "reference_lambda": old["lambda"], **full_model_contrast(new, old, bank["denominator"])})
        flags = cell["exact_reuse"]
        equal_i["original_I_HW"].append(flags["prior_I_HW"])
        equal_i["full_blend"].append(flags["full_blend"])
        equal_i["original_R_HW"].append(all(flags["prior_R_HW_draws"]))
        rflags = [view["blend_bank"]["cells"][index]["exact_reuse"] for view in vi[1:]]
        equal_r["original_R_HW"].append(all(flags["prior_R_HW_draws"][r] for r, flags in enumerate(rflags)))
        equal_r[RECIPES[0]].append(all(flags["current_I_blend"] for flags in rflags))
        equal_r["original_I_HW"].append(all(flags["prior_I_HW"] for flags in rflags))
        equal_r["full_blend"].append(all(flags["full_blend"] for flags in rflags))
        for r, view in enumerate(vi[1:]):
            draw = view["blend_bank"]["cells"][index]
            seed_rows[r].append({**identity, **draw, "baseline_R2": bank["baseline_R2"]})
            seed_effects[r].append({**identity, "lambda": cell["lambda"], "alpha": cell["alpha"],
                **full_model_contrast(cell, draw, bank["denominator"])})
            for name, old_index in (("blend_lambda_matched", li), ("original_R_HW", bank["original_I_HW_lambda_index"])):
                old = view["HW_bank"]["lambda_rows"][old_index]
                draw_contrasts[name][r].append({**identity, "lambda": cell["lambda"], "alpha": cell["alpha"],
                    "reference_lambda": old["lambda"], **full_model_contrast(draw, old, bank["denominator"])})
    duplicates = {RECIPES[0]: [key for key, flags in equal_i.items() if all(flags)],
                  RECIPES[1]: [key for key, flags in equal_r.items() if all(flags)]}
    charges = {recipe: int(not duplicates[recipe]) for recipe in RECIPES}
    if prior_valid+sum(charges.values()) > 64:
        raise ValueError("Original maximum scientific depth cannot be exceeded")
    panels = []
    for recipe in RECIPES:
        value = {"recipe": recipe, **panel(selected[recipe])}
        screens = {"primary_at_least_0p01": value["primary_increment"] >= .01,
            "identity_at_least_0p005": value["identity_increment"] >= .005, **directional_screens(value),
            "every_within_animal_primary_omission_positive": all(row["primary_increment"] > 0
                for rows in value["leave_one_session_within_animal"].values() for row in rows),
            "every_within_animal_identity_omission_positive": all(row["identity_increment"] > 0
                for rows in value["leave_one_session_within_animal"].values() for row in rows)}
        panels.append({**value, "reference_margin_screens": screens, "all_reference_margin_screens_pass": all(screens.values()),
            "scientific_trial_count_increment": charges[recipe], "sensitivity_only": True, "promotion_authorized": False})
    count, patience, transitions = prior_valid, prior_patience, []
    for recipe in RECIPES:
        count += charges[recipe]
        patience += charges[recipe]
        transitions.append({"recipe": recipe, "charge": charges[recipe], "completed_count": count,
            "patience": patience, "incumbent_reset": False})
    return {"recipes": list(RECIPES), "recipe_order": list(RECIPES), "panels": panels,
        "intersection_minus_random_support": panel(paired),
        "random_support_panels": [{"replicate": r, **panel(rows)} for r, rows in enumerate(seed_rows)],
        "intersection_minus_each_random_support": [{"replicate": r, **panel(rows)} for r, rows in enumerate(seed_effects)],
        "reference_contrasts": {name: panel(rows) for name, rows in contrasts.items()},
        "random_support_HW_contrasts": {name: [{"replicate": r, **panel(rows)} for r, rows in enumerate(draws)]
                                        for name, draws in draw_contrasts.items()},
        "scientific_trial_count_by_recipe": charges, "structural_duplicate_of": duplicates,
        "scientific_trial_count_increment": sum(charges.values()), "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": sum(charges.values()), "stress_diagnostic_count_increment": sum(charges.values()),
        "prior_valid_comparisons": prior_valid, "prior_patience": prior_patience,
        "prior_alternative_checks": prior_alternatives, "prior_literal_null_controls": prior_nulls,
        "total_valid_primary_trials_including_prior": count, "post_initial_comparisons": count-6,
        "post_initial_alternative_checks": prior_alternatives+sum(charges.values()), "post_initial_null_controls": prior_nulls,
        "patience_after": patience, "count_transitions": transitions, "incumbent_reset": False,
        "eligible_for_comparable_primary_incumbent": False, "sensitivity_only": True,
        "original_HW_refitted": False, "full_W_refitted": False, "promotion_authorized": False, "search_complete": False}
