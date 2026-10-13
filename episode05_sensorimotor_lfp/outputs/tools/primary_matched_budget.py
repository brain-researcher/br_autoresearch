"""Fixed HW resource sensitivities with metadata-only C/T/CT support views."""
import random

import numpy as np

from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_history import history_blocks, ridge_predictions
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_pca import projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_HW_matched_channels8_v1", "lmp_HW_matched_trials8_v1",
           "lmp_HW_matched_channels_trials8_v1")
REFERENCE = "lmp_history4_repeatability_weight_ridge_v1"
VIEWS = tuple(range(8))
ELECTRODE_BUDGET = 86
TRIAL_BUDGET = 12


def support_views(ids, labels, electrode_ids, session, *, electrode_budget=ELECTRODE_BUDGET,
                  trials_per_direction=TRIAL_BUDGET):
    ids, labels, electrodes = np.asarray(ids), np.asarray(labels), np.asarray(electrode_ids)
    numeric = numeric_trial_ids(ids, session)
    if (labels.shape != ids.shape or set(labels.tolist()) != set(range(8))
            or electrodes.ndim != 1 or not np.issubdtype(electrodes.dtype, np.integer)
            or not np.array_equal(electrodes, np.unique(electrodes)) or np.any(electrodes < 1)
            or type(electrode_budget) is not int or not 1 <= electrode_budget <= len(electrodes)
            or type(trials_per_direction) is not int or trials_per_direction < 2
            or any(np.count_nonzero(labels == d) < trials_per_direction for d in range(8))):
        raise ValueError("Original IDs, ordered physical support and feasible fixed budgets required")
    output = {recipe: [] for recipe in RECIPES}
    for view in VIEWS:
        literal = f"ep05-HW-budget-electrode-v1|20261004|{session}|{view}"
        queue = electrodes.tolist()
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(queue)
        chosen_e = set(queue[:electrode_budget])
        ep = [i for i, value in enumerate(electrodes) if value in chosen_e]
        chosen_t, trial_queues = set(), []
        for direction in range(8):
            positions = sorted(np.flatnonzero(labels == direction).tolist(), key=lambda i: int(numeric[i]))
            trial_literal = f"ep05-HW-budget-trial-v1|20261004|{session}|{view}|{direction}"
            rng = random.Random()
            rng.seed(trial_literal, version=2)
            rng.shuffle(positions)
            chosen_t.update(positions[:trials_per_direction])
            trial_queues.append({"direction": direction, "seed_literal": trial_literal,
                                 "trial_id_queue": ids[positions].tolist()})
        tp = [i for i in range(len(ids)) if i in chosen_t]
        for recipe, trial_positions, electrode_positions in (
                (RECIPES[0], list(range(len(ids))), ep),
                (RECIPES[1], tp, list(range(len(electrodes)))), (RECIPES[2], tp, ep)):
            output[recipe].append({"view": view, "trial_positions": trial_positions,
                "electrode_positions": electrode_positions, "trial_ids": ids[trial_positions].tolist(),
                "electrode_ids": electrodes[electrode_positions].tolist(),
                "full_support": len(trial_positions) == len(ids) and len(electrode_positions) == len(electrodes),
                "electrode_seed_literal": literal, "electrode_id_queue": queue,
                "trial_queues": trial_queues})
    return output


def metadata_direction_counts(row):
    counts = row["direction_counts"]
    if (not isinstance(counts, dict) or set(counts) != {str(d) for d in range(8)}
            or any(type(counts[str(d)]) is not int or counts[str(d)] < 2 for d in range(8))):
        raise ValueError("Metadata direction counts must contain exactly string keys0..7 and integral counts>=2")
    return [counts[str(d)] for d in range(8)]


def validate_global_metadata(metadata, records):
    """Reconcile the complete released metadata and original trial-ID table."""
    rows = metadata["sessions"]
    if (len(rows) != 16 or {row["session_id"] for row in rows} != set(SESSIONS)
            or set(records) != set(SESSIONS) or metadata["source_values_read"] is not False
            or metadata["score_values_released"] is not False
            or metadata["minimum_electrode_count"] != ELECTRODE_BUDGET
            or metadata["minimum_direction_trial_count"] != TRIAL_BUDGET):
        raise ValueError("Complete released outcome-blind support and exact86/12 budgets required")
    if (min(row["physical_LMP_electrode_count"] for row in rows) != ELECTRODE_BUDGET
            or min(value for row in rows for value in metadata_direction_counts(row)) != TRIAL_BUDGET):
        raise ValueError("Frozen global minima changed")
    plans = {}
    for row in rows:
        session = row["session_id"]
        ids = np.asarray([r["trial_id"] for r in records[session]])
        labels = np.asarray([int(r["direction"]) for r in records[session]])
        electrodes = row["physical_LMP_electrode_ids"]
        if (len(electrodes) != row["physical_LMP_electrode_count"]
                or len(ids) != row["complete_trials"]
                or [int(np.count_nonzero(labels == d)) for d in range(8)] != metadata_direction_counts(row)):
            raise ValueError("Frozen physical/trial support did not reconcile: "+session)
        plans[session] = support_views(ids, labels, electrodes, session)
    return plans


def subset_arrays(x, y, labels, ids, support):
    tp, ep = support["trial_positions"], support["electrode_positions"]
    if (tp != sorted(set(tp)) or ep != sorted(set(ep)) or not tp or not ep
            or min(tp) < 0 or max(tp) >= len(y) or min(ep) < 0 or max(ep) >= x.shape[-1]
            or np.asarray(ids)[tp].tolist() != support["trial_ids"]):
        raise ValueError("Original-order complete selected identities required")
    # This declared order preserves the advanced channel-indexing layout;
    # never force the current slice to C order.
    return x[tp][:, :, ep], y[tp], np.asarray(labels)[tp], np.asarray(ids)[tp]


def fit_view(x, y, labels, check=lambda: None):
    x, y, labels = np.asarray(x), np.asarray(y), np.asarray(labels)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),)
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Complete reduced native history/XY support required")
    baseline = np.full_like(y, np.nan)
    predictions = np.full((len(LAMBDAS), *y.shape), np.nan)
    blocks, current, metadata = history_blocks(x), x[:, 3:18, :], []
    for held in range(8):
        check()
        split = prepare_fold(blocks, current, y, labels, held)
        test = split["test"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        fitting, evaluation = split["fitting_history"], split["evaluation_history"]
        flat = fitting.reshape(-1, fitting.shape[-1])
        gram = flat.T@flat/len(flat)
        weights, info = history_trace_weights(split["repeatability"]["a"], gram)
        repeated = np.tile(weights, 4)
        weighted_fit, weighted_test = fitting*repeated, evaluation*repeated
        covariance = projected_covariance(weighted_fit)
        fitted = ridge_predictions(weighted_fit, weighted_test, split["residual"], check)
        for index in range(len(LAMBDAS)):
            predictions[index, test] = base+fitted[index].reshape(base.shape)
        metadata.append({"held_direction": held, "fitting_bin_rows": len(flat),
            "physical_mean": split["weight_state"]["physical_mean"].tolist(),
            "physical_sd": split["weight_state"]["physical_sd"].tolist(),
            "history_physical_mean": split["history_state"].feature_mean.tolist(),
            "history_physical_sd": split["history_state"].feature_sd.tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "repeatability": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in split["repeatability"].items()},
            "paired_weights": info, "paired_weighted_covariance": covariance})
    return baseline, predictions, metadata


def support_signature(support):
    return tuple(support["trial_ids"]), tuple(support["electrode_ids"])


def fit_reduced(x, y, labels, ids, supports, check=lambda: None):
    fits, routes, seen = {}, {}, {}
    for recipe in RECIPES:
        if [row["view"] for row in supports[recipe]] != list(VIEWS):
            raise ValueError("All eight ordered fixed views required")
        routes[recipe] = []
        for support in supports[recipe]:
            check()
            signature = support_signature(support)
            if support["full_support"]:
                key = "full_HW"
            elif signature in seen:
                key = seen[signature]
            else:
                key = f"fit{len(fits):03d}"
                sx, sy, sl, si = subset_arrays(x, y, labels, ids, support)
                baseline, predictions, metadata = fit_view(sx, sy, sl, check)
                fits[key] = {"baseline": baseline, "predictions": predictions, "metadata": metadata,
                             "trial_ids": si.tolist(), "electrode_ids": support["electrode_ids"]}
                seen[signature] = key
            routes[recipe].append(key)
    return fits, routes


def align_reference(reference, y, labels, ids):
    source_ids = np.asarray(reference["trial_ids"])
    ids = np.asarray(ids)
    if (source_ids.shape != ids.shape or len(set(source_ids.tolist())) != len(ids)
            or len(set(ids.tolist())) != len(ids) or set(source_ids.tolist()) != set(ids.tolist())):
        raise ValueError("Exact complete saved HW trial identities required")
    positions = {value: i for i, value in enumerate(source_ids.tolist())}
    order = [positions[value] for value in ids.tolist()]
    actual, directions = np.asarray(reference["actual_xy"])[order], np.asarray(reference["directions"])[order]
    baseline = np.asarray(reference["baseline_xy"])[order]
    predictions = np.asarray(reference["candidate_xy_by_lambda"])[:, order]
    if (not np.array_equal(actual, y) or not np.array_equal(directions, labels)
            or baseline.shape != y.shape or predictions.shape != (5, *y.shape)
            or not np.isfinite(baseline).all() or not np.isfinite(predictions).all()):
        raise ValueError("Exact native HW target/direction replay and finite five-penalty arrays required")
    return {**reference, "actual_xy": actual, "directions": directions, "trial_ids": ids,
            "baseline_xy": baseline, "candidate_xy_by_lambda": predictions}


def average_views(views, recipe):
    if len(views) != 8 or [row["view"] for row in views] != list(VIEWS):
        raise ValueError("Eight fixed ordered subset views required")
    banks = [view["bank"] for view in views]
    if any(not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0
           or bank["lambdas"] != list(LAMBDAS) for bank in banks):
        raise ValueError("Positive per-view denominators and original penalties required")
    baseline = float(np.mean([b["baseline_SSE"]/b["denominator"] for b in banks]))
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        if any(b["lambda_rows"][index]["lambda"] != penalty for b in banks):
            raise ValueError("Original complete ordered penalty rows required")
        correct, wrong = (float(np.mean([b["lambda_rows"][index][key]/b["denominator"] for b in banks]))
                          for key in ("candidate_SSE", "wrong_pair_SSE"))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct,
            "candidate_normalized_SSE": correct, "wrong_pair_normalized_SSE": wrong,
            "delta_R2": float(np.mean([b["lambda_rows"][index]["delta_R2"] for b in banks])),
            "identity_increment": float(np.mean([b["lambda_rows"][index]["identity_increment"] for b in banks]))})
    return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": 1-baseline,
            "baseline_normalized_SSE": baseline, "lambda_rows": rows, "views": views,
            "view_aggregation": "equal mean of each view's losses divided by its own target denominator",
            "sensitivity_only": True, "promotion_authorized": False}


def finalize_banks(y, labels, ids, supports, fits, routes, reference):
    reference = align_reference(reference, y, labels, ids)
    banks, payload = {}, {}
    for key, fit in fits.items():
        payload[key+"_baseline_xy"] = fit["baseline"]
        payload[key+"_candidate_xy_by_lambda"] = fit["predictions"]
        payload[key+"_trial_ids"] = np.asarray(fit["trial_ids"])
    for recipe in RECIPES:
        views = []
        for support, key in zip(supports[recipe], routes[recipe]):
            tp = support["trial_positions"]
            sy, sl = y[tp], np.asarray(labels)[tp]
            full_base, full_pred = reference["baseline_xy"][tp], reference["candidate_xy_by_lambda"][:, tp]
            if key == "full_HW":
                baseline, prediction, metadata = full_base, full_pred, reference["fold_metadata"]
            else:
                fit = fits[key]
                if fit["trial_ids"] != support["trial_ids"] or fit["electrode_ids"] != support["electrode_ids"]:
                    raise ValueError("Reused within-attempt fit must have identical original support")
                baseline, prediction, metadata = fit["baseline"], fit["predictions"], fit["metadata"]
            bank = score_bank(sy, sl, baseline, prediction, recipe)
            full = score_bank(sy, sl, baseline, full_pred, REFERENCE)
            own = score_bank(sy, sl, full_base, full_pred, REFERENCE)
            if bank["denominator"] <= 0 or not np.isfinite(bank["denominator"]):
                raise ValueError("Positive retained-target denominator required")
            compact = [{"held_direction": row["held_direction"], "fitting_bin_rows": row["fitting_bin_rows"],
                "trace": row["paired_weighted_covariance"]["trace"],
                "numerical_rank": row["paired_weighted_covariance"]["numerical_rank"],
                "ridge_df_by_lambda": row["paired_weighted_covariance"]["ridge_df_by_lambda"]} for row in metadata]
            views.append({"view": support["view"], "support": support, "fit_key": key,
                "bank": bank, "reference_bank": full, "reference_own_baseline_bank": own,
                "baseline_change": (bank["baseline_SSE"]-own["baseline_SSE"])/bank["denominator"],
                "fit_diagnostics": compact})
        banks[recipe] = average_views(views, recipe)
    return banks, payload


def recipe_charges(sessions):
    signatures, charges, duplicate_of = {}, {}, {}
    for recipe in RECIPES:
        views = [sessions[s][recipe]["views"] for s in SESSIONS]
        signature = tuple(tuple(support_signature(v["support"]) for v in row) for row in views)
        previous = next((name for name, old in signatures.items() if signature == old), None)
        duplicate = "full_HW" if all(v["support"]["full_support"] for row in views for v in row) else previous
        charges[recipe], duplicate_of[recipe] = int(duplicate is None), duplicate
        signatures[recipe] = signature
    return charges, duplicate_of


def summarize_banks(sessions, reference_panel):
    if (set(sessions) != set(SESSIONS) or reference_panel["recipe"] != REFERENCE
            or len(reference_panel["sessions"]) != 16
            or {r["session_id"] for r in reference_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("All original sessions and complete original HW selection required")
    full_rows = {row["session_id"]: row for row in reference_panel["sessions"]}
    selected, contrasts, baseline_changes = {}, {}, {}
    for recipe in RECIPES:
        selected[recipe] = []
        contrasts[recipe] = {name: [] for name in ("independently_selected", "diagnostic_lambda_matched")}
        baseline_changes[recipe] = []
        banks = {s: sessions[s][recipe] for s in SESSIONS}
        for session in SESSIONS:
            bank = banks[session]
            if bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS):
                raise ValueError("Complete original diagnostic penalty banks required")
            chosen, training, scores = select_paired_lambda(banks, session)
            original = full_rows[session]
            full_index = original["lambda_index"]
            if original["lambda"] != LAMBDAS[full_index] or set(original["outer_training_sessions"]) != set(SESSIONS)-{session}:
                raise ValueError("Original independent full-HW outer selection required")
            row = {"session_id": session, "animal": ANIMALS[session], "lambda_index": chosen,
                "outer_training_sessions": training, "inner_animal_balanced_scores": scores,
                "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            row["selected_view_scores"] = [{"view": v["view"], "denominator": v["bank"]["denominator"],
                "baseline_R2": v["bank"]["baseline_R2"], "baseline_change": v["baseline_change"],
                "trial_count": len(v["support"]["trial_ids"]), "electrode_count": len(v["support"]["electrode_ids"]),
                "fit_key": v["fit_key"], **v["bank"]["lambda_rows"][chosen],
                "full_HW_at_original_lambda": v["reference_bank"]["lambda_rows"][full_index],
                "full_HW_at_diagnostic_lambda": v["reference_bank"]["lambda_rows"][chosen],
                "full_HW_original_baseline_R2": v["reference_own_baseline_bank"]["baseline_R2"],
                "selected_fit_diagnostics": [{**{k: d[k] for k in ("held_direction", "fitting_bin_rows", "trace", "numerical_rank")},
                    "effective_df": d["ridge_df_by_lambda"][chosen]["effective_df"]} for d in v["fit_diagnostics"]]}
                for v in bank["views"]]
            selected[recipe].append(row)
            baseline_changes[recipe].append({"session_id": session, "animal": ANIMALS[session],
                "mean_normalized_baseline_SSE_change": float(np.mean([v["baseline_change"] for v in bank["views"]]))})
            for name, index in (("independently_selected", full_index), ("diagnostic_lambda_matched", chosen)):
                view_contrasts = []
                for view in bank["views"]:
                    actual, full = view["bank"]["lambda_rows"][chosen], view["reference_bank"]["lambda_rows"][index]
                    view_contrasts.append({"view": view["view"],
                        "delta_R2": (full["candidate_SSE"]-actual["candidate_SSE"])/view["bank"]["denominator"],
                        "identity_increment": actual["identity_increment"]-full["identity_increment"]})
                contrasts[recipe][name].append({"session_id": session, "animal": ANIMALS[session],
                    "lambda": LAMBDAS[chosen], "full_HW_lambda": LAMBDAS[index],
                    **{key: float(np.mean([v[key] for v in view_contrasts])) for key in ("delta_R2", "identity_increment")},
                    "view_contrasts": view_contrasts})
    charges, duplicates = recipe_charges(sessions)
    panels, summarized_contrasts = [], {}
    for recipe in RECIPES:
        panel = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": panel["primary_increment"] >= .01,
            "identity_at_least_0p005": panel["identity_increment"] >= .005, **directional_screens(panel)}
        panel.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
            scientific_trial_count_increment=charges[recipe], role="nonpromotable resource stress diagnostic",
            sensitivity_only=True, promotion_authorized=False, search_complete=False)
        panels.append(panel)
        summarized_contrasts[recipe] = {}
        for name, rows in contrasts[recipe].items():
            item = aggregate_rows(rows)
            item.update(contrast="reduced HW minus full HW on identical retained target support",
                primary_formula="(SSE_full_HW-SSE_reduced)/D_retained", directional_screens=directional_screens(item))
            summarized_contrasts[recipe][name] = item
    count = sum(charges.values())
    cursor, patience, transitions = 27, 0, []
    for recipe in RECIPES:
        if charges[recipe]:
            cursor += 1
            if cursor >= 29:
                patience += 1
        transitions.append({"recipe": recipe, "charge": charges[recipe], "completed_count": cursor,
            "patience": "not_started" if cursor <= 28 else patience, "incumbent_reset": False})
    return {"panels": panels, "reference_contrasts": summarized_contrasts,
        "baseline_changes": baseline_changes, "scientific_trial_count_by_recipe": charges,
        "structural_duplicate_of": duplicates, "scientific_trial_count_increment": count,
        "stress_diagnostic_count_increment": count, "null_control_trial_count_increment": 0,
        "total_valid_primary_trials_including_prior": 27+count, "post_initial_comparisons": 21+count,
        "post_initial_null_controls": 11, "post_initial_alternative_checks": 11+count,
        "recipe_order": list(RECIPES), "prospective_patience_transitions": transitions,
        "sensitivity_only": True, "incumbent_reset": False, "promotion_authorized": False, "search_complete": False}
