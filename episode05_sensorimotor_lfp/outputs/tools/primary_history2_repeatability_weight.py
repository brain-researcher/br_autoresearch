"""Fixed two-lag weighted LMP ridge and conditional-capacity training control."""
import numpy as np

from primary_capacity import score_bank
from primary_capacity_matched_surrogate import REPLICATES, average_banks, trial_map
from primary_covariance_svd import numeric_trial_ids
from primary_elapsed_only import full_model_contrast, panel
from primary_history import history_blocks
from primary_history_blend import COMPONENTS as ANCHORS, SCORE_KEYS, validate_original_selections
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import directional_screens, select_paired_lambda

RECIPES = ("lmp_history2_repeatability_weight_ridge_v1",
           "lmp_history2_repeatability_weight_training_surrogate8_v1")
LAGS = (0, -1)


def history2_trace_weights(base_weights, gram):
    a, g = np.asarray(base_weights, dtype=float), np.asarray(gram, dtype=float)
    if (a.ndim != 1 or len(a) < 1 or g.shape != (2*len(a), 2*len(a))
            or not np.isfinite(a).all() or not np.isfinite(g).all()
            or np.any(a < .1) or np.any(a > 1.1) or np.any(np.diag(g) < 0)):
        raise ValueError("Positive current-only prior and finite two-lag Gram required")
    diagonal = np.diag(g).reshape(2, len(a))
    q = diagonal.sum(axis=0)
    active = q > 0
    trace, denominator = float(np.trace(g)), float(a@q)
    if trace == 0:
        if np.any(g != 0):
            raise ArithmeticError("Zero trace requires exactly zero two-lag Gram")
        formula, identity = np.ones(len(a)), True
    else:
        if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(trace):
            raise ArithmeticError("Positive finite two-lag weight normalization required")
        formula = np.sqrt(a*trace/denominator)
        identity = bool(np.all(a[active] == a[active][0]))
    weights = np.ones(len(a)) if identity else formula
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ArithmeticError("Finite positive effective physical weights required")
    return weights, {"structural_identity": identity, "history_energy_by_electrode": q.tolist(),
        "history_energy_by_lag_electrode": diagonal.tolist(), "positive_energy_columns": np.flatnonzero(active).tolist(),
        "base_weights": a.tolist(), "weights": weights.tolist(), "formula_weights": formula.tolist(),
        "original_trace": trace, "normalization_denominator": denominator,
        "weighted_trace_from_diagonal": float(np.square(weights)@q)}


def prepare_two_lag_fold(blocks, current, y, labels, held):
    split = prepare_fold(blocks, current, y, labels, held)
    width = 2*blocks.shape[-1]
    return {**split, "fitting_two": split["fitting_history"][:, :, :width],
            "evaluation_two": split["evaluation_history"][:, :, :width]}


def shared_predictions(fitting, evaluation, residual, maps, check=lambda: None):
    fitting, evaluation, residual = (np.asarray(value, dtype=float) for value in (fitting, evaluation, residual))
    if (fitting.ndim != 3 or fitting.shape[1] != 15 or evaluation.ndim != 3
            or evaluation.shape[1:] != fitting.shape[1:] or residual.shape != (15*len(fitting), 2)
            or len(maps) != 8 or not all(np.isfinite(value).all() for value in (fitting, evaluation, residual))):
        raise ValueError("Complete finite two-lag features, eight maps and original residual required")
    phi, test = fitting.reshape(-1, fitting.shape[-1]), evaluation.reshape(-1, evaluation.shape[-1])
    gram = phi.T@phi
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    true_cross = phi.T@residual
    candidate = np.empty((5, len(test), 2))
    controls = np.empty((8, 5, len(test), 2))
    equal_cross, equal_design, identities = [], [], []
    for arm in range(9):
        check()
        if arm == 0:
            cross = true_cross
        else:
            mapping = np.asarray(maps[arm-1])
            if (mapping.shape != (len(fitting),) or not np.issubdtype(mapping.dtype, np.integer)
                    or not np.array_equal(np.sort(mapping), np.arange(len(fitting)))):
                raise ValueError("Complete whole fitting-trial map required")
            permuted = fitting[mapping]
            cross = permuted.reshape(phi.shape).T@residual
            equal_cross.append(bool(np.array_equal(cross, true_cross)))
            equal_design.append(bool(np.array_equal(permuted, fitting)))
            identities.append(bool(np.array_equal(mapping, np.arange(len(fitting)))))
        rhs = vectors.T@cross
        for index, penalty in enumerate(LAMBDAS):
            check()
            denominator = values+len(phi)*penalty
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ArithmeticError("Finite positive prescribed ridge denominator required")
            prediction = test@(vectors@(rhs/denominator[:, None]))
            if arm == 0:
                candidate[index] = prediction
            else:
                controls[arm-1, index] = prediction
    spectrum = np.maximum(values[::-1]/len(phi), 0.)
    tolerance = 64*np.finfo(np.float64).eps*phi.shape[1]*max(1., float(spectrum[0]))
    covariance = {"eigenvalues": spectrum.tolist(), "trace": float(np.trace(gram)/len(phi)),
        "numerical_rank": int(np.count_nonzero(spectrum > tolerance)), "numerical_rank_tolerance": tolerance,
        "ridge_df_by_lambda": [{"lambda": p, "effective_df": float(np.sum(spectrum/(spectrum+p)))} for p in LAMBDAS],
        "diagnostic_only": True, "eigensystem_reused_from_fit": True}
    return candidate, controls, {"fitting_design_rows": len(phi), "feature_columns": phi.shape[-1],
        "shared_Gram_eigendecompositions": 1, "surrogate_specific_Gram_eigendecompositions": 0,
        "original_HW_refitted": False, "capacity_matching": "one exact shared Gram/eigensystem for all nine heads",
        "surrogate_cross_product_equals_original_by_replicate": equal_cross,
        "surrogate_design_equals_original_by_replicate": equal_design,
        "whole_fitting_map_identity_by_replicate": identities, "shared_covariance": covariance}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(x), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or set(labels.tolist()) != set(range(8)) or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or electrode_ids is None or len(electrode_ids) != x.shape[-1]):
        raise ValueError("Original complete eighteen-bin reader support and fifteen-bin target required")
    blocks, current = history_blocks(x), x[:, 3:18, :]
    baseline = np.full_like(y, np.nan)
    predictions = {RECIPES[0]: np.full((5, *y.shape), np.nan), RECIPES[1]: np.full((8, 5, *y.shape), np.nan)}
    metadata = []
    for held in range(8):
        check()
        split = prepare_two_lag_fold(blocks, current, y, labels, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        fitting, evaluation = split["fitting_two"], split["evaluation_two"]
        phi = fitting.reshape(-1, fitting.shape[-1])
        gram = phi.T@phi/len(phi)
        statistics, weight_state = split["repeatability"], split["weight_state"]
        weights, info = history2_trace_weights(statistics["a"], gram)
        repeated = np.tile(weights, 2)
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        candidate, controls, capacity = shared_predictions(fitting*repeated, evaluation*repeated,
            split["residual"], [mapping for mapping, _ in maps], check)
        for index in range(5):
            predictions[RECIPES[0]][index, test] = base+candidate[index].reshape(base.shape)
            for replicate in REPLICATES:
                predictions[RECIPES[1]][replicate, index, test] = base+controls[replicate, index].reshape(base.shape)
        full = split["fitting_history"].reshape(-1, 4*x.shape[-1])
        gram4 = full.T@full/len(full)
        _, old_weights = history_trace_weights(statistics["a"], gram4)
        metadata.append({"held_direction": held, "fitting_bin_rows": len(split["residual"]),
            "physical_mean": weight_state["physical_mean"].tolist(), "physical_sd": weight_state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(weight_state["physical_sd"] == 0).tolist(),
            "history_physical_mean": state.feature_mean.tolist(), "history_physical_sd": state.feature_sd.tolist(),
            "history_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "repeatability": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in statistics.items()},
            "paired_weights": old_weights, "h2_weights": info,
            "h2_weighted_covariance": capacity["shared_covariance"], "capacity_identity": capacity,
            "surrogate_trial_maps": [records for _, records in maps],
            "retained_past_fitting_energy_exactly_zero": bool(np.all(np.diag(gram)[x.shape[-1]:] == 0)),
            "discarded_past_fitting_energy_exactly_zero": bool(np.all(np.diag(gram4)[2*x.shape[-1]:] == 0))})
    candidate = score_bank(y, labels, baseline, predictions[RECIPES[0]], RECIPES[0])
    control = average_banks([score_bank(y, labels, baseline, predictions[RECIPES[1]][r], RECIPES[1]) for r in REPLICATES])
    control["lambda_selection"] = "H2W other-fifteen-session-selected penalty shared by all eight draws"
    for bank in (candidate, control):
        bank.update(physical_electrode_ids=np.asarray(electrode_ids).tolist(), physical_electrode_count=x.shape[-1],
            original_trial_ids=ids.tolist(), original_direction_labels=labels.tolist(), history_column_count=2*x.shape[-1],
            lag_offsets=list(LAGS), reader_LMP_offsets=[-3, 14], effective_LMP_offsets=[-1, 14],
            native_XY_scaled=False, weights_use_velocity=False, positive_floor=.1, post_weight_standardization=False,
            post_weight_centering=False, original_HW_refitted=False)
    candidate.update(direction_fit_metadata=metadata, shared_Gram_eigendecompositions=8,
        surrogate_specific_Gram_eigendecompositions=0,
        structural_control_equals_candidate=all(all(row["capacity_identity"][
            "surrogate_cross_product_equals_original_by_replicate"]) for row in metadata),
        zero_energy_is_duplicate_proof=False)
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, predictions


def summarize_banks(sessions, anchor_banks, anchor_panels, blend_panel=None):
    if set(sessions) != set(SESSIONS) or set(anchor_banks) != set(ANCHORS) or set(anchor_panels) != set(ANCHORS):
        raise ValueError("All sixteen new and original component banks required")
    originals = {name: validate_original_selections(anchor_banks[name], anchor_panels[name]) for name in ANCHORS}
    blend_rows = None if blend_panel is None else {row["session_id"]: row for row in blend_panel["sessions"]}
    if blend_rows is not None and (len(blend_panel["sessions"]) != 16 or set(blend_rows) != set(SESSIONS)):
        raise ValueError("Complete fixed selected blend required")
    selected, controls, null_contrasts = [], [], []
    refs = {name+"_"+kind: [] for name in ANCHORS for kind in ("independently_selected", "H2W_lambda_matched")}
    if blend_rows is not None:
        refs["blend_originally_selected"] = []
    seeds, capacity = {r: [] for r in REPLICATES}, []
    candidate_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    equal_anchors, equal_old_null = {name: [] for name in ANCHORS}, []
    if blend_rows is not None:
        equal_anchors["blend"] = []
    for session in SESSIONS:
        candidate, control = (sessions[session][recipe] for recipe in RECIPES)
        for bank, recipe in ((candidate, RECIPES[0]), (control, RECIPES[1])):
            if bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS):
                raise ValueError("Complete prescribed five-penalty banks required")
            for other in (candidate, *[anchor_banks[name][session] for name in ANCHORS]):
                if any(bank[key] != other[key] for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts")):
                    raise ValueError("Exact original B/native-XY scoring support required")
        if [seed["replicate"] for seed in control["surrogate_seed_rows"]] != list(REPLICATES):
            raise ValueError("Every fixed surrogate draw required")
        index, training, scores = select_paired_lambda(candidate_banks, session)
        identity = {"session_id": session, "animal": ANIMALS[session]}
        chosen = {**identity, "lambda_index": index, "outer_training_sessions": training,
                  "paired_inner_animal_balanced_scores": scores, **candidate["lambda_rows"][index]}
        null = {**identity, "lambda_index": index, **control["lambda_rows"][index]}
        selected.append(chosen)
        controls.append(null)
        null_contrasts.append({**identity, "lambda": chosen["lambda"], **full_model_contrast(chosen, null, candidate["denominator"])})
        for seed in control["surrogate_seed_rows"]:
            seeds[seed["replicate"]].append({**identity, "lambda": chosen["lambda"],
                **full_model_contrast(chosen, seed["lambda_rows"][index], candidate["denominator"])})
        for name in ANCHORS:
            old = originals[name][session]
            flags = candidate["exact_prediction_equals_components"][name]
            if np.asarray(flags).shape != (5, 5) or any(type(v) is not bool for row in flags for v in row):
                raise ValueError("Literal full prediction equality matrix required")
            equal_anchors[name].append(flags[index][old["lambda_index"]])
            for kind, reference in (("independently_selected", old), ("H2W_lambda_matched", anchor_banks[name][session]["lambda_rows"][index])):
                refs[name+"_"+kind].append({**identity, "lambda": chosen["lambda"], "reference_lambda": reference["lambda"],
                    **full_model_contrast(chosen, reference, candidate["denominator"])})
        null_flags = candidate["exact_surrogate_equals_original_HW_surrogate_by_lambda"]
        if np.asarray(null_flags).shape != (5, 5) or any(type(v) is not bool for row in null_flags for v in row):
            raise ValueError("Literal equality of all eight saved original surrogate predictions required")
        equal_old_null.append(null_flags[index][originals["HW"][session]["lambda_index"]])
        if blend_rows is not None:
            original = blend_rows[session]
            if any(candidate["blend_reference_row"][key] != original[key] for key in SCORE_KEYS):
                raise ValueError("Fixed blend score replay required")
            equal_anchors["blend"].append(candidate["exact_prediction_equals_blend"][index])
            refs["blend_originally_selected"].append({**identity, "lambda": chosen["lambda"],
                "reference_lambda": original["lambda"], **full_model_contrast(chosen, original, candidate["denominator"])})
        capacity.append({**identity, "lambda": chosen["lambda"], "physical_electrode_count": candidate["physical_electrode_count"],
            "history_column_count": candidate["history_column_count"], "direction_fits": [
                {"held_direction": row["held_direction"], "fitting_bin_rows": row["fitting_bin_rows"],
                 "shared_trace": row["h2_weighted_covariance"]["trace"],
                 "shared_numerical_rank": row["h2_weighted_covariance"]["numerical_rank"],
                 "shared_ridge_df_by_lambda": row["h2_weighted_covariance"]["ridge_df_by_lambda"],
                 "shared_selected_effective_df": row["h2_weighted_covariance"]["ridge_df_by_lambda"][index]["effective_df"],
                 "capacity_identity": row["capacity_identity"],
                 "retained_past_fitting_energy_exactly_zero": row["retained_past_fitting_energy_exactly_zero"],
                 "discarded_past_fitting_energy_exactly_zero": row["discarded_past_fitting_energy_exactly_zero"]}
                for row in candidate["direction_fit_metadata"]]})
    duplicates = {RECIPES[0]: [name for name, flags in equal_anchors.items() if all(flags)], RECIPES[1]: []}
    if all(candidate_banks[s]["structural_control_equals_candidate"] for s in SESSIONS):
        duplicates[RECIPES[1]].append("H2W")
    if all(equal_old_null):
        duplicates[RECIPES[1]].append("original_HW_capacity_surrogate")
    counts = {recipe: int(not duplicates[recipe]) for recipe in RECIPES}
    candidate_panel = panel(selected)
    screens = {"primary_at_least_0p01": candidate_panel["primary_increment"] >= .01,
        "identity_at_least_0p005": candidate_panel["identity_increment"] >= .005,
        **directional_screens(candidate_panel),
        "every_within_animal_primary_omission_positive": all(row["primary_increment"] > 0 for rows in candidate_panel["leave_one_session_within_animal"].values() for row in rows),
        "every_within_animal_identity_omission_positive": all(row["identity_increment"] > 0 for rows in candidate_panel["leave_one_session_within_animal"].values() for row in rows)}
    return {"recipes": list(RECIPES), "recipe_order": list(RECIPES), "candidate": candidate_panel,
        "mean_surrogate": panel(controls), "candidate_minus_mean_surrogate": panel(null_contrasts),
        "candidate_minus_each_surrogate": [{"replicate": r, **panel(seeds[r])} for r in REPLICATES],
        "reference_contrasts": {name: panel(rows) for name, rows in refs.items()},
        "original_components": {name: panel([originals[name][s] for s in SESSIONS]) for name in ANCHORS},
        "blend_reference_available": blend_rows is not None, "shared_capacity_by_session": capacity,
        "structural_duplicate_of": duplicates, "scientific_trial_count_by_recipe": counts,
        "scientific_trial_count_increment": sum(counts.values()), "literal_null_control_count_increment": counts[RECIPES[1]],
        "alternative_explanation_count_increment": counts[RECIPES[1]],
        "eligible_for_comparable_primary_incumbent": bool(counts[RECIPES[0]]),
        "reference_margin_screens": screens, "all_reference_margin_screens_pass": all(screens.values()),
        "original_HW_refitted": False, "original_W_refitted": False, "new_promotion_gate": False,
        "promotion_authorized": False, "search_complete": False, "evidence": "exposed original-support development"}
