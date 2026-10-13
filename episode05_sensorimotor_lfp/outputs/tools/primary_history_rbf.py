"""Fixed HW Nyström predictor and eight feature-row training-pairing controls."""
import numpy as np

from primary_capacity import fit_nystrom, landmark_rows, raw_nystrom, score_bank
from primary_capacity_matched_surrogate import REPLICATES, average_banks, trial_map
from primary_covariance_svd import numeric_trial_ids
from primary_elapsed_only import full_model_contrast, panel
from primary_history import history_blocks
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_pca import projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import directional_screens, select_paired_lambda

RECIPES = ("lmp_history4_repeatability_weight_nystrom105_v1",
           "lmp_history4_repeatability_weight_nystrom105_training_surrogate8_v1")
ANCHORS = {"HW": "lmp_history4_repeatability_weight_ridge_v1",
           "K": "lmp_nystrom_rbf105_v1", "blend": "lmp_current_history_repeatability_convex_blend_v1"}
SUPPORT_KEYS = ("lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts")


def prepare_kernel_fold(blocks, current, y, labels, ids, session, held):
    split = prepare_fold(blocks, current, y, labels, held)
    fitting, evaluation = split["fitting_history"], split["evaluation_history"]
    flat = fitting.reshape(-1, fitting.shape[-1])
    weights, info = history_trace_weights(split["repeatability"]["a"], flat.T@flat/len(flat))
    weighted_fit, weighted_test = fitting*np.tile(weights, 4), evaluation*np.tile(weights, 4)
    # Preserve the unmodified HW state for exact anchor replay. The established
    # RBF rule alone zeroes evaluation coordinates constant in fitting data.
    active = split["history_state"].feature_sd.reshape(-1) > 0
    kernel_fit, kernel_test = weighted_fit.copy(), weighted_test.copy()
    kernel_fit[..., ~active] = 0.
    kernel_test[..., ~active] = 0.
    rows, chosen_ids = landmark_rows(ids[split["train"]], labels[split["train"]], session, held)
    phi, state = fit_nystrom(kernel_fit.reshape(-1, fitting.shape[-1]), rows, int(active.sum()))
    phi_test = raw_nystrom(kernel_test.reshape(-1, fitting.shape[-1]), state)-state.feature_mean
    return {**split, "weighted_fit": weighted_fit, "weighted_test": weighted_test,
            "paired_weights": info, "active_kernel_columns": active, "kernel_state": state,
            "kernel_fit": kernel_fit, "kernel_test": kernel_test,
            "landmark_rows": rows, "landmark_trial_ids": chosen_ids,
            "phi": phi.reshape(len(fitting), 15, 105),
            "phi_test": phi_test.reshape(len(evaluation), 15, 105)}


def shared_heads(fitting, evaluation, residual, maps, check=lambda: None):
    fitting, evaluation, residual = (np.asarray(x, dtype=float) for x in (fitting, evaluation, residual))
    if (fitting.ndim != 3 or fitting.shape[1] != 15 or fitting.shape[2] < 1
            or evaluation.ndim != 3 or evaluation.shape[1:] != fitting.shape[1:]
            or residual.shape != (15*len(fitting), 2) or len(maps) != 8
            or not all(np.isfinite(x).all() for x in (fitting, evaluation, residual))):
        raise ValueError("Finite whole-trial feature trajectories and eight maps required")
    phi, test = fitting.reshape(-1, fitting.shape[-1]), evaluation.reshape(-1, fitting.shape[-1])
    gram = phi.T@phi
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    true_cross = phi.T@residual
    output = np.empty((9, len(LAMBDAS), len(test), 2))
    cross_equal, design_equal, identities = [], [], []
    for arm in range(9):
        check()
        if arm == 0:
            cross = true_cross
        else:
            mapping = np.asarray(maps[arm-1])
            if (mapping.shape != (len(fitting),) or not np.issubdtype(mapping.dtype, np.integer)
                    or not np.array_equal(np.sort(mapping), np.arange(len(fitting)))):
                raise ValueError("Complete fitting-trial bijections required")
            moved = fitting[mapping]
            cross = moved.reshape(phi.shape).T@residual
            cross_equal.append(bool(np.array_equal(cross, true_cross)))
            design_equal.append(bool(np.array_equal(moved, fitting)))
            identities.append(bool(np.array_equal(mapping, np.arange(len(fitting)))))
        rhs = vectors.T@cross
        for index, penalty in enumerate(LAMBDAS):
            check()
            denominator = values+len(phi)*penalty
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ArithmeticError("Positive finite original-penalty eigensystem required")
            output[arm, index] = test@(vectors@(rhs/denominator[:, None]))
    return output[0], output[1:], {"fitting_design_rows": len(phi), "feature_columns": phi.shape[-1],
        "shared_head_Gram_eigendecompositions": 1, "surrogate_specific_Gram_eigendecompositions": 0,
        "surrogate_cross_product_equals_candidate_by_replicate": cross_equal,
        "surrogate_design_equals_candidate_by_replicate": design_equal,
        "whole_fitting_map_identity_by_replicate": identities,
        "kernel_feature_covariance": projected_covariance(fitting)}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8)) or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or electrode_ids is None or len(electrode_ids) != x.shape[-1]):
        raise ValueError("Original complete history/XY/trial/physical support required")
    blocks, current = history_blocks(x), x[:, 3:18, :]
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    controls = np.full((8, len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_kernel_fold(blocks, current, y, labels, ids, session, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        candidate, surrogate, capacity = shared_heads(split["phi"], split["phi_test"], split["residual"],
            [mapping for mapping, _ in maps], check)
        for index in range(len(LAMBDAS)):
            paired[index, test] = base+candidate[index].reshape(base.shape)
            for replicate in REPLICATES:
                controls[replicate, index, test] = base+surrogate[replicate, index].reshape(base.shape)
        statistics, weight_state = split["repeatability"], split["weight_state"]
        metadata.append({"held_direction": held, "fitting_bin_rows": len(split["residual"]),
            "physical_mean": weight_state["physical_mean"].tolist(), "physical_sd": weight_state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(weight_state["physical_sd"] == 0).tolist(),
            "history_physical_mean": state.feature_mean.tolist(), "history_physical_sd": state.feature_sd.tolist(),
            "history_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "repeatability": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in statistics.items()},
            "paired_weights": split["paired_weights"],
            "paired_weighted_covariance": projected_covariance(split["weighted_fit"]),
            "active_RBF_columns": int(split["active_kernel_columns"].sum()),
            "RBF_sigma_squared": split["kernel_state"].sigma_squared,
            "landmark_trial_ids": split["landmark_trial_ids"], "landmark_count": 105,
            "landmark_bins_per_trial": 15, "landmark_diagonal_regularizer": 1e-8,
            "surrogate_trial_maps": [records for _, records in maps], "capacity_identity": capacity})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    control = average_banks([score_bank(y, labels, baseline, controls[r], RECIPES[1]) for r in REPLICATES])
    control["lambda_selection"] = "new paired HW105 candidate outer-session-selected penalty; controls never retune"
    common = {"physical_electrode_ids": np.asarray(electrode_ids).tolist(),
        "physical_electrode_count": x.shape[-1], "history_column_count": 4*x.shape[-1],
        "original_trial_ids": ids.tolist(), "original_direction_labels": labels.tolist(),
        "lag_offsets": [0, -1, -2, -3], "native_XY_scaled": False, "weights_use_velocity": False,
        "positive_floor": .1, "post_weight_standardization": False, "post_weight_centering": False,
        "direction_fit_metadata": metadata, "original_HW_refitted": False,
        "kernel_feature_columns": 105, "kernel_feature_centering": "fitting only; no standardization",
        "kernel_training_constant_coordinates": "zero in fitting and evaluation; no dropped identities"}
    candidate.update(common)
    control.update(common)
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, paired, controls


def summarize_banks(sessions, anchor_rows, anchor_banks):
    if set(sessions) != set(SESSIONS) or set(anchor_rows) != set(ANCHORS):
        raise ValueError("All sixteen new sessions and three original selected anchors required")
    candidates = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, controls, seed_rows = [], [], {r: [] for r in REPLICATES}
    contrasts = {name: [] for name in ("candidate_minus_control", *["candidate_minus_"+n for n in ANCHORS],
                                     "candidate_minus_HW_lambda_matched", "candidate_minus_K_lambda_matched")}
    for s in SESSIONS:
        index, training, scores = select_paired_lambda(candidates, s)
        candidate, control = sessions[s][RECIPES[0]], sessions[s][RECIPES[1]]
        if any(candidate[k] != control[k] for k in SUPPORT_KEYS):
            raise ValueError("Candidate/control target support differs")
        label = {"session_id": s, "animal": ANIMALS[s], "lambda_index": index,
                 "outer_training_sessions": training}
        chosen, null = candidate["lambda_rows"][index], control["lambda_rows"][index]
        selected.append({**label, "training_primary_scores": scores, "baseline_R2": candidate["baseline_R2"], **chosen})
        controls.append({**label, "baseline_R2": control["baseline_R2"], **null})
        refs = {"control": null, **{name: anchor_rows[name][s] for name in ANCHORS},
                **{name+"_lambda_matched": anchor_banks[name][s]["lambda_rows"][index] for name in ("HW", "K")}}
        for name, row in refs.items():
            contrasts["candidate_minus_"+name].append({**label, "reference_lambda": row["lambda"],
                **full_model_contrast(chosen, row, candidate["denominator"])})
        for seed in control["surrogate_seed_rows"]:
            seed_rows[seed["replicate"]].append({**label,
                **full_model_contrast(chosen, seed["lambda_rows"][index], candidate["denominator"])})
    result = panel(selected)
    screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
        "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result),
        "every_within_animal_primary_omission_positive": all(r["primary_increment"] > 0
            for rows in result["leave_one_session_within_animal"].values() for r in rows),
        "every_within_animal_identity_omission_positive": all(r["identity_increment"] > 0
            for rows in result["leave_one_session_within_animal"].values() for r in rows)}
    return {"recipes": list(RECIPES), "candidate": result, "control": panel(controls),
        "reference_contrasts": {name: panel(rows) for name, rows in contrasts.items()},
        "candidate_minus_each_control": [{"replicate": r, **panel(seed_rows[r])} for r in REPLICATES],
        "original_anchors": {name: panel([anchor_rows[name][s] for s in SESSIONS]) for name in ANCHORS},
        "reference_margin_screens": screens, "all_reference_margin_screens_pass": all(screens.values()),
        "promotion_authorized": False, "search_complete": False, "original_HW_refitted": False,
        "evidence": "exposed original-support development; nonlinear weighted history; no mechanism attribution"}


def duplicate_accounting(candidate_flags, control_flags):
    names = (*ANCHORS, "baseline")
    if set(candidate_flags) != set(SESSIONS) or set(control_flags) != set(SESSIONS):
        raise ValueError("All sixteen literal selected-array duplicate records required")
    for s in SESSIONS:
        if set(candidate_flags[s]) != set(names) or set(control_flags[s]) != set((*names, "candidate")):
            raise ValueError("Every fixed anchor and candidate identity comparison required")
        if any(type(v) is not bool for v in candidate_flags[s].values()):
            raise ValueError("Literal candidate duplicate flags required")
        if any(len(v) != 8 or any(type(x) is not bool for x in v) for v in control_flags[s].values()):
            raise ValueError("All eight literal control duplicate flags required")
    candidate_dup = [n for n in names if all(candidate_flags[s][n] for s in SESSIONS)]
    control_dup = [n for n in (*names, "candidate") if all(all(control_flags[s][n]) for s in SESSIONS)]
    c, n = int(not candidate_dup), int(not control_dup)
    return {"candidate_exact_duplicate_of": candidate_dup, "control_exact_duplicate_of": control_dup,
        "candidate_trial_count_increment": c, "control_trial_count_increment": n,
        "scientific_trial_count_increment": c+n, "literal_null_control_count_increment": n,
        "alternative_explanation_count_increment": n, "count_order": list(RECIPES),
        "eligible_for_comparable_primary_incumbent": bool(c), "control_updates_primary_incumbent": False,
        "selected_candidate_exact_equality": candidate_flags, "selected_control_exact_equality": control_flags}
