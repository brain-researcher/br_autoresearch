"""One fixed HW training-feature surrogate with exactly shared ridge capacity."""
import random

import numpy as np

from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_elapsed_only import full_model_contrast, panel
from primary_history import history_blocks
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_pca import projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS

RECIPE = "lmp_history4_repeatability_weight_training_surrogate8_v1"
ANCHOR = "lmp_history4_repeatability_weight_ridge_v1"
REPLICATES = tuple(range(8))


def trial_map(ids, labels, session, held, replicate):
    ids, labels = np.asarray(ids), np.asarray(labels)
    numeric = numeric_trial_ids(ids, session)
    if (held not in range(8) or replicate not in REPLICATES or labels.shape != ids.shape
            or set(labels.tolist()) != set(range(8))-{held}):
        raise ValueError("Seven fitting directions and one fixed replicate required")
    mapping, records = np.arange(len(ids)), []
    for direction in sorted(set(range(8))-{held}):
        positions = sorted(np.flatnonzero(labels == direction).tolist(), key=lambda i: int(numeric[i]))
        if len(positions) < 2:
            raise ValueError("Original complete fitting directions require at least two trials")
        donors = positions.copy()
        literal = ("ep05-lmp-hw-capacity-training-feature-surrogate-v1|20261004|"
                   f"{session}|{held}|{replicate}|{direction}")
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donors)
        mapping[positions] = donors
        records.append({"direction": direction, "replicate": replicate, "seed_literal": literal,
            "recipient_trial_ids": ids[positions].tolist(), "donor_trial_ids": ids[donors].tolist(),
            "identity_map": donors == positions,
            "fixed_point_count": sum(a == b for a, b in zip(positions, donors))})
    if (not np.array_equal(np.sort(mapping), np.arange(len(ids)))
            or not np.array_equal(labels[mapping], labels)):
        raise ValueError("Complete within-direction fitting-trial bijection required")
    return mapping, records


def shared_surrogate_predictions(fitting, evaluation, residual, maps, check=lambda: None):
    """One unchanged HW Gram/eigensystem; only surrogate cross-products vary."""
    fitting, evaluation, residual = (np.asarray(x, dtype=float) for x in (fitting, evaluation, residual))
    if (fitting.ndim != 3 or fitting.shape[1] != 15 or evaluation.ndim != 3
            or evaluation.shape[1:] != fitting.shape[1:]
            or residual.shape != (15*len(fitting), 2) or len(maps) != len(REPLICATES)
            or not all(np.isfinite(x).all() for x in (fitting, evaluation, residual))):
        raise ValueError("Finite complete HW trajectories and unchanged native residual required")
    phi, test = fitting.reshape(-1, fitting.shape[-1]), evaluation.reshape(-1, evaluation.shape[-1])
    gram = phi.T@phi
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    true_cross = phi.T@residual
    predictions = np.empty((len(REPLICATES), len(LAMBDAS), len(test), 2))
    equal_cross, equal_design, identity_maps = [], [], []
    for replicate, mapping in enumerate(maps):
        check()
        mapping = np.asarray(mapping)
        if (mapping.shape != (len(fitting),) or not np.issubdtype(mapping.dtype, np.integer)
                or not np.array_equal(np.sort(mapping), np.arange(len(fitting)))):
            raise ValueError("Each surrogate requires one complete fitting-trial permutation")
        permuted = fitting[mapping]
        cross = permuted.reshape(phi.shape).T@residual
        equal_cross.append(bool(np.array_equal(cross, true_cross)))
        equal_design.append(bool(np.array_equal(permuted, fitting)))
        identity_maps.append(bool(np.array_equal(mapping, np.arange(len(fitting)))))
        rhs = vectors.T@cross
        for index, penalty in enumerate(LAMBDAS):
            check()
            denominator = values+len(phi)*penalty
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ArithmeticError("Original positive-penalty HW eigensystem required")
            predictions[replicate, index] = test@(vectors@(rhs/denominator[:, None]))
    return predictions, {"fitting_design_rows": len(phi), "feature_columns": phi.shape[-1],
        "shared_Gram_eigendecompositions": 1, "surrogate_specific_Gram_eigendecompositions": 0,
        "original_HW_refitted": False, "capacity_matching": "exact whole-trial row-permutation Gram identity",
        "surrogate_cross_product_equals_original_by_replicate": equal_cross,
        "surrogate_design_equals_original_by_replicate": equal_design,
        "whole_fitting_map_identity_by_replicate": identity_maps}


def average_banks(banks):
    if len(banks) != len(REPLICATES):
        raise ValueError("All eight prescribed surrogate banks required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError("All surrogate baselines/targets/support/penalties must match")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        seeds = [bank["lambda_rows"][index] for bank in banks]
        if any(row["lambda"] != penalty for row in seeds):
            raise ValueError("All five prescribed penalties required")
        correct = float(np.mean([row["candidate_SSE"] for row in seeds]))
        wrong = float(np.mean([row["wrong_pair_SSE"] for row in seeds]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "ridge_fits": 8*8*len(LAMBDAS),
        "surrogate_replicate_ids": list(REPLICATES), "surrogate_Monte_Carlo_draws": 8,
        "surrogate_seed_rows": [{"replicate": r, "lambda_rows": bank["lambda_rows"]}
                                for r, bank in zip(REPLICATES, banks)],
        "surrogate_aggregation": "mean correct/wrong SSE; never mean predictions",
        "lambda_selection": "original independently selected HW penalty shared by all eight surrogate refits"}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or electrode_ids is None or len(electrode_ids) != x.shape[-1]):
        raise ValueError("Complete original native history/XY/trial/physical support required")
    blocks, current = history_blocks(x), x[:, 3:18, :]
    baseline = np.full_like(y, np.nan)
    predictions = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_fold(blocks, current, y, labels, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        fitting, evaluation = split["fitting_history"], split["evaluation_history"]
        flat = fitting.reshape(-1, fitting.shape[-1])
        gram = flat.T@flat/len(flat)
        statistics, weight_state = split["repeatability"], split["weight_state"]
        weights, info = history_trace_weights(statistics["a"], gram)
        repeated = np.tile(weights, 4)
        weighted_fit, weighted_test = fitting*repeated, evaluation*repeated
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        fitted, capacity = shared_surrogate_predictions(weighted_fit, weighted_test, split["residual"],
            [mapping for mapping, _ in maps], check)
        for replicate in REPLICATES:
            for index in range(len(LAMBDAS)):
                predictions[replicate, index, test] = base+fitted[replicate, index].reshape(base.shape)
        metadata.append({"held_direction": held, "fitting_bin_rows": len(split["residual"]),
            "physical_mean": weight_state["physical_mean"].tolist(),
            "physical_sd": weight_state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(weight_state["physical_sd"] == 0).tolist(),
            "history_physical_mean": state.feature_mean.tolist(), "history_physical_sd": state.feature_sd.tolist(),
            "history_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "repeatability": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in statistics.items()},
            "paired_weights": info, "paired_weighted_covariance": projected_covariance(weighted_fit),
            "surrogate_trial_maps": [records for _, records in maps], "capacity_identity": capacity})
    bank = average_banks([score_bank(y, labels, baseline, predictions[r], RECIPE) for r in REPLICATES])
    bank.update(physical_electrode_ids=np.asarray(electrode_ids).tolist(), physical_electrode_count=x.shape[-1],
        original_trial_ids=ids.tolist(), original_direction_labels=labels.tolist(),
        history_column_count=4*x.shape[-1], lag_offsets=[0, -1, -2, -3], native_XY_scaled=False,
        weights_use_velocity=False, positive_floor=.1, post_weight_standardization=False,
        post_weight_centering=False, direction_fit_metadata=metadata, original_HW_refitted=False,
        shared_Gram_eigendecompositions=8, surrogate_specific_Gram_eigendecompositions=0,
        structural_original_HW_duplicate=all(all(row["capacity_identity"][
            "surrogate_cross_product_equals_original_by_replicate"]) for row in metadata))
    return bank, baseline, predictions


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != 16
            or {row["session_id"] for row in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete original sixteen-session surrogate/HW panels required")
    original = {row["session_id"]: row for row in anchor_panel["sessions"]}
    selected, contrasts, seed_contrasts = [], [], {r: [] for r in REPLICATES}
    capacity_records = []
    for session in SESSIONS:
        bank, old, chosen = sessions[session], anchor_banks[session], original[session]
        if (bank["recipe"] != RECIPE or old["recipe"] != ANCHOR
                or bank["lambdas"] != list(LAMBDAS) or old["lambdas"] != list(LAMBDAS)
                or bank["surrogate_replicate_ids"] != list(REPLICATES)
                or bank["original_HW_saved_prediction_rows"] != old["lambda_rows"]
                or bank["denominator"] <= 0):
            raise ValueError("Complete fixed surrogate and saved original HW bank required")
        for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
            if bank[key] != old[key]:
                raise ValueError("Original B/native-XY/trial support must match exactly")
        index = chosen["lambda_index"]
        if (index not in range(len(LAMBDAS))
                or any(chosen[k] != old["lambda_rows"][index][k] for k in old["lambda_rows"][index])
                or set(chosen["outer_training_sessions"]) != set(SESSIONS)-{session}):
            raise ValueError("Unchanged other-fifteen-session-selected original HW penalty required")
        if [r["replicate"] for r in bank["surrogate_seed_rows"]] != list(REPLICATES):
            raise ValueError("All eight prescribed seed banks required")
        row = bank["lambda_rows"][index]
        identity = {"session_id": session, "animal": ANIMALS[session]}
        selected.append({**identity, "lambda_index": index,
            "outer_training_sessions": chosen["outer_training_sessions"],
            "baseline_R2": bank["baseline_R2"], **row})
        # First argument is HW so positive means HW has smaller full prediction loss.
        contrasts.append({**identity, "lambda": row["lambda"],
            **full_model_contrast(chosen, row, bank["denominator"])})
        for seed in bank["surrogate_seed_rows"]:
            value = seed["lambda_rows"][index]
            seed_contrasts[seed["replicate"]].append({**identity, "lambda": value["lambda"],
                **full_model_contrast(chosen, value, bank["denominator"])})
        capacity_records.append({**identity, "lambda": row["lambda"],
            "physical_electrode_count": bank["physical_electrode_count"],
            "history_column_count": bank["history_column_count"],
            "direction_fits": [{"held_direction": fit["held_direction"],
                "fitting_bin_rows": fit["fitting_bin_rows"],
                "shared_trace": fit["paired_weighted_covariance"]["trace"],
                "shared_numerical_rank": fit["paired_weighted_covariance"]["numerical_rank"],
                "shared_ridge_df_by_lambda": fit["paired_weighted_covariance"]["ridge_df_by_lambda"],
                "shared_selected_effective_df": fit["paired_weighted_covariance"]["ridge_df_by_lambda"][index]["effective_df"],
                "capacity_identity": fit["capacity_identity"]}
                for fit in bank["direction_fit_metadata"]]})
    duplicate = all(sessions[s]["structural_original_HW_duplicate"] for s in SESSIONS)
    charge = int(not duplicate)
    return {"recipe": RECIPE, "surrogate": panel(selected),
        "original_HW": panel([original[s] for s in SESSIONS]),
        "original_HW_minus_mean_surrogate": panel(contrasts),
        "original_HW_minus_each_surrogate": [{"replicate": r, **panel(seed_contrasts[r])} for r in REPLICATES],
        "shared_capacity_by_session": capacity_records,
        "structural_original_HW_duplicate": duplicate,
        "scientific_trial_count_increment": charge, "literal_null_control_count_increment": charge,
        "alternative_explanation_count_increment": charge, "updates_primary_incumbent": False,
        "promotion_authorized": False, "search_complete": False, "new_promotion_gate": False,
        "original_HW_refitted": False, "evidence": "exposed exploratory development; conditional-capacity feature-label surrogate"}
