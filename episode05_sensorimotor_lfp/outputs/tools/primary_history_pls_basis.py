"""Frozen shuffled PLS projections with true-target heads and literal old replay."""
import time

import numpy as np

from primary_capacity import score_bank
from primary_capacity_matched_surrogate import average_banks
from primary_covariance_svd import numeric_trial_ids, projected_covariance, ridge_predictions, trial_map
from primary_elapsed_only import full_model_contrast, panel
from primary_history import history_blocks
from primary_history_pls import (DUPLICATE_ANCHORS, RECIPES as PLS_RECIPES, REPLICATES, SCORE_KEYS,
                                 SUPPORT_KEYS, _selected_scalar_row, apply_pls, contrast_screens)
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from run_primary_elapsed_only import STATE_KEYS

RECIPE = "lmp_history4_repeatability_weight_pls2_basis_labels8_true_head_v1"
SINGLE_REFERENCES = (*DUPLICATE_ANCHORS, "PLS")
ENSEMBLE_REFERENCES = ("PLS_full_target", "covariance_SVD_basis")
CONTRAST_NAMES = ("paired_PLS_minus_basis_only", "basis_only_minus_full_target")


def replay_fold_heads(fitting, evaluation, residual, maps, details, check=lambda: None):
    """Never construct a projection: apply saved states and refit old/new heads."""
    x, test, u = (np.asarray(value, dtype=np.float64) for value in (fitting, evaluation, residual))
    if (x.ndim != 3 or x.shape[1] != 15 or len(x) < 1 or x.shape[-1] < 1
            or test.ndim != 3 or test.shape[1:] != x.shape[1:]
            or u.shape != (15*len(x), 2) or len(maps) != 8
            or any(not np.isfinite(value).all() for value in (x, test, u))):
        raise ValueError("Complete original weighted-history inputs and native residual required")
    flat, held = x.reshape(-1, x.shape[-1]), test.reshape(-1, x.shape[-1])
    targets = [u]
    for mapping in maps:
        mapping = np.asarray(mapping)
        if (mapping.shape != (len(x),) or not np.issubdtype(mapping.dtype, np.integer)
                or not np.array_equal(np.sort(mapping), np.arange(len(x)))):
            raise ValueError("Complete original whole-fitting-trial maps required")
        targets.append(u.reshape(len(x), 15, 2)[mapping].reshape(u.shape))
    states = [details["candidate_transform"], *details["control_transforms"]]
    diagnostics = [details["candidate_score_covariance"], *details["control_score_covariances"]]
    if len(states) != 9 or len(diagnostics) != 9:
        raise ValueError("Exactly nine stored original projection states required")
    old_predictions, new_predictions = [], []
    old_seconds = new_seconds = 0.
    for index, (state, target, diagnostic) in enumerate(zip(states, targets, diagnostics)):
        check()
        if state["fitting_rows"] != len(flat):
            raise ValueError("Stored transform fitting support changed")
        scores, held_scores = apply_pls(flat, state), apply_pls(held, state)
        if projected_covariance(scores) != diagnostic:
            raise ValueError("Stored PLS score covariance did not replay literally")
        started = time.monotonic()
        old = ridge_predictions(scores, held_scores, target, check)
        old_seconds += time.monotonic()-started
        if not np.isfinite(old).all():
            raise ArithmeticError("Nonfinite old-head reconstruction")
        old_predictions.append(old)
        if index:
            started = time.monotonic()
            new = ridge_predictions(scores, held_scores, u, check)
            new_seconds += time.monotonic()-started
            if not np.isfinite(new).all():
                raise ArithmeticError("Nonfinite basis-only true-target head")
            new_predictions.append(new)
    return old_predictions[0], np.stack(old_predictions[1:]), np.stack(new_predictions), {
        "stored_transforms_reused": 9, "new_projection_fits": 0,
        "paired_candidate_score_covariance": diagnostics[0],
        "basis_score_covariances": diagnostics[1:],
        "paired_candidate_components": states[0]["retained_components"],
        "basis_components": [state["retained_components"] for state in states[1:]],
        "basis_and_full_target_score_Grams_identical": True,
        "candidate_and_basis_capacity_matched": False,
        "old_head_reconstruction_seconds": old_seconds, "new_true_head_seconds": new_seconds}


def fit_banks(features, targets, directions, trial_ids, session, old_banks,
              old_candidate, old_full_target, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    old_p, old_f = old_banks[PLS_RECIPES[0]], old_banks[PLS_RECIPES[1]]
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or any(not np.isfinite(value).all() for value in (x, y))
            or set(labels.tolist()) != set(range(8)) or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or electrode_ids is None or list(electrode_ids) != old_p["physical_electrode_ids"]
            or len(electrode_ids) != x.shape[-1] or ids.tolist() != old_p["original_trial_ids"]
            or labels.tolist() != old_p["original_direction_labels"]
            or old_candidate.shape != (5, *y.shape) or old_full_target.shape != (8, 5, *y.shape)):
        raise ValueError("Original complete source support and P/F arrays required")
    blocks, current = history_blocks(x), x[:, 3:18, :]
    baseline = np.full_like(y, np.nan)
    predictions = np.full((8, 5, *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        previous = old_p["direction_fit_metadata"][held]
        if previous["held_direction"] != held:
            raise ValueError("Original fold order changed")
        split = prepare_fold(blocks, current, y, labels, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        fitting, evaluation = split["fitting_history"], split["evaluation_history"]
        flat = fitting.reshape(-1, fitting.shape[-1])
        weights, info = history_trace_weights(split["repeatability"]["a"], flat.T@flat/len(flat))
        weighted_fit, weighted_test = fitting*np.tile(weights, 4), evaluation*np.tile(weights, 4)
        statistics, weight_state = split["repeatability"], split["weight_state"]
        row = {"held_direction": held, "fitting_bin_rows": len(split["residual"]),
            "physical_mean": weight_state["physical_mean"].tolist(), "physical_sd": weight_state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(weight_state["physical_sd"] == 0).tolist(),
            "history_physical_mean": state.feature_mean.tolist(), "history_physical_sd": state.feature_sd.tolist(),
            "history_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "repeatability": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in statistics.items()},
            "paired_weights": info, "paired_weighted_covariance": projected_covariance(weighted_fit)}
        if any(row[key] != previous[key] for key in (*STATE_KEYS, "paired_weighted_covariance", "fitting_bin_rows")):
            raise ValueError("Original HW/PLS fitting state did not replay literally")
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        if [records for _, records in maps] != previous["target_trial_maps"]:
            raise ValueError("Original target-label maps changed")
        p, f, b, diagnostics = replay_fold_heads(weighted_fit, weighted_test, split["residual"],
            [mapping for mapping, _ in maps], previous["pls"], check)
        for index in range(5):
            if not np.array_equal(base+p[index].reshape(base.shape), old_candidate[index, test]):
                raise ValueError("Stored-transform original paired PLS head did not replay literally")
            for replicate in REPLICATES:
                if not np.array_equal(base+f[replicate, index].reshape(base.shape), old_full_target[replicate, index, test]):
                    raise ValueError("Stored-transform original full-target head did not replay literally")
                predictions[replicate, index, test] = base+b[replicate, index].reshape(base.shape)
        row.update(target_trial_maps=[records for _, records in maps], basis_diagnostic=diagnostics,
                   original_paired_and_full_target_heads_replayed=True)
        metadata.append(row)
    for values, old, recipe in ((old_candidate, old_p, PLS_RECIPES[0]),):
        replay = score_bank(y, labels, baseline, values, recipe)
        if any(replay[key] != old[key] for key in (*SUPPORT_KEYS, "lambda_rows")):
            raise ValueError("Original paired PLS scoring did not replay literally")
    old_seed_banks = [score_bank(y, labels, baseline, old_full_target[r], PLS_RECIPES[1]) for r in REPLICATES]
    replay = average_banks(old_seed_banks)
    if (any(replay[key] != old_f[key] for key in (*SUPPORT_KEYS, "lambda_rows"))
            or any(replay["surrogate_seed_rows"][r]["lambda_rows"] != old_f["surrogate_seed_rows"][r]["lambda_rows"]
                   for r in REPLICATES)):
        raise ValueError("All original full-target seed/mean scores must replay")
    bank = average_banks([score_bank(y, labels, baseline, predictions[r], RECIPE) for r in REPLICATES])
    bank.update({key: old_p[key] for key in ("physical_electrode_ids", "physical_electrode_count", "history_column_count",
        "original_trial_ids", "original_direction_labels", "lag_offsets", "native_XY_scaled", "weights_use_velocity",
        "positive_floor", "post_weight_standardization", "post_weight_centering", "component_cap", "score_standardization")})
    bank.update(direction_fit_metadata=metadata, new_projection_fits=0, stored_transforms_reused=72,
        ridge_heads_use_true_targets=True, target_shuffle_scope="stored_projection_only",
        lambda_selection="preserved_PLS_candidate_outer_session_selection_only",
        original_HW_refitted=False, original_paired_and_full_target_heads_replayed=True,
        original_PLS_saved_scores_replayed=True, candidate_and_basis_capacity_matched=False)
    return bank, baseline, predictions


def duplicate_accounting(single_flags, ensemble_flags):
    if set(single_flags) != set(SESSIONS) or set(ensemble_flags) != set(SESSIONS):
        raise ValueError("All sixteen literal duplicate records required")
    for flags, names in ((single_flags, SINGLE_REFERENCES), (ensemble_flags, ENSEMBLE_REFERENCES)):
        for session in SESSIONS:
            if (set(flags[session]) != set(names)
                    or any(len(values) != 8 or any(type(value) is not bool for value in values)
                           for values in flags[session].values())):
                raise ValueError("Every fixed reference and all eight literal draw identities required")
    single = [name for name in SINGLE_REFERENCES if all(all(single_flags[s][name]) for s in SESSIONS)]
    ensemble = [name for name in ENSEMBLE_REFERENCES if all(all(ensemble_flags[s][name]) for s in SESSIONS)]
    count = int(not single and not ensemble)
    return {"exact_duplicate_single_panels": single, "exact_duplicate_ensembles": ensemble,
        "scientific_trial_count_increment": count, "alternative_explanation_count_increment": count,
        "literal_null_control_count_increment": 0, "candidate_trial_count_increment": 0,
        "eligible_for_comparable_primary_incumbent": False, "updates_primary_incumbent": False,
        "count_order": [RECIPE], "selected_exact_single_equality": single_flags,
        "selected_exact_ensemble_equality": ensemble_flags}


def summarize_banks(sessions, original_p, original_f, selected_p, selected_f, anchor_rows):
    if any(set(value) != set(SESSIONS) for value in (sessions, original_p, original_f, selected_p, selected_f)):
        raise ValueError("All sixteen complete new and preserved P/F sessions required")
    basis, projection, head = [], [], []
    each_projection, each_head = {r: [] for r in REPLICATES}, {r: [] for r in REPLICATES}
    diagnostics = []
    for session in SESSIONS:
        bank, p, f = sessions[session], selected_p[session], selected_f[session]
        index = p["lambda_index"]
        if f["lambda_index"] != index or any(bank[key] != original_p[session][key] or bank[key] != original_f[session][key] for key in SUPPORT_KEYS):
            raise ValueError("Preserved candidate penalty and identical native P/F/B support required")
        chosen = bank["lambda_rows"][index]
        label = {"session_id": session, "animal": ANIMALS[session], "lambda_index": index,
                 "lambda": chosen["lambda"], "outer_training_sessions": p["outer_training_sessions"]}
        basis.append({**label, "baseline_R2": bank["baseline_R2"], **chosen})
        projection.append({**label, **full_model_contrast(p, chosen, bank["denominator"])})
        head.append({**label, **full_model_contrast(chosen, f, bank["denominator"])})
        for r in REPLICATES:
            b_r = bank["surrogate_seed_rows"][r]["lambda_rows"][index]
            f_r = original_f[session]["surrogate_seed_rows"][r]["lambda_rows"][index]
            each_projection[r].append({**label, **full_model_contrast(p, b_r, bank["denominator"])})
            each_head[r].append({**label, **full_model_contrast(b_r, f_r, bank["denominator"])})
        folds = []
        for row in bank["direction_fit_metadata"]:
            d = row["basis_diagnostic"]
            pc, bc = d["paired_candidate_score_covariance"], d["basis_score_covariances"]
            folds.append({"held_direction": row["held_direction"],
                "paired_components": d["paired_candidate_components"], "basis_components": d["basis_components"],
                "paired_trace": pc["trace"], "basis_traces": [item["trace"] for item in bc],
                "paired_effective_df": pc["ridge_df_by_lambda"][index]["effective_df"],
                "basis_effective_df": [item["ridge_df_by_lambda"][index]["effective_df"] for item in bc]})
        diagnostics.append({"session_id": session, "lambda": chosen["lambda"], "folds": folds})
    contrasts = dict(zip(CONTRAST_NAMES, (panel(projection), panel(head))))
    originals = {"paired_PLS": selected_p, "full_target_control": selected_f, **anchor_rows}
    return {"recipe": RECIPE, "basis_only": panel(basis), "reference_contrasts": contrasts,
        "paired_PLS_minus_each_basis": [{"replicate": r, **panel(each_projection[r])} for r in REPLICATES],
        "each_basis_minus_matched_full_target": [{"replicate": r, **panel(each_head[r])} for r in REPLICATES],
        "original_anchors": {name: panel([_selected_scalar_row(rows[s]) for s in SESSIONS]) for name, rows in originals.items()},
        "comparison_screens": {name: contrast_screens(value) for name, value in contrasts.items()},
        "projection_diagnostics": diagnostics, "new_selector_fitted": False,
        "basis_and_full_target_score_Grams_identical": True, "candidate_and_basis_capacity_matched": False,
        "promotion_authorized": False, "search_complete": False}
