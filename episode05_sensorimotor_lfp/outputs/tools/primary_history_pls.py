"""Adopted fixed two-component regression-deflation transform on original HW."""
import time

import numpy as np

from primary_capacity import score_bank
from primary_capacity_matched_surrogate import average_banks
from primary_covariance_svd import numeric_trial_ids, projected_covariance, ridge_predictions, trial_map
from primary_elapsed_only import full_model_contrast, panel
from primary_history import history_blocks
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import directional_screens, select_paired_lambda

RECIPES = ("lmp_history4_repeatability_weight_pls2_ridge_v1",
           "lmp_history4_repeatability_weight_pls2_target_labels8_v1")
REPLICATES = tuple(range(8))
COMPONENT_CAP = 2
ANCHORS = {"HW": "lmp_history4_repeatability_weight_ridge_v1",
           "blend": "lmp_current_history_repeatability_convex_blend_v1",
           "HW105": "lmp_history4_repeatability_weight_nystrom105_v1",
           "covariance_SVD": "lmp_target_covariance_svd_ridge_v1",
           "K": "lmp_nystrom_rbf105_v1"}
DUPLICATE_ANCHORS = (*ANCHORS, "baseline")
SUPPORT_KEYS = ("lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts")
SCORE_KEYS = ("candidate_R2", "candidate_SSE", "wrong_pair_SSE", "delta_R2", "identity_increment")
CONTRAST_NAMES = ("candidate_minus_control", "candidate_minus_HW", "candidate_minus_blend",
                  "candidate_minus_HW_lambda_matched")


def _finite(*values):
    if not all(np.isfinite(value).all() for value in values):
        raise ArithmeticError("Nonfinite PLS intermediate; no numerical rescue")


def _tolerance(columns, leading):
    return 64*np.finfo(np.float64).eps*max(columns, 2)*max(1., float(leading))


def fit_pls(fitting, residual, check=lambda: None):
    """Return centered scores and a JSON-safe sequential PLS2 inference state."""
    x, u = np.asarray(fitting, dtype=np.float64), np.asarray(residual, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or u.shape != (len(x), 2):
        raise ValueError("Nonempty paired n-by-D input and native n-by-2 residual required")
    _finite(x, u)
    xh, uh = x.copy(), u.copy()
    state = {"input_columns": x.shape[1], "fitting_rows": len(x), "component_cap": COMPONENT_CAP,
             "components": [], "rank_steps": [], "stop_reason": "component_cap",
             "native_XY_scaled": False, "score_standardization": False,
             "deflation": "both_X_and_U_use_input_score", "inference": "sequential_input_deflation"}
    scores = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for step in range(COMPONENT_CAP):
            check()
            cross = xh.T@uh/len(x)
            _finite(cross)
            left, singular, _ = np.linalg.svd(cross, full_matrices=False)
            _finite(left, singular)
            tolerance = _tolerance(x.shape[1], singular[0])
            accepted = bool(singular[0] > tolerance)
            details = {"step": step, "singular_values": singular.tolist(),
                       "tolerance": tolerance, "accepted": accepted}
            state["rank_steps"].append(details)
            if not accepted:
                state["stop_reason"] = "numerical_zero_cross_covariance"
                break
            # The projector of the exact tied leading subspace fixes signs and
            # ties without choosing a native target coordinate or a new floor.
            tied = left[:, singular == singular[0]]
            coordinate = int(np.argmax(np.sum(tied*tied, axis=1)))
            direction = tied@tied[coordinate, :]
            norm = np.linalg.norm(direction)
            _finite(direction, norm)
            if norm <= 0:
                raise ArithmeticError("Invalid leading-subspace canonical direction")
            weight = direction/norm
            score = xh@weight
            energy = float(score@score)
            _finite(weight, score, energy)
            if energy <= 0:
                raise ArithmeticError("Nonpositive score energy after an accepted PLS direction")
            loading = xh.T@score/energy
            target_loading = uh.T@score/energy
            _finite(loading, target_loading)
            state["components"].append({**details, "weight": weight.tolist(),
                "input_loading": loading.tolist(), "target_loading": target_loading.tolist(),
                "score_energy": energy, "tie_multiplicity": tied.shape[1],
                "canonical_coordinate": coordinate})
            scores.append(score)
            xh = xh-score[:, None]*loading[None, :]
            uh = uh-score[:, None]*target_loading[None, :]
            _finite(xh, uh)
        raw = np.column_stack(scores) if scores else np.empty((len(x), 0), dtype=np.float64)
        mean = raw.mean(axis=0)
        centered = raw-mean
        _finite(mean, centered)
    state.update(retained_components=len(scores), score_mean=mean.tolist())
    return centered, state


def validate_pls_state(state, columns=None, rows=None):
    d, n, k = state["input_columns"], state["fitting_rows"], state["retained_components"]
    if (type(d) is not int or d < 1 or type(n) is not int or n < 1
            or type(k) is not int or not 0 <= k <= COMPONENT_CAP
            or state["component_cap"] != COMPONENT_CAP
            or (columns is not None and d != columns) or (rows is not None and n != rows)
            or state["native_XY_scaled"] is not False or state["score_standardization"] is not False
            or state["deflation"] != "both_X_and_U_use_input_score"
            or state["inference"] != "sequential_input_deflation"
            or len(state["components"]) != k or len(state["score_mean"]) != k):
        raise ValueError("Complete fixed PLS2 state required")
    expected_steps = k if k == COMPONENT_CAP else k+1
    if (len(state["rank_steps"]) != expected_steps
            or state["stop_reason"] != ("component_cap" if k == COMPONENT_CAP else "numerical_zero_cross_covariance")):
        raise ValueError("Exact component-cap or numerical-zero stop required")
    for index, step in enumerate(state["rank_steps"]):
        singular = np.asarray(step["singular_values"], dtype=float)
        if (step["step"] != index or singular.shape != (min(d, 2),)
                or not np.isfinite(singular).all() or np.any(singular < 0)
                or np.any(np.diff(singular) > 0)
                or step["tolerance"] != _tolerance(d, singular[0])
                or type(step["accepted"]) is not bool
                or step["accepted"] != bool(singular[0] > step["tolerance"])
                or step["accepted"] != (index < k)):
            raise ValueError("Recorded PLS rank decision differs from its fixed threshold")
        if index < k:
            component = state["components"][index]
            if (any(component[key] != step[key] for key in step)
                    or component["tie_multiplicity"] != int(np.count_nonzero(singular == singular[0]))
                    or type(component["canonical_coordinate"]) is not int
                    or not 0 <= component["canonical_coordinate"] < d
                    or not np.isfinite(component["score_energy"]) or component["score_energy"] <= 0):
                raise ValueError("Accepted PLS component identity differs")
            for key, shape in (("weight", (d,)), ("input_loading", (d,)), ("target_loading", (2,))):
                value = np.asarray(component[key], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("Finite complete PLS weight/loading required")
    _finite(np.asarray(state["score_mean"]))


def apply_pls(evaluation, state):
    """Project with stored sequential input deflation, never raw X@W."""
    x = np.asarray(evaluation, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("A two-dimensional evaluation input is required")
    validate_pls_state(state, columns=x.shape[1])
    _finite(x)
    residual, scores = x.copy(), []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for component in state["components"]:
            score = residual@np.asarray(component["weight"])
            residual = residual-score[:, None]*np.asarray(component["input_loading"])[None, :]
            _finite(score, residual)
            scores.append(score)
        raw = np.column_stack(scores) if scores else np.empty((len(x), 0), dtype=np.float64)
        result = raw-np.asarray(state["score_mean"])
        _finite(result)
    return result


def fit_fold_arms(fitting, evaluation, residual, maps, check=lambda: None):
    x, test, u = (np.asarray(value, dtype=np.float64) for value in (fitting, evaluation, residual))
    if (x.ndim != 3 or x.shape[1] != 15 or x.shape[-1] < 1 or len(x) < 1
            or test.ndim != 3 or test.shape[1:] != x.shape[1:]
            or u.shape != (15*len(x), 2) or len(maps) != len(REPLICATES)):
        raise ValueError("Whole-trial fitting/evaluation arrays and eight target maps required")
    _finite(x, test, u)
    flat, held = x.reshape(-1, x.shape[-1]), test.reshape(-1, x.shape[-1])
    targets = [u]
    identities, equal_targets = [], []
    for mapping in maps:
        mapping = np.asarray(mapping)
        if (mapping.shape != (len(x),) or not np.issubdtype(mapping.dtype, np.integer)
                or not np.array_equal(np.sort(mapping), np.arange(len(x)))):
            raise ValueError("A complete fitting-trial recipient-to-donor bijection is required")
        moved = u.reshape(len(x), 15, 2)[mapping].reshape(u.shape)
        targets.append(moved)
        identities.append(bool(np.array_equal(mapping, np.arange(len(x)))))
        equal_targets.append(bool(np.array_equal(moved, u)))
    predictions, states, covariances = [], [], []
    transform_seconds = head_seconds = 0.
    for target in targets:
        check()
        started = time.monotonic()
        scores, state = fit_pls(flat, target, check)
        eval_scores = apply_pls(held, state)
        transform_seconds += time.monotonic()-started
        states.append(state)
        covariances.append(projected_covariance(scores))
        started = time.monotonic()
        # The control's same shuffled native target fits BOTH projection and head.
        fitted = ridge_predictions(scores, eval_scores, target, check)
        _finite(fitted)
        predictions.append(fitted)
        head_seconds += time.monotonic()-started
    stacked = np.stack(predictions)
    return stacked[0], stacked[1:], {"candidate_transform": states[0], "control_transforms": states[1:],
        "candidate_score_covariance": covariances[0], "control_score_covariances": covariances[1:],
        "whole_fitting_map_identity_by_replicate": identities,
        "control_targets_equal_candidate_by_replicate": equal_targets,
        "supervised_projection_fits": 9, "exact_capacity_match": False,
        "target_shuffle_scope": "projection_and_head", "transform_seconds": transform_seconds,
        "ridge_head_seconds": head_seconds}


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
    controls = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_fold(blocks, current, y, labels, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        fitting, evaluation = split["fitting_history"], split["evaluation_history"]
        flat = fitting.reshape(-1, fitting.shape[-1])
        weights, info = history_trace_weights(split["repeatability"]["a"], flat.T@flat/len(flat))
        weighted_fit, weighted_test = fitting*np.tile(weights, 4), evaluation*np.tile(weights, 4)
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        candidate, surrogate, transform = fit_fold_arms(weighted_fit, weighted_test, split["residual"],
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
            "paired_weights": info, "paired_weighted_covariance": projected_covariance(weighted_fit),
            "target_trial_maps": [records for _, records in maps], "pls": transform})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    control = average_banks([score_bank(y, labels, baseline, controls[r], RECIPES[1]) for r in REPLICATES])
    common = {"physical_electrode_ids": np.asarray(electrode_ids).tolist(),
        "physical_electrode_count": x.shape[-1], "history_column_count": 4*x.shape[-1],
        "original_trial_ids": ids.tolist(), "original_direction_labels": labels.tolist(),
        "lag_offsets": [0, -1, -2, -3], "native_XY_scaled": False, "weights_use_velocity": False,
        "positive_floor": .1, "post_weight_standardization": False, "post_weight_centering": False,
        "direction_fit_metadata": metadata, "original_HW_refitted": False,
        "component_cap": COMPONENT_CAP, "score_standardization": False, "exact_capacity_match": False,
        "target_shuffle_scope": "projection_and_head", "supervised_projection_fits": 72}
    candidate.update(common)
    control.update(common)
    control["lambda_selection"] = "candidate outer-session-selected original penalty; controls never retune"
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, paired, controls


def _selected_scalar_row(row):
    # Old selected rows may contain entire nested training banks. Never carry
    # those into a new summary merely to show a preserved anchor's five scores.
    fields = ("session_id", "animal", "lambda_index", "lambda", "alpha", "cell_index", "baseline_R2", *SCORE_KEYS)
    return {key: row[key] for key in fields if key in row}


def contrast_screens(value):
    return {**directional_screens(value),
        "every_within_animal_primary_omission_positive": all(row["primary_increment"] > 0
            for rows in value["leave_one_session_within_animal"].values() for row in rows),
        "every_within_animal_identity_omission_positive": all(row["identity_increment"] > 0
            for rows in value["leave_one_session_within_animal"].values() for row in rows)}


def summarize_banks(sessions, anchor_rows, hw_banks):
    if (set(sessions) != set(SESSIONS) or set(hw_banks) != set(SESSIONS)
            or any(set(anchor_rows[name]) != set(SESSIONS) for name in ("HW", "blend"))):
        raise ValueError("All sixteen candidate/control and selected HW/blend anchors required")
    candidates = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, controls, seed_rows = [], [], {r: [] for r in REPLICATES}
    contrasts = {name: [] for name in CONTRAST_NAMES}
    projection_diagnostics = []
    for session in SESSIONS:
        index, training, scores = select_paired_lambda(candidates, session)
        candidate, control = sessions[session][RECIPES[0]], sessions[session][RECIPES[1]]
        if any(candidate[k] != control[k] or candidate[k] != hw_banks[session][k] for k in SUPPORT_KEYS):
            raise ValueError("Identical candidate/control/HW target support required")
        label = {"session_id": session, "animal": ANIMALS[session], "lambda_index": index,
                 "outer_training_sessions": training}
        chosen, null = candidate["lambda_rows"][index], control["lambda_rows"][index]
        selected.append({**label, "training_primary_scores": scores, "baseline_R2": candidate["baseline_R2"], **chosen})
        controls.append({**label, "baseline_R2": control["baseline_R2"], **null})
        refs = {"control": null, "HW": anchor_rows["HW"][session], "blend": anchor_rows["blend"][session],
                "HW_lambda_matched": hw_banks[session]["lambda_rows"][index]}
        for name, reference in refs.items():
            contrasts["candidate_minus_"+name].append({**label, "reference_lambda": reference["lambda"],
                **full_model_contrast(chosen, reference, candidate["denominator"])})
        for seed in control["surrogate_seed_rows"]:
            seed_rows[seed["replicate"]].append({**label,
                **full_model_contrast(chosen, seed["lambda_rows"][index], candidate["denominator"])})
        def diagnostic(state, covariance):
            return {"retained_components": state["retained_components"], "stop_reason": state["stop_reason"],
                "trace": covariance["trace"], "numerical_rank": covariance["numerical_rank"],
                "effective_df": covariance["ridge_df_by_lambda"][index]["effective_df"]}
        projection_diagnostics.append({"session_id": session, "lambda": chosen["lambda"], "folds": [
            {"held_direction": row["held_direction"],
             "candidate": diagnostic(row["pls"]["candidate_transform"], row["pls"]["candidate_score_covariance"]),
             "controls": [diagnostic(state, covariance) for state, covariance in zip(
                 row["pls"]["control_transforms"], row["pls"]["control_score_covariances"])]}
            for row in candidate["direction_fit_metadata"]]})
    result = panel(selected)
    reference_contrasts = {name: panel(rows) for name, rows in contrasts.items()}
    screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
               "identity_at_least_0p005": result["identity_increment"] >= .005, **contrast_screens(result)}
    comparison_screens = {name: contrast_screens(reference_contrasts[name]) for name in
                         ("candidate_minus_control", "candidate_minus_HW", "candidate_minus_blend")}
    return {"recipes": list(RECIPES), "candidate": result, "control": panel(controls),
        "reference_contrasts": reference_contrasts,
        "candidate_minus_each_control": [{"replicate": r, **panel(seed_rows[r])} for r in REPLICATES],
        "original_anchors": {name: panel([_selected_scalar_row(anchor_rows[name][s]) for s in SESSIONS])
                             for name in ("HW", "blend")},
        "projection_diagnostics": projection_diagnostics, "comparison_screens": comparison_screens,
        "reference_margin_screens": screens, "all_reference_margin_screens_pass": all(screens.values()),
        "promotion_authorized": False, "search_complete": False, "original_HW_refitted": False,
        "evidence": "exposed original-support development; fixed supervised compression; no mechanism attribution"}


def duplicate_accounting(candidate_flags, control_flags):
    names = DUPLICATE_ANCHORS
    if set(candidate_flags) != set(SESSIONS) or set(control_flags) != set(SESSIONS):
        raise ValueError("All sixteen literal selected-array duplicate records required")
    for session in SESSIONS:
        if (set(candidate_flags[session]) != set(names)
                or set(control_flags[session]) != set((*names, "candidate"))
                or any(type(value) is not bool for value in candidate_flags[session].values())
                or any(len(values) != 8 or any(type(value) is not bool for value in values)
                       for values in control_flags[session].values())):
            raise ValueError("Literal complete candidate and eight-draw anchor comparisons required")
    candidate_dup = [name for name in names if all(candidate_flags[s][name] for s in SESSIONS)]
    control_dup = [name for name in (*names, "candidate") if all(all(control_flags[s][name]) for s in SESSIONS)]
    c, n = int(not candidate_dup), int(not control_dup)
    return {"candidate_exact_duplicate_of": candidate_dup, "control_exact_duplicate_of": control_dup,
        "candidate_trial_count_increment": c, "control_trial_count_increment": n,
        "scientific_trial_count_increment": c+n, "literal_null_control_count_increment": n,
        "alternative_explanation_count_increment": n, "count_order": list(RECIPES),
        "eligible_for_comparable_primary_incumbent": bool(c), "control_updates_primary_incumbent": False,
        "selected_candidate_exact_equality": candidate_flags, "selected_control_exact_equality": control_flags}
