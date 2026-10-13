"""Fixed current-bin repeatability prior on the preserved four-lag LMP model."""
import numpy as np

from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_history import (LAGS, center_arm, history_blocks, prepare_split as history_split,
                             ridge_predictions, standardize_history)
from primary_lmp_high import _contrast
from primary_pca import prepare_split as current_split, projected_covariance
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_repeatability_weight import REPLICATES, average_control_banks, repeatability_statistics, weight_map
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda

RECIPES = ("lmp_history4_repeatability_weight_ridge_v1", "lmp_history4_repeatability_weight_labels8_v1")
ANCHORS = {"history": "lmp_history4_ridge_v1", "repeatability": "lmp_repeatability_weight_ridge_v1"}


def history_trace_weights(base_weights, gram):
    a, g = np.asarray(base_weights, dtype=float), np.asarray(gram, dtype=float)
    if (a.ndim != 1 or len(a) < 1 or g.shape != (4*len(a), 4*len(a))
            or not np.isfinite(a).all() or not np.isfinite(g).all()
            or np.any(a < .1) or np.any(a > 1.1) or np.any(np.diag(g) < 0)):
        raise ValueError("Positive fixed physical weights and finite four-lag Gram required")
    diagonal = np.diag(g).reshape(4, len(a))
    q = diagonal.sum(axis=0)
    active = q > 0
    trace, denominator = float(np.trace(g)), float(a@q)
    if trace == 0:
        if np.any(g != 0):
            raise ArithmeticError("Zero trace requires an exactly zero history Gram")
        formula, identity = np.ones(len(a)), True
    else:
        if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(trace):
            raise ArithmeticError("Positive finite history normalization required")
        formula = np.sqrt(a*trace/denominator)
        identity = bool(np.all(a[active] == a[active][0]))
    weights = np.ones(len(a)) if identity else formula
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ArithmeticError("Positive finite applied physical weights required")
    return weights, {"structural_identity": identity,
        "history_energy_by_electrode": q.tolist(), "history_energy_by_lag_electrode": diagonal.tolist(),
        "positive_energy_columns": np.flatnonzero(active).tolist(),
        "base_weights": a.tolist(), "weights": weights.tolist(), "formula_weights": formula.tolist(),
        "original_trace": trace, "normalization_denominator": denominator,
        "weighted_trace_from_diagonal": float(np.square(weights)@q)}


def prepare_fold(blocks, current, y, labels, held):
    """Separate preserved W prior state from preserved H decoder state."""
    split = history_split(blocks, y, labels, held)
    train, test, state = split["train"], split["test"], split["history_state"]
    z0, weight_state = current_split(current, train)
    statistics = repeatability_statistics(z0[train], labels[train])
    fitting, evaluation, center = center_arm(split["fitting_standardized"], standardize_history(blocks[test], state))
    return {**split, "fitting_history": fitting, "evaluation_history": evaluation,
            "history_center": center, "weight_state": weight_state, "repeatability": statistics}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None, electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete native eighteen-bin LMP and fifteen-bin XY support required")
    if electrode_ids is None or len(electrode_ids) != x.shape[-1]:
        raise ValueError("Exact original physical electrode identities required")
    maps = [weight_map(electrode_ids, session, r) for r in REPLICATES]
    # Preserve the original reader's time-fast layout for exact W reductions;
    # a C-order copy changes the floating-point mean/SD/statistics path.
    blocks, current = history_blocks(x), x[:, 3:18, :]
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    shuffled = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
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
        diagonal = np.diag(gram).reshape(4, x.shape[-1])
        active = diagonal.sum(axis=0) > 0
        no_past = bool(np.all(diagonal[1:] == 0))
        arms, covariances, equal_candidate = [], [], []
        candidate_weights = None
        for arm in range(1+len(REPLICATES)):
            check()
            a = statistics["a"] if arm == 0 else statistics["a"][maps[arm-1][0]]
            weights, info = history_trace_weights(a, gram)
            if arm == 0:
                candidate_weights = weights
            else:
                equal_candidate.append(bool(np.array_equal(weights[active], candidate_weights[active])))
            repeated = np.tile(weights, len(LAGS))
            weighted_fit, weighted_test = fitting*repeated, evaluation*repeated
            arms.append(info)
            covariances.append(projected_covariance(weighted_fit))
            fitted = ridge_predictions(weighted_fit, weighted_test, split["residual"], check)
            for index in range(len(LAMBDAS)):
                value = base+fitted[index].reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(split["residual"]),
            "physical_mean": weight_state["physical_mean"].tolist(),
            "physical_sd": weight_state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(weight_state["physical_sd"] == 0).tolist(),
            "history_physical_mean": state.feature_mean.tolist(), "history_physical_sd": state.feature_sd.tolist(),
            "history_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "all_past_lags_zero": no_past,
            "repeatability": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in statistics.items()},
            "paired_weights": arms[0], "control_weights_by_replicate": arms[1:],
            "paired_weighted_covariance": covariances[0],
            "control_weighted_covariance_by_replicate": covariances[1:],
            "controls_equal_candidate_by_replicate": equal_candidate})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    no_past = all(row["all_past_lags_zero"] for row in metadata)
    candidate.update(physical_electrode_count=x.shape[-1], physical_electrode_ids=np.asarray(electrode_ids).tolist(),
        lag_offsets=list(LAGS), history_column_count=4*x.shape[-1],
        column_order="lag0,-1,-2,-3 then frozen physical electrode",
        direction_fit_metadata=metadata, native_XY_scaled=False, positive_floor=.1,
        weights_use_velocity=False, post_weight_standardization=False, post_weight_centering=False,
        weight_state="unchanged current-bin W physical normalization; before H numerical centering",
        decoder_state="unchanged H lag/electrode normalization and fitting-column centering",
        weight_label_maps=[details for _, details in maps], all_past_lags_zero=no_past,
        structural_identity=all(row["paired_weights"]["structural_identity"] for row in metadata),
        all_controls_equal_candidate=all(all(row["controls_equal_candidate_by_replicate"]) for row in metadata))
    control = average_control_banks([score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES])
    control.update(lag_offsets=list(LAGS), physical_normalization_preserved=True, true_native_targets=True,
        controls_have_own_Gram=True, all_past_lags_zero=no_past,
        structural_identity=all(arm["structural_identity"] for row in metadata for arm in row["control_weights_by_replicate"]),
        diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}


def summarize_banks(sessions, anchor_banks, anchor_panels):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(ANCHORS)
            or set(anchor_panels) != set(ANCHORS)):
        raise ValueError("All original sessions and both preserved anchor families required")
    for name, recipe in ANCHORS.items():
        panel = anchor_panels[name]
        if (set(anchor_banks[name]) != set(SESSIONS) or panel["recipe"] != recipe
                or len(panel["sessions"]) != 16 or {r["session_id"] for r in panel["sessions"]} != set(SESSIONS)):
            raise ValueError("Complete sixteen-session H and W anchors required")
    for session, banks in sessions.items():
        if set(banks) != set(RECIPES):
            raise ValueError("Both complete HW banks required")
        combined = [*banks.items(), *((recipe, anchor_banks[name][session]) for name, recipe in ANCHORS.items())]
        for recipe, bank in combined:
            if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in bank["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("Complete finite original penalty banks required")
            for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"):
                if bank[key] != banks[RECIPES[0]][key]:
                    raise ValueError("HW/control/H/W baseline or target support differs")
        if (banks[RECIPES[1]]["weight_label_replicate_ids"] != list(REPLICATES)
                or any(banks[r]["lag_offsets"] != list(LAGS) for r in RECIPES)
                or banks[RECIPES[0]]["all_past_lags_zero"] != banks[RECIPES[1]]["all_past_lags_zero"]):
            raise ValueError("Fixed lags, all controls and shared history support required")
    candidate_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, anchor_matched = {r: [] for r in RECIPES}, {name: [] for name in ANCHORS}
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(candidate_banks, held)
        for recipe in RECIPES:
            bank = sessions[held][recipe]
            row = {"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                   "outer_training_sessions": training, "paired_inner_animal_balanced_scores": scores,
                   "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][chosen]}
            if recipe == RECIPES[1]:
                seeds = bank["weight_label_seed_rows"]
                if [r["replicate"] for r in seeds] != list(REPLICATES):
                    raise ValueError("Complete ordered seed scores required")
                row["selected_seed_scores"] = [{"replicate": r["replicate"], **r["lambda_rows"][chosen]} for r in seeds]
            if recipe == RECIPES[0] and "direction_fit_metadata" in bank:
                def diagnostic(weight, covariance):
                    return {**weight, "trace": covariance["trace"], "numerical_rank": covariance["numerical_rank"],
                            "effective_df": covariance["ridge_df_by_lambda"][chosen]["effective_df"]}
                row["selected_weight_diagnostics"] = [{"held_direction": d["held_direction"],
                    "repeatability": d["repeatability"], "all_past_lags_zero": d["all_past_lags_zero"],
                    "paired": diagnostic(d["paired_weights"], d["paired_weighted_covariance"]),
                    "controls": [diagnostic(w, c) for w, c in zip(d["control_weights_by_replicate"], d["control_weighted_covariance_by_replicate"])],
                    "controls_equal_candidate_by_replicate": d["controls_equal_candidate_by_replicate"]}
                    for d in bank["direction_fit_metadata"]]
            selected[recipe].append(row)
        for name in ANCHORS:
            anchor_matched[name].append({"session_id": held, "animal": ANIMALS[held],
                                        **anchor_banks[name][held]["lambda_rows"][chosen]})
    identities = {r: all(sessions[s][r]["structural_identity"] for s in SESSIONS) for r in RECIPES}
    same = all(candidate_banks[s]["all_controls_equal_candidate"] for s in SESSIONS)
    no_past = all(candidate_banks[s]["all_past_lags_zero"] for s in SESSIONS)
    charges = {RECIPES[0]: int(not identities[RECIPES[0]] and not no_past),
               RECIPES[1]: int(not identities[RECIPES[1]] and not same and not no_past)}
    panels = []
    for recipe in RECIPES:
        result = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": result["primary_increment"] >= .01,
                   "identity_at_least_0p005": result["identity_increment"] >= .005, **directional_screens(result)}
        result.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
                      structural_identity=identities[recipe], scientific_trial_count_increment=charges[recipe],
                      promotion_authorized=False, search_complete=False, evidence="exposed exploratory development")
        panels.append(result)
    control = _contrast(selected[RECIPES[0]], selected[RECIPES[1]], "HW minus mean weight-label control")
    contrasts = {}
    for name in ANCHORS:
        contrasts[name+"_independently_selected"] = _contrast(selected[RECIPES[0]], anchor_panels[name]["sessions"],
                                                              "HW minus independently selected "+name)
        contrasts[name+"_HW_lambda_matched"] = _contrast(selected[RECIPES[0]], anchor_matched[name],
                                                         "HW minus "+name+" at HW-selected lambda")
    screens = {k: c["all_directional_screens_pass"] for k, c in contrasts.items()}
    screens["weight_labels"] = control["all_directional_screens_pass"]
    return {"panels": panels, "weight_label_control": control, "anchor_contrasts": contrasts,
            "matched_lambda_anchor_sessions": anchor_matched, "weighting_screens": screens,
            "all_weighting_screens_pass": all(screens.values()), "structural_identity_by_recipe": identities,
            "structural_current_only_duplicate": no_past, "all_controls_equal_candidate": same,
            "scientific_trial_count_by_recipe": charges, "scientific_trial_count_increment": sum(charges.values()),
            "control_trial_count_increment": charges[RECIPES[1]]}
