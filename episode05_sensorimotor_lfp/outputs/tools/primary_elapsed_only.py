"""Fixed elapsed-only behavioral sensitivity of the original full-support HW."""
import numpy as np

from method_components import DirectionTime
from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_history import history_blocks, ridge_predictions
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, select_paired_lambda

RECIPE = "lmp_history4_repeatability_weight_elapsed_only_sensitivity_v1"
ANCHOR = "lmp_history4_repeatability_weight_ridge_v1"


def elapsed_baseline(y, labels, held):
    """Equal-direction intercept-only template, then the original pooled offset."""
    y, labels = np.asarray(y, dtype=float), np.asarray(labels)
    if (y.ndim != 3 or y.shape[1:] != (15, 2) or labels.shape != (len(y),)
            or not np.isfinite(y).all() or set(labels.tolist()) != set(range(8))
            or held not in range(8)):
        raise ValueError("Complete original XY/direction support required")
    train, test = labels != held, labels == held
    template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 0, labels[train])
    uncalibrated = template.predict(labels[train]*np.pi/4)
    residual = (y[train]-uncalibrated).reshape(-1, 2)
    intercept = residual.mean(axis=0)
    fitting = uncalibrated+intercept
    evaluation = template.predict(labels[test]*np.pi/4)+intercept
    # Preserve HW's subtraction order, using the new T0/intercept rather than B0.
    return {"train": train, "test": test, "template": template, "intercept": intercept,
            "fitting_baseline": fitting, "evaluation_baseline": evaluation,
            "residual": residual-intercept}


def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None,
              electrode_ids=None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 18) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),)
            or ids.shape != (len(y),) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or electrode_ids is None or len(electrode_ids) != x.shape[-1]):
        raise ValueError("Complete original history, XY, trial and physical support required")
    blocks, current = history_blocks(x), x[:, 3:18, :]
    original_baseline, baseline = np.full_like(y, np.nan), np.full_like(y, np.nan)
    predictions = np.full((len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        split = prepare_fold(blocks, current, y, labels, held)
        time = elapsed_baseline(y, labels, held)
        train, test, state = split["train"], split["test"], split["history_state"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        old_fit_base = split["template"].predict(labels[train]*np.pi/4)+split["intercept"]
        original_baseline[test], baseline[test] = base, time["evaluation_baseline"]
        fitting, evaluation = split["fitting_history"], split["evaluation_history"]
        flat = fitting.reshape(-1, fitting.shape[-1])
        gram = flat.T@flat/len(flat)
        statistics, weight_state = split["repeatability"], split["weight_state"]
        weights, weight_info = history_trace_weights(statistics["a"], gram)
        repeated = np.tile(weights, 4)
        weighted_fit, weighted_test = fitting*repeated, evaluation*repeated
        cross = weighted_fit.reshape(-1, weighted_fit.shape[-1]).T@time["residual"]
        fitted = ridge_predictions(weighted_fit, weighted_test, time["residual"], check)
        for index in range(len(LAMBDAS)):
            predictions[index, test] = time["evaluation_baseline"]+fitted[index].reshape(base.shape)
        metadata.append({"held_direction": held, "fitting_bin_rows": len(time["residual"]),
            "physical_mean": weight_state["physical_mean"].tolist(),
            "physical_sd": weight_state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(weight_state["physical_sd"] == 0).tolist(),
            "history_physical_mean": state.feature_mean.tolist(),
            "history_physical_sd": state.feature_sd.tolist(),
            "history_zero_SD_coordinates": np.argwhere(state.feature_sd == 0).tolist(),
            "history_fitting_column_mean": split["history_center"].tolist(),
            "repeatability": {k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in statistics.items()},
            "paired_weights": weight_info,
            "elapsed_template_coefficients": time["template"].coefficients.tolist(),
            "elapsed_common_residual_intercept": time["intercept"].tolist(),
            "original_common_residual_intercept": split["intercept"].tolist(),
            "exact_elapsed_equals_original_baseline_on_fitting_and_evaluation": bool(
                np.array_equal(old_fit_base, time["fitting_baseline"])
                and np.array_equal(base, time["evaluation_baseline"])),
            "exact_zero_weighted_design_response_cross_product": bool(np.all(cross == 0))})
    bank = score_bank(y, labels, baseline, predictions, RECIPE)
    bank.update(physical_electrode_ids=np.asarray(electrode_ids).tolist(),
        physical_electrode_count=x.shape[-1], history_column_count=4*x.shape[-1],
        lag_offsets=[0, -1, -2, -3], native_XY_scaled=False, weights_use_velocity=False,
        elapsed_harmonics=0, baseline_direction_weighting="equal seven fitting directions",
        positive_floor=.1, post_weight_standardization=False, post_weight_centering=False,
        direction_fit_metadata=metadata,
        structural_original_HW_duplicate=all(d["exact_elapsed_equals_original_baseline_on_fitting_and_evaluation"]
                                            for d in metadata),
        structural_time_only_duplicate=all(d["exact_zero_weighted_design_response_cross_product"]
                                          for d in metadata))
    return bank, baseline, original_baseline, predictions


def full_model_contrast(candidate, anchor, denominator):
    """Compare full prediction loss; the two recorded delta_R2 baselines may differ."""
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Finite positive common original denominator required")
    return {"delta_R2": (anchor["candidate_SSE"]-candidate["candidate_SSE"])/denominator,
            "identity_increment": candidate["identity_increment"]-anchor["identity_increment"]}


def omissions_by_animal(rows):
    result = {}
    for animal in ("Mihili", "Chewie"):
        original = [r for r in rows if r["animal"] == animal]
        if len(original) != (6 if animal == "Mihili" else 10):
            raise ValueError("All six Mihili and ten Chewie sessions required")
        result[animal] = [{"omitted_session": omit["session_id"],
            "primary_increment": float(np.mean([r["delta_R2"] for r in original if r is not omit])),
            "identity_increment": float(np.mean([r["identity_increment"] for r in original if r is not omit]))}
            for omit in original]
    return result


def panel(rows):
    return {**aggregate_rows(rows), "leave_one_session_within_animal": omissions_by_animal(rows)}


def elapsed_screens(summary, *, require_identity):
    counts = summary["primary_positive_session_counts"]
    screens = {"both_primary_animals_positive": all(v > 0 for v in summary["primary_by_animal"].values()),
        "strict_primary_session_majorities": counts["Mihili"] >= 4 and counts["Chewie"] >= 6,
        "every_within_animal_primary_omission_positive": all(r["primary_increment"] > 0
            for rows in summary["leave_one_session_within_animal"].values() for r in rows)}
    if require_identity:
        screens.update(both_identity_animals_positive=all(v > 0 for v in summary["identity_by_animal"].values()),
            every_within_animal_identity_omission_positive=all(r["identity_increment"] > 0
                for rows in summary["leave_one_session_within_animal"].values() for r in rows))
    return screens


def summarize_banks(sessions, anchor_banks, anchor_panel):
    if (set(sessions) != set(SESSIONS) or set(anchor_banks) != set(SESSIONS)
            or anchor_panel["recipe"] != ANCHOR or len(anchor_panel["sessions"]) != 16
            or {r["session_id"] for r in anchor_panel["sessions"]} != set(SESSIONS)):
        raise ValueError("Complete original sixteen-session sensitivity and HW anchor required")
    original_selected = {r["session_id"]: r for r in anchor_panel["sessions"]}
    for session in SESSIONS:
        bank, old = sessions[session], anchor_banks[session]
        if (bank["recipe"] != RECIPE or old["recipe"] != ANCHOR
                or bank["lambdas"] != list(LAMBDAS) or old["lambdas"] != list(LAMBDAS)
                or bank["denominator"] <= 0 or bank["denominator"] != old["denominator"]
                or bank["direction_counts"] != old["direction_counts"]
                or bank["original_baseline_SSE"] != old["baseline_SSE"]
                or bank["original_HW_saved_prediction_rows"] != old["lambda_rows"]):
            raise ValueError("Original support, baseline and saved prediction rows must replay")
        for item in (bank, old):
            if ([r["lambda"] for r in item["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(r[k]) for r in item["lambda_rows"]
                        for k in ("delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE"))):
                raise ValueError("All five complete finite original penalty rows required")
        saved = original_selected[session]
        if (saved["lambda_index"] not in range(5)
                or any(saved[k] != old["lambda_rows"][saved["lambda_index"]][k]
                       for k in old["lambda_rows"][saved["lambda_index"]])):
            raise ValueError("Original independently selected HW rows must be preserved")
    chosen_rows, against_original, hw_time, b_time = [], [], [], []
    contrasts = {"HW_independently_selected": [], "HW_ET_HW_lambda_matched": []}
    for held in SESSIONS:
        chosen, training, scores = select_paired_lambda(sessions, held)
        bank, old = sessions[held], anchor_banks[held]
        selected, independent = bank["lambda_rows"][chosen], original_selected[held]
        label = {"session_id": held, "animal": ANIMALS[held]}
        chosen_rows.append({**label, "lambda_index": chosen, "outer_training_sessions": training,
            "paired_inner_animal_balanced_scores": scores, "baseline_R2": bank["baseline_R2"], **selected})
        against_original.append({**label, **selected,
            "delta_R2": (old["baseline_SSE"]-selected["candidate_SSE"])/bank["denominator"]})
        hw_time.append({**label, **{k: independent[k] for k in old["lambda_rows"][0]},
            "delta_R2": (bank["baseline_SSE"]-independent["candidate_SSE"])/bank["denominator"]})
        b_time.append({**label,
            "delta_R2": (bank["baseline_SSE"]-old["baseline_SSE"])/bank["denominator"],
            "identity_increment": 0.0})
        for name, anchor in (("HW_independently_selected", independent),
                             ("HW_ET_HW_lambda_matched", old["lambda_rows"][chosen])):
            contrasts[name].append({**label, "ET_HW_lambda": selected["lambda"],
                "HW_lambda": anchor["lambda"], **full_model_contrast(selected, anchor, bank["denominator"])})
    sensitivity, original_time = panel(chosen_rows), panel(hw_time)
    sensitivity_screens = elapsed_screens(sensitivity, require_identity=True)
    original_screens = elapsed_screens(original_time, require_identity=False)
    duplicate_hw = all(sessions[s]["structural_original_HW_duplicate"] for s in SESSIONS)
    duplicate_time = all(sessions[s]["structural_time_only_duplicate"] for s in SESSIONS)
    charge = int(not duplicate_hw and not duplicate_time)
    return {"recipe": RECIPE, "elapsed_sensitivity": sensitivity,
        "original_HW_versus_elapsed_baseline": original_time,
        "original_B_versus_elapsed_baseline": panel(b_time),
        "ET_HW_versus_original_B_descriptive": panel(against_original),
        "original_HW_versus_original_B_replayed": panel([original_selected[s] for s in SESSIONS]),
        "full_model_anchor_contrasts": {name: panel(rows) for name, rows in contrasts.items()},
        "elapsed_sensitivity_screens": sensitivity_screens,
        "all_elapsed_sensitivity_screens_pass": all(sensitivity_screens.values()),
        "original_HW_elapsed_baseline_screens": original_screens,
        "all_original_HW_elapsed_baseline_screens_pass": all(original_screens.values()),
        "structural_original_HW_duplicate": duplicate_hw, "structural_time_only_duplicate": duplicate_time,
        "scientific_trial_count_increment": charge, "alternative_explanation_stress_count_increment": charge,
        "null_control_trial_count_increment": 0, "updates_primary_incumbent": False,
        "sensitivity_only": True, "promotion_authorized": False, "search_complete": False,
        "evidence": "exposed exploratory development; changed-baseline nonpromotable sensitivity"}
