"""One finite linear time schedule over the preserved original W/HW predictions."""
import numpy as np

from method_components import r2_sse
from primary_elapsed_only import full_model_contrast, panel
from primary_history_blend import (COMPONENTS, SCORE_KEYS, blend_prediction, prediction_scores,
    select_cell as select_constant_cell, validate_bank as validate_constant_bank,
    validate_original_selections)
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, animal_balanced
from primary_temporal import directional_screens

RECIPE = "lmp_current_history_repeatability_linear_time_blend_v1"
SLOPES = (-.5, -.25, 0., .25, .5)
ANCHORS = {**COMPONENTS, "blend": "lmp_current_history_repeatability_convex_blend_v1",
           "H2": "lmp_history2_repeatability_weight_ridge_v1"}
SUPPORT_KEYS = ("baseline_R2", "baseline_SSE", "denominator", "direction_counts")


def alpha_schedule(slope):
    if slope not in SLOPES:
        raise ValueError("Only the five adopted slopes are permitted")
    return .5+slope*(2*np.arange(15, dtype=np.float64)/14-1)


def timevarying_prediction(w, hw, slope):
    alpha = alpha_schedule(slope)
    if (w.shape != hw.shape or w.ndim != 3 or w.shape[1:] != (15, 2)
            or w.dtype != np.dtype("float64") or hw.dtype != np.dtype("float64")
            or not np.isfinite(w).all() or not np.isfinite(hw).all()):
        raise ValueError("Finite native float64 complete XY components required")
    if slope == 0:
        return blend_prediction(w, hw, .5)
    prediction = np.empty(w.shape, dtype=np.float64)
    for t, weight in enumerate(alpha):
        if weight == 0:
            prediction[:, t] = w[:, t]
        elif weight == 1:
            prediction[:, t] = hw[:, t]
        else:
            prediction[:, t] = (1-weight)*w[:, t]+weight*hw[:, t]
    return prediction


def score_session(y, baseline, labels, w_predictions, hw_predictions, original_predictions, check=lambda: None):
    expected = (len(LAMBDAS), *y.shape)
    if (y.ndim != 3 or y.shape[1:] != (15, 2) or baseline.shape != y.shape
            or labels.shape != (len(y),) or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or w_predictions.shape != expected or hw_predictions.shape != expected
            or set(original_predictions) != set(ANCHORS)
            or any(value.shape != y.shape for value in original_predictions.values())
            or any(value.dtype != np.dtype("float64") or not np.isfinite(value).all()
                   for value in (y, baseline, w_predictions, hw_predictions, *original_predictions.values()))):
        raise ValueError("Complete original scoring support and four fixed original predictions required")
    cells = []
    for li, penalty in enumerate(LAMBDAS):
        for si, slope in enumerate(SLOPES):
            check()
            prediction = timevarying_prediction(w_predictions[li], hw_predictions[li], slope)
            cells.append({"lambda_index": li, "slope_index": si, "lambda": penalty, "slope": slope,
                "alpha_by_bin": alpha_schedule(slope).tolist(), **prediction_scores(y, baseline, labels, prediction),
                "exact_prediction_equals_originals": {name: bool(np.array_equal(prediction, original))
                    for name, original in original_predictions.items()}})
    return {"recipe": RECIPE, "lambdas": list(LAMBDAS), "slopes": list(SLOPES), "cells": cells,
        "baseline_R2": r2_sse(y, baseline), "baseline_SSE": float(np.square(y-baseline).sum()),
        "denominator": float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum()),
        "direction_counts": [int(np.count_nonzero(labels == d)) for d in range(8)],
        "neural_refits": 0, "native_XY_scaled": False,
        "wrong_pair_mean": "exact within-direction whole-trajectory derangement mean",
        "mean_coefficient_is_equal_capacity_claim": False}


def validate_bank(bank):
    if (bank["recipe"] != RECIPE or bank["lambdas"] != list(LAMBDAS) or bank["slopes"] != list(SLOPES)
            or len(bank["cells"]) != 25 or bank["neural_refits"] != 0 or bank["native_XY_scaled"] is not False
            or not all(np.isfinite(bank[key]) for key in SUPPORT_KEYS[:3]) or bank["denominator"] <= 0
            or len(bank["direction_counts"]) != 8 or any(type(n) is not int or n < 2 for n in bank["direction_counts"])):
        raise ValueError("Complete native-support finite 25-cell time schedule bank required")
    for index, cell in enumerate(bank["cells"]):
        li, si = divmod(index, 5)
        if (cell["lambda_index"] != li or cell["slope_index"] != si or cell["lambda"] != LAMBDAS[li]
                or cell["slope"] != SLOPES[si] or cell["alpha_by_bin"] != alpha_schedule(SLOPES[si]).tolist()
                or not all(np.isfinite(cell[key]) for key in SCORE_KEYS)
                or set(cell["exact_prediction_equals_originals"]) != set(ANCHORS)
                or any(type(flag) is not bool for flag in cell["exact_prediction_equals_originals"].values())):
            raise ValueError("Ordered prescribed cells and literal four-anchor equality evidence required")


def select_cell(sessions, held):
    if set(sessions) != set(SESSIONS) or held not in SESSIONS:
        raise ValueError("All sixteen original sessions required")
    training = [session for session in SESSIONS if session != held]
    scores = [animal_balanced({s: sessions[s]["cells"][i]["delta_R2"] for s in training})[0] for i in range(25)]
    chosen = max(range(25), key=lambda i: (scores[i], LAMBDAS[i//5], -abs(SLOPES[i % 5]), -SLOPES[i % 5]))
    return chosen, training, scores


def validate_anchor_selections(anchor_banks, anchor_panels, blend_banks):
    if set(anchor_banks) != {"W", "HW", "H2"} or set(anchor_panels) != set(ANCHORS) or set(blend_banks) != set(SESSIONS):
        raise ValueError("All four complete original anchor selections required")
    originals = {}
    for name, banks in anchor_banks.items():
        if anchor_panels[name]["recipe"] != ANCHORS[name]:
            raise ValueError("Original anchor recipe changed")
        originals[name] = validate_original_selections(banks, anchor_panels[name])
    old_panel = anchor_panels["blend"]
    if len(old_panel["sessions"]) != 16 or {r["session_id"] for r in old_panel["sessions"]} != set(SESSIONS):
        raise ValueError("Complete original blend selection required")
    originals["blend"] = {row["session_id"]: row for row in old_panel["sessions"]}
    for session in SESSIONS:
        bank = blend_banks[session]
        validate_constant_bank(bank)
        index, training, scores = select_constant_cell(blend_banks, session)
        row, cell = originals["blend"][session], bank["cells"][index]
        if (row["animal"] != ANIMALS[session] or row["cell_index"] != index
                or row["outer_training_sessions"] != training or row["training_primary_scores"] != scores
                or any(row[key] != cell[key] for key in ("lambda_index", "alpha_index", "lambda", "alpha", *SCORE_KEYS))):
            raise ValueError("Exact original constant-blend selector replay failed")
    return originals


def summarize_banks(sessions, anchor_banks, anchor_panels, blend_banks):
    if set(sessions) != set(SESSIONS):
        raise ValueError("Complete sixteen-session time schedule panel required")
    originals = validate_anchor_selections(anchor_banks, anchor_panels, blend_banks)
    selected, reversed_rows, reversal_contrasts = [], [], []
    contrast_keys = [name+"_independently_selected" for name in ANCHORS]
    contrast_keys += ["W_schedule_lambda_matched", "HW_schedule_lambda_matched", "half_blend_schedule_lambda_matched"]
    contrasts = {key: [] for key in contrast_keys}
    for session in SESSIONS:
        bank = sessions[session]
        validate_bank(bank)
        for old in (*[banks[session] for banks in anchor_banks.values()], blend_banks[session]):
            if any(bank[key] != old[key] for key in SUPPORT_KEYS):
                raise ValueError("All anchors must retain exact original baseline, denominator and direction support")
        for li in range(5):
            if any(bank["cells"][5*li+2][key] != blend_banks[session]["cells"][5*li+2][key] for key in SCORE_KEYS):
                raise ValueError("Zero slope must exactly replay each original half-blend cell")
        index, training, scores = select_cell(sessions, session)
        cell = bank["cells"][index]
        identity = {"session_id": session, "animal": ANIMALS[session]}
        selected.append({**identity, **cell, "cell_index": index, "outer_training_sessions": training,
            "training_primary_scores": scores, "baseline_R2": bank["baseline_R2"]})
        references = {name+"_independently_selected": originals[name][session] for name in ANCHORS}
        references.update({name+"_schedule_lambda_matched": anchor_banks[name][session]["lambda_rows"][cell["lambda_index"]]
                           for name in COMPONENTS})
        references["half_blend_schedule_lambda_matched"] = bank["cells"][5*cell["lambda_index"]+2]
        for name, reference in references.items():
            contrasts[name].append({**identity, "lambda": cell["lambda"], "slope": cell["slope"],
                "reference_lambda": reference["lambda"], **full_model_contrast(cell, reference, bank["denominator"])})
        reverse = bank["cells"][5*cell["lambda_index"]+SLOPES.index(-cell["slope"])]
        reversed_rows.append({**identity, **reverse, "baseline_R2": bank["baseline_R2"]})
        reversal_contrasts.append({**identity, "lambda": cell["lambda"], "slope": cell["slope"],
            "reference_slope": reverse["slope"], **full_model_contrast(cell, reverse, bank["denominator"])})
    duplicate = [name for name in ANCHORS if all(row["exact_prediction_equals_originals"][name] for row in selected)]
    candidate = panel(selected)
    screens = {"primary_at_least_0p01": candidate["primary_increment"] >= .01,
        "identity_at_least_0p005": candidate["identity_increment"] >= .005, **directional_screens(candidate),
        "every_within_animal_primary_omission_positive": all(row["primary_increment"] > 0
            for rows in candidate["leave_one_session_within_animal"].values() for row in rows),
        "every_within_animal_identity_omission_positive": all(row["identity_increment"] > 0
            for rows in candidate["leave_one_session_within_animal"].values() for row in rows)}
    return {"recipe": RECIPE, "candidate": candidate,
        "reference_contrasts": {key: panel(rows) for key, rows in contrasts.items()},
        "original_anchors": {name: panel([originals[name][s] for s in SESSIONS]) for name in ANCHORS},
        "reversed_slope": panel(reversed_rows), "selected_minus_reversed": panel(reversal_contrasts),
        "reversal_descriptive_only": True, "reversal_scientific_trial_count_increment": 0,
        "reference_margin_screens": screens, "all_reference_margin_screens_pass": all(screens.values()),
        "structural_duplicate_of": duplicate, "scientific_trial_count_increment": int(not duplicate),
        "literal_null_control_count_increment": 0, "alternative_explanation_count_increment": 0,
        "eligible_for_comparable_primary_incumbent": not bool(duplicate), "neural_refits": 0,
        "promotion_authorized": False, "search_complete": False,
        "evidence": "exposed original-support development; no equal-capacity or physiological-slope claim"}
