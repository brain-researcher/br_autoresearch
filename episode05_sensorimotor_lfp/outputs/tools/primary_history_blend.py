"""One prospectively selected convex mixture of preserved W and HW predictions."""
import numpy as np

from method_components import r2_sse
from primary_elapsed_only import full_model_contrast, panel
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, animal_balanced, wrong_pair_sse
from primary_temporal import directional_screens, select_paired_lambda

RECIPE = "lmp_current_history_repeatability_convex_blend_v1"
COMPONENTS = {"W": "lmp_repeatability_weight_ridge_v1",
              "HW": "lmp_history4_repeatability_weight_ridge_v1"}
ALPHAS = (0., .25, .5, .75, 1.)
SCORE_KEYS = ("candidate_R2", "candidate_SSE", "wrong_pair_SSE", "delta_R2", "identity_increment")


def blend_prediction(w, hw, alpha):
    if (alpha not in ALPHAS or w.shape != hw.shape or w.ndim != 3 or w.shape[1:] != (15, 2)
            or not np.isfinite(w).all() or not np.isfinite(hw).all()):
        raise ValueError("Finite complete native XY endpoints and prescribed mixture weight required")
    if alpha == 0:
        return w
    if alpha == 1:
        return hw
    return (1-alpha)*w+alpha*hw


def prediction_scores(y, baseline, labels, prediction):
    denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Finite positive original native-XY denominator required")
    sse, baseline_sse = float(np.square(y-prediction).sum()), float(np.square(y-baseline).sum())
    wrong = wrong_pair_sse(y, prediction, labels)
    return {"candidate_R2": r2_sse(y, prediction), "candidate_SSE": sse,
            "wrong_pair_SSE": wrong, "delta_R2": (baseline_sse-sse)/denominator,
            "identity_increment": (wrong-sse)/denominator}


def score_session(y, baseline, labels, w_predictions, hw_predictions, check=lambda: None):
    y, baseline, labels = np.asarray(y), np.asarray(baseline), np.asarray(labels)
    expected = (len(LAMBDAS), *y.shape)
    if (y.ndim != 3 or y.shape[1:] != (15, 2) or baseline.shape != y.shape
            or labels.shape != (len(y),) or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))
            or w_predictions.shape != expected or hw_predictions.shape != expected
            or not all(np.isfinite(a).all() for a in (y, baseline, w_predictions, hw_predictions))):
        raise ValueError("Original complete target/baseline and both five-penalty prediction banks required")
    cells = []
    for lambda_index, penalty in enumerate(LAMBDAS):
        for alpha_index, alpha in enumerate(ALPHAS):
            check()
            prediction = blend_prediction(w_predictions[lambda_index], hw_predictions[lambda_index], alpha)
            cells.append({"lambda_index": lambda_index, "alpha_index": alpha_index,
                "lambda": penalty, "alpha": alpha, **prediction_scores(y, baseline, labels, prediction),
                "literal_endpoint_reused": "W" if alpha == 0 else "HW" if alpha == 1 else None,
                "exact_prediction_equals_components": {
                    name: [bool(np.array_equal(prediction, other)) for other in values]
                    for name, values in (("W", w_predictions), ("HW", hw_predictions))}})
    return {"recipe": RECIPE, "lambdas": list(LAMBDAS), "alphas": list(ALPHAS), "cells": cells,
        "baseline_R2": r2_sse(y, baseline), "baseline_SSE": float(np.square(y-baseline).sum()),
        "denominator": float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum()),
        "direction_counts": [int(np.count_nonzero(labels == d)) for d in range(8)],
        "neural_refits": 0, "native_XY_scaled": False,
        "prediction_rule": "literal endpoint reuse; interior (1-alpha)*P_W(lambda)+alpha*P_HW(lambda)",
        "wrong_pair_mean": "exact within-direction whole-trajectory derangement mean"}


def validate_bank(bank):
    if (bank["recipe"] != RECIPE or bank["lambdas"] != list(LAMBDAS) or bank["alphas"] != list(ALPHAS)
            or len(bank["cells"]) != 25 or bank["neural_refits"] != 0 or bank["native_XY_scaled"] is not False
            or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0
            or not np.isfinite(bank["baseline_SSE"]) or not np.isfinite(bank["baseline_R2"])):
        raise ValueError("One complete original-support 25-cell blend bank required")
    for index, cell in enumerate(bank["cells"]):
        li, ai = divmod(index, len(ALPHAS))
        if (cell["lambda_index"] != li or cell["alpha_index"] != ai
                or cell["lambda"] != LAMBDAS[li] or cell["alpha"] != ALPHAS[ai]
                or not all(np.isfinite(cell[key]) for key in SCORE_KEYS)
                or cell["literal_endpoint_reused"] != ("W" if ai == 0 else "HW" if ai == 4 else None)
                or set(cell["exact_prediction_equals_components"]) != set(COMPONENTS)):
            raise ValueError("Ordered finite prescribed parameter cells and literal endpoints required")
        for flags in cell["exact_prediction_equals_components"].values():
            if len(flags) != 5 or any(type(value) is not bool for value in flags):
                raise ValueError("Exact equality to all five preserved endpoints must be explicit")
        if ai in (0, 4) and not cell["exact_prediction_equals_components"]["W" if ai == 0 else "HW"][li]:
            raise ValueError("Literal endpoint identity did not replay")


def select_cell(sessions, held):
    if set(sessions) != set(SESSIONS) or held not in SESSIONS:
        raise ValueError("All original sessions required for outer selection")
    training = [session for session in SESSIONS if session != held]
    scores = [animal_balanced({s: sessions[s]["cells"][index]["delta_R2"] for s in training})[0]
              for index in range(25)]
    chosen = max(range(25), key=lambda index: (scores[index], LAMBDAS[index//5], -ALPHAS[index % 5]))
    return chosen, training, scores


def validate_original_selections(anchor_banks, anchor_panel):
    rows = anchor_panel["sessions"]
    if (set(anchor_banks) != set(SESSIONS) or len(rows) != 16
            or {r["session_id"] for r in rows} != set(SESSIONS)):
        raise ValueError("Complete preserved original endpoint selection required")
    by_session = {row["session_id"]: row for row in rows}
    for session in SESSIONS:
        chosen, training, _ = select_paired_lambda(anchor_banks, session)
        row = by_session[session]
        if (row["animal"] != ANIMALS[session] or row["lambda_index"] != chosen
                or row["outer_training_sessions"] != training
                or any(row[key] != anchor_banks[session]["lambda_rows"][chosen][key]
                       for key in ("lambda", *SCORE_KEYS))):
            raise ValueError("Original other-fifteen-session selector or selected endpoint row changed")
    return by_session


def summarize_banks(sessions, anchor_banks, anchor_panels):
    if set(sessions) != set(SESSIONS) or set(anchor_banks) != set(COMPONENTS) or set(anchor_panels) != set(COMPONENTS):
        raise ValueError("All sixteen blend banks and both preserved original component families required")
    originals = {}
    for name, recipe in COMPONENTS.items():
        if anchor_panels[name]["recipe"] != recipe:
            raise ValueError("Original endpoint recipe changed")
        originals[name] = validate_original_selections(anchor_banks[name], anchor_panels[name])
    for session, bank in sessions.items():
        validate_bank(bank)
        for name in COMPONENTS:
            old = anchor_banks[name][session]
            if (old["recipe"] != COMPONENTS[name] or old["lambdas"] != list(LAMBDAS)
                    or [row["lambda"] for row in old["lambda_rows"]] != list(LAMBDAS)
                    or any(bank[key] != old[key] for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"))):
                raise ValueError("Both endpoints must use the exact original common target scoring support")
    selected, contrasts = [], {name+"_"+kind: [] for name in COMPONENTS
                               for kind in ("independently_selected", "blend_lambda_matched")}
    for session in SESSIONS:
        index, training, scores = select_cell(sessions, session)
        cell, bank = sessions[session]["cells"][index], sessions[session]
        identity = {"session_id": session, "animal": ANIMALS[session]}
        selected.append({**identity, **cell, "cell_index": index,
            "outer_training_sessions": training, "training_primary_scores": scores,
            "baseline_R2": bank["baseline_R2"],
            "exactly_matches_original_selected_components": {
                name: cell["exact_prediction_equals_components"][name][originals[name][session]["lambda_index"]]
                for name in COMPONENTS}})
        for name in COMPONENTS:
            for kind, reference in (("independently_selected", originals[name][session]),
                                    ("blend_lambda_matched", anchor_banks[name][session]["lambda_rows"][cell["lambda_index"]])):
                contrasts[name+"_"+kind].append({**identity, "lambda": cell["lambda"], "alpha": cell["alpha"],
                    "reference_lambda": reference["lambda"], **full_model_contrast(cell, reference, bank["denominator"])})
    duplicates = [name for name in COMPONENTS if all(row["exactly_matches_original_selected_components"][name] for row in selected)]
    charge = int(not duplicates)
    candidate = panel(selected)
    screens = {"primary_at_least_0p01": candidate["primary_increment"] >= .01,
               "identity_at_least_0p005": candidate["identity_increment"] >= .005, **directional_screens(candidate),
               "every_within_animal_primary_omission_positive": all(row["primary_increment"] > 0
                   for rows in candidate["leave_one_session_within_animal"].values() for row in rows),
               "every_within_animal_identity_omission_positive": all(row["identity_increment"] > 0
                   for rows in candidate["leave_one_session_within_animal"].values() for row in rows)}
    return {"recipe": RECIPE, "blend": candidate,
        "reference_contrasts": {name: panel(rows) for name, rows in contrasts.items()},
        "original_components": {name: panel([originals[name][s] for s in SESSIONS]) for name in COMPONENTS},
        "reference_margin_screens": screens, "all_reference_margin_screens_pass": all(screens.values()),
        "structural_duplicate_of": duplicates, "scientific_trial_count_increment": charge,
        "literal_null_control_count_increment": 0, "alternative_explanation_count_increment": 0,
        "eligible_for_comparable_primary_incumbent": bool(charge), "neural_refits": 0,
        "promotion_authorized": False, "search_complete": False,
        "evidence": "exposed original-support development; outer-session-selected two-component convex mixture"}
