"""Adopted EP05 first-family equations; no source or scheduler access."""
from __future__ import annotations

import numpy as np

from method_components import DirectionTime, RidgeResidual, r2_sse
from source_adapter import M1_SESSIONS

RECIPE = "ep05_primary_lmp_mean_v1"
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
OFFSETS = (0, 14)
SESSIONS = tuple(f"{animal}_{day}" for animal, _, day in M1_SESSIONS)
ANIMALS = {f"{animal}_{day}": animal for animal, _, day in M1_SESSIONS}


def derangement_count(count):
    previous, current = 1, 0
    if count == 0:
        return previous
    for size in range(2, count+1):
        previous, current = current, (size-1)*(previous+current)
    return current


def wrong_pair_sse(actual, prediction, directions):
    """Exact uniform whole-trajectory derangement mean within each direction."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if actual.shape != prediction.shape or not np.isfinite([actual, prediction]).all():
        raise ValueError("Complete finite matched trajectory arrays required")
    total = 0.0
    for direction in range(8):
        selected = np.asarray(directions) == direction
        y = actual[selected].reshape(np.count_nonzero(selected), -1)
        p = prediction[selected].reshape(np.count_nonzero(selected), -1)
        count = len(y)
        if count < 2:
            raise ValueError("At least two trials per direction required")
        # A common translation improves stability without changing distances.
        center = y.mean(axis=0)
        y, p = y-center, p-center
        all_pairs = count*(np.square(y).sum()+np.square(p).sum()) - 2*(y.sum(axis=0) @ p.sum(axis=0))
        diagonal = np.square(y-p).sum()
        contribution = float((all_pairs-diagonal)/(count-1))
        if not np.isfinite(contribution) or contribution < 0:
            raise ArithmeticError("Invalid wrong-pair SSE; no clipping")
        total += contribution
    return total


def fit_bank(features, targets, directions, check=lambda: None):
    """Eight whole-direction fits per lambda, with shared fair comparator."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    labels = np.asarray(directions)
    if (x.shape != (len(y), 15, 1) or y.shape != (len(y), 15, 2)
            or labels.shape != (len(y),) or set(labels) != set(range(8))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Fixed finite complete-trial/window/direction support required")
    angles = labels*np.pi/4
    baseline = np.full_like(y, np.nan)
    predictions = np.full((len(LAMBDAS), *y.shape), np.nan)
    fitting_rows, constants = [], []
    for direction in range(8):
        check()
        test, train = labels == direction, labels != direction
        template = DirectionTime.fit(angles[train], y[train], 1, labels[train])
        residual = (y[train]-template.predict(angles[train])).reshape(-1, 2)
        flat = x[train].reshape(-1, 1)
        mean, scale = flat.mean(axis=0), flat.std(axis=0)
        constants.append(bool(scale[0] == 0))
        scale = np.where(scale > 0, scale, 1.0)
        z = (flat-mean)/scale
        common_intercept = residual.mean(axis=0)
        centered = residual-common_intercept
        baseline[test] = template.predict(angles[test])+common_intercept
        gram, rhs = z.T @ z, z.T @ centered
        fitting_rows.append(int(len(flat)))
        for index, penalty in enumerate(LAMBDAS):
            check()
            coefficients = np.linalg.solve(gram+len(flat)*penalty*np.eye(1), rhs)
            model = RidgeResidual(template, mean, scale, common_intercept, coefficients)
            predictions[index, test] = model.predict(x[test], angles[test])
    denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
    baseline_r2 = r2_sse(y, baseline)
    baseline_sse = float(np.square(y-baseline).sum())
    rows = []
    for penalty, prediction in zip(LAMBDAS, predictions):
        candidate_r2 = r2_sse(y, prediction)
        sse = float(np.square(y-prediction).sum())
        wrong = wrong_pair_sse(y, prediction, labels)
        rows.append({"lambda": penalty, "candidate_R2": candidate_r2,
                     "candidate_SSE": sse, "wrong_pair_SSE": wrong,
                     "delta_R2": (baseline_sse-sse)/denominator,
                     "identity_increment": (wrong-sse)/denominator})
    counts = [int(np.count_nonzero(labels == d)) for d in range(8)]
    return {"recipe": RECIPE, "lambdas": list(LAMBDAS), "baseline_R2": baseline_r2,
            "baseline_SSE": baseline_sse, "denominator": denominator,
            "direction_counts": counts,
            "wrong_pair_mean": "exact finite uniform direction-wise whole-trajectory derangement mean",
            "derangement_counts_decimal_by_direction": [str(derangement_count(n)) for n in counts],
            "Monte_Carlo_draws": 0,
            "training_bin_rows_by_held_direction": fitting_rows,
            "constant_LMP_by_held_direction": constants,
            "baseline_fits": 8, "ridge_fits": 40, "lambda_rows": rows}, baseline, predictions


def animal_balanced(values):
    groups = {animal: [value for session, value in values.items() if ANIMALS[session] == animal]
              for animal in ("Mihili", "Chewie")}
    if any(not group for group in groups.values()):
        raise ValueError("Both biological animals required")
    means = {animal: float(np.mean(group)) for animal, group in groups.items()}
    return float(np.mean(list(means.values()))), means


def choose_outer_lambdas(banks):
    if set(banks) != set(SESSIONS):
        raise ValueError("All sixteen fixed sessions required; no partial selection")
    for session, bank in banks.items():
        if (bank["recipe"] != RECIPE or bank["lambdas"] != list(LAMBDAS)
                or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(r["delta_R2"]) for r in bank["lambda_rows"])):
            raise ValueError(f"Fixed finite recipe bank required: {session}")
    selected = {}
    for held in SESSIONS:
        training = [session for session in SESSIONS if session != held]
        scores = [animal_balanced({s: banks[s]["lambda_rows"][index]["delta_R2"] for s in training})[0]
                  for index in range(len(LAMBDAS))]
        index = max(range(len(LAMBDAS)), key=lambda i: (scores[i], LAMBDAS[i]))
        selected[held] = {"lambda_index": index, "lambda": LAMBDAS[index],
                          "outer_training_sessions": training, "inner_animal_balanced_scores": scores}
    return selected


def summarize_banks(banks):
    selection = choose_outer_lambdas(banks)
    rows = []
    for session in SESSIONS:
        choice = selection[session]
        rows.append({"session_id": session, "animal": ANIMALS[session], **choice,
                     **banks[session]["lambda_rows"][choice["lambda_index"]],
                     "baseline_R2": banks[session]["baseline_R2"]})
    primary, primary_animals = animal_balanced({r["session_id"]: r["delta_R2"] for r in rows})
    identity, identity_animals = animal_balanced({r["session_id"]: r["identity_increment"] for r in rows})
    majority = {animal: sum(r["delta_R2"] > 0 for r in rows if r["animal"] == animal)
                for animal in ("Mihili", "Chewie")}
    screens = {"primary_at_least_0p01": primary >= .01, "identity_at_least_0p005": identity >= .005,
               "both_primary_animals_positive": all(x > 0 for x in primary_animals.values()),
               "both_identity_animals_positive": all(x > 0 for x in identity_animals.values()),
               "strict_session_majorities": majority["Mihili"] >= 4 and majority["Chewie"] >= 6}
    influence = []
    for excluded in SESSIONS:
        retained = [r for r in rows if r["session_id"] != excluded]
        influence.append({"excluded_session": excluded,
            "primary_increment": animal_balanced({r["session_id"]: r["delta_R2"] for r in retained})[0],
            "identity_increment": animal_balanced({r["session_id"]: r["identity_increment"] for r in retained})[0]})
    return {"recipe": RECIPE, "status": "complete_first_primary_reference", "sessions": rows,
            "primary_increment": primary, "identity_increment": identity,
            "primary_by_animal": primary_animals, "identity_by_animal": identity_animals,
            "primary_positive_session_counts": majority, "reference_margin_screens": screens,
            "all_reference_margin_screens_pass": all(screens.values()),
            "leave_one_session_fixed_prediction_influence": influence,
            "promotion_authorized": False, "search_complete": False,
            "scientific_trial_count_increment": 1, "evidence": "exposed exploratory development"}
