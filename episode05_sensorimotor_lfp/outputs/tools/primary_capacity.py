"""Two adopted EP05 capacity references; pure fitting arrays and fixed metrics."""
from dataclasses import dataclass
import random

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

from method_components import DirectionTime, r2_sse
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, animal_balanced, derangement_count, wrong_pair_sse

RECIPES = ("lmp_rank1_ridge_v1", "lmp_nystrom_rbf105_v1")
LANDMARK_JITTER = 1e-8
KERNEL_CHUNK = 1024


def coordinates(values, mean, raw_sd, *, rbf):
    scale = np.where(raw_sd > 0, raw_sd, 1.)
    result = (np.asarray(values)-mean)/scale
    if rbf:
        result[..., raw_sd == 0] = 0.
    return result


def rank_one_coefficients(z, residual, penalty):
    coefficient = np.linalg.solve(z.T@z+len(z)*penalty*np.eye(z.shape[1]), z.T@residual)
    metric = residual.T@z@coefficient
    metric = (metric+metric.T)/2
    if not np.isfinite(metric).all():
        raise ArithmeticError("Nonfinite regularized response metric")
    if np.all(metric == 0):
        return np.zeros_like(coefficient)
    values, vectors = np.linalg.eigh(metric)
    # Exact two-dimensional isotropic tie has every axis as an optimum.
    vector = np.array([1., 0.]) if values[0] == values[1] else vectors[:, -1]
    return (coefficient@vector)[:, None]*vector[None, :]


def landmark_rows(trial_ids, directions, session_id, held_direction):
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    legal = [d for d in range(8) if d != held_direction]
    if len(set(ids.tolist())) != len(ids) or set(labels.tolist()) != set(legal):
        raise ValueError("Unique original fitting trial identities and exactly seven directions required")
    chosen = []
    for direction in legal:
        available = sorted(np.flatnonzero(labels == direction), key=lambda i: str(ids[i]))
        rng = random.Random()
        rng.seed(f"ep05-rbf-landmark-v1|20261003|{session_id}|{held_direction}|{direction}", version=2)
        chosen.append(int(available[rng.randrange(len(available))]))
    return np.array([15*i+t for i in chosen for t in range(15)]), [str(ids[i]) for i in chosen]


def rbf_kernel(left, right, sigma_squared):
    return np.exp(-cdist(left, right, metric="sqeuclidean")/(2*sigma_squared))


@dataclass
class NystromState:
    landmarks: np.ndarray
    sigma_squared: float
    cholesky: object
    feature_mean: np.ndarray


def raw_nystrom(z, state):
    output = np.zeros((len(z), len(state.landmarks)), dtype=float)
    if state.sigma_squared == 0:
        return output
    for start in range(0, len(z), KERNEL_CHUNK):
        kernel = rbf_kernel(z[start:start+KERNEL_CHUNK], state.landmarks, state.sigma_squared)
        output[start:start+len(kernel)] = solve_triangular(state.cholesky, kernel.T, lower=True).T
    return output


def fit_nystrom(z, rows, active_columns):
    landmarks = np.array(z[rows], copy=True)
    factor = None
    if active_columns:
        kernel = rbf_kernel(landmarks, landmarks, active_columns)
        factor = np.linalg.cholesky((kernel+kernel.T)/2 + LANDMARK_JITTER*np.eye(len(rows)))
    state = NystromState(landmarks, float(active_columns), factor, np.zeros(len(rows)))
    features = raw_nystrom(z, state)
    state.feature_mean = features.mean(axis=0)
    return features-state.feature_mean, state


def prepare_split(x, y, labels, ids, session_id, held):
    train, test = labels != held, labels == held
    angles = labels*np.pi/4
    template = DirectionTime.fit(angles[train], y[train], 1, labels[train])
    residual = (y[train]-template.predict(angles[train])).reshape(-1, 2)
    intercept = residual.mean(axis=0)
    flat = x[train].reshape(-1, x.shape[-1])
    mean, sd = flat.mean(axis=0), flat.std(axis=0)
    z = coordinates(flat, mean, sd, rbf=False)
    kernel_z = coordinates(flat, mean, sd, rbf=True)
    rows, chosen_ids = landmark_rows(ids[train], labels[train], session_id, held)
    phi, state = fit_nystrom(kernel_z, rows, int(np.count_nonzero(sd > 0)))
    return {"test": test, "template": template, "intercept": intercept,
            "feature_mean": mean, "feature_sd": sd, "z": z, "residual": residual-intercept,
            "phi": phi, "kernel_state": state, "landmark_trial_ids": chosen_ids}


def score_bank(y, labels, baseline, predictions, recipe):
    denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
    baseline_r2, baseline_sse = r2_sse(y, baseline), float(np.square(y-baseline).sum())
    rows = []
    for penalty, prediction in zip(LAMBDAS, predictions):
        r2, sse = r2_sse(y, prediction), float(np.square(y-prediction).sum())
        wrong = wrong_pair_sse(y, prediction, labels)
        rows.append({"lambda": penalty, "candidate_R2": r2, "candidate_SSE": sse,
                     "wrong_pair_SSE": wrong, "delta_R2": (baseline_sse-sse)/denominator,
                     "identity_increment": (wrong-sse)/denominator})
    counts = [int(np.count_nonzero(labels == d)) for d in range(8)]
    return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": baseline_r2,
            "baseline_SSE": baseline_sse, "denominator": denominator, "direction_counts": counts,
            "lambda_rows": rows, "ridge_fits": 40, "shared_baseline_fits": 8,
            "derangement_counts_decimal_by_direction": [str(derangement_count(n)) for n in counts],
            "wrong_pair_mean": "exact finite whole-trajectory direction-wise derangement mean", "Monte_Carlo_draws": 0}


def fit_banks(features, targets, directions, trial_ids, session_id, check=lambda: None):
    x, y, labels, ids = np.asarray(features, dtype=float), np.asarray(targets, dtype=float), np.asarray(directions), np.asarray(trial_ids)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[2] < 1 or y.shape != (len(y), 15, 2)
            or labels.shape != (len(y),) or ids.shape != (len(y),) or len(set(ids.tolist())) != len(y)
            or set(labels.tolist()) != set(range(8)) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original trial/window/electrode/direction support required")
    baseline = np.full_like(y, np.nan)
    predictions = {r: np.full((len(LAMBDAS), *y.shape), np.nan) for r in RECIPES}
    fit_metadata = []
    for held in range(8):
        check()
        split = prepare_split(x, y, labels, ids, session_id, held)
        test, sd, mean = split["test"], split["feature_sd"], split["feature_mean"]
        base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
        baseline[test] = base
        linear_test = coordinates(x[test], mean, sd, rbf=False)
        kernel_test = coordinates(x[test].reshape(-1, x.shape[-1]), mean, sd, rbf=True)
        state = split["kernel_state"]
        phi_test = raw_nystrom(kernel_test, state)-state.feature_mean
        phi, residual = split["phi"], split["residual"]
        gram, rhs = phi.T@phi, phi.T@residual
        for index, penalty in enumerate(LAMBDAS):
            check()
            coefficient = rank_one_coefficients(split["z"], residual, penalty)
            predictions[RECIPES[0]][index, test] = base+linear_test@coefficient
            weights = np.linalg.solve(gram+len(phi)*penalty*np.eye(phi.shape[1]), rhs)
            predictions[RECIPES[1]][index, test] = base+(phi_test@weights).reshape(base.shape)
        fit_metadata.append({"held_direction": held, "fitting_bin_rows": len(phi),
            "training_zero_SD_columns": np.flatnonzero(sd == 0).tolist(),
            "active_RBF_columns": int(np.count_nonzero(sd > 0)),
            "RBF_sigma_squared": state.sigma_squared, "landmark_trial_ids": split["landmark_trial_ids"],
            "landmark_bins_per_trial": 15, "landmark_count": len(state.landmarks)})
    banks = {recipe: score_bank(y, labels, baseline, predictions[recipe], recipe) for recipe in RECIPES}
    banks[RECIPES[0]].update(rank_constraint=1, projection_metric="regularized response metric R'Z(H^-1 Z'R)")
    banks[RECIPES[1]].update(kernel="explicitly approximate105-landmark RBF", landmark_diagonal_regularizer=LANDMARK_JITTER,
        kernel_feature_centering="training-only; no subsequent feature standardization",
        training_zero_SD_columns="set to zero in both fitting and evaluation kernel coordinates; no columns removed")
    for bank in banks.values():
        bank["direction_fit_metadata"] = fit_metadata
    return banks, baseline, predictions


def summarize_banks(banks, recipe):
    if recipe not in RECIPES or set(banks) != set(SESSIONS):
        raise ValueError("All sixteen fixed sessions of one capacity recipe required")
    for session, bank in banks.items():
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(r["delta_R2"]) for r in bank["lambda_rows"])):
            raise ValueError(f"Invalid capacity bank: {session}")
    rows = []
    for held in SESSIONS:
        training = [s for s in SESSIONS if s != held]
        scores = [animal_balanced({s: banks[s]["lambda_rows"][i]["delta_R2"] for s in training})[0] for i in range(len(LAMBDAS))]
        chosen = max(range(len(LAMBDAS)), key=lambda i: (scores[i], LAMBDAS[i]))
        rows.append({"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
            "outer_training_sessions": training, "inner_animal_balanced_scores": scores,
            "baseline_R2": banks[held]["baseline_R2"], **banks[held]["lambda_rows"][chosen]})
    primary, primary_animals = animal_balanced({r["session_id"]: r["delta_R2"] for r in rows})
    identity, identity_animals = animal_balanced({r["session_id"]: r["identity_increment"] for r in rows})
    majority = {a: sum(r["delta_R2"] > 0 for r in rows if r["animal"] == a) for a in ("Mihili", "Chewie")}
    screens = {"primary_at_least_0p01": primary >= .01, "identity_at_least_0p005": identity >= .005,
        "both_primary_animals_positive": all(v > 0 for v in primary_animals.values()),
        "both_identity_animals_positive": all(v > 0 for v in identity_animals.values()),
        "strict_session_majorities": majority["Mihili"] >= 4 and majority["Chewie"] >= 6}
    influence = [{"excluded_session": s,
        "primary_increment": animal_balanced({r["session_id"]: r["delta_R2"] for r in rows if r["session_id"] != s})[0],
        "identity_increment": animal_balanced({r["session_id"]: r["identity_increment"] for r in rows if r["session_id"] != s})[0]}
        for s in SESSIONS]
    return {"recipe": recipe, "sessions": rows, "primary_increment": primary, "identity_increment": identity,
        "primary_by_animal": primary_animals, "identity_by_animal": identity_animals,
        "primary_positive_session_counts": majority, "reference_margin_screens": screens,
        "all_reference_margin_screens_pass": all(screens.values()),
        "leave_one_session_fixed_prediction_influence": influence, "scientific_trial_count_increment": 1,
        "promotion_authorized": False, "search_complete": False, "evidence": "exposed exploratory development"}
