"""Adopted full-feature-span EP05 ridge; no source or scheduler access."""
import numpy as np

from method_components import DirectionTime, RidgeResidual, r2_sse
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, animal_balanced, derangement_count, wrong_pair_sse

BLOCKS = {
    "lmp_full_channel_ridge_v1": ((0., 0.),),
    "low_full_channel_ridge_v1": ((.5, 4.), (4., 8.), (8., 12.)),
    "high_full_channel_ridge_v1": ((100., 200.), (200., 400.)),
}


def guide_columns(guide, channels, bands):
    guide = np.asarray(guide)
    if (guide.ndim != 2 or guide.shape[1] != 3 or not np.isfinite(guide).all()
            or not channels or len(set(channels)) != len(channels)):
        raise ValueError("Finite guide and distinct fixed electrodes required")
    columns = []
    for channel in channels:
        for lower, upper in bands:
            found = np.flatnonzero((guide[:, 0] == channel) & (guide[:, 1] == lower) & (guide[:, 2] == upper))
            if len(found) != 1:
                raise ValueError(f"Electrode {channel} lacks unique exact band {(lower, upper)}")
            columns.append(int(found[0]))
    return columns


def ridge_solve(standardized, centered_target, penalty):
    """Full orthonormal PCA-span equivalent; no numerical PCA/whitening/rank cut."""
    z, target = np.asarray(standardized), np.asarray(centered_target)
    return np.linalg.solve(z.T @ z + len(z)*penalty*np.eye(z.shape[1]), z.T @ target)


def lmp_scale_diagnostic(features, stored_dtypes):
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    mean = x.mean(axis=1)
    rms_mean, rms_electrode = float(np.sqrt(np.mean(mean**2))), float(np.sqrt(np.mean(x**2)))
    sd_mean, sd_electrode = float(mean.std()), float(np.sqrt(np.mean(x.var(axis=0))))
    return {"stored_dtypes": sorted(stored_dtypes), "spatial_mean_RMS": rms_mean,
            "all_electrode_RMS": rms_electrode,
            "RMS_ratio": rms_mean/rms_electrode if rms_electrode > 0 else None,
            "spatial_mean_population_SD": sd_mean, "root_mean_electrode_population_variance": sd_electrode,
            "SD_ratio": sd_mean/sd_electrode if sd_electrode > 0 else None,
            "spatial_mean_exactly_zero": bool(np.all(mean == 0)),
            "spatial_mean_exactly_constant": bool(sd_mean == 0),
            "used_for_model_eligibility_or_preprocessing": False,
            "scope": "all fixed selected trials/bins; descriptive amplitude only, no cutoff or behavior association"}


def fit_bank(features, targets, directions, recipe, check=lambda: None):
    x, y, labels = np.asarray(features, dtype=float), np.asarray(targets, dtype=float), np.asarray(directions)
    if (recipe not in BLOCKS or x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[2] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),)
            or set(labels) != set(range(8)) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete fixed finite feature/trial/window/direction support required")
    angles, width = labels*np.pi/4, x.shape[2]
    baseline, predictions = np.full_like(y, np.nan), np.full((len(LAMBDAS), *y.shape), np.nan)
    fitting_rows, constants = [], []
    for direction in range(8):
        check()
        test, train = labels == direction, labels != direction
        template = DirectionTime.fit(angles[train], y[train], 1, labels[train])
        residual = (y[train]-template.predict(angles[train])).reshape(-1, 2)
        flat = x[train].reshape(-1, width)
        mean, scale = flat.mean(axis=0), flat.std(axis=0)
        constants.append(np.flatnonzero(scale == 0).tolist())
        scale = np.where(scale > 0, scale, 1.)
        z, common_intercept = (flat-mean)/scale, residual.mean(axis=0)
        centered = residual-common_intercept
        baseline[test] = template.predict(angles[test])+common_intercept
        fitting_rows.append(len(flat))
        gram, rhs = z.T @ z, z.T @ centered
        for index, penalty in enumerate(LAMBDAS):
            check()
            coefficients = np.linalg.solve(gram+len(flat)*penalty*np.eye(width), rhs)
            model = RidgeResidual(template, mean, scale, common_intercept, coefficients)
            predictions[index, test] = model.predict(x[test], angles[test])
    denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
    baseline_r2, baseline_sse = r2_sse(y, baseline), float(np.square(y-baseline).sum())
    scores = []
    for penalty, prediction in zip(LAMBDAS, predictions):
        r2, sse = r2_sse(y, prediction), float(np.square(y-prediction).sum())
        wrong = wrong_pair_sse(y, prediction, labels)
        scores.append({"lambda": penalty, "candidate_R2": r2, "candidate_SSE": sse,
                       "wrong_pair_SSE": wrong, "delta_R2": (baseline_sse-sse)/denominator,
                       "identity_increment": (wrong-sse)/denominator})
    counts = [int(np.count_nonzero(labels == d)) for d in range(8)]
    return {"recipe": recipe, "lambdas": list(LAMBDAS), "feature_width": width,
            "basis": "direct standardized features; equivalent to complete unwhitened orthonormal PCA span",
            "numerical_PCA_computed": False, "rank_truncation": False, "postrotation_standardization": False,
            "baseline_R2": baseline_r2, "baseline_SSE": baseline_sse, "denominator": denominator,
            "direction_counts": counts, "training_bin_rows_by_held_direction": fitting_rows,
            "constant_feature_columns_by_held_direction": constants,
            "derangement_counts_decimal_by_direction": [str(derangement_count(n)) for n in counts],
            "wrong_pair_mean": "exact finite uniform direction-wise whole-trajectory derangement mean",
            "Monte_Carlo_draws": 0, "baseline_fits": 8, "ridge_fits": 40, "lambda_rows": scores}, baseline, predictions


def summarize_banks(banks, recipe):
    if recipe not in BLOCKS or set(banks) != set(SESSIONS):
        raise ValueError("Exactly sixteen sessions of one adopted recipe required")
    for session, bank in banks.items():
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or [row["lambda"] for row in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(row["delta_R2"]) for row in bank["lambda_rows"])):
            raise ValueError(f"Invalid recipe bank: {session}")
    rows = []
    for held in SESSIONS:
        training = [s for s in SESSIONS if s != held]
        scores = [animal_balanced({s: banks[s]["lambda_rows"][i]["delta_R2"] for s in training})[0]
                  for i in range(len(LAMBDAS))]
        selected = max(range(len(LAMBDAS)), key=lambda i: (scores[i], LAMBDAS[i]))
        rows.append({"session_id": held, "animal": ANIMALS[held], "lambda_index": selected,
                     "outer_training_sessions": training, "inner_animal_balanced_scores": scores,
                     "baseline_R2": banks[held]["baseline_R2"], **banks[held]["lambda_rows"][selected]})
    primary, primary_animals = animal_balanced({r["session_id"]: r["delta_R2"] for r in rows})
    identity, identity_animals = animal_balanced({r["session_id"]: r["identity_increment"] for r in rows})
    majority = {a: sum(r["delta_R2"] > 0 for r in rows if r["animal"] == a) for a in ("Mihili", "Chewie")}
    screens = {"primary_at_least_0p01": primary >= .01, "identity_at_least_0p005": identity >= .005,
               "both_primary_animals_positive": all(v > 0 for v in primary_animals.values()),
               "both_identity_animals_positive": all(v > 0 for v in identity_animals.values()),
               "strict_session_majorities": majority["Mihili"] >= 4 and majority["Chewie"] >= 6}
    influence = [{"excluded_session": excluded,
        "primary_increment": animal_balanced({r["session_id"]: r["delta_R2"] for r in rows if r["session_id"] != excluded})[0],
        "identity_increment": animal_balanced({r["session_id"]: r["identity_increment"] for r in rows if r["session_id"] != excluded})[0]}
        for excluded in SESSIONS]
    return {"recipe": recipe, "sessions": rows, "primary_increment": primary, "identity_increment": identity,
            "primary_by_animal": primary_animals, "identity_by_animal": identity_animals,
            "primary_positive_session_counts": majority, "reference_margin_screens": screens,
            "all_reference_margin_screens_pass": all(screens.values()),
            "leave_one_session_fixed_prediction_influence": influence,
            "scientific_trial_count_increment": 1, "promotion_authorized": False, "search_complete": False,
            "evidence": "exposed exploratory development"}
