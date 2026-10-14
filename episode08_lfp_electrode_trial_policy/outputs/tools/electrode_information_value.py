"""Pure fitting-array predictors and complete-session cross-fitting for EP08.

No source reader, evaluation callback, file access or tunable model search lives
here. Conditional value is a fixed-ridge predictive quantity, not mutual information.
"""
from __future__ import annotations

import random
import numpy as np
from scipy.stats import rankdata

from direction_training import residual_statistics, residual_coordinates
from ridge_training import ridge_coefficients

PREFIXES = (32, 64)
BASE_FEATURES = (
    "direction_repeatability", "log_LMP_residual_RMS", "log_HF100_200_residual_RMS",
    "log_HF200_400_residual_RMS", "constant_band_fraction", "marginal_correlation_squared",
    "marginal_CV_gain", "context_redundancy", "log_context_size",
)
CONDITIONAL_FEATURE = "conditional_CV_gain"


def contexts(filename, channels):
    """One full-array context and three balanced, outcome-independent partitions."""
    if len(channels) != 96 or list(channels) != sorted(set(channels)):
        raise ValueError("Preserve the original 96 canonical physical channels")
    result = [{"id": "full96", "indices": list(range(96)), "family": "full96",
               "partition": None, "row_weight": .5/96}]
    for seed in range(3):
        order = list(range(96))
        literal = f"ep08-information-value-context-v1|20261013|{filename}|{seed}"
        rng = random.Random(); rng.seed(literal, version=2); rng.shuffle(order)
        for block in range(6):
            result.append({"id": f"partition{seed}_block{block}",
                           "indices": sorted(order[16*block:16*(block+1)]),
                           "family": "partition16", "partition": seed,
                           "seed_literal": literal, "row_weight": .5/(3*96)})
    return result


def columns(electrodes):
    return np.asarray([3*int(e)+band for e in electrodes for band in range(3)], dtype=int)


def balanced_folds(directions):
    directions = np.asarray(directions)
    if directions.shape not in ((32,), (64,)) or not np.array_equal(
            np.bincount(directions, minlength=8), np.full(8, len(directions)//8)):
        raise ValueError("Balanced 32/64-trial prefixes with eight directions required")
    groups = np.empty(len(directions), dtype=int)
    for direction in range(8):
        positions = np.flatnonzero(directions == direction)
        groups[positions] = np.arange(len(positions)) % 4
    return [(np.flatnonzero(groups != f), np.flatnonzero(groups == f)) for f in range(4)]


def residual_quality(x, y, directions):
    """Recorded-signal proxies only; no claim of calibrated noise/artifact quality."""
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    n = len(x); repeats = n//8
    raw_sd = np.asarray(x).std(0)
    z = (x-np.asarray(x).mean(0))/np.where(raw_sd == 0, 1., raw_sd)
    means = np.stack([z[directions == d].mean(0) for d in range(8)])
    between = repeats*np.sum((means-z.mean(0))**2, axis=0)/7
    within = sum(np.sum((z[directions == d]-means[d])**2, axis=0) for d in range(8))/(n-8)
    component = np.maximum((between-within)/repeats, 0.)
    reliability = np.divide(component, component+within, out=np.zeros_like(component), where=(component+within)>0)
    raw_residual = x-stats.direction_input_mean[directions]
    rms = np.sqrt(np.mean(raw_residual**2, axis=0)).reshape(96, 3)
    ys = yy.std(0); xs = xx.std(0)
    zx = (xx-xx.mean(0))/np.where(xs == 0, 1., xs)
    zy = (yy-yy.mean(0))/np.where(ys == 0, 1., ys)
    corr = zx.T@zy/n
    marginal = np.mean(corr.reshape(96, 3, -1)**2, axis=(1, 2))
    correlation = zx.T@zx/n
    quality = np.column_stack((reliability.reshape(96, 3).mean(1),
                               np.log(np.maximum(rms, 1e-12)), (rms == 0).mean(1), marginal))
    return quality, correlation


def coefficient(x, y, cols, check, progress):
    check(); progress["ridge_fits"] += 1
    return ridge_coefficients(x[:, cols], y, 1.)


def predictor_rows(x, y, directions, channels, context_menu, check, progress):
    """Only fitting-prefix arrays enter, including all foldwise normalizers."""
    x, y, directions = np.asarray(x), np.asarray(y), np.asarray(directions)
    if x.shape != (len(x), 288) or len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Complete finite fitting-prefix representations required")
    folds = balanced_folds(directions)
    quality, correlation = residual_quality(x, y, directions)
    gains = {c["id"]: np.zeros(len(c["indices"])) for c in context_menu}
    marginal = np.zeros(96)
    for fit, valid in folds:
        stats = residual_statistics(x[fit], y[fit], directions[fit])
        xf, yf = residual_coordinates(stats, x[fit], y[fit], directions[fit])
        xv, yv = residual_coordinates(stats, x[valid], y[valid], directions[valid])
        denominator = float(np.sum(((y[valid]-stats.target_mean)/stats.target_sd)**2))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ArithmeticError("Positive training-normalized held-fold target denominator required")
        null = float(np.sum(yv**2))
        for e in range(96):
            cols = columns([e]); beta = coefficient(xf, yf, cols, check, progress)
            marginal[e] += (null-float(np.sum((yv-xv[:, cols]@beta)**2)))/denominator/4
        for context in context_menu:
            ids = context["indices"]; cols = columns(ids)
            beta = coefficient(xf, yf, cols, check, progress)
            full_sse = float(np.sum((yv-xv[:, cols]@beta)**2))
            for index, electrode in enumerate(ids):
                reduced = columns([e for e in ids if e != electrode])
                beta = coefficient(xf, yf, reduced, check, progress)
                drop_sse = float(np.sum((yv-xv[:, reduced]@beta)**2))
                gains[context["id"]][index] += (drop_sse-full_sse)/denominator/4
    records = []
    for context in context_menu:
        for index, electrode in enumerate(context["indices"]):
            others = columns([e for e in context["indices"] if e != electrode])
            redundancy = float(np.mean(correlation[np.ix_(columns([electrode]), others)]**2))
            base = [*quality[electrode].tolist(), float(marginal[electrode]), redundancy,
                    float(np.log(len(context["indices"])))]
            records.append({"prefix_trials": len(x), "context": context["id"],
                            "context_family": context["family"], "electrode": int(channels[electrode]),
                            "row_weight": context["row_weight"], "base_features": base,
                            CONDITIONAL_FEATURE: float(gains[context["id"]][index])})
    if len(records) != 384 or not np.isfinite([r[CONDITIONAL_FEATURE] for r in records]).all():
        raise ArithmeticError("Complete 384-row predictor panel required")
    return records


def fit_removals(x, y, directions, context_menu, check, progress):
    """Freeze every 96-trial decoder before an evaluation array is supplied."""
    if np.shape(x) != (96, 288) or len(y) != 96:
        raise ValueError("Original full 96-trial fitting support required")
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    fitted = []
    for context in context_menu:
        ids = context["indices"]; cols = columns(ids)
        full = coefficient(xx, yy, cols, check, progress)
        dropped = []
        for electrode in ids:
            reduced = columns([e for e in ids if e != electrode])
            dropped.append((electrode, reduced, coefficient(xx, yy, reduced, check, progress)))
        fitted.append((context["id"], cols, full, dropped))
    return stats, fitted


def removal_values(frozen, x, y, directions, channels):
    stats, fitted = frozen
    xx, yy = residual_coordinates(stats, x, y, directions)
    denominator = float(np.sum(((y-stats.target_mean)/stats.target_sd)**2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ArithmeticError("Positive common original-evaluation denominator required")
    values, contexts_out = {}, []
    for name, cols, full, dropped in fitted:
        full_sse = float(np.sum((yy-xx[:, cols]@full)**2))
        contexts_out.append({"context": name, "full_SSE": full_sse, "denominator": denominator,
                             "full_Q": 1-full_sse/denominator})
        for electrode, reduced, beta in dropped:
            drop_sse = float(np.sum((yy-xx[:, reduced]@beta)**2))
            values[(name, int(channels[electrode]))] = (drop_sse-full_sse)/denominator
    if len(values) != 384 or not np.isfinite(list(values.values())).all():
        raise ArithmeticError("All finite removal outcomes required")
    return values, contexts_out


def fit_value_predictor(sessions, excluded_session, prefix, augmented, cross_animal=False):
    """Exclude the whole target session/animal before fitting any statistics."""
    target = next(s for s in sessions if s["filename"] == excluded_session)
    donors = [s for s in sessions if s["filename"] != excluded_session
              and (not cross_animal or s["animal"] != target["animal"])]
    if len(donors) != (4 if cross_animal else 7):
        raise ValueError("All fixed non-target donors required")
    animals = sorted({s["animal"] for s in donors})
    animal_counts = {a: sum(s["animal"] == a for s in donors) for a in animals}
    features, labels, weights = [], [], []
    for session in donors:
        rows = [r for r in session["rows"] if r["prefix_trials"] == prefix]
        if len(rows) != 384:
            raise ValueError("Complete donor electrode/context panel required")
        for row in rows:
            features.append(row["base_features"]+([row[CONDITIONAL_FEATURE]] if augmented else []))
            labels.append(row["removal_value"])
            weights.append(row["row_weight"]/len(animals)/animal_counts[session["animal"]])
    x, y, w = np.asarray(features), np.asarray(labels), np.asarray(weights)
    w /= w.sum(); mean = w@x
    sd = np.sqrt(w@((x-mean)**2)); sd[sd == 0] = 1.
    z = (x-mean)/sd; ym = float(w@y)
    projection = None
    if augmented:
        # A duplicate feature must not merely weaken baseline ridge shrinkage.
        # Learn its baseline-linear component from donors only, without Y.
        baseline = np.column_stack((np.ones(len(z)), z[:, :-1]))
        root_weight = np.sqrt(w)
        projection = np.linalg.lstsq(baseline*root_weight[:, None], z[:, -1]*root_weight, rcond=None)[0]
        z[:, -1] -= baseline@projection
        # Deliberately do not restandardize a near-zero residual feature.
    beta = np.linalg.solve(z.T@(w[:, None]*z)+np.eye(z.shape[1]), z.T@(w*(y-ym)))
    return {"mean": mean, "sd": sd, "beta": beta, "intercept": ym,
            "conditional_projection": projection,
            "donor_sessions": [s["filename"] for s in donors]}


def predict_value(state, rows, augmented):
    x = np.asarray([r["base_features"]+([r[CONDITIONAL_FEATURE]] if augmented else []) for r in rows])
    z = (x-state["mean"])/state["sd"]
    if augmented:
        z[:, -1] -= np.column_stack((np.ones(len(z)), z[:, :-1]))@state["conditional_projection"]
    return z@state["beta"]+state["intercept"]


def rank_correlation(a, b):
    a, b = rankdata(a), rankdata(b)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def cross_session_report(sessions, check):
    if len(sessions) != 8 or sorted(sum(s["animal"] == a for s in sessions) for a in ("Chewie", "Mihili")) != [4, 4]:
        raise ValueError("All eight original sessions and both animals required")
    output = []
    for prefix in PREFIXES:
        for cross_animal in (False, True):
            results = []
            for session in sessions:
                check()
                rows = [r for r in session["rows"] if r["prefix_trials"] == prefix]
                states = [fit_value_predictor(sessions, session["filename"], prefix, flag, cross_animal) for flag in (False, True)]
                predictions = [predict_value(state, rows, flag) for state, flag in zip(states, (False, True))]
                y, w = np.asarray([r["removal_value"] for r in rows]), np.asarray([r["row_weight"] for r in rows])
                mse = [float(w@((y-p)**2)) for p in predictions]
                ranking = []
                for context in dict.fromkeys(r["context"] for r in rows):
                    indices = [i for i, r in enumerate(rows) if r["context"] == context]
                    ranking.append({"context": context, "family": rows[indices[0]]["context_family"],
                                    "baseline": rank_correlation(y[indices], predictions[0][indices]),
                                    "augmented": rank_correlation(y[indices], predictions[1][indices]),
                                    "conditional_proxy": rank_correlation(y[indices], [rows[i][CONDITIONAL_FEATURE] for i in indices])})
                results.append({"filename": session["filename"], "animal": session["animal"],
                                "baseline_MSE": mse[0], "augmented_MSE": mse[1],
                                "MSE_reduction": mse[0]-mse[1], "donor_sessions": states[0]["donor_sessions"],
                                "baseline_predictions": predictions[0].tolist(),
                                "augmented_predictions": predictions[1].tolist(), "rankings": ranking})
            animal_results = []
            for animal in ("Chewie", "Mihili"):
                group = [r for r in results if r["animal"] == animal]
                animal_results.append({"animal": animal, **{key: float(np.mean([r[key] for r in group]))
                    for key in ("baseline_MSE", "augmented_MSE", "MSE_reduction")},
                    "positive_sessions": sum(r["MSE_reduction"] > 0 for r in group)})
            output.append({"prefix_trials": prefix, "transfer": "opposite_animal" if cross_animal else "leave_one_session_out",
                           "sessions": results, "animals": animal_results,
                           "equal_animal_MSE_reduction": float(np.mean([r["MSE_reduction"] for r in animal_results])),
                           "fixed_prediction_session_omission": [{"omitted": omitted["filename"],
                               "MSE_reduction": float(np.mean([np.mean([r["MSE_reduction"] for r in results
                                   if r["animal"] == a and r["filename"] != omitted["filename"]]) for a in ("Chewie", "Mihili")]))}
                               for omitted in results]})
    return output
