"""Pure rank fitting conditional on a fixed observed initializer."""
import numpy as np

from capacity_training import CapacityModel, rank_projection
from direction_training import residual_statistics, residual_coordinates
from ridge_training import ALPHAS, ridge_coefficients


def validate_conditioned_folds(n, initial, folds):
    initial, all_rows = set(initial), set(range(n))
    if not initial or not initial < all_rows or len(folds) != 4:
        raise ValueError("Initializer and post-initial observations with four folds required")
    seen = []
    for fit, valid in folds:
        f, v = set(fit), set(valid)
        if (len(f) != len(fit) or len(v) != len(valid) or not v
                or f & v or f | v != all_rows or not initial <= f or initial & v):
            raise ValueError("Each fold must condition on every initial row and validate only acquired later rows")
        seen.extend(valid)
    if sorted(seen) != sorted(all_rows-initial):
        raise ValueError("Post-initial validation rows must partition acquired later rows exactly once")


def conditioned_folds(labels, initial_mask, post_positions):
    initial = np.flatnonzero(initial_mask)
    folds = []
    for fold in range(4):
        valid = np.flatnonzero(~initial_mask & ((post_positions+labels) % 4 == fold))
        fit = np.flatnonzero(~np.isin(np.arange(len(labels)), valid))
        folds.append((fit, valid))
    validate_conditioned_folds(len(labels), initial, folds)
    return initial, folds


def fit_conditioned_rank(x, y, labels, initial, folds, progress):
    x, y, labels = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True), np.asarray(labels).copy()
    validate_conditioned_folds(len(x), initial, folds)
    bound = min(x.shape[1], y.shape[1], *(len(f)-8 for f, _ in folds), len(x)-8)
    ranks = tuple(r for r in (1, 2, 4, 8) if r <= bound)+("full",)
    candidates = [(a, r) for r in ranks for a in ALPHAS]
    errors = np.zeros(len(candidates))
    for fit, valid in folds:
        stats = residual_statistics(x[fit], y[fit], labels[fit])
        progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
        progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
        xf, yf = residual_coordinates(stats, x[fit], y[fit], labels[fit])
        xv, yv = residual_coordinates(stats, x[valid], y[valid], labels[valid])
        for alpha in ALPHAS:
            base = ridge_coefficients(xf, yf, alpha)
            progress["inner_ridge_fits"] += 1
            for index, (candidate_alpha, rank) in enumerate(candidates):
                if alpha != candidate_alpha:
                    continue
                error = float(np.sum((yv-xv@rank_projection(xf, yf, base, rank))**2))
                if not np.isfinite(error):
                    raise ArithmeticError("Nonfinite initializer-conditioned CV error")
                errors[index] += error
                progress["capacity_inner_candidate_scores"] += 1
    best = min(range(len(candidates)), key=lambda i: (errors[i], ranks.index(candidates[i][1]), -candidates[i][0]))
    alpha, rank = candidates[best]
    stats = residual_statistics(x, y, labels)
    progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
    progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
    xx, yy = residual_coordinates(stats, x, y, labels)
    coefficients = rank_projection(xx, yy, ridge_coefficients(xx, yy, alpha), rank)
    progress["outer_training_refits"] += 1
    return CapacityModel("reduced_rank", stats, coefficients, alpha, rank, ranks, None, len(candidates))
