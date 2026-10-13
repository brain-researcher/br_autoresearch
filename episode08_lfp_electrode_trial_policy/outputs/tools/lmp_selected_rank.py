"""True-target nested reduced-rank fitting on selected scalar LMP columns."""
import numpy as np

from capacity_training import CapacityModel, rank_projection
from direction_training import residual_coordinates, residual_statistics
from ridge_training import ALPHAS, ridge_coefficients
from selector_training import validate_folds


def fit_selected_rank(x, y, labels, folds, inner_orders, outer_order, count, progress):
    validate_folds(len(x), folds)
    if x.ndim != 2 or x.shape[1] != 96 or len(inner_orders) != len(folds):
        raise ValueError("Original96 physical electrodes must each have one native-LMP column")
    bound = min(count, y.shape[1], *(len(f)-8 for f, _ in folds), len(x)-8)
    ranks = tuple(r for r in (1, 2, 4, 8) if r <= bound)+("full",)
    candidates = [(alpha, rank) for rank in ranks for alpha in ALPHAS]
    errors = np.zeros(len(candidates))
    for (fit, valid), order in zip(folds, inner_orders):
        columns = np.asarray(order[:count])
        xf0, xv0 = x[np.ix_(fit, columns)], x[np.ix_(valid, columns)]
        stats = residual_statistics(xf0, y[fit], labels[fit])
        progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
        progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
        xf, yf = residual_coordinates(stats, xf0, y[fit], labels[fit])
        xv, yv = residual_coordinates(stats, xv0, y[valid], labels[valid])
        for alpha in ALPHAS:
            base = ridge_coefficients(xf, yf, alpha)
            progress["inner_ridge_fits"] += 1
            for index, (candidate_alpha, rank) in enumerate(candidates):
                if candidate_alpha != alpha:
                    continue
                error = float(np.sum((yv-xv@rank_projection(xf, yf, base, rank))**2))
                if not np.isfinite(error):
                    raise ArithmeticError("Nonfinite nested scalar-LMP selection error")
                errors[index] += error
                progress["capacity_inner_candidate_scores"] += 1
    best = min(range(len(candidates)), key=lambda i: (errors[i], ranks.index(candidates[i][1]), -candidates[i][0]))
    alpha, rank = candidates[best]
    columns = np.asarray(outer_order[:count])
    stats = residual_statistics(x[:, columns], y, labels)
    progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
    progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
    xx, yy = residual_coordinates(stats, x[:, columns], y, labels)
    coefficient = rank_projection(xx, yy, ridge_coefficients(xx, yy, alpha), rank)
    progress["outer_training_refits"] += 1
    return CapacityModel("reduced_rank", stats, coefficient, alpha, rank, ranks, None, len(candidates)), columns
