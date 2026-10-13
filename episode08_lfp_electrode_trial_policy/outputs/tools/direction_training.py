"""Pure fitting-array direction means and residual ridge; no source/evaluation I/O."""
from dataclasses import dataclass

import numpy as np

from ridge_training import ALPHAS, ridge_coefficients, scaling
from selector_training import validate_folds


def direction_means(values, directions):
    values, directions = np.asarray(values), np.asarray(directions)
    if (values.ndim != 2 or len(values) != len(directions)
            or not np.isfinite(values).all()
            or not np.array_equal(np.unique(directions), np.arange(8))):
        raise ValueError("Finite fitting arrays and all eight directions are required")
    return np.stack([values[directions == d].mean(axis=0) for d in range(8)])


@dataclass(frozen=True)
class DirectionNull:
    direction_target_mean: np.ndarray
    target_mean: np.ndarray
    target_sd: np.ndarray


def fit_direction_null(y, directions):
    # Deliberately no LFP argument: this comparator uses zero LFP electrodes.
    ym, ys, _ = scaling(y)
    return DirectionNull(direction_means(y, directions), ym, ys)


@dataclass(frozen=True)
class ResidualStatistics:
    direction_input_mean: np.ndarray
    direction_target_mean: np.ndarray
    input_mean: np.ndarray
    input_sd: np.ndarray
    target_mean: np.ndarray
    target_sd: np.ndarray
    target_residual_mean: np.ndarray
    constant_input_columns: tuple
    constant_target_columns: tuple


def residual_statistics(x, y, directions):
    """Only fitting rows enter; raw-Y SD deliberately differs from residual SD."""
    mx, my = direction_means(x, directions), direction_means(y, directions)
    xm, xs, xc = scaling(x - mx[directions])
    ym, ys, yc = scaling(y)
    residual_y = (y - my[directions]) / ys
    return ResidualStatistics(mx, my, xm, xs, ym, ys, residual_y.mean(axis=0), xc, yc)


def residual_coordinates(stats, x, y, directions):
    xx = (x - stats.direction_input_mean[directions] - stats.input_mean) / stats.input_sd
    yy = (y - stats.direction_target_mean[directions]) / stats.target_sd - stats.target_residual_mean
    return xx, yy


@dataclass(frozen=True)
class ResidualModel:
    statistics: ResidualStatistics
    coefficients: np.ndarray
    alpha: float


def fit_residual_candidate(x, y, directions, folds, progress):
    x, y = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
    directions = np.asarray(directions).copy()
    validate_folds(len(x), folds)
    errors = np.zeros(len(ALPHAS))
    for fit, valid in folds:
        stats = residual_statistics(x[fit], y[fit], directions[fit])
        progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
        progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
        xf, yf = residual_coordinates(stats, x[fit], y[fit], directions[fit])
        xv, yv = residual_coordinates(stats, x[valid], y[valid], directions[valid])
        for index, alpha in enumerate(ALPHAS):
            coefficients = ridge_coefficients(xf, yf, alpha)
            progress["inner_ridge_fits"] += 1
            # Equivalent to true raw-Y prediction error after adding back the
            # fitting direction mean, in raw fitting-Y SD coordinates.
            error = float(np.sum((yv - xv @ coefficients) ** 2))
            if not np.isfinite(error):
                raise ArithmeticError("Nonfinite direction-residual CV error")
            errors[index] += error
    chosen = [i for i, error in enumerate(errors) if error == errors.min()][-1]
    stats = residual_statistics(x, y, directions)
    progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
    progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
    xx, yy = residual_coordinates(stats, x, y, directions)
    coefficients = ridge_coefficients(xx, yy, ALPHAS[chosen])
    progress["outer_training_refits"] += 1
    return ResidualModel(stats, coefficients, ALPHAS[chosen])
