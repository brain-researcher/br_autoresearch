"""Fixed EP08 fitter: training arrays and relative inner-fold indices only.

No source paths, source trial identities, evaluation arrays, scoring callback,
filesystem I/O or candidate plugins enter this module.
"""
from dataclasses import dataclass

import numpy as np

ALPHAS = (0.0001, 0.01, 1., 100., 10000.)


@dataclass(frozen=True)
class FittedRidge:
    input_mean: np.ndarray
    input_sd: np.ndarray
    target_mean: np.ndarray
    target_sd: np.ndarray
    coefficients: np.ndarray
    alpha: float
    constant_input_columns: tuple
    constant_target_columns: tuple


def ridge_coefficients(x, y, alpha):
    """Minimize ||Y-XB||_F^2/n + alpha*||B||_F^2 after centering/scaling."""
    n = len(x)
    dual = np.linalg.solve(x @ x.T + (n * alpha) * np.eye(n), y)
    result = x.T @ dual
    if not np.isfinite(result).all():
        raise ArithmeticError("Nonfinite ridge coefficient")
    return result


def scaling(values):
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=0)
    constant = tuple(int(index) for index in np.flatnonzero(sd == 0))
    sd[sd == 0] = 1.
    return mean, sd, constant


def fit_candidate(x, y, folds, progress):
    """Fit the frozen alpha grid/CV/refit using only this candidate's training.

`folds` contains (fitting, validation) row indices relative to x/y. `progress`
contains counters only and is shared to retain completed fits on interruption.
"""
    x, y = np.array(x, dtype=np.float64, copy=True), np.array(y, dtype=np.float64, copy=True)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Fitter requires aligned finite two-dimensional training arrays")
    all_rows, validation_partition = set(range(len(x))), []
    if len(folds) != 4:
        raise ValueError("Expected four training-only folds")
    for fitting, validation in folds:
        f, v = list(fitting), list(validation)
        if (not f or not v or len(f) != len(set(f)) or len(v) != len(set(v))
                or set(f).intersection(v) or set(f).union(v) != all_rows):
            raise ValueError("Each inner fold must partition this training array")
        validation_partition.extend(v)
    if sorted(validation_partition) != list(range(len(x))):
        raise ValueError("Inner validation folds must partition the training rows exactly once")
    errors = np.zeros(len(ALPHAS), dtype=np.float64)
    for fitting, validation in folds:
        xf, xv, yf, yv = x[fitting], x[validation], y[fitting], y[validation]
        xm, xs, xc = scaling(xf)
        ym, ys, yc = scaling(yf)
        progress["zero_SD_input_column_fit_occurrences"] += len(xc)
        progress["zero_SD_target_column_fit_occurrences"] += len(yc)
        xf, xv, yf, yv = (xf - xm) / xs, (xv - xm) / xs, (yf - ym) / ys, (yv - ym) / ys
        for index, alpha in enumerate(ALPHAS):
            coefficients = ridge_coefficients(xf, yf, alpha)
            progress["inner_ridge_fits"] += 1
            error = float(np.sum((yv - xv @ coefficients) ** 2))
            if not np.isfinite(error):
                raise ArithmeticError("Nonfinite inner-training CV error")
            errors[index] += error
    selected = [index for index, value in enumerate(errors) if value == errors.min()][-1]
    xm, xs, xc = scaling(x)
    ym, ys, yc = scaling(y)
    progress["zero_SD_input_column_fit_occurrences"] += len(xc)
    progress["zero_SD_target_column_fit_occurrences"] += len(yc)
    coefficients = ridge_coefficients((x - xm) / xs, (y - ym) / ys, ALPHAS[selected])
    progress["outer_training_refits"] += 1
    return FittedRidge(xm, xs, ym, ys, coefficients, ALPHAS[selected], xc, yc)


def new_progress():
    return {"inner_ridge_fits": 0, "outer_training_refits": 0,
            "zero_SD_input_column_fit_occurrences": 0, "zero_SD_target_column_fit_occurrences": 0}
