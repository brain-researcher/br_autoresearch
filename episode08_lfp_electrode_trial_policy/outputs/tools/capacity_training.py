"""Training-array-only reduced-rank and RBF residual models; no source I/O."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from direction_training import residual_statistics, residual_coordinates
from ridge_training import ALPHAS, ridge_coefficients
from selector_training import validate_folds


def rank_projection(x, y, coefficients, rank):
    if rank == "full":
        return coefficients
    metric = y.T @ x @ coefficients
    _, vectors = np.linalg.eigh((metric + metric.T) / 2)
    selected = vectors[:, -rank:]
    return coefficients @ selected @ selected.T


@dataclass(frozen=True)
class KernelState:
    anchors: np.ndarray
    sigma_squared: float
    column_mean: np.ndarray
    grand_mean: float


def training_kernel(x):
    distances = cdist(x, x, metric="sqeuclidean")
    positive = distances[np.triu_indices(len(x), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        return np.zeros((len(x), len(x))), KernelState(x.copy(), 0., np.ones(len(x)), 1.)
    sigma_squared = float(np.median(positive))
    kernel = np.exp(-distances / (2 * sigma_squared))
    means, grand = kernel.mean(axis=0), float(kernel.mean())
    return kernel-means[None, :]-means[:, None]+grand, KernelState(x.copy(), sigma_squared, means, grand)


@dataclass(frozen=True)
class CapacityModel:
    method: str
    statistics: object
    coefficients: np.ndarray
    alpha: float
    rank: object
    legal_ranks: tuple
    kernel_state: object
    candidate_count: int


def fit_capacity(x, y, directions, folds, method, progress):
    x, y, directions = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True), np.asarray(directions).copy()
    validate_folds(len(x), folds)
    if method not in ("reduced_rank", "rbf"):
        raise ValueError("Only the two selected capacity families are allowed")
    bound = min(x.shape[1], y.shape[1], *(len(fit)-8 for fit, _ in folds), len(x)-8)
    legal_ranks = tuple(r for r in (1, 2, 4, 8) if r <= bound) + ("full",)
    candidates = [(alpha, rank) for rank in (legal_ranks if method == "reduced_rank" else ("full",)) for alpha in ALPHAS]
    errors = np.zeros(len(candidates))
    for fit, valid in folds:
        stats = residual_statistics(x[fit], y[fit], directions[fit])
        progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
        progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
        xf, yf = residual_coordinates(stats, x[fit], y[fit], directions[fit])
        xv, yv = residual_coordinates(stats, x[valid], y[valid], directions[valid])
        if method == "rbf":
            kernel, state = training_kernel(xf)
            # Inner validation is part of the supplied training pool; no outer
            # evaluator array reaches this module.
            if state.sigma_squared == 0:
                valid_kernel = np.zeros((len(xv), len(xf)))
            else:
                raw = np.exp(-cdist(xv, state.anchors, metric="sqeuclidean")/(2*state.sigma_squared))
                valid_kernel = raw-state.column_mean[None, :]-raw.mean(axis=1)[:, None]+state.grand_mean
        for alpha in ALPHAS:
            if method == "reduced_rank":
                base = ridge_coefficients(xf, yf, alpha)
            else:
                base = np.linalg.solve(kernel+len(xf)*alpha*np.eye(len(xf)), yf)
            progress["inner_ridge_fits"] += 1
            for index, (candidate_alpha, rank) in enumerate(candidates):
                if candidate_alpha != alpha:
                    continue
                prediction = (xv @ rank_projection(xf, yf, base, rank) if method == "reduced_rank" else valid_kernel @ base)
                error = float(np.sum((yv-prediction)**2))
                if not np.isfinite(error):
                    raise ArithmeticError("Nonfinite nested capacity error")
                errors[index] += error
                progress["capacity_inner_candidate_scores"] += 1
    # Rank candidates are ascending, full last; ties prefer lower rank then
    # larger alpha. RBF has a single fixed bandwidth rule.
    selected = min(range(len(candidates)), key=lambda i: (errors[i],
                   len(legal_ranks) if candidates[i][1] == "full" else legal_ranks.index(candidates[i][1]), -candidates[i][0]))
    alpha, rank = candidates[selected]
    stats = residual_statistics(x, y, directions)
    progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
    progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
    xx, yy = residual_coordinates(stats, x, y, directions)
    state = None
    if method == "reduced_rank":
        coefficients = rank_projection(xx, yy, ridge_coefficients(xx, yy, alpha), rank)
    else:
        kernel, state = training_kernel(xx)
        coefficients = np.linalg.solve(kernel+len(xx)*alpha*np.eye(len(xx)), yy)
    if not np.isfinite(coefficients).all():
        raise ArithmeticError("Nonfinite capacity refit")
    progress["outer_training_refits"] += 1
    return CapacityModel(method, stats, coefficients, alpha, rank, legal_ranks if method == "reduced_rank" else (), state, len(candidates))
