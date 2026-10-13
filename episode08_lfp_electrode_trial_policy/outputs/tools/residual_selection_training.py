"""Residual selectors and rank decoder consume fitting arrays only."""
from dataclasses import dataclass
import random
import time

import numpy as np

from capacity_training import CapacityModel, rank_projection
from direction_training import residual_statistics, residual_coordinates
from ridge_training import ALPHAS, ridge_coefficients
from selector_training import feature_columns, validate_folds

METHODS = ("residual_marginal", "residual_conditional", "permuted_residual_conditional")


def within_direction_permutation(labels, seed):
    permutation = np.arange(len(labels))
    if not np.array_equal(np.unique(labels), np.arange(8)):
        raise ValueError("Preserve eight directions")
    for direction in range(8):
        positions = np.flatnonzero(labels == direction).tolist()
        if len(positions) < 2:
            raise ValueError("Direction has insufficient permutation support")
        rng = random.Random()
        rng.seed(f"{seed}|{direction}", version=2)
        shuffled = positions.copy()
        # Include fixed points: ordinary permutations have zero expected
        # centered covariance, unlike small-group derangements.
        rng.shuffle(shuffled)
        permutation[positions] = shuffled
    return permutation


def selector_coordinates(x, y, directions):
    stats = residual_statistics(x, y, directions)
    return residual_coordinates(stats, x, y, directions)


def selector_order(x, y, directions, method, maximum, seed):
    if method not in METHODS or x.shape[1] % 3:
        raise ValueError("Selected residual method and three-coordinate electrodes required")
    x, y = selector_coordinates(x, y, directions)
    if method == "permuted_residual_conditional":
        y = y[within_direction_permutation(directions, seed)]
    n, groups = len(x), x.shape[1] // 3
    if not 0 < maximum <= groups:
        raise ValueError("Unsupported requested electrode count")
    blocks, ax, ay = x.reshape(n, groups, 3), x.copy(), y.copy()
    chosen, available = [], np.ones(groups, dtype=bool)
    for _ in range(1 if method == "residual_marginal" else maximum):
        gram = np.einsum("ngb,ngc->gbc", blocks, ax.reshape(n, groups, 3))/n
        matrix = (gram+gram.transpose(0, 2, 1))/2+np.eye(3)[None, :, :]
        cross = np.einsum("ngb,nu->gbu", blocks, ay)/n
        scores = np.sum(cross*np.linalg.solve(matrix, cross), axis=(1, 2))/y.shape[1]
        if not np.isfinite(scores).all():
            raise ArithmeticError("Nonfinite residual selector scores")
        if method == "residual_marginal":
            return np.argsort(-scores, kind="stable")[:maximum]
        electrode = int(np.argmax(np.where(available, scores, -np.inf)))
        chosen.append(electrode)
        available[electrode] = False
        columns = feature_columns([electrode])
        update = ax[:, columns].copy()
        rhs_x, rhs_y = x[:, columns].T @ ax/n, x[:, columns].T @ ay/n
        ax -= update @ np.linalg.solve(matrix[electrode], rhs_x)
        ay -= update @ np.linalg.solve(matrix[electrode], rhs_y)
    return np.asarray(chosen)


def order_bank(x, y, labels, folds, method, seed, progress):
    started, inner = time.monotonic(), []
    for fold, (fit, _valid) in enumerate(folds):
        inner.append(selector_order(x[fit], y[fit], labels[fit], method, 32, f"{seed}|{fold}"))
        progress["selector_fits"] += 1
    outer = selector_order(x, y, labels, method, 32, f"{seed}|outer")
    progress["selector_fits"] += 1
    progress["selection_seconds"] += time.monotonic()-started
    return inner, outer


def fit_selected_rank(x, y, labels, folds, inner_orders, outer_order, count, progress):
    validate_folds(len(x), folds)
    bound = min(3*count, y.shape[1], *(len(f)-8 for f, _ in folds), len(x)-8)
    ranks = tuple(r for r in (1, 2, 4, 8) if r <= bound)+( "full",)
    candidates = [(alpha, rank) for rank in ranks for alpha in ALPHAS]
    errors = np.zeros(len(candidates))
    for (fit, valid), order in zip(folds, inner_orders):
        columns = feature_columns(order[:count])
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
                coefficients = rank_projection(xf, yf, base, rank)
                error = float(np.sum((yv-xv@coefficients)**2))
                if not np.isfinite(error):
                    raise ArithmeticError("Nonfinite nested residual-selection error")
                errors[index] += error
                progress["capacity_inner_candidate_scores"] += 1
    best = min(range(len(candidates)), key=lambda i: (errors[i], ranks.index(candidates[i][1]), -candidates[i][0]))
    alpha, rank = candidates[best]
    columns = feature_columns(outer_order[:count])
    stats = residual_statistics(x[:, columns], y, labels)
    progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
    progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
    xx, yy = residual_coordinates(stats, x[:, columns], y, labels)
    coefficients = rank_projection(xx, yy, ridge_coefficients(xx, yy, alpha), rank)
    progress["outer_training_refits"] += 1
    return CapacityModel("reduced_rank", stats, coefficients, alpha, rank, ranks, None, len(candidates)), columns


@dataclass(frozen=True)
class SelectedRank:
    method: str
    electrodes: int
    replicate: int
    columns: np.ndarray
    fitted: CapacityModel
