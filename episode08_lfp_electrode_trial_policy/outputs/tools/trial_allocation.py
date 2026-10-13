"""Pure acquired-history allocation utilities and fixed-capacity refits."""
from dataclasses import replace

import numpy as np

from capacity_training import CapacityModel, rank_projection
from direction_training import residual_statistics, residual_coordinates
from ridge_training import ridge_coefficients

POLICIES = ("balanced", "mean_value", "mean_error")
CHECKPOINTS = (32, 48, 64, 80, 96)


def choose_direction(targets, labels, initial_sd, counts, policy):
    """No future rows, source handles, trial IDs or evaluation inputs."""
    y, labels = np.asarray(targets), np.asarray(labels)
    counts, scale = np.asarray(counts), np.asarray(initial_sd)
    if (policy not in POLICIES or y.ndim != 2 or labels.shape != (len(y),)
            or counts.shape != (8,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 4) or np.any(counts > 12)
            or not np.array_equal(counts, np.array([np.count_nonzero(labels == d) for d in range(8)]))
            or scale.shape != (y.shape[1],) or np.any(scale <= 0)
            or not np.isfinite(y).all() or not np.isfinite(scale).all()):
        raise ValueError("Complete finite acquired history and fixed initial scaling required")
    legal = np.flatnonzero(counts < 12)
    if not len(legal):
        raise ValueError("The finite trial pool is exhausted")
    z = y / scale
    variance = np.array([np.var(z[labels == d], axis=0, ddof=1).mean() for d in range(8)])
    pooled = float(np.sum((counts-1)*variance) / np.sum(counts-1))
    shrunk = ((counts-1)*variance + 3*pooled) / (counts+2)
    if policy == "mean_value":
        utility = shrunk / (counts*(counts+1))
    elif policy == "mean_error":
        utility = shrunk * (1 + 1/counts)
    else:
        utility = -counts.astype(float)
    if not np.isfinite(utility).all():
        raise ArithmeticError("Nonfinite pre-choice utility")
    chosen = min(legal, key=lambda d: (-utility[d], counts[d], d))
    return int(chosen), {"counts_before": counts.tolist(), "utility": utility.tolist(),
                         "direction_variance": variance.tolist(), "pooled_variance": pooled,
                         "shrunk_variance": shrunk.tolist()}


def frozen_numeric_rank(initial_model, feature_width, target_width):
    """Full resolves to the initializer's structural bound, never a growing alias."""
    rank = min(feature_width, target_width, 32-8) if initial_model.rank == "full" else int(initial_model.rank)
    if not 1 <= rank <= min(feature_width, target_width, 32-8):
        raise ValueError("Initializer rank exceeds its declared structural support")
    return rank


def freeze_initial_model(initial_model, feature_width, target_width):
    rank = frozen_numeric_rank(initial_model, feature_width, target_width)
    # Keep the already qualified initializer coefficients exactly; only resolve
    # the full alias and identify the fixed subsequent capacity.
    return replace(initial_model, rank=rank, legal_ranks=(rank,)), rank


def fit_frozen_rank(x, y, labels, alpha, rank, progress):
    x, y, labels = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True), np.asarray(labels).copy()
    bound = min(x.shape[1], y.shape[1], len(x)-8)
    if not isinstance(rank, int) or not 1 <= rank <= bound or alpha <= 0:
        raise ValueError("The frozen positive rank and penalty must remain supported")
    stats = residual_statistics(x, y, labels)
    progress["zero_SD_input_column_fit_occurrences"] += len(stats.constant_input_columns)
    progress["zero_SD_target_column_fit_occurrences"] += len(stats.constant_target_columns)
    xx, yy = residual_coordinates(stats, x, y, labels)
    coefficients = ridge_coefficients(xx, yy, alpha)
    # When the fixed cap equals the complete structural support, ridge already
    # attains the rank-constrained optimum; avoid redundant numerical rotation.
    if rank < bound:
        coefficients = rank_projection(xx, yy, coefficients, rank)
    if not np.isfinite(coefficients).all():
        raise ArithmeticError("Nonfinite fixed-capacity refit")
    progress["outer_training_refits"] += 1
    return CapacityModel("reduced_rank", stats, coefficients, alpha, rank, (rank,), None, 0)
