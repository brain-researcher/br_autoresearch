"""EP16 declared score/window/scaling operations; no file or reader access.

Callers supply the adopted analysis_definition.json contents and already
authorized fitting/evaluation arrays. These operations never select trials,
assign participant roles or authorize a reader. An unresolved packet remains
in its registered pair/component disposition; callers must not discard it.
"""
from __future__ import annotations

import numpy as np


class UnresolvedPacket(ValueError):
    """The fixed packet cannot support the requested registered operation."""


def direction_weights(labels, definition):
    """Equal total weight in each of the eight half-open direction octants."""
    y = np.asarray(labels, dtype=np.float64)
    score = definition["score"]
    if score["octants"] != 8 or score["loss"] != "sum_squared_error_over_two_coordinates":
        raise ValueError("Unsupported EP16 score definition")
    if y.ndim != 2 or y.shape[1] != 2 or not len(y) or not np.isfinite(y).all():
        raise UnresolvedPacket("Labels must be finite nonempty two-coordinate vectors")
    if np.any(np.abs(np.linalg.norm(y, axis=1) - 1) > score["label_unit_norm_numerical_tolerance"]):
        raise UnresolvedPacket("Direction labels must already be unit vectors")
    angles = np.remainder(np.arctan2(y[:, 1], y[:, 0]), 2 * np.pi)
    # Remainder can round a tiny negative angle to 2*pi; this is the zero boundary.
    octants = np.floor(angles / (np.pi / 4)).astype(np.int64) % 8
    counts = np.bincount(octants, minlength=8)
    if np.any(counts == 0):
        raise UnresolvedPacket("Missing octant in fixed packet; do not replace trials")
    return 1.0 / (8 * counts[octants])


def direction_balanced_score(labels, predictions, definition):
    y = np.asarray(labels, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    weights = direction_weights(y, definition)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise UnresolvedPacket("Predictions must be finite and match the fixed labels")
    loss = np.sum((y - prediction) ** 2, axis=1)
    result = -float(weights @ loss)
    if not np.isfinite(result):
        raise UnresolvedPacket("Nonfinite score remains unresolved")
    return result


def complete_window_means(windows, definition):
    """trial x bin x physical-channel windows; invalid channel windows stay NaN."""
    values = np.asarray(windows, dtype=np.float64)
    bins = definition["feature_window"]["bins"]
    if bins != 30 or values.ndim != 3 or values.shape[1] != bins:
        raise UnresolvedPacket("Require every declared 30-bin window; no padding")
    if not values.shape[0] or not values.shape[2]:
        raise UnresolvedPacket("Empty trial or physical-channel axis")
    complete = np.isfinite(values).all(axis=1)
    means = np.full((values.shape[0], values.shape[2]), np.nan)
    # Never nanmean: it would change the physical window on invalid channels.
    safe_values = np.where(np.isfinite(values), values, 0.0)
    sums = np.sum(safe_values, axis=1, dtype=np.float64)
    if np.any(complete & ~np.isfinite(sums)):
        raise UnresolvedPacket("Finite-window arithmetic overflow is not physical missingness")
    means[complete] = sums[complete] / bins
    return means


def fit_source_scaler(fitting_features, definition):
    """Fit only to supplied source fitting trials, retaining the entire axis."""
    x = np.asarray(fitting_features, dtype=np.float64)
    quality = definition["channel_quality"]
    if quality["minimum_available_fitting_trials"] != 2 or quality["scale_ddof"] != 0:
        raise ValueError("Unsupported EP16 source-scaling definition")
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise UnresolvedPacket("Nonempty trial x physical-channel features required")
    center = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    usable = np.zeros(x.shape[1], dtype=bool)
    counts = np.isfinite(x).sum(axis=0)
    for column in range(x.shape[1]):
        finite = x[np.isfinite(x[:, column]), column]
        if len(finite) < 2:
            continue
        mean, deviation = float(finite.mean()), float(finite.std(ddof=0))
        if np.isfinite(mean) and np.isfinite(deviation) and deviation > 0:
            center[column], scale[column], usable[column] = mean, deviation, True
    return {"center": center, "scale": scale, "usable": usable,
            "finite_fitting_trials": counts}


def apply_source_scaler(features, fitted):
    """No target refit: absent/invalid/inactive coordinates are zero, not deleted."""
    x = np.asarray(features, dtype=np.float64)
    center, scale, usable = fitted["center"], fitted["scale"], fitted["usable"]
    if x.ndim != 2 or x.shape[1] != len(center):
        raise ValueError("Physical-channel identity/axis must be aligned upstream")
    result = np.zeros_like(x)
    observed = np.isfinite(x) & usable[None, :]
    rows, columns = np.nonzero(observed)
    result[rows, columns] = (x[rows, columns] - center[columns]) / scale[columns]
    if not np.isfinite(result).all():
        raise UnresolvedPacket("Nonfinite standardized features remain unresolved")
    return result
