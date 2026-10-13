"""EP05 train-only direction/time and ridge-residual components.

These are reusable components, not the full adaptive search, a scientific
configuration selection, or an authenticated publication-code reproduction.
All numerical scientific choices are explicit call arguments. Callers own the
outer whole-session policy-selection boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def circular_basis(angles, harmonics):
    if not isinstance(harmonics, int) or harmonics < 0:
        raise ValueError("harmonics must be an explicit nonnegative integer")
    angles = np.asarray(angles, dtype=np.float64)
    columns = [np.ones_like(angles)]
    for order in range(1, harmonics + 1):
        columns.extend((np.cos(order * angles), np.sin(order * angles)))
    return np.column_stack(columns)


@dataclass
class DirectionTime:
    coefficients: np.ndarray
    harmonics: int
    time_bins: int

    @classmethod
    def fit(cls, angles, targets, harmonics, direction_labels):
        """Fit cyclic angle coefficients at each fixed native elapsed-time bin.

        Equal direction weights avoid a direction's trial count changing its
        contribution to the behavioral estimate. Only supplied training trials
        can contribute to this operation.
        """
        y = np.asarray(targets, dtype=np.float64)
        labels = np.asarray(direction_labels)
        if y.ndim != 3 or y.shape[2] != 2 or len(y) != len(labels):
            raise ValueError("targets must be complete trial×time×XY arrays")
        design = circular_basis(angles, harmonics)
        if len(design) != len(y) or not np.isfinite(y).all():
            raise ValueError("invalid training target support")
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("training directions cannot identify the selected cyclic basis")
        counts = {label: np.count_nonzero(labels == label) for label in np.unique(labels)}
        weight = np.sqrt(np.array([1.0 / counts[label] for label in labels]))
        coefficient = np.linalg.lstsq(design * weight[:, None],
                                      y.reshape(len(y), -1) * weight[:, None], rcond=None)[0]
        return cls(coefficient.reshape(design.shape[1], y.shape[1], 2), harmonics, y.shape[1])

    def predict(self, angles):
        return np.einsum("np,ptk->ntk", circular_basis(angles, self.harmonics), self.coefficients)


@dataclass
class RidgeResidual:
    baseline: DirectionTime
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_mean: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(cls, features, angles, targets, labels, *, harmonics, ridge_alpha):
        if ridge_alpha < 0:
            raise ValueError("ridge penalty cannot be negative")
        x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        if x.ndim != 3 or x.shape[:2] != y.shape[:2] or not np.isfinite(x).all():
            raise ValueError("features must align with complete training trials and native bins")
        baseline = DirectionTime.fit(angles, y, harmonics, labels)
        residual = (y - baseline.predict(angles)).reshape(-1, 2)
        flat = x.reshape(-1, x.shape[2])
        mean, scale = flat.mean(axis=0), flat.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        standardized = (flat - mean) / scale
        residual_mean = residual.mean(axis=0)
        target = residual - residual_mean
        if ridge_alpha == 0:
            coefficient = np.linalg.lstsq(standardized, target, rcond=None)[0]
        else:
            coefficient = np.linalg.solve(standardized.T @ standardized + ridge_alpha * np.eye(flat.shape[1]),
                                           standardized.T @ target)
        return cls(baseline, mean, scale, residual_mean, coefficient)

    def predict(self, features, angles):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != self.baseline.time_bins or not np.isfinite(x).all():
            raise ValueError("evaluation feature support does not match the fixed fit")
        residual = ((x - self.feature_mean) / self.feature_scale) @ self.coefficients + self.residual_mean
        return self.baseline.predict(angles) + residual


def lodo_predictions(features, targets, angles, direction_labels, *, harmonics,
                     ridge_alpha, minimum_trials_per_direction):
    """One fixed recipe in one session; no candidate selection or score access.

    Every fold fits full training trials from the seven other directions. The
    caller must have selected this recipe using only allowed outer sessions.
    """
    x, y = np.asarray(features), np.asarray(targets)
    angles, labels = np.asarray(angles), np.asarray(direction_labels)
    if set(np.unique(labels)) != set(range(8)):
        raise ValueError("all eight directions must be present")
    if minimum_trials_per_direction < 1:
        raise ValueError("support must be an explicit positive trial count")
    candidate, comparator = np.empty_like(y, dtype=float), np.empty_like(y, dtype=float)
    models = {}
    for direction in range(8):
        test = labels == direction
        if np.count_nonzero(test) < minimum_trials_per_direction:
            raise ValueError("a complete direction fails the specified support rule")
        train = ~test
        model = RidgeResidual.fit(x[train], angles[train], y[train], labels[train],
                                  harmonics=harmonics, ridge_alpha=ridge_alpha)
        candidate[test] = model.predict(x[test], angles[test])
        comparator[test] = model.baseline.predict(angles[test])
        models[direction] = model
    return {"candidate": candidate, "baseline": comparator, "models": models}


def r2_sse(actual, prediction):
    """Joint XY/whole-trial/native-bin score; negative values remain negative."""
    y, estimate = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != estimate.shape or y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError("score requires matching complete trial×time×XY arrays")
    if not np.isfinite(y).all() or not np.isfinite(estimate).all():
        raise ValueError("nonfinite complete predictions are not silently removed")
    center = y.mean(axis=(0, 1), keepdims=True)
    denominator = np.square(y - center).sum()
    if denominator <= 0:
        raise ValueError("velocity target has no evaluable total variance")
    return float(1.0 - np.square(y - estimate).sum() / denominator)


def matched_electrode_mean(lfp, guide, lower_hz, upper_hz, electrodes):
    """Select by physical guide identity, never by numeric column position."""
    values, mapping = np.asarray(lfp), np.asarray(guide)
    if values.shape[-1] != len(mapping) or mapping.shape[1] != 3:
        raise ValueError("LFP guide and feature axis disagree")
    electrode_set = set(electrodes)
    select = np.array([row[0] in electrode_set and row[1] == lower_hz and row[2] == upper_hz
                       for row in mapping])
    selected_ids = mapping[select, 0]
    if set(selected_ids) != electrode_set or len(selected_ids) != len(electrode_set):
        raise ValueError("selected physical electrodes lack unique support for requested band")
    return values[..., select].mean(axis=-1, keepdims=True)
