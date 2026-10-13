"""Prospective EP06 direction/bin and linear elapsed-time control.

This array/metadata-only helper accepts the caller's frozen initial09 CCA
population context and permitted training counts. It never reads LFP, files,
calibration outcomes or reserved data, selects a model, or fits a classifier.
The stored source fit remains a reference to existing frozen state, not a new
CCA fit. Its population axes and native direction/bin template define exactly
the same residual target as the LFP comparison.

The source inspector requires idx_startTime==1. The exporter stores zero-based
window_start=idx_movement_on-1+5 with .03-second bins. At native position k,
elapsed time from the first-bin origin is .03*(window_start+k); no assertion
about bin centers is needed. Training cell centering removes the common k:
z=.03*(window_start-mean_training_window_start_for_direction). Every native bin
retains its own response and equal row weight. One pooled training ddof0 SD
scales z (zero becomes1), without normalizing evaluation timing.

The fixed .01 ridge predicts both residual population coordinates from this
one scalar with an unpenalized intercept: .5*mean_rows||R-z@B-a||^2 +
.5*.01||B||^2. A linear timing association is the sole baseline scope; success
or failure does not exhaust nonlinear timing or unmeasured behavioral effects.

The direction/bin-only prediction is zero in the training-template residual
space. Its energy score is therefore0 only when the common residual energy is
finite and positive; otherwise both scores are unavailable. No nine identical
band blocks or centered-shape classifier are manufactured from this baseline.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from bandwise_prediction import (BANDS, FittedCCABandwise, _bind_fitted, _check,
                                _finite, _readonly, transform_observations)
from component_scoring import BoundActivity
from development_views import DevelopmentRole


BIN_SECONDS = .03
RIDGE_PENALTY = .01


@dataclass(frozen=True)
class FittedElapsedBaseline:
    source_context: FittedCCABandwise
    training_window_starts: tuple
    training_window_stops: tuple
    start_mean_by_direction: np.ndarray  # [8], original zero-based native bins
    elapsed_scale: float                # pooled training SD in seconds
    coefficients: np.ndarray            # [2], single predictor -> both PCs
    intercept: np.ndarray               # [2], unpenalized
    penalty: float = RIDGE_PENALTY


@dataclass(frozen=True)
class ElapsedBaselineScore:
    region: str
    design: object
    residual_energy: float | None
    squared_error: float | None
    score: float | None
    direction_time_score: float | None
    unavailable_reason: str | None


def _context(context):
    if (not isinstance(context, FittedCCABandwise)
            or context.recipe.feature_family != "pca2_cca" or context.recipe.rank != 2
            or context.recipe.penalty != .01 or context.recipe.covariance_regularization != .01
            or len(context.recipe.blocks) != 9
            or tuple(block.bands for block in context.recipe.blocks) != tuple((band,) for band in BANDS)):
        raise ValueError("the frozen full-nine initial09 rank2 CCA context is required")
    if context.training_design.role != "training":
        raise PermissionError("the source population context must be training-fitted")
    return context


def _metadata(metadata, context, role):
    if not isinstance(metadata, DevelopmentRole):
        raise TypeError("explicit admitted development-role metadata required")
    if metadata.design.role != role:
        raise PermissionError("elapsed baseline requires its exact training/evaluation role")
    design = _bind_fitted(context, metadata.design, metadata.region)
    if metadata.design.time_ids != tuple(range(5, 15)):
        raise ValueError("the frozen ten-bin movement window is required")
    count = len(metadata.design.trial_ids)
    if any(not isinstance(values, tuple) or len(values) != count
           or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                  or value < 0 for value in values)
           for values in (metadata.window_starts, metadata.window_stops)):
        raise ValueError("each whole trial requires explicit integer native window bounds")
    if any(stop - start != 10 for start, stop in zip(metadata.window_starts, metadata.window_stops)):
        raise ValueError("native window bounds must retain all ten bins")
    return design


def _timing(metadata, means, scale):
    starts = _finite(np.asarray(metadata.window_starts, dtype=np.float64))
    values = np.empty(len(starts), dtype=np.float64)
    for index, direction in enumerate(metadata.design.directions):
        rows = np.asarray([label == direction for label in metadata.design.trial_directions])
        values[rows] = (starts[rows] - means[index]) * BIN_SECONDS / scale
    return _finite(np.broadcast_to(values[:, None], (len(starts), 10)).copy())


def fit_elapsed_time_baseline(frozen_initial09_fit, training_metadata, training_counts, *, check):
    """Fit only training counts/timing while reusing the original population map."""
    _check(check)
    context = _context(frozen_initial09_fit)
    _metadata(training_metadata, context, "training")
    if (not isinstance(training_counts, BoundActivity)
            or training_counts.design != training_metadata.design):
        raise ValueError("training counts and elapsed metadata need identical native trial axes")
    # All scope/role/window checks above precede response-array access.
    starts = _finite(np.asarray(training_metadata.window_starts, dtype=np.float64))
    means = []
    for direction in training_metadata.design.directions:
        _check(check)
        rows = [index for index, label in enumerate(training_metadata.design.trial_directions)
                if label == direction]
        values = starts[rows]
        # Exact constant timing must not acquire a tiny numerical variance
        # when a floating-point mean sums many copies of the same integer.
        means.append(values[0] if np.all(values == values[0]) else np.mean(values))
    means = _finite(np.asarray(means))
    timing = _timing(training_metadata, means, 1.)
    scale = float(np.std(timing, ddof=0))
    if not math.isfinite(scale):
        raise FloatingPointError("nonfinite training elapsed-time scale")
    if scale == 0:
        scale = 1.
    x = _finite(timing.reshape(-1, 1) / scale)
    target = transform_observations(context, training_counts, region=context.region, check=check).values
    target = target.reshape(-1, 2)
    _check(check)
    with np.errstate(over="ignore", invalid="ignore"):
        x_mean, y_mean = np.mean(x, axis=0), np.mean(target, axis=0)
        a = _finite(np.vstack(((x - x_mean) / math.sqrt(len(x)), [[math.sqrt(RIDGE_PENALTY)]])))
        b = _finite(np.vstack(((target - y_mean) / math.sqrt(len(x)), np.zeros((1, 2)))))
        coefficients = _finite(np.linalg.lstsq(a, b, rcond=None)[0])
        intercept = _finite(y_mean - x_mean @ coefficients)
    _check(check)
    return FittedElapsedBaseline(context, training_metadata.window_starts, training_metadata.window_stops,
        _readonly(means), scale, _readonly(coefficients[0]), _readonly(intercept))


def predict_elapsed_time_baseline(fitted, evaluation_metadata, *, check):
    """One genuine frozen prediction per region, with no outcome/count input."""
    _check(check)
    if not isinstance(fitted, FittedElapsedBaseline) or fitted.penalty != RIDGE_PENALTY:
        raise TypeError("the fixed fitted elapsed-time baseline is required")
    context = _context(fitted.source_context)
    design = _metadata(evaluation_metadata, context, "evaluation")
    if (fitted.start_mean_by_direction.shape != (8,) or fitted.coefficients.shape != (2,)
            or fitted.intercept.shape != (2,) or not math.isfinite(fitted.elapsed_scale)
            or fitted.elapsed_scale <= 0):
        raise ValueError("frozen timing state has inconsistent dimensions or scaling")
    timing = _timing(evaluation_metadata, fitted.start_mean_by_direction, fitted.elapsed_scale)
    with np.errstate(over="ignore", invalid="ignore"):
        result = _finite(timing[:, :, None] * fitted.coefficients + fitted.intercept)
    _check(check)
    return BoundActivity(design, _readonly(result))


def score_elapsed_time_baseline(fitted, observed, predicted, *, check):
    """Use the same common residual energy as the LFP score; no clipping or epsilon."""
    _check(check)
    if not isinstance(fitted, FittedElapsedBaseline):
        raise TypeError("one frozen elapsed-time fit is required")
    context = _context(fitted.source_context)
    if not isinstance(observed, BoundActivity) or observed.design.role != "evaluation":
        raise PermissionError("baseline scoring accepts evaluation counts only")
    expected = _bind_fitted(context, observed.design, context.region)
    if not isinstance(predicted, BoundActivity) or predicted.design != expected:
        raise ValueError("the single baseline prediction must retain the complete frozen evaluation design")
    raw = np.asarray(predicted.values)
    if raw.dtype.kind not in "fiu" or raw.shape != expected.shape:
        raise ValueError("a complete numeric baseline prediction is required")
    try:
        residual = transform_observations(context, observed, region=context.region, check=check).values
    except FloatingPointError:
        return ElapsedBaselineScore(context.region, expected, None, None, None, None,
                                    "nonfinite_residual_energy")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        energy = float(np.sum(np.square(residual), dtype=np.float64))
        error = float(np.sum(np.square(residual - raw), dtype=np.float64))
    _check(check)
    if not math.isfinite(energy) or energy <= 0:
        return ElapsedBaselineScore(context.region, expected, energy if math.isfinite(energy) else None,
            None, None, None, "zero_residual_energy" if energy == 0 else "nonfinite_residual_energy")
    score = 1. - error / energy
    if not math.isfinite(error) or not math.isfinite(score):
        return ElapsedBaselineScore(context.region, expected, energy,
            error if math.isfinite(error) else None, None, 0., "nonfinite_prediction_error_or_score")
    return ElapsedBaselineScore(context.region, expected, energy, error, score, 0., None)
