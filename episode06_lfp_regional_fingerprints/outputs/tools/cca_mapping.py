"""Prospective EP06 two-input, two-output regularized CCA mapping primitive.

The caller supplies training electrode-PC2 scores X and training-template
residual spike-PC2 targets Y with identical rows. Native identities, permitted
roles, electrode standardization/PCA, spike PCA and template construction
belong to the surrounding caller. This primitive has no reader, experiment
recipe selection, profile scorer, classifier or outcome-opening authority.

Center X and Y with their training means and form uniform-row covariances
with divisor n. Whiten Cxx+kappa*I and Cyy+kappa*I using symmetric eigenvalue
decompositions. If u is the leading left singular vector of the whitened
cross-covariance, a=(Cxx+kappa*I)^(-1/2)u; fix its sign by making its first
largest-absolute element positive. Do not rescale a or its canonical score.
Fit the full two-coordinate Y from that one score using
.5*mean_rows||Y-t*gamma-intercept||^2 + .5*penalty*||gamma||^2,
with an unpenalized intercept. Prediction applies the frozen complete map.

The requested X rank is exactly two, with numerical rank determined by
eps64*max(n,2)*largest singular value of centered X. No lower-rank fallback,
row/coordinate drop or alternate mode is allowed. Y may have rank one or
zero: positive covariance regularization permits semidefinite Cyy. Tied
canonical singular values use the numerical SVD's first returned vector;
its direction within a tied subspace can differ across numerical libraries.
No positive association or spectral-gap gate is imposed. Exactly zero
cross-covariance gives a valid zero map and the frozen Y mean prediction.

Positive covariance regularization and prediction penalty are explicit caller
arguments; implementing this primitive does not adopt an empirical recipe.
The proposed initial CCA branch supplies .01 for each parameter.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class FittedCCAMapping:
    training_rows: int
    covariance_regularization: float
    prediction_penalty: float
    x_mean: np.ndarray
    y_mean: np.ndarray
    x_whitener: np.ndarray
    y_whitener: np.ndarray
    canonical_singular_values: np.ndarray
    canonical_direction: np.ndarray       # a; no additional normalization
    canonical_score_mean: float           # centering correction at floating-point precision
    score_coefficients: np.ndarray        # gamma: one score -> full two-coordinate Y
    coefficients: np.ndarray              # a[:,None] * gamma[None,:], shape(2,2)
    intercept: np.ndarray                 # raw-X intercept, shape(2,)
    x_singular_values: np.ndarray
    x_numerical_rank: int
    x_rank_tolerance: float


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f"{name} must be an explicit positive finite number")
    return float(value)


def _matrix(value, name):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 2
            or not raw.shape[0] or not np.isfinite(raw).all()):
        raise ValueError(f"{name} must contain finite real rows with exactly two columns")
    with np.errstate(over="ignore", invalid="ignore"):
        return _finite(np.array(raw, dtype=np.float64, copy=True))


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite CCA arithmetic; no fallback or support reduction permitted")
    return value


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _whitener(covariance, regularization):
    regularized = _finite(.5 * (covariance + covariance.T) + regularization * np.eye(2))
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    _finite(eigenvalues)
    _finite(eigenvectors)
    if np.any(eigenvalues <= 0):
        raise FloatingPointError("regularized covariance is not numerically positive definite")
    return _finite((eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T)


def fit_cca_mapping(training_x, training_y, *, covariance_regularization, prediction_penalty, check):
    """Fit one frozen rank-one map from explicitly supplied training rows only."""
    _check(check)
    regularization = _positive(covariance_regularization, "covariance_regularization")
    penalty = _positive(prediction_penalty, "prediction_penalty")
    x, y = _matrix(training_x, "training X"), _matrix(training_y, "training Y")
    if x.shape != y.shape:
        raise ValueError("training X and Y require identical row counts and two coordinates each")
    n = len(x)
    _check(check)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x_mean, y_mean = _finite(np.mean(x, axis=0)), _finite(np.mean(y, axis=0))
        xc, yc = _finite(x - x_mean), _finite(y - y_mean)
        singular_values = _finite(np.linalg.svd(xc, compute_uv=False))
        tolerance = float(np.finfo(np.float64).eps * max(n, 2) * singular_values[0])
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank != 2:
            raise ValueError("training electrode-PC scores do not support the requested rank two")
        cxx, cyy = _finite(xc.T @ xc / n), _finite(yc.T @ yc / n)
        cxy = _finite(xc.T @ yc / n)
        x_whitener = _whitener(cxx, regularization)
        y_whitener = _whitener(cyy, regularization)
        _check(check)
        left, canonical_values, _ = np.linalg.svd(_finite(x_whitener @ cxy @ y_whitener), full_matrices=False)
        _finite(left)
        _finite(canonical_values)
        direction = _finite(x_whitener @ left[:, 0])
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            direction = -direction
        score = _finite(xc @ direction)
        score_mean = float(np.mean(score))
        score_centered = _finite(score - score_mean)
        variance = float(np.dot(score_centered, score_centered) / n)
        if not math.isfinite(variance) or variance <= 0 or not math.isfinite(score_mean):
            raise FloatingPointError("canonical input score has invalid variance despite required rank two")
        gamma = _finite((score_centered @ yc / n) / (variance + penalty))
        coefficients = _finite(direction[:, None] * gamma[None, :])
        intercept = _finite(y_mean - x_mean @ coefficients - score_mean * gamma)
    _check(check)
    return FittedCCAMapping(n, regularization, penalty, _readonly(x_mean), _readonly(y_mean),
        _readonly(x_whitener), _readonly(y_whitener), _readonly(canonical_values), _readonly(direction),
        score_mean, _readonly(gamma), _readonly(coefficients), _readonly(intercept),
        _readonly(singular_values), rank, tolerance)


def predict_cca_mapping(fitted, x, *, check):
    """Predict both output coordinates from features, using only the frozen map."""
    _check(check)
    if not isinstance(fitted, FittedCCAMapping):
        raise TypeError("frozen CCA mapping required")
    for name, shape in (("x_mean", (2,)), ("y_mean", (2,)), ("score_coefficients", (2,)),
                        ("coefficients", (2, 2)), ("intercept", (2,))):
        value = getattr(fitted, name)
        if (not isinstance(value, np.ndarray) or value.dtype.kind != "f"
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError("frozen CCA mapping requires complete finite two-coordinate state")
    if not math.isfinite(fitted.canonical_score_mean):
        raise ValueError("frozen canonical score mean must be finite")
    features = _matrix(x, "prediction X")
    # This is X@coefficients+intercept, evaluated in centered form to avoid
    # cancelling a large frozen feature mean against a large intercept.
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = _finite((features - fitted.x_mean) @ fitted.coefficients
                           + fitted.y_mean - fitted.canonical_score_mean * fitted.score_coefficients)
    _check(check)
    return _readonly(predicted)
