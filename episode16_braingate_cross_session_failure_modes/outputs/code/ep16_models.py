"""Batched EP16 model arithmetic: octant weights, source scaling, weighted ridge, scores.

Conventions follow config/analysis_definition.json: summed two-coordinate squared error,
eight half-open octants with weights 1/(8 n_octant), source-only scaling (ddof 0, at least
two finite fitting values, zero/nonfinite scale inactive), missing or inactive coordinates
zero in fitting coordinates. Arrays carry a leading batch axis where noted.
"""
from __future__ import annotations

import numpy as np


class Unresolved(ValueError):
    """A registered quantity cannot be formed (missing octant, nonfinite value)."""


def octant_weights(octant):
    """1/(8 n_o) weights for one packet; raises if an octant is missing."""
    octant = np.asarray(octant, dtype=np.int64)
    if octant.ndim != 1 or not len(octant) or octant.min() < 0 or octant.max() > 7:
        raise Unresolved("Octants must be integers 0-7")
    counts = np.bincount(octant, minlength=8)
    if np.any(counts == 0):
        raise Unresolved("Missing octant in fixed packet")
    return 1.0 / (8.0 * counts[octant])


def fit_scaler(X):
    """X: (..., n, E) raw with NaN missing. Returns center, scale, usable with shape (..., E)."""
    finite = np.isfinite(X)
    count = finite.sum(axis=-2)
    safe = np.where(finite, X, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        center = safe.sum(axis=-2) / count
        dev = np.where(finite, X - center[..., None, :], 0.0)
        scale = np.sqrt((dev ** 2).sum(axis=-2) / count)
    usable = (count >= 2) & np.isfinite(center) & np.isfinite(scale) & (scale > 0)
    center = np.where(usable, center, 0.0)
    scale = np.where(usable, scale, 1.0)
    return center, scale, usable


def standardize(X, center, scale, usable):
    """Apply a fitted scaler; absent, invalid or inactive coordinates are exactly zero."""
    observed = np.isfinite(X) & usable[..., None, :]
    with np.errstate(invalid="ignore"):
        Z = (X - center[..., None, :]) / scale[..., None, :]
    return np.where(observed, Z, 0.0)


def ridge(Z, Y, w, lam):
    """Weighted ridge with unpenalized intercept, batched over the leading axis.

    Z: (m, n, E); Y: (m, n, 2); w: (m, n) summing to one. Returns beta (m, E, 2), c (m, 2).
    Dual form: beta = A^T (A A^T + lam I)^-1 D Yc with A = D Zc, D = sqrt(w).
    """
    zbar = np.einsum("mn,mne->me", w, Z)
    ybar = np.einsum("mn,mnd->md", w, Y)
    d = np.sqrt(w)[..., None]
    A = (Z - zbar[:, None, :]) * d
    B = (Y - ybar[:, None, :]) * d
    K = A @ np.swapaxes(A, 1, 2)
    K[:, np.arange(K.shape[1]), np.arange(K.shape[1])] += lam
    alpha = np.linalg.solve(K, B)
    beta = np.swapaxes(A, 1, 2) @ alpha
    c = ybar - np.einsum("me,med->md", zbar, beta)
    return beta, c


def score(Y, P, w):
    """Direction-balanced negative summed squared error; batched over leading axes."""
    loss = ((Y - P) ** 2).sum(axis=-1)
    return -(w * loss).sum(axis=-1)


def predict_raw(X, center, scale, usable, beta, c):
    """Predictions for raw features X (..., n, E) through a fitted scaler and ridge."""
    return standardize(X, center, scale, usable) @ beta + c[..., None, :]


def effective_df(Z, w, lam):
    """Trace of the weighted ridge hat matrix (intercept excluded), batched."""
    zbar = np.einsum("mn,mne->me", w, Z)
    A = (Z - zbar[:, None, :]) * np.sqrt(w)[..., None]
    K = A @ np.swapaxes(A, 1, 2)
    eig = np.linalg.eigvalsh(K)
    eig = np.clip(eig, 0, None)
    return (eig / (eig + lam)).sum(axis=-1)
