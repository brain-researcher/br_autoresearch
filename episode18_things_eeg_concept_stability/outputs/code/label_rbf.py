"""Training-only registered RBF label kernel with the correct ridge geometry.

The fitted linear-label panel supplies the already-transformed continuous
name/property/category vectors. No identifiers enter distances or the kernel.
For K=U diag(d) U', training features are U sqrt(d), and evaluation features
are K(eval,train) U / sqrt(d). Ridge on these coordinates therefore uses the
retained RKHS kernel, rather than the squared kernel induced by raw K columns.
This array-only component performs no data reads, temporal expansion, EDF
matching, EEG fit, penalty selection or candidate-scientific release.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from training_transforms import FeaturePanelTransform, LABEL, orient_component_signs


BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def _real(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric data")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def _ids(values, *, unique):
    if isinstance(values, (str, bytes)):
        raise ValueError("Supply explicit concept IDs in row order")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError("Concept IDs must be nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError("Training concept IDs must be unique; repeated events are not fitting units")
    return result


def _tolerance(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be an explicit finite nonnegative real tolerance")
    return float(value)


def _rbf_from_distances(distances, sigma):
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise Euclidean distances are not numerically representable")
    # A large finite distance may underflow to the correct zero-kernel limit.
    # Scaling before squaring avoids a needless distance-squared overflow.
    with np.errstate(over="ignore", under="ignore", invalid="raise", divide="raise"):
        values = np.exp(-0.5 * (distances / sigma) ** 2)
    if not np.isfinite(values).all():
        raise ValueError("RBF kernel is not finite")
    return values


@dataclass(frozen=True)
class RBFLabelBasis:
    training_unit_ids: tuple[str, ...]
    family_order: tuple[str, ...]
    family_effective_ranks: tuple[int, ...]
    training_rows: np.ndarray
    median_positive_training_distance: float
    bandwidth_multiplier: float
    bandwidth: float
    eigenvalue_threshold: float
    complete_eigenvalue_spectrum: np.ndarray
    discarded_eigenvalues: np.ndarray
    retained_eigenvalues: np.ndarray
    retained_eigenvectors: np.ndarray
    training_features: np.ndarray
    reconstruction_relative_frobenius_error: float
    reconstruction_max_absolute_error: float

    @property
    def effective_rank(self):
        return len(self.retained_eigenvalues)

    def apply(self, transformed_concept_rows):
        """Apply frozen training bandwidth/eigensystem to already-transformed rows."""
        values = _real(transformed_concept_rows, 2, "transformed label rows")
        if values.shape[1] != self.training_rows.shape[1]:
            raise ValueError("Evaluation columns must match the same transformed label panel")
        kernel = _rbf_from_distances(cdist(values, self.training_rows, metric="euclidean"),
                                     self.bandwidth)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            features = (kernel @ self.retained_eigenvectors) / np.sqrt(self.retained_eigenvalues)
        if not np.isfinite(features).all():
            raise ValueError("RBF evaluation coordinates are not numerically representable")
        return features

    def event_features(self, event_concept_ids):
        """Repeat/reorder fitted concept coordinates for event rows; never refit."""
        ids = _ids(event_concept_ids, unique=False)
        positions = {value: index for index, value in enumerate(self.training_unit_ids)}
        if any(value not in positions for value in ids):
            raise ValueError("Every event concept must belong to the fitted training concept inventory")
        return self.training_features[[positions[value] for value in ids]]


def fit_label_rbf_basis(transformed_training_concepts, *, training_unit_ids,
                        label_panel_transform, bandwidth_multiplier,
                        eigenvalue_rtol, eigenvalue_atol,
                        reconstruction_rtol, reconstruction_max_atol):
    """Fit one of the three registered bandwidths on unique training concepts.

    Explicit rank tolerances define threshold=max(atol, rtol*largest_eigenvalue).
    Eigenvalues strictly above it are retained. Materially negative eigenvalues
    refuse; discarded numerical zeros/negative roundoff remain reported. The
    retained kernel must meet both explicit reconstruction limits. No diagonal
    epsilon, arbitrary bandwidth, extra centering/rescaling or held-out fit is
    introduced. Qualification refers to this finite kernel representation only.

    The fitted panel record verifies required family presence, dimensions and
    training IDs, but cannot authenticate the provenance of caller arrays.
    The caller must use the panel's unchanged training-derived apply method.
    """
    rows = _real(transformed_training_concepts, 2, "transformed training concepts")
    ids = _ids(training_unit_ids, unique=True)
    if len(rows) < 2 or len(ids) != len(rows):
        raise ValueError("Supply at least two unique concept IDs matching training rows")
    panel = label_panel_transform
    if (not isinstance(panel, FeaturePanelTransform) or panel.panel != "label"
            or panel.family_order != LABEL or len(panel.families) != len(LABEL)):
        raise ValueError("Use the fitted label panel retaining all three registered families")
    ranks = tuple(family.effective_rank for family in panel.families)
    if any(family.training_unit_ids != ids for family in panel.families):
        raise ValueError("Kernel training IDs/order must equal the fitted label panel's unique units")
    if rows.shape[1] != sum(ranks):
        raise ValueError("Kernel columns must include every transformed label-family dimension")
    if (isinstance(bandwidth_multiplier, (bool, np.bool_))
            or not isinstance(bandwidth_multiplier, Real)
            or bandwidth_multiplier not in BANDWIDTH_MULTIPLIERS):
        raise ValueError("Use one of the registered bandwidth multipliers 0.5, 1, 2")
    rtol = _tolerance(eigenvalue_rtol, "eigenvalue_rtol")
    atol = _tolerance(eigenvalue_atol, "eigenvalue_atol")
    fit_rtol = _tolerance(reconstruction_rtol, "reconstruction_rtol")
    fit_atol = _tolerance(reconstruction_max_atol, "reconstruction_max_atol")
    if rtol >= 1:
        raise ValueError("Eigenvalue relative tolerance must be less than one")
    pair_distances = pdist(rows, metric="euclidean")
    if not np.isfinite(pair_distances).all():
        raise ValueError("Training pairwise distances are not numerically representable")
    positive = pair_distances[pair_distances > 0.0]
    if not len(positive):
        raise ValueError("No strictly positive training distance; RBF bandwidth is underidentified")
    median = float(np.median(positive))
    bandwidth = float(bandwidth_multiplier * median)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Training-derived bandwidth is not numerically representable")
    kernel = _rbf_from_distances(squareform(pair_distances), bandwidth)
    eigenvalues, vectors = np.linalg.eigh(kernel)
    threshold = max(atol, rtol * float(eigenvalues[-1]))
    if eigenvalues[0] < -threshold:
        raise ValueError("RBF kernel has a materially negative eigenvalue at the supplied tolerance")
    retained = np.flatnonzero(eigenvalues > threshold)[::-1]
    if not len(retained):
        raise ValueError("No RBF direction remains at the supplied numerical rank threshold")
    values = eigenvalues[retained]
    # This fixes only coordinate signs; no rotations, centering or model change.
    directions = orient_component_signs(vectors[:, retained].T).T
    features = directions * np.sqrt(values)
    residual = features @ features.T - kernel
    relative_error = float(np.linalg.norm(residual, ord="fro") / np.linalg.norm(kernel, ord="fro"))
    max_error = float(np.max(np.abs(residual)))
    if (not np.isfinite(relative_error) or not np.isfinite(max_error)
            or relative_error > fit_rtol or max_error > fit_atol):
        raise ValueError("Retained RBF kernel exceeds the explicit reconstruction tolerance")
    return RBFLabelBasis(ids, LABEL, ranks, _readonly(rows), median,
                         float(bandwidth_multiplier), bandwidth, threshold,
                         _readonly(eigenvalues[::-1]), _readonly(eigenvalues[eigenvalues <= threshold][::-1]),
                         _readonly(values), _readonly(directions), _readonly(features), relative_error, max_error)
