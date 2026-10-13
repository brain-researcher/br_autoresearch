"""EP18 array-only training transforms and fixed full-rank sensor coordinates.

Feature conventions implement config/feature_registry.json exactly. Fit rows
must be unique permitted training images/concepts, never trial-weighted copies.
The caller refits inside each inner/outer training fold and supplies the frozen
channel order. This module has no data reader or access-isolation mechanism.
Covariance centering, normalization, weights and shrinkage are explicit caller
choices, not new registered scientific defaults or EEG-outcome tuning here.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np


CORE_IMAGE = ("low_level_851", "supervised_resnet50_v1", "dino_resnet50",
              "clip_vit_b32_image", "blip_caption_clip_text")
EXPANDED_IMAGE = CORE_IMAGE + ("dinov2_vit_b14",)
LABEL = ("concept_name_clip_text", "human_properties_12", "human_categories_53")
PANELS = {"core_image": CORE_IMAGE, "expanded_image": EXPANDED_IMAGE, "label": LABEL}
DIMENSIONS = {"low_level_851": 851, "supervised_resnet50_v1": 2048,
              "dino_resnet50": 2048, "clip_vit_b32_image": 512,
              "blip_caption_clip_text": 512, "dinov2_vit_b14": 1536,
              "concept_name_clip_text": 512, "human_properties_12": 12,
              "human_categories_53": 53}
RANK_GRID = (32, 64, 128)


def _real(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric data")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and {ndim}-dimensional")
    return result


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _ids(values, count, *, unique):
    if isinstance(values, (str, bytes)):
        raise ValueError("Supply one explicit unit ID per row")
    result = tuple(values)
    if len(result) != count or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("Unit IDs must be nonempty strings matching the rows")
    if unique and len(set(result)) != len(result):
        raise ValueError("Training unit IDs must be unique; no implicit trial weighting")
    return result


def orient_component_signs(loadings):
    """Loading rows: largest absolute entry positive, first index breaks ties."""
    result = _real(loadings, 2, "loadings")
    if result.shape[1] == 0:
        raise ValueError("Loadings need original feature columns")
    if len(result):
        pivot = np.argmax(np.abs(result), axis=1)
        signs = np.where(result[np.arange(len(result)), pivot] < 0., -1., 1.)
        result *= signs[:, None]
    return result


@dataclass(frozen=True)
class FeatureTransform:
    training_unit_ids: tuple[str, ...]
    mean: np.ndarray
    population_sd: np.ndarray
    zero_sd_columns: tuple[int, ...]
    loadings: np.ndarray
    singular_values: np.ndarray
    rank_threshold: float
    numerical_rank: int
    requested_rank: int
    effective_rank: int

    def apply(self, rows):
        """Use training statistics unchanged, including zero-SD→zero columns."""
        values = _real(rows, 2, "feature rows")
        if values.shape[1] != len(self.mean):
            raise ValueError("Feature dimension differs from the fitted family")
        scaled = np.zeros_like(values)
        retained = self.population_sd != 0.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaled[:, retained] = ((values[:, retained] - self.mean[retained])
                                   / self.population_sd[retained])
            transformed = scaled @ self.loadings.T
        if not np.isfinite(transformed).all():
            raise ValueError("Nonfinite transformed features")
        return transformed


def fit_feature_transform(training_rows, *, training_unit_ids, requested_rank):
    """Fit the registered population scaling and full float64 SVD, without whitening."""
    rows = _real(training_rows, 2, "training features")
    samples, dimensions = rows.shape
    if samples < 2 or dimensions < 1:
        raise ValueError("Training features need at least two unique rows and one column")
    ids = _ids(training_unit_ids, samples, unique=True)
    if (isinstance(requested_rank, (bool, np.bool_))
            or not isinstance(requested_rank, Integral) or requested_rank not in RANK_GRID):
        raise ValueError("Requested rank must be one of the registered 32, 64, 128")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = rows.mean(axis=0)
        scale = rows.std(axis=0, ddof=0)
        # Equal represented values have exactly zero population variance;
        # avoid a roundoff mean manufacturing variance in a constant column.
        constant = np.all(rows == rows[0], axis=0)
        mean[constant], scale[constant] = rows[0, constant], 0.
        retained = scale != 0.
        scaled = np.zeros_like(rows)
        scaled[:, retained] = (rows[:, retained] - mean[retained]) / scale[retained]
    if not np.isfinite(scaled).all():
        raise ValueError("Training scaling is not numerically representable")
    # Complete deterministic SVD, not randomized/truncated PCA. Economy-shaped
    # factors retain every singular value while avoiding an unused n-by-n U.
    _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    threshold = float(max(scaled.shape) * np.finfo(np.float64).eps * singular[0])
    numerical_rank = int(np.count_nonzero(singular > threshold))
    effective = min(int(requested_rank), dimensions, samples - 1, numerical_rank)
    loadings = orient_component_signs(right[:effective])
    return FeatureTransform(ids, _readonly(mean), _readonly(scale),
                            tuple(int(i) for i in np.flatnonzero(~retained)),
                            _readonly(loadings), _readonly(singular), threshold,
                            numerical_rank, int(requested_rank), effective)


@dataclass(frozen=True)
class FeaturePanelTransform:
    panel: str
    family_order: tuple[str, ...]
    families: tuple[FeatureTransform, ...]

    def apply(self, feature_rows: Mapping[str, np.ndarray], *, row_ids: Mapping[str, Sequence[str]]):
        """Concatenate every registered family in order; zero-rank records persist."""
        expected = set(self.family_order)
        if set(feature_rows) != expected or set(row_ids) != expected:
            raise ValueError("Every registered family must remain in the panel")
        transformed, first_ids = [], None
        for name, transform in zip(self.family_order, self.families):
            result = transform.apply(feature_rows[name])
            ids = _ids(row_ids[name], len(result), unique=False)
            if first_ids is not None and ids != first_ids:
                raise ValueError("Family rows must align by the same explicit unit IDs")
            first_ids = ids
            transformed.append(result)
        return np.concatenate(transformed, axis=1)


def fit_feature_panel(training_families, *, training_unit_ids, requested_ranks, panel):
    """Fit each required image or label family separately, on identical unique units."""
    if panel not in PANELS:
        raise ValueError("Use a registered core_image, expanded_image or label panel")
    order = PANELS[panel]
    expected = set(order)
    if any(set(mapping) != expected for mapping in
           (training_families, training_unit_ids, requested_ranks)):
        raise ValueError("Every registered family needs training rows, IDs and its rank choice")
    fitted, first_ids = [], None
    for name in order:
        transform = fit_feature_transform(training_families[name],
                                           training_unit_ids=training_unit_ids[name],
                                           requested_rank=requested_ranks[name])
        if len(transform.mean) != DIMENSIONS[name]:
            raise ValueError(f"{name} differs from its registered raw feature dimension")
        if first_ids is not None and transform.training_unit_ids != first_ids:
            raise ValueError("All families in a panel need the same training units/order")
        first_ids = transform.training_unit_ids
        fitted.append(transform)
    return FeaturePanelTransform(panel, order, tuple(fitted))


class CommonSensorSpace:
    """Fixed 62→61 Helmert contrasts in the caller's frozen common-channel order.

    Column j-1 (j=1..61) has 1/sqrt(j(j+1)) in rows 0..j-1 and
    -j/sqrt(j(j+1)) in row j. No signal-dependent sign/order, bad-channel
    deletion, interpolation or participant-specific basis is used.
    """
    def __init__(self, frozen_channel_order):
        self.channel_order = _ids(frozen_channel_order, 62, unique=True)
        basis = np.zeros((62, 61))
        for j in range(1, 62):
            denominator = np.sqrt(j * (j + 1.))
            basis[:j, j - 1] = 1. / denominator
            basis[j, j - 1] = -j / denominator
        self.basis = _readonly(basis)

    def apply(self, samples, *, channel_order):
        values = _real(samples, 2, "sensor samples")
        names = _ids(channel_order, values.shape[1], unique=True)
        if not set(self.channel_order).issubset(names):
            raise ValueError("All 62 frozen channels are required; no interpolation or deletion")
        selected = values[:, [names.index(name) for name in self.channel_order]]
        with np.errstate(over="raise", invalid="raise"):
            selected -= selected.mean(axis=1, keepdims=True)
            result = selected @ self.basis
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite fixed sensor transform")
        return result


@dataclass(frozen=True)
class ResidualWhitener:
    residual_mean_for_covariance: np.ndarray
    empirical_covariance: np.ndarray
    shrinkage_target: np.ndarray
    shrinkage_strength: float
    covariance_denominator: float
    covariance_centered: bool
    regularized_covariance: np.ndarray
    inverse_root: np.ndarray

    def apply(self, sensor_contrasts):
        """Apply the same training metric; do not residualize or recenter EEG."""
        values = _real(sensor_contrasts, 2, "61D sensor contrasts")
        if values.shape[1] != 61:
            raise ValueError("Whitening requires the fixed full-rank 61D sensor space")
        with np.errstate(over="raise", invalid="raise"):
            result = values @ self.inverse_root
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite whitened contrasts")
        return result


def fit_residual_whitener(training_nuisance_residuals, *, sample_weights,
                          center_covariance, covariance_denominator,
                          shrinkage_target, shrinkage_strength):
    """Fit W from training nuisance residuals, with every covariance choice explicit.

    S = R_centered' diag(weights) R_centered / covariance_denominator;
    C = (1-alpha) S + alpha target; W = C^(-1/2), symmetric.
    The caller supplies only training nuisance-model residuals in 61D and
    freezes/reuses this same metric across candidate models in the fold. Target
    and alpha must be outcome-blind or training-derived; this array-only API
    cannot authenticate provenance. No alpha/target/normalization is selected,
    no eigenvalues are clipped, and no singular 62D covariance is inverted.
    Covariance centering affects S only, not subsequent response/prediction
    application. The empirical joint fit still uses the original EEG targets.
    """
    residuals = _real(training_nuisance_residuals, 2, "training nuisance residuals")
    if residuals.shape[1] != 61 or len(residuals) < 1:
        raise ValueError("Supply nonempty training nuisance residuals in fixed 61D space")
    weights = _real(sample_weights, 1, "covariance sample weights")
    if weights.shape != (len(residuals),) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Covariance weights need nonnegative values and retained training support")
    if not isinstance(center_covariance, bool):
        raise ValueError("Covariance centering must be an explicit Boolean choice")
    for name, value in (("covariance denominator", covariance_denominator),
                        ("shrinkage strength", shrinkage_strength)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
    if covariance_denominator <= 0 or not 0 <= shrinkage_strength <= 1:
        raise ValueError("Denominator must be positive and shrinkage strength in [0,1]")
    target = _real(shrinkage_target, 2, "shrinkage target")
    if target.shape != (61, 61) or not np.array_equal(target, target.T):
        raise ValueError("Supply an explicitly symmetric 61-by-61 shrinkage target")
    if np.linalg.eigvalsh(target)[0] < 0:
        raise ValueError("Shrinkage target must be positive semidefinite; no eigenvalue repair")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        weight_sum = weights.sum()
        mean = (weights @ residuals / weight_sum if center_covariance else np.zeros(61))
        centered = residuals - mean
        weighted = np.sqrt(weights)[:, None] * centered
        covariance = weighted.T @ weighted / covariance_denominator
        regularized = (1. - shrinkage_strength) * covariance + shrinkage_strength * target
    if not np.isfinite(regularized).all():
        raise ValueError("Covariance is not numerically representable")
    eigenvalues, vectors = np.linalg.eigh(regularized)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise ValueError("Regularized covariance must be positive definite; no eigenvalue clipping")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        inverse_root = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    if not np.isfinite(inverse_root).all():
        raise ValueError("Covariance inverse root is not numerically representable")
    return ResidualWhitener(_readonly(mean), _readonly(covariance), _readonly(target),
                            float(shrinkage_strength), float(covariance_denominator),
                            center_covariance, _readonly(regularized), _readonly(inverse_root))
