"""Actual count-likelihood T/U/M models for EP12 generated-data qualification.

The models in this module all score the same canonical partner-count vector
under a conditional multinomial likelihood.  Fitting sees only a transfer's
source side.  Target latent coordinates and mixture labels are integrated out;
there is no target-side fitting, clustering, or component alignment.

This module deliberately depends on Sherlock's pinned scientific Python
modules.  The synthetic control-plane package remains standard-library-only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .canonical_target import CanonicalDecoder, count_matrix

try:
    import numpy as np
    from scipy.special import gammaln, logsumexp
    from scipy.stats import norm, qmc
    from sklearn.covariance import LedoitWolf
    from sklearn.mixture import GaussianMixture
except ImportError as exc:  # pragma: no cover - exercised by CLI environment guard
    raise RuntimeError(
        "scientific models require the pinned Sherlock Python/numpy/scipy/"
        "scikit-learn module environment"
    ) from exc


Array = np.ndarray


BOUNDED_LATENT_SUPPORT_PROTOCOL = "ep12.pure_bounded_latent_support.v1"
MAX_LINEAR_LATENT_RANK = 2
CURVED_LATENT_DIMENSION = 1
CURVED_HERMITE_DEGREE = 3


class ScientificModelError(RuntimeError):
    """The scientific fitter or predictive integrator failed closed."""


def seed_from(*parts: object) -> int:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def _require_finite(name: str, value: Array) -> None:
    if not np.all(np.isfinite(value)):
        raise ScientificModelError(f"{name} contains non-finite values")


def validate_profile_arrays(counts: Array, covariates: Array) -> tuple[Array, Array]:
    counts = np.asarray(counts)
    covariates = np.asarray(covariates, dtype=float)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] < 3:
        raise ScientificModelError("counts must be a neuron-by-partner matrix")
    if covariates.ndim != 2 or covariates.shape[0] != counts.shape[0]:
        raise ScientificModelError("covariates must have one row per neuron")
    if not np.issubdtype(counts.dtype, np.integer):
        if not np.all(np.equal(counts, np.round(counts))):
            raise ScientificModelError("counts must be integer-valued")
        counts = np.round(counts).astype(np.int64)
    else:
        counts = counts.astype(np.int64, copy=False)
    if np.any(counts < 0):
        raise ScientificModelError("counts cannot be negative")
    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise ScientificModelError("every scored neuron needs positive observed mass")
    _require_finite("covariates", covariates)
    return counts, covariates


def helmert_ilr_basis(partners: int) -> Array:
    """Return a deterministic orthonormal basis for the sum-zero simplex."""

    if partners < 2:
        raise ScientificModelError("ILR needs at least two partner bins")
    basis = np.zeros((partners, partners - 1), dtype=float)
    for column in range(partners - 1):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    if not np.allclose(basis.T @ basis, np.eye(partners - 1), atol=1e-12):
        raise ScientificModelError("failed to construct an orthonormal ILR basis")
    return basis


def counts_to_ilr(counts: Array, basis: Array, pseudocount: float = 0.5) -> Array:
    if pseudocount <= 0:
        raise ScientificModelError("fit pseudocount must be positive")
    smoothed = counts.astype(float) + pseudocount
    composition = smoothed / smoothed.sum(axis=1, keepdims=True)
    return np.log(composition) @ basis


def _design(covariates: Array) -> Array:
    return np.column_stack([np.ones(covariates.shape[0]), covariates])


def _ridge_regression(design: Array, target: Array, penalty: float) -> Array:
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0]) * penalty
    regularizer[0, 0] = 0.0
    try:
        return np.linalg.solve(gram + regularizer, design.T @ target)
    except np.linalg.LinAlgError as exc:
        raise ScientificModelError("nuisance regression is singular") from exc


def _bounded_covariance(
    covariance: Array,
    *,
    latent_rank: int,
) -> Array:
    """Return a PSD covariance supported on at most ``latent_rank`` axes.

    No eigenvalue floor is added: a direction unsupported by the fitted data
    stays absent rather than becoming either latent or ambient variation.
    """

    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    covariance = (covariance + covariance.T) / 2.0
    dimension = covariance.shape[0]
    rank = max(0, min(int(latent_rank), dimension))
    if rank == 0:
        return np.zeros((dimension, dimension), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    retained = eigenvalues[:rank]
    result = (eigenvectors[:, :rank] * retained) @ eigenvectors[:, :rank].T
    return (result + result.T) / 2.0


def _shrunk_covariance(values: Array, latent_rank: int) -> Array:
    dimension = values.shape[1]
    if values.shape[0] <= 1:
        covariance = np.zeros((dimension, dimension), dtype=float)
    else:
        covariance = np.cov(values, rowvar=False, ddof=1)
    return _bounded_covariance(
        covariance,
        latent_rank=latent_rank,
    )


def _covariance_for_branch(
    values: Array,
    latent_rank: int,
    covariance_mode: str,
) -> Array:
    """Estimate one of the three covariance branches declared in the policy."""

    dimension = values.shape[1]
    if values.shape[0] <= 1:
        return _bounded_covariance(
            np.zeros((dimension, dimension), dtype=float),
            latent_rank=latent_rank,
        )
    if covariance_mode == "diagonal":
        variances = np.var(values, axis=0, ddof=1)
        return _bounded_covariance(
            np.diag(np.maximum(variances, 0.0)),
            latent_rank=latent_rank,
        )
    if covariance_mode == "shrinkage":
        covariance = LedoitWolf(assume_centered=False).fit(values).covariance_
        return _bounded_covariance(
            covariance,
            latent_rank=latent_rank,
        )
    if covariance_mode == "low_rank":
        return _shrunk_covariance(values, latent_rank)
    raise ScientificModelError(f"unknown covariance branch: {covariance_mode}")


def _representation_residuals(
    counts: Array,
    residuals: Array,
    basis: Array,
    *,
    representation: str,
    latent_rank: int,
    pseudocount: float,
) -> tuple[Array, dict[str, Any]]:
    """Apply a bounded fitting representation in the common ILR space.

    Every branch is decoded and scored against the same raw multinomial count
    vector.  The composition branch retains a rank-bounded empirical subspace;
    the count-model branch additionally removes the mean first-order sampling
    variance before reconstructing that subspace.
    """

    diagnostics: dict[str, Any] = {
        "representation": representation,
        "representation_rank_definition": "representation_latent_effective_rank",
    }
    if representation == "count_aware_log_ratio":
        diagnostics["representation_rank"] = max(
            1, min(residuals.shape[1], residuals.shape[0] - 1)
        )
        return residuals, diagnostics
    if representation not in {"low_rank_composition", "low_rank_count_model"}:
        raise ScientificModelError(f"unknown representation branch: {representation}")

    centered = residuals - residuals.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance).astype(float)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    rank = max(1, min(int(latent_rank), centered.shape[1], centered.shape[0] - 1))
    retained_values = eigenvalues[:rank].copy()
    if representation == "low_rank_count_model":
        smoothed = counts.astype(float) + pseudocount
        noise_traces = []
        for row in smoothed:
            measurement = basis.T @ np.diag(1.0 / row) @ basis
            noise_traces.append(float(np.trace(measurement)) / measurement.shape[0])
        noise_floor = float(np.mean(noise_traces))
        retained_values = np.maximum(retained_values - noise_floor, 0.0)
        diagnostics["mean_measurement_variance_removed"] = noise_floor
    scales = np.zeros(rank, dtype=float)
    positive = eigenvalues[:rank] > 1e-12
    scales[positive] = np.sqrt(retained_values[positive] / eigenvalues[:rank][positive])
    scores = centered @ eigenvectors[:, :rank]
    reconstructed = (scores * scales) @ eigenvectors[:, :rank].T
    diagnostics.update(
        {
            "representation_rank": rank,
            "empirical_eigenvalues": eigenvalues[:rank].tolist(),
            "retained_eigenvalues": retained_values.tolist(),
        }
    )
    return reconstructed, diagnostics


def _normal_qmc(draws: int, dimension: int, seed: int) -> Array:
    """Nested scrambled-Sobol standard-normal draws."""

    if draws < 1 or dimension < 1:
        raise ScientificModelError("predictive integration needs draws >= 1 and dimension >= 1")
    exponent = int(math.ceil(math.log2(draws)))
    uniforms = qmc.Sobol(
        dimension, scramble=True, seed=int(seed)
    ).random_base2(exponent)
    uniforms = np.clip(uniforms[:draws], 1e-9, 1.0 - 1e-9)
    return norm.ppf(uniforms)


def _cubic_hermite_basis(z: Array) -> Array:
    values = np.asarray(z, dtype=float).reshape(-1)
    return np.column_stack(
        [values, values**2 - 1.0, values**3 - 3.0 * values]
    )


def _covariance_root(covariance: Array) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ScientificModelError("latent covariance is not positive semidefinite")
    eigenvalues = np.where(eigenvalues > 1e-12 * scale, eigenvalues, 0.0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def _log_gaussian_density(points: Array, mean: Array, covariance: Array) -> Array:
    points = np.atleast_2d(points).astype(float)
    covariance = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    centered = (points - np.asarray(mean, dtype=float)[None, :]) @ eigenvectors
    quadratic = np.sum(centered**2 / eigenvalues[None, :], axis=1)
    log_determinant = float(np.sum(np.log(eigenvalues)))
    dimension = points.shape[1]
    return -0.5 * (dimension * math.log(2.0 * math.pi) + log_determinant + quadratic)


def _log_mixture_density(
    points: Array,
    means: Array,
    covariances: Array,
    weights: Array,
) -> Array:
    terms = [
        math.log(float(weight)) + _log_gaussian_density(points, mean, covariance)
        for mean, covariance, weight in zip(means, covariances, weights)
    ]
    return logsumexp(np.vstack(terms), axis=0)


def _draw_gaussian_mixture(
    means: Array,
    covariances: Array,
    weights: Array,
    *,
    draws: int,
    seed: int,
) -> Array:
    pieces: list[Array] = []
    exact = np.asarray(weights, dtype=float) * draws
    allocations = np.floor(exact).astype(int)
    remaining = int(draws - allocations.sum())
    remainder_order = np.argsort(-(exact - allocations), kind="stable")
    allocations[remainder_order[:remaining]] += 1
    if allocations.sum() != draws:
        raise ScientificModelError("mixture draw allocation did not conserve draws")
    for index, allocation in enumerate(allocations):
        if allocation == 0:
            continue
        component_normals = _normal_qmc(
            int(allocation), means.shape[1], seed_from(seed, index)
        )
        pieces.append(
            means[index][None, :]
            + component_normals @ _covariance_root(covariances[index]).T
        )
    if not pieces:
        raise ScientificModelError("mixture draw allocation produced no samples")
    return np.vstack(pieces)


@dataclass(frozen=True)
class FittedPredictiveModel:
    model_id: str
    basis: Array
    nuisance_coefficients: Array
    latent_offsets: Array
    log_weights: Array
    prior_means: Array
    prior_covariances: Array
    prior_weights: Array
    integration_draws: int
    integration_seed: int
    parameter_count: int
    diagnostics: Mapping[str, Any]
    pseudocount: float
    canonical_decoder: CanonicalDecoder | None = None

    def validated_latent_support(self) -> tuple[Array, Array]:
        """Return the exact finite support and normalized log weights used.

        ``latent_offsets`` is the deterministic integration rule for this
        fitted model.  Scoring and fitted-null generation must both consume
        this same object rather than reconstructing a different Gaussian law.
        """

        offsets = np.asarray(self.latent_offsets, dtype=float)
        log_weights = np.asarray(self.log_weights, dtype=float)
        dimension = self.basis.shape[1]
        if (
            offsets.ndim != 2
            or offsets.shape[0] < 1
            or offsets.shape[1] != dimension
        ):
            raise ScientificModelError("latent integration support is malformed")
        if log_weights.shape != (offsets.shape[0],):
            raise ScientificModelError("latent integration weights are malformed")
        _require_finite("latent integration support", offsets)
        _require_finite("latent integration log weights", log_weights)
        if not np.isclose(float(logsumexp(log_weights)), 0.0, atol=1e-10):
            raise ScientificModelError("latent integration weights are not normalized")
        return offsets, log_weights

    def score_neurons(self, counts: Array, covariates: Array) -> Array:
        counts, covariates = validate_profile_arrays(counts, covariates)
        if counts.shape[1] != self.basis.shape[0]:
            raise ScientificModelError("target partner vocabulary differs from source")
        design = _design(covariates)
        if design.shape[1] != self.nuisance_coefficients.shape[0]:
            raise ScientificModelError("target nuisance schema differs from source")
        base_ilr = design @ self.nuisance_coefficients
        totals = counts.sum(axis=1).astype(float)
        result = np.empty(counts.shape[0], dtype=float)
        latent_support: tuple[Array, Array] | None = None
        if self.model_id != "T":
            latent_support = self.validated_latent_support()
        for index, (row, base, total) in enumerate(zip(counts, base_ilr, totals)):
            coefficient = gammaln(total + 1.0) - np.sum(gammaln(row.astype(float) + 1.0))
            if self.model_id == "T":
                clr_logits = base @ self.basis.T
                log_probabilities = clr_logits - logsumexp(clr_logits)
                integrated = float(coefficient + log_probabilities @ row.astype(float))
                result[index] = integrated / total
                continue

            assert latent_support is not None
            residual_draws, support_log_weights = latent_support
            clr_logits = (base[None, :] + residual_draws) @ self.basis.T
            clr_logits -= np.max(clr_logits, axis=1, keepdims=True)
            log_probabilities = clr_logits - logsumexp(clr_logits, axis=1, keepdims=True)
            log_likelihoods = coefficient + log_probabilities @ row.astype(float)
            integrated = float(logsumexp(log_likelihoods + support_log_weights))
            result[index] = integrated / total
        _require_finite("held-out predictive scores", result)
        return result

    def mean_score(self, counts: Array, covariates: Array) -> float:
        return float(np.mean(self.score_neurons(counts, covariates)))


def _fit_common(
    counts: Array,
    covariates: Array,
    *,
    ridge: float,
    pseudocount: float,
    representation: str,
    latent_rank: int,
) -> tuple[Array, Array, Array, dict[str, Any]]:
    counts, covariates = validate_profile_arrays(counts, covariates)
    basis = helmert_ilr_basis(counts.shape[1])
    ilr = counts_to_ilr(counts, basis, pseudocount=pseudocount)
    coefficients = _ridge_regression(_design(covariates), ilr, ridge)
    residuals = ilr - _design(covariates) @ coefficients
    residuals, diagnostics = _representation_residuals(
        counts,
        residuals,
        basis,
        representation=representation,
        latent_rank=latent_rank,
        pseudocount=pseudocount,
    )
    return basis, coefficients, residuals, diagnostics


def fit_predictive_model(
    counts: Array,
    covariates: Array,
    model_id: str,
    *,
    seed: int,
    integration_seed: int | None = None,
    integration_draws: int = 64,
    latent_rank: int = 2,
    ridge: float = 0.1,
    pseudocount: float = 0.5,
    representation: str = "count_aware_log_ratio",
    covariance_mode: str = "shrinkage",
    covariance_rank: int | None = None,
) -> FittedPredictiveModel:
    counts, covariates = validate_profile_arrays(counts, covariates)
    bounded_latent_rank = max(
        1,
        min(int(latent_rank), MAX_LINEAR_LATENT_RANK),
    )
    resolved_integration_seed = int(
        seed if integration_seed is None else integration_seed
    )
    basis, coefficients, residuals, representation_diagnostics = _fit_common(
        counts,
        covariates,
        ridge=ridge,
        pseudocount=pseudocount,
        representation=representation,
        latent_rank=bounded_latent_rank,
    )
    dimension = residuals.shape[1]
    offsets: Array
    weights: Array
    diagnostics: dict[str, Any] = dict(representation_diagnostics)
    diagnostics["covariance_mode"] = covariance_mode
    requested_covariance_rank = (
        bounded_latent_rank if covariance_rank is None else int(covariance_rank)
    )
    fitted_covariance_rank = max(
        1,
        min(requested_covariance_rank, bounded_latent_rank),
    )
    diagnostics["latent_support_protocol"] = BOUNDED_LATENT_SUPPORT_PROTOCOL
    diagnostics["ambient_residual_tube"] = False
    diagnostics["covariance_rank"] = fitted_covariance_rank
    prior_means: Array
    prior_covariances: Array
    prior_weights: Array

    if model_id == "T":
        offsets = np.zeros((1, dimension), dtype=float)
        weights = np.ones(1, dtype=float)
        prior_means = np.zeros((1, dimension), dtype=float)
        prior_covariances = np.repeat(
            (np.eye(dimension) * 1e-10)[None, :, :], 1, axis=0
        )
        prior_weights = np.ones(1, dtype=float)
        parameter_count = coefficients.size
    elif model_id == "U_linear":
        covariance = _covariance_for_branch(
            residuals, fitted_covariance_rank, covariance_mode
        )
        normals = _normal_qmc(
            integration_draws, dimension, resolved_integration_seed
        )
        offsets = normals @ _covariance_root(covariance).T
        weights = np.full(offsets.shape[0], 1.0 / offsets.shape[0])
        prior_means = np.zeros((1, dimension), dtype=float)
        prior_covariances = covariance[None, :, :]
        prior_weights = np.ones(1, dtype=float)
        parameter_count = coefficients.size + dimension * min(
            fitted_covariance_rank, dimension
        )
        support_rank = int(
            np.linalg.matrix_rank(
                offsets - offsets.mean(axis=0, keepdims=True),
                tol=1e-10,
            )
        )
        if support_rank > MAX_LINEAR_LATENT_RANK:
            raise ScientificModelError("U_linear support exceeds rank two")
        diagnostics["latent_support_family"] = "rank_bounded_Gaussian_manifold"
        diagnostics["latent_support_rank"] = support_rank
        diagnostics["covariance_eigenvalues"] = np.linalg.eigvalsh(covariance).tolist()
    elif model_id == "U_curved":
        covariance = _covariance_for_branch(residuals, 1, covariance_mode)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        raw_z = residuals @ direction
        scale = float(np.std(raw_z, ddof=1)) if len(raw_z) > 1 else 1.0
        scale = max(scale, 1e-6)
        z = raw_z / scale
        hermite = _cubic_hermite_basis(z)
        curve = _ridge_regression(
            np.column_stack([np.ones(len(z)), hermite]), residuals, ridge
        )[1:, :]
        quadrature_z, quadrature_weights = np.polynomial.hermite_e.hermegauss(9)
        quadrature_weights = quadrature_weights / math.sqrt(2.0 * math.pi)
        node_h = _cubic_hermite_basis(quadrature_z)
        prior_means = node_h @ curve
        prior_covariances = np.zeros(
            (len(quadrature_z), dimension, dimension), dtype=float
        )
        prior_weights = quadrature_weights / quadrature_weights.sum()
        offsets = np.array(prior_means, copy=True)
        weights = np.array(prior_weights, copy=True)
        parameter_count = coefficients.size + curve.size
        diagnostics["latent_support_family"] = (
            "one_dimensional_Gaussian_cubic_Hermite_curve"
        )
        diagnostics["latent_dimension"] = CURVED_LATENT_DIMENSION
        diagnostics["curve_degree"] = CURVED_HERMITE_DEGREE
        diagnostics["integration_rule"] = "nine_node_Gauss_Hermite"
        diagnostics["integration_nodes"] = int(len(quadrature_z))
        diagnostics["curve_coefficients"] = curve.tolist()
        diagnostics["curve_frobenius_norm"] = float(np.linalg.norm(curve))
    elif model_id in {"M2", "M3"}:
        components = int(model_id[1:])
        if counts.shape[0] < components:
            raise ScientificModelError(
                f"{model_id} needs at least one source neuron per component"
            )
        mixture = GaussianMixture(
            n_components=components,
            covariance_type="tied",
            reg_covar=1e-4,
            init_params="kmeans",
            n_init=5,
            max_iter=150,
            random_state=int(seed),
        )
        mixture.fit(residuals)
        if not mixture.converged_:
            raise ScientificModelError(f"{model_id} mixture did not converge")
        covariance = _covariance_for_branch(
            residuals - mixture.means_[mixture.predict(residuals)],
            fitted_covariance_rank,
            covariance_mode,
        )
        per_component = max(2, int(math.ceil(integration_draws / components)))
        pieces: list[Array] = []
        piece_weights: list[Array] = []
        root = _covariance_root(covariance)
        for component in range(components):
            normals = _normal_qmc(
                per_component,
                dimension,
                seed_from(resolved_integration_seed, model_id, component),
            )
            pieces.append(mixture.means_[component][None, :] + normals @ root.T)
            piece_weights.append(
                np.full(per_component, mixture.weights_[component] / per_component)
            )
        offsets = np.vstack(pieces)
        weights = np.concatenate(piece_weights)
        weights /= weights.sum()
        prior_means = np.asarray(mixture.means_, dtype=float)
        prior_covariances = np.repeat(
            covariance[None, :, :], components, axis=0
        )
        prior_weights = np.asarray(mixture.weights_, dtype=float)
        parameter_count = (
            coefficients.size
            + (components - 1)
            + components * dimension
            + dimension * min(fitted_covariance_rank, dimension)
        )
        diagnostics.update(
            {
                "within_group_latent_support_family": (
                    "rank_bounded_Gaussian_manifold"
                ),
                "within_group_latent_support_rank": int(
                    np.linalg.matrix_rank(covariance, tol=1e-10)
                ),
                "group_offsets": True,
                "component_weights": mixture.weights_.tolist(),
                "component_means": mixture.means_.tolist(),
                "converged": bool(mixture.converged_),
                "iterations": int(mixture.n_iter_),
            }
        )
    else:
        raise ScientificModelError(f"unknown scientific model: {model_id}")

    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ScientificModelError("predictive integration weights are invalid")
    _require_finite("predictive latent offsets", offsets)
    return FittedPredictiveModel(
        model_id=model_id,
        basis=basis,
        nuisance_coefficients=coefficients,
        latent_offsets=offsets,
        log_weights=np.log(weights),
        prior_means=prior_means,
        prior_covariances=prior_covariances,
        prior_weights=prior_weights,
        integration_draws=integration_draws,
        integration_seed=resolved_integration_seed,
        parameter_count=int(parameter_count),
        diagnostics=diagnostics,
        pseudocount=float(pseudocount),
    )


@dataclass(frozen=True)
class CandidateSelection:
    model_id: str
    mean_score: float
    standard_error: float
    fold_scores: tuple[float, ...]
    parameter_count: int


def cross_validated_candidate(
    counts: Array,
    covariates: Array,
    candidates: Sequence[str],
    *,
    seed: int,
    folds: int = 3,
    integration_draws: int = 64,
    latent_rank: int = 2,
    ridge: float = 0.1,
    pseudocount: float = 0.5,
    representation: str = "count_aware_log_ratio",
    covariance_mode: str = "shrinkage",
    covariance_rank: int | None = None,
    tie_tolerance: float = 1e-6,
    canonical_counts: Any = None,
    canonical_to_pool: Array | None = None,
) -> CandidateSelection:
    counts, covariates = validate_profile_arrays(counts, covariates)
    if (canonical_counts is None) != (canonical_to_pool is None):
        raise ScientificModelError("canonical counts and pool mapping must be supplied together")
    canonical = None if canonical_counts is None else count_matrix(canonical_counts)
    if canonical is not None and canonical.shape[0] != counts.shape[0]:
        raise ScientificModelError("canonical counts must retain every source neuron")
    folds = max(2, min(int(folds), counts.shape[0]))
    permutation = np.random.default_rng(seed).permutation(counts.shape[0])
    fold_ids = np.arange(counts.shape[0]) % folds
    fold_ids = fold_ids[np.argsort(permutation)]
    # Each family keeps its frozen fold seed. No validation/transfer-target
    # counts contribute to the decoder used to score that fold.
    fold_corrections = {}
    if canonical is not None:
        for fold in range(folds):
            validation = fold_ids == fold
            training = ~validation
            decoder = CanonicalDecoder.fit(
                canonical[training], canonical_to_pool, counts.shape[1]
            )
            if not np.array_equal(decoder.pooled_counts(canonical[training]), counts[training]):
                raise ScientificModelError("canonical source counts disagree with internal pools")
            fold_corrections[fold] = decoder.correction(canonical[validation], counts[validation])
    summaries: list[CandidateSelection] = []
    for candidate in candidates:
        scores: list[float] = []
        parameter_count = 0
        failed = False
        for fold in range(folds):
            validation = fold_ids == fold
            training = ~validation
            if training.sum() < 2 or validation.sum() < 1:
                failed = True
                break
            try:
                fitted = fit_predictive_model(
                    counts[training],
                    covariates[training],
                    candidate,
                    seed=seed_from(seed, candidate, fold),
                    integration_seed=seed_from(seed, "cv-integration", fold),
                    integration_draws=integration_draws,
                    latent_rank=latent_rank,
                    ridge=ridge,
                    pseudocount=pseudocount,
                    representation=representation,
                    covariance_mode=covariance_mode,
                    covariance_rank=covariance_rank,
                )
                neuron_scores = fitted.score_neurons(counts[validation], covariates[validation])
                if canonical is not None:
                    neuron_scores = neuron_scores + fold_corrections[fold]
                scores.append(float(np.mean(neuron_scores)))
                parameter_count = max(parameter_count, fitted.parameter_count)
            except ScientificModelError:
                failed = True
                break
        if failed or not scores:
            continue
        values = np.asarray(scores, dtype=float)
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        summaries.append(
            CandidateSelection(
                model_id=candidate,
                mean_score=float(np.mean(values)),
                standard_error=standard_error,
                fold_scores=tuple(float(value) for value in values),
                parameter_count=parameter_count,
            )
        )
    if not summaries:
        raise ScientificModelError("every source-side candidate failed cross-validation")
    summaries.sort(key=lambda item: (-item.mean_score, item.parameter_count, item.model_id))
    best = summaries[0]
    selection_tolerance = max(tie_tolerance, best.standard_error)
    near = [
        item
        for item in summaries
        if best.mean_score - item.mean_score <= selection_tolerance
    ]
    return min(near, key=lambda item: (item.parameter_count, item.model_id))


@dataclass(frozen=True)
class FittedTriplet:
    T: FittedPredictiveModel
    U: FittedPredictiveModel
    M: FittedPredictiveModel
    T_cv: CandidateSelection
    U_cv: CandidateSelection
    M_cv: CandidateSelection
    reference_model_id: str
    null_generator_model_id: str
    canonical_decoder: CanonicalDecoder | None = None

    def score_target(
        self, counts: Array, covariates: Array, *, canonical_counts: Any = None
    ) -> dict[str, Any]:
        t_scores = self.T.score_neurons(counts, covariates)
        u_scores = self.U.score_neurons(counts, covariates)
        m_scores = self.M.score_neurons(counts, covariates)
        if self.canonical_decoder is not None:
            if canonical_counts is None:
                raise ScientificModelError("canonical target counts are required by the fitted decoder")
            correction = self.canonical_decoder.correction(canonical_counts, counts)
            t_scores = t_scores + correction
            u_scores = u_scores + correction
            m_scores = m_scores + correction
        elif canonical_counts is not None:
            raise ScientificModelError("canonical target supplied without a source-fitted decoder")
        reference_scores = u_scores if self.reference_model_id.startswith("U") else t_scores
        return {
            "T": t_scores,
            "U": u_scores,
            "M": m_scores,
            "reference": reference_scores,
            "delta_M_minus_reference": m_scores - reference_scores,
            "mean_scores": {
                "T": float(np.mean(t_scores)),
                "U": float(np.mean(u_scores)),
                "M": float(np.mean(m_scores)),
            },
            "mean_delta_M_minus_reference": float(np.mean(m_scores - reference_scores)),
            "reference_model_id": self.reference_model_id,
            "null_generator_model_id": self.null_generator_model_id,
        }


def fit_triplet(
    counts: Array,
    covariates: Array,
    *,
    seed: int,
    integration_draws: int = 64,
    latent_rank: int = 2,
    ridge: float = 0.1,
    folds: int = 3,
    tie_tolerance: float = 1e-6,
    pseudocount: float = 0.5,
    representation: str = "count_aware_log_ratio",
    covariance_mode: str = "shrinkage",
    covariance_rank: int | None = None,
    component_candidates: Sequence[int] = (2, 3),
    canonical_counts: Any = None,
    canonical_to_pool: Array | None = None,
) -> FittedTriplet:
    counts, covariates = validate_profile_arrays(counts, covariates)
    common = dict(
        folds=folds,
        integration_draws=integration_draws,
        latent_rank=latent_rank,
        ridge=ridge,
        pseudocount=pseudocount,
        representation=representation,
        covariance_mode=covariance_mode,
        covariance_rank=covariance_rank,
        tie_tolerance=tie_tolerance,
        canonical_counts=canonical_counts,
        canonical_to_pool=canonical_to_pool,
    )
    t_cv = cross_validated_candidate(
        counts, covariates, ["T"], seed=seed_from(seed, "T-cv"), **common
    )
    u_cv = cross_validated_candidate(
        counts,
        covariates,
        ["U_linear", "U_curved"],
        seed=seed_from(seed, "U-cv"),
        **common,
    )
    m_cv = cross_validated_candidate(
        counts,
        covariates,
        [f"M{int(component)}" for component in component_candidates],
        seed=seed_from(seed, "M-cv"),
        **common,
    )
    final_integration_seed = seed_from(seed, "final-integration")
    t_model = fit_predictive_model(
        counts,
        covariates,
        "T",
        seed=seed_from(seed, "T-final"),
        integration_seed=final_integration_seed,
        integration_draws=integration_draws,
        latent_rank=latent_rank,
        ridge=ridge,
        pseudocount=pseudocount,
        representation=representation,
        covariance_mode=covariance_mode,
        covariance_rank=covariance_rank,
    )
    u_model = fit_predictive_model(
        counts,
        covariates,
        u_cv.model_id,
        seed=seed_from(seed, "U-final", u_cv.model_id),
        integration_seed=final_integration_seed,
        integration_draws=integration_draws,
        latent_rank=latent_rank,
        ridge=ridge,
        pseudocount=pseudocount,
        representation=representation,
        covariance_mode=covariance_mode,
        covariance_rank=covariance_rank,
    )
    m_model = fit_predictive_model(
        counts,
        covariates,
        m_cv.model_id,
        seed=seed_from(seed, "M-final", m_cv.model_id),
        integration_seed=final_integration_seed,
        integration_draws=integration_draws,
        latent_rank=latent_rank,
        ridge=ridge,
        pseudocount=pseudocount,
        representation=representation,
        covariance_mode=covariance_mode,
        covariance_rank=covariance_rank,
    )
    reference = u_model.model_id if u_cv.mean_score > t_cv.mean_score + tie_tolerance else "T"
    comparability = max(t_cv.standard_error, u_cv.standard_error, tie_tolerance)
    null_generator = u_model.model_id if u_cv.mean_score >= t_cv.mean_score - comparability else "T"
    decoder = (None if canonical_counts is None else CanonicalDecoder.fit(
        canonical_counts, canonical_to_pool, counts.shape[1]
    ))
    if decoder is not None:
        t_model = replace(t_model, canonical_decoder=decoder)
        u_model = replace(u_model, canonical_decoder=decoder)
        m_model = replace(m_model, canonical_decoder=decoder)
    return FittedTriplet(
        T=t_model,
        U=u_model,
        M=m_model,
        T_cv=t_cv,
        U_cv=u_cv,
        M_cv=m_cv,
        reference_model_id=reference,
        null_generator_model_id=null_generator,
        canonical_decoder=decoder,
    )


def reciprocal_transfer(
    left_counts: Array,
    left_covariates: Array,
    right_counts: Array,
    right_covariates: Array,
    *,
    seed: int,
    left_canonical_counts: Any = None,
    right_canonical_counts: Any = None,
    canonical_to_pool: Array | None = None,
    **fit_options: Any,
) -> dict[str, Any]:
    if (left_canonical_counts is None) != (right_canonical_counts is None):
        raise ScientificModelError("both reciprocal canonical targets must be supplied")
    left_fit = fit_triplet(
        left_counts, left_covariates, seed=seed_from(seed, "left-source"),
        canonical_counts=left_canonical_counts, canonical_to_pool=canonical_to_pool,
        **fit_options
    )
    right_fit = fit_triplet(
        right_counts, right_covariates, seed=seed_from(seed, "right-source"),
        canonical_counts=right_canonical_counts, canonical_to_pool=canonical_to_pool,
        **fit_options
    )
    return {
        "left_to_right": left_fit.score_target(
            right_counts, right_covariates, canonical_counts=right_canonical_counts
        ),
        "right_to_left": right_fit.score_target(
            left_counts, left_covariates, canonical_counts=left_canonical_counts
        ),
        "source_fit_metadata": {
            "left": {
                "U": left_fit.U.model_id,
                "M": left_fit.M.model_id,
                "reference": left_fit.reference_model_id,
                "null_generator": left_fit.null_generator_model_id,
            },
            "right": {
                "U": right_fit.U.model_id,
                "M": right_fit.M.model_id,
                "reference": right_fit.reference_model_id,
                "null_generator": right_fit.null_generator_model_id,
            },
        },
    }
