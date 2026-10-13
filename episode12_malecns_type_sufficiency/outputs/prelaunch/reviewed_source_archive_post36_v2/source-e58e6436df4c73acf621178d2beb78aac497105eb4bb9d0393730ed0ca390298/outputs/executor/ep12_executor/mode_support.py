"""EP12 residual-mode diagnostic primitives.

The functions in this module calculate point diagnostics only.  They never
promote point estimates to scientific support: callers must obtain the frozen
whole-type simultaneous lower bounds before any mode-support gate can pass.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from .scientific_models import FittedPredictiveModel


Array = np.ndarray


class ModeSupportError(ValueError):
    """The fitted-mode diagnostic input is malformed or numerically invalid."""


def _finite_array(value: Array, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ModeSupportError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _gaussian_log_density(points: Array, mean: Array, covariance: Array) -> Array:
    points = _finite_array(points, ndim=2, name="points")
    mean = _finite_array(mean, ndim=1, name="mean")
    covariance = _finite_array(covariance, ndim=2, name="covariance")
    if points.shape[1] != len(mean) or covariance.shape != (len(mean), len(mean)):
        raise ModeSupportError("Gaussian dimensions disagree")
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ModeSupportError("component covariance must be positive definite")
    centered = points - mean
    try:
        solved = np.linalg.solve(covariance, centered.T).T
    except np.linalg.LinAlgError as exc:
        raise ModeSupportError("component covariance is singular") from exc
    quadratic = np.sum(centered * solved, axis=1)
    return -0.5 * (len(mean) * math.log(2.0 * math.pi) + logdet + quadratic)


def mixture_responsibilities(
    residuals: Array,
    component_means: Array,
    component_covariances: Array,
    component_weights: Array,
) -> Array:
    """Return normalized source-side component responsibilities."""

    residuals = _finite_array(residuals, ndim=2, name="residuals")
    means = _finite_array(component_means, ndim=2, name="component_means")
    covariances = _finite_array(
        component_covariances, ndim=3, name="component_covariances"
    )
    weights = _finite_array(component_weights, ndim=1, name="component_weights")
    components, dimension = means.shape
    if residuals.shape[1] != dimension:
        raise ModeSupportError("residual and component dimensions disagree")
    if covariances.shape != (components, dimension, dimension):
        raise ModeSupportError("component covariance dimensions disagree")
    if len(weights) != components or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ModeSupportError("component weights must be positive and sum to one")
    log_joint = np.column_stack(
        [
            math.log(float(weights[index]))
            + _gaussian_log_density(residuals, means[index], covariances[index])
            for index in range(components)
        ]
    )
    result = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    if not np.all(np.isfinite(result)) or not np.allclose(result.sum(axis=1), 1.0):
        raise ModeSupportError("component responsibilities are invalid")
    return result


def effective_component_memberships(responsibilities: Array) -> Array:
    """Kish effective membership for each soft component."""

    values = _finite_array(responsibilities, ndim=2, name="responsibilities")
    if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0):
        raise ModeSupportError("responsibility rows must be probabilities")
    numerator = values.sum(axis=0) ** 2
    denominator = np.sum(values**2, axis=0)
    if np.any(denominator <= 0):
        raise ModeSupportError("every fitted component needs positive membership")
    return numerator / denominator


def pairwise_symmetric_gaussian_kl(
    component_means: Array,
    component_covariances: Array,
) -> Array:
    """Return the full pairwise symmetric-KL matrix for Gaussian components."""

    means = _finite_array(component_means, ndim=2, name="component_means")
    covariances = _finite_array(
        component_covariances, ndim=3, name="component_covariances"
    )
    components, dimension = means.shape
    if covariances.shape != (components, dimension, dimension):
        raise ModeSupportError("component covariance dimensions disagree")
    inverses: list[Array] = []
    logdets: list[float] = []
    for covariance in covariances:
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ModeSupportError("component covariance must be positive definite")
        try:
            inverses.append(np.linalg.inv(covariance))
        except np.linalg.LinAlgError as exc:
            raise ModeSupportError("component covariance is singular") from exc
        logdets.append(float(logdet))
    result = np.zeros((components, components), dtype=float)
    for left in range(components):
        for right in range(left + 1, components):
            difference = means[right] - means[left]
            left_to_right = 0.5 * (
                np.trace(inverses[right] @ covariances[left])
                + difference @ inverses[right] @ difference
                - dimension
                + logdets[right]
                - logdets[left]
            )
            right_to_left = 0.5 * (
                np.trace(inverses[left] @ covariances[right])
                + difference @ inverses[left] @ difference
                - dimension
                + logdets[left]
                - logdets[right]
            )
            symmetric = 0.5 * float(left_to_right + right_to_left)
            if not math.isfinite(symmetric) or symmetric < -1e-10:
                raise ModeSupportError("symmetric component KL is invalid")
            result[left, right] = result[right, left] = max(0.0, symmetric)
    return result


def component_partner_contrasts(model: FittedPredictiveModel) -> Array:
    """Decode each fitted M component into a centered partner-probability contrast."""

    if not model.model_id.startswith("M") or len(model.prior_means) < 2:
        raise ModeSupportError("partner contrasts require a fitted mixture model")
    logits = np.asarray(model.prior_means, dtype=float) @ np.asarray(model.basis, dtype=float).T
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    weights = np.asarray(model.prior_weights, dtype=float)
    center = np.sum(weights[:, None] * probabilities, axis=0)
    contrasts = probabilities - center[None, :]
    if not np.all(np.isfinite(contrasts)) or not np.allclose(contrasts.sum(axis=1), 0.0):
        raise ModeSupportError("decoded partner contrasts are invalid")
    return contrasts


def match_component_contrasts(left: Array, right: Array) -> dict[str, Any]:
    """Label-invariant Hungarian matching by partner-contrast cosine."""

    left_values = _finite_array(left, ndim=2, name="left_contrasts")
    right_values = _finite_array(right, ndim=2, name="right_contrasts")
    if left_values.shape[1] != right_values.shape[1]:
        raise ModeSupportError("left and right partner vocabularies differ")
    left_norm = np.linalg.norm(left_values, axis=1)
    right_norm = np.linalg.norm(right_values, axis=1)
    if np.any(left_norm <= 0) or np.any(right_norm <= 0):
        raise ModeSupportError("every component needs a nonzero partner contrast")
    cosine = (left_values @ right_values.T) / (left_norm[:, None] * right_norm[None, :])
    row, column = linear_sum_assignment(-cosine)
    matched = cosine[row, column]
    return {
        "left_component_indices": row.astype(int).tolist(),
        "right_component_indices": column.astype(int).tolist(),
        "matched_cosines": matched.astype(float).tolist(),
        "minimum_matched_cosine": float(np.min(matched)),
        "mean_matched_cosine": float(np.mean(matched)),
        "cosine_matrix": cosine.tolist(),
        "point_estimate_only": True,
        "scientific_gate_passed": False,
        "gate_requires": "whole_type_simultaneous_lower_bound_above_zero",
    }


def point_mode_diagnostics(
    model: FittedPredictiveModel,
    residuals: Array,
    *,
    fitted_component_parameter_rank: int,
) -> dict[str, Any]:
    """Collect point diagnostics while explicitly leaving all gates closed."""

    if fitted_component_parameter_rank < 1:
        raise ModeSupportError("fitted component parameter rank must be positive")
    responsibilities = mixture_responsibilities(
        residuals,
        model.prior_means,
        model.prior_covariances,
        model.prior_weights,
    )
    effective = effective_component_memberships(responsibilities)
    pairwise = pairwise_symmetric_gaussian_kl(
        model.prior_means, model.prior_covariances
    )
    off_diagonal = pairwise[np.triu_indices(len(pairwise), k=1)]
    return {
        "model_id": model.model_id,
        "effective_component_memberships": effective.tolist(),
        "fitted_component_parameter_rank": int(fitted_component_parameter_rank),
        "all_effective_memberships_exceed_rank_pointwise": bool(
            np.all(effective > fitted_component_parameter_rank)
        ),
        "pairwise_symmetric_kl": pairwise.tolist(),
        "minimum_pairwise_symmetric_kl": float(np.min(off_diagonal)),
        "component_partner_contrasts": component_partner_contrasts(model).tolist(),
        "point_estimate_only": True,
        "separation_gate_passed": False,
        "prevalence_gate_passed": False,
        "partner_identifiability_gate_passed": False,
        "required_uncertainty": "whole_type_paired_simultaneous_intervals",
    }
