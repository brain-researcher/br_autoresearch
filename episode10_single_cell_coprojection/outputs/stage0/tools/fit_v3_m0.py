#!/usr/bin/env python3
"""Fit the frozen EP10 v3 development-only M0 observation-pattern model.

The production path accepts only the EP10-local target replay and the sealed
target-blind model interface.  ``--self-test`` uses deterministic synthetic
data only and never opens either real input.  Final and unassigned data are
not addressable by this program.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PANEL = (
    "allen_993_MOs__ipsilateral",
    "allen_993_MOs__contralateral",
    "allen_453_SS__ipsilateral",
    "allen_453_SS__contralateral",
    "allen_31_ACA__ipsilateral",
    "allen_31_ACA__contralateral",
    "allen_477_STR__ipsilateral",
    "allen_477_STR__contralateral",
    "allen_549_TH__ipsilateral",
    "allen_549_TH__contralateral",
    "allen_313_MB__ipsilateral",
    "allen_313_MB__contralateral",
    "allen_771_P__ipsilateral",
    "allen_771_P__contralateral",
    "allen_354_MY__ipsilateral",
    "allen_354_MY__contralateral",
)
GROUP_SIZES = {
    "212074": 6,
    "220187": 35,
    "220188": 25,
    "220190": 16,
    "221091": 33,
    "221225": 39,
    "221418": 12,
    "221514": 5,
    "221539": 1,
    "221588": 7,
    "221751": 13,
    "233281": 2,
}
LAYERS = ("MOp1", "MOp2/3", "MOp5", "MOp6a")
COVARIATES = (
    "intercept",
    "K_legendre_P1",
    "K_legendre_P2",
    "soma_x",
    "soma_y",
    "soma_z",
    "soma_x2",
    "soma_y2",
    "soma_z2",
    "soma_xy",
    "soma_xz",
    "soma_yz",
    "layer_H1",
    "layer_H2",
    "layer_H3",
    "hemisphere_right",
    "segment_unmapped",
)
LAMBDAS = tuple(10.0**power for power in range(-8, 4))
TARGET_ROOT_BASENAME = "gao_mop_development_targets_v3_ep10_assets"
REJECTED_TARGET_ROOT_BASENAME = "gao_mop_development_targets_v3"
INTERFACE_ROOT_BASENAME = "v3_model_interface_preflight_v1"
EPISODE_SCRATCH_RELATIVE = Path(
    "br_autoresearch/episode10_single_cell_coprojection"
)
INTERFACE_FIELDS = (
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "provider_hemisphere",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
    "segment_unmapped",
    "terminal_unmapped_report_only",
    "mapping_overlap_class_report_only",
)
LN2 = math.log(2.0)


class ModelError(RuntimeError):
    """A frozen model or input invariant was violated."""


class BoundaryReview(ModelError):
    """The prospectively frozen ridge grid requires contract review."""


@dataclass(frozen=True)
class Record:
    provider_neuron_id: str
    filename: str
    group: str
    sample_id: str
    source_acronym: str
    hemisphere: str
    soma_um: tuple[float, float, float]
    segment_unmapped: int
    terminal_unmapped_report_only: int
    outcome: tuple[int, ...]
    k: int


@dataclass(frozen=True)
class DesignTransform:
    soma_mean: np.ndarray
    soma_scale: np.ndarray


@dataclass(frozen=True)
class OptimizerResult:
    coefficients: np.ndarray
    objective: float
    gradient_infinity_norm: float
    iterations: int
    accepted_steps: int
    relative_objective_change: float


@dataclass(frozen=True)
class PredictionBlock:
    eta: np.ndarray
    log_probability_nats: np.ndarray
    marginals: np.ndarray


def helmert(rows: int) -> np.ndarray:
    """Return the frozen rows-by-(rows-1) orthonormal Helmert matrix."""
    if rows < 2:
        raise ValueError("Helmert coding requires at least two rows")
    result = np.zeros((rows, rows - 1), dtype=float)
    for column in range(rows - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        result[: column + 1, column] = 1.0 / denominator
        result[column + 1, column] = -(column + 1) / denominator
    return result


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def fixed_k_logz_marginals(eta: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    """Exact log partition and inclusion marginals from prefix/suffix ESFs."""
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 1 or not np.all(np.isfinite(eta)):
        raise ValueError("eta must be a finite one-dimensional array")
    p = eta.size
    if not 0 <= k <= p:
        raise ValueError(f"K={k} is outside 0..{p}")
    if k == 0:
        return 0.0, np.zeros(p, dtype=float)
    if k == p:
        return float(np.sum(eta)), np.ones(p, dtype=float)

    prefix = np.full((p + 1, k + 1), -np.inf, dtype=float)
    suffix = np.full((p + 1, k + 1), -np.inf, dtype=float)
    prefix[0, 0] = 0.0
    for target in range(p):
        prefix[target + 1, 0] = 0.0
        for degree in range(1, min(k, target + 1) + 1):
            prefix[target + 1, degree] = np.logaddexp(
                prefix[target, degree],
                prefix[target, degree - 1] + eta[target],
            )
    suffix[p, 0] = 0.0
    for target in range(p - 1, -1, -1):
        suffix[target, 0] = 0.0
        remaining = p - target
        for degree in range(1, min(k, remaining) + 1):
            suffix[target, degree] = np.logaddexp(
                suffix[target + 1, degree],
                suffix[target + 1, degree - 1] + eta[target],
            )

    logz = float(prefix[p, k])
    marginals = np.zeros(p, dtype=float)
    for target in range(p):
        terms = np.asarray(
            [
                prefix[target, left]
                + suffix[target + 1, k - 1 - left]
                for left in range(k)
                if left <= target and k - 1 - left <= p - target - 1
            ],
            dtype=float,
        )
        marginals[target] = math.exp(eta[target] + _logsumexp(terms) - logz)
    marginals = np.clip(marginals, 0.0, 1.0)
    return logz, marginals


def batch_fixed_k_logz_marginals(
    eta: np.ndarray, k_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact prefix/suffix DP, batched separately for each K."""
    eta = np.asarray(eta, dtype=float)
    k_values = np.asarray(k_values, dtype=int)
    if eta.ndim != 2 or k_values.shape != (eta.shape[0],):
        raise ValueError("eta and K arrays are not aligned")
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta contains a nonfinite value")
    n, p = eta.shape
    if np.any(k_values < 0) or np.any(k_values > p):
        raise ValueError("a K value is outside the panel")
    logz = np.empty(n, dtype=float)
    marginals = np.zeros((n, p), dtype=float)

    for k in np.unique(k_values):
        indexes = np.flatnonzero(k_values == k)
        block = eta[indexes]
        count = indexes.size
        if k == 0:
            logz[indexes] = 0.0
            continue
        if k == p:
            logz[indexes] = np.sum(block, axis=1)
            marginals[indexes] = 1.0
            continue
        prefix = np.full((count, p + 1, k + 1), -np.inf, dtype=float)
        suffix = np.full((count, p + 1, k + 1), -np.inf, dtype=float)
        prefix[:, 0, 0] = 0.0
        for target in range(p):
            prefix[:, target + 1, 0] = 0.0
            for degree in range(1, min(k, target + 1) + 1):
                prefix[:, target + 1, degree] = np.logaddexp(
                    prefix[:, target, degree],
                    prefix[:, target, degree - 1] + block[:, target],
                )
        suffix[:, p, 0] = 0.0
        for target in range(p - 1, -1, -1):
            suffix[:, target, 0] = 0.0
            remaining = p - target
            for degree in range(1, min(k, remaining) + 1):
                suffix[:, target, degree] = np.logaddexp(
                    suffix[:, target + 1, degree],
                    suffix[:, target + 1, degree - 1] + block[:, target],
                )
        block_logz = prefix[:, p, k]
        logz[indexes] = block_logz
        for target in range(p):
            pieces = [
                prefix[:, target, left]
                + suffix[:, target + 1, k - 1 - left]
                for left in range(k)
                if left <= target and k - 1 - left <= p - target - 1
            ]
            log_without = np.logaddexp.reduce(np.stack(pieces, axis=1), axis=1)
            marginals[indexes, target] = np.exp(
                block[:, target] + log_without - block_logz
            )
    np.clip(marginals, 0.0, 1.0, out=marginals)
    return logz, marginals


def equal_group_weights(groups: Sequence[str]) -> np.ndarray:
    if not groups:
        raise ValueError("group weights require at least one row")
    counts = Counter(str(group) for group in groups)
    number_groups = len(counts)
    return np.asarray(
        [1.0 / (number_groups * counts[str(group)]) for group in groups],
        dtype=float,
    )


def equal_group_mean(values: Sequence[float], groups: Sequence[str]) -> float:
    values_array = np.asarray(values, dtype=float)
    if values_array.shape != (len(groups),) or values_array.size == 0:
        raise ValueError("values and groups must be nonempty and aligned")
    return float(np.dot(equal_group_weights(groups), values_array))


def objective_and_gradient(
    coefficient_vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
) -> tuple[float, np.ndarray]:
    """Return frozen equal-group ridge NLL and its analytic gradient."""
    n, d = x.shape
    p = y.shape[1]
    if y.shape != (n, p) or k_values.shape != (n,):
        raise ValueError("training arrays are not aligned")
    if target_contrasts.shape != (p, p - 1):
        raise ValueError("target contrast matrix has the wrong shape")
    if coefficient_vector.shape != (d * (p - 1),):
        raise ValueError("coefficient vector has the wrong shape")
    if ridge_lambda < 0.0 or not math.isfinite(ridge_lambda):
        raise ValueError("ridge lambda must be finite and nonnegative")
    coefficients = coefficient_vector.reshape(d, p - 1)
    eta = (x @ coefficients) @ target_contrasts.T
    logz, marginals = batch_fixed_k_logz_marginals(eta, k_values)
    observed_score = np.sum(eta * y, axis=1)
    weights = equal_group_weights(groups)
    nll = logz - observed_score
    objective = float(np.dot(weights, nll))
    objective += 0.5 * ridge_lambda * float(np.sum(coefficients * coefficients))
    weighted_residual = (marginals - y) * weights[:, None]
    gradient = x.T @ weighted_residual @ target_contrasts
    gradient += ridge_lambda * coefficients
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise ModelError("nonfinite objective or gradient")
    return objective, gradient.ravel()


def _lbfgs_direction(
    gradient: np.ndarray,
    s_history: Sequence[np.ndarray],
    y_history: Sequence[np.ndarray],
) -> np.ndarray:
    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s_value, y_value in zip(reversed(s_history), reversed(y_history)):
        rho = 1.0 / float(np.dot(y_value, s_value))
        alpha = rho * float(np.dot(s_value, q))
        q -= alpha * y_value
        alphas.append(alpha)
        rhos.append(rho)
    if s_history:
        last_s = s_history[-1]
        last_y = y_history[-1]
        scale = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
        r = scale * q
    else:
        r = q
    for s_value, y_value, alpha, rho in zip(
        s_history, y_history, reversed(alphas), reversed(rhos)
    ):
        beta = rho * float(np.dot(y_value, r))
        r += s_value * (alpha - beta)
    return -r


def fit_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
    initial: np.ndarray | None = None,
    *,
    history_size: int = 10,
    maximum_iterations: int = 500,
    gradient_tolerance: float = 1.0e-7,
    relative_objective_tolerance: float = 1.0e-10,
    armijo_c1: float = 1.0e-4,
    backtrack_factor: float = 0.5,
) -> OptimizerResult:
    """Deterministic L-BFGS with Armijo search and no fallback optimizer."""
    dimension = x.shape[1] * (target_contrasts.shape[1])
    if initial is None:
        vector = np.zeros(dimension, dtype=float)
    else:
        initial_array = np.asarray(initial, dtype=float)
        if initial_array.shape == (x.shape[1], target_contrasts.shape[1]):
            vector = initial_array.ravel().copy()
        elif initial_array.shape == (dimension,):
            vector = initial_array.copy()
        else:
            raise ValueError("optimizer initial value has the wrong shape")
    value, gradient = objective_and_gradient(
        vector, x, y, k_values, groups, target_contrasts, ridge_lambda
    )
    gradient_norm = float(np.max(np.abs(gradient)))
    if gradient_norm <= gradient_tolerance:
        return OptimizerResult(
            coefficients=vector.reshape(x.shape[1], target_contrasts.shape[1]),
            objective=value,
            gradient_infinity_norm=gradient_norm,
            iterations=0,
            accepted_steps=0,
            relative_objective_change=0.0,
        )

    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    accepted_steps = 0
    relative_change = math.inf
    for iteration in range(1, maximum_iterations + 1):
        direction = _lbfgs_direction(gradient, s_history, y_history)
        directional_derivative = float(np.dot(gradient, direction))
        if not directional_derivative < 0.0:
            raise ModelError("L-BFGS produced a non-descent direction")

        step = 1.0
        accepted: tuple[np.ndarray, float, np.ndarray] | None = None
        for _ in range(80):
            candidate = vector + step * direction
            candidate_value, candidate_gradient = objective_and_gradient(
                candidate,
                x,
                y,
                k_values,
                groups,
                target_contrasts,
                ridge_lambda,
            )
            if candidate_value <= value + armijo_c1 * step * directional_derivative:
                accepted = (candidate, candidate_value, candidate_gradient)
                break
            step *= backtrack_factor
        if accepted is None:
            raise ModelError("deterministic Armijo line search failed")

        new_vector, new_value, new_gradient = accepted
        s_value = new_vector - vector
        y_value = new_gradient - gradient
        curvature = float(np.dot(s_value, y_value))
        curvature_floor = (
            1.0e-14
            * max(1.0, float(np.linalg.norm(s_value)))
            * max(1.0, float(np.linalg.norm(y_value)))
        )
        if curvature > curvature_floor:
            s_history.append(s_value)
            y_history.append(y_value)
            if len(s_history) > history_size:
                del s_history[0]
                del y_history[0]

        relative_change = abs(new_value - value) / max(
            1.0, abs(value), abs(new_value)
        )
        vector, value, gradient = new_vector, new_value, new_gradient
        accepted_steps += 1
        gradient_norm = float(np.max(np.abs(gradient)))
        if (
            gradient_norm <= gradient_tolerance
            and relative_change <= relative_objective_tolerance
        ):
            return OptimizerResult(
                coefficients=vector.reshape(x.shape[1], target_contrasts.shape[1]),
                objective=value,
                gradient_infinity_norm=gradient_norm,
                iterations=iteration,
                accepted_steps=accepted_steps,
                relative_objective_change=relative_change,
            )
    raise ModelError(
        "deterministic L-BFGS did not meet both convergence tolerances in "
        f"{maximum_iterations} iterations (gradient_inf={gradient_norm:.6g}, "
        f"relative_change={relative_change:.6g})"
    )


def predict(
    coefficients: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    target_contrasts: np.ndarray,
) -> PredictionBlock:
    eta = (x @ coefficients) @ target_contrasts.T
    logz, marginals = batch_fixed_k_logz_marginals(eta, k_values)
    log_probability = np.sum(eta * y, axis=1) - logz
    return PredictionBlock(eta=eta, log_probability_nats=log_probability, marginals=marginals)


def _soma_monomials(records: Sequence[Record]) -> np.ndarray:
    coordinates = np.asarray([record.soma_um for record in records], dtype=float)
    if coordinates.shape != (len(records), 3) or not np.all(np.isfinite(coordinates)):
        raise ModelError("invalid soma-coordinate matrix")
    x, y, z = coordinates.T
    return np.column_stack((x, y, z, x * x, y * y, z * z, x * y, x * z, y * z))


def fit_design_transform(records: Sequence[Record]) -> DesignTransform:
    if not records:
        raise ModelError("cannot fit a design transform without training rows")
    monomials = _soma_monomials(records)
    mean = np.mean(monomials, axis=0)
    scale = np.sqrt(np.mean((monomials - mean) ** 2, axis=0))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ModelError("a training-only soma monomial has zero or nonfinite scale")
    return DesignTransform(soma_mean=mean, soma_scale=scale)


def build_design(records: Sequence[Record], transform: DesignTransform) -> np.ndarray:
    if not records:
        return np.empty((0, len(COVARIATES)), dtype=float)
    k_values = np.asarray([record.k for record in records], dtype=float)
    u = k_values / 8.0 - 1.0
    standardized_soma = (
        _soma_monomials(records) - transform.soma_mean[None, :]
    ) / transform.soma_scale[None, :]
    layer_contrasts = helmert(4)
    layer_rows: list[np.ndarray] = []
    hemispheres: list[float] = []
    segment: list[float] = []
    for record in records:
        try:
            layer_index = LAYERS.index(record.source_acronym)
        except ValueError as error:
            raise ModelError(
                f"unexpected frozen layer {record.source_acronym!r} for "
                f"cell {record.provider_neuron_id}"
            ) from error
        layer_rows.append(layer_contrasts[layer_index])
        if record.hemisphere == "left":
            hemispheres.append(-0.5)
        elif record.hemisphere == "right":
            hemispheres.append(0.5)
        else:
            raise ModelError(
                f"unexpected hemisphere {record.hemisphere!r} for "
                f"cell {record.provider_neuron_id}"
            )
        if record.segment_unmapped not in (0, 1):
            raise ModelError("segment_unmapped is not binary")
        segment.append(float(record.segment_unmapped))
    design = np.column_stack(
        (
            np.ones(len(records), dtype=float),
            u,
            0.5 * (3.0 * u * u - 1.0),
            standardized_soma,
            np.vstack(layer_rows),
            np.asarray(hemispheres),
            np.asarray(segment),
        )
    )
    if design.shape != (len(records), 17) or not np.all(np.isfinite(design)):
        raise ModelError("frozen 17-column design construction failed")
    return design


def outcome_matrix(records: Sequence[Record]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = np.asarray([record.outcome for record in records], dtype=float)
    k_values = np.asarray([record.k for record in records], dtype=int)
    groups = [record.group for record in records]
    if y.shape != (len(records), 16):
        raise ModelError("outcome matrix is not N by 16")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ModelError("outcome matrix is not binary")
    if not np.array_equal(np.sum(y, axis=1).astype(int), k_values):
        raise ModelError("observed K disagrees with the fixed panel calls")
    return y, k_values, groups


def logo_splits(groups: Sequence[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    group_array = np.asarray(groups, dtype=object)
    unique = sorted(set(str(value) for value in groups))
    return [
        (
            held,
            np.flatnonzero(group_array != held),
            np.flatnonzero(group_array == held),
        )
        for held in unique
    ]


def _take(records: Sequence[Record], indexes: np.ndarray) -> list[Record]:
    return [records[int(index)] for index in indexes]


def fit_lambda_path(
    records: Sequence[Record], lambdas: Sequence[float]
) -> tuple[DesignTransform, dict[float, OptimizerResult]]:
    transform = fit_design_transform(records)
    x = build_design(records, transform)
    y, k_values, groups = outcome_matrix(records)
    contrasts = helmert(16)
    fitted: dict[float, OptimizerResult] = {}
    initial: np.ndarray | None = None
    for ridge_lambda in sorted((float(value) for value in lambdas), reverse=True):
        try:
            result = fit_lbfgs(
                x,
                y,
                k_values,
                groups,
                contrasts,
                ridge_lambda,
                initial=initial,
            )
        except ModelError as error:
            raise ModelError(
                f"ridge path failed at lambda={ridge_lambda:.12g}: {error}"
            ) from error
        fitted[ridge_lambda] = result
        initial = result.coefficients
    return transform, fitted


def select_lambda_one_se(
    validation_nll_by_lambda: Mapping[float, Sequence[float]],
) -> tuple[float, dict[str, Any]]:
    summaries: dict[float, tuple[float, float]] = {}
    for ridge_lambda in LAMBDAS:
        values = np.asarray(validation_nll_by_lambda[ridge_lambda], dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ModelError("one-SE selection lacks finite group-level losses")
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
        summaries[ridge_lambda] = (mean, standard_error)
    minimum_lambda = min(LAMBDAS, key=lambda value: (summaries[value][0], -value))
    threshold = summaries[minimum_lambda][0] + summaries[minimum_lambda][1]
    qualifying = [
        value for value in LAMBDAS if summaries[value][0] <= threshold + 1.0e-15
    ]
    selected = max(qualifying)
    details = {
        "minimum_mean_lambda": minimum_lambda,
        "minimum_mean_nll_nats": summaries[minimum_lambda][0],
        "standard_error_at_minimum_nats": summaries[minimum_lambda][1],
        "one_se_threshold_nats": threshold,
        "selected_largest_qualifying_lambda": selected,
        "grid": [
            {
                "lambda": value,
                "mean_group_nll_nats": summaries[value][0],
                "group_standard_error_nats": summaries[value][1],
            }
            for value in LAMBDAS
        ],
    }
    return selected, details


def _fit_single(records: Sequence[Record], ridge_lambda: float) -> tuple[DesignTransform, OptimizerResult]:
    transform = fit_design_transform(records)
    x = build_design(records, transform)
    y, k_values, groups = outcome_matrix(records)
    result = fit_lbfgs(x, y, k_values, groups, helmert(16), ridge_lambda)
    return transform, result


def tune_lambda_logo(records: Sequence[Record]) -> tuple[float, dict[str, Any]]:
    """Tune ridge strength by group-level LOGO validation."""
    groups = [record.group for record in records]
    splits = logo_splits(groups)
    validation: dict[float, list[float]] = {value: [] for value in LAMBDAS}
    for held_group, train_indexes, validation_indexes in splits:
        training = _take(records, train_indexes)
        held = _take(records, validation_indexes)
        try:
            transform, path = fit_lambda_path(training, LAMBDAS)
        except ModelError as error:
            raise ModelError(
                f"inner LOGO held group {held_group} failed: {error}"
            ) from error
        held_x = build_design(held, transform)
        held_y, held_k, _ = outcome_matrix(held)
        for ridge_lambda in LAMBDAS:
            block = predict(
                path[ridge_lambda].coefficients,
                held_x,
                held_y,
                held_k,
                helmert(16),
            )
            validation[ridge_lambda].append(
                float(-np.mean(block.log_probability_nats))
            )
        if len({record.group for record in training}) != len(splits) - 1:
            raise ModelError(f"LOGO training groups changed for {held_group}")
        print(
            json.dumps(
                {
                    "event": "inner_LOGO_path_complete",
                    "held_group": held_group,
                    "training_cells": len(training),
                    "validation_cells": len(held),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return select_lambda_one_se(validation)


def nested_outer_logo(
    records: Sequence[Record],
) -> tuple[PredictionBlock, list[dict[str, Any]], int]:
    """Run 12 outer LOGO folds with 11-fold inner LOGO tuning."""
    n = len(records)
    eta = np.full((n, 16), np.nan, dtype=float)
    log_probability = np.full(n, np.nan, dtype=float)
    marginals = np.full((n, 16), np.nan, dtype=float)
    outer_details: list[dict[str, Any]] = []
    groups = [record.group for record in records]
    outer_splits = logo_splits(groups)
    if len(outer_splits) != 12:
        raise ModelError(f"outer LOGO requires 12 groups, found {len(outer_splits)}")

    for held_group, outer_train_indexes, outer_test_indexes in outer_splits:
        outer_training = _take(records, outer_train_indexes)
        outer_test = _take(records, outer_test_indexes)
        if len(set(record.group for record in outer_training)) != 11:
            raise ModelError("an outer training split does not contain 11 groups")
        try:
            selected_lambda, inner_details = tune_lambda_logo(outer_training)
            transform, fit = _fit_single(outer_training, selected_lambda)
        except ModelError as error:
            raise ModelError(
                f"outer LOGO held group {held_group} failed: {error}"
            ) from error
        test_x = build_design(outer_test, transform)
        test_y, test_k, _ = outcome_matrix(outer_test)
        block = predict(fit.coefficients, test_x, test_y, test_k, helmert(16))
        eta[outer_test_indexes] = block.eta
        log_probability[outer_test_indexes] = block.log_probability_nats
        marginals[outer_test_indexes] = block.marginals
        outer_details.append(
            {
                "held_out_group": held_group,
                "held_out_cells": len(outer_test),
                "training_groups": 11,
                "inner_folds": 11,
                "selected_lambda": selected_lambda,
                "selected_lambda_is_grid_boundary": selected_lambda
                in (LAMBDAS[0], LAMBDAS[-1]),
                "optimizer": {
                    "objective": fit.objective,
                    "gradient_infinity_norm": fit.gradient_infinity_norm,
                    "iterations": fit.iterations,
                    "accepted_steps": fit.accepted_steps,
                    "relative_objective_change": fit.relative_objective_change,
                },
                "inner_selection": inner_details,
            }
        )
        print(
            json.dumps(
                {
                    "event": "outer_LOGO_fold_complete",
                    "held_group": held_group,
                    "held_out_cells": len(outer_test),
                    "selected_lambda": selected_lambda,
                    "boundary": selected_lambda in (LAMBDAS[0], LAMBDAS[-1]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not (
        np.all(np.isfinite(eta))
        and np.all(np.isfinite(log_probability))
        and np.all(np.isfinite(marginals))
    ):
        raise ModelError("outer cross-fitting left an unpredicted cell")
    boundary_count = sum(
        bool(detail["selected_lambda_is_grid_boundary"])
        for detail in outer_details
    )
    return (
        PredictionBlock(
            eta=eta,
            log_probability_nats=log_probability,
            marginals=marginals,
        ),
        outer_details,
        boundary_count,
    )


def score_and_calibration(
    records: Sequence[Record], block: PredictionBlock
) -> dict[str, Any]:
    y, k_values, groups = outcome_matrix(records)
    log2_probability = block.log_probability_nats / LN2
    uniform_log2_probability = np.asarray(
        [-math.log2(math.comb(16, int(k))) for k in k_values], dtype=float
    )
    gain = log2_probability - uniform_log2_probability
    per_group: dict[str, Any] = {}
    for group in sorted(set(groups)):
        indexes = np.asarray([value == group for value in groups], dtype=bool)
        group_y = y[indexes]
        group_marginals = block.marginals[indexes]
        group_k = k_values[indexes]
        reference = group_k[:, None] / 16.0
        residual = np.mean(group_y - group_marginals, axis=0)
        per_group[group] = {
            "cells": int(np.sum(indexes)),
            "mean_log2_probability": float(np.mean(log2_probability[indexes])),
            "uniform_fixed_K_mean_log2_probability": float(
                np.mean(uniform_log2_probability[indexes])
            ),
            "paired_M0_minus_uniform_gain_bits_per_cell": float(
                np.mean(gain[indexes])
            ),
            "marginal_Brier_M0": float(
                np.mean((group_y - group_marginals) ** 2)
            ),
            "marginal_Brier_K_over_16_reference": float(
                np.mean((group_y - reference) ** 2)
            ),
            "observed_minus_expected_by_target": {
                coordinate: float(residual[index])
                for index, coordinate in enumerate(PANEL)
            },
        }
    return {
        "scoring_scale": "bits",
        "aggregate_equal_group_mean_log2_probability": equal_group_mean(
            log2_probability, groups
        ),
        "aggregate_equal_group_uniform_fixed_K_mean_log2_probability": equal_group_mean(
            uniform_log2_probability, groups
        ),
        "aggregate_equal_group_paired_M0_minus_uniform_gain_bits_per_cell": equal_group_mean(
            gain, groups
        ),
        "per_group": per_group,
        "degenerate_K_counts": {
            "K_0": int(np.sum(k_values == 0)),
            "K_16": int(np.sum(k_values == 16)),
        },
        "calibration": {
            "prediction_type": "outer_cross_fitted_exact_inclusion_marginals",
            "bins_used": False,
            "group_by_target_residual_definition": "mean(observed_indicator - exact_marginal)",
            "per_group_marginal_Brier_reported": True,
        },
    }


def _temperature_objective_gradient(
    slope: float,
    eta: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
) -> tuple[float, float]:
    scaled = slope * eta
    logz, marginals = batch_fixed_k_logz_marginals(scaled, k_values)
    observed_energy = np.sum(eta * y, axis=1)
    expected_energy = np.sum(eta * marginals, axis=1)
    weights = equal_group_weights(groups)
    objective = float(np.dot(weights, logz - slope * observed_energy))
    gradient = float(np.dot(weights, expected_energy - observed_energy))
    return objective, gradient


def fit_temperature_slope(
    eta: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
) -> dict[str, Any]:
    """Fit the optional bounded slope diagnostic without changing predictions."""
    lower, upper = 0.0, 4.0
    _, lower_gradient = _temperature_objective_gradient(
        lower, eta, y, k_values, groups
    )
    _, upper_gradient = _temperature_objective_gradient(
        upper, eta, y, k_values, groups
    )
    if lower_gradient >= 0.0:
        slope = lower
    elif upper_gradient <= 0.0:
        slope = upper
    else:
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            _, gradient = _temperature_objective_gradient(
                midpoint, eta, y, k_values, groups
            )
            if gradient <= 0.0:
                lower = midpoint
            else:
                upper = midpoint
        slope = 0.5 * (lower + upper)
    objective, gradient = _temperature_objective_gradient(
        slope, eta, y, k_values, groups
    )
    return {
        "slope": slope,
        "objective_nats": objective,
        "gradient": gradient,
        "at_bound": slope in (0.0, 4.0),
    }


def temperature_diagnostic(
    records: Sequence[Record], block: PredictionBlock
) -> dict[str, Any]:
    y, k_values, groups = outcome_matrix(records)
    point = fit_temperature_slope(block.eta, y, k_values, groups)
    jackknife: dict[str, float] = {}
    group_array = np.asarray(groups, dtype=object)
    for held_group in sorted(set(groups)):
        keep = group_array != held_group
        result = fit_temperature_slope(
            block.eta[keep],
            y[keep],
            k_values[keep],
            [group for group in groups if group != held_group],
        )
        jackknife[held_group] = float(result["slope"])
    values = np.asarray(list(jackknife.values()), dtype=float)
    jackknife_mean = float(np.mean(values))
    standard_error = math.sqrt(
        (len(values) - 1)
        / len(values)
        * float(np.sum((values - jackknife_mean) ** 2))
    )
    return {
        "status": "diagnostic_only_not_applied_to_scores_or_predictions",
        "bounds": [0.0, 4.0],
        "point": point,
        "leave_one_group_out": jackknife,
        "jackknife_standard_error": standard_error,
        "outcome_selected_bins_used": False,
    }


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise ModelError("M0 execution requires PyYAML") from error
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ModelError(f"YAML document is not a mapping: {path}")
    return payload


def validate_contract(path: Path) -> Mapping[str, Any]:
    contract = _load_yaml(path)
    if (
        contract.get("schema_version") != "episode10_v3_m0_model_v1"
        or contract.get("status") != "frozen_before_real_M0_execution"
        or contract.get("sealed_scope", {}).get("role") != "development_only"
        or contract.get("sealed_scope", {}).get("final_or_unassigned_access")
        != "forbidden"
        or int(contract.get("sealed_scope", {}).get("required_cells", -1)) != 194
    ):
        raise ModelError("M0 contract is not the frozen development-only contract")
    inputs = contract.get("required_inputs", {})
    if (
        inputs.get("exact_scratch_parent_below_SCRATCH")
        != EPISODE_SCRATCH_RELATIVE.as_posix()
        or
        inputs.get("target_root", {}).get("required_basename")
        != TARGET_ROOT_BASENAME
        or inputs.get("target_root", {}).get("historical_rejected_basename")
        != REJECTED_TARGET_ROOT_BASENAME
        or inputs.get("interface_root", {}).get("required_basename")
        != INTERFACE_ROOT_BASENAME
        or {str(key): int(value) for key, value in inputs.get("expected_group_sizes", {}).items()}
        != GROUP_SIZES
    ):
        raise ModelError("M0 input identity contract changed")
    if tuple(contract.get("fixed_panel", {}).get("ordered_coordinate_ids", ())) != PANEL:
        raise ModelError("M0 fixed target panel changed")
    if tuple(contract.get("design", {}).get("ordered_covariates", ())) != COVARIATES:
        raise ModelError("M0 design columns changed")
    observed_grid = tuple(
        float(value) for value in contract.get("estimation", {}).get("lambda_grid", ())
    )
    if observed_grid != LAMBDAS:
        raise ModelError("M0 ridge grid changed")
    optimizer = contract.get("estimation", {}).get("optimizer", {})
    if (
        optimizer.get("algorithm") != "deterministic_L_BFGS"
        or int(optimizer.get("history_size", -1)) != 10
        or int(optimizer.get("maximum_iterations", -1)) != 500
        or float(optimizer.get("gradient_infinity_tolerance", math.nan)) != 1.0e-7
        or float(optimizer.get("relative_objective_tolerance", math.nan)) != 1.0e-10
        or optimizer.get("convergence_requires_both_tolerances") is not True
        or optimizer.get("fallback_optimizer") != "forbidden"
    ):
        raise ModelError("M0 optimizer contract changed")
    return contract


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.is_symlink() or not path.is_file():
        raise ModelError(f"CSV input is absent or a symlink: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ModelError(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ModelError(f"nonnumeric {label}") from error
    if not math.isfinite(parsed):
        raise ModelError(f"nonfinite {label}")
    return parsed


def _require_scratch_child(path: Path, label: str) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise ModelError("SCRATCH is unset; real M0 execution must run in a Sherlock job")
    scratch = Path(scratch_value).resolve(strict=True)
    if not path.is_absolute():
        raise ModelError(f"{label} must be an absolute Scratch path")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(scratch):
        raise ModelError(f"{label} must be below SCRATCH")
    if path.is_symlink() or resolved.is_symlink():
        raise ModelError(f"{label} may not be a symlink")
    return resolved


def load_real_records(
    target_root_argument: Path, interface_root_argument: Path
) -> list[Record]:
    """Load only the explicitly named development target/interface surfaces."""
    if REJECTED_TARGET_ROOT_BASENAME in target_root_argument.parts:
        raise ModelError(
            "historical sibling-sourced gao_mop_development_targets_v3 is rejected"
        )
    target_root = _require_scratch_child(target_root_argument, "target root")
    interface_root = _require_scratch_child(interface_root_argument, "interface root")
    scratch = Path(os.environ["SCRATCH"]).resolve(strict=True)
    episode_root = (scratch / EPISODE_SCRATCH_RELATIVE).resolve(strict=True)
    if target_root.parent != episode_root or interface_root.parent != episode_root:
        raise ModelError(
            "target and interface roots must be direct children of the canonical "
            "EP10 Scratch parent"
        )
    if target_root.name != TARGET_ROOT_BASENAME:
        raise ModelError(
            f"target root basename must be exactly {TARGET_ROOT_BASENAME!r}"
        )
    if interface_root.name != INTERFACE_ROOT_BASENAME:
        raise ModelError(
            f"interface root basename must be exactly {INTERFACE_ROOT_BASENAME!r}"
        )
    if target_root == interface_root:
        raise ModelError("target and interface roots must be distinct")

    target_gate_path = target_root / "v3_gate_summary.json"
    interface_gate_path = interface_root / "v3_model_interface_summary.json"
    if (
        target_gate_path.is_symlink()
        or interface_gate_path.is_symlink()
        or not target_gate_path.is_file()
        or not interface_gate_path.is_file()
    ):
        raise ModelError("required upstream gate summary is absent or a symlink")
    try:
        target_gate = json.loads(target_gate_path.read_text(encoding="utf-8"))
        interface_gate = json.loads(interface_gate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ModelError("required upstream gate summary is unreadable") from error
    if not (
        isinstance(target_gate, dict)
        and target_gate.get("status") == "pass"
        and int(target_gate.get("eligible_cells", -1)) == 194
        and target_gate.get("final_or_unassigned_files_opened") is False
        and isinstance(interface_gate, dict)
        and interface_gate.get("status") == "pass"
        and int(interface_gate.get("interface_rows", -1)) == 194
        and interface_gate.get("projection_outcomes_opened_or_emitted") is False
        and interface_gate.get("final_or_unassigned_files_opened") is False
    ):
        raise ModelError("required upstream v3 target/interface gate is not passed and sealed")

    interface_fields, interface_rows = _read_csv(
        interface_root / "v3_model_interface.csv"
    )
    if tuple(interface_fields) != INTERFACE_FIELDS or len(interface_rows) != 194:
        raise ModelError("v3 model interface schema or row count changed")
    interface_by_id: dict[str, dict[str, str]] = {}
    for row in interface_rows:
        identifier = row.get("provider_neuron_id", "")
        if not identifier or identifier in interface_by_id:
            raise ModelError("interface neuron identifiers are empty or duplicated")
        interface_by_id[identifier] = row

    target_fields, target_rows = _read_csv(
        target_root / "primary_16_target_calls.csv"
    )
    if len(target_rows) != 200:
        raise ModelError("EP10-local target replay must contain 200 frozen rows")
    detected_fields = [field for field in target_fields if field.startswith("detected__")]
    expected_detected_fields = [f"detected__{coordinate}" for coordinate in PANEL]
    if detected_fields != expected_detected_fields:
        raise ModelError("target replay does not contain the ordered fixed 16 panel")
    required_target_fields = {
        "provider_neuron_id",
        "file",
        "fMOST_brain_id",
        "provider_sample_id",
        "global_eligible",
        "K",
        "S_coordinate_ids",
        *expected_detected_fields,
    }
    if not required_target_fields.issubset(target_fields):
        raise ModelError("target replay lacks a required frozen field")
    eligible_rows = [row for row in target_rows if row.get("global_eligible") == "yes"]
    if len(eligible_rows) != 194:
        raise ModelError("target replay does not contain exactly 194 eligible rows")
    if any(row.get("global_eligible") not in {"yes", "no"} for row in target_rows):
        raise ModelError("target replay eligibility is not binary")

    records: list[Record] = []
    target_ids: set[str] = set()
    for target_row in eligible_rows:
        identifier = target_row["provider_neuron_id"]
        if not identifier or identifier in target_ids:
            raise ModelError("eligible target identifiers are empty or duplicated")
        target_ids.add(identifier)
        try:
            interface_row = interface_by_id[identifier]
        except KeyError as error:
            raise ModelError(f"eligible target {identifier} is absent from the interface") from error
        for field in ("file", "fMOST_brain_id", "provider_sample_id"):
            if target_row[field] != interface_row[field]:
                raise ModelError(f"target/interface {field} mismatch for {identifier}")
        calls_text = [target_row[field] for field in expected_detected_fields]
        if any(value not in {"0", "1"} for value in calls_text):
            raise ModelError(f"nonbinary recorded-positive call for {identifier}")
        calls = tuple(int(value) for value in calls_text)
        try:
            k_value = int(target_row["K"])
        except ValueError as error:
            raise ModelError(f"invalid K for {identifier}") from error
        if k_value != sum(calls):
            raise ModelError(f"K disagrees with recorded-positive calls for {identifier}")
        reported_set = (
            target_row["S_coordinate_ids"].split(";")
            if target_row["S_coordinate_ids"]
            else []
        )
        expected_set = [coordinate for coordinate, call in zip(PANEL, calls) if call]
        if reported_set != expected_set:
            raise ModelError(f"recorded-positive set disagrees with calls for {identifier}")
        segment_text = interface_row["segment_unmapped"]
        terminal_text = interface_row["terminal_unmapped_report_only"]
        if segment_text not in {"0", "1"} or terminal_text not in {"0", "1"}:
            raise ModelError(f"nonbinary mapping flag for {identifier}")
        records.append(
            Record(
                provider_neuron_id=identifier,
                filename=interface_row["file"],
                group=interface_row["fMOST_brain_id"],
                sample_id=interface_row["provider_sample_id"],
                source_acronym=interface_row["source_acronym"],
                hemisphere=interface_row["provider_hemisphere"],
                soma_um=(
                    _finite_float(interface_row["soma_x_ccfv3_um"], "soma x"),
                    _finite_float(interface_row["soma_y_ccfv3_um"], "soma y"),
                    _finite_float(interface_row["soma_z_ccfv3_um"], "soma z"),
                ),
                segment_unmapped=int(segment_text),
                terminal_unmapped_report_only=int(terminal_text),
                outcome=calls,
                k=k_value,
            )
        )
    if target_ids != set(interface_by_id):
        raise ModelError("target/interface eligible identity sets differ")
    observed_groups = dict(sorted(Counter(record.group for record in records).items()))
    if observed_groups != GROUP_SIZES:
        raise ModelError(f"frozen 12-group support changed: {observed_groups}")
    records.sort(key=lambda record: (record.group, record.filename, record.provider_neuron_id))
    return records


def _prepare_output_root(path: Path, forbidden_roots: Sequence[Path]) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise ModelError("SCRATCH is unset; real M0 execution must run in a Sherlock job")
    scratch = Path(scratch_value).resolve(strict=True)
    if not path.is_absolute():
        raise ModelError("model output root must be an absolute Scratch path")
    if path.is_symlink():
        raise ModelError("model output root may not be a symlink")
    if path.exists():
        resolved = path.resolve(strict=True)
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise ModelError("model output root must be absent or empty")
    else:
        parent = path.parent.resolve(strict=True)
        resolved = parent / path.name
    if not resolved.is_relative_to(scratch):
        raise ModelError("model output root must be below SCRATCH")
    forbidden_resolved = [root.resolve(strict=True) for root in forbidden_roots]
    if any(
        resolved == root or resolved.is_relative_to(root) or root.is_relative_to(resolved)
        for root in forbidden_resolved
    ):
        raise ModelError("model output root must be disjoint from both input roots")
    resolved.mkdir(parents=False, exist_ok=True)
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or partial.is_symlink():
        raise ModelError(f"refusing symlink output below {path.parent}")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or partial.is_symlink():
        raise ModelError(f"refusing symlink output below {path.parent}")
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _optimizer_payload(result: OptimizerResult) -> dict[str, Any]:
    return {
        "objective": result.objective,
        "gradient_infinity_norm": result.gradient_infinity_norm,
        "iterations": result.iterations,
        "accepted_steps": result.accepted_steps,
        "relative_objective_change": result.relative_objective_change,
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    validate_contract(args.contract)
    records = load_real_records(args.target_root, args.interface_root)
    cross_fitted, outer_details, boundary_count = nested_outer_logo(records)

    output_root = _prepare_output_root(
        args.output_root, (args.target_root, args.interface_root)
    )
    if boundary_count >= 6:
        review = {
            "status": "stop_for_contract_review",
            "reason": "ridge_grid_boundary_selected_in_at_least_6_outer_folds",
            "outer_boundary_fold_count": boundary_count,
            "outer_folds": outer_details,
            "real_target_root_basename": args.target_root.name,
            "development_cells": 194,
            "development_groups": 12,
            "final_or_unassigned_opened": False,
        }
        _atomic_json(output_root / "v3_m0_fit_summary.json", review)
        raise BoundaryReview(
            f"ridge-grid boundary selected in {boundary_count} outer folds"
        )

    full_selected_lambda, full_selection = tune_lambda_logo(records)
    full_transform, full_fit = _fit_single(records, full_selected_lambda)
    score = score_and_calibration(records, cross_fitted)
    if args.temperature_diagnostic:
        score["calibration"]["scalar_temperature_slope"] = temperature_diagnostic(
            records, cross_fitted
        )
    else:
        score["calibration"]["scalar_temperature_slope"] = {
            "status": "not_requested_optional_diagnostic"
        }

    y, k_values, groups = outcome_matrix(records)
    log2_probability = cross_fitted.log_probability_nats / LN2
    uniform = np.asarray(
        [-math.log2(math.comb(16, int(k))) for k in k_values], dtype=float
    )
    prediction_fields = [
        "provider_neuron_id",
        "file",
        "fMOST_brain_id",
        "provider_sample_id",
        "K",
        "M0_outer_cross_fitted_log2_probability",
        "uniform_fixed_K_log2_probability",
        "M0_minus_uniform_gain_bits",
        *[f"exact_inclusion_marginal__{coordinate}" for coordinate in PANEL],
    ]
    prediction_rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        row: dict[str, object] = {
            "provider_neuron_id": record.provider_neuron_id,
            "file": record.filename,
            "fMOST_brain_id": record.group,
            "provider_sample_id": record.sample_id,
            "K": record.k,
            "M0_outer_cross_fitted_log2_probability": format(
                log2_probability[index], ".17g"
            ),
            "uniform_fixed_K_log2_probability": format(uniform[index], ".17g"),
            "M0_minus_uniform_gain_bits": format(
                log2_probability[index] - uniform[index], ".17g"
            ),
        }
        for target_index, coordinate in enumerate(PANEL):
            row[f"exact_inclusion_marginal__{coordinate}"] = format(
                cross_fitted.marginals[index, target_index], ".17g"
            )
        prediction_rows.append(row)
    _atomic_csv(
        output_root / "v3_m0_cross_fitted_predictions.csv",
        prediction_fields,
        prediction_rows,
    )

    full_refit = {
        "status": "pass",
        "purpose": "full_development_coefficient_artifact_not_performance_estimate",
        "selected_lambda": full_selected_lambda,
        "selection": full_selection,
        "optimizer": _optimizer_payload(full_fit),
        "ordered_covariates": list(COVARIATES),
        "ordered_targets": list(PANEL),
        "soma_training_mean": full_transform.soma_mean.tolist(),
        "soma_training_population_sd": full_transform.soma_scale.tolist(),
        "target_Helmert_C_16_by_15": helmert(16).tolist(),
        "coefficient_B_17_by_15": full_fit.coefficients.tolist(),
        "final_or_unassigned_opened": False,
    }
    _atomic_json(output_root / "v3_m0_full_refit.json", full_refit)

    summary = {
        "status": "pass",
        "stage": "v3_M0_development_outer_cross_fit_and_full_refit",
        "analysis_label": "exploratory",
        "estimand": "mapped_positive_terminal_observation_pattern_conditional_on_recorded_K",
        "development_cells": len(records),
        "development_groups": len(set(groups)),
        "target_panel_size": y.shape[1],
        "target_root_basename": args.target_root.name,
        "historical_target_root_rejected": True,
        "outer_folds": outer_details,
        "outer_boundary_fold_count": boundary_count,
        "full_development_selected_lambda": full_selected_lambda,
        "scores_and_calibration": score,
        "identifiers_or_groups_used_as_covariates": False,
        "terminal_unmapped_used_as_covariate": False,
        "raw_lengths_or_terminal_counts_used": False,
        "zeros_interpreted_as_absence_or_negative": False,
        "final_or_unassigned_opened": False,
        "temperature_diagnostic_applied_to_predictions_or_scores": False,
        "output_root": str(output_root),
    }
    _atomic_json(output_root / "v3_m0_fit_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def _brute_fixed_k(eta: np.ndarray, k: int) -> tuple[float, np.ndarray, tuple[tuple[int, ...], ...], np.ndarray]:
    states = tuple(itertools.combinations(range(eta.size), k))
    scores = np.asarray([float(np.sum(eta[list(state)])) for state in states])
    logz = _logsumexp(scores)
    probabilities = np.exp(scores - logz)
    marginals = np.zeros(eta.size, dtype=float)
    for state, probability in zip(states, probabilities):
        if state:
            marginals[list(state)] += probability
    return logz, marginals, states, probabilities


def _synthetic_optimizer_and_recovery() -> dict[str, Any]:
    p = 6
    k = 3
    contrasts = helmert(p)
    planted_reduced = np.asarray([[0.62, -0.41, 0.28, -0.19, 0.11]])
    planted_eta = (np.ones((1, 1)) @ planted_reduced) @ contrasts.T
    _, planted_marginals, states, probabilities = _brute_fixed_k(planted_eta[0], k)
    counts = np.maximum(1, np.rint(8000.0 * probabilities).astype(int))
    observations: list[tuple[int, ...]] = []
    for state, count in zip(states, counts):
        observations.extend([state] * int(count))
    y = np.zeros((len(observations), p), dtype=float)
    for row, state in enumerate(observations):
        y[row, list(state)] = 1.0
    x = np.ones((len(observations), 1), dtype=float)
    k_values = np.full(len(observations), k, dtype=int)
    groups = ["synthetic"] * len(observations)
    first = fit_lbfgs(
        x,
        y,
        k_values,
        groups,
        contrasts,
        1.0e-8,
    )
    second = fit_lbfgs(
        x,
        y,
        k_values,
        groups,
        contrasts,
        1.0e-8,
    )
    fitted_eta = (x[:1] @ first.coefficients) @ contrasts.T
    repeated_eta = (x[:1] @ second.coefficients) @ contrasts.T
    _, fitted_marginals, _, fitted_probabilities = _brute_fixed_k(fitted_eta[0], k)
    _, _, _, repeated_probabilities = _brute_fixed_k(repeated_eta[0], k)
    repeat_error = float(np.max(np.abs(fitted_probabilities - repeated_probabilities)))
    positive = probabilities > 0.0
    mean_kl_bits = float(
        np.sum(
            probabilities[positive]
            * np.log2(probabilities[positive] / fitted_probabilities[positive])
        )
    )
    marginal_mae = float(np.mean(np.abs(planted_marginals - fitted_marginals)))
    if repeat_error > 1.0e-10:
        raise AssertionError(f"optimizer repeat probability error {repeat_error}")
    if first.gradient_infinity_norm > 1.0e-6 or second.gradient_infinity_norm > 1.0e-6:
        raise AssertionError("synthetic optimizer did not reach the required gradient")
    if mean_kl_bits > 0.01 or marginal_mae > 0.02:
        raise AssertionError(
            f"planted recovery failed: KL={mean_kl_bits}, marginal_MAE={marginal_mae}"
        )
    return {
        "repeat_probability_max_absolute_error": repeat_error,
        "gradient_infinity_norm": first.gradient_infinity_norm,
        "iterations": first.iterations,
        "planted_mean_KL_bits": mean_kl_bits,
        "planted_marginal_MAE": marginal_mae,
    }


def run_self_test(contract_path: Path) -> dict[str, Any]:
    validate_contract(contract_path)
    checks: list[str] = []
    target_contrasts = helmert(16)
    orthonormal_error = float(
        np.max(np.abs(target_contrasts.T @ target_contrasts - np.eye(15)))
    )
    zero_sum_error = float(np.max(np.abs(target_contrasts.T @ np.ones(16))))
    if orthonormal_error > 1.0e-14 or zero_sum_error > 1.0e-14:
        raise AssertionError("frozen Helmert target coding failed")
    checks.append("orthonormal_16_by_15_Helmert_target_coding")

    eta = np.linspace(-0.71, 0.83, 16) + 0.07 * np.sin(np.arange(16))
    max_logz_error = 0.0
    max_marginal_error = 0.0
    max_offset_error = 0.0
    for k in range(17):
        dp_logz, dp_marginals = fixed_k_logz_marginals(eta, k)
        brute_logz, brute_marginals, states, _ = _brute_fixed_k(eta, k)
        max_logz_error = max(max_logz_error, abs(dp_logz - brute_logz))
        max_marginal_error = max(
            max_marginal_error,
            float(np.max(np.abs(dp_marginals - brute_marginals))),
        )
        offset = 0.713
        shifted_logz, shifted_marginals = fixed_k_logz_marginals(eta + offset, k)
        state = states[len(states) // 2]
        base_log_probability = float(np.sum(eta[list(state)])) - dp_logz
        shifted_log_probability = (
            float(np.sum((eta + offset)[list(state)])) - shifted_logz
        )
        max_offset_error = max(
            max_offset_error,
            abs(base_log_probability - shifted_log_probability),
            float(np.max(np.abs(dp_marginals - shifted_marginals))),
        )
    if max_logz_error > 1.0e-10 or max_marginal_error > 1.0e-10:
        raise AssertionError(
            f"DP/brute mismatch logZ={max_logz_error} marginal={max_marginal_error}"
        )
    if max_offset_error > 1.0e-12:
        raise AssertionError(f"common-offset invariance error {max_offset_error}")
    checks.extend(
        (
            "exact_prefix_suffix_DP_matches_bruteforce_for_K_0_through_16",
            "exact_inclusion_marginals_match_bruteforce",
            "common_target_offset_invariance",
        )
    )

    p_gradient = 6
    d_gradient = 3
    gradient_contrasts = helmert(p_gradient)
    x_gradient = np.column_stack(
        (
            np.ones(18),
            np.linspace(-1.0, 1.0, 18),
            np.cos(np.arange(18) * 0.31),
        )
    )
    k_gradient = np.asarray([(index % 5) + 1 for index in range(18)], dtype=int)
    y_gradient = np.zeros((18, p_gradient), dtype=float)
    for row, k_value in enumerate(k_gradient):
        active = tuple((row + offset * 2) % p_gradient for offset in range(k_value))
        if len(set(active)) != k_value:
            active = tuple(range(k_value))
        y_gradient[row, list(active)] = 1.0
    groups_gradient = [f"g{index % 3}" for index in range(18)]
    vector = np.linspace(-0.17, 0.21, d_gradient * (p_gradient - 1))
    _, analytic = objective_and_gradient(
        vector,
        x_gradient,
        y_gradient,
        k_gradient,
        groups_gradient,
        gradient_contrasts,
        0.13,
    )
    finite = np.empty_like(analytic)
    epsilon = 1.0e-6
    for index in range(vector.size):
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value, _ = objective_and_gradient(
            plus,
            x_gradient,
            y_gradient,
            k_gradient,
            groups_gradient,
            gradient_contrasts,
            0.13,
        )
        minus_value, _ = objective_and_gradient(
            minus,
            x_gradient,
            y_gradient,
            k_gradient,
            groups_gradient,
            gradient_contrasts,
            0.13,
        )
        finite[index] = (plus_value - minus_value) / (2.0 * epsilon)
    gradient_absolute_error = float(np.max(np.abs(analytic - finite)))
    gradient_relative_error = gradient_absolute_error / max(
        1.0e-15, float(np.max(np.abs(analytic))), float(np.max(np.abs(finite)))
    )
    if gradient_absolute_error > 1.0e-6 or gradient_relative_error > 1.0e-5:
        raise AssertionError(
            f"analytic gradient mismatch abs={gradient_absolute_error}, "
            f"rel={gradient_relative_error}"
        )
    checks.append("analytic_gradient_matches_central_finite_difference")

    optimizer = _synthetic_optimizer_and_recovery()
    checks.extend(
        (
            "deterministic_optimizer_repeat_probabilities",
            "planted_conditional_distribution_recovery",
        )
    )

    synthetic_groups = [
        group for group, size in GROUP_SIZES.items() for _ in range(size)
    ]
    if len(synthetic_groups) != 194:
        raise AssertionError("synthetic exact group vector does not total 194")
    outer_splits = logo_splits(synthetic_groups)
    if len(outer_splits) != 12:
        raise AssertionError("synthetic outer LOGO does not have 12 folds")
    for held_group, training_indexes, held_indexes in outer_splits:
        training_groups = [synthetic_groups[int(index)] for index in training_indexes]
        if len(logo_splits(training_groups)) != 11:
            raise AssertionError(f"inner LOGO for {held_group} does not have 11 folds")
        if len(held_indexes) != GROUP_SIZES[held_group]:
            raise AssertionError(f"held-out group size changed for {held_group}")
    group_order = {group: index for index, group in enumerate(GROUP_SIZES)}
    synthetic_values = [float(group_order[group]) for group in synthetic_groups]
    equal_score = equal_group_mean(synthetic_values, synthetic_groups)
    expected_equal_score = float(np.mean(list(group_order.values())))
    if abs(equal_score - expected_equal_score) > 1.0e-14:
        raise AssertionError("equal-group scoring does not match exact accounting")
    checks.append("exact_194_row_12_outer_11_inner_equal_group_fold_accounting")

    return {
        "status": "pass",
        "stage": "v3_M0_synthetic_self_test",
        "uses_real_target_interface_morphology_or_atlas_data": False,
        "final_or_unassigned_opened": False,
        "panel_size": 16,
        "covariate_count": 17,
        "target_contrast_count": 15,
        "max_DP_logZ_absolute_error": max_logz_error,
        "max_DP_marginal_absolute_error": max_marginal_error,
        "max_common_offset_invariance_absolute_error": max_offset_error,
        "gradient_max_absolute_error": gradient_absolute_error,
        "gradient_relative_error": gradient_relative_error,
        "optimizer_and_recovery": optimizer,
        "synthetic_fold_rows": len(synthetic_groups),
        "synthetic_outer_folds": len(outer_splits),
        "synthetic_inner_folds_per_outer": 11,
        "equal_group_accounting_value": equal_score,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    default_contract = Path(__file__).resolve().parent.parent / "v3_m0_model_contract.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path, default=default_contract)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--temperature-diagnostic",
        action="store_true",
        help="report the optional group-jackknife calibration slope diagnostic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if any(
            value is not None
            for value in (args.target_root, args.interface_root, args.output_root)
        ):
            parser.error("--self-test does not accept real target/interface/output paths")
        if args.temperature_diagnostic:
            parser.error("--temperature-diagnostic is available only with --execute")
        try:
            result = run_self_test(args.contract)
        except (ModelError, AssertionError, OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    missing = [
        name
        for name in ("target_root", "interface_root", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("--execute requires: " + ", ".join("--" + value.replace("_", "-") for value in missing))
    try:
        run_real(args)
    except BoundaryReview as error:
        print(f"CONTRACT REVIEW REQUIRED: {error}", file=sys.stderr)
        return 3
    except (ModelError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
