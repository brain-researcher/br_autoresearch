#!/usr/bin/env python3
"""Frozen, target-free qualification for the EP10 v3 M2 core engine.

This executable has no real-data input.  It constructs only deterministic
synthetic fixtures and independently checks the numerical contracts required
before any real M2 outcome may be opened.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402


FROZEN_SEED = 2026092810
FROZEN_GROUPS = 8
FROZEN_ROWS_PER_GROUP = 17
GRAM_CROSS_TOLERANCE = 1.0e-10
ORTHOGONALITY_TOLERANCE = 1.0e-10
PROJECTOR_IDEMPOTENCE_TOLERANCE = 1.0e-12
DERIVATIVE_ABSOLUTE_TOLERANCE = 1.0e-6
DERIVATIVE_RELATIVE_TOLERANCE = 1.0e-5
PROX_KKT_TOLERANCE = 1.0e-7
RANGE_NULL_PROBABILITY_TOLERANCE = 1.0e-10
RECOVERY_KL_MAXIMUM_BITS = 1.0e-6
RECOVERY_CANDIDATE_PROBABILITY_MAXIMUM = 1.0e-5
RECOVERY_PAIR_EVENT_MAE_MAXIMUM = 1.0e-5
RECOVERY_RHO = 1.0e-6
REPEAT_OBJECTIVE_TOLERANCE = 1.0e-12
REPEAT_CANDIDATE_PROBABILITY_TOLERANCE = 1.0e-10
REPEAT_IDENTIFIED_PAIR_CONTRAST_TOLERANCE = 1.0e-10
TINY_NONZERO_GRADIENT_NORM = 5.0e-10
LN2 = math.log(2.0)

SMOKE_GATE_ORDER = (
    "exact_all_K_enumeration_and_beta_zero_nesting",
    "independent_brute_force_Gram_cross",
    "all_120_coordinate_central_derivatives",
    "independent_small_convex_prox_KKT",
    "range_null_probability_invariance",
)
FULL_GATE_ORDER = SMOKE_GATE_ORDER + (
    "lambda_maximum_rho_1_zero_solution_and_zero_gradient",
    "warm_and_zero_deterministic_repeats",
    "frozen_population_recovery",
)
LAMBDA_PENALTY_CASES = (
    ("lasso", 1.0, "lasso"),
    ("elastic_net", 0.25, "elastic_net_alpha_0.25"),
    ("elastic_net", 0.5, "elastic_net_alpha_0.5"),
    ("elastic_net", 0.75, "elastic_net_alpha_0.75"),
    ("group_lasso", 1.0, "group_lasso"),
)
CORE_PENALTY_CASE_NAMES = ("lasso", "elastic_net", "group_lasso")


class QualificationError(RuntimeError):
    """A frozen target-free M2 qualification gate failed closed."""


@dataclass(frozen=True)
class FrozenFixture:
    cells: tuple[m1.Cell, ...]
    moments: tuple[m1.UniformCellMoments, ...]
    candidates: m1.CandidateLibrary
    geometry: m1.GeometryEngine
    pairs: m2.PairFeatureLibrary
    fixed_m1_theta: np.ndarray
    k2_indexes: tuple[int, ...]
    small_transform: m2.ResidualTransform
    small_prepared: m2.PreparedM2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _maximum_absolute(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=float)
    _require(bool(np.all(np.isfinite(values))), "qualification value is nonfinite")
    return float(np.max(np.abs(values))) if values.size else 0.0


def _independent_lexicographic_combinations(
    size: int, choose: int
) -> Sequence[tuple[int, ...]]:
    """Build lexicographic combinations without the engine's itertools path."""
    _require(0 <= choose <= size, "invalid independent combination request")
    if choose == 0:
        return ((),)
    indexes = list(range(choose))
    result: list[tuple[int, ...]] = []
    while True:
        result.append(tuple(indexes))
        position = choose - 1
        while position >= 0 and indexes[position] == size - choose + position:
            position -= 1
        if position < 0:
            break
        indexes[position] += 1
        for suffix in range(position + 1, choose):
            indexes[suffix] = indexes[suffix - 1] + 1
    return tuple(result)


def _frozen_fixture() -> FrozenFixture:
    cells, moments, _, candidates, geometry, fixed_m1_theta = (
        m1.generate_synthetic_dataset(
            groups=FROZEN_GROUPS,
            rows_per_group=FROZEN_ROWS_PER_GROUP,
            seed=FROZEN_SEED,
        )
    )
    expected_rows = FROZEN_GROUPS * FROZEN_ROWS_PER_GROUP
    _require(len(cells) == expected_rows, "frozen recovery row count changed")
    counts = Counter(cell.group for cell in cells)
    _require(
        len(counts) == FROZEN_GROUPS
        and set(counts.values()) == {FROZEN_ROWS_PER_GROUP},
        "frozen recovery group inventory changed",
    )
    for group in sorted(counts):
        observed_k = sorted(cell.k for cell in cells if cell.group == group)
        _require(
            observed_k == list(range(m1.PANEL_SIZE + 1)),
            f"frozen recovery K schedule changed for {group}",
        )
    pairs = m2.PairFeatureLibrary(candidates)
    k2_indexes = tuple(index for index, cell in enumerate(cells) if cell.k == 2)
    _require(
        len(k2_indexes) == FROZEN_GROUPS
        and len({cells[index].group for index in k2_indexes}) == FROZEN_GROUPS,
        "small K=2 qualification fixture must have one row per group",
    )
    small_transform, small_prepared = m2.fit_residual_transform(
        cells,
        moments,
        candidates,
        geometry,
        pairs,
        k2_indexes,
    )
    return FrozenFixture(
        cells=tuple(cells),
        moments=tuple(moments),
        candidates=candidates,
        geometry=geometry,
        pairs=pairs,
        fixed_m1_theta=np.asarray(fixed_m1_theta, dtype=float),
        k2_indexes=k2_indexes,
        small_transform=small_transform,
        small_prepared=small_prepared,
    )


def _independent_m1_candidate_probabilities(
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    cell_index: int,
    fixture: FrozenFixture,
) -> np.ndarray:
    """Compute fixed-M1 probabilities without calling the M2 score helper."""
    vector = np.asarray(fixed_m1_theta, dtype=float)
    b_matrix = vector[: m1.B_PARAMETER_DIMENSION].reshape(
        m1.DESIGN_DIMENSION, m1.TARGET_CONTRAST_DIMENSION
    )
    gamma = vector[m1.B_PARAMETER_DIMENSION :]
    cell = prepared.cells[cell_index]
    basis = fixture.candidates.basis(cell.k)
    target_coefficient = b_matrix.T @ prepared.m1_data.x[cell_index]
    scores = np.empty(basis.count, dtype=float)
    for start in range(0, basis.count, m1.BATCH_MAXIMUM):
        stop = min(start + m1.BATCH_MAXIMUM, basis.count)
        raw = fixture.geometry.raw_batch(cell, start, stop)
        z = (
            raw - prepared.m1_data.geometry_centers[cell_index]
        ) / prepared.m1_data.geometry_scales
        scores[start:stop] = (
            basis.contrasts[start:stop] @ target_coefficient + z @ gamma
        )
    maximum = float(np.max(scores))
    weights = np.exp(scores - maximum)
    return weights / float(np.sum(weights, dtype=float))


def qualify_exact_enumeration_and_beta_zero_nesting(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    expected_pair_index = tuple(
        (left, right)
        for left in range(m1.PANEL_SIZE)
        for right in range(left + 1, m1.PANEL_SIZE)
    )
    _require(
        m2.PAIR_INDEX == expected_pair_index,
        "physical pair index is not the exact parent lexicographic order",
    )
    expected_counts = {
        k: math.comb(m1.PANEL_SIZE, k)
        for k in range(m1.PANEL_SIZE + 1)
    }
    candidate_counts = fixture.candidates.all_counts()
    pair_counts = fixture.pairs.all_counts()
    _require(candidate_counts == expected_counts, "all-K candidate counts changed")
    _require(pair_counts == expected_counts, "all-K pair-basis counts changed")
    maximum_pair_count_error = 0
    candidate_rows_checked = 0
    pair_indicator_rows_checked = 0
    candidate_row_order_by_k: dict[str, bool] = {}
    pair_indicator_order_by_k: dict[str, bool] = {}
    for k in range(m1.PANEL_SIZE + 1):
        candidate_basis = fixture.candidates.basis(k)
        indicators = fixture.pairs.basis(k).indicators
        expected_combinations = _independent_lexicographic_combinations(
            m1.PANEL_SIZE, k
        )
        _require(
            len(expected_combinations) == math.comb(m1.PANEL_SIZE, k),
            f"independent lexicographic enumeration count changed for K={k}",
        )
        rows_match = True
        indicator_rows_match = True
        for row_index, expected_combination in enumerate(expected_combinations):
            expected_membership = np.zeros(m1.PANEL_SIZE, dtype=np.uint8)
            expected_membership[list(expected_combination)] = 1
            if not np.array_equal(
                candidate_basis.membership[row_index], expected_membership
            ):
                rows_match = False
                break
            expected_pair_indicators = np.asarray(
                [
                    expected_membership[left] * expected_membership[right]
                    for left, right in expected_pair_index
                ],
                dtype=np.uint8,
            )
            if not np.array_equal(
                indicators[row_index], expected_pair_indicators
            ):
                indicator_rows_match = False
                break
            candidate_rows_checked += 1
            pair_indicator_rows_checked += 1
        candidate_row_order_by_k[str(k)] = rows_match
        pair_indicator_order_by_k[str(k)] = indicator_rows_match
        _require(rows_match, f"candidate row order is not lexicographic for K={k}")
        _require(
            indicator_rows_match,
            f"pair indicator columns do not follow the parent pair order for K={k}",
        )
        observed = np.sum(indicators, axis=1, dtype=int)
        expected = math.comb(k, 2)
        maximum_pair_count_error = max(
            maximum_pair_count_error,
            int(np.max(np.abs(observed - expected))) if observed.size else 0,
        )
    _require(
        maximum_pair_count_error == 0,
        "a fixed-K candidate has the wrong choose(K,2) pair count",
    )

    first_group = fixture.cells[0].group
    indexes = tuple(
        index for index, cell in enumerate(fixture.cells) if cell.group == first_group
    )
    _require(
        len(indexes) == 17
        and sorted(fixture.cells[index].k for index in indexes) == list(range(17)),
        "beta-zero nesting fixture does not cover every K exactly once",
    )
    cells = tuple(fixture.cells[index] for index in indexes)
    moments = tuple(fixture.moments[index] for index in indexes)
    prepared = m2.apply_residual_transform(
        cells, moments, fixture.small_transform
    )
    fixed_m1_theta = 0.12 * fixture.fixed_m1_theta
    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    m2_evaluation = m2.evaluate_m2(
        zero,
        fixed_m1_theta,
        prepared,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
        need_predictions=True,
    )
    m1_evaluation = m1.evaluate_model(
        fixed_m1_theta,
        prepared.m1_data,
        fixture.candidates,
        fixture.geometry,
        0.0,
        need_hessian=False,
        need_predictions=True,
    )
    assert m2_evaluation.log_probability_nats is not None
    assert m2_evaluation.marginals is not None
    assert m1_evaluation.log_probability_nats is not None
    assert m1_evaluation.marginals is not None
    log_error = _maximum_absolute(
        m2_evaluation.log_probability_nats - m1_evaluation.log_probability_nats
    )
    marginal_error = _maximum_absolute(
        m2_evaluation.marginals - m1_evaluation.marginals
    )
    candidate_probability_error = 0.0
    degenerate_exact_zero = True
    for cell_index, cell in enumerate(prepared.cells):
        m2_distribution = m2.candidate_distribution(
            zero,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        m1_probability = _independent_m1_candidate_probabilities(
            fixed_m1_theta, prepared, cell_index, fixture
        )
        candidate_probability_error = max(
            candidate_probability_error,
            _maximum_absolute(m2_distribution.probabilities - m1_probability),
        )
        if cell.k in (0, m1.PANEL_SIZE):
            degenerate_exact_zero = bool(
                degenerate_exact_zero
                and m2_distribution.probabilities.shape == (1,)
                and m2_distribution.probabilities[0] == 1.0
                and m2_distribution.log_probabilities[0] == 0.0
                and m2_evaluation.log_probability_nats[cell_index] == 0.0
                and m1_evaluation.log_probability_nats[cell_index] == 0.0
            )
    _require(log_error <= 1.0e-12, "beta-zero M1 log-probability nesting failed")
    _require(
        marginal_error <= 1.0e-12,
        "beta-zero M1 inclusion-marginal nesting failed",
    )
    _require(
        candidate_probability_error <= 1.0e-12,
        "beta-zero M1 candidate-probability nesting failed",
    )
    _require(
        degenerate_exact_zero,
        "K=0 or K=16 did not have singleton probability one and exact zero gain",
    )
    return {
        "status": "pass",
        "K_values_checked": list(range(m1.PANEL_SIZE + 1)),
        "candidate_counts_by_K": {
            str(k): candidate_counts[k] for k in range(m1.PANEL_SIZE + 1)
        },
        "pair_basis_counts_by_K": {
            str(k): pair_counts[k] for k in range(m1.PANEL_SIZE + 1)
        },
        "total_panel_subsets": sum(candidate_counts.values()),
        "candidate_rows_checked_for_exact_lexicographic_order": (
            candidate_rows_checked
        ),
        "pair_indicator_rows_checked_against_exact_PAIR_INDEX": (
            pair_indicator_rows_checked
        ),
        "candidate_row_order_exact_by_K": candidate_row_order_by_k,
        "pair_indicator_order_exact_by_K": pair_indicator_order_by_k,
        "PAIR_INDEX_exact_lexicographic_order": True,
        "PAIR_INDEX_coordinates_checked": len(expected_pair_index),
        "maximum_choose_K_2_pair_count_error": maximum_pair_count_error,
        "beta_zero_M1_log_probability_maximum_absolute_error": log_error,
        "beta_zero_M1_candidate_probability_maximum_absolute_error": (
            candidate_probability_error
        ),
        "beta_zero_M1_marginal_maximum_absolute_error": marginal_error,
        "K_0_and_K_16_M2_minus_M1_gain_exact_zero": degenerate_exact_zero,
        "nesting_tolerance": 1.0e-12,
    }


def _explicit_centered_features(
    fixture: FrozenFixture,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate F and H directly, without M2 cached moment helpers."""
    prepared = fixture.small_prepared
    gram_ff = np.zeros((m1.PARAMETER_DIMENSION, m1.PARAMETER_DIMENSION))
    gram_fh = np.zeros((m1.PARAMETER_DIMENSION, m2.PAIR_DIMENSION))
    gram_hh = np.zeros((m2.PAIR_DIMENSION, m2.PAIR_DIMENSION))
    residual_gram = np.zeros((m2.PAIR_DIMENSION, m2.PAIR_DIMENSION))
    projection = fixture.small_transform.projection_f_to_h

    for cell_index, cell in enumerate(prepared.cells):
        basis = fixture.candidates.basis(cell.k)
        membership = basis.membership.astype(float, copy=False)
        pair_values = (
            membership[:, fixture.pairs.left]
            * membership[:, fixture.pairs.right]
        )
        h = pair_values - np.mean(pair_values, axis=0)
        contrast = basis.contrasts - np.mean(basis.contrasts, axis=0)
        raw_geometry = fixture.geometry.raw_batch(cell, 0, basis.count)
        z = (
            raw_geometry - prepared.m1_data.geometry_centers[cell_index]
        ) / prepared.m1_data.geometry_scales
        target = np.einsum(
            "a,nr->nar",
            prepared.m1_data.x[cell_index],
            contrast,
            optimize=True,
        ).reshape(basis.count, m1.B_PARAMETER_DIMENSION)
        f = np.column_stack((target, z))
        denominator = float(basis.count)
        weight = float(prepared.weights[cell_index])
        gram_ff += weight * (f.T @ f) / denominator
        gram_fh += weight * (f.T @ h) / denominator
        gram_hh += weight * (h.T @ h) / denominator
        residual = h - f @ projection
        residual_gram += weight * (residual.T @ residual) / denominator
    return (
        0.5 * (gram_ff + gram_ff.T),
        gram_fh,
        0.5 * (gram_hh + gram_hh.T),
        0.5 * (residual_gram + residual_gram.T),
    )


def qualify_brute_force_gram_cross(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    gram_ff, gram_fh, gram_hh, direct_residual = _explicit_centered_features(
        fixture
    )
    engine_ff, engine_prepared = m1.fold_gram(
        fixture.cells, fixture.moments, fixture.k2_indexes
    )
    _require(
        np.array_equal(
            engine_prepared.x, fixture.small_prepared.m1_data.x
        ),
        "independent Gram fixture and M2 fixture are not aligned",
    )
    projection = fixture.small_transform.projection_f_to_h
    fitted_cross = gram_ff @ projection
    orthogonality_error = _maximum_absolute(gram_fh - fitted_cross)
    schur = (
        gram_hh
        - projection.T @ gram_fh
        - gram_fh.T @ projection
        + projection.T @ gram_ff @ projection
    )
    schur = 0.5 * (schur + schur.T)
    scales = np.sqrt(np.diag(direct_residual))
    scaled = direct_residual / scales[:, None] / scales[None, :]
    errors = {
        "G_FF": _maximum_absolute(gram_ff - engine_ff),
        "G_FH_fitted_cross": orthogonality_error,
        "Schur_vs_direct_residual_Gram": _maximum_absolute(
            schur - direct_residual
        ),
        "direct_residual_Gram": _maximum_absolute(
            direct_residual - fixture.small_transform.residual_gram
        ),
        "pair_residual_RMS": _maximum_absolute(
            scales - fixture.small_transform.pair_residual_scales
        ),
        "scaled_residual_Gram": _maximum_absolute(
            scaled - fixture.small_transform.scaled_residual_gram
        ),
    }
    maximum = max(errors.values())
    _require(
        maximum <= GRAM_CROSS_TOLERANCE,
        f"independent brute-force Gram/cross error {maximum:.17g} exceeds "
        f"{GRAM_CROSS_TOLERANCE:.17g}",
    )
    _require(
        orthogonality_error <= ORTHOGONALITY_TOLERANCE,
        "independent residual features are not orthogonal to M1 F",
    )
    projector = fixture.small_transform.range_projector
    symmetry = _maximum_absolute(projector - projector.T)
    idempotence = _maximum_absolute(projector @ projector - projector)
    _require(
        symmetry <= PROJECTOR_IDEMPOTENCE_TOLERANCE,
        f"range projector symmetry {symmetry:.17g} exceeds "
        f"{PROJECTOR_IDEMPOTENCE_TOLERANCE:.17g}",
    )
    _require(
        idempotence <= PROJECTOR_IDEMPOTENCE_TOLERANCE,
        f"range projector idempotence {idempotence:.17g} exceeds "
        f"{PROJECTOR_IDEMPOTENCE_TOLERANCE:.17g}",
    )
    return {
        "status": "pass",
        "training_cells": len(fixture.small_prepared.cells),
        "training_groups": len(
            {cell.group for cell in fixture.small_prepared.cells}
        ),
        "candidate_rows_enumerated": sum(
            fixture.candidates.basis(cell.k).count
            for cell in fixture.small_prepared.cells
        ),
        "component_maximum_absolute_errors": errors,
        "maximum_Gram_or_cross_error": maximum,
        "maximum_orthogonality_error": orthogonality_error,
        "projector_symmetry_maximum_absolute_error": symmetry,
        "projector_idempotence_maximum_absolute_error": idempotence,
        "Gram_cross_tolerance": GRAM_CROSS_TOLERANCE,
        "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
        "projector_idempotence_tolerance": PROJECTOR_IDEMPOTENCE_TOLERANCE,
    }


def qualify_all_coordinate_derivatives(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    projector = fixture.small_transform.range_projector
    raw = np.sin(np.arange(1, m2.PAIR_DIMENSION + 1, dtype=float) * 0.271)
    beta = 0.035 * (projector @ raw)
    analytic = m2.evaluate_m2(
        beta,
        0.12 * fixture.fixed_m1_theta,
        fixture.small_prepared,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
    ).gradient
    step = 1.0e-6
    numeric = np.empty(m2.PAIR_DIMENSION, dtype=float)
    for coordinate in range(m2.PAIR_DIMENSION):
        direction = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        direction[coordinate] = step
        plus = m2.evaluate_m2(
            beta + direction,
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        ).objective
        minus = m2.evaluate_m2(
            beta - direction,
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        ).objective
        numeric[coordinate] = (plus - minus) / (2.0 * step)
    absolute_error = np.abs(numeric - analytic)
    scale = np.maximum(np.abs(numeric), np.abs(analytic))
    relative_error = np.divide(
        absolute_error,
        scale,
        out=np.zeros_like(absolute_error),
        where=scale > 0.0,
    )
    accepted = (absolute_error <= DERIVATIVE_ABSOLUTE_TOLERANCE) | (
        relative_error <= DERIVATIVE_RELATIVE_TOLERANCE
    )
    _require(
        bool(np.all(accepted)),
        "at least one of the 120 physical-coordinate central derivatives "
        "failed both absolute and relative tolerances",
    )
    return {
        "status": "pass",
        "coordinates_checked": m2.PAIR_DIMENSION,
        "central_difference_step": step,
        "maximum_absolute_error": float(np.max(absolute_error)),
        "maximum_relative_error": float(np.max(relative_error)),
        "absolute_tolerance": DERIVATIVE_ABSOLUTE_TOLERANCE,
        "relative_tolerance": DERIVATIVE_RELATIVE_TOLERANCE,
        "coordinates_passing_absolute_or_relative_rule": int(
            np.sum(accepted)
        ),
    }


def _coordinate_range_transform(
    source: m2.ResidualTransform,
) -> tuple[m2.ResidualTransform, tuple[int, ...]]:
    selected_groups = (0, 9, 28)
    selected = tuple(
        sorted(
            edge
            for group_index in selected_groups
            for edge in m2.BILATERAL_GROUPS[group_index][1]
        )
    )
    _require(len(selected) == 9, "small convex coordinate range changed")
    basis = np.eye(m2.PAIR_DIMENSION, dtype=float)[:, list(selected)]
    projector = basis @ basis.T
    transform = dataclasses.replace(
        source,
        range_basis=basis,
        range_projector=projector,
        residual_gram=projector.copy(),
        scaled_residual_gram=projector.copy(),
    )
    return transform, selected


def _independent_coordinate_prox(
    source: np.ndarray,
    amount: float,
    specification: m2.PenaltySpec,
    selected: Sequence[int],
) -> np.ndarray:
    chosen = set(int(index) for index in selected)
    result = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    if specification.kind in {"lasso", "elastic_net"}:
        alpha = 1.0 if specification.kind == "lasso" else specification.l1_ratio
        denominator = (
            1.0
            if specification.kind == "lasso"
            else 1.0 + amount * (1.0 - alpha)
        )
        index = np.asarray(sorted(chosen), dtype=int)
        values = source[index]
        result[index] = (
            np.sign(values)
            * np.maximum(np.abs(values) - amount * alpha, 0.0)
            / denominator
        )
        return result
    for _, indexes in m2.BILATERAL_GROUPS:
        block_set = set(indexes)
        overlap = block_set & chosen
        _require(
            not overlap or overlap == block_set,
            "small group-lasso reference range split a frozen group",
        )
        if not overlap:
            continue
        index = np.asarray(indexes, dtype=int)
        values = source[index]
        norm = float(np.linalg.norm(values))
        threshold = amount * math.sqrt(float(len(indexes)))
        if norm > threshold:
            result[index] = (1.0 - threshold / norm) * values
    return result


def _independent_coordinate_kkt(
    solution: np.ndarray,
    source: np.ndarray,
    amount: float,
    specification: m2.PenaltySpec,
    selected: Sequence[int],
) -> float:
    selected_set = set(int(index) for index in selected)
    residuals: list[float] = []
    if specification.kind in {"lasso", "elastic_net"}:
        alpha = 1.0 if specification.kind == "lasso" else specification.l1_ratio
        for index in sorted(selected_set):
            value = float(solution[index])
            smooth = value - float(source[index])
            if specification.kind == "elastic_net":
                smooth += amount * (1.0 - alpha) * value
            if value != 0.0:
                residuals.append(abs(smooth + amount * alpha * math.copysign(1.0, value)))
            else:
                residuals.append(max(abs(smooth) - amount * alpha, 0.0))
        return max(residuals, default=0.0)
    for _, indexes in m2.BILATERAL_GROUPS:
        block_set = set(indexes)
        if not block_set.issubset(selected_set):
            continue
        index = np.asarray(indexes, dtype=int)
        block = solution[index]
        gradient = block - source[index]
        norm = float(np.linalg.norm(block))
        radius = amount * math.sqrt(float(len(indexes)))
        if norm > 0.0:
            residuals.append(
                float(np.linalg.norm(gradient + radius * block / norm, ord=np.inf))
            )
        else:
            residuals.append(max(float(np.linalg.norm(gradient)) - radius, 0.0))
    return max(residuals, default=0.0)


def _oblique_range_transform(
    source: m2.ResidualTransform,
) -> tuple[m2.ResidualTransform, np.ndarray]:
    coordinates = np.arange(1, m2.PAIR_DIMENSION + 1, dtype=float)
    direction = np.sin(0.173 * coordinates) + 0.37 * np.cos(0.091 * coordinates)
    direction /= float(np.linalg.norm(direction))
    basis = direction[:, None]
    projector = np.outer(direction, direction)
    transform = dataclasses.replace(
        source,
        range_basis=basis,
        range_projector=projector,
        residual_gram=projector.copy(),
        scaled_residual_gram=projector.copy(),
    )
    return transform, direction


def _independent_unconstrained_prox(
    source: np.ndarray,
    amount: float,
    specification: m2.PenaltySpec,
) -> np.ndarray:
    if specification.kind == "lasso":
        return np.sign(source) * np.maximum(np.abs(source) - amount, 0.0)
    if specification.kind == "elastic_net":
        alpha = specification.l1_ratio
        return (
            np.sign(source)
            * np.maximum(np.abs(source) - amount * alpha, 0.0)
            / (1.0 + amount * (1.0 - alpha))
        )
    result = np.zeros_like(source)
    for _, indexes in m2.BILATERAL_GROUPS:
        index = np.asarray(indexes, dtype=int)
        block = source[index]
        norm = float(np.linalg.norm(block))
        threshold = amount * math.sqrt(float(len(indexes)))
        if norm > threshold:
            result[index] = (1.0 - threshold / norm) * block
    return result


def _independent_oblique_prox_and_kkt(
    source: np.ndarray,
    amount: float,
    specification: m2.PenaltySpec,
    direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    scalar_source = float(np.dot(direction, source))
    if specification.kind == "lasso":
        nonsmooth_scale = float(np.sum(np.abs(direction), dtype=float))
        ridge_scale = 0.0
    elif specification.kind == "elastic_net":
        nonsmooth_scale = specification.l1_ratio * float(
            np.sum(np.abs(direction), dtype=float)
        )
        ridge_scale = 1.0 - specification.l1_ratio
    else:
        nonsmooth_scale = math.fsum(
            math.sqrt(float(len(indexes)))
            * float(np.linalg.norm(direction[list(indexes)]))
            for _, indexes in m2.BILATERAL_GROUPS
        )
        ridge_scale = 0.0
    threshold = amount * nonsmooth_scale
    scalar = (
        math.copysign(max(abs(scalar_source) - threshold, 0.0), scalar_source)
        / (1.0 + amount * ridge_scale)
        if scalar_source != 0.0
        else 0.0
    )
    solution = scalar * direction
    smooth_derivative = scalar - scalar_source + amount * ridge_scale * scalar
    if scalar != 0.0:
        kkt = abs(
            smooth_derivative + threshold * math.copysign(1.0, scalar)
        )
    else:
        kkt = max(abs(smooth_derivative) - threshold, 0.0)
    return solution, kkt


def qualify_small_convex_prox_kkt(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    transform, selected = _coordinate_range_transform(fixture.small_transform)
    source = (
        0.9 * np.sin(np.arange(1, m2.PAIR_DIMENSION + 1, dtype=float) * 0.19)
        + 0.4 * np.cos(np.arange(1, m2.PAIR_DIMENSION + 1, dtype=float) * 0.07)
    )
    amount = 0.17
    coordinate_results: dict[str, Any] = {}
    oblique_results: dict[str, Any] = {}
    oblique_transform, oblique_direction = _oblique_range_transform(
        fixture.small_transform
    )
    maximum_dykstra_certificate_residual = 0.0
    for kind, alpha in (
        ("lasso", 1.0),
        ("elastic_net", 0.5),
        ("group_lasso", 1.0),
    ):
        specification = m2.PenaltySpec(kind, 1.0, alpha)
        reference = _independent_coordinate_prox(
            source, amount, specification, selected
        )
        observed = m2.prox_penalty_range(
            source, amount, specification, transform
        )
        error = _maximum_absolute(observed.vector - reference)
        kkt = _independent_coordinate_kkt(
            observed.vector, source, amount, specification, selected
        )
        _require(observed.certified, f"{kind} constrained prox was not certified")
        observed_certificate_residual = max(
            observed.fixed_point_residual,
            observed.range_residual,
            observed.invariant_residual,
        )
        maximum_dykstra_certificate_residual = max(
            maximum_dykstra_certificate_residual,
            observed_certificate_residual,
        )
        _require(
            observed_certificate_residual <= m2.PROX_TOLERANCE,
            f"{kind} Dykstra certificate residual "
            f"{observed_certificate_residual:.17g} exceeds the absolute tolerance",
        )
        _require(
            error <= PROX_KKT_TOLERANCE,
            f"{kind} prox differs from independent convex reference by {error:.17g}",
        )
        _require(
            kkt <= PROX_KKT_TOLERANCE,
            f"{kind} independent KKT residual {kkt:.17g} exceeds tolerance",
        )
        coordinate_results[kind] = {
            "reference_maximum_absolute_error": error,
            "independent_KKT_infinity_residual": kkt,
            "engine_fixed_point_residual": observed.fixed_point_residual,
            "engine_range_residual": observed.range_residual,
            "engine_invariant_residual": observed.invariant_residual,
            "iterations": observed.iterations,
        }
        oblique_reference, oblique_kkt = _independent_oblique_prox_and_kkt(
            source, amount, specification, oblique_direction
        )
        oblique_observed = m2.prox_penalty_range(
            source, amount, specification, oblique_transform
        )
        oblique_error = _maximum_absolute(
            oblique_observed.vector - oblique_reference
        )
        projected_unconstrained = oblique_transform.range_projector @ (
            _independent_unconstrained_prox(source, amount, specification)
        )
        noncommutation_witness = _maximum_absolute(
            projected_unconstrained - oblique_reference
        )
        _require(
            noncommutation_witness > 1.0e-6,
            f"{kind} oblique prox fixture did not witness noncommutation",
        )
        _require(
            oblique_observed.certified,
            f"{kind} oblique constrained prox was not certified",
        )
        oblique_certificate_residual = max(
            oblique_observed.fixed_point_residual,
            oblique_observed.range_residual,
            oblique_observed.invariant_residual,
        )
        maximum_dykstra_certificate_residual = max(
            maximum_dykstra_certificate_residual,
            oblique_certificate_residual,
        )
        _require(
            oblique_certificate_residual <= m2.PROX_TOLERANCE,
            f"{kind} oblique Dykstra certificate residual "
            f"{oblique_certificate_residual:.17g} exceeds the absolute tolerance",
        )
        _require(
            oblique_error <= PROX_KKT_TOLERANCE,
            f"{kind} oblique prox reference error {oblique_error:.17g} exceeds tolerance",
        )
        _require(
            oblique_kkt <= PROX_KKT_TOLERANCE,
            f"{kind} oblique scalar KKT residual {oblique_kkt:.17g} exceeds tolerance",
        )
        oblique_results[kind] = {
            "reference_maximum_absolute_error": oblique_error,
            "independent_scalar_KKT_residual": oblique_kkt,
            "prox_then_project_noncommutation_witness": noncommutation_witness,
            "engine_fixed_point_residual": oblique_observed.fixed_point_residual,
            "engine_range_residual": oblique_observed.range_residual,
            "engine_invariant_residual": oblique_observed.invariant_residual,
            "iterations": oblique_observed.iterations,
        }
    return {
        "status": "pass",
        "embedded_convex_dimension": len(selected),
        "oblique_convex_dimension": 1,
        "penalty_case_order": list(CORE_PENALTY_CASE_NAMES),
        "prox_amount": amount,
        "tolerance": PROX_KKT_TOLERANCE,
        "Dykstra_absolute_infinity_norm_tolerance": m2.PROX_TOLERANCE,
        "Dykstra_maximum_certificate_residual": (
            maximum_dykstra_certificate_residual
        ),
        "coordinate_aligned_penalties": coordinate_results,
        "oblique_noncommuting_penalties": oblique_results,
    }


def _maximum_probability_difference(
    left_beta: np.ndarray,
    right_beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    fixture: FrozenFixture,
) -> float:
    maximum = 0.0
    for cell_index in range(len(prepared.cells)):
        left = m2.candidate_distribution(
            left_beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        right = m2.candidate_distribution(
            right_beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        maximum = max(
            maximum,
            _maximum_absolute(left.probabilities - right.probabilities),
        )
    return maximum


def _equal_group_m2_minus_m1_pair_contrasts(
    beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    fixture: FrozenFixture,
) -> np.ndarray:
    """Return all 120 identified equal-group M2-minus-M1 pair contrasts."""
    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    group_sums: dict[str, np.ndarray] = {}
    group_counts: Counter[str] = Counter()
    for cell_index, cell in enumerate(prepared.cells):
        fitted = m2.candidate_distribution(
            beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        baseline = m2.candidate_distribution(
            zero,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        indicators = fixture.pairs.basis(cell.k).indicators.astype(
            float, copy=False
        )
        difference = (
            fitted.probabilities @ indicators
            - baseline.probabilities @ indicators
        )
        if cell.group not in group_sums:
            group_sums[cell.group] = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        group_sums[cell.group] += difference
        group_counts[cell.group] += 1
    groups = sorted(group_sums)
    _require(groups, "identified pair-contrast fixture has no groups")
    group_means = np.vstack(
        [group_sums[group] / float(group_counts[group]) for group in groups]
    )
    contrasts = np.mean(group_means, axis=0)
    _require(
        contrasts.shape == (m2.PAIR_DIMENSION,)
        and bool(np.all(np.isfinite(contrasts))),
        "identified pair-contrast vector is malformed or nonfinite",
    )
    return contrasts


def qualify_lambda_maximum_rho_one_and_zero_gradient(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    """Exercise every frozen rho=1 penalty and the zero-gradient boundary."""
    fixed_m1_theta = 0.12 * fixture.fixed_m1_theta
    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    base_gradient = m2.evaluate_m2(
        zero,
        fixed_m1_theta,
        fixture.small_prepared,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
    ).gradient
    _require(
        np.count_nonzero(base_gradient) > 0,
        "lambda qualification base gradient is unexpectedly zero",
    )
    degenerate_indexes = tuple(
        index
        for index, cell in enumerate(fixture.cells)
        if cell.k in (0, m1.PANEL_SIZE)
    )
    boundary_prepared = m2.apply_residual_transform(
        tuple(fixture.cells[index] for index in degenerate_indexes),
        tuple(fixture.moments[index] for index in degenerate_indexes),
        fixture.small_transform,
    )
    exact_zero_gradient = m2.evaluate_m2(
        zero,
        fixed_m1_theta,
        boundary_prepared,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
    ).gradient
    _require(
        np.count_nonzero(exact_zero_gradient) == 0,
        "synthetic exact-zero lambda fixture is not bit-exact zero",
    )

    direction = fixture.small_transform.range_basis[:, 0].copy()
    direction /= float(np.linalg.norm(direction, ord=np.inf))
    tiny_correction = TINY_NONZERO_GRADIENT_NORM * direction
    tiny_gradient = m2.evaluate_m2(
        zero,
        fixed_m1_theta,
        boundary_prepared,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
        linear_correction=tiny_correction,
    ).gradient
    tiny_norm = float(np.linalg.norm(tiny_gradient, ord=np.inf))
    _require(
        0.0 < tiny_norm < 0.01 * m2.KKT_TOLERANCE,
        "tiny-nonzero lambda fixture does not lie below the removed zero floor",
    )

    cases: dict[str, Any] = {}
    for kind, alpha, case_name in LAMBDA_PENALTY_CASES:
        maximum = m2.fold_lambda_maximum(
            fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            kind,
            l1_ratio=alpha,
        )
        _require(maximum.value > 0.0, f"{case_name} lambda_max is not positive")
        relative_width = (
            (maximum.upper_bracket - maximum.lower_bracket)
            / maximum.upper_bracket
        )
        _require(
            relative_width <= m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE,
            f"{case_name} lambda_max bracket did not meet relative tolerance",
        )
        fitted = m2.fit_m2(
            fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            m2.PenaltySpec(kind, maximum.value, alpha),
        )
        _require(
            fitted.status == "pass"
            and fitted.iterations == 0
            and np.count_nonzero(fitted.beta) == 0
            and fitted.kkt_residual <= m2.KKT_TOLERANCE,
            f"{case_name} rho=1 did not certify the exact zero solution",
        )

        exact_zero_maximum = m2.fold_lambda_maximum(
            fixed_m1_theta,
            boundary_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            kind,
            l1_ratio=alpha,
        )
        _require(
            exact_zero_maximum
            == m2.LambdaMaximumResult(0.0, 0.0, 0.0, 0.0, 0),
            f"{case_name} exact-zero gradient did not return exact M1 lambda behavior",
        )

        tiny_maximum = m2.fold_lambda_maximum(
            fixed_m1_theta,
            boundary_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            kind,
            l1_ratio=alpha,
            linear_correction=tiny_correction,
        )
        _require(
            tiny_maximum.value > 0.0,
            f"{case_name} tiny nonzero gradient was collapsed to lambda_max=0",
        )
        tiny_relative_width = (
            (tiny_maximum.upper_bracket - tiny_maximum.lower_bracket)
            / tiny_maximum.upper_bracket
        )
        _require(
            tiny_relative_width
            <= m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE,
            f"{case_name} tiny-gradient bracket did not meet relative tolerance",
        )
        cases[case_name] = {
            "family": kind,
            "alpha": alpha,
            "rho": 1.0,
            "lambda_max": maximum.value,
            "lambda_bracket_relative_width": relative_width,
            "rho_1_beta_exact_zero": True,
            "rho_1_fit_iterations": fitted.iterations,
            "rho_1_KKT_residual": fitted.kkt_residual,
            "exact_zero_gradient_lambda_max": exact_zero_maximum.value,
            "tiny_nonzero_gradient_lambda_max": tiny_maximum.value,
            "tiny_nonzero_lambda_bracket_relative_width": tiny_relative_width,
        }
    return {
        "status": "pass",
        "penalty_case_order": [case[2] for case in LAMBDA_PENALTY_CASES],
        "rho_1_KKT_tolerance": m2.KKT_TOLERANCE,
        "relative_bracket_tolerance": (
            m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE
        ),
        "initial_upper_bound": m2.LAMBDA_MAXIMUM_INITIAL_UPPER,
        "maximum_expansions": m2.LAMBDA_MAXIMUM_MAXIMUM_EXPANSIONS,
        "zero_KKT_solver": {
            "accelerated_iteration_budget": (
                m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS
            ),
            "smooth_minimax_iteration_budget": (
                m2.ZERO_KKT_MINIMAX_MAXIMUM_ITERATIONS
            ),
            "smooth_minimax_smoothing_exponent_range": [
                m2.ZERO_KKT_MINIMAX_SMOOTHING_MINIMUM_EXPONENT,
                m2.ZERO_KKT_MINIMAX_SMOOTHING_MAXIMUM_EXPONENT,
            ],
            "smooth_minimax_newton_steps_per_smoothing": (
                m2.ZERO_KKT_MINIMAX_NEWTON_STEPS_PER_SMOOTHING
            ),
            "active_face_multiplier_polish": (
                "candidate_only_guarded_separator_acceptance"
            ),
            "active_face_roundoff_multiplier": (
                m2.ZERO_KKT_ACTIVE_FACE_ROUNDOFF_MULTIPLIER
            ),
            "active_face_backward_error_multiplier": (
                m2.ZERO_KKT_ACTIVE_FACE_BACKWARD_ERROR_MULTIPLIER
            ),
            "active_face_budget": "shared_smooth_minimax_iteration_budget",
            "shared_total_iteration_cap": m2.PROX_MAXIMUM_ITERATIONS,
            "feasible_acceptance": "direct_dual_witness_only",
            "infeasible_acceptance": (
                "guarded_range_space_support_separator_only"
            ),
        },
        "exact_zero_gradient_bitwise_all_zero": True,
        "tiny_nonzero_gradient_infinity_norm": tiny_norm,
        "removed_absolute_zero_floor": 0.01 * m2.KKT_TOLERANCE,
        "cases": cases,
    }


def qualify_range_null_probability_invariance(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    projector = fixture.small_transform.range_projector
    raw_range = np.cos(
        np.arange(1, m2.PAIR_DIMENSION + 1, dtype=float) * 0.113
    )
    beta = 0.04 * (projector @ raw_range)
    raw_null = np.sin(
        np.arange(1, m2.PAIR_DIMENSION + 1, dtype=float) * 0.347
    )
    null = raw_null - projector @ raw_null
    norm = float(np.linalg.norm(null))
    _require(norm > 1.0e-8, "range-null invariance fixture has no null direction")
    null *= 3.0 / norm
    range_leakage = _maximum_absolute(projector @ null)
    error = _maximum_probability_difference(
        beta,
        beta + null,
        0.12 * fixture.fixed_m1_theta,
        fixture.small_prepared,
        fixture,
    )
    _require(
        error <= RANGE_NULL_PROBABILITY_TOLERANCE,
        f"range-null probability error {error:.17g} exceeds tolerance",
    )
    return {
        "status": "pass",
        "null_vector_l2_norm": float(np.linalg.norm(null)),
        "projected_null_maximum_absolute_value": range_leakage,
        "candidate_probability_maximum_absolute_error": error,
        "tolerance": RANGE_NULL_PROBABILITY_TOLERANCE,
    }


def _small_plants(fixture: FrozenFixture) -> Mapping[str, np.ndarray]:
    plants: dict[str, np.ndarray] = {}
    sparse = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    sparse[[3, 41, 89]] = (0.11, -0.09, 0.08)
    plants["lasso"] = sparse
    elastic = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    elastic[[5, 17, 52, 78, 101]] = (0.08, -0.07, 0.09, -0.06, 0.07)
    plants["elastic_net"] = elastic
    grouped = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    group_indexes = m2.BILATERAL_GROUPS[7][1]
    grouped[list(group_indexes)] = (0.07, -0.09, 0.08, -0.06)
    plants["group_lasso"] = grouped
    return plants


def _same_result(left: m2.M2OptimizerResult, right: m2.M2OptimizerResult) -> bool:
    return bool(
        np.array_equal(left.beta, right.beta)
        and left.objective == right.objective
        and left.smooth_objective == right.smooth_objective
        and left.penalty_value == right.penalty_value
        and left.iterations == right.iterations
        and left.lipschitz == right.lipschitz
        and left.kkt_residual == right.kkt_residual
        and left.relative_objective_change == right.relative_objective_change
        and left.range_residual == right.range_residual
        and left.status == right.status
        and left.telemetry == right.telemetry
    )


def qualify_deterministic_zero_and_warm_repeats(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    results: dict[str, Any] = {}
    for kind, alpha in (
        ("lasso", 1.0),
        ("elastic_net", 0.5),
        ("group_lasso", 1.0),
    ):
        planted = _small_plants(fixture)[kind]
        correction = m2.population_linear_correction(
            planted,
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        maximum = m2.fold_lambda_maximum(
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            kind,
            l1_ratio=alpha,
            linear_correction=correction,
        )
        _require(maximum.value > 0.0, f"{kind} deterministic fixture lambda_max is zero")
        specification = m2.PenaltySpec(kind, 0.1 * maximum.value, alpha)
        arguments = (
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            specification,
        )
        zero_first = m2.fit_m2(
            *arguments, linear_correction=correction
        )
        zero_second = m2.fit_m2(
            *arguments, linear_correction=correction
        )
        _require(
            _same_result(zero_first, zero_second),
            f"{kind} zero-start deterministic repeat changed",
        )
        _require(
            float(np.linalg.norm(zero_first.beta, ord=np.inf)) > 0.0,
            f"{kind} rho=0.1 qualification solution remained zero",
        )
        warm_first = m2.fit_m2(
            *arguments,
            initial=zero_first.beta,
            linear_correction=correction,
        )
        warm_second = m2.fit_m2(
            *arguments,
            initial=zero_first.beta,
            linear_correction=correction,
        )
        _require(
            _same_result(warm_first, warm_second),
            f"{kind} warm-start deterministic repeat changed",
        )
        for label, fitted in (
            ("zero", zero_first),
            ("warm", warm_first),
        ):
            _require(fitted.status == "pass", f"{kind} {label} fit did not pass")
            _require(
                fitted.kkt_residual <= m2.KKT_TOLERANCE,
                f"{kind} {label} fit failed KKT tolerance",
            )
            _require(
                fitted.relative_objective_change
                <= m2.RELATIVE_OBJECTIVE_TOLERANCE,
                f"{kind} {label} fit failed relative-objective tolerance",
            )
        cross_probability_error = _maximum_probability_difference(
            zero_first.beta,
            warm_first.beta,
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture,
        )
        zero_pair_contrasts = _equal_group_m2_minus_m1_pair_contrasts(
            zero_first.beta,
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture,
        )
        warm_pair_contrasts = _equal_group_m2_minus_m1_pair_contrasts(
            warm_first.beta,
            0.12 * fixture.fixed_m1_theta,
            fixture.small_prepared,
            fixture,
        )
        cross_pair_contrast_error = _maximum_absolute(
            zero_pair_contrasts - warm_pair_contrasts
        )
        cross_objective_error = abs(zero_first.objective - warm_first.objective)
        _require(
            cross_objective_error <= REPEAT_OBJECTIVE_TOLERANCE,
            f"{kind} zero-vs-warm objective changed by {cross_objective_error:.17g}",
        )
        _require(
            cross_probability_error <= REPEAT_CANDIDATE_PROBABILITY_TOLERANCE,
            f"{kind} zero-vs-warm probability changed by {cross_probability_error:.17g}",
        )
        _require(
            cross_pair_contrast_error
            <= REPEAT_IDENTIFIED_PAIR_CONTRAST_TOLERANCE,
            f"{kind} zero-vs-warm identified pair contrast changed by "
            f"{cross_pair_contrast_error:.17g}",
        )
        results[kind] = {
            "lambda_max": maximum.value,
            "rho": 0.1,
            "zero_repeat_exact": True,
            "warm_repeat_exact": True,
            "zero_iterations": zero_first.iterations,
            "warm_iterations": warm_first.iterations,
            "zero_KKT_residual": zero_first.kkt_residual,
            "warm_KKT_residual": warm_first.kkt_residual,
            "zero_vs_warm_objective_absolute_difference": cross_objective_error,
            "zero_vs_warm_candidate_probability_maximum_absolute_error": (
                cross_probability_error
            ),
            "identified_equal_group_pair_contrasts_checked": m2.PAIR_DIMENSION,
            "zero_vs_warm_identified_pair_contrast_maximum_absolute_difference": (
                cross_pair_contrast_error
            ),
        }
    return {
        "status": "pass",
        "repeat_definition": "two_exact_repeats_for_each_zero_and_warm_initialization",
        "penalty_case_order": list(CORE_PENALTY_CASE_NAMES),
        "zero_vs_warm_objective_tolerance": REPEAT_OBJECTIVE_TOLERANCE,
        "zero_vs_warm_candidate_probability_tolerance": (
            REPEAT_CANDIDATE_PROBABILITY_TOLERANCE
        ),
        "zero_vs_warm_identified_pair_contrast_tolerance": (
            REPEAT_IDENTIFIED_PAIR_CONTRAST_TOLERANCE
        ),
        "identified_equal_group_pair_contrasts_per_penalty": m2.PAIR_DIMENSION,
        "penalties": results,
    }


def _frozen_recovery_plants() -> Mapping[str, np.ndarray]:
    plants: dict[str, np.ndarray] = {}

    lasso = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    lasso_pairs = ((0, 3), (2, 7), (4, 11), (5, 14), (8, 13), (10, 15))
    lasso_values = (0.090, -0.080, 0.075, -0.085, 0.070, -0.065)
    for pair, value in zip(lasso_pairs, lasso_values, strict=True):
        lasso[m2.PAIR_TO_INDEX[pair]] = value
    _require(np.count_nonzero(lasso) == 6, "frozen lasso plant is not six-edge sparse")
    plants["lasso"] = lasso

    elastic = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    elastic_pairs = tuple((0, target) for target in range(1, 13))
    elastic_values = tuple(
        (0.055 + 0.002 * index) * (1.0 if index % 2 == 0 else -1.0)
        for index in range(12)
    )
    for pair, value in zip(elastic_pairs, elastic_values, strict=True):
        elastic[m2.PAIR_TO_INDEX[pair]] = value
    _require(
        np.count_nonzero(elastic) == 12
        and all(0 in pair for pair in elastic_pairs),
        "frozen elastic-net plant is not twelve correlated physical edges",
    )
    plants["elastic_net"] = elastic

    grouped = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    group_ordinals = (4, 19)
    group_values = (
        (0.070, -0.060, 0.055, -0.065),
        (-0.060, 0.070, -0.065, 0.055),
    )
    expected_edges: set[int] = set()
    for ordinal, values in zip(group_ordinals, group_values, strict=True):
        indexes = m2.BILATERAL_GROUPS[ordinal][1]
        _require(len(indexes) == 4, "frozen recovery group is not four-edge")
        grouped[list(indexes)] = values
        expected_edges.update(indexes)
    _require(
        np.count_nonzero(grouped) == 8
        and set(np.flatnonzero(grouped)) == expected_edges,
        "frozen group-lasso plant is not two complete four-edge groups",
    )
    plants["group_lasso"] = grouped
    return plants


def _pair_event_mae(
    left_beta: np.ndarray,
    right_beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    fixture: FrozenFixture,
) -> float:
    absolute_sum = 0.0
    count = 0
    for cell_index, cell in enumerate(prepared.cells):
        left = m2.candidate_distribution(
            left_beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        right = m2.candidate_distribution(
            right_beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        pair_indicators = fixture.pairs.basis(cell.k).indicators.astype(
            float, copy=False
        )
        left_probability = left.probabilities @ pair_indicators
        right_probability = right.probabilities @ pair_indicators
        absolute_sum += float(
            np.sum(np.abs(left_probability - right_probability), dtype=float)
        )
        count += m2.PAIR_DIMENSION
    return absolute_sum / float(count)


def qualify_frozen_population_recovery(
    fixture: FrozenFixture,
) -> Mapping[str, Any]:
    full_transform, full_prepared = m2.fit_residual_transform(
        fixture.cells,
        fixture.moments,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
    )
    _require(
        bool(full_transform.diagnostics["rank_stable"]),
        "frozen recovery transform rank is unstable",
    )
    plants = _frozen_recovery_plants()
    results: dict[str, Any] = {}
    for kind, alpha in (
        ("lasso", 1.0),
        ("elastic_net", 0.5),
        ("group_lasso", 1.0),
    ):
        planted = plants[kind]
        correction = m2.population_linear_correction(
            planted,
            fixture.fixed_m1_theta,
            full_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        planted_gradient = m2.evaluate_m2(
            planted,
            fixture.fixed_m1_theta,
            full_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            linear_correction=correction,
        ).gradient
        planted_gradient_error = _maximum_absolute(planted_gradient)
        _require(
            planted_gradient_error <= 1.0e-10,
            f"{kind} exact population correction did not center the planted model",
        )
        maximum = m2.fold_lambda_maximum(
            fixture.fixed_m1_theta,
            full_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            kind,
            l1_ratio=alpha,
            linear_correction=correction,
        )
        _require(maximum.value > 0.0, f"{kind} recovery lambda_max is zero")
        specification = m2.PenaltySpec(
            kind, RECOVERY_RHO * maximum.value, alpha
        )
        fitted = m2.fit_m2(
            fixture.fixed_m1_theta,
            full_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            specification,
            linear_correction=correction,
        )
        _require(fitted.status == "pass", f"{kind} population recovery did not pass")
        comparison = m2.compare_conditional_probabilities(
            planted,
            fitted.beta,
            fixture.fixed_m1_theta,
            full_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        kl_bits = comparison["mean_KL_nats"] / LN2
        probability_error = comparison[
            "maximum_candidate_probability_absolute_error"
        ]
        pair_mae = _pair_event_mae(
            planted,
            fitted.beta,
            fixture.fixed_m1_theta,
            full_prepared,
            fixture,
        )
        _require(
            kl_bits <= RECOVERY_KL_MAXIMUM_BITS,
            f"{kind} recovery KL {kl_bits:.17g} bits exceeds threshold",
        )
        _require(
            probability_error <= RECOVERY_CANDIDATE_PROBABILITY_MAXIMUM,
            f"{kind} candidate probability error {probability_error:.17g} exceeds threshold",
        )
        _require(
            pair_mae <= RECOVERY_PAIR_EVENT_MAE_MAXIMUM,
            f"{kind} pair-event MAE {pair_mae:.17g} exceeds threshold",
        )
        results[kind] = {
            "alpha": alpha,
            "lambda_max": maximum.value,
            "rho": RECOVERY_RHO,
            "penalty_strength": specification.strength,
            "plant_nonzero_physical_edges": int(np.count_nonzero(planted)),
            "population_correction_planted_gradient_maximum_absolute_error": (
                planted_gradient_error
            ),
            "fit_iterations": fitted.iterations,
            "fit_KKT_residual": fitted.kkt_residual,
            "fit_relative_objective_change": fitted.relative_objective_change,
            "mean_conditional_KL_bits": kl_bits,
            "candidate_probability_maximum_absolute_error": probability_error,
            "pair_event_probability_MAE": pair_mae,
        }
    return {
        "status": "pass",
        "penalty_case_order": list(CORE_PENALTY_CASE_NAMES),
        "fixture": {
            "seed": FROZEN_SEED,
            "groups": FROZEN_GROUPS,
            "rows_per_group": FROZEN_ROWS_PER_GROUP,
            "cells": len(fixture.cells),
            "K_schedule": "each_group_contains_every_K_0_through_16_once",
            "residual_rank": full_transform.rank,
            "rank_at_tau_over_10_tau_ten_tau": full_transform.diagnostics[
                "rank_at_tau_over_10_tau_ten_tau"
            ],
        },
        "outcome": "exact_population_cross_entropy_not_finite_sample_acceptance",
        "thresholds": {
            "mean_conditional_KL_maximum_bits": RECOVERY_KL_MAXIMUM_BITS,
            "candidate_probability_maximum_absolute_error": (
                RECOVERY_CANDIDATE_PROBABILITY_MAXIMUM
            ),
            "pair_event_probability_MAE_maximum": (
                RECOVERY_PAIR_EVENT_MAE_MAXIMUM
            ),
        },
        "plant_contract": {
            "lasso": "six_nonzero_sparse_physical_edges",
            "elastic_net": (
                "twelve_nonzero_physical_edges_sharing_target_0_alpha_0.5"
            ),
            "group_lasso": {
                "complete_four_edge_group_ordinals": [4, 19],
                "complete_four_edge_group_names": [
                    m2.BILATERAL_GROUPS[4][0],
                    m2.BILATERAL_GROUPS[19][0],
                ],
            },
            "raw_nonidentified_coefficient_recovery_required": False,
        },
        "cases": results,
    }


def run_qualification(*, smoke: bool = False) -> Mapping[str, Any]:
    started = time.monotonic()
    fixture = _frozen_fixture()
    gates: dict[str, Any] = {}
    gates["exact_all_K_enumeration_and_beta_zero_nesting"] = (
        qualify_exact_enumeration_and_beta_zero_nesting(fixture)
    )
    gates["independent_brute_force_Gram_cross"] = (
        qualify_brute_force_gram_cross(fixture)
    )
    gates["all_120_coordinate_central_derivatives"] = (
        qualify_all_coordinate_derivatives(fixture)
    )
    gates["independent_small_convex_prox_KKT"] = (
        qualify_small_convex_prox_kkt(fixture)
    )
    gates["range_null_probability_invariance"] = (
        qualify_range_null_probability_invariance(fixture)
    )
    if not smoke:
        gates["lambda_maximum_rho_1_zero_solution_and_zero_gradient"] = (
            qualify_lambda_maximum_rho_one_and_zero_gradient(fixture)
        )
        gates["warm_and_zero_deterministic_repeats"] = (
            qualify_deterministic_zero_and_warm_repeats(fixture)
        )
        gates["frozen_population_recovery"] = (
            qualify_frozen_population_recovery(fixture)
        )
    expected_gate_order = SMOKE_GATE_ORDER if smoke else FULL_GATE_ORDER
    observed_gate_order = tuple(gates)
    passing_gate_order = tuple(
        gate_name
        for gate_name, gate in gates.items()
        if gate.get("status") == "pass"
    )
    _require(
        observed_gate_order == expected_gate_order,
        "qualification gate inventory differs from the frozen mode inventory",
    )
    _require(
        passing_gate_order == expected_gate_order,
        "qualification report contains a nonpassing or vacuous required gate",
    )
    case_inventory: dict[str, Any] = {
        "K_values": list(range(m1.PANEL_SIZE + 1)),
        "candidate_rows": 2**m1.PANEL_SIZE,
        "physical_pair_coordinates": m2.PAIR_DIMENSION,
        "prox_penalty_cases": list(CORE_PENALTY_CASE_NAMES),
    }
    if not smoke:
        case_inventory.update(
            {
                "lambda_rho_1_penalty_cases": [
                    case[2] for case in LAMBDA_PENALTY_CASES
                ],
                "deterministic_repeat_penalty_cases": list(
                    CORE_PENALTY_CASE_NAMES
                ),
                "population_recovery_penalty_cases": list(
                    CORE_PENALTY_CASE_NAMES
                ),
                "identified_pair_contrasts_per_repeat_case": (
                    m2.PAIR_DIMENSION
                ),
            }
        )
    return {
        "status": "smoke_pass_not_qualification" if smoke else "pass",
        "overall_gate_status": "pass",
        "stage": "v3_M2_target_free_frozen_core_contract_qualification",
        "mode": "smoke" if smoke else "full",
        "elapsed_wall_seconds": time.monotonic() - started,
        "execution": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "frozen_fixture": {
            "seed": FROZEN_SEED,
            "groups": FROZEN_GROUPS,
            "rows_per_group": FROZEN_ROWS_PER_GROUP,
            "cells": FROZEN_GROUPS * FROZEN_ROWS_PER_GROUP,
            "every_K_0_through_16_once_per_group": True,
        },
        "gate_inventory": {
            "expected_gate_order": list(expected_gate_order),
            "observed_gate_order": list(observed_gate_order),
            "passing_gate_order": list(passing_gate_order),
            "required_gate_count": len(expected_gate_order),
            "all_required_gates_present_and_passed": True,
        },
        "case_inventory": case_inventory,
        "gates": gates,
        "controller_replay_qualification_included": False,
        "real_target_membership_or_M2_effects_opened": False,
        "actual_K_opened": False,
        "M0_full_refit_or_telemetry_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only non-optimizer smoke gates; not a qualification pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run_qualification(smoke=arguments.smoke)
    except (
        QualificationError,
        m2.M2Error,
        m1.M1Error,
        AssertionError,
        ArithmeticError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if arguments.output is not None:
        _atomic_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
