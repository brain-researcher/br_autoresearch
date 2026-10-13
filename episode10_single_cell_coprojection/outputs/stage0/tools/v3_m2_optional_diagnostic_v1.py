#!/usr/bin/env python3
"""Deterministic optional-family diagnostic for the EP10 v3 M2 controller.

The diagnostic consumes only in-memory, outer-cross-fitted M1 pair
probabilities and the corresponding observed fixed-K memberships.  It applies
equal-cell weighting within each conservative group and equal-group weighting
across the twelve groups.  This module has no filesystem entry point and does
not persist cell-level values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import fit_v3_m1 as m1
import fit_v3_m2 as m2
import v3_search_controller_v1 as controller


EXPECTED_GROUPS = 12
PANEL_SIZE = 16
PAIR_DIMENSION = 120
GRAM_SCHMIDT_THRESHOLD = 256.0 * PANEL_SIZE * np.finfo(float).eps
EIGENSYSTEM_CERTIFICATE_TOLERANCE = (
    1024.0 * PANEL_SIZE * np.finfo(float).eps
)
PROBABILITY_TOLERANCE = 2.0e-12


class OptionalDiagnosticError(RuntimeError):
    """The optional-family diagnostic failed a frozen invariant."""


def _readonly(array: np.ndarray, label: str) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise OptionalDiagnosticError(f"{label} is nonfinite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PairResidualRow:
    """One held-out cell and its outer-cross-fitted M1 pair probabilities."""

    cell_key: str
    group: str
    k: int
    observed_membership: np.ndarray
    m1_pair_probabilities: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.cell_key, str) or not self.cell_key.strip():
            raise OptionalDiagnosticError("diagnostic cell key must be nonempty")
        if self.group not in m1.GROUP_ORDER:
            raise OptionalDiagnosticError("diagnostic group is outside the frozen order")
        if (
            isinstance(self.k, (bool, np.bool_))
            or not isinstance(self.k, (int, np.integer))
            or not 0 <= int(self.k) <= PANEL_SIZE
        ):
            raise OptionalDiagnosticError("diagnostic K is outside 0..16")

        membership = np.asarray(self.observed_membership, dtype=float)
        if (
            membership.shape != (PANEL_SIZE,)
            or not np.all(np.isfinite(membership))
            or not np.all((membership == 0.0) | (membership == 1.0))
            or int(np.sum(membership, dtype=float)) != int(self.k)
        ):
            raise OptionalDiagnosticError(
                "observed membership is not a binary fixed-K panel vector"
            )

        probabilities = np.asarray(self.m1_pair_probabilities, dtype=float)
        if (
            probabilities.shape != (PAIR_DIMENSION,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < -PROBABILITY_TOLERANCE)
            or np.any(probabilities > 1.0 + PROBABILITY_TOLERANCE)
        ):
            raise OptionalDiagnosticError(
                "outer-cross-fitted M1 pair probabilities are malformed"
            )
        expected_pair_count = math.comb(int(self.k), 2)
        probability_sum = math.fsum(float(value) for value in probabilities)
        if not math.isclose(
            probability_sum,
            float(expected_pair_count),
            rel_tol=0.0,
            abs_tol=PROBABILITY_TOLERANCE * PAIR_DIMENSION,
        ):
            raise OptionalDiagnosticError(
                "M1 pair probabilities do not preserve the observed fixed K"
            )

        object.__setattr__(
            self,
            "observed_membership",
            _readonly(membership, "observed membership"),
        )
        object.__setattr__(
            self,
            "m1_pair_probabilities",
            _readonly(probabilities, "M1 pair probabilities"),
        )
        object.__setattr__(self, "k", int(self.k))


@dataclass(frozen=True)
class OptionalDiagnosticResult:
    """Controller metrics plus in-memory numerical witnesses."""

    diagnostics: controller.OptionalDiagnostics
    group_mean_residuals: np.ndarray
    full_matrix: np.ndarray
    leading_eigenvalue: float
    leading_loading: np.ndarray
    leave_one_group_cosines: tuple[float, ...]


def _observed_pair_indicator(membership: np.ndarray) -> np.ndarray:
    result = np.asarray(
        [membership[left] * membership[right] for left, right in m2.PAIR_INDEX],
        dtype=float,
    )
    return _readonly(result, "observed pair indicator")


def _group_mean_residuals(rows: Sequence[PairResidualRow]) -> np.ndarray:
    supplied = tuple(rows)
    if not supplied:
        raise OptionalDiagnosticError("optional diagnostic requires held-out rows")
    if any(type(row) is not PairResidualRow for row in supplied):
        raise OptionalDiagnosticError("optional diagnostic row has the wrong type")
    identifiers = tuple(row.cell_key for row in supplied)
    if len(set(identifiers)) != len(identifiers):
        raise OptionalDiagnosticError("optional diagnostic cell keys are not unique")

    by_group: dict[str, list[PairResidualRow]] = {
        group: [] for group in m1.GROUP_ORDER
    }
    for row in supplied:
        by_group[row.group].append(row)
    if len(m1.GROUP_ORDER) != EXPECTED_GROUPS or any(
        not by_group[group] for group in m1.GROUP_ORDER
    ):
        raise OptionalDiagnosticError(
            "optional diagnostic does not cover all twelve conservative groups"
        )

    group_means = np.empty((EXPECTED_GROUPS, PAIR_DIMENSION), dtype=float)
    for group_index, group in enumerate(m1.GROUP_ORDER):
        group_rows = sorted(by_group[group], key=lambda row: row.cell_key)
        residuals = tuple(
            _observed_pair_indicator(row.observed_membership)
            - row.m1_pair_probabilities
            for row in group_rows
        )
        for edge in range(PAIR_DIMENSION):
            group_means[group_index, edge] = (
                math.fsum(float(residual[edge]) for residual in residuals)
                / len(residuals)
            )
    return _readonly(group_means, "equal-cell group residual means")


def _edge_vector_to_matrix(edges: np.ndarray) -> np.ndarray:
    vector = np.asarray(edges, dtype=float)
    if vector.shape != (PAIR_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise OptionalDiagnosticError("pair-residual edge vector is malformed")
    matrix = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=float)
    for value, (left, right) in zip(vector, m2.PAIR_INDEX, strict=True):
        matrix[left, right] = float(value)
        matrix[right, left] = float(value)
    return _readonly(matrix, "symmetric pair-residual matrix")


def _canonicalize_sign(vector: np.ndarray) -> np.ndarray:
    result = np.array(vector, dtype=float, copy=True)
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _leading_eigenpair(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Apply the parent contract's exact-tie eigenpair order."""

    source = np.asarray(matrix, dtype=float)
    if (
        source.shape != (PANEL_SIZE, PANEL_SIZE)
        or not np.all(np.isfinite(source))
        or not np.array_equal(source, source.T)
        or not np.array_equal(np.diag(source), np.zeros(PANEL_SIZE, dtype=float))
    ):
        raise OptionalDiagnosticError("diagnostic matrix is not symmetric zero-diagonal")
    try:
        values, vectors = np.linalg.eigh(source)
    except np.linalg.LinAlgError as error:
        raise OptionalDiagnosticError("diagnostic eigensystem failed") from error
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise OptionalDiagnosticError("diagnostic eigensystem is nonfinite")

    records: list[tuple[float, int, np.ndarray]] = []
    start = 0
    while start < PANEL_SIZE:
        stop = start + 1
        while stop < PANEL_SIZE and values[stop] == values[start]:
            stop += 1
        cardinality = stop - start
        tied_vectors = np.asarray(vectors[:, start:stop], dtype=float)
        if cardinality == 1:
            basis = _canonicalize_sign(tied_vectors[:, 0])[:, np.newaxis]
        else:
            projector = tied_vectors @ tied_vectors.T
            projector = 0.5 * (projector + projector.T)
            selected: list[np.ndarray] = []
            for coordinate in range(PANEL_SIZE):
                residual = np.array(projector[:, coordinate], dtype=float, copy=True)
                for _ in range(2):
                    for accepted in selected:
                        residual -= float(np.dot(accepted, residual)) * accepted
                norm = float(np.linalg.norm(residual))
                if not math.isfinite(norm):
                    raise OptionalDiagnosticError(
                        "diagnostic tied-eigenspace Gram-Schmidt is nonfinite"
                    )
                if norm <= GRAM_SCHMIDT_THRESHOLD:
                    continue
                selected.append(_canonicalize_sign(residual / norm))
                if len(selected) == cardinality:
                    break
            if len(selected) != cardinality:
                raise OptionalDiagnosticError(
                    "diagnostic tied-eigenspace basis has the wrong cardinality"
                )
            basis = np.column_stack(selected)
            orthogonality_error = float(
                np.max(np.abs(basis.T @ basis - np.eye(cardinality)))
            )
            projector_error = float(
                np.max(np.abs(basis @ basis.T - projector))
            )
            if (
                not math.isfinite(orthogonality_error)
                or not math.isfinite(projector_error)
                or orthogonality_error > EIGENSYSTEM_CERTIFICATE_TOLERANCE
                or projector_error > EIGENSYSTEM_CERTIFICATE_TOLERANCE
            ):
                raise OptionalDiagnosticError(
                    "diagnostic tied-eigenspace basis failed its certificate"
                )
        for within in range(cardinality):
            records.append((float(values[start]), start + within, basis[:, within]))
        start = stop

    records.sort(key=lambda item: (-abs(item[0]), -item[0], item[1]))
    leading_value, _, leading_vector = records[0]
    residual = float(np.linalg.norm(source @ leading_vector - leading_value * leading_vector))
    scale = max(1.0, float(np.linalg.norm(source, ord=2)))
    if residual > EIGENSYSTEM_CERTIFICATE_TOLERANCE * scale:
        raise OptionalDiagnosticError("diagnostic leading eigenpair failed its residual certificate")
    return leading_value, _readonly(leading_vector, "diagnostic leading eigenvector")


def _leading_loading(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    eigenvalue, eigenvector = _leading_eigenpair(matrix)
    loading = np.asarray(
        [
            eigenvalue * eigenvector[left] * eigenvector[right]
            for left, right in m2.PAIR_INDEX
        ],
        dtype=float,
    )
    norm = math.hypot(*(float(value) for value in loading))
    if norm == 0.0:
        return float(eigenvalue), _readonly(
            np.zeros(PAIR_DIMENSION, dtype=float),
            "zero leading edge loading",
        )
    return float(eigenvalue), _readonly(
        loading / norm,
        "normalized leading edge loading",
    )


def diagnose_optional_families(
    rows: Sequence[PairResidualRow],
) -> OptionalDiagnosticResult:
    """Compute the frozen post-coverage optional-family diagnostic."""

    group_means = _group_mean_residuals(rows)
    full_edges = np.asarray(
        [
            math.fsum(float(group_means[group, edge]) for group in range(EXPECTED_GROUPS))
            / EXPECTED_GROUPS
            for edge in range(PAIR_DIMENSION)
        ],
        dtype=float,
    )
    full_matrix = _edge_vector_to_matrix(full_edges)
    edge_norm = math.hypot(*(float(value) for value in full_edges))

    if edge_norm == 0.0:
        diagnostics = controller.OptionalDiagnostics(0.0, 1.0, 0.0, 0)
        return OptionalDiagnosticResult(
            diagnostics=diagnostics,
            group_mean_residuals=group_means,
            full_matrix=full_matrix,
            leading_eigenvalue=0.0,
            leading_loading=_readonly(
                np.zeros(PAIR_DIMENSION, dtype=float),
                "zero full-development leading loading",
            ),
            leave_one_group_cosines=tuple(0.0 for _ in range(EXPECTED_GROUPS)),
        )

    leading_eigenvalue, leading_loading = _leading_loading(full_matrix)
    leading_energy = (abs(leading_eigenvalue) / (math.sqrt(2.0) * edge_norm)) ** 2
    if not math.isfinite(leading_energy) or not -1.0e-12 <= leading_energy <= 1.0 + 1.0e-12:
        raise OptionalDiagnosticError("diagnostic leading energy is invalid")
    leading_energy = min(1.0, max(0.0, leading_energy))

    fourth_sum = math.fsum(float(value) ** 4 for value in leading_loading)
    if not math.isfinite(fourth_sum) or fourth_sum <= 0.0:
        raise OptionalDiagnosticError("diagnostic edge participation is undefined")
    participation = 1.0 / fourth_sum
    if not 1.0 - 1.0e-10 <= participation <= PAIR_DIMENSION + 1.0e-10:
        raise OptionalDiagnosticError("diagnostic edge participation is outside 1..120")
    participation = min(float(PAIR_DIMENSION), max(1.0, participation))

    cosines: list[float] = []
    for omitted in range(EXPECTED_GROUPS):
        leave_edges = np.asarray(
            [
                math.fsum(
                    float(group_means[group, edge])
                    for group in range(EXPECTED_GROUPS)
                    if group != omitted
                )
                / (EXPECTED_GROUPS - 1)
                for edge in range(PAIR_DIMENSION)
            ],
            dtype=float,
        )
        if math.hypot(*(float(value) for value in leave_edges)) == 0.0:
            cosine = 0.0
        else:
            _, leave_loading = _leading_loading(_edge_vector_to_matrix(leave_edges))
            if not np.any(leave_loading):
                cosine = 0.0
            else:
                raw_cosine = float(np.dot(leading_loading, leave_loading))
                cosine = abs(raw_cosine)
                if not math.isfinite(cosine) or cosine > 1.0 + 1.0e-12:
                    raise OptionalDiagnosticError("diagnostic LOGO cosine is invalid")
                cosine = min(1.0, max(0.0, cosine))
        cosines.append(cosine)

    ordered_cosines = sorted(cosines)
    median = math.fsum(ordered_cosines[5:7]) / 2.0
    count = sum(value >= 0.70 for value in cosines)
    diagnostics = controller.OptionalDiagnostics(
        float(leading_energy),
        float(participation),
        float(median),
        int(count),
    )
    return OptionalDiagnosticResult(
        diagnostics=diagnostics,
        group_mean_residuals=group_means,
        full_matrix=full_matrix,
        leading_eigenvalue=float(leading_eigenvalue),
        leading_loading=leading_loading,
        leave_one_group_cosines=tuple(float(value) for value in cosines),
    )


__all__ = (
    "OptionalDiagnosticError",
    "OptionalDiagnosticResult",
    "PairResidualRow",
    "diagnose_optional_families",
)
