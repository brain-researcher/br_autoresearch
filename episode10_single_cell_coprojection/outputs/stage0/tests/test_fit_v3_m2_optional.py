#!/usr/bin/env python3
"""Focused target-free tests for the EP10 v3 optional low-rank engine."""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m2 as core  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


def _dct_columns(rank: int) -> np.ndarray:
    rows = np.arange(16, dtype=float)[:, np.newaxis]
    columns = np.arange(rank, dtype=float)[np.newaxis, :]
    result = np.cos(math.pi * (rows + 0.5) * columns / 16.0)
    result[:, 0] *= 1.0 / 4.0
    if rank > 1:
        result[:, 1:] *= math.sqrt(2.0 / 16.0)
    return result


def _hadamard16() -> np.ndarray:
    result = np.ones((1, 1), dtype=float)
    for _ in range(4):
        result = np.block([[result, result], [result, -result]])
    return result / 4.0


def _ordered_projector_basis(projector: np.ndarray, cardinality: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    for coordinate in range(16):
        residual = np.array(projector[:, coordinate], dtype=float, copy=True)
        for _ in range(2):
            for accepted in selected:
                residual -= float(np.dot(accepted, residual)) * accepted
        norm = float(np.linalg.norm(residual))
        if norm <= optional.CLUSTER_GRAM_SCHMIDT_RESIDUAL_THRESHOLD:
            continue
        accepted = residual / norm
        pivot = int(np.argmax(np.abs(accepted)))
        if accepted[pivot] < 0.0:
            accepted *= -1.0
        selected.append(accepted)
        if len(selected) == cardinality:
            break
    if len(selected) != cardinality:
        raise AssertionError("test projector did not yield its expected cardinality")
    return np.column_stack(selected)


class OptionalLowRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.cells,
            cls.moments,
            cls.prepared,
            cls.candidates,
            cls.geometry,
            cls.pairs,
            cls.fixed_m1_theta,
        ) = core.synthetic_k2_problem(rows=24)
        cls.fold_scales = optional.fold_low_rank_scales(
            cls.fixed_m1_theta,
            cls.prepared,
            cls.candidates,
            cls.geometry,
            cls.pairs,
        )

    def test_symmetric_gradient_map_is_the_offdiagonal_adjoint(self) -> None:
        edge_gradient = np.sin(np.arange(1, 121, dtype=float) * 0.17)
        raw = np.cos(
            np.arange(1, 257, dtype=float).reshape(16, 16) * 0.11
        )
        perturbation = 0.5 * (raw + raw.T)
        np.fill_diagonal(perturbation, 0.0)
        matrix_gradient = optional.physical_gradient_matrix(edge_gradient)
        left = float(
            np.dot(edge_gradient, optional.offdiagonal_vector(perturbation))
        )
        right = float(np.sum(matrix_gradient * perturbation, dtype=float))
        self.assertLess(abs(left - right), 2.0e-14)
        np.testing.assert_array_equal(np.diag(matrix_gradient), np.zeros(16))

    def test_deterministic_eigentree_exact_ties_order_and_sign(self) -> None:
        diagonal = np.asarray(
            (3.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0, *([0.0] * 9))
        )
        matrix = np.diag(diagonal)
        first = optional.deterministic_symmetric_eigh(matrix)
        second = optional.deterministic_symmetric_eigh(matrix)
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.vectors, second.vectors)
        np.testing.assert_array_equal(
            first.values[:7], np.asarray((3.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0))
        )
        np.testing.assert_array_equal(first.vectors[:, 0], np.eye(16)[:, 0])
        np.testing.assert_array_equal(first.vectors[:, 1], np.eye(16)[:, 1])
        for column in range(16):
            pivot = int(np.argmax(np.abs(first.vectors[:, column])))
            self.assertGreater(first.vectors[pivot, column], 0.0)

    def test_rotated_exact_tie_uses_canonical_cluster_basis_and_rank_cut(self) -> None:
        rotation = _hadamard16()
        diagonal = np.asarray(
            (3.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.25, 0.0, -0.25, -0.5,
             -0.75, -1.0, -1.25, -1.5, -1.75, -2.0)
        )
        matrix = (rotation * diagonal[np.newaxis, :]) @ rotation.T
        first = optional.deterministic_symmetric_eigh(matrix)
        second = optional.deterministic_symmetric_eigh(matrix)

        self.assertEqual(first.cluster_sizes[0], 2)
        self.assertEqual(first.values[0], first.values[1])
        self.assertLessEqual(
            abs(float(first.raw_values[-1] - first.raw_values[-2])),
            float(first.interval_radii[-1] + first.interval_radii[-2]),
        )
        expected_projector = rotation[:, :2] @ rotation[:, :2].T
        expected_basis = _ordered_projector_basis(expected_projector, 2)
        np.testing.assert_allclose(
            first.vectors[:, :2], expected_basis, atol=2.0e-14, rtol=0.0
        )
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.vectors, second.vectors)
        np.testing.assert_allclose(
            first.vectors[:, 0], expected_basis[:, 0], atol=2.0e-14, rtol=0.0
        )

    def test_transitive_near_cluster_and_separated_eigenpairs(self) -> None:
        unit_guard = (
            optional.EIGENPAIR_ROUNDING_GUARD_FACTOR * np.finfo(float).eps
        )
        near = (
            1.0,
            1.0 + 1.5 * unit_guard,
            1.0 + 3.0 * unit_guard,
        )
        separated = tuple(np.linspace(0.75, -0.45, 13))
        matrix = np.diag(np.asarray((*near, *separated), dtype=float))
        eigensystem = optional.deterministic_symmetric_eigh(matrix)

        self.assertEqual(eigensystem.cluster_sizes[0], 3)
        self.assertTrue(all(size == 1 for size in eigensystem.cluster_sizes[1:]))
        expected_mean = math.fsum(
            float(value) for value in eigensystem.raw_values[-3:]
        ) / 3
        np.testing.assert_array_equal(
            eigensystem.values[:3], np.full(3, expected_mean)
        )
        np.testing.assert_array_equal(eigensystem.vectors[:, :3], np.eye(16)[:, :3])

        lower = eigensystem.raw_values - eigensystem.interval_radii
        upper = eigensystem.raw_values + eigensystem.interval_radii
        self.assertLess(float(upper[-3]), float(lower[-1]))
        self.assertGreaterEqual(float(upper[-3]), float(lower[-2]))
        self.assertGreaterEqual(float(upper[-2]), float(lower[-1]))

    def test_eigencluster_certificates_fail_closed(self) -> None:
        matrix = np.diag(np.arange(16, dtype=float))
        malformed_vectors = 2.0 * np.eye(16, dtype=float)
        with mock.patch.object(
            optional.np.linalg,
            "eigh",
            return_value=(np.arange(16, dtype=float), malformed_vectors),
        ):
            with self.assertRaisesRegex(
                optional.LowRankError, "orthogonality/projector certificate"
            ):
                optional.deterministic_symmetric_eigh(matrix)

    def test_fold_scales_initialization_and_effective_beta(self) -> None:
        scales = self.fold_scales
        self.assertGreater(scales.smooth_lipschitz_upper_bound, 0.0)
        expected_spectral = float(
            np.max(np.abs(np.linalg.eigvalsh(scales.null_gradient_matrix)))
        )
        self.assertEqual(scales.raw_gradient_spectral_norm, expected_spectral)
        self.assertEqual(
            scales.lambda_reference,
            max(optional.LAMBDA_REFERENCE_FLOOR, expected_spectral),
        )
        initialization = optional.initialize_low_rank(
            2,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            fold_scales=scales,
        )
        state = initialization.state
        np.testing.assert_allclose(state.U.T @ state.U, np.eye(2), atol=2.0e-15)
        expected_beta = self.prepared.transform.range_projector @ optional.offdiagonal_vector(
            optional.low_rank_matrix(state)
        )
        np.testing.assert_allclose(
            optional.low_rank_effective_beta(state, self.prepared.transform),
            expected_beta,
            atol=3.0e-15,
            rtol=0.0,
        )
        self.assertEqual(
            initialization.negative_gradient_step,
            1.0 / max(1.0, scales.smooth_lipschitz_upper_bound),
        )
        eigensystem = optional.deterministic_symmetric_eigh(
            scales.null_gradient_matrix
        )
        expected_state = optional.canonicalize_state(
            optional.LowRankState(
                eigensystem.vectors[:, :2],
                -initialization.negative_gradient_step * eigensystem.values[:2],
            )
        )
        np.testing.assert_array_equal(
            initialization.selected_gradient_eigenvalues,
            eigensystem.values[:2],
        )
        np.testing.assert_array_equal(state.U, expected_state.U)
        np.testing.assert_array_equal(state.d, expected_state.d)

    def test_stiefel_tangent_qr_and_exact_objective_gradients(self) -> None:
        state = optional.LowRankState(
            _dct_columns(2), np.asarray((0.08, -0.035))
        )
        specification = optional.LowRankSpec(
            2, 0.1, self.fold_scales.lambda_reference
        )
        evaluation = optional.evaluate_low_rank(
            state,
            specification,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        tangent_error = state.U.T @ evaluation.gradient_U
        tangent_error += tangent_error.T
        self.assertLess(float(np.max(np.abs(tangent_error))), 2.0e-15)

        ambient_direction = np.sin(
            np.arange(1, 33, dtype=float).reshape(16, 2) * 0.29
        )
        tangent_direction = np.array(
            optional.stiefel_tangent_projection(state.U, ambient_direction),
            copy=True,
        )
        tangent_direction /= float(np.linalg.norm(tangent_direction))
        d_direction = np.asarray((0.37, -0.23), dtype=float)
        analytic = float(np.sum(evaluation.gradient_U * tangent_direction))
        analytic += float(np.dot(evaluation.gradient_d, d_direction))
        step = 1.0e-6

        def shifted(sign: float) -> optional.LowRankState:
            U = optional.qr_retraction(
                state.U, sign * tangent_direction, step
            )
            return optional.canonicalize_state(
                optional.LowRankState(U, state.d + sign * step * d_direction)
            )

        plus = optional.evaluate_low_rank(
            shifted(1.0),
            specification,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        minus = optional.evaluate_low_rank(
            shifted(-1.0),
            specification,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        numeric = (plus.objective - minus.objective) / (2.0 * step)
        self.assertLess(abs(numeric - analytic), 1.0e-7)
        retracted = optional.qr_retraction(state.U, tangent_direction, 0.3)
        np.testing.assert_allclose(retracted.T @ retracted, np.eye(2), atol=1.0e-15)

    def test_hybrid_formula_penalties_and_exact_chain_gradients(self) -> None:
        specification = optional.HybridSpec(
            rank=2,
            rho_low=0.1,
            rho_sparse=0.3,
            lambda_reference=self.fold_scales.lambda_reference,
            core_lasso_lambda_maximum=0.8,
            named_edges=(core.PAIR_INDEX[0], core.PAIR_INDEX[7], core.PAIR_INDEX[25]),
        )
        low_rank = optional.LowRankState(
            _dct_columns(2), np.asarray((0.08, -0.035))
        )
        state = optional.HybridState(
            low_rank, np.asarray((0.04, -0.025, 0.015))
        )
        sparse_lipschitz = 3.0
        evaluation = optional.evaluate_sparse_plus_low_rank(
            state,
            specification,
            sparse_lipschitz,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )

        embedded = np.zeros(core.PAIR_DIMENSION)
        embedded[list(specification.edge_indexes)] = state.sparse_coefficients
        expected_physical = optional.offdiagonal_vector(
            optional.low_rank_matrix(low_rank)
        ) + embedded
        expected_beta = core._range_project_vector(
            expected_physical, self.prepared.transform
        )
        np.testing.assert_allclose(
            evaluation.physical_coefficient,
            expected_physical,
            atol=2.0e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            evaluation.beta, expected_beta, atol=3.0e-15, rtol=0.0
        )
        expected_objective = evaluation.core_evaluation.objective
        expected_objective += specification.low_strength * 0.5 * float(
            np.dot(low_rank.d, low_rank.d)
        )
        expected_objective += specification.sparse_strength * float(
            np.sum(np.abs(state.sparse_coefficients))
        )
        self.assertLess(abs(evaluation.objective - expected_objective), 2.0e-15)
        projected_gradient = core._range_project_vector(
            evaluation.core_evaluation.gradient, self.prepared.transform
        )
        np.testing.assert_allclose(
            evaluation.gradient_a,
            projected_gradient[list(specification.edge_indexes)],
            atol=2.0e-15,
            rtol=0.0,
        )

        ambient = np.sin(np.arange(1, 33, dtype=float).reshape(16, 2) * 0.19)
        tangent = np.array(
            optional.stiefel_tangent_projection(low_rank.U, ambient), copy=True
        )
        tangent /= float(np.linalg.norm(tangent))
        d_direction = np.asarray((0.31, -0.27))
        a_direction = np.asarray((0.23, -0.17, 0.11))
        step = 1.0e-6

        def shifted_low(sign: float) -> optional.HybridState:
            U = optional.qr_retraction(low_rank.U, sign * tangent, step)
            shifted = optional.canonicalize_state(
                optional.LowRankState(
                    U, low_rank.d + sign * step * d_direction
                )
            )
            return optional.HybridState(shifted, state.sparse_coefficients)

        low_plus = optional.evaluate_sparse_plus_low_rank(
            shifted_low(1.0), specification, sparse_lipschitz,
            self.fixed_m1_theta, self.prepared, self.candidates, self.geometry, self.pairs
        )
        low_minus = optional.evaluate_sparse_plus_low_rank(
            shifted_low(-1.0), specification, sparse_lipschitz,
            self.fixed_m1_theta, self.prepared, self.candidates, self.geometry, self.pairs
        )
        low_numeric = (low_plus.objective - low_minus.objective) / (2.0 * step)
        low_analytic = float(np.sum(evaluation.gradient_U * tangent))
        low_analytic += float(np.dot(evaluation.gradient_d, d_direction))
        self.assertLess(abs(low_numeric - low_analytic), 1.0e-7)

        def shifted_sparse(sign: float) -> optional.HybridState:
            return optional.HybridState(
                low_rank,
                state.sparse_coefficients + sign * step * a_direction,
            )

        sparse_plus = optional.evaluate_sparse_plus_low_rank(
            shifted_sparse(1.0), specification, sparse_lipschitz,
            self.fixed_m1_theta, self.prepared, self.candidates, self.geometry, self.pairs
        )
        sparse_minus = optional.evaluate_sparse_plus_low_rank(
            shifted_sparse(-1.0), specification, sparse_lipschitz,
            self.fixed_m1_theta, self.prepared, self.candidates, self.geometry, self.pairs
        )
        smooth_numeric = (
            sparse_plus.smooth_objective - sparse_minus.smooth_objective
        ) / (2.0 * step)
        self.assertLess(
            abs(smooth_numeric - float(np.dot(evaluation.gradient_a, a_direction))),
            1.0e-7,
        )
        full_numeric = (sparse_plus.objective - sparse_minus.objective) / (
            2.0 * step
        )
        full_analytic = float(np.dot(evaluation.gradient_a, a_direction))
        full_analytic += specification.sparse_strength * float(
            np.dot(np.sign(state.sparse_coefficients), a_direction)
        )
        self.assertLess(abs(full_numeric - full_analytic), 1.0e-7)

        initial_L, projected_norm = optional._initial_sparse_lipschitz(
            specification,
            self.prepared.transform,
            self.fold_scales.smooth_lipschitz_upper_bound,
        )
        basis = self.prepared.transform.range_basis
        projected_embedding = basis @ basis[
            list(specification.edge_indexes), :
        ].T
        expected_norm = float(
            np.linalg.svd(projected_embedding, compute_uv=False)[0]
        )
        self.assertEqual(projected_norm, expected_norm)
        self.assertEqual(
            initial_L,
            max(
                1.0,
                self.fold_scales.smooth_lipschitz_upper_bound
                * expected_norm
                * expected_norm,
            ),
        )

    def test_hybrid_spec_requires_sorted_one_to_sixteen_named_edges(self) -> None:
        arguments = dict(
            rank=1,
            rho_low=1.0,
            rho_sparse=0.001,
            lambda_reference=self.fold_scales.lambda_reference,
            core_lasso_lambda_maximum=1.0,
        )
        with self.assertRaisesRegex(optional.LowRankError, "between 1 and 16"):
            optional.HybridSpec(**arguments, named_edges=())
        with self.assertRaisesRegex(optional.LowRankError, "between 1 and 16"):
            optional.HybridSpec(**arguments, named_edges=tuple(core.PAIR_INDEX[:17]))
        with self.assertRaisesRegex(optional.LowRankError, "lexicographic"):
            optional.HybridSpec(
                **arguments, named_edges=(core.PAIR_INDEX[1], core.PAIR_INDEX[0])
            )
        with self.assertRaisesRegex(optional.LowRankError, "lexicographic"):
            optional.HybridSpec(
                **arguments, named_edges=(core.PAIR_INDEX[0], core.PAIR_INDEX[0])
            )
        accepted = optional.HybridSpec(
            **arguments, named_edges=tuple(core.PAIR_INDEX[:16])
        )
        self.assertEqual(accepted.edge_indexes, tuple(range(16)))

    def test_zero_d_exactly_nests_core_M1_probabilities(self) -> None:
        state = optional.LowRankState(_dct_columns(2), np.zeros(2))
        specification = optional.LowRankSpec(
            2, 1.0, self.fold_scales.lambda_reference
        )
        low_rank = optional.evaluate_low_rank(
            state,
            specification,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            need_predictions=True,
        )
        nested = core.evaluate_m2(
            np.zeros(120),
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            need_predictions=True,
        )
        np.testing.assert_array_equal(low_rank.beta, np.zeros(120))
        np.testing.assert_array_equal(
            low_rank.core_evaluation.log_probability_nats,
            nested.log_probability_nats,
        )
        np.testing.assert_array_equal(
            low_rank.core_evaluation.marginals, nested.marginals
        )

    @staticmethod
    def _mock_evaluation(
        state: optional.LowRankState,
        objective: float,
        gradient_d: np.ndarray,
    ) -> optional.LowRankEvaluation:
        matrix = optional.low_rank_matrix(state)
        gradient_U = np.zeros_like(state.U)
        gradient_infinity = max(
            float(np.max(np.abs(gradient_U))),
            float(np.max(np.abs(gradient_d))),
        )
        return optional.LowRankEvaluation(
            objective=float(objective),
            smooth_objective=float(objective),
            penalty_value=0.5 * float(np.dot(state.d, state.d)),
            beta=np.zeros(120),
            physical_matrix=matrix,
            gradient_U=gradient_U,
            gradient_d=np.asarray(gradient_d, dtype=float),
            gradient_infinity=gradient_infinity,
            core_evaluation=core.M2Evaluation(
                float(objective), np.zeros(120), None, None, 0
            ),
        )

    def _mock_full_development_prepared(self) -> core.PreparedM2:
        cells = tuple(
            dataclasses.replace(
                self.prepared.cells[index % len(self.prepared.cells)],
                identifier=identifier,
                group=group,
            )
            for index, (identifier, group) in enumerate(
                controller.FROZEN_CELL_INVENTORY.entries
            )
        )
        indexes = np.arange(optional.EXPECTED_FULL_DEVELOPMENT_ROWS) % len(
            self.prepared.cells
        )
        data = self.prepared.m1_data
        return dataclasses.replace(
            self.prepared,
            m1_data=dataclasses.replace(
                data,
                cells=cells,
                x=np.asarray(data.x)[indexes].copy(),
                geometry_centers=np.asarray(data.geometry_centers)[indexes].copy(),
                weights=core.m1.equal_group_weights(
                    tuple(cell.group for cell in cells)
                ),
            ),
        )

    def _registered_synthetic_anchor(
        self,
        *,
        rank: int = 1,
        rho: float = 0.001,
        lambda_reference: float = 1.0,
        prepared: core.PreparedM2 | None = None,
    ) -> tuple[
        optional.LowRankOptimizerResult,
        optional.LowRankSpec,
        np.ndarray,
        optional.LowRankFoldScales,
    ]:
        active_prepared = self.prepared if prepared is None else prepared
        correction = np.zeros(core.PAIR_DIMENSION, dtype=float)
        scales = optional.LowRankFoldScales(
            lambda_reference,
            lambda_reference,
            1.0,
            np.zeros(core.PAIR_DIMENSION),
            np.zeros((core.m1.PANEL_SIZE, core.m1.PANEL_SIZE)),
        )
        specification = optional.LowRankSpec(rank, rho, lambda_reference)
        state = optional.LowRankState(_dct_columns(rank), np.zeros(rank))
        initialization = optional.LowRankInitialization(
            state,
            scales,
            1.0,
            np.zeros(rank),
        )
        stationary = self._mock_evaluation(state, 0.0, np.zeros(rank))
        with (
            mock.patch.object(
                optional,
                "fold_low_rank_scales_synthetic_qualification",
                return_value=scales,
            ),
            mock.patch.object(
                optional,
                "initialize_low_rank_synthetic_qualification",
                return_value=initialization,
            ),
            mock.patch.object(
                optional,
                "evaluate_low_rank_synthetic_qualification",
                return_value=stationary,
            ),
        ):
            result = optional.fit_low_rank_synthetic_qualification(
                self.fixed_m1_theta,
                active_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                linear_correction=correction,
                maximum_iterations=1,
            )
        return result, specification, correction, scales

    def _production_hybrid_fixture(self):
        core_results = tuple(
            controller.TrialResult(trial, 0.0, 0)
            for trial in controller.coverage_trials()
        )
        diagnostic = controller.OptionalDiagnostics(0.40, 16.0, 0.80, 9)
        contrasts = (1.0, *(0.0 for _ in range(core.PAIR_DIMENSION - 1)))
        evidence = controller.bind_coverage_evidence(
            core_results, diagnostic, contrasts
        )
        low_rank_trials = controller.coverage_trials(
            coverage_evidence=evidence
        )[controller.COVERAGE_TRIALS :]
        low_rank_results = tuple(
            controller.TrialResult(
                trial,
                1.0 if offset == 0 else 0.0,
                0,
            )
            for offset, trial in enumerate(low_rank_trials)
        )
        selected_configuration = low_rank_results[0].trial.configuration
        assert selected_configuration is not None
        selected_specification = optional.LowRankSpec(
            int(selected_configuration.rank),
            float(selected_configuration.rho),
            self.fold_scales.lambda_reference,
        )
        initialization = optional.initialize_low_rank(
            selected_specification.rank,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            fold_scales=self.fold_scales,
        )
        stationary = self._mock_evaluation(
            initialization.state,
            0.0,
            np.zeros(selected_specification.rank),
        )
        with (
            mock.patch.object(
                optional, "fold_low_rank_scales", return_value=self.fold_scales
            ),
            mock.patch.object(
                optional, "initialize_low_rank", return_value=initialization
            ),
            mock.patch.object(
                optional, "evaluate_low_rank", return_value=stationary
            ),
        ):
            selected_anchor = optional.fit_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                selected_specification,
                maximum_iterations=1,
            )
        complete_coverage = controller.coverage_trials(
            coverage_evidence=evidence,
            low_rank_anchor_results=low_rank_results,
        )
        target_configuration = complete_coverage[-1].configuration
        assert target_configuration is not None
        named_edges = tuple(
            core.PAIR_INDEX[index] for index in evidence.localized.edges
        )
        hybrid_specification = optional.HybridSpec(
            selected_specification.rank,
            float(target_configuration.rho),
            float(target_configuration.sparse_rho),
            self.fold_scales.lambda_reference,
            1.0,
            named_edges,
        )
        receipt = optional.authorize_production_hybrid(
            evidence,
            low_rank_results,
            selected_anchor,
            target_configuration,
            hybrid_specification,
        )
        return (
            evidence,
            low_rank_results,
            selected_anchor,
            target_configuration,
            hybrid_specification,
            receipt,
        )

    def test_exact_zero_gradient_uses_explicit_fixed_point_recheck(self) -> None:
        initial = optional.LowRankState(_dct_columns(1), np.zeros(1))
        specification = optional.LowRankSpec(1, 1.0, 1.0)
        scales = optional.LowRankFoldScales(
            1.0, 1.0, 1.0, np.zeros(120), np.zeros((16, 16))
        )
        initialization = optional.LowRankInitialization(
            initial, scales, 1.0, np.zeros(1)
        )

        def quadratic(state, *_args, **_kwargs):
            difference = np.asarray(state.d) - 1.0
            return self._mock_evaluation(
                state,
                0.5 * float(np.dot(difference, difference)),
                difference,
            )

        with (
            mock.patch.object(optional, "fold_low_rank_scales", return_value=scales),
            mock.patch.object(optional, "initialize_low_rank", return_value=initialization),
            mock.patch.object(
                optional, "evaluate_low_rank", side_effect=quadratic
            ) as evaluate,
        ):
            result = optional.fit_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                maximum_iterations=3,
            )
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.state.d[0], 1.0)
        self.assertEqual(result.gradient_infinity, 0.0)
        self.assertEqual(result.relative_objective_change, 0.0)
        self.assertEqual(result.telemetry["total_line_trials"], 1)
        self.assertTrue(result.telemetry["fixed_point_recheck"])
        self.assertEqual(result.telemetry["fixed_point_rechecks"], 1)
        self.assertEqual(evaluate.call_count, 3)

    def test_small_nonzero_gradient_takes_another_real_Armijo_step(self) -> None:
        initial = optional.LowRankState(_dct_columns(1), np.zeros(1))
        specification = optional.LowRankSpec(1, 1.0, 1.0)
        scales = optional.LowRankFoldScales(
            1.0, 1.0, 1.0, np.zeros(120), np.zeros((16, 16))
        )
        initialization = optional.LowRankInitialization(
            initial, scales, 1.0, np.zeros(1)
        )
        curvature = 1.0 - 0.5 * optional.LOW_RANK_GRADIENT_INFINITY_TOLERANCE

        def nearly_quadratic_stationary(state, *_args, **_kwargs):
            difference = np.asarray(state.d) - 1.0
            return self._mock_evaluation(
                state,
                0.5 * curvature * float(np.dot(difference, difference)),
                curvature * difference,
            )

        with (
            mock.patch.object(optional, "fold_low_rank_scales", return_value=scales),
            mock.patch.object(optional, "initialize_low_rank", return_value=initialization),
            mock.patch.object(
                optional,
                "evaluate_low_rank",
                side_effect=nearly_quadratic_stationary,
            ),
        ):
            result = optional.fit_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                maximum_iterations=3,
            )
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.telemetry["total_line_trials"], 2)
        self.assertFalse(result.telemetry["fixed_point_recheck"])
        self.assertEqual(result.telemetry["fixed_point_rechecks"], 0)
        self.assertGreater(result.relative_objective_change, 0.0)
        self.assertLessEqual(
            result.relative_objective_change,
            optional.LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE,
        )

    def test_Armijo_line_failure_is_fail_closed(self) -> None:
        initial = optional.LowRankState(_dct_columns(1), np.zeros(1))
        specification = optional.LowRankSpec(1, 1.0, 1.0)
        scales = optional.LowRankFoldScales(
            1.0, 1.0, 1.0, np.zeros(120), np.zeros((16, 16))
        )
        initialization = optional.LowRankInitialization(
            initial, scales, 1.0, np.zeros(1)
        )

        def uphill(state, *_args, **_kwargs):
            return self._mock_evaluation(
                state, -float(state.d[0]), np.ones(1, dtype=float)
            )

        with (
            mock.patch.object(optional, "fold_low_rank_scales", return_value=scales),
            mock.patch.object(optional, "initialize_low_rank", return_value=initialization),
            mock.patch.object(optional, "evaluate_low_rank", side_effect=uphill),
        ):
            with self.assertRaisesRegex(optional.LowRankError, "Armijo search failed"):
                optional.fit_low_rank(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    maximum_iterations=1,
                )

    def test_zero_corrected_null_repeats_exactly_and_uses_floor(self) -> None:
        null = core.evaluate_m2(
            np.zeros(120),
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        correction = -null.gradient
        scales = optional.fold_low_rank_scales_synthetic_qualification(
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            linear_correction=correction,
            smooth_lipschitz_upper_bound=self.fold_scales.smooth_lipschitz_upper_bound,
        )
        self.assertLess(scales.raw_gradient_spectral_norm, 1.0e-14)
        self.assertEqual(scales.lambda_reference, optional.LAMBDA_REFERENCE_FLOOR)
        specification = optional.LowRankSpec(1, 1.0, scales.lambda_reference)
        with mock.patch.object(
            optional,
            "fold_low_rank_scales_synthetic_qualification",
            return_value=scales,
        ) as recompute:
            first = optional.fit_low_rank_synthetic_qualification(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                linear_correction=correction,
                maximum_iterations=3,
            )
            second = optional.fit_low_rank_synthetic_qualification(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                linear_correction=correction,
                maximum_iterations=3,
            )
        self.assertEqual(recompute.call_count, 2)
        np.testing.assert_array_equal(first.state.U, second.state.U)
        np.testing.assert_array_equal(first.state.d, second.state.d)
        np.testing.assert_array_equal(first.beta, second.beta)
        self.assertEqual(first.objective, second.objective)
        self.assertEqual(first.telemetry, second.telemetry)
        self.assertEqual(first.telemetry["objective_mode"], "synthetic_qualification")
        self.assertEqual(first.iterations, 1)
        self.assertGreaterEqual(first.telemetry["total_line_trials"], 1)
        self.assertFalse(first.telemetry["fixed_point_recheck"])
        self.assertLessEqual(
            first.relative_objective_change,
            optional.LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE,
        )

    def test_public_fit_rejects_arbitrary_initial_wrong_scale_and_large_cap(self) -> None:
        state = optional.LowRankState(_dct_columns(1), np.zeros(1))
        specification = optional.LowRankSpec(
            1, 1.0, self.fold_scales.lambda_reference
        )
        with self.assertRaises(TypeError):
            optional.fit_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                initial=state,
            )
        with self.assertRaises(TypeError):
            optional.fit_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                linear_correction=np.zeros(120),
            )
        with self.assertRaises(TypeError):
            optional.fit_low_rank_synthetic_qualification(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
            )
        production_helper_arguments = (
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        with self.assertRaises(TypeError):
            optional.fold_low_rank_scales(
                *production_helper_arguments,
                linear_correction=np.zeros(120),
            )
        with self.assertRaises(TypeError):
            optional.low_rank_lambda_reference(
                *production_helper_arguments,
                linear_correction=np.zeros(120),
            )
        with self.assertRaises(TypeError):
            optional.initialize_low_rank(
                1,
                *production_helper_arguments,
                linear_correction=np.zeros(120),
            )
        with self.assertRaises(TypeError):
            optional.evaluate_low_rank(
                state,
                specification,
                *production_helper_arguments,
                linear_correction=np.zeros(120),
            )

        wrong_reference = np.nextafter(
            self.fold_scales.lambda_reference, math.inf
        )
        wrong = optional.LowRankSpec(1, 1.0, wrong_reference)
        with mock.patch.object(
            optional, "fold_low_rank_scales", return_value=self.fold_scales
        ) as recompute:
            with self.assertRaisesRegex(optional.LowRankError, "recomputed fold"):
                optional.fit_low_rank(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    wrong,
                    maximum_iterations=1,
                )
        recompute.assert_called_once()

        with mock.patch.object(optional, "fold_low_rank_scales") as recompute:
            with self.assertRaisesRegex(optional.LowRankError, "1..20000"):
                optional.fit_low_rank(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    maximum_iterations=20_001,
                )
        recompute.assert_not_called()

        for invalid_cap in (0, -1, True, 3.0):
            with self.subTest(invalid_cap=invalid_cap):
                with mock.patch.object(optional, "fold_low_rank_scales") as recompute:
                    with self.assertRaisesRegex(optional.LowRankError, "1..20000"):
                        optional.fit_low_rank(
                            self.fixed_m1_theta,
                            self.prepared,
                            self.candidates,
                            self.geometry,
                            self.pairs,
                            specification,
                            maximum_iterations=invalid_cap,
                        )
                recompute.assert_not_called()

        initialization = optional.initialize_low_rank(
            1,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            fold_scales=self.fold_scales,
        )
        stationary = self._mock_evaluation(
            initialization.state, 0.0, np.zeros(1)
        )
        with (
            mock.patch.object(
                optional, "fold_low_rank_scales", return_value=self.fold_scales
            ),
            mock.patch.object(
                optional, "initialize_low_rank", return_value=initialization
            ),
            mock.patch.object(
                optional, "evaluate_low_rank", return_value=stationary
            ),
        ):
            cap_boundary = optional.fit_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                maximum_iterations=20_000,
            )
        self.assertEqual(cap_boundary.iterations, 0)

    def test_unmocked_exact_likelihood_fit_moves_U_and_d_with_backtracking(self) -> None:
        specification = optional.LowRankSpec(
            1, 1.0, self.fold_scales.lambda_reference
        )
        initialization = optional.initialize_low_rank(
            1,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            fold_scales=self.fold_scales,
        )
        initial = optional.evaluate_low_rank(
            initialization.state,
            specification,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        result = optional.fit_low_rank(
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            specification,
            maximum_iterations=500,
        )
        repeated = optional.fit_low_rank(
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            specification,
            maximum_iterations=500,
        )
        self.assertGreater(result.iterations, 0)
        self.assertLessEqual(result.iterations, 500)
        self.assertLess(result.objective, initial.objective)
        self.assertLessEqual(
            result.gradient_infinity,
            optional.LOW_RANK_GRADIENT_INFINITY_TOLERANCE,
        )
        self.assertLessEqual(
            result.relative_objective_change,
            optional.LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE,
        )
        self.assertGreater(
            result.telemetry["total_line_trials"], result.iterations
        )
        self.assertTrue(result.telemetry["QR_retraction"])
        initial_projector = initialization.state.U @ initialization.state.U.T
        fitted_projector = result.state.U @ result.state.U.T
        self.assertGreater(
            float(np.max(np.abs(fitted_projector - initial_projector))), 1.0e-6
        )
        self.assertGreater(
            float(np.max(np.abs(result.state.d - initialization.state.d))),
            1.0e-6,
        )
        np.testing.assert_array_equal(result.state.U, repeated.state.U)
        np.testing.assert_array_equal(result.state.d, repeated.state.d)
        np.testing.assert_array_equal(result.beta, repeated.beta)
        self.assertEqual(result.objective, repeated.objective)
        self.assertEqual(result.telemetry, repeated.telemetry)
        self.assertEqual(result.telemetry["objective_mode"], "production")
        rebound = optional._validated_selected_low_rank_anchor(
            result,
            "production",
            self.fold_scales.lambda_reference,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            linear_correction=None,
        )
        np.testing.assert_array_equal(rebound.U, result.state.U)
        np.testing.assert_array_equal(rebound.d, result.state.d)
        with self.assertRaisesRegex(optional.LowRankError, "objective mode"):
            optional._validated_selected_low_rank_anchor(
                result,
                "synthetic_qualification",
                self.fold_scales.lambda_reference,
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                linear_correction=np.zeros(120),
            )

    def test_low_rank_warm_refit_requires_exact_registered_bound_result(self) -> None:
        full_prepared = self._mock_full_development_prepared()
        specification = optional.LowRankSpec(
            1, 1.0, self.fold_scales.lambda_reference
        )
        initialization = optional.initialize_low_rank(
            1,
            self.fixed_m1_theta,
            full_prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            fold_scales=self.fold_scales,
        )
        stationary = self._mock_evaluation(
            initialization.state, 0.0, np.zeros(1)
        )
        patches = (
            mock.patch.object(
                optional, "fold_low_rank_scales", return_value=self.fold_scales
            ),
            mock.patch.object(
                optional, "initialize_low_rank", return_value=initialization
            ),
            mock.patch.object(
                optional, "evaluate_low_rank", return_value=stationary
            ),
        )
        with patches[0], patches[1], patches[2]:
            cold = optional.fit_low_rank(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                maximum_iterations=1,
            )
            warm = optional.refit_low_rank_from_converged(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                cold,
                maximum_iterations=1,
            )
            np.testing.assert_array_equal(warm.state.U, cold.state.U)
            np.testing.assert_array_equal(warm.state.d, cold.state.d)
            self.assertEqual(
                warm.telemetry["initialization"],
                "exact_registered_converged_low_rank_result",
            )

            with self.assertRaisesRegex(optional.LowRankError, "solver process"):
                optional.refit_low_rank_from_converged(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    dataclasses.replace(cold),
                    maximum_iterations=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "different prepared"):
                optional.refit_low_rank_from_converged(
                    self.fixed_m1_theta.copy(),
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    cold,
                    maximum_iterations=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "different prepared"):
                optional.refit_low_rank_from_converged(
                    self.fixed_m1_theta,
                    dataclasses.replace(full_prepared),
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    cold,
                    maximum_iterations=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "not an outer fold"):
                optional.refit_low_rank_from_converged(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    cold,
                    maximum_iterations=1,
                )
            wrong_specification = optional.LowRankSpec(
                1, 0.3, self.fold_scales.lambda_reference
            )
            with self.assertRaisesRegex(optional.LowRankError, "specification"):
                optional.refit_low_rank_from_converged(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    wrong_specification,
                    cold,
                    maximum_iterations=1,
                )

        cold.telemetry["rho"] = 0.3
        with self.assertRaisesRegex(optional.LowRankError, "changed after fitting"):
            optional.refit_low_rank_from_converged(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                cold,
                maximum_iterations=1,
            )

    def test_synthetic_low_rank_warm_refit_seals_objective_mode_and_correction(
        self,
    ) -> None:
        full_prepared = self._mock_full_development_prepared()
        cold, specification, correction, scales = (
            self._registered_synthetic_anchor(prepared=full_prepared)
        )
        stationary = self._mock_evaluation(cold.state, 0.0, np.zeros(1))
        with (
            mock.patch.object(
                optional,
                "fold_low_rank_scales_synthetic_qualification",
                return_value=scales,
            ),
            mock.patch.object(
                optional,
                "evaluate_low_rank_synthetic_qualification",
                return_value=stationary,
            ),
        ):
            warm = optional.refit_low_rank_from_converged_synthetic_qualification(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                cold,
                linear_correction=correction,
                maximum_iterations=1,
            )
            self.assertEqual(
                warm.telemetry["initialization"],
                "exact_registered_converged_low_rank_result",
            )
            changed = correction.copy()
            changed[0] = np.nextafter(0.0, 1.0)
            with self.assertRaisesRegex(optional.LowRankError, "objective"):
                optional.refit_low_rank_from_converged_synthetic_qualification(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    cold,
                    linear_correction=changed,
                    maximum_iterations=1,
                )
        with self.assertRaisesRegex(optional.LowRankError, "objective mode"):
            optional.refit_low_rank_from_converged(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                cold,
                maximum_iterations=1,
            )

    def test_hybrid_block_order_carried_sparse_L_and_determinism(self) -> None:
        full_prepared = self._mock_full_development_prepared()
        selected_anchor, _, correction, _ = self._registered_synthetic_anchor(
            prepared=full_prepared
        )
        specification = optional.HybridSpec(
            rank=1,
            rho_low=0.001,
            rho_sparse=0.001,
            lambda_reference=1.0,
            core_lasso_lambda_maximum=1.0,
            named_edges=(core.PAIR_INDEX[0],),
        )
        anchor_state = optional.LowRankState(_dct_columns(1), np.zeros(1))
        scales = optional.LowRankFoldScales(
            1.0, 1.0, 1.0, np.zeros(120), np.zeros((16, 16))
        )
        calls: list[tuple[float, float, float]] = []

        def block_quadratic(
            state, active_specification, sparse_lipschitz, *_args, **_kwargs
        ):
            d_value = float(state.low_rank.d[0])
            a_value = float(state.sparse_coefficients[0])
            calls.append((d_value, a_value, float(sparse_lipschitz)))
            smooth = 0.5 * (d_value - 1.0) ** 2
            smooth += 2.0 * (a_value - 1.0) ** 2
            gradient_d = np.asarray((d_value - 1.0,))
            gradient_a = np.asarray((4.0 * (a_value - 1.0),))
            prox = optional._soft_threshold(
                state.sparse_coefficients - gradient_a / sparse_lipschitz,
                active_specification.sparse_strength / sparse_lipschitz,
            )
            mapping = sparse_lipschitz * (
                state.sparse_coefficients - prox
            )
            objective = smooth + active_specification.sparse_strength * abs(a_value)
            joint = max(
                abs(float(gradient_d[0])), abs(float(mapping[0]))
            )
            return optional.HybridEvaluation(
                objective=objective,
                smooth_objective=smooth,
                low_rank_penalty_value=0.5 * d_value * d_value,
                sparse_penalty_value=abs(a_value),
                beta=np.zeros(120),
                physical_coefficient=np.zeros(120),
                gradient_U=np.zeros_like(state.low_rank.U),
                gradient_d=gradient_d,
                gradient_a=gradient_a,
                sparse_stationarity_mapping=mapping,
                joint_stationarity_infinity=joint,
                core_evaluation=core.M2Evaluation(
                    smooth, np.zeros(120), None, None, 0
                ),
            )

        patches = (
            mock.patch.object(
                optional,
                "fold_low_rank_scales_synthetic_qualification",
                return_value=scales,
            ),
            mock.patch.object(
                core, "fold_lambda_maximum", return_value=mock.Mock(value=1.0)
            ),
            mock.patch.object(
                optional,
                "_validated_selected_low_rank_anchor",
                return_value=anchor_state,
            ),
            mock.patch.object(
                optional, "_initial_sparse_lipschitz", return_value=(1.0, 1.0)
            ),
            mock.patch.object(
                optional, "_evaluate_hybrid_impl", side_effect=block_quadratic
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            first = optional.fit_sparse_plus_low_rank_synthetic_qualification(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                linear_correction=correction,
                maximum_sweeps=5,
            )
            second = optional.fit_sparse_plus_low_rank_synthetic_qualification(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                linear_correction=correction,
                maximum_sweeps=5,
            )
            warm = (
                optional.refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    first,
                    linear_correction=correction,
                    maximum_sweeps=5,
                )
            )
            with self.assertRaisesRegex(optional.LowRankError, "solver process"):
                optional.refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    dataclasses.replace(first),
                    linear_correction=correction,
                    maximum_sweeps=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "different prepared"):
                optional.refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
                    self.fixed_m1_theta,
                    dataclasses.replace(full_prepared),
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    first,
                    linear_correction=correction,
                    maximum_sweeps=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "not an outer fold"):
                optional.refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    first,
                    linear_correction=correction,
                    maximum_sweeps=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "registered converged"):
                optional.refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    warm,
                    linear_correction=correction,
                    maximum_sweeps=1,
                )

        self.assertEqual(first.status, "pass")
        self.assertEqual(first.sweeps, 2)
        self.assertEqual(first.state.low_rank.d[0], 1.0)
        self.assertLess(abs(first.state.sparse_coefficients[0] - 0.99975), 1.0e-15)
        self.assertEqual(first.sparse_lipschitz, 4.0)
        self.assertEqual(first.telemetry["block_order"], ("low_rank", "sparse"))
        self.assertEqual(first.telemetry["low_rank_line_trials"], 1)
        self.assertEqual(first.telemetry["sparse_line_trials"], 4)
        self.assertEqual(first.telemetry["sparse_backtracking_steps"], 2)
        self.assertEqual(first.telemetry["post_sweep_evaluations"], 2)
        self.assertLessEqual(
            first.joint_stationarity_infinity,
            optional.HYBRID_JOINT_STATIONARITY_TOLERANCE,
        )
        self.assertLessEqual(
            first.relative_objective_change,
            optional.HYBRID_RELATIVE_OBJECTIVE_TOLERANCE,
        )
        np.testing.assert_array_equal(first.state.low_rank.U, second.state.low_rank.U)
        np.testing.assert_array_equal(first.state.low_rank.d, second.state.low_rank.d)
        np.testing.assert_array_equal(
            first.state.sparse_coefficients, second.state.sparse_coefficients
        )
        self.assertEqual(first.objective, second.objective)
        self.assertEqual(first.telemetry, second.telemetry)
        np.testing.assert_array_equal(warm.state.low_rank.U, first.state.low_rank.U)
        np.testing.assert_array_equal(warm.state.low_rank.d, first.state.low_rank.d)
        np.testing.assert_array_equal(
            warm.state.sparse_coefficients, first.state.sparse_coefficients
        )
        self.assertEqual(warm.telemetry["initial_sparse_lipschitz"], 4.0)
        self.assertEqual(warm.sparse_lipschitz, 4.0)
        self.assertEqual(
            warm.telemetry["initialization"],
            "exact_registered_converged_hybrid_result",
        )
        self.assertTrue(any(d == 1.0 and a == 0.0 for d, a, _ in calls))
        self.assertTrue(any(d == 1.0 and a > 0.0 for d, a, _ in calls))

    def test_hybrid_sparse_backtracking_failure_is_fail_closed(self) -> None:
        specification = optional.HybridSpec(
            1, 0.001, 0.001, 1.0, 1.0, (core.PAIR_INDEX[0],)
        )
        anchor_state = optional.LowRankState(_dct_columns(1), np.zeros(1))
        scales = optional.LowRankFoldScales(
            1.0, 1.0, 1.0, np.zeros(120), np.zeros((16, 16))
        )

        def never_majorizes(
            state, active_specification, sparse_lipschitz, *_args, **_kwargs
        ):
            a_value = float(state.sparse_coefficients[0])
            smooth = 0.0 if a_value == 0.0 else 100.0
            gradient_a = np.asarray((-1.0,))
            prox = optional._soft_threshold(
                state.sparse_coefficients - gradient_a / sparse_lipschitz,
                active_specification.sparse_strength / sparse_lipschitz,
            )
            mapping = sparse_lipschitz * (
                state.sparse_coefficients - prox
            )
            return optional.HybridEvaluation(
                objective=smooth
                + active_specification.sparse_strength * abs(a_value),
                smooth_objective=smooth,
                low_rank_penalty_value=0.0,
                sparse_penalty_value=abs(a_value),
                beta=np.zeros(120),
                physical_coefficient=np.zeros(120),
                gradient_U=np.zeros_like(state.low_rank.U),
                gradient_d=np.zeros(1),
                gradient_a=gradient_a,
                sparse_stationarity_mapping=mapping,
                joint_stationarity_infinity=abs(float(mapping[0])),
                core_evaluation=core.M2Evaluation(
                    smooth, np.zeros(120), None, None, 0
                ),
            )

        with (
            mock.patch.object(
                optional,
                "fold_low_rank_scales_synthetic_qualification",
                return_value=scales,
            ),
            mock.patch.object(
                core, "fold_lambda_maximum", return_value=mock.Mock(value=1.0)
            ),
            mock.patch.object(
                optional,
                "_validated_selected_low_rank_anchor",
                return_value=anchor_state,
            ),
            mock.patch.object(
                optional, "_initial_sparse_lipschitz", return_value=(1.0, 1.0)
            ),
            mock.patch.object(
                optional, "_evaluate_hybrid_impl", side_effect=never_majorizes
            ),
        ):
            with self.assertRaisesRegex(optional.LowRankError, "sparse backtracking"):
                optional.fit_sparse_plus_low_rank_synthetic_qualification(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    mock.sentinel.selected_anchor,
                    linear_correction=np.zeros(120),
                    maximum_sweeps=1,
                )

    def test_hybrid_public_apis_reject_arbitrary_initialization_and_mode_mixing(self) -> None:
        specification = optional.HybridSpec(
            1, 0.001, 0.001, 1.0, 1.0, (core.PAIR_INDEX[0],)
        )
        arguments = (
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            specification,
            mock.sentinel.selected_anchor,
        )
        with self.assertRaises(TypeError):
            optional.fit_sparse_plus_low_rank(
                *arguments,
                initial=optional.HybridState(
                    optional.LowRankState(_dct_columns(1), np.zeros(1)),
                    np.zeros(1),
                ),
            )
        with self.assertRaises(TypeError):
            optional.fit_sparse_plus_low_rank(
                *arguments, linear_correction=np.zeros(120)
            )
        with self.assertRaises(TypeError):
            optional.fit_sparse_plus_low_rank_synthetic_qualification(*arguments)
        for invalid_cap in (0, -1, True, 3.0, 20_001):
            with self.subTest(invalid_cap=invalid_cap):
                with self.assertRaisesRegex(optional.LowRankError, "1..20000"):
                    optional.fit_sparse_plus_low_rank_synthetic_qualification(
                        *arguments,
                        linear_correction=np.zeros(120),
                        maximum_sweeps=invalid_cap,
                    )

    def test_production_hybrid_warm_refit_carries_full_state_L_and_receipt(self) -> None:
        (
            evidence,
            low_rank_results,
            selected_anchor,
            target_configuration,
            specification,
            receipt,
        ) = self._production_hybrid_fixture()
        full_prepared = self._mock_full_development_prepared()

        def stationary_hybrid(
            state, active_specification, sparse_lipschitz, *_args, **_kwargs
        ):
            beta = optional.hybrid_effective_beta(
                state, active_specification, full_prepared.transform
            )
            return optional.HybridEvaluation(
                objective=1.0,
                smooth_objective=1.0,
                low_rank_penalty_value=0.5
                * float(np.dot(state.low_rank.d, state.low_rank.d)),
                sparse_penalty_value=float(
                    np.sum(np.abs(state.sparse_coefficients))
                ),
                beta=beta,
                physical_coefficient=np.zeros(core.PAIR_DIMENSION),
                gradient_U=np.zeros_like(state.low_rank.U),
                gradient_d=np.zeros_like(state.low_rank.d),
                gradient_a=np.zeros_like(state.sparse_coefficients),
                sparse_stationarity_mapping=np.zeros_like(
                    state.sparse_coefficients
                ),
                joint_stationarity_infinity=0.0,
                core_evaluation=core.M2Evaluation(
                    1.0, np.zeros(core.PAIR_DIMENSION), None, None, 0
                ),
            )

        with (
            mock.patch.object(
                optional, "fold_low_rank_scales", return_value=self.fold_scales
            ),
            mock.patch.object(
                core, "fold_lambda_maximum", return_value=mock.Mock(value=1.0)
            ),
            mock.patch.object(
                optional,
                "_validated_selected_low_rank_anchor",
                return_value=selected_anchor.state,
            ),
            mock.patch.object(
                optional, "_initial_sparse_lipschitz", return_value=(2.0, 1.0)
            ),
            mock.patch.object(
                optional, "_evaluate_hybrid_impl", side_effect=stationary_hybrid
            ),
        ):
            cold = optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )
            warm = optional.refit_sparse_plus_low_rank_from_converged(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                cold,
                production_receipt=receipt,
                maximum_sweeps=1,
            )
            np.testing.assert_array_equal(warm.state.low_rank.U, cold.state.low_rank.U)
            np.testing.assert_array_equal(warm.state.low_rank.d, cold.state.low_rank.d)
            np.testing.assert_array_equal(
                warm.state.sparse_coefficients, cold.state.sparse_coefficients
            )
            self.assertEqual(warm.telemetry["initial_sparse_lipschitz"], 2.0)
            self.assertEqual(warm.sparse_lipschitz, 2.0)
            self.assertEqual(
                warm.telemetry["initialization"],
                "exact_registered_converged_hybrid_result",
            )

            alternate_receipt = optional.authorize_production_hybrid(
                evidence,
                low_rank_results,
                selected_anchor,
                target_configuration,
                specification,
            )
            with self.assertRaisesRegex(optional.LowRankError, "different production"):
                optional.refit_sparse_plus_low_rank_from_converged(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    cold,
                    production_receipt=alternate_receipt,
                    maximum_sweeps=1,
                )
            with self.assertRaisesRegex(optional.LowRankError, "solver process"):
                optional.refit_sparse_plus_low_rank_from_converged(
                    self.fixed_m1_theta,
                    full_prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    selected_anchor,
                    dataclasses.replace(cold),
                    production_receipt=receipt,
                    maximum_sweeps=1,
                )

        object.__setattr__(cold, "sparse_lipschitz", 4.0)
        with self.assertRaisesRegex(optional.LowRankError, "changed after fitting"):
            optional.refit_sparse_plus_low_rank_from_converged(
                self.fixed_m1_theta,
                full_prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                cold,
                production_receipt=receipt,
                maximum_sweeps=1,
            )

    def test_production_hybrid_receipt_binds_winner_edges_and_object_identity(
        self,
    ) -> None:
        (
            evidence,
            low_rank_results,
            selected_anchor,
            target_configuration,
            specification,
            receipt,
        ) = self._production_hybrid_fixture()
        expected = mock.sentinel.production_hybrid_result
        with mock.patch.object(
            optional, "_fit_sparse_plus_low_rank_impl", return_value=expected
        ) as implementation:
            actual = optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=7,
            )
        self.assertIs(actual, expected)
        implementation.assert_called_once()

        wrong_edges = optional.HybridSpec(
            specification.rank,
            specification.rho_low,
            specification.rho_sparse,
            specification.lambda_reference,
            specification.core_lasso_lambda_maximum,
            (core.PAIR_INDEX[1],),
        )
        with self.assertRaisesRegex(
            optional.LowRankError, "target and initializer|objects are unavailable"
        ):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                wrong_edges,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )

        equivalent_specification = dataclasses.replace(specification)
        with self.assertRaisesRegex(optional.LowRankError, "objects are unavailable"):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                equivalent_specification,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )

        replaced_anchor = dataclasses.replace(selected_anchor)
        with self.assertRaisesRegex(optional.LowRankError, "unavailable"):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                replaced_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )
        with self.assertRaisesRegex(optional.LowRankError, "solver process"):
            optional.authorize_production_hybrid(
                evidence,
                low_rank_results,
                replaced_anchor,
                target_configuration,
                specification,
            )

        with self.assertRaises(TypeError):
            dataclasses.replace(receipt)
        forged_receipt = object.__new__(type(receipt))
        with self.assertRaisesRegex(optional.LowRankError, "receipt is invalid"):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                production_receipt=forged_receipt,
                maximum_sweeps=1,
            )

        original_target_rho = target_configuration.rho
        object.__setattr__(target_configuration, "rho", 0.3)
        with self.assertRaisesRegex(optional.LowRankError, "changed after authorization"):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )
        object.__setattr__(target_configuration, "rho", original_target_rho)

        original_specification_rho = specification.rho_low
        object.__setattr__(specification, "rho_low", 0.3)
        with self.assertRaisesRegex(optional.LowRankError, "changed after authorization"):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )
        object.__setattr__(specification, "rho_low", original_specification_rho)

        selected_anchor.telemetry["rho"] = 1.0
        with self.assertRaisesRegex(optional.LowRankError, "changed after fitting"):
            optional.fit_sparse_plus_low_rank(
                self.fixed_m1_theta,
                self.prepared,
                self.candidates,
                self.geometry,
                self.pairs,
                specification,
                selected_anchor,
                production_receipt=receipt,
                maximum_sweeps=1,
            )

    def test_v3_hybrid_initializer_derivation_covers_every_rank_pair(self) -> None:
        core_results = tuple(
            controller.TrialResult(trial, 0.0, 0)
            for trial in controller.coverage_trials()
        )
        evidence = controller.bind_coverage_evidence(
            core_results,
            controller.OptionalDiagnostics(0.40, 16.0, 0.80, 9),
            (1.0, *(0.0 for _ in range(core.PAIR_DIMENSION - 1))),
        )
        low_rank_trials = controller.coverage_trials(
            coverage_evidence=evidence
        )[controller.COVERAGE_TRIALS :]
        self.assertEqual(len(low_rank_trials), 8)

        # The first six winners exercise every r_star.  The two boundary-only
        # nodes additionally prove that an r_star rho of 1 or .001 is preserved
        # exactly, while a different target rank still uses only rho .1.
        for winner_offset in range(8):
            low_rank_results = tuple(
                controller.TrialResult(
                    trial,
                    1.0 if offset == winner_offset else 0.0,
                    0,
                )
                for offset, trial in enumerate(low_rank_trials)
            )
            selected = low_rank_results[winner_offset].trial.configuration
            assert selected is not None
            for target_rank in optional.LOW_RANKS:
                target = controller.Configuration(
                    "sparse_plus_low_rank",
                    0.3,
                    rank=target_rank,
                    sparse_rho=0.03,
                    sparse_edges=evidence.localized.edges,
                )
                derived = optional.derive_hybrid_low_rank_initializer_configuration(
                    evidence,
                    low_rank_results,
                    target,
                )
                if target_rank == selected.rank:
                    self.assertIs(derived, selected)
                else:
                    prescribed = next(
                        result.trial.configuration
                        for result in low_rank_results
                        if result.trial.configuration is not None
                        and result.trial.configuration.rank == target_rank
                        and result.trial.configuration.rho == 0.1
                    )
                    self.assertIs(derived, prescribed)

            with self.assertRaisesRegex(optional.LowRankError, "controller rejected"):
                optional.derive_hybrid_low_rank_initializer_configuration(
                    evidence,
                    tuple(reversed(low_rank_results)),
                    controller.Configuration(
                        "sparse_plus_low_rank",
                        0.3,
                        rank=selected.rank,
                        sparse_rho=0.03,
                        sparse_edges=evidence.localized.edges,
                    ),
                )

        original_contrasts = evidence.equal_group_pair_contrasts
        object.__setattr__(
            evidence,
            "equal_group_pair_contrasts",
            (0.5, *original_contrasts[1:]),
        )
        with self.assertRaisesRegex(optional.LowRankError, "changed after issuance"):
            optional.derive_hybrid_low_rank_initializer_configuration(
                evidence,
                low_rank_results,
                controller.Configuration(
                    "sparse_plus_low_rank",
                    0.3,
                    rank=1,
                    sparse_rho=0.03,
                    sparse_edges=evidence.localized.edges,
                ),
            )
        object.__setattr__(
            evidence,
            "equal_group_pair_contrasts",
            original_contrasts,
        )

    def test_v3_rank_change_and_successor_production_issuance_stay_closed(self) -> None:
        (
            evidence,
            low_rank_results,
            selected_anchor,
            target_configuration,
            specification,
            _receipt,
        ) = self._production_hybrid_fixture()
        named_edges = tuple(
            core.PAIR_INDEX[index] for index in evidence.localized.edges
        )

        rank_change_target = controller.Configuration(
            "sparse_plus_low_rank",
            1.0,
            rank=2,
            sparse_rho=1.0,
            sparse_edges=evidence.localized.edges,
        )
        rank_change_specification = optional.HybridSpec(
            2,
            1.0,
            1.0,
            specification.lambda_reference,
            specification.core_lasso_lambda_maximum,
            named_edges,
        )
        with self.assertRaisesRegex(optional.LowRankError, "named session adapter"):
            optional.authorize_production_hybrid(
                evidence,
                low_rank_results,
                selected_anchor,
                rank_change_target,
                rank_change_specification,
            )

        successor_target = controller.Configuration(
            "sparse_plus_low_rank",
            0.3,
            rank=target_configuration.rank,
            sparse_rho=0.3,
            sparse_edges=evidence.localized.edges,
        )
        successor_specification = optional.HybridSpec(
            int(successor_target.rank),
            0.3,
            0.3,
            specification.lambda_reference,
            specification.core_lasso_lambda_maximum,
            named_edges,
        )
        with self.assertRaisesRegex(optional.LowRankError, "named session adapter"):
            optional.authorize_production_hybrid(
                evidence,
                low_rank_results,
                selected_anchor,
                successor_target,
                successor_specification,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
