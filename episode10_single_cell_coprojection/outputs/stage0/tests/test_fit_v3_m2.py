#!/usr/bin/env python3
"""Target-free mathematical tests for the EP10 v3 M2 core engine."""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402


class M2CoreTests(unittest.TestCase):
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
        ) = m2.synthetic_k2_problem(rows=24)

    def test_all_k_pair_enumeration_and_choose_k_two(self) -> None:
        counts = self.pairs.all_counts()
        self.assertEqual(counts, {k: math.comb(16, k) for k in range(17)})
        self.assertEqual(sum(counts.values()), 65_536)
        expected_pairs = tuple(
            (left, right)
            for left in range(16)
            for right in range(left + 1, 16)
        )
        self.assertEqual(m2.PAIR_INDEX, expected_pairs)
        for k in range(17):
            pair_basis = self.pairs.basis(k)
            self.assertEqual(pair_basis.indicators.shape, (math.comb(16, k), 120))
            np.testing.assert_array_equal(
                np.sum(pair_basis.indicators, axis=1, dtype=int),
                np.full(math.comb(16, k), math.comb(k, 2), dtype=int),
            )

    def test_bilateral_anatomical_group_inventory(self) -> None:
        self.assertEqual(len(m2.BILATERAL_GROUPS), 36)
        sizes = [len(indexes) for _, indexes in m2.BILATERAL_GROUPS]
        self.assertEqual(sizes[:28], [4] * 28)
        self.assertEqual(sizes[28:], [1] * 8)
        flattened = [
            pair_index
            for _, indexes in m2.BILATERAL_GROUPS
            for pair_index in indexes
        ]
        self.assertEqual(sorted(flattened), list(range(120)))
        expected_bases = (
            "allen_993_MOs",
            "allen_453_SS",
            "allen_31_ACA",
            "allen_477_STR",
            "allen_549_TH",
            "allen_313_MB",
            "allen_771_P",
            "allen_354_MY",
        )
        self.assertEqual(m2.ANATOMICAL_BASES, expected_bases)

    def test_uniform_projection_orthogonality_range_and_validation_isolation(self) -> None:
        transform = self.prepared.transform
        diagnostics = transform.diagnostics
        self.assertLessEqual(
            diagnostics["G_FH_projection_maximum_absolute_orthogonality_error"],
            m2.PROJECTION_TOLERANCE,
        )
        self.assertEqual(m2.PROJECTION_TOLERANCE, 1.0e-10)
        self.assertEqual(m2.PROJECTOR_TOLERANCE, 1.0e-12)
        self.assertTrue(diagnostics["rank_stable"])
        self.assertEqual(
            len(set(diagnostics["rank_at_tau_over_10_tau_ten_tau"].values())), 1
        )
        projector = transform.range_projector
        np.testing.assert_allclose(projector, projector.T, atol=2.0e-13, rtol=0.0)
        np.testing.assert_allclose(
            projector @ projector, projector, atol=2.0e-12, rtol=0.0
        )
        self.assertLessEqual(
            diagnostics["range_projector_symmetry_maximum_absolute_error"],
            m2.PROJECTOR_TOLERANCE,
        )
        self.assertLessEqual(
            diagnostics["range_projector_idempotence_maximum_absolute_error"],
            m2.PROJECTOR_TOLERANCE,
        )
        null_gram = (np.eye(120) - projector) @ transform.scaled_residual_gram
        self.assertLess(float(np.max(np.abs(null_gram))), 2.0e-9)

        design_mean_before = transform.design_transform.soma_mean.copy()
        geometry_scale_before = transform.geometry_scales.copy()
        projection_before = transform.projection_f_to_h.copy()
        validation_cells = tuple(
            dataclasses.replace(
                cell,
                identifier=f"validation_{index}",
                soma_um=tuple(value + 100_000.0 for value in cell.soma_um),
            )
            for index, cell in enumerate(self.cells[:4])
        )
        validation_moments = tuple(
            self.geometry.uniform_moments(cell) for cell in validation_cells
        )
        first = m2.apply_residual_transform(
            validation_cells, validation_moments, transform
        )
        second = m2.apply_residual_transform(
            validation_cells, validation_moments, transform
        )
        self.assertIs(first.transform, transform)
        self.assertIs(second.transform, transform)
        np.testing.assert_array_equal(
            transform.design_transform.soma_mean, design_mean_before
        )
        np.testing.assert_array_equal(transform.geometry_scales, geometry_scale_before)
        np.testing.assert_array_equal(transform.projection_f_to_h, projection_before)
        np.testing.assert_array_equal(first.m1_data.x, second.m1_data.x)

    def test_beta_zero_exact_nesting_probability_and_marginal_identities(self) -> None:
        zero = np.zeros(120, dtype=float)
        m2_evaluation = m2.evaluate_m2(
            zero,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
            need_predictions=True,
        )
        m1_evaluation = m1.evaluate_model(
            self.fixed_m1_theta,
            self.prepared.m1_data,
            self.candidates,
            self.geometry,
            0.0,
            need_hessian=False,
            need_predictions=True,
        )
        assert m2_evaluation.log_probability_nats is not None
        assert m2_evaluation.marginals is not None
        assert m1_evaluation.log_probability_nats is not None
        assert m1_evaluation.marginals is not None
        np.testing.assert_array_equal(
            m2_evaluation.log_probability_nats,
            m1_evaluation.log_probability_nats,
        )
        np.testing.assert_array_equal(m2_evaluation.marginals, m1_evaluation.marginals)

        for cell_index, cell in enumerate(self.prepared.cells):
            distribution = m2.candidate_distribution(
                zero,
                self.fixed_m1_theta,
                self.prepared,
                cell_index,
                self.candidates,
                self.geometry,
                self.pairs,
            )
            self.assertAlmostEqual(float(np.sum(distribution.probabilities)), 1.0, 13)
            self.assertAlmostEqual(float(np.sum(distribution.marginals)), cell.k, 12)

        degenerate_cells = (
            dataclasses.replace(
                self.cells[0], identifier="degenerate_K0", k=0, observed_index=0
            ),
            dataclasses.replace(
                self.cells[1], identifier="degenerate_K16", k=16, observed_index=0
            ),
        )
        degenerate_moments = tuple(
            self.geometry.uniform_moments(cell) for cell in degenerate_cells
        )
        degenerate = m2.apply_residual_transform(
            degenerate_cells, degenerate_moments, self.prepared.transform
        )
        for cell_index, expected_k in enumerate((0, 16)):
            distribution = m2.candidate_distribution(
                zero,
                self.fixed_m1_theta,
                degenerate,
                cell_index,
                self.candidates,
                self.geometry,
                self.pairs,
            )
            np.testing.assert_array_equal(distribution.probabilities, np.ones(1))
            np.testing.assert_array_equal(distribution.log_probabilities, np.zeros(1))
            self.assertEqual(float(np.sum(distribution.marginals)), float(expected_k))
        degenerate_evaluation = m2.evaluate_m2(
            zero,
            self.fixed_m1_theta,
            degenerate,
            self.candidates,
            self.geometry,
            self.pairs,
            need_predictions=True,
        )
        self.assertEqual(degenerate_evaluation.objective, 0.0)
        np.testing.assert_array_equal(degenerate_evaluation.gradient, np.zeros(120))

    def test_exact_analytic_gradient_matches_finite_difference(self) -> None:
        raw = np.sin(np.arange(1, 121, dtype=float) * 0.31)
        direction = m2._range_project_vector(raw, self.prepared.transform)
        direction /= float(np.linalg.norm(direction))
        beta = 0.04 * m2._range_project_vector(
            np.cos(np.arange(1, 121, dtype=float) * 0.17),
            self.prepared.transform,
        )
        evaluation = m2.evaluate_m2(
            beta,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        step = 1.0e-6
        plus = m2.evaluate_m2(
            beta + step * direction,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        minus = m2.evaluate_m2(
            beta - step * direction,
            self.fixed_m1_theta,
            self.prepared,
            self.candidates,
            self.geometry,
            self.pairs,
        )
        observed = (plus.objective - minus.objective) / (2.0 * step)
        expected = float(np.dot(evaluation.gradient, direction))
        self.assertLess(abs(observed - expected), 2.0e-7)

    def test_closed_form_prox_and_range_dykstra_certification(self) -> None:
        identity = np.eye(120)
        full_transform = dataclasses.replace(
            self.prepared.transform,
            range_basis=identity,
            range_projector=identity,
            scaled_residual_gram=identity,
        )
        vector = np.linspace(-1.2, 1.1, 120)
        amount = 0.23

        lasso = m2.PenaltySpec("lasso", 1.0)
        lasso_result = m2.prox_penalty_range(
            vector, amount, lasso, full_transform
        )
        np.testing.assert_allclose(
            lasso_result.vector,
            np.sign(vector) * np.maximum(np.abs(vector) - amount, 0.0),
            atol=0.0,
            rtol=0.0,
        )

        elastic = m2.PenaltySpec("elastic_net", 1.0, 0.5)
        elastic_result = m2.prox_penalty_range(
            vector, amount, elastic, full_transform
        )
        expected_elastic = (
            np.sign(vector)
            * np.maximum(np.abs(vector) - 0.5 * amount, 0.0)
            / (1.0 + 0.5 * amount)
        )
        np.testing.assert_allclose(
            elastic_result.vector, expected_elastic, atol=0.0, rtol=0.0
        )

        group = m2.PenaltySpec("group_lasso", 1.0)
        group_result = m2.prox_penalty_range(
            vector, amount, group, full_transform
        )
        expected_group = np.zeros(120)
        for _, indexes in m2.BILATERAL_GROUPS:
            index = np.asarray(indexes, dtype=int)
            block = vector[index]
            norm = float(np.linalg.norm(block))
            threshold = amount * math.sqrt(len(indexes))
            if norm > threshold:
                expected_group[index] = (1.0 - threshold / norm) * block
        np.testing.assert_allclose(
            group_result.vector, expected_group, atol=2.0e-16, rtol=0.0
        )

        constrained = m2.prox_penalty_range(
            vector, amount, lasso, self.prepared.transform
        )
        self.assertTrue(constrained.certified)
        self.assertEqual(m2.PROX_TOLERANCE, 1.0e-12)
        self.assertLessEqual(constrained.fixed_point_residual, m2.PROX_TOLERANCE)
        self.assertLessEqual(constrained.range_residual, m2.PROX_TOLERANCE)
        self.assertLessEqual(constrained.invariant_residual, m2.PROX_TOLERANCE)
        np.testing.assert_allclose(
            constrained.vector,
            self.prepared.transform.range_projector @ constrained.vector,
            atol=m2.PROX_TOLERANCE,
            rtol=0.0,
        )

        first, warm_state, first_warm_started = m2._prox_penalty_range_with_state(
            vector, amount, lasso, self.prepared.transform
        )
        self.assertFalse(first_warm_started)
        self.assertIsNotNone(warm_state)
        changed_vector = vector + 0.01 * np.sin(np.arange(120, dtype=float))
        warmed, _, second_warm_started = m2._prox_penalty_range_with_state(
            changed_vector,
            amount,
            lasso,
            self.prepared.transform,
            warm_state=warm_state,
        )
        cold = m2.prox_penalty_range(
            changed_vector, amount, lasso, self.prepared.transform
        )
        self.assertTrue(second_warm_started)
        self.assertTrue(warmed.certified)
        self.assertTrue(cold.certified)
        for result in (warmed, cold):
            self.assertLessEqual(result.fixed_point_residual, m2.PROX_TOLERANCE)
            self.assertLessEqual(result.range_residual, m2.PROX_TOLERANCE)
            self.assertLessEqual(result.invariant_residual, m2.PROX_TOLERANCE)
        np.testing.assert_allclose(
            warmed.vector, cold.vector, atol=2.0e-11, rtol=0.0
        )

    def test_lambda_maximum_exact_zero_and_tiny_nonzero_gradient(self) -> None:
        zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        tiny = zero.copy()
        tiny[0] = 5.0e-14

        def evaluation(gradient: np.ndarray) -> m2.M2Evaluation:
            return m2.M2Evaluation(0.0, gradient, None, None, 0)

        with mock.patch.object(m2, "evaluate_m2", return_value=evaluation(zero)):
            with mock.patch.object(m2, "zero_kkt_certificate") as certificate:
                maximum = m2.fold_lambda_maximum(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    "lasso",
                )
        self.assertEqual(maximum, m2.LambdaMaximumResult(0.0, 0.0, 0.0, 0.0, 0))
        certificate.assert_not_called()

        def exact_scalar_certificate(
            gradient: np.ndarray,
            specification: m2.PenaltySpec,
            transform: m2.ResidualTransform,
            **kwargs,
        ) -> m2.ZeroKKTCertificate:
            del transform
            residual = max(
                float(np.linalg.norm(gradient, ord=np.inf))
                - float(specification.strength),
                0.0,
            )
            decision_tolerance = float(
                kwargs.get("decision_tolerance", m2.PROX_TOLERANCE)
            )
            feasible = residual <= decision_tolerance
            initial_correction = kwargs.get("initial_correction")
            if initial_correction is None:
                correction = (0.0,) * m2.PAIR_DIMENSION
            else:
                correction = tuple(float(value) for value in initial_correction)
            return m2.ZeroKKTCertificate(
                residual,
                0.0,
                0,
                True,
                feasible,
                0.0 if feasible else residual - decision_tolerance,
                correction,
            )

        with mock.patch.object(m2, "evaluate_m2", return_value=evaluation(tiny)):
            with mock.patch.object(
                m2,
                "zero_kkt_certificate",
                side_effect=exact_scalar_certificate,
            ):
                maximum = m2.fold_lambda_maximum(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    "lasso",
                )
        self.assertGreater(maximum.value, 0.0)
        self.assertLess(abs(maximum.value - 5.0e-14), 1.0e-22)
        self.assertLessEqual(
            maximum.upper_bracket - maximum.lower_bracket,
            m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE
            * maximum.upper_bracket,
        )

    def test_lambda_maximum_subdivides_indeterminate_midpoints(self) -> None:
        gradient = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        gradient[0] = 1.0
        transition = 0.6
        indeterminate_strengths = {0.25, 0.5}
        calls: list[
            tuple[
                float,
                tuple[float, ...] | None,
                str,
                tuple[float, ...] | None,
            ]
        ] = []

        def evaluation(*args, **kwargs) -> m2.M2Evaluation:
            del args, kwargs
            return m2.M2Evaluation(0.0, gradient.copy(), None, None, 0)

        def tri_state_certificate(
            observed_gradient: np.ndarray,
            specification: m2.PenaltySpec,
            transform: m2.ResidualTransform,
            **kwargs,
        ) -> m2.ZeroKKTCertificate:
            del observed_gradient, transform
            strength = float(specification.strength)
            supplied = kwargs.get("initial_correction")
            frozen_initial = (
                None
                if supplied is None
                else tuple(float(value) for value in supplied)
            )
            if strength in indeterminate_strengths:
                calls.append((strength, frozen_initial, "indeterminate", None))
                raise m2._ZeroKKTIndeterminate("synthetic boundary ambiguity")
            feasible = strength >= transition
            correction = (strength,) * m2.PAIR_DIMENSION
            calls.append(
                (
                    strength,
                    frozen_initial,
                    "feasible" if feasible else "infeasible",
                    correction,
                )
            )
            return m2.ZeroKKTCertificate(
                0.0 if feasible else 1.0,
                0.0,
                1,
                True,
                feasible,
                0.0 if feasible else 1.0,
                correction,
            )

        with mock.patch.object(m2, "evaluate_m2", side_effect=evaluation):
            with mock.patch.object(
                m2,
                "zero_kkt_certificate",
                side_effect=tri_state_certificate,
            ):
                maximum = m2.fold_lambda_maximum(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    "lasso",
                )

        self.assertIn(0.5, [strength for strength, _, _, _ in calls])
        self.assertIn(0.25, [strength for strength, _, _, _ in calls])
        self.assertNotIn(
            maximum.lower_bracket,
            indeterminate_strengths,
        )
        self.assertNotIn(
            maximum.upper_bracket,
            indeterminate_strengths,
        )
        self.assertLess(maximum.lower_bracket, transition)
        self.assertGreaterEqual(maximum.upper_bracket, transition)
        self.assertLessEqual(
            maximum.upper_bracket - maximum.lower_bracket,
            m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE
            * maximum.upper_bracket,
        )
        self.assertEqual(maximum.value, maximum.upper_bracket)
        self.assertEqual(maximum.zero_kkt_residual, 0.0)
        self.assertEqual(maximum.bisections, len(calls) - 2)
        self.assertLessEqual(
            maximum.bisections,
            m2.LAMBDA_MAXIMUM_BISECTIONS,
        )

        current_feasible_correction: tuple[float, ...] | None = None
        for call_index, (_, initial, outcome, correction) in enumerate(calls):
            if call_index == 0:
                self.assertIsNone(initial)
            else:
                self.assertEqual(initial, current_feasible_correction)
            if outcome == "feasible":
                self.assertIsNotNone(correction)
                current_feasible_correction = correction
        self.assertIsNotNone(current_feasible_correction)

    def test_zero_kkt_two_sided_oblique_certificates(self) -> None:
        direction = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        direction[0] = 1.0
        direction[1] = 2.0
        direction /= float(np.linalg.norm(direction))
        projector = np.outer(direction, direction)
        transform = dataclasses.replace(
            self.prepared.transform,
            range_basis=direction[:, None],
            range_projector=projector,
            scaled_residual_gram=projector,
        )
        source = 0.2 * direction
        gradient = -source
        decision_tolerance = 1.0e-11
        for kind, ratio in (
            ("lasso", 1.0),
            ("elastic_net", 0.25),
            ("elastic_net", 0.5),
            ("elastic_net", 0.75),
            ("group_lasso", 1.0),
        ):
            with self.subTest(kind=kind, ratio=ratio):
                unit_shape = m2.PenaltySpec(kind, 1.0, ratio)
                analytic_maximum = float(np.dot(source, direction)) / (
                    m2._dual_ball_support(direction, unit_shape)
                )
                below = m2.zero_kkt_certificate(
                    gradient,
                    m2.PenaltySpec(kind, 0.9 * analytic_maximum, ratio),
                    transform,
                    decision_tolerance=decision_tolerance,
                )
                repeat = m2.zero_kkt_certificate(
                    gradient,
                    m2.PenaltySpec(kind, 0.9 * analytic_maximum, ratio),
                    transform,
                    decision_tolerance=decision_tolerance,
                )
                above = m2.zero_kkt_certificate(
                    gradient,
                    m2.PenaltySpec(kind, 1.1 * analytic_maximum, ratio),
                    transform,
                    decision_tolerance=decision_tolerance,
                )
                self.assertEqual(below, repeat)
                self.assertTrue(below.certified)
                self.assertFalse(below.feasible)
                self.assertGreater(below.separation_margin, 0.0)
                self.assertLess(below.iterations, m2.PROX_MAXIMUM_ITERATIONS)
                self.assertTrue(above.certified)
                self.assertTrue(above.feasible)
                self.assertEqual(above.separation_margin, 0.0)
                self.assertLessEqual(above.residual, decision_tolerance)

    def test_zero_kkt_separator_roundoff_boundary_is_fail_closed(self) -> None:
        direction = np.ones(m2.PAIR_DIMENSION, dtype=float)
        direction /= float(np.linalg.norm(direction))
        projector = np.outer(direction, direction)
        transform = dataclasses.replace(
            self.prepared.transform,
            range_basis=direction[:, None],
            range_projector=projector,
            scaled_residual_gram=projector,
        )
        strength = 0.125
        decision_tolerance = float.fromhex("0x1.0000000000000p-30")
        boundary_source = np.full(
            m2.PAIR_DIMENSION,
            strength + decision_tolerance,
            dtype=float,
        )
        for kind in ("lasso", "group_lasso"):
            with self.subTest(kind=kind):
                specification = m2.PenaltySpec(kind, strength)
                for scale in (1.0, 2.0, 4.0, 8.0):
                    dual = boundary_source - scale * decision_tolerance
                    margin = m2._zero_kkt_separation_margin(
                        boundary_source,
                        dual,
                        specification,
                        transform,
                        decision_tolerance,
                        refine_active_normal=False,
                    )
                    self.assertLessEqual(margin, 0.0)
                near_source = np.nextafter(boundary_source, math.inf)
                near_margin = m2._zero_kkt_separation_margin(
                    near_source,
                    np.full(m2.PAIR_DIMENSION, strength, dtype=float),
                    specification,
                    transform,
                    decision_tolerance,
                    refine_active_normal=False,
                )
                self.assertLessEqual(near_margin, 0.0)
                far_source = boundary_source + float.fromhex(
                    "0x1.0000000000000p-35"
                )
                far_margin = m2._zero_kkt_separation_margin(
                    far_source,
                    np.full(m2.PAIR_DIMENSION, strength, dtype=float),
                    specification,
                    transform,
                    decision_tolerance,
                    refine_active_normal=False,
                )
                self.assertGreater(far_margin, 0.0)

        # Coordinate products round upward to the minimum subnormal here while
        # the exact homogeneous gap is zero.  Additive product-underflow guards
        # must keep the certificate fail-closed; relative gamma alone cannot.
        eta = math.ulp(0.0)
        tiny = 0.75 * math.ldexp(1.0, -537)
        self.assertEqual(tiny * tiny, eta)
        subnormal_product_margin = m2._zero_kkt_separation_margin(
            np.full(m2.PAIR_DIMENSION, tiny, dtype=float),
            np.zeros(m2.PAIR_DIMENSION, dtype=float),
            m2.PenaltySpec("lasso", 0.0),
            transform,
            tiny,
            refine_active_normal=False,
        )
        self.assertLessEqual(subnormal_product_margin, 0.0)

    def test_zero_kkt_minimax_direction_is_projected_before_rejection(self) -> None:
        range_basis = np.zeros((m2.PAIR_DIMENSION, 2), dtype=float)
        range_basis[0, 0] = 1.0
        range_basis[1, 1] = 1.0
        range_projector = range_basis @ range_basis.T
        transform = dataclasses.replace(
            self.prepared.transform,
            range_basis=range_basis,
            range_projector=range_projector,
            scaled_residual_gram=range_projector,
        )
        source = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        source[0] = 1.1
        dual = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        dual[:2] = -1.0
        specification = m2.PenaltySpec("lasso", 1.0)
        decision_tolerance = 1.0e-12

        # This valid but deliberately nonoptimal dual produces a range-residual
        # direction whose gap is negative.
        baseline = m2._zero_kkt_separation_margin(
            source,
            dual,
            specification,
            transform,
            decision_tolerance,
            refine_active_normal=False,
        )
        self.assertLessEqual(baseline, 0.0)

        # The raw minimax-style direction contains a null component, so its
        # cheap unprojected gap is also negative.  Its exact stored projection
        # is e_0, however, and the full guarded calculation must retain that
        # valid positive separator.
        raw_direction = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        raw_direction[[0, 2]] = 1.0
        raw_gap = (
            float(np.dot(source, raw_direction))
            - m2._dual_ball_support(raw_direction, specification)
            - decision_tolerance * float(np.sum(np.abs(raw_direction)))
        )
        self.assertLess(raw_gap, 0.0)
        projected_margin = m2._zero_kkt_separation_margin(
            source,
            dual,
            specification,
            transform,
            decision_tolerance,
            refine_active_normal=False,
            additional_direction=raw_direction,
        )
        self.assertGreater(projected_margin, 0.0)

    def test_tangential_midpoints_have_separation_certificates(self) -> None:
        import qualify_v3_m2_core as qualification

        fixture = qualification._frozen_fixture()
        fixed_m1_theta = 0.12 * fixture.fixed_m1_theta
        lasso_planted = qualification._small_plants(fixture)["lasso"]
        lasso_correction = m2.population_linear_correction(
            lasso_planted,
            fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        lasso_gradient = m2.evaluate_m2(
            np.zeros(m2.PAIR_DIMENSION, dtype=float),
            fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            linear_correction=lasso_correction,
        ).gradient
        epsilon = np.finfo(float).eps
        roundoff_gamma = (
            m2.PAIR_DIMENSION
            * epsilon
            / (1.0 - m2.PAIR_DIMENSION * epsilon)
        )
        lasso_decision_tolerance = min(
            m2.PROX_TOLERANCE,
            roundoff_gamma * float(np.linalg.norm(lasso_gradient, ord=np.inf)),
        )
        self.assertLess(lasso_decision_tolerance, m2.PROX_TOLERANCE)
        lasso_midpoint = float.fromhex("0x1.afed980000000p-5")
        self.assertEqual(lasso_midpoint, 0.052725598216056824)
        lasso_certificate = m2.zero_kkt_certificate(
            lasso_gradient,
            m2.PenaltySpec("lasso", lasso_midpoint),
            fixture.small_transform,
            decision_tolerance=lasso_decision_tolerance,
        )
        self.assertTrue(lasso_certificate.certified)
        self.assertFalse(lasso_certificate.feasible)
        self.assertGreater(lasso_certificate.separation_margin, 0.0)
        self.assertLess(
            lasso_certificate.iterations, m2.PROX_MAXIMUM_ITERATIONS
        )

        planted = qualification._small_plants(fixture)["group_lasso"]
        correction = m2.population_linear_correction(
            planted,
            fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
        )
        gradient = m2.evaluate_m2(
            np.zeros(m2.PAIR_DIMENSION, dtype=float),
            fixed_m1_theta,
            fixture.small_prepared,
            fixture.candidates,
            fixture.geometry,
            fixture.pairs,
            linear_correction=correction,
        ).gradient
        decision_tolerance = min(
            m2.PROX_TOLERANCE,
            roundoff_gamma * float(np.linalg.norm(gradient, ord=np.inf)),
        )
        self.assertLess(decision_tolerance, m2.PROX_TOLERANCE)
        failing_midpoint = float.fromhex("0x1.5b00000000000p-5")
        self.assertEqual(failing_midpoint, 0.0423583984375)
        specification = m2.PenaltySpec("group_lasso", failing_midpoint)
        first = m2.zero_kkt_certificate(
            gradient,
            specification,
            fixture.small_transform,
            decision_tolerance=decision_tolerance,
        )
        second = m2.zero_kkt_certificate(
            gradient,
            specification,
            fixture.small_transform,
            decision_tolerance=decision_tolerance,
        )
        self.assertEqual(first, second)
        self.assertTrue(first.certified)
        self.assertFalse(first.feasible)
        self.assertGreater(first.separation_margin, 0.0)
        self.assertLess(first.iterations, m2.PROX_MAXIMUM_ITERATIONS)

        # This exact midpoint was the tangential group-lasso ambiguity exposed
        # by the failed comprehensive rerun.  It must reach the normalized
        # minimax phase and may certify only through the guarded separator.
        minimax_midpoint = float.fromhex("0x1.5b96000000000p-5")
        self.assertEqual(minimax_midpoint, 0.04242992401123047)
        minimax_certificate = m2.zero_kkt_certificate(
            gradient,
            m2.PenaltySpec("group_lasso", minimax_midpoint),
            fixture.small_transform,
            decision_tolerance=decision_tolerance,
        )
        self.assertTrue(minimax_certificate.certified)
        self.assertFalse(minimax_certificate.feasible)
        self.assertGreater(minimax_certificate.separation_margin, 0.0)
        self.assertGreater(
            minimax_certificate.iterations,
            m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS,
        )
        self.assertLessEqual(
            minimax_certificate.iterations,
            m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS
            + m2.ZERO_KKT_MINIMAX_MAXIMUM_ITERATIONS,
        )

        # With one iteration available beyond the accelerated phase and the
        # separator deliberately disabled, the minimax phase must consume that
        # one iteration and then fail closed.  This pins shared-cap accounting
        # at the minimax-to-fallback boundary without changing acceptance.
        with mock.patch.object(
            m2,
            "_zero_kkt_separation_margin",
            return_value=-math.inf,
        ):
            with self.assertRaisesRegex(
                m2.M2Error,
                r"minimax_iterations=1(?:,|$)",
            ):
                m2.zero_kkt_certificate(
                    gradient,
                    m2.PenaltySpec("group_lasso", minimax_midpoint),
                    fixture.small_transform,
                    decision_tolerance=decision_tolerance,
                    maximum_iterations=(
                        m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS + 1
                    ),
                )

        # Give the smooth phase enough room to reach the under-resolved final
        # face, then disable every separator so the active-face polish and
        # fallback must fail closed.  The reported phase counts must add up and
        # never exceed the caller's deliberately small shared cap.
        active_face_cap = (
            m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS + 900
        )
        with mock.patch.object(
            m2,
            "_zero_kkt_separation_margin",
            return_value=-math.inf,
        ):
            with self.assertRaises(m2.M2Error) as active_face_cap_error:
                m2.zero_kkt_certificate(
                    gradient,
                    m2.PenaltySpec("group_lasso", minimax_midpoint),
                    fixture.small_transform,
                    decision_tolerance=decision_tolerance,
                    maximum_iterations=active_face_cap,
                )
        active_face_message = str(active_face_cap_error.exception)

        def integer_field(name: str) -> int:
            return int(
                active_face_message.split(f"{name}=", 1)[1].split(",", 1)[0]
            )

        self.assertGreater(integer_field("minimax_active_face_growths"), 0)
        self.assertGreater(integer_field("minimax_active_face_qp_events"), 0)
        accelerated_count = m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS
        minimax_count = integer_field("minimax_iterations")
        fallback_count = integer_field("fallback_iterations")
        total_count = integer_field("total_iterations_consumed")
        self.assertEqual(
            total_count,
            accelerated_count + minimax_count + fallback_count,
        )
        self.assertLessEqual(total_count, active_face_cap)

        upper = m2.zero_kkt_certificate(
            gradient,
            m2.PenaltySpec("group_lasso", 0.04248046875),
            fixture.small_transform,
            decision_tolerance=decision_tolerance,
        )
        self.assertTrue(upper.certified)
        self.assertTrue(upper.feasible)
        self.assertLessEqual(upper.residual, decision_tolerance)

        def run_recorded_fold():
            calls: list[tuple[float, np.ndarray | None, m2.ZeroKKTCertificate]] = []
            original_certificate = m2.zero_kkt_certificate

            def recording_certificate(*args, **kwargs):
                initial = kwargs.get("initial_correction")
                frozen_initial = (
                    None
                    if initial is None
                    else np.asarray(initial, dtype=float).copy()
                )
                observed = original_certificate(*args, **kwargs)
                calls.append((float(args[1].strength), frozen_initial, observed))
                return observed

            with mock.patch.object(
                m2,
                "zero_kkt_certificate",
                side_effect=recording_certificate,
            ):
                maximum = m2.fold_lambda_maximum(
                    fixed_m1_theta,
                    fixture.small_prepared,
                    fixture.candidates,
                    fixture.geometry,
                    fixture.pairs,
                    "group_lasso",
                    linear_correction=correction,
                )
            return maximum, calls

        first_maximum, first_calls = run_recorded_fold()
        second_maximum, second_calls = run_recorded_fold()
        self.assertEqual(first_maximum, second_maximum)
        self.assertGreater(first_maximum.value, 0.0)
        self.assertLessEqual(
            first_maximum.upper_bracket - first_maximum.lower_bracket,
            m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE
            * first_maximum.upper_bracket,
        )
        self.assertLessEqual(
            first_maximum.zero_kkt_residual,
            decision_tolerance,
        )
        self.assertEqual(
            [strength for strength, _, _ in first_calls],
            [strength for strength, _, _ in second_calls],
        )
        self.assertEqual(
            [certificate for _, _, certificate in first_calls],
            [certificate for _, _, certificate in second_calls],
        )
        for calls in (first_calls, second_calls):
            self.assertIsNone(calls[0][1])
            self.assertTrue(calls[0][2].feasible)
            feasible_upper = calls[0][2]
            saw_infeasible_midpoint = False
            for _, initial, observed in calls[1:]:
                self.assertIsNotNone(initial)
                np.testing.assert_array_equal(
                    initial,
                    np.asarray(feasible_upper.correction, dtype=float),
                )
                if observed.feasible:
                    feasible_upper = observed
                else:
                    saw_infeasible_midpoint = True
            self.assertTrue(saw_infeasible_midpoint)
            polished = [
                observed
                for _, _, observed in calls
                if observed.active_face_growths > 0
            ]
            self.assertTrue(polished)
            saw_materially_nonuniform_face = False
            for observed in polished:
                self.assertTrue(observed.certified)
                self.assertFalse(observed.feasible)
                self.assertGreater(observed.separation_margin, 0.0)
                self.assertGreater(observed.active_face_qp_events, 0)
                self.assertGreaterEqual(observed.active_face_size, 2)
                self.assertGreaterEqual(observed.active_face_support_size, 2)
                self.assertGreater(
                    observed.active_face_effective_constraint_count, 1.0
                )
                self.assertLess(observed.active_face_maximum_weight, 1.0)
                uniform_weight = 1.0 / float(
                    observed.active_face_support_size
                )
                if observed.active_face_maximum_weight > (
                    uniform_weight + 64.0 * np.finfo(float).eps
                ):
                    saw_materially_nonuniform_face = True
            self.assertTrue(saw_materially_nonuniform_face)
        self.assertEqual(m2.PROX_TOLERANCE, 1.0e-12)
        self.assertEqual(m2.PROX_MAXIMUM_ITERATIONS, 10_000)
        self.assertEqual(m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS, 512)
        self.assertEqual(m2.ZERO_KKT_MINIMAX_MAXIMUM_ITERATIONS, 1_024)
        self.assertEqual(
            m2.ZERO_KKT_MINIMAX_SMOOTHING_MINIMUM_EXPONENT, 1
        )
        self.assertEqual(
            m2.ZERO_KKT_MINIMAX_SMOOTHING_MAXIMUM_EXPONENT, 18
        )
        self.assertEqual(m2.ZERO_KKT_MINIMAX_NEWTON_STEPS_PER_SMOOTHING, 40)
        self.assertLess(
            m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS
            + m2.ZERO_KKT_MINIMAX_MAXIMUM_ITERATIONS,
            m2.PROX_MAXIMUM_ITERATIONS,
        )

    def test_fold_local_lambda_maximum_zero_kkt_all_penalties(self) -> None:
        for kind, ratio in (
            ("lasso", 1.0),
            ("elastic_net", 0.25),
            ("elastic_net", 0.5),
            ("elastic_net", 0.75),
            ("group_lasso", 1.0),
        ):
            with self.subTest(kind=kind, ratio=ratio):
                strengths: list[float] = []
                certificates: list[m2.ZeroKKTCertificate] = []
                original_certificate = m2.zero_kkt_certificate

                def recording_certificate(*args, **kwargs):
                    strengths.append(float(args[1].strength))
                    certificate = original_certificate(*args, **kwargs)
                    certificates.append(certificate)
                    return certificate

                with mock.patch.object(
                    m2, "zero_kkt_certificate", side_effect=recording_certificate
                ):
                    maximum = m2.fold_lambda_maximum(
                        self.fixed_m1_theta,
                        self.prepared,
                        self.candidates,
                        self.geometry,
                        self.pairs,
                        kind,
                        l1_ratio=ratio,
                    )
                self.assertEqual(strengths[0], m2.LAMBDA_MAXIMUM_INITIAL_UPPER)
                self.assertEqual(m2.LAMBDA_MAXIMUM_INITIAL_UPPER, 1.0)
                self.assertEqual(m2.LAMBDA_MAXIMUM_MAXIMUM_EXPANSIONS, 80)
                self.assertGreater(maximum.value, 0.0)
                self.assertLessEqual(
                    maximum.upper_bracket - maximum.lower_bracket,
                    m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE
                    * maximum.upper_bracket,
                )
                self.assertLessEqual(maximum.zero_kkt_residual, 1.0e-7)
                self.assertTrue(
                    any(
                        certificate.iterations
                        > m2.ZERO_KKT_ACCELERATED_MAXIMUM_ITERATIONS
                        for certificate in certificates
                    ),
                    "fold-local search did not exercise the minimax route",
                )
                self.assertTrue(
                    all(
                        certificate.iterations <= m2.PROX_MAXIMUM_ITERATIONS
                        for certificate in certificates
                    )
                )
                fitted = m2.fit_m2(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    m2.PenaltySpec(kind, maximum.value, ratio),
                    maximum_iterations=20,
                )
                self.assertEqual(fitted.status, "pass")
                self.assertEqual(fitted.iterations, 0)
                np.testing.assert_array_equal(fitted.beta, np.zeros(120))
                self.assertLessEqual(fitted.kkt_residual, m2.KKT_TOLERANCE)

    def test_deterministic_planted_probability_recovery_all_penalties(self) -> None:
        plants: dict[str, np.ndarray] = {}
        sparse = np.zeros(120)
        sparse[[3, 41, 89]] = (0.12, -0.10, 0.09)
        plants["lasso"] = m2._range_project_vector(
            sparse, self.prepared.transform
        )
        elastic = np.zeros(120)
        elastic[[5, 17, 52, 78, 101]] = (0.08, -0.07, 0.09, -0.06, 0.07)
        plants["elastic_net"] = m2._range_project_vector(
            elastic, self.prepared.transform
        )
        group = np.zeros(120)
        first_group = m2.BILATERAL_GROUPS[7][1]
        group[list(first_group)] = (0.07, -0.09, 0.08, -0.06)
        plants["group_lasso"] = m2._range_project_vector(
            group, self.prepared.transform
        )

        for kind, ratio in (
            ("lasso", 1.0),
            ("elastic_net", 0.5),
            ("group_lasso", 1.0),
        ):
            with self.subTest(kind=kind):
                planted = plants[kind]
                correction = m2.population_linear_correction(
                    planted,
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                )
                planted_gradient = m2.evaluate_m2(
                    planted,
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    linear_correction=correction,
                ).gradient
                self.assertLess(float(np.max(np.abs(planted_gradient))), 2.0e-12)
                maximum = m2.fold_lambda_maximum(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    kind,
                    l1_ratio=ratio,
                    linear_correction=correction,
                )
                specification = m2.PenaltySpec(
                    kind, 0.001 * maximum.value, ratio
                )
                first = m2.fit_m2(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    linear_correction=correction,
                    maximum_iterations=2_000,
                )
                second = m2.fit_m2(
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                    specification,
                    linear_correction=correction,
                    maximum_iterations=2_000,
                )
                np.testing.assert_array_equal(first.beta, second.beta)
                self.assertEqual(first.telemetry, second.telemetry)
                self.assertLessEqual(first.kkt_residual, m2.KKT_TOLERANCE)
                self.assertGreater(first.telemetry["Dykstra_warm_start_calls"], 0)
                self.assertEqual(
                    first.telemetry["initial_Lipschitz_estimate"], 1.0
                )
                self.assertEqual(
                    first.telemetry["majorization_epsilon_multiplier"], 64.0
                )
                self.assertEqual(m2.FISTA_MAXIMUM_ITERATIONS, 20_000)
                self.assertLessEqual(
                    first.relative_objective_change,
                    m2.RELATIVE_OBJECTIVE_TOLERANCE,
                )
                recovered = m2.compare_conditional_probabilities(
                    planted,
                    first.beta,
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                )
                null = m2.compare_conditional_probabilities(
                    planted,
                    np.zeros(120),
                    self.fixed_m1_theta,
                    self.prepared,
                    self.candidates,
                    self.geometry,
                    self.pairs,
                )
                self.assertLess(recovered["mean_KL_nats"], null["mean_KL_nats"])
                self.assertLess(
                    recovered["maximum_candidate_probability_absolute_error"],
                    null["maximum_candidate_probability_absolute_error"],
                )
                self.assertLess(recovered["mean_KL_nats"], 2.0e-5)

    def test_cli_is_synthetic_only(self) -> None:
        parser = m2.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(option_strings, {"-h", "--help", "--self-test"})
        for forbidden in (
            "--target-root",
            "--interface-root",
            "--final-root",
            "--unassigned-root",
            "--swc-root",
            "--output-root",
        ):
            self.assertNotIn(forbidden, option_strings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
