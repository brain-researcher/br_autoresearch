#!/usr/bin/env python3
"""Focused tests for the frozen EP10 v3 M1 target-free engine."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ENGINE = Path(__file__).resolve().parents[1] / "tools" / "fit_v3_m1.py"
SPEC = importlib.util.spec_from_file_location("ep10_fit_v3_m1_tested", ENGINE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import M1 engine: {ENGINE}")
m1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m1
SPEC.loader.exec_module(m1)


class M1DiagnosticSurfaceTests(unittest.TestCase):
    def test_cli_has_no_target_or_m0_input(self) -> None:
        parser = m1.build_parser()
        actions = parser._actions
        destinations = {action.dest for action in actions}
        self.assertNotIn("target_root", destinations)
        self.assertNotIn("m0_root", destinations)
        self.assertNotIn("swc_root", destinations)
        options = {
            option
            for action in actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {
                "-h",
                "--help",
                "--contract",
                "--synthetic-design-addendum",
                "--synthetic-recovery-addendum",
                "--solver-addendum",
                "--unit-test",
                "--synthetic-rank-probe",
                "--synthetic-population-recovery-probe",
                "--synthetic-warm-zero-diagnostic",
                "--probe",
                "--interface-root",
                "--geometry-root",
                "--output-root",
            },
        )
        for forbidden in ("target", "m0", "swc", "final", "unassigned"):
            self.assertFalse(any(forbidden in option.lower() for option in options))
        parsed = parser.parse_args(
            [
                "--synthetic-population-recovery-probe",
                "--output-root",
                "/tmp/not_opened_by_parser",
            ]
        )
        self.assertTrue(parsed.synthetic_population_recovery_probe)
        self.assertEqual(
            parsed.synthetic_recovery_addendum,
            m1.SYNTHETIC_RECOVERY_ADDENDUM_PATH,
        )
        self.assertEqual(parsed.solver_addendum, m1.SOLVER_ADDENDUM_PATH)
        diagnostic = parser.parse_args(
            [
                "--synthetic-warm-zero-diagnostic",
                "--output-root",
                "/tmp/not_opened_by_parser",
            ]
        )
        self.assertTrue(diagnostic.synthetic_warm_zero_diagnostic)
        self.assertIsNone(diagnostic.interface_root)
        self.assertIsNone(diagnostic.geometry_root)

    def test_warm_zero_failure_context_is_scalar_and_sealed(self) -> None:
        line_failure = m1.LineSearchFailure(
            "strong-Wolfe zoom bracket reached floating-point resolution",
            evaluations=18,
            expansions=0,
            zoom_iterations=18,
            all_values_finite=True,
            scalar_context={
                "initial_objective": 1.0,
                "initial_directional_derivative": -1.0e-8,
                "terminal_bracket_low": 0.25,
                "terminal_bracket_high": math.nextafter(0.25, 1.0),
                "terminal_midpoint": 0.25,
                "terminal_bracket_absolute_width": (
                    math.nextafter(0.25, 1.0) - 0.25
                ),
                "last_armijo_slack": -1.0e-16,
                "last_curvature_ratio": 0.95,
            },
        )
        gradient = np.zeros(m1.PARAMETER_DIMENSION)
        gradient[0] = m1.GRADIENT_TOLERANCE
        certificate = m1._certificate(
            line_failure,
            1.0,
            gradient,
            1.0e-8,
            2.0e-10,
        )
        self.assertTrue(certificate["eligible_failure"])
        self.assertTrue(certificate["gradient_predicate_pass"])
        self.assertFalse(certificate["relative_gap_predicate_pass"])
        self.assertFalse(certificate["certificate_pass"])
        context = {
            "ridge_lambda": 1.0e-8,
            "optimizer_iteration": 7,
            "completed_iterations": 6,
            "accepted_steps": 6,
            "current_objective": 1.0,
            "current_gradient_infinity_norm": m1.GRADIENT_TOLERANCE,
            "prior_relative_objective_change": 2.0e-10,
            "last_Newton_solve": {
                "diagonal_shift": 0.0,
                "scaled_backward_error": 1.0e-17,
                "directional_derivative": -1.0e-8,
            },
            "line_search_failure": {
                "evaluations": line_failure.evaluations,
                "expansions": line_failure.expansions,
                "zoom_iterations": line_failure.zoom_iterations,
                "all_values_finite": line_failure.all_values_finite,
                **line_failure.scalar_context,
            },
            "certificate": certificate,
        }
        error = m1.UncertifiedLineSearchFailure(
            "strong-Wolfe search failed without the frozen 261D certificate",
            context,
        )
        record = m1._diagnostic_failure_record(
            "warm_repeat1",
            1.0e-8,
            "immediately_preceding_converged_joint_B_and_gamma",
            1.0e-7,
            12,
            12,
            error,
        )
        encoded = json.dumps(record, allow_nan=False, sort_keys=True)
        self.assertIn('"phase": "warm_repeat1"', encoded)
        self.assertIn('"lambda": 1e-08', encoded)
        self.assertNotIn("[", encoded)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        seals = m1._synthetic_warm_zero_diagnostic_seals()
        self.assertTrue(seals)
        self.assertTrue(all(value is False for value in seals.values()))


class M1TargetFreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.cells,
            cls.moments,
            cls.prepared,
            cls.library,
            cls.geometry,
            cls.planted,
        ) = m1.generate_synthetic_dataset(
            groups=2, rows_per_group=17, seed=20260931
        )

    def test_frozen_contracts_and_all_addenda_validate(self) -> None:
        contract = m1.validate_contract()
        self.assertEqual(contract["schema_version"], "episode10_v3_m1_model_v1")
        self.assertEqual(
            m1.WARM_ZERO_OUTPUT_PREFIX,
            "v3_m1_synthetic_warm_zero_diagnostic_v2",
        )
        self.assertEqual(m1.POST_WITNESS_CONFIRMATION_ATTEMPTS, 1)

    def test_stable_line_difference_preserves_absolute_evaluator(self) -> None:
        indexes = tuple(
            index for index, cell in enumerate(self.cells) if cell.k in (2, 3)
        )[:4]
        prepared = m1.prepare_data(self.cells, self.moments, indexes)
        theta = 0.01 * np.cos(np.arange(m1.PARAMETER_DIMENSION) + 0.3)
        direction = 0.003 * np.sin(np.arange(m1.PARAMETER_DIMENSION) + 0.7)
        step = 0.25
        candidate = theta + step * direction
        origin = m1.evaluate_model(
            theta,
            prepared,
            self.library,
            self.geometry,
            0.1,
            need_hessian=True,
        )
        self.assertIsNotNone(origin.joint_residuals)
        assert origin.joint_residuals is not None
        initial_direct, _ = m1._direct_directional_derivative(
            prepared,
            self.library,
            self.geometry,
            theta,
            direction,
            0.1,
        )
        self.assertAlmostEqual(
            initial_direct,
            float(np.dot(origin.gradient, direction)),
            delta=2.0e-12,
        )
        ordinary_candidate = m1.evaluate_model(
            candidate,
            prepared,
            self.library,
            self.geometry,
            0.1,
            need_hessian=False,
        )
        line_candidate = m1.evaluate_model(
            candidate,
            prepared,
            self.library,
            self.geometry,
            0.1,
            need_hessian=False,
            line_origin=theta,
            line_direction=direction,
            line_step=step,
        )
        self.assertEqual(line_candidate.objective, ordinary_candidate.objective)
        np.testing.assert_array_equal(
            line_candidate.gradient, ordinary_candidate.gradient
        )
        self.assertIsNotNone(line_candidate.line_objective_difference)
        assert line_candidate.line_objective_difference is not None
        self.assertAlmostEqual(
            line_candidate.line_objective_difference,
            ordinary_candidate.objective - origin.objective,
            delta=2.0e-12,
        )
        self.assertIsNotNone(line_candidate.direct_directional_derivative)
        assert line_candidate.direct_directional_derivative is not None
        self.assertAlmostEqual(
            line_candidate.direct_directional_derivative,
            float(np.dot(ordinary_candidate.gradient, direction)),
            delta=2.0e-12,
        )
        self.assertEqual(
            line_candidate.line_small_increment_cells
            + line_candidate.line_general_cells,
            len(prepared.cells),
        )

        large_direction = 0.8 * np.sin(
            np.arange(m1.PARAMETER_DIMENSION) + 0.41
        )
        large_step = 1.0
        large_candidate_vector = theta + large_step * large_direction
        large_plain = m1.evaluate_model(
            large_candidate_vector,
            prepared,
            self.library,
            self.geometry,
            0.1,
            need_hessian=False,
        )
        large_line = m1.evaluate_model(
            large_candidate_vector,
            prepared,
            self.library,
            self.geometry,
            0.1,
            need_hessian=False,
            line_origin=theta,
            line_direction=large_direction,
            line_step=large_step,
        )
        self.assertGreater(large_line.line_general_cells, 0)
        self.assertIsNotNone(large_line.line_objective_difference)
        assert large_line.line_objective_difference is not None
        self.assertAlmostEqual(
            large_line.line_objective_difference,
            large_plain.objective - origin.objective,
            delta=2.0e-12,
        )

    def test_stable_line_forms_boundary_zero_and_revival(self) -> None:
        q = np.asarray((-0.2, 0.0, 0.3), dtype=float)
        r = np.asarray((-0.5, 0.1, 0.5), dtype=float)
        q_maximum = float(np.max(q))
        baseline = np.exp(q - q_maximum)
        combined = q + r
        combined_maximum = float(np.max(combined))
        small = m1._finish_line_cell_difference(
            [m1._fsum_batch(baseline, "test baseline")],
            [m1._fsum_batch(baseline * np.expm1(r), "test small")],
            form="small_log1p_expm1",
            maximum_shift=0.0,
        )
        general = m1._finish_line_cell_difference(
            [m1._fsum_batch(baseline, "test baseline")],
            [
                m1._fsum_batch(
                    np.exp(combined - combined_maximum), "test combined"
                )
            ],
            form="general_shifted",
            maximum_shift=combined_maximum - q_maximum,
        )
        self.assertAlmostEqual(small, general, delta=5.0e-16)
        self.assertEqual(m1._line_difference_form(0.5), "small_log1p_expm1")
        self.assertEqual(
            m1._line_difference_form(math.nextafter(0.5, math.inf)),
            "general_shifted",
        )
        zero = m1._finish_line_cell_difference(
            [1.0, 2.0],
            [0.0, 0.0],
            form="small_log1p_expm1",
            maximum_shift=0.0,
        )
        self.assertEqual(zero, 0.0)

        revived_q = np.asarray((0.0, -1000.0), dtype=float)
        revived_r = np.asarray((0.0, 2000.0), dtype=float)
        revived_base_maximum = float(np.max(revived_q))
        revived_baseline = np.exp(revived_q - revived_base_maximum)
        self.assertEqual(float(revived_baseline[1]), 0.0)
        revived_combined = revived_q + revived_r
        revived_combined_maximum = float(np.max(revived_combined))
        revived = m1._finish_line_cell_difference(
            [m1._fsum_batch(revived_baseline, "revived baseline")],
            [
                m1._fsum_batch(
                    np.exp(revived_combined - revived_combined_maximum),
                    "revived combined",
                )
            ],
            form="general_shifted",
            maximum_shift=(
                revived_combined_maximum - revived_base_maximum
            ),
        )
        self.assertEqual(revived, 1000.0)
        with self.assertRaises(m1.M1Error):
            m1._finish_line_cell_difference(
                [1.0],
                [-1.0],
                form="small_log1p_expm1",
                maximum_shift=0.0,
            )
        with self.assertRaises(m1.M1Error):
            m1._fsum_batch(np.asarray((1.0, math.inf)), "nonfinite test")

    def test_sub_ulp_total_objective_uses_stable_Wolfe_difference(self) -> None:
        anchor = 1.0
        slope = -1.0e-16
        curvature = 1.0e-16
        vector = np.asarray((0.0,))
        gradient = np.asarray((slope,))
        direction = np.asarray((1.0,))

        def evaluate(candidate: np.ndarray, step: float) -> m1.Evaluation:
            difference = slope * step + 0.5 * curvature * step * step
            derivative = slope + curvature * step
            return m1.Evaluation(
                objective=anchor + difference,
                gradient=np.asarray((derivative,)),
                hessian=None,
                log_probability_nats=None,
                marginals=None,
                candidate_visits=0,
                raw_hessian_symmetry_error=None,
                elapsed_seconds=0.0,
                joint_residuals=None,
                line_objective_difference=difference,
                direct_directional_derivative=derivative,
                line_small_increment_cells=1,
                line_general_cells=0,
            )

        result = m1._strong_wolfe_search(
            vector,
            anchor,
            gradient,
            direction,
            slope,
            evaluate,
        )
        self.assertEqual(result.step, 1.0)
        self.assertEqual(result.value, anchor)
        self.assertLess(result.objective_difference, 0.0)
        self.assertGreaterEqual(result.armijo_slack, 0.0)
        self.assertLessEqual(result.curvature_ratio, m1.CURVATURE_C2)

    def test_singleton_line_terms_are_ridge_and_correction_only(self) -> None:
        source = self.cells[:2]
        cells = (
            m1.Cell(
                source[0].identifier,
                "zero_group",
                source[0].layer,
                source[0].hemisphere,
                source[0].soma_um,
                source[0].soma_distance_mm,
                0,
                0,
                source[0].segment_unmapped,
            ),
            m1.Cell(
                source[1].identifier,
                "full_group",
                source[1].layer,
                source[1].hemisphere,
                source[1].soma_um,
                source[1].soma_distance_mm,
                16,
                0,
                source[1].segment_unmapped,
            ),
        )
        prepared = m1.PreparedData(
            cells,
            self.prepared.x[:2],
            np.zeros((2, 6)),
            np.ones(6),
            np.asarray((0.5, 0.5)),
            self.prepared.design_transform,
        )
        theta = 0.001 * np.cos(np.arange(m1.PARAMETER_DIMENSION) + 0.2)
        direction = 0.002 * np.sin(np.arange(m1.PARAMETER_DIMENSION) + 0.4)
        step = 0.125
        candidate = theta + step * direction
        displacement = candidate - theta
        ridge = 0.1
        correction = 0.0003 * np.cos(
            np.arange(m1.PARAMETER_DIMENSION) + 0.9
        )
        result = m1.evaluate_model(
            candidate,
            prepared,
            self.library,
            self.geometry,
            ridge,
            need_hessian=False,
            linear_correction=correction,
            line_origin=theta,
            line_direction=direction,
            line_step=step,
        )
        expected_difference = math.fsum(
            (
                ridge
                * math.fsum(
                    float(theta[index]) * float(displacement[index])
                    for index in range(m1.PARAMETER_DIMENSION)
                ),
                0.5
                * ridge
                * math.fsum(
                    float(value) * float(value) for value in displacement
                ),
                math.fsum(
                    float(correction[index]) * float(displacement[index])
                    for index in range(m1.PARAMETER_DIMENSION)
                ),
            )
        )
        expected_derivative = math.fsum(
            (
                ridge
                * math.fsum(
                    float(candidate[index]) * float(direction[index])
                    for index in range(m1.PARAMETER_DIMENSION)
                ),
                math.fsum(
                    float(correction[index]) * float(direction[index])
                    for index in range(m1.PARAMETER_DIMENSION)
                ),
            )
        )
        self.assertEqual(result.line_objective_difference, expected_difference)
        self.assertEqual(
            result.direct_directional_derivative, expected_derivative
        )
        self.assertEqual(result.line_small_increment_cells, 2)
        self.assertEqual(result.line_general_cells, 0)

        unchanged = m1.evaluate_model(
            theta,
            prepared,
            self.library,
            self.geometry,
            ridge,
            need_hessian=False,
            linear_correction=correction,
            line_origin=theta,
            line_direction=direction,
            line_step=0.0,
        )
        self.assertEqual(unchanged.line_objective_difference, 0.0)

    def test_certificate_still_requires_gradient_and_gap(self) -> None:
        failure = m1.LineSearchFailure(
            "strong-Wolfe zoom bracket reached floating-point resolution",
            evaluations=58,
            expansions=0,
            zoom_iterations=58,
            all_values_finite=True,
        )
        gradient = np.zeros(m1.PARAMETER_DIMENSION)
        gradient[0] = 6.9385468515204e-7
        certificate = m1._certificate(
            failure,
            6.215630573370164,
            gradient,
            1000.0,
            1.5526577550129445e-4,
        )
        self.assertFalse(certificate["gradient_predicate_pass"])
        self.assertTrue(certificate["relative_gap_predicate_pass"])
        self.assertFalse(certificate["certificate_pass"])

    def test_fixed_panel_enumeration_is_complete(self) -> None:
        library = m1.CandidateLibrary()
        observed = library.all_counts()
        expected = {k: math.comb(16, k) for k in range(17)}
        self.assertEqual(observed, expected)
        self.assertEqual(sum(observed.values()), 65536)

    def test_candidate_order_is_lexicographic_for_every_K(self) -> None:
        for k in range(17):
            observed = tuple(
                tuple(int(value) for value in np.flatnonzero(row))
                for row in self.library.basis(k).membership
            )
            expected = tuple(itertools.combinations(range(16), k))
            self.assertEqual(observed, expected)

    def test_scalar_and_vector_geometry_agree(self) -> None:
        library = m1.CandidateLibrary()
        pairs = m1.pair_geometry_from_coordinates(m1.synthetic_target_coordinates())
        geometry = m1.GeometryEngine(library, pairs)
        cell = m1._synthetic_cell(
            0,
            "g0",
            3,
            np.asarray((0.2, -0.4, 0.6)),
            0.5,
        )
        basis = library.basis(3)
        for index in (0, basis.count // 2, basis.count - 1):
            selected = tuple(np.flatnonzero(basis.membership[index]))
            scalar = m1._scalar_set_features(cell, selected, pairs)
            vector = geometry.raw_batch(cell, index, index + 1)[0]
            np.testing.assert_allclose(vector, scalar, atol=1.0e-12, rtol=0.0)

    def test_streamed_Hessian_has_exact_cross_block(self) -> None:
        indexes = tuple(
            i for i, cell in enumerate(self.cells) if cell.k in (2, 3)
        )[:4]
        prepared = m1.prepare_data(self.cells, self.moments, indexes)
        theta = 0.01 * np.cos(np.arange(m1.PARAMETER_DIMENSION))
        result = m1.evaluate_model(
            theta,
            prepared,
            self.library,
            self.geometry,
            0.1,
            need_hessian=True,
        )
        self.assertIsNotNone(result.hessian)
        assert result.hessian is not None
        self.assertGreater(float(np.max(np.abs(result.hessian[:255, 255:]))), 0.0)
        self.assertEqual(result.hessian.shape, (261, 261))
        self.assertIsNotNone(result.raw_hessian_symmetry_error)
        assert result.raw_hessian_symmetry_error is not None
        self.assertLessEqual(result.raw_hessian_symmetry_error, 1.0e-12)

    def test_validation_uses_training_transforms_and_own_Omega_center(self) -> None:
        first_group = self.cells[0].group
        training_indexes = tuple(
            index for index, cell in enumerate(self.cells) if cell.group == first_group
        )
        validation_indexes = tuple(
            index for index, cell in enumerate(self.cells) if cell.group != first_group
        )
        training = m1.prepare_data(self.cells, self.moments, training_indexes)
        validation_cells = tuple(self.cells[index] for index in validation_indexes)
        validation_moments = tuple(self.moments[index] for index in validation_indexes)
        validation = m1.apply_fixed_transforms(
            validation_cells,
            validation_moments,
            training.design_transform,
            training.geometry_scales,
        )
        np.testing.assert_array_equal(validation.geometry_scales, training.geometry_scales)
        np.testing.assert_array_equal(
            validation.geometry_centers,
            np.asarray([moment.geometry_mean for moment in validation_moments]),
        )
        np.testing.assert_array_equal(
            validation.x,
            m1.build_design(validation_cells, training.design_transform),
        )

    def test_79_unique_rank_sets_cover_all_fold_roles(self) -> None:
        representatives = tuple(
            m1._synthetic_cell(
                index,
                group,
                (index % 15) + 1,
                np.asarray((0.1, -0.2, 0.3)),
                0.5,
            )
            for index, group in enumerate(m1.GROUP_ORDER)
        )
        folds = m1._all_training_folds(representatives)
        self.assertEqual(len(folds), 79)
        self.assertEqual(sum(label.startswith("leave_one__") for label, _ in folds), 12)
        self.assertEqual(sum(label.startswith("leave_two__") for label, _ in folds), 66)
        self.assertEqual(len({indexes for _, indexes in folds}), 79)
        self.assertEqual(2 * 66, 132)
        self.assertEqual(132 + 24 + 1, 157)

    def test_certificate_rejects_ineligible_and_nonfinite_without_state(self) -> None:
        result = m1._certificate_gate()
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["ineligible_pass"])
        self.assertFalse(result["nonfinite_pass"])
        self.assertFalse(result["state_or_history_mutated"])

    def test_frozen_synthetic_rng_blocks(self) -> None:
        result = m1._synthetic_rng_order_gate()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["normal_block_precedes_categorical_uniform_block"])
        self.assertEqual([row["rows"] for row in result["schedules"]], [544, 68])

    def test_synthetic_design_addendum_restores_full_rank_without_rng(self) -> None:
        result = m1._synthetic_design_rank_gate()
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["consumes_random_variates"])
        self.assertEqual(
            [row["rank"] for row in result["schedules"]],
            [m1.DESIGN_DIMENSION, m1.DESIGN_DIMENSION],
        )
        self.assertEqual(
            result["replacement_schedule"],
            "hemisphere_index=(row_index//4)%2",
        )

    def test_zero_gradient_requires_repeated_objective_predicate(self) -> None:
        source = self.cells[:2]
        singleton_cells = (
            m1.Cell(
                source[0].identifier,
                "zero_group",
                source[0].layer,
                source[0].hemisphere,
                source[0].soma_um,
                source[0].soma_distance_mm,
                0,
                0,
                source[0].segment_unmapped,
            ),
            m1.Cell(
                source[1].identifier,
                "full_group",
                source[1].layer,
                source[1].hemisphere,
                source[1].soma_um,
                source[1].soma_distance_mm,
                16,
                0,
                source[1].segment_unmapped,
            ),
        )
        singleton = m1.PreparedData(
            singleton_cells,
            self.prepared.x[:2],
            np.zeros((2, 6)),
            np.ones(6),
            np.asarray((0.5, 0.5)),
            self.prepared.design_transform,
        )
        fit = m1.fit_newton(
            singleton,
            self.library,
            self.geometry,
            0.1,
            initial=np.zeros(m1.PARAMETER_DIMENSION),
        )
        self.assertEqual(
            fit.telemetry["convergence_witness"],
            "gradient_and_repeated_unchanged_initial_objective",
        )
        self.assertEqual(fit.relative_objective_change, 0.0)
        self.assertFalse(
            fit.telemetry["post_witness_confirmation_attempted"]
        )

    def test_population_correction_matches_direct_exhaustive_math(self) -> None:
        indexes = tuple(
            index for index, cell in enumerate(self.cells) if cell.k in (2, 3)
        )[:4]
        prepared = m1.prepare_data(self.cells, self.moments, indexes)
        expected = m1.expected_cell_joint_statistics(
            self.planted, prepared, self.library, self.geometry
        )
        observed = m1.observed_cell_joint_statistics(
            prepared, self.library, self.geometry
        )
        correction = m1.aggregate_population_linear_correction(
            prepared, expected.joint_mean, observed
        )
        one_hot = m1.aggregate_population_linear_correction(
            prepared, observed, observed
        )
        np.testing.assert_array_equal(
            one_hot.vector, np.zeros(m1.PARAMETER_DIMENSION)
        )

        trial = 0.015 * np.sin(
            np.arange(m1.PARAMETER_DIMENSION, dtype=float) + 0.41
        )
        ridge = 0.1
        analytic = m1.evaluate_model(
            trial,
            prepared,
            self.library,
            self.geometry,
            ridge,
            need_hessian=True,
            linear_correction=correction.vector,
        )
        self.assertIsNotNone(analytic.hessian)
        assert analytic.hessian is not None

        direct_objective = 0.0
        direct_gradient = np.zeros(m1.PARAMETER_DIMENSION)
        direct_hessian = np.zeros(
            (m1.PARAMETER_DIMENSION, m1.PARAMETER_DIMENSION)
        )
        for cell_index, cell in enumerate(prepared.cells):
            basis = self.library.basis(cell.k)
            raw = self.geometry.raw_batch(cell, 0, basis.count)
            z = (
                raw - prepared.geometry_centers[cell_index]
            ) / prepared.geometry_scales
            joint = np.column_stack((basis.contrasts, z))
            planted_scores = m1._candidate_score_batch(
                self.planted,
                prepared,
                cell_index,
                0,
                basis.count,
                self.library,
                self.geometry,
            )
            trial_scores = m1._candidate_score_batch(
                trial,
                prepared,
                cell_index,
                0,
                basis.count,
                self.library,
                self.geometry,
            )
            planted_maximum = float(np.max(planted_scores))
            trial_maximum = float(np.max(trial_scores))
            planted_weights = np.exp(planted_scores - planted_maximum)
            trial_weights = np.exp(trial_scores - trial_maximum)
            planted_probabilities = planted_weights / np.sum(planted_weights)
            trial_probabilities = trial_weights / np.sum(trial_weights)
            logz = trial_maximum + math.log(float(np.sum(trial_weights)))
            weight = float(prepared.weights[cell_index])
            direct_objective += weight * (
                logz - float(np.dot(planted_probabilities, trial_scores))
            )
            planted_mean = planted_probabilities @ joint
            trial_mean = trial_probabilities @ joint
            delta = trial_mean - planted_mean
            x_row = prepared.x[cell_index]
            direct_gradient[:255] += weight * np.outer(
                x_row, delta[:15]
            ).ravel()
            direct_gradient[255:] += weight * delta[15:]
            covariance = (
                joint.T @ (trial_probabilities[:, None] * joint)
                - np.outer(trial_mean, trial_mean)
            )
            covariance = 0.5 * (covariance + covariance.T)
            cov_aa = covariance[:15, :15]
            cov_az = covariance[:15, 15:]
            cov_zz = covariance[15:, 15:]
            direct_hessian[:255, :255] += weight * np.einsum(
                "a,b,rs->arbs", x_row, x_row, cov_aa, optimize=True
            ).reshape(255, 255)
            cross = np.einsum(
                "a,rg->arg", x_row, cov_az, optimize=True
            ).reshape(255, 6)
            direct_hessian[:255, 255:] += weight * cross
            direct_hessian[255:, :255] += weight * cross.T
            direct_hessian[255:, 255:] += weight * cov_zz
        direct_objective += 0.5 * ridge * float(np.dot(trial, trial))
        direct_gradient += ridge * trial
        direct_hessian.flat[:: m1.PARAMETER_DIMENSION + 1] += ridge
        np.testing.assert_allclose(
            analytic.objective, direct_objective, atol=2.0e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            analytic.gradient, direct_gradient, atol=2.0e-11, rtol=0.0
        )
        np.testing.assert_allclose(
            analytic.hessian, direct_hessian, atol=2.0e-10, rtol=0.0
        )

        step = 1.0e-6
        for coordinate in (0, 17, 127, 254, 255, 260):
            offset = np.zeros(m1.PARAMETER_DIMENSION)
            offset[coordinate] = step
            plus = m1.evaluate_model(
                trial + offset,
                prepared,
                self.library,
                self.geometry,
                ridge,
                need_hessian=False,
                linear_correction=correction.vector,
            ).objective
            minus = m1.evaluate_model(
                trial - offset,
                prepared,
                self.library,
                self.geometry,
                ridge,
                need_hessian=False,
                linear_correction=correction.vector,
            ).objective
            finite = (plus - minus) / (2.0 * step)
            self.assertAlmostEqual(
                finite, analytic.gradient[coordinate], delta=1.0e-6
            )
        for phase in (0.2, 0.7):
            direction = np.cos(
                np.arange(m1.PARAMETER_DIMENSION, dtype=float) + phase
            )
            direction /= np.linalg.norm(direction)
            hv_step = 1.0e-5
            plus = m1.evaluate_model(
                trial + hv_step * direction,
                prepared,
                self.library,
                self.geometry,
                ridge,
                need_hessian=False,
                linear_correction=correction.vector,
            ).gradient
            minus = m1.evaluate_model(
                trial - hv_step * direction,
                prepared,
                self.library,
                self.geometry,
                ridge,
                need_hessian=False,
                linear_correction=correction.vector,
            ).gradient
            finite_hv = (plus - minus) / (2.0 * hv_step)
            np.testing.assert_allclose(
                finite_hv,
                analytic.hessian @ direction,
                atol=1.0e-5,
                rtol=1.0e-4,
            )

        planted_unpenalized = m1.evaluate_model(
            self.planted,
            prepared,
            self.library,
            self.geometry,
            0.0,
            need_hessian=False,
            linear_correction=correction.vector,
        )
        planted_penalized = m1.evaluate_model(
            self.planted,
            prepared,
            self.library,
            self.geometry,
            m1.RECOVERY_LAMBDA,
            need_hessian=False,
            linear_correction=correction.vector,
        )
        self.assertLessEqual(
            float(np.max(np.abs(planted_unpenalized.gradient))), 1.0e-12
        )
        np.testing.assert_allclose(
            planted_penalized.gradient,
            m1.RECOVERY_LAMBDA * self.planted,
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_population_newton_fit_threads_fixed_correction(self) -> None:
        indexes = tuple(
            index for index, cell in enumerate(self.cells) if cell.k in (2, 3)
        )[:4]
        prepared = m1.prepare_data(self.cells, self.moments, indexes)
        expected = m1.expected_cell_joint_statistics(
            self.planted, prepared, self.library, self.geometry
        )
        observed = m1.observed_cell_joint_statistics(
            prepared, self.library, self.geometry
        )
        correction = m1.aggregate_population_linear_correction(
            prepared, expected.joint_mean, observed
        )
        initial = m1.evaluate_model(
            np.zeros(m1.PARAMETER_DIMENSION),
            prepared,
            self.library,
            self.geometry,
            m1.RECOVERY_LAMBDA,
            need_hessian=False,
            linear_correction=correction.vector,
        )
        fitted = m1.fit_newton(
            prepared,
            self.library,
            self.geometry,
            m1.RECOVERY_LAMBDA,
            initial=None,
            linear_correction=correction.vector,
        )
        self.assertEqual(fitted.telemetry["status"], "pass")
        self.assertLess(fitted.objective, initial.objective)
        self.assertLessEqual(
            fitted.gradient_infinity_norm, m1.GRADIENT_TOLERANCE
        )

    def test_exact_seed_20260928_lambda_1000_cancellation_regression(self) -> None:
        _, _, prepared, library, geometry, _ = m1.generate_synthetic_dataset(
            groups=4, rows_per_group=17, seed=20260928
        )
        fitted = m1.fit_newton(
            prepared,
            library,
            geometry,
            1000.0,
            initial=None,
        )
        self.assertEqual(
            fitted.telemetry["convergence_witness"],
            "gradient_and_relative_objective_change",
        )
        self.assertLessEqual(
            fitted.gradient_infinity_norm, m1.GRADIENT_TOLERANCE
        )
        self.assertLessEqual(
            fitted.relative_objective_change, m1.RELATIVE_TOLERANCE
        )
        self.assertTrue(
            fitted.telemetry["post_witness_confirmation_attempted"]
        )
        self.assertEqual(
            fitted.telemetry["post_witness_confirmation_accepted_steps"], 1
        )
        first_iteration = fitted.telemetry[
            "first_ordinary_convergence_iteration"
        ]
        self.assertIsInstance(first_iteration, int)
        assert isinstance(first_iteration, int)
        self.assertEqual(fitted.iterations, first_iteration + 1)
        records = fitted.telemetry.get("line_records")
        self.assertIsInstance(records, list)
        assert isinstance(records, list)
        self.assertGreaterEqual(len(records), 2)
        for record in records:
            self.assertGreaterEqual(record["armijo_slack"], 0.0)
            self.assertLessEqual(record["curvature_ratio"], m1.CURVATURE_C2)

    def test_full_unit_gate_bundle(self) -> None:
        result = m1.run_unit_gates()
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["candidate_by_261_materialized"])
        self.assertFalse(result["final_or_unassigned_or_SWC_opened"])
        parity = result["gamma_zero_and_fold_scaling"]
        self.assertEqual(
            parity["paired_frozen_M0_source"],
            "outputs/stage0/tools/fit_v3_m0_v7.py",
        )
        self.assertFalse(parity["M0_prediction_or_fit_artifact_opened"])
        population = result["exact_population_objective"]
        self.assertEqual(population["status"], "pass")
        self.assertFalse(population["candidate_by_261_materialized"])


if __name__ == "__main__":
    unittest.main()
