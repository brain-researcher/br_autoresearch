"""Small M2 algebra/fit/dataflow checks, not empirical or class qualification."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, M2Prediction, M2_VARIANTS, DevelopmentParticipant,
    FoldContext, exploratory_points, fit_development_lopo,
)
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from registered_endpoint_metrics import map_covariance
from smooth_relocation import (
    RelocationTaskAObjective, SmoothRelocationPrediction, fit_relocation_task_a,
)
import test_development_lopo as lopo_fixture


def center(value):
    return value-value.mean(axis=-1, keepdims=True)


class SmoothRelocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coordinates = 3*np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        tolerances = BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                                    rank_rtol=1e-8, numerical_atol=1e-10)
        cls.anatomy = build_local_anatomical_basis(coordinates, np.arange(500, 708), tolerances=tolerances)
        cls.recipe = LocalFunctionalSource(cls.anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
        cls.predictor = SmoothRelocationPrediction(cls.recipe, cls.anatomy)
        cls.representation, cls.field = cls.predictor.representation, cls.predictor.fields[0]
        # One deterministic algebra fixture; no synthetic-cohort program.
        rng = np.random.default_rng(15102)
        cls.A = center(rng.standard_normal((29, 208)))
        base = center(rng.standard_normal((18, 208)))
        extra = center(rng.standard_normal((18, 208)))
        cls.partitions = np.stack([(1+q/4)*base+(q*q/20)*extra for q in range(4)])
        records = tuple(DevelopmentParticipant(pid, TASK_A_CONDITIONS, cls.anatomy.voxel_ids,
                                               cls.A, cls.partitions) for pid in DEVELOPMENT_IDS)
        cls.context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)
        cls.source = cls.recipe.fit_shared(records[1:], cls.context)

    def gain(self, source, target):
        return float(np.sum(center(source)*center(target))/np.sum(center(source)**2))

    def test_exact_nodes_keep_analytic_derivatives_and_distinct_amplification_bounds(self):
        representation = self.representation
        self.assertEqual(representation.bandwidth_mm, 3.)
        W, derivative = representation.weights_and_derivatives(self.anatomy.coordinates)
        np.testing.assert_array_equal(W, np.eye(208))
        self.assertGreater(np.linalg.norm(derivative), .01)
        query = self.anatomy.coordinates[[0, 70, 137]]
        step = np.array([2e-5, -1e-5, 1e-5])
        plus, _ = representation.weights_and_derivatives(query+step)
        minus, _ = representation.weights_and_derivatives(query-step)
        np.testing.assert_allclose((plus-minus)/2, derivative[[0, 70, 137]]@step, atol=2e-9, rtol=1e-5)
        constant, gradient = representation.evaluate(np.full((2, 208), 3.), query+step)
        np.testing.assert_allclose(constant, 3., atol=1e-10)
        np.testing.assert_allclose(gradient, 0., atol=1e-10)
        # Raw interpolation can satisfy cap4 while final centering approaches8.
        raw = np.zeros((208, 208))
        raw[:, :2] = [2.5, -1.5]
        raw[-1, :2] = [-1.5, 2.5]
        certificate = representation.certificate(raw)
        self.assertEqual(certificate["interpolation_amplification"], 4.)
        self.assertGreater(certificate["centered_interpolation_amplification"], 7.)
        self.assertFalse(certificate["off_grid_biological_accuracy_certified"])
        raw[0, :2] = [3., -2.]
        with self.assertRaisesRegex(ValueError, "amplification"):
            representation.certificate(raw)

    def test_global_field_bounds_pull_jacobian_and_inverse(self):
        for field, lower in zip(self.predictor.fields, (np.exp(-.5), np.exp(-.125), np.exp(-2), np.exp(-.5))):
            own = field.native_basis[np.arange(208), self.anatomy.cell_labels]
            self.assertGreaterEqual(own.min()+1e-14, lower)
            T = np.arange(-12., 12.).reshape(8, 3)
            T *= .7*field.radius/np.linalg.norm(T)
            certificate = field.certificate(T)
            points = np.vstack((self.anatomy.coordinates, [[-20., 50., 80.], [500., -300., 80.]]))
            displacement, jacobian = field.displacement(points, T)
            self.assertLessEqual(np.linalg.norm(displacement, axis=1).max(), certificate["global_displacement_upper_bound_mm"])
            self.assertLessEqual(np.linalg.svd(jacobian, compute_uv=False).max(), certificate["global_derivative_norm_upper_bound"])
            self.assertTrue(np.all(np.linalg.det(np.eye(3)+jacobian) > 0))
            inverse, report = field.inverse(points+displacement, T)
            np.testing.assert_allclose(inverse, points, atol=2e-10, rtol=0)
            self.assertLessEqual(report["inverse_error_upper_bound_mm"], 1e-10)
            direction = np.array([.3, -.2, .5])
            plus, _ = field.displacement(points+1e-5*direction, T)
            minus, _ = field.displacement(points-1e-5*direction, T)
            np.testing.assert_allclose((plus-minus)/(2e-5), jacobian@direction, atol=1e-9)
            with self.assertRaisesRegex(ValueError, "global displacement/Lipschitz"):
                field.certificate(2*T)

    def test_objective_and_constraint_derivatives_include_identity_centering_and_gain(self):
        target = center(self.A+.03*np.roll(self.A, 1, axis=1))
        objective = RelocationTaskAObjective(self.representation, self.field, self.A, target,
                                            common_gain=self.gain(self.A, target))
        direction = np.cos(np.arange(24.)+.3)
        direction /= np.linalg.norm(direction)
        for z in (np.zeros(24), .07*np.sin(np.arange(24.)+.7)):
            value, gradient = objective.value_gradient(z)
            epsilon = 2e-6
            plus = objective.evaluation(z+epsilon*direction)
            minus = objective.evaluation(z-epsilon*direction)
            self.assertAlmostEqual((plus["value"]-minus["value"])/(2*epsilon), gradient@direction, places=7)
            self.assertTrue(np.isfinite(value))
            if np.any(z):
                exact = objective.evaluation(z)["interpolation_jacobian"]@direction
                observed = (plus["interpolation_constraints"]-minus["interpolation_constraints"])/(2*epsilon)
                np.testing.assert_allclose(observed, exact, atol=2e-7, rtol=2e-5)
        self.assertGreater(np.linalg.norm(objective.value_gradient(np.zeros(24))[1]), 1e-6)
        scaled = RelocationTaskAObjective(self.representation, self.field, self.A, 3*target,
                                         common_gain=3*self.gain(self.A, target))
        first, second = objective.value_gradient(direction*.1), scaled.value_gradient(direction*.1)
        self.assertAlmostEqual(first[0], second[0], places=13)
        np.testing.assert_allclose(first[1], second[1], atol=1e-12)

    def test_four_partition_prediction_uses_pull_and_exact_zero_reduction(self):
        zero = self.predictor.predict_partitions(self.field, self.partitions, np.zeros((8, 3)), 2.)
        np.testing.assert_array_equal(zero, center(2*center(self.partitions)))
        T = np.zeros((8, 3))
        T[0] = [.03, -.02, .04]
        actual = self.predictor.predict_partitions(self.field, self.partitions, T, 1.7)
        displacement, _ = self.field.displacement(self.anatomy.coordinates, T)
        W, _ = self.representation.weights_and_derivatives(self.anatomy.coordinates+displacement)
        expected = center(1.7*(center(self.partitions)@W.T))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        wrong, _ = self.representation.weights_and_derivatives(self.anatomy.coordinates-displacement)
        self.assertGreater(np.linalg.norm(actual-center(1.7*(center(self.partitions)@wrong.T))), .01)
        geometry = sum(actual[a]@actual[b].T+actual[b]@actual[a].T
                       for a, b in ((0, 1), (2, 3)))/(4*208)
        from registered_endpoint_metrics import crossvalidated_geometries
        np.testing.assert_allclose(crossvalidated_geometries(actual[None])[0].mean(axis=0), geometry, atol=1e-12)
        mean = actual.mean(axis=0)
        self.assertGreater(np.linalg.norm(geometry-mean@mean.T/208), 1e-4)

    def test_all_four_callback_variants_share_gain_source_and_unverified_claim_flags(self):
        predictions = self.predictor(self.source, 2*self.source.task_a_maps, 2., self.context)
        self.assertEqual(tuple(value.name for value in predictions), M2_VARIANTS)
        for value in predictions:
            diagnostics = value.diagnostics
            np.testing.assert_allclose(diagnostics["coefficients"], 0., atol=1e-8)
            np.testing.assert_allclose(value.maps, 2*self.source.G, atol=1e-10)
            np.testing.assert_allclose(value.geometry, 4*self.source.K0, atol=1e-10)
            self.assertEqual(diagnostics["common_gain"], 2.)
            self.assertEqual(diagnostics["training_ids"], self.context.training_ids)
            self.assertFalse(diagnostics["global_solution_certified"])
            self.assertFalse(diagnostics["general_data_identifiability_certified"])
            self.assertFalse(diagnostics["native_signature_certified"])
        with self.assertRaisesRegex(ValueError, "common untransformed-source"):
            self.predictor(self.source, self.source.task_a_maps, 1.5, self.context)
        # A rank diagnostic must not become another execution gate.
        with patch("smooth_relocation.RelocationTaskAObjective.tangent_singular_values", return_value=np.zeros(24)):
            result = fit_relocation_task_a(self.representation, self.field, self.A, self.A, common_gain=1.)
        self.assertEqual(result["local_tangent_rank_diagnostic"], 0)
        self.assertFalse(result["local_tangent_full_rank_diagnostic"])

    def test_nonzero_task_a_fit_improves_identity_and_failed_solver_refuses(self):
        T = np.zeros((8, 3))
        T[2] = [.05, -.04, .02]
        displacement, _ = self.field.displacement(self.anatomy.coordinates, T)
        target, _ = self.representation.evaluate(self.A, self.anatomy.coordinates+displacement)
        gain = self.gain(self.A, target)
        result = fit_relocation_task_a(self.representation, self.field, self.A, target, common_gain=gain)
        self.assertLess(result["normalized_penalized_task_a_loss"], .99*result["identity_task_a_loss"])
        self.assertGreater(np.linalg.norm(result["coefficients"]), 1e-5)
        self.assertFalse(result["global_solution_certified"])
        with patch("smooth_relocation.minimize", return_value=SimpleNamespace(success=False)):
            with self.assertRaisesRegex(ArithmeticError, "SLSQP failed"):
                fit_relocation_task_a(self.representation, self.field, self.A, target, common_gain=gain)

    def test_lopo_optional_m2_keeps_all_nested_contexts_variants_and_target_b_exclusion(self):
        fixture = lopo_fixture.DevelopmentLOPOTests()
        fixture.setUp()
        calls = []
        # Orchestration-only callback; actual four-variant fitting is checked
        # above. Its source K0 comes from the fixture's actual four B maps.
        def predict(source, target_a, gain, context):
            calls.append(context)
            self.assertNotIn(context.target_id, source.training_ids)
            self.assertNotIn(context.deleted_id, source.training_ids)
            self.assertEqual(target_a.shape, (3, 208))
            return tuple(M2Prediction(name, (1+j/4)*gain*source.G,
                                       ((1+j/4)*gain)**2*source.K0, {})
                         for j, name in enumerate(M2_VARIANTS))
        def run(records, nested):
            return fit_development_lopo(records, fit_shared=fixture.shared, calibrate_m3=fixture.calibrate,
                certify_m3=fixture.certificate, predict_m3_geometry=fixture.geometry, predict_m2=predict,
                check=lambda: None, include_participant_deletions=nested)
        result = run(fixture.records, True)
        self.assertEqual(len(calls), 144)
        for item in (result.full, *result.deletions):
            for fold in item.folds:
                self.assertEqual(tuple(value.name for value in fold.m2_predictions), M2_VARIANTS)
                self.assertEqual(fold.m1_predictions, ())
        points = exploratory_points(result.full)
        observed = np.stack([fold.observed_partitions for fold in result.full.folds])
        halves = observed.reshape(12, 2, 2, 18, 208).mean(axis=2)
        for j, name in enumerate(M2_VARIANTS):
            maps = np.stack([fold.m2_predictions[j].maps for fold in result.full.folds])
            self.assertAlmostEqual(points["R_map"][name], map_covariance(halves[:, 0]-maps, halves[:, 1]-maps), places=12)
        changed = list(fixture.records)
        changed[0] = replace(changed[0], task_b_partitions=3*changed[0].task_b_partitions)
        after = run(changed, False).full
        for first, second in zip(result.full.folds[0].m2_predictions, after.folds[0].m2_predictions):
            np.testing.assert_array_equal(first.maps, second.maps)
            np.testing.assert_array_equal(first.geometry, second.geometry)


if __name__ == "__main__":
    unittest.main()
