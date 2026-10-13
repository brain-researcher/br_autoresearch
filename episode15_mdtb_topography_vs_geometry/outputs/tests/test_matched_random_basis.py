"""Four bounded algebra/dataflow checks; no empirical or null-power campaign."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, DevelopmentParticipant, FoldContext, RandomBasisPrediction,
    exploratory_points, fit_development_lopo,
)
from local_anatomical_basis import BasisTolerances, _centered_coordinates, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from local_isometry_variants import BlockLocalIsometry, VARIANTS
from matched_random_basis import (
    MASTER_SEED, STREAM_TAG, MatchedRandomLocalBasis, RandomBasisUnresolved,
    _draw_cell, _prepare_cell,
)
from registered_endpoint_metrics import crossvalidated_geometries, map_covariance
import test_development_lopo as lopo_fixture


def center(value):
    return value-value.mean(axis=-1, keepdims=True)


class MatchedRandomBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coordinates = 3*np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        coordinates *= [1., 1.3, 1.7]
        index = np.arange(208.)
        coordinates += .005*np.column_stack((np.sin(index), np.cos(index), np.sin(2*index)))
        cls.tolerances = BasisTolerances(1e-10, 1e-10, 1e-8, 1e-10)
        cls.anatomy = build_local_anatomical_basis(coordinates, np.arange(500, 708), tolerances=cls.tolerances)
        cls.recipe = LocalFunctionalSource(cls.anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
        cls.predictor = MatchedRandomLocalBasis(cls.recipe, cls.anatomy, check=lambda: None)
        rng = np.random.default_rng(15031)
        A = center(rng.standard_normal((29, 208)))
        base = center(rng.standard_normal((18, 208)))
        extra = center(rng.standard_normal((18, 208)))
        partitions = np.stack([(1+q/4)*base+(q*q/20)*extra for q in range(4)])
        records = tuple(DevelopmentParticipant(pid, TASK_A_CONDITIONS, cls.anatomy.voxel_ids, A, partitions)
                        for pid in DEVELOPMENT_IDS)
        cls.context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)
        cls.source = cls.recipe.fit_shared(records[1:], cls.context)
        cls.target = center(cls.source.task_a_maps+.05*np.roll(cls.source.task_a_maps, 1, axis=1))
        cls.numerical = cls.recipe.calibrate_m3(cls.source, cls.target, cls.context)

    def test_random_subspaces_are_distinct_while_a_basis_gauge_rotation_is_vacuous(self):
        B = self.anatomy.B
        R = np.eye(16)
        angle = .27
        for start, stop in self.anatomy.blocks:
            R[start:stop, start:stop] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        gauge = BlockLocalIsometry(replace(self.anatomy, B=B@R), VARIANTS[0])
        fit = gauge.fit(self.source.task_a_maps, self.target, common_gain=self.numerical["gain"])
        original_Q = np.eye(208)+B@(self.numerical["O"]-np.eye(16))@B.T
        gauge_Q = np.eye(208)+gauge.B@(fit["O"]-np.eye(16))@gauge.B.T
        np.testing.assert_allclose(gauge_Q, original_Q, atol=1e-10, rtol=0)
        self.assertEqual(len(self.predictor.operators), 8)
        for draw, operator in enumerate(self.predictor.operators):
            self.assertGreater(np.linalg.norm(operator.B@operator.B.T-B@B.T), 1e-8)
            self.assertEqual(operator.dimension, 16)
            self.assertEqual(operator.continuous_degrees_of_freedom, 8)
            for cell, report in enumerate(self.predictor.cell_diagnostics[draw]):
                self.assertEqual(report["seed_entropy"], (MASTER_SEED, cell, draw, STREAM_TAG))
                self.assertLessEqual(report["maximum_rayleigh_matching_abs"], 1e-10)
                self.assertLessEqual(report["energy_matrix_matching_max_abs"], 1e-10)
                self.assertAlmostEqual(report["trace_increase"], report["expected_trace_increase"], places=10)
                self.assertGreater(report["projector_frobenius_distance"], 1e-8)
                self.assertAlmostEqual(np.sin(report["maximum_principal_angle_radians"]),
                                       report["projector_frobenius_distance"]/np.sqrt(2), places=10)
                self.assertFalse(report["scientific_effectiveness_certified"])
                self.assertFalse(report["degenerate_control"])

    def test_actual_eight_predictions_keep_common_source_gain_and_partition_geometry(self):
        original_A = self.source.task_a_maps.copy()
        original_B = self.source.geometry_state.source_partitions.copy()
        original_K0 = self.source.K0.copy()
        with patch.object(self.recipe, "fit_shared", side_effect=AssertionError("functional source refit")):
            result = self.predictor(self.source, self.target, self.numerical, self.context)
        self.assertEqual(tuple(value.draw_index for value in result), tuple(range(8)))
        np.testing.assert_array_equal(self.source.task_a_maps, original_A)
        np.testing.assert_array_equal(self.source.geometry_state.source_partitions, original_B)
        np.testing.assert_array_equal(self.source.K0, original_K0)
        for prediction, operator in zip(result, self.predictor.operators):
            diagnostics = prediction.diagnostics
            self.assertEqual(diagnostics["common_gain"], self.numerical["gain"])
            self.assertEqual(diagnostics["voxel_ids"], self.source.voxel_ids)
            self.assertTrue(diagnostics["draws_shared_across_participants_and_deletions"])
            self.assertFalse(diagnostics["source_refitted"])
            self.assertFalse(diagnostics["average_roughness_matched"])
            self.assertFalse(diagnostics["M4_output_smoothness_signature_certified"])
            np.testing.assert_allclose(prediction.geometry,
                self.numerical["gain"]**2*original_K0, atol=1e-10, rtol=0)
            certificate = operator.operator_certificate(diagnostics["O"])
            self.assertEqual(certificate["determinant_sign_factors"], 8)
        # Geometry remains the actual cross-partition estimator, not mean-map Gram.
        mean_map_gram = result[0].maps@result[0].maps.T/208
        self.assertGreater(np.linalg.norm(result[0].geometry-mean_map_gram), 1e-4)

    def test_genuine_boundary_tie_passes_but_isolated_or_uncertified_ties_and_bad_context_refuse(self):
        n = 6
        B = _centered_coordinates(n)[:, :2]
        centered_identity = np.eye(n)-np.ones((n, n))/n
        prepared = _prepare_cell(centered_identity, B, cell=0, tolerances=self.tolerances)
        mixed, report = _draw_cell(prepared, 0)
        again, _ = _draw_cell(prepared, 0)
        np.testing.assert_array_equal(mixed, again)
        self.assertTrue(report["numerical_boundary_tie_branch"])
        self.assertFalse(report["exact_eigenvalue_tie_proven"])
        self.assertAlmostEqual(report["squared_sine"], .5)
        self.assertLessEqual(abs(report["trace_increase"]), 1e-10)
        self.assertGreater(report["projector_frobenius_distance"], .9)
        for higher in (1., 5e-10):
            L = centered_identity+higher*(centered_identity-B@B.T)
            candidate = _prepare_cell(L, B, cell=0, tolerances=self.tolerances)
            with self.assertRaisesRegex(RandomBasisUnresolved, "boundary-tied"):
                _draw_cell(candidate, 0)
        with self.assertRaisesRegex(ValueError, "common pre-operator"):
            self.predictor(self.source, self.target, dict(self.numerical, gain=2*self.numerical["gain"]), self.context)
        with self.assertRaisesRegex(ValueError, "actual current-fold fit"):
            self.predictor(replace(self.source, training_ids=()), self.target, self.numerical, self.context)
        def expired():
            raise TimeoutError("allocation expired")
        with self.assertRaisesRegex(TimeoutError, "allocation expired"):
            MatchedRandomLocalBasis(self.recipe, self.anatomy, check=expired)
        with patch("matched_random_basis._draw_cell", side_effect=RandomBasisUnresolved("one draw unresolved", {})):
            with self.assertRaisesRegex(RandomBasisUnresolved, "one draw unresolved"):
                MatchedRandomLocalBasis(self.recipe, self.anatomy, check=lambda: None)

    def test_lopo_averages_draw_statistics_and_keeps_target_b_out_of_prediction(self):
        fixture = lopo_fixture.DevelopmentLOPOTests()
        fixture.setUp()
        calls = []
        def controls(source, target_a, numerical, context):
            calls.append(context)
            self.assertEqual(target_a.shape, (3, 208))
            self.assertNotIn(context.target_id, source.training_ids)
            self.assertNotIn(context.deleted_id, source.training_ids)
            gain = numerical["gain"]
            return tuple(RandomBasisPrediction(draw,
                (1+.15*(draw-3.5))*gain*source.G,
                gain**2*source.K0+(draw-3.5)*gain*np.diag(np.arange(1., 19.)), {})
                for draw in range(8))
        def run(records, callback=controls):
            return fit_development_lopo(records, fit_shared=fixture.shared,
                calibrate_m3=fixture.calibrate, certify_m3=fixture.certificate,
                predict_m3_geometry=fixture.geometry, check=lambda: None,
                include_participant_deletions=False, predict_random_bases=callback).full
        full = run(fixture.records)
        self.assertEqual(len(calls), 12)
        points = exploratory_points(full)
        halves = np.stack([record.task_b_partitions.reshape(2, 2, 18, 208).mean(axis=1)
                           for record in fixture.records])
        maps = np.stack([[fold.random_basis_predictions[draw].maps for fold in full.folds] for draw in range(8)])
        expected = [map_covariance(halves[:, 0]-draw, halves[:, 1]-draw) for draw in maps]
        self.assertAlmostEqual(points["R_map"]["M3_random_local_basis"], np.mean(expected), places=10)
        np.testing.assert_allclose(points["random_basis_draw_residuals"]["R_map"], expected, atol=1e-10)
        wrong = map_covariance(halves[:, 0]-maps.mean(axis=0), halves[:, 1]-maps.mean(axis=0))
        self.assertGreater(abs(np.mean(expected)-wrong), 1e-6)
        changed = list(fixture.records)
        changed[0] = replace(changed[0], task_b_partitions=3*changed[0].task_b_partitions)
        after = run(changed)
        for first, second in zip(full.folds[0].random_basis_predictions, after.folds[0].random_basis_predictions):
            np.testing.assert_array_equal(first.maps, second.maps)
            np.testing.assert_array_equal(first.geometry, second.geometry)
        with self.assertRaisesRegex(ValueError, "eight"):
            run(fixture.records, callback=lambda *args: controls(*args)[:-1])


if __name__ == "__main__":
    unittest.main()
