"""Small M3 operator/source/dataflow checks, not empirical class qualification."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    M3Prediction, M3_VARIANTS, exploratory_points, fit_development_lopo,
)
from local_isometry_variants import LocalIsometryVariants
from registered_endpoint_metrics import crossvalidated_geometries, map_covariance
from spatial_deformation_operators import apply_m3
import test_development_lopo as lopo_fixture
import test_smooth_relocation as relocation_fixture


def center(value):
    return value-value.mean(axis=-1, keepdims=True)


class LocalIsometryVariantsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = relocation_fixture.SmoothRelocationTests
        if not hasattr(fixture, "predictor"):
            fixture.setUpClass()
        cls.fixture = fixture
        cls.predictor = LocalIsometryVariants(fixture.recipe, fixture.anatomy, check=lambda: None)

    def test_actual_four_operator_fits_have_selected_dimensions_locality_and_geometry(self):
        self.assertEqual(tuple(op.dimension for op in self.predictor.operators), (16, 16, 16, 24))
        self.assertEqual(tuple(op.continuous_degrees_of_freedom for op in self.predictor.operators), (8, 0, 24, 24))
        reference, tight, broad, stress = self.predictor.operators
        self.assertLessEqual(tight.diameter, reference.diameter)
        self.assertLessEqual(reference.diameter, broad.diameter)
        self.assertEqual(stress.diameter, reference.diameter)
        for operator in self.predictor.operators:
            expected = np.eye(operator.dimension)
            if operator.variant.modes_per_cell == 1:
                expected[0, 0] = -1
            else:
                angle = .31
                expected[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]]
            target = 1.7*apply_m3(self.fixture.A, operator.B, expected)
            fitted = operator.fit(self.fixture.A, target, common_gain=1.7)
            np.testing.assert_allclose(fitted["O"], expected, atol=1e-10, rtol=0)
            self.assertTrue(fitted["unique_within_fixed_block_family"])
            self.assertEqual(len(fitted["source_design_singular_values"]), operator.dimension)
            prediction = 1.7*apply_m3(self.fixture.partitions.reshape(72, 208),
                                      operator.B, fitted["O"]).reshape(4, 18, 208)
            geometry = crossvalidated_geometries(prediction[None])[0]
            expected_geometry = 1.7**2*crossvalidated_geometries(self.fixture.partitions[None])[0]
            np.testing.assert_allclose(geometry, expected_geometry, atol=1e-10, rtol=0)
            self.assertEqual(fitted["operator_certificate"]["max_transport_distance_bound_mm"], operator.diameter)

    def test_tight_variant_is_sixteen_independent_identified_sign_factors(self):
        tight = self.predictor.operators[1]
        signs = np.where(np.arange(16) % 2, -1., 1.)
        target = apply_m3(self.fixture.A, tight.B, np.diag(signs))
        fitted = tight.fit(self.fixture.A, target, common_gain=1.)
        np.testing.assert_allclose(fitted["O"], np.diag(signs), atol=1e-12, rtol=0)
        certificate = fitted["operator_certificate"]
        self.assertEqual(certificate["continuous_degrees_of_freedom"], 0)
        self.assertEqual(certificate["determinant_sign_factors"], 16)
        self.assertEqual(tuple(len(values) for values in fitted["local_cross_singular_values"]), (1,)*16)
        self.assertLessEqual(certificate["Q_orthogonality_max_abs"], 1e-10)

    def test_rank_locality_and_nonorthogonal_failures_do_not_drop_modes(self):
        for operator in self.predictor.operators:
            with self.assertRaisesRegex(ValueError, "global Task-A active source design"):
                operator.fit(np.zeros((29, 208)), self.fixture.A, common_gain=1.)
            b0 = operator.B[:, 0]
            missing_direction = self.fixture.A-(self.fixture.A@b0)[:, None]*b0
            with self.assertRaisesRegex(ValueError, "local Task-A cross-product"):
                operator.fit(self.fixture.A, missing_direction, common_gain=1.)
            O = np.eye(operator.dimension)
            first = 0
            second = operator.blocks[1][0]
            angle = .2
            O[np.ix_([first, second], [first, second])] = [[np.cos(angle), -np.sin(angle)],
                                                        [np.sin(angle), np.cos(angle)]]
            with self.assertRaisesRegex(ValueError, "algebra/locality"):
                operator.operator_certificate(O)
            with self.assertRaisesRegex(ValueError, "algebra/locality"):
                operator.operator_certificate(1.01*np.eye(operator.dimension))

    def test_callback_reuses_reference_fit_source_gain_and_actual_partition_geometry(self):
        fixture = self.fixture
        target = 2*fixture.source.task_a_maps
        numerical = fixture.recipe.calibrate_m3(fixture.source, target, fixture.context)
        original_a = fixture.source.task_a_maps.copy()
        original_b = fixture.source.geometry_state.source_partitions.copy()
        original_k0 = fixture.source.K0.copy()
        with patch.object(fixture.recipe, "fit_shared", side_effect=AssertionError("source refit")), \
                patch.object(self.predictor.operators[0], "fit", side_effect=AssertionError("reference refit")):
            result = self.predictor(fixture.source, target, numerical, fixture.context)
        self.assertEqual(tuple(value.name for value in result), M3_VARIANTS)
        expected_map = numerical["gain"]*apply_m3(center(fixture.source.G), fixture.source.B, numerical["O"])
        expected_geometry = fixture.recipe.predict_m3_geometry(fixture.source, numerical, fixture.context)
        np.testing.assert_array_equal(result[0].maps, expected_map)
        np.testing.assert_array_equal(result[0].geometry, expected_geometry)
        np.testing.assert_array_equal(fixture.source.task_a_maps, original_a)
        np.testing.assert_array_equal(fixture.source.geometry_state.source_partitions, original_b)
        np.testing.assert_array_equal(fixture.source.K0, original_k0)
        for index, value in enumerate(result):
            self.assertEqual(value.diagnostics["common_gain"], numerical["gain"])
            self.assertEqual(value.diagnostics["voxel_ids"], fixture.source.voxel_ids)
            self.assertEqual(value.diagnostics["reference_fit_reused"], index == 0)
            self.assertFalse(value.diagnostics["source_refitted_for_variant"])
            self.assertFalse(value.diagnostics["native_signature_certified"])
            np.testing.assert_array_equal(value.diagnostics["source_basis"], fixture.source.B)
            np.testing.assert_allclose(value.geometry, numerical["gain"]**2*original_k0, atol=1e-10, rtol=0)
        wrong_gain = dict(numerical, gain=3.)
        with self.assertRaisesRegex(ValueError, "common pre-operator Task-A gain"):
            self.predictor(fixture.source, target, wrong_gain, fixture.context)
        with self.assertRaisesRegex(ValueError, "actual current-fold fit"):
            self.predictor(replace(fixture.source, training_ids=()), target, numerical, fixture.context)

    def test_lopo_hook_preserves_reference_summary_and_target_b_exclusion(self):
        fixture = lopo_fixture.DevelopmentLOPOTests()
        fixture.setUp()
        calls = []
        def variants(source, target_a, numerical, context):
            calls.append(context)
            self.assertEqual(target_a.shape, (3, 208))
            self.assertNotIn(context.target_id, source.training_ids)
            self.assertNotIn(context.deleted_id, source.training_ids)
            reference = numerical["gain"]*apply_m3(center(source.G), source.B, numerical["O"])
            geometry = fixture.geometry(source, numerical, context)
            return tuple(M3Prediction(name, (1+index/5)*reference,
                                      (1+index/5)**2*geometry, {})
                         for index, name in enumerate(M3_VARIANTS))
        def run(records, callback=variants):
            return fit_development_lopo(records, fit_shared=fixture.shared,
                calibrate_m3=fixture.calibrate, certify_m3=fixture.certificate,
                predict_m3_geometry=fixture.geometry, check=lambda: None,
                include_participant_deletions=False, predict_m3_variants=callback).full
        full = run(fixture.records)
        self.assertEqual(len(calls), 12)
        points = exploratory_points(full)
        self.assertAlmostEqual(points["R_map"][M3_VARIANTS[0]], points["R_map"]["M3"], places=12)
        self.assertAlmostEqual(points["R_geometry"][M3_VARIANTS[0]], points["R_geometry"]["M3"], places=12)
        halves = np.stack([record.task_b_partitions.reshape(2, 2, 18, 208).mean(axis=1)
                           for record in fixture.records])
        for index, name in enumerate(M3_VARIANTS):
            maps = np.stack([fold.m3_variant_predictions[index].maps for fold in full.folds])
            expected = map_covariance(halves[:, 0]-maps, halves[:, 1]-maps)
            self.assertAlmostEqual(points["R_map"][name], expected, places=10)
        changed = list(fixture.records)
        changed[0] = replace(changed[0], task_b_partitions=-5*changed[0].task_b_partitions)
        after = run(changed)
        for before_value, after_value in zip(full.folds[0].m3_variant_predictions,
                                             after.folds[0].m3_variant_predictions):
            np.testing.assert_array_equal(before_value.maps, after_value.maps)
            np.testing.assert_array_equal(before_value.geometry, after_value.geometry)
        with self.assertRaisesRegex(ValueError, "four.*M3"):
            run(fixture.records, callback=lambda *args: variants(*args)[:-1])
        def wrong_reference(*args):
            result = list(variants(*args))
            result[0] = replace(result[0], maps=2*result[0].maps)
            return result
        with self.assertRaisesRegex(ValueError, "reference"):
            run(fixture.records, callback=wrong_reference)


if __name__ == "__main__":
    unittest.main()
