"""Focused M2 control checks; no nonlinear refits or empirical qualification."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, M2Prediction, M2_VARIANTS, MatchedWarpPrediction,
    exploratory_points, fit_development_lopo,
)
from matched_random_warp import (
    DRAW_COUNT, MASTER_SEED, MAX_PROPOSALS_PER_DRAW, MatchedRandomField,
    MatchedRandomWarpPrediction, _haar_rotation, matched_random_fields,
)
from registered_endpoint_metrics import (
    crossvalidated_geometries, geometry_covariance, map_covariance, project_gain_direction,
)
import test_development_lopo as lopo_fixture
import test_smooth_relocation as relocation_fixture


class MatchedRandomWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing = relocation_fixture.SmoothRelocationTests
        if not hasattr(existing, "predictor"):
            existing.setUpClass()
        cls.fixture = existing
        cls.predictor = existing.predictor
        cls.field, cls.representation = cls.predictor.fields[0], cls.predictor.representation
        cls.T = np.arange(-12., 12.).reshape(8, 3)
        cls.T *= .04*cls.field.radius/np.linalg.norm(cls.T)

    def predictions(self):
        return tuple(M2Prediction(field.variant.name, self.fixture.source.G,
            self.fixture.source.K0, {"coefficients": self.T.copy(), "common_gain": 1.,
                "training_ids": self.fixture.context.training_ids, "variant": field.variant.name})
            for field in self.predictor.fields)

    def test_actual_draws_preserve_global_field_geometry_and_fixed_streams(self):
        checks = []
        global_before = np.random.get_state()
        draws = matched_random_fields(self.representation, self.field, self.T,
            participant_id=DEVELOPMENT_IDS[0], check=lambda: checks.append(True))
        self.assertEqual(len(draws), DRAW_COUNT)
        self.assertGreaterEqual(len(checks), DRAW_COUNT+2)
        after = np.random.get_state()
        self.assertEqual(global_before[0], after[0])
        np.testing.assert_array_equal(global_before[1], after[1])
        self.assertEqual(global_before[2:], after[2:])
        points = np.vstack((self.field.coordinates[[0, 73, 207]], [[-100., 30., 70.]]))
        parent_u, parent_j = self.field.displacement(points, self.T)
        parent_singular = np.linalg.svd(parent_j, compute_uv=False)
        for index, draw in enumerate(draws):
            self.assertEqual(draw.draw_index, index)
            self.assertEqual(draw.diagnostics["seed_entropy"], (MASTER_SEED, 8, 0, index))
            self.assertFalse(draw.diagnostics["degenerate_identity_control"])
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((MASTER_SEED, 8, 0, index))))
            for _ in range(draw.diagnostics["accepted_proposal"]):
                expected = _haar_rotation(rng)
            np.testing.assert_array_equal(draw.rotation, expected)
            np.testing.assert_allclose(draw.coefficients, self.T@expected, atol=0, rtol=0)
            actual_u, actual_j = self.field.displacement(points, draw.coefficients)
            np.testing.assert_allclose(actual_u@actual_u.T, parent_u@parent_u.T, atol=1e-14)
            np.testing.assert_allclose(np.linalg.svd(actual_j, compute_uv=False), parent_singular, atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(draw.coefficients), np.linalg.norm(self.T), places=14)
            self.assertFalse(draw.coefficients.flags.writeable)
            self.assertFalse(draw.diagnostics["condition_geometry_preserved"])
            self.assertFalse(draw.diagnostics["exact_phi_jacobian_determinant_matched"])

    def test_geometry_rejection_is_bounded_and_never_shrinks_or_returns_partial(self):
        seen = []
        def reject(representation, field, coefficients):
            seen.append(coefficients.copy())
            if len(seen) > 1:  # parent admission must succeed
                raise ValueError("interpolation cap")
            return {}
        with patch("matched_random_warp._geometry", side_effect=reject):
            with self.assertRaisesRegex(ArithmeticError, "no admissible proposal among 32"):
                matched_random_fields(self.representation, self.field, self.T,
                    participant_id=DEVELOPMENT_IDS[0], check=lambda: None)
        self.assertEqual(len(seen), 1+MAX_PROPOSALS_PER_DRAW)
        for coefficients in seen:
            self.assertAlmostEqual(np.linalg.norm(coefficients), np.linalg.norm(self.T), places=14)
        calls = []
        def reject_first_two(representation, field, coefficients):
            calls.append(True)
            if len(calls) in (2, 3):
                raise ArithmeticError("inverse certificate")
            return {}
        with patch("matched_random_warp._geometry", side_effect=reject_first_two):
            draws = matched_random_fields(self.representation, self.field, self.T,
                participant_id=DEVELOPMENT_IDS[0], check=lambda: None)
        self.assertEqual(len(draws), 8)
        self.assertEqual(draws[0].diagnostics["accepted_proposal"], 3)
        self.assertTrue(all(draw.diagnostics["accepted_proposal"] == 1 for draw in draws[1:]))

    def test_identity_is_explicitly_degenerate_with_no_false_native_pass(self):
        zero = np.zeros((8, 3))
        draws = matched_random_fields(self.representation, self.field, zero,
            participant_id=DEVELOPMENT_IDS[0], check=lambda: None)
        for draw in draws:
            np.testing.assert_array_equal(draw.coefficients, zero)
            self.assertTrue(draw.diagnostics["degenerate_identity_control"])
            self.assertFalse(draw.diagnostics["native_signature_certified"])
            self.assertEqual(draw.diagnostics["accepted_proposal"], 1)
            predicted = self.predictor.predict_partitions(self.field,
                self.fixture.partitions, draw.coefficients, 1.)
            expected = self.predictor.predict_partitions(self.field,
                self.fixture.partitions, zero, 1.)
            np.testing.assert_array_equal(predicted, expected)

    def test_four_variant_adapter_keeps_draws_gain_and_cross_partition_geometry(self):
        recorded = []
        def fields(representation, field, coefficients, *, participant_id, check):
            self.assertEqual(participant_id, self.fixture.context.target_id)
            np.testing.assert_array_equal(coefficients, self.T)
            self.assertIs(representation, self.representation)
            return tuple(MatchedRandomField(field.variant.name, index,
                (index+1)*self.T, np.eye(3), {}) for index in range(DRAW_COUNT))
        def predict(field, source_partitions, coefficients, common_gain):
            recorded.append((field, coefficients.copy(), common_gain))
            self.assertIs(source_partitions, self.fixture.source.geometry_state.source_partitions)
            # Numerical pull evaluation already has focused M2 coverage. This
            # adapter check uses unequal partitions and draw values to expose
            # mean-map geometry or premature draw averaging.
            index = len(recorded)-1
            return (1+index/20)*source_partitions
        with patch("matched_random_warp.matched_random_fields", side_effect=fields), \
                patch.object(self.predictor, "predict_partitions", side_effect=predict):
            result = MatchedRandomWarpPrediction(self.predictor, check=lambda: None)(
                self.fixture.source, self.predictions(), 1., self.fixture.context)
        self.assertEqual(len(result), 32)
        self.assertEqual(len(recorded), 32)
        for index, prediction in enumerate(result):
            self.assertEqual(prediction.maps.shape, (18, 208))
            self.assertEqual(prediction.geometry.shape, (18, 18))
            self.assertEqual(prediction.variant, self.predictor.fields[index//8].variant.name)
            self.assertEqual(prediction.draw_index, index % 8)
            self.assertEqual(prediction.diagnostics["aggregation"], "mean_of_per_draw_statistics")
            self.assertFalse(prediction.diagnostics["native_signature_certified"])
            partitions = (1+index/20)*self.fixture.source.geometry_state.source_partitions
            np.testing.assert_allclose(prediction.maps, partitions.mean(axis=0), atol=1e-12)
            expected = crossvalidated_geometries(partitions[None])[0].mean(axis=0)
            np.testing.assert_allclose(prediction.geometry, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(result[0].maps-result[7].maps), .01)

        # Cheap full LOPO callbacks exercise the new hook/aggregation without
        # fitting 48 nonlinear M2 models or generating any additional fields.
        fixture = lopo_fixture.DevelopmentLOPOTests()
        fixture.setUp()
        calls = []
        def m2(source, task_a, gain, context):
            return tuple(M2Prediction(name, gain*source.G, gain**2*source.K0, {})
                         for name in M2_VARIANTS)
        def controls(source, existing, gain, context):
            calls.append(context)
            self.assertNotIn(context.target_id, source.training_ids)
            self.assertNotIn(context.deleted_id, source.training_ids)
            self.assertEqual(tuple(value.name for value in existing), M2_VARIANTS)
            pattern = np.diag(np.arange(1., 19.))
            return tuple(MatchedWarpPrediction(name, draw,
                gain*(1+.15*(draw-3.5))*source.G,
                gain**2*source.K0+gain*(draw-3.5)*pattern, {})
                for name in M2_VARIANTS for draw in range(8))
        def run(records, callback=controls):
            return fit_development_lopo(records, fit_shared=fixture.shared,
                calibrate_m3=fixture.calibrate, certify_m3=fixture.certificate,
                predict_m3_geometry=fixture.geometry, check=lambda: None,
                include_participant_deletions=False, predict_m2=m2,
                predict_matched_warps=callback).full
        full = run(fixture.records)
        points = exploratory_points(full)
        self.assertEqual(len(calls), 12)
        halves = np.stack([record.task_b_partitions.reshape(2, 2, 18, 208).mean(axis=1)
                           for record in fixture.records])
        observed_geometry = crossvalidated_geometries(np.stack([record.task_b_partitions for record in fixture.records]))
        k0 = np.stack([fold.K0 for fold in full.folds])
        all_maps, all_geometry = [], []
        expected_map, expected_geometry = [], []
        for draw in range(8):
            maps = np.stack([fold.matched_warp_predictions[draw].maps for fold in full.folds])
            geometry = np.stack([fold.matched_warp_predictions[draw].geometry for fold in full.folds])
            all_maps.append(maps)
            all_geometry.append(geometry)
            expected_map.append(map_covariance(halves[:, 0]-maps, halves[:, 1]-maps))
            residual = project_gain_direction(observed_geometry-geometry[:, None], k0[:, None])
            expected_geometry.append(geometry_covariance(residual[:, 0], residual[:, 1]))
        for variant in M2_VARIANTS:
            self.assertAlmostEqual(points["R_map"][variant+"_matched_random_warp"], np.mean(expected_map), places=10)
            self.assertAlmostEqual(points["R_geometry"][variant+"_matched_random_warp"], np.mean(expected_geometry), places=10)
            np.testing.assert_allclose(points["matched_warp_draw_residuals"][variant]["R_map"], expected_map, atol=1e-10)
            np.testing.assert_allclose(points["matched_warp_draw_residuals"][variant]["R_geometry"], expected_geometry, atol=1e-10)
        mean_maps = np.mean(all_maps, axis=0)
        wrong_map = map_covariance(halves[:, 0]-mean_maps, halves[:, 1]-mean_maps)
        self.assertGreater(abs(np.mean(expected_map)-wrong_map), 1e-6)
        mean_geometry = np.mean(all_geometry, axis=0)
        wrong_residual = project_gain_direction(observed_geometry-mean_geometry[:, None], k0[:, None])
        wrong_geometry = geometry_covariance(wrong_residual[:, 0], wrong_residual[:, 1])
        self.assertGreater(abs(np.mean(expected_geometry)-wrong_geometry), 1e-6)
        changed = list(fixture.records)
        changed[0] = replace(changed[0], task_b_partitions=2*changed[0].task_b_partitions)
        after = run(changed)
        for before_value, after_value in zip(full.folds[0].matched_warp_predictions,
                                             after.folds[0].matched_warp_predictions):
            np.testing.assert_array_equal(before_value.maps, after_value.maps)
            np.testing.assert_array_equal(before_value.geometry, after_value.geometry)
        with self.assertRaisesRegex(ValueError, "eight matched-warp draws"):
            run(fixture.records, callback=lambda *args: controls(*args)[:-1])

    def test_wrong_roles_fold_gain_partial_variants_and_expired_budget_refuse(self):
        control = MatchedRandomWarpPrediction(self.predictor, check=lambda: None)
        predictions = self.predictions()
        with self.assertRaisesRegex(ValueError, "all four"):
            control(self.fixture.source, predictions[:3], 1., self.fixture.context)
        with self.assertRaisesRegex(ValueError, "exact fold, variant and common gain"):
            control(self.fixture.source, predictions, 2., self.fixture.context)
        changed = list(predictions)
        changed[0] = replace(changed[0], diagnostics=dict(changed[0].diagnostics, training_ids=()))
        with self.assertRaisesRegex(ValueError, "exact fold, variant and common gain"):
            control(self.fixture.source, changed, 1., self.fixture.context)
        with self.assertRaisesRegex(ValueError, "assigned development"):
            matched_random_fields(self.representation, self.field, self.T,
                participant_id="sub-01", check=lambda: None)
        with self.assertRaisesRegex(ValueError, "real array"):
            matched_random_fields(self.representation, self.field, self.T.astype(complex),
                participant_id=DEVELOPMENT_IDS[0], check=lambda: None)
        def expired():
            raise TimeoutError("allocation expired")
        with self.assertRaisesRegex(TimeoutError, "allocation expired"):
            matched_random_fields(self.representation, self.field, self.T,
                participant_id=DEVELOPMENT_IDS[0], check=expired)


if __name__ == "__main__":
    unittest.main()
