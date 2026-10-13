"""LOPO dependency and geometry tests; no empirical reader or qualification."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, DevelopmentParticipant, SharedM3Source,
    exploratory_points, fit_development_lopo,
)
from registered_endpoint_metrics import (
    CONDITIONS, condition_weights, crossvalidated_geometries, geometry_weights, map_covariance,
)
from spatial_deformation_operators import apply_m3
from fit_m3_task_a import fit_m3_task_a


class DevelopmentLOPOTests(unittest.TestCase):
    def setUp(self):
        self.voxels = tuple(f"location-{i:03d}" for i in range(208))
        self.conditions_a = ("A1", "A2", "A3")
        self.B = np.zeros((208, 2))
        self.B[:4] = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])/np.sqrt(2.)
        self.A = np.array([[1., 0.], [0., 2.], [1., 1.]])@self.B.T
        self.pattern = np.column_stack((np.arange(1., 19.), np.arange(18., 0., -1.)))@self.B.T
        self.records = tuple(DevelopmentParticipant(
            participant, self.conditions_a, self.voxels, (1+index/10)*self.A,
            np.stack([np.column_stack((
                np.arange(1., 19.)*(1+index/8+quarter/13),
                np.arange(18., 0., -1.)*(1+index**2/100+quarter**2/11),
            ))@self.B.T for quarter in range(4)]),
        ) for index, participant in enumerate(DEVELOPMENT_IDS))
        self.calls = []
        self.calibrations = []
        self.certificates = []

    def shared(self, training, context):
        self.calls.append((tuple(item.participant_id for item in training), context))
        # Intentionally training-dependent, so each deletion changes source
        # estimates and merely deleting fitted rows cannot reproduce the run.
        partitions = np.mean([item.task_b_partitions for item in training], axis=0)
        G = partitions.mean(axis=0)
        K0 = crossvalidated_geometries(partitions[None])[0].mean(axis=0)
        halves = partitions.reshape(2, 2, 18, 208).mean(axis=1)
        scale = np.sum(condition_weights()*np.mean(halves[0]*halves[1], axis=-1))
        return SharedM3Source(
            context.training_ids, self.conditions_a, self.voxels, self.A,
            G, K0, scale, self.B, np.ones(3), True, 1e-10, 1e-12, partitions,
        )

    def geometry(self, source, numerical, context):
        transformed = np.stack([
            numerical["gain"]*apply_m3(partition, source.B, numerical["O"])
            for partition in source.geometry_state
        ])
        return crossvalidated_geometries(transformed[None])[0].mean(axis=0)

    def calibrate(self, source, target_task_a, context):
        self.calibrations.append(context)
        self.assertIsInstance(target_task_a, np.ndarray)
        self.assertEqual(target_task_a.shape, (3, 208))
        return fit_m3_task_a(
            source.task_a_maps, target_task_a, source.B,
            condition_weights=source.task_a_weights, fit_global_gain=source.fit_global_gain,
            rank_rtol=source.rank_rtol, numerical_atol=source.numerical_atol,
        )

    def certificate(self, source, numerical, context):
        # This fixture's only allowed operator is the identity on its fixed
        # local basis. It is not an empirical locality certificate.
        np.testing.assert_allclose(numerical["O"], np.eye(2), atol=1e-12, rtol=0)
        self.certificates.append(context)

    def run_fits(self, records=None, nested=False, shared=None, certificate=None, geometry=None):
        return fit_development_lopo(
            self.records if records is None else records,
            fit_shared=self.shared if shared is None else shared,
            calibrate_m3=self.calibrate,
            certify_m3=self.certificate if certificate is None else certificate,
            # Explicit split-source fixture; not an empirical recipe selection.
            predict_m3_geometry=self.geometry if geometry is None else geometry,
            check=lambda: None, include_participant_deletions=nested,
        )

    def test_all_nested_sources_refit_with_exact_target_and_deletion_exclusion(self):
        result = self.run_fits(nested=True)
        self.assertEqual(len(self.calls), 144)
        self.assertEqual(len(self.calibrations), 144)
        self.assertEqual(len(self.certificates), 144)
        self.assertEqual(result.full.participant_ids, DEVELOPMENT_IDS)
        self.assertEqual(tuple(run.deleted_id for run in result.deletions), DEVELOPMENT_IDS)
        for training, context in self.calls:
            self.assertEqual(training, context.training_ids)
            self.assertNotIn(context.target_id, training)
            self.assertNotIn(context.deleted_id, training)
            expected = tuple(item for item in DEVELOPMENT_IDS
                             if item not in (context.target_id, context.deleted_id))
            self.assertEqual(training, expected)
            self.assertEqual(len(training), 11 if context.deleted_id is None else 10)
        for index, fold in enumerate(result.full.folds):
            self.assertAlmostEqual(fold.gain, 1+index/10)
        # Target sub-12 has a newly fitted source after deleting sub-08.
        full = result.full.folds[1]
        deleted = result.deletions[0].folds[0]
        self.assertEqual(full.context.target_id, deleted.context.target_id)
        self.assertFalse(np.array_equal(full.G, deleted.G))
        self.assertFalse(np.array_equal(full.K0, deleted.K0))
        self.assertNotEqual(full.S_map, deleted.S_map)

    def test_target_b_perturbation_cannot_change_its_fit_or_source(self):
        first = self.run_fits().full
        altered = list(self.records)
        altered[3] = replace(altered[3], task_b_partitions=-123*altered[3].task_b_partitions)
        second = self.run_fits(altered).full
        before, after = first.folds[3], second.folds[3]
        for field in ("G", "K0", "m3_maps", "m3_geometry", "O"):
            np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
        self.assertEqual(before.gain, after.gain)
        self.assertFalse(np.array_equal(before.observed_partitions, after.observed_partitions))
        # The perturbed participant is a legitimate training source in other folds.
        self.assertFalse(np.array_equal(first.folds[0].G, second.folds[0].G))

    def test_geometry_prediction_comes_from_explicit_source_recipe(self):
        calls = []
        outputs = []
        def source_geometry(source, numerical, context):
            calls.append(context)
            result = self.geometry(source, numerical, context)
            outputs.append(result)
            return result
        run = self.run_fits(geometry=source_geometry).full
        self.assertEqual(len(calls), 12)
        for fold, supplied in zip(run.folds, outputs):
            np.testing.assert_array_equal(fold.m3_geometry, supplied)
            self.assertFalse(np.allclose(fold.m3_geometry, fold.m3_maps@fold.m3_maps.T/208))
        def failed_geometry(source, numerical, context):
            raise ValueError("source geometry construction failed")
        with self.assertRaisesRegex(ValueError, "source geometry construction"):
            self.run_fits(geometry=failed_geometry)

    def test_fold_geometry_is_preserved_and_projected_before_participant_covariance(self):
        run = self.run_fits().full
        point = exploratory_points(run)
        geometries = crossvalidated_geometries(np.stack([item.task_b_partitions for item in self.records]))
        weights = geometry_weights()
        projected = []
        for index, fold in enumerate(run.folds):
            # K0 uses the actual four source maps and registered replicate
            # pairing; it differs from the Gram of their arithmetic mean.
            self.assertFalse(np.allclose(fold.K0, fold.G@fold.G.T/208))
            centered = geometries[index]-fold.K0
            coefficient = np.sum(centered*fold.K0*weights, axis=(-2, -1))/np.sum(fold.K0**2*weights)
            projected.append(centered-coefficient[:, None, None]*fold.K0)
        projected = np.stack(projected)
        expected = 0.
        for i in range(12):
            for j in range(i+1, 12):
                expected += np.sum((projected[i, 0]-projected[j, 0])*
                                   (projected[i, 1]-projected[j, 1])*weights)/132
        self.assertAlmostEqual(point["V_geometry"], expected, places=11)
        self.assertAlmostEqual(point["R_geometry"]["M3"], expected, places=11)
        averaged = np.mean([fold.K0 for fold in run.folds], axis=0)
        wrong = geometries-averaged
        coefficients = np.sum(wrong*averaged*weights, axis=(-2, -1))/np.sum(averaged**2*weights)
        wrong -= coefficients[..., None, None]*averaged
        wrong_value = 0.
        for i in range(12):
            for j in range(i+1, 12):
                wrong_value += np.sum((wrong[i, 0]-wrong[j, 0])*(wrong[i, 1]-wrong[j, 1])*weights)/132
        self.assertGreater(abs(point["V_geometry"]-wrong_value), 1e-6)
        self.assertAlmostEqual(point["S_geometry"], np.mean([fold.S_geometry for fold in run.folds]))
        halves = np.stack([item.task_b_partitions.reshape(2, 2, 18, 208).mean(axis=1) for item in self.records])
        predictions = np.stack([fold.m3_maps for fold in run.folds])
        self.assertAlmostEqual(point["R_map"]["M3"], map_covariance(
            halves[:, 0]-predictions, halves[:, 1]-predictions), places=12)
        self.assertNotIn("confidence_bounds", point)

    def test_identity_errors_refuse_before_fitting(self):
        changes = (
            lambda records: records[:-1],
            lambda records: records[:-1]+(replace(records[-1], participant_id="sub-02"),),
            lambda records: records[:-1]+(records[0],),
            lambda records: (replace(records[0], voxel_ids=tuple(reversed(self.voxels))),)+records[1:],
            lambda records: (replace(records[0], task_a_condition_ids=("A2", "A1", "A3")),)+records[1:],
            lambda records: (replace(records[0], task_b_condition_ids=tuple(reversed(CONDITIONS))),)+records[1:],
            lambda records: tuple(replace(item, voxel_ids=(True,)+self.voxels[1:]) for item in records),
            lambda records: tuple(replace(item, voxel_ids=(1,)+self.voxels[1:]) for item in records),
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_fits(change(self.records))
        self.assertEqual(self.calls, [])
        def incorrect_source(training, context):
            return replace(self.shared(training, context), training_ids=DEVELOPMENT_IDS)
        with self.assertRaisesRegex(ValueError, "shared source"):
            self.run_fits(shared=incorrect_source)
        integer_ids = tuple(range(208))
        integer_records = tuple(replace(item, voxel_ids=integer_ids) for item in self.records)
        def integer_source(training, context):
            return replace(self.shared(training, context), voxel_ids=integer_ids)
        accepted = self.run_fits(integer_records, shared=integer_source)
        self.assertEqual(len(accepted.full.folds), 12)

    def test_certificate_and_nested_failure_propagate_without_reduced_n_result(self):
        def failed_certificate(source, numerical, context):
            raise ValueError("fixed locality certificate failed")
        with self.assertRaisesRegex(ValueError, "locality"):
            self.run_fits(certificate=failed_certificate)
        self.assertEqual(len(self.calls), 1)
        self.calls.clear()
        def fails_in_deletion(training, context):
            if context.deleted_id == DEVELOPMENT_IDS[0]:
                raise RuntimeError("nested source fit failed")
            return self.shared(training, context)
        with self.assertRaisesRegex(RuntimeError, "nested source"):
            self.run_fits(shared=fails_in_deletion, nested=True)
        self.assertEqual(len(self.calls), 12)
        self.calls.clear()
        def invalid_scale(training, context):
            return replace(self.shared(training, context), S_map=0.)
        with self.assertRaisesRegex(ValueError, "scale"):
            self.run_fits(shared=invalid_scale)


if __name__ == "__main__":
    unittest.main()
