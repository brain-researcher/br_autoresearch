"""Three raw-mean/exclusion/LOPO fixtures; no participant data or fitting sweep."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from anatomical_group_baseline import AnatomicalGroupBaseline
from development_lopo import (
    DEVELOPMENT_IDS, DevelopmentParticipant, FoldContext,
    exploratory_points, fit_development_lopo,
)
from registered_endpoint_metrics import (
    crossvalidated_geometries, geometry_covariance, map_covariance, project_gain_direction,
)
import test_development_lopo as lopo_fixture


class UnreadableArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("this array must not be read")


class AnatomicalGroupBaselineTests(unittest.TestCase):
    def setUp(self):
        self.voxel_ids = tuple(range(500, 708))
        spatial = np.linspace(-1., 1., 208)
        pattern = np.arange(1., 19.)[:, None]*spatial
        self.records = tuple(DevelopmentParticipant(person, ("unread Task A",), self.voxel_ids,
            UnreadableArray(), np.stack([(1+index/10)*factor*pattern+index+quarter
                                       for quarter, factor in enumerate((1., -2., 3., -1.))]))
            for index, person in enumerate(DEVELOPMENT_IDS))
        self.context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)
        self.predictor = AnatomicalGroupBaseline(self.voxel_ids, check=lambda: None)

    def test_raw_equal_person_mean_and_actual_signed_partition_geometry(self):
        before = [record.task_b_partitions.copy() for record in self.records]
        result = self.predictor(self.records[1:], self.context)
        raw = np.stack([record.task_b_partitions for record in self.records[1:]])
        centered = raw-raw.mean(axis=-1, keepdims=True)
        partitions = sum(centered)/len(centered)
        expected_map = sum(partitions)/4
        cross12 = partitions[0]@partitions[1].T
        cross34 = partitions[2]@partitions[3].T
        expected_geometry = (cross12+cross12.T+cross34+cross34.T)/(4*208)
        np.testing.assert_allclose(result.maps, expected_map, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result.geometry, expected_geometry, atol=1e-12, rtol=1e-12)
        self.assertTrue(np.all(np.diag(result.geometry) < 0))
        self.assertGreater(np.linalg.norm(result.geometry-result.maps@result.maps.T/208), 1.)
        self.assertEqual(result.training_ids, self.context.training_ids)
        self.assertFalse(result.maps.flags.writeable)
        self.assertFalse(result.geometry.flags.writeable)
        nested = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[2:], DEVELOPMENT_IDS[1])
        nested_result = self.predictor(self.records[2:], nested)
        self.assertEqual(nested_result.training_ids, DEVELOPMENT_IDS[2:])
        self.assertFalse(np.array_equal(result.maps, nested_result.maps))
        for record, original in zip(self.records, before):
            np.testing.assert_array_equal(record.task_b_partitions, original)

    def test_exclusion_and_axis_checks_precede_payload_and_invalid_arrays_refuse(self):
        hidden = tuple(replace(record, task_b_partitions=UnreadableArray()) for record in self.records)
        for records, context in (
                (hidden, self.context), (hidden[1:][::-1], self.context),
                (hidden[1:-1], self.context),
                (hidden[1:], replace(self.context, training_ids=DEVELOPMENT_IDS)),
                (hidden[1:], replace(self.context, deleted_id=DEVELOPMENT_IDS[1]))):
            with self.subTest(ids=tuple(record.participant_id for record in records)), self.assertRaisesRegex(ValueError, "training identities"):
                self.predictor(records, context)
        for replacement in (
                replace(hidden[-1], voxel_ids=self.voxel_ids[::-1]),
                replace(hidden[-1], task_b_condition_ids=hidden[-1].task_b_condition_ids[::-1])):
            with self.assertRaisesRegex(ValueError, "ordered anatomical and 18-condition"):
                self.predictor(hidden[1:-1]+(replacement,), self.context)
        for value in (np.zeros((3, 18, 208)), np.full((4, 18, 208), np.nan),
                      np.ones((4, 18, 208), dtype=complex)):
            changed = self.records[1:-1]+(replace(self.records[-1], task_b_partitions=value),)
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                self.predictor(changed, self.context)

    def test_lopo_points_keep_m0b_origin_and_ignore_target_a_and_b(self):
        fixture = lopo_fixture.DevelopmentLOPOTests()
        fixture.setUp()
        baseline = AnatomicalGroupBaseline(fixture.voxels, check=lambda: None)
        calls = []
        def fit_baseline(training, context):
            calls.append((tuple(record.participant_id for record in training), context))
            return baseline(training, context)
        def distinct_functional_source(training, context):
            # A deliberately distinct supplied M0b fixture makes accidental
            # substitution of M0a for the residual origin observable. This is
            # orchestration arithmetic, not a newly selected source recipe.
            source = fixture.shared(training, context)
            partitions = 1.2*source.geometry_state
            return replace(source, G=partitions.mean(axis=0),
                K0=crossvalidated_geometries(partitions[None])[0].mean(axis=0),
                S_map=1.2**2*source.S_map, geometry_state=partitions)
        def run(records):
            return fit_development_lopo(records, fit_shared=distinct_functional_source,
                calibrate_m3=fixture.calibrate, certify_m3=fixture.certificate,
                predict_m3_geometry=fixture.geometry, check=lambda: None,
                include_participant_deletions=False, fit_m0a=fit_baseline).full
        full = run(fixture.records)
        self.assertEqual(len(calls), 12)
        for ids, context in calls:
            self.assertEqual(ids, context.training_ids)
            self.assertNotIn(context.target_id, ids)
        points = exploratory_points(full)
        observed = np.stack([record.task_b_partitions for record in fixture.records])
        halves = observed.reshape(12, 2, 2, 18, 208).mean(axis=2)
        maps = np.stack([fold.m0a_prediction.maps for fold in full.folds])
        geometry = np.stack([fold.m0a_prediction.geometry for fold in full.folds])
        self.assertFalse(np.allclose(full.folds[0].m0a_prediction.maps, full.folds[0].G))
        self.assertAlmostEqual(points["R_map"]["M0a"], map_covariance(
            halves[:, 0]-maps, halves[:, 1]-maps), places=10)
        K0 = np.stack([fold.K0 for fold in full.folds])
        residual = project_gain_direction(crossvalidated_geometries(observed)-geometry[:, None], K0[:, None])
        self.assertAlmostEqual(points["R_geometry"]["M0a"], geometry_covariance(
            residual[:, 0], residual[:, 1]), places=10)
        self.assertEqual(points["R_map"]["M0b"], points["V_map"])
        self.assertEqual(points["R_geometry"]["M0b"], points["V_geometry"])
        before = full.folds[0].m0a_prediction
        for field, replacement in (("task_a_maps", 7*fixture.records[0].task_a_maps),
                                   ("task_b_partitions", -5*fixture.records[0].task_b_partitions)):
            changed = list(fixture.records)
            changed[0] = replace(changed[0], **{field: replacement})
            after_run = run(changed)
            after = after_run.folds[0].m0a_prediction
            np.testing.assert_array_equal(before.maps, after.maps)
            np.testing.assert_array_equal(before.geometry, after.geometry)
            self.assertEqual(before.training_ids, after.training_ids)
            if field == "task_b_partitions":
                self.assertFalse(np.array_equal(full.folds[1].m0a_prediction.maps,
                                               after_run.folds[1].m0a_prediction.maps))


if __name__ == "__main__":
    unittest.main()
