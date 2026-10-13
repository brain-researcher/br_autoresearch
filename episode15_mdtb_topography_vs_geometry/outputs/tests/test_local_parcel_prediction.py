"""Focused M1 predictor/dataflow checks; no empirical data or class qualification."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, M1_VARIANTS, DevelopmentParticipant, FoldContext,
    exploratory_points, fit_development_lopo,
)
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from local_parcel_prediction import LocalParcelPrediction
from registered_endpoint_metrics import (
    crossvalidated_geometries, geometry_covariance, map_covariance,
    project_gain_direction,
)


def center(value):
    return value-value.mean(axis=-1, keepdims=True)


def explicit_four_map_geometry(partitions):
    # Independent expression, including signed symmetric cross-partition terms.
    return sum(partitions[a]@partitions[b].T+partitions[b]@partitions[a].T
               for a, b in ((0, 1), (2, 3)))/(4*208)


class LocalParcelPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xyz = 3*np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        tolerances = BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                                    rank_rtol=1e-8, numerical_atol=1e-10)
        cls.anatomy = build_local_anatomical_basis(xyz, np.arange(500, 708), tolerances=tolerances)
        cls.recipe = LocalFunctionalSource(cls.anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
        cls.predictor = LocalParcelPrediction(cls.recipe, cls.anatomy)
        indicator = np.eye(8)[cls.anatomy.cell_labels]
        active = np.zeros((29, 16))
        active[8:24] = np.eye(16)
        # Cell profiles are separated, with an independently identifiable full
        # active source design. This is an algebra fixture, not a data model.
        cls.A = center(10*np.eye(29, 8)@indicator.T+.05*active@cls.anatomy.B.T)
        base_b = center(np.eye(18, 8)@indicator.T+.1*np.eye(18, 16)@cls.anatomy.B.T)
        extra_b = center(np.arange(1., 19.)[:, None]*(cls.anatomy.cell_labels[None]+1)/30)
        cls.partitions = np.stack([(1+q/4)*base_b+(1+q*q/5)*extra_b for q in range(4)])
        cls.records = tuple(DevelopmentParticipant(
            participant, TASK_A_CONDITIONS, cls.anatomy.voxel_ids,
            (1+index/100)*cls.A,
            (1+index/50)*cls.partitions+index**2/200*extra_b[None],
        ) for index, participant in enumerate(DEVELOPMENT_IDS))
        cls.context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)
        cls.source = cls.recipe.fit_shared(cls.records[1:], cls.context)

    def gain(self, source, target):
        A, Y = center(source.task_a_maps), center(target)
        return float(np.sum(source.task_a_weights[:, None]*A*Y)
                     /np.sum(source.task_a_weights[:, None]*A*A))

    def run_lopo(self, records=None, predictor=None, nested=False):
        return fit_development_lopo(
            self.records if records is None else records,
            fit_shared=self.recipe.fit_shared, calibrate_m3=self.recipe.calibrate_m3,
            certify_m3=self.recipe.certify_m3,
            predict_m3_geometry=self.recipe.predict_m3_geometry,
            predict_m1=self.predictor if predictor is None else predictor,
            check=lambda: None, include_participant_deletions=nested,
        )

    def test_four_variants_temperature_memberships_and_actual_partition_geometry(self):
        predictions = self.predictor(self.source, 2.3*self.source.task_a_maps, 2.3, self.context)
        self.assertEqual(tuple(value.name for value in predictions), M1_VARIANTS)
        for prediction, count in zip(predictions, (8, 8, 4, 16)):
            labels = self.predictor.cells[count]
            self.assertEqual(tuple(np.bincount(labels)), (208//count,)*count)
            a_profiles = np.column_stack([self.source.task_a_maps[:, labels == k].mean(axis=1)
                                          for k in range(count)])
            distances = [np.sum(self.source.task_a_weights*(a_profiles[:, a]-a_profiles[:, b])**2)
                         for a in range(count) for b in range(a+1, count)]
            expected_temperature = np.median([value for value in distances if value > 0])
            self.assertAlmostEqual(prediction.diagnostics["temperature"], expected_temperature, places=12)
            memberships = prediction.diagnostics["memberships"]
            np.testing.assert_allclose(memberships.sum(axis=1), 1., atol=1e-14)
            self.assertTrue(np.all(memberships >= 0))
            if prediction.diagnostics["assignment"] == "probabilistic":
                distances = np.sum(self.source.task_a_weights[:, None, None]
                                   *(self.source.task_a_maps[:, :, None]-a_profiles[:, None, :])**2, axis=0)
                expected_memberships = np.exp(-distances/expected_temperature)
                expected_memberships /= expected_memberships.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(memberships, expected_memberships, atol=1e-13)
            b_profiles = np.stack([
                self.source.geometry_state.source_partitions[..., labels == k].mean(axis=-1)
                for k in range(count)], axis=-1)
            expected = center(2.3*(b_profiles@memberships.T))
            np.testing.assert_allclose(prediction.maps, expected.mean(axis=0), atol=1e-12)
            np.testing.assert_allclose(prediction.geometry, explicit_four_map_geometry(expected), atol=1e-12)
            self.assertGreater(np.linalg.norm(prediction.geometry-prediction.maps@prediction.maps.T/208), 1e-5)
            np.testing.assert_allclose(prediction.maps.mean(axis=1), 0., atol=1e-14)
            self.assertFalse(prediction.diagnostics["native_signature_certified"])
            self.assertFalse(prediction.diagnostics["spatial_smoothness_certified"])
            self.assertFalse(prediction.diagnostics["class_adequacy_claimed"])
        np.testing.assert_array_equal(predictions[1].diagnostics["memberships"].argmax(axis=1),
                                      self.anatomy.cell_labels)
        self.assertTrue(predictions[1].diagnostics["hard_assignment_identified"])
        soft = predictions[0].diagnostics["memberships"]
        self.assertTrue(np.all((soft > 0) & (soft < 1)))

    def test_training_b_changes_predictions_but_not_temperature_or_memberships(self):
        first = self.predictor(self.source, self.source.task_a_maps, 1., self.context)
        altered = replace(self.source,
            geometry_state=replace(self.source.geometry_state,
                                   source_partitions=2*self.source.geometry_state.source_partitions),
            G=2*self.source.G, K0=4*self.source.K0, S_map=4*self.source.S_map)
        second = self.predictor(altered, altered.task_a_maps, 1., self.context)
        for before, after in zip(first, second):
            np.testing.assert_array_equal(before.diagnostics["memberships"], after.diagnostics["memberships"])
            self.assertEqual(before.diagnostics["temperature"], after.diagnostics["temperature"])
            np.testing.assert_allclose(after.maps, 2*before.maps, atol=1e-12)
            np.testing.assert_allclose(after.geometry, 4*before.geometry, atol=1e-12)

    def test_hard_ties_use_task_a_only_and_are_explicitly_unidentified(self):
        labels = self.anatomy.cell_labels
        # Replace both equally sized source cells by the same A profile. These
        # target columns remain identical after centering/common-gain scaling;
        # nearest source labels 0 and 1 are symmetric and indistinguishable.
        target = self.source.task_a_maps.copy()
        midpoint = (target[:, labels == 0].mean(axis=1)+target[:, labels == 1].mean(axis=1))/2
        target[:, (labels == 0) | (labels == 1)] = midpoint[:, None]
        predictions = self.predictor(self.source, target, self.gain(self.source, target), self.context)
        hard = predictions[1].diagnostics
        expected = tuple(np.flatnonzero((labels == 0) | (labels == 1)))
        self.assertEqual(hard["hard_assignment_ambiguous_voxel_indices"], expected)
        self.assertFalse(hard["hard_assignment_identified"])
        np.testing.assert_array_equal(hard["memberships"][list(expected)].argmax(axis=1),
                                      np.zeros(len(expected), dtype=int))
        self.assertEqual(hard["hard_tie_rule"], "first_anatomy_ordered_label_within_fixed_numerical_tolerance")

    def test_no_informative_temperature_or_wrong_common_gain_refuses(self):
        # Exact zero means in every fine cell, hence every coarser cell. This
        # isolates the predictor's computability boundary, not source fitting.
        uninformative = np.zeros((29, 208))
        for label in range(16):
            members = np.flatnonzero(self.predictor.cells[16] == label)
            uninformative[:, members[0]] = 1.
            uninformative[:, members[1]] = -1.
        source = replace(self.source, task_a_maps=uninformative)
        with self.assertRaisesRegex(ValueError, "no informative positive temperature"):
            self.predictor(source, uninformative, 1., self.context)
        for invalid_gain in (0., -1., True, np.nan, 1.2):
            with self.assertRaises(ValueError):
                self.predictor(self.source, self.source.task_a_maps, invalid_gain, self.context)

    def test_lopo_refits_all_variants_and_keeps_fold_specific_point_geometry(self):
        calls = []
        def predict(source, target_a, gain, context):
            calls.append(context)
            self.assertEqual(source.training_ids, context.training_ids)
            self.assertNotIn(context.target_id, source.training_ids)
            self.assertNotIn(context.deleted_id, source.training_ids)
            self.assertEqual(target_a.shape, (29, 208))
            self.assertAlmostEqual(gain, self.gain(source, target_a), places=12)
            return self.predictor(source, target_a, gain, context)
        result = self.run_lopo(predictor=predict, nested=True)
        self.assertEqual(len(calls), 144)
        self.assertEqual(len(result.deletions), 12)
        for run in (result.full, *result.deletions):
            for fold in run.folds:
                self.assertEqual(tuple(value.name for value in fold.m1_predictions), M1_VARIANTS)
        run = result.full
        points = exploratory_points(run)
        observed = np.stack([fold.observed_partitions for fold in run.folds])
        K0 = np.stack([fold.K0 for fold in run.folds])
        self.assertGreater(np.linalg.norm(K0[0]-K0[-1]), 1e-4)
        halves = observed.reshape(12, 2, 2, 18, 208).mean(axis=2)
        geometry = crossvalidated_geometries(observed)
        for index, name in enumerate(M1_VARIANTS):
            maps = np.stack([fold.m1_predictions[index].maps for fold in run.folds])
            matrices = np.stack([fold.m1_predictions[index].geometry for fold in run.folds])
            expected_map = map_covariance(halves[:, 0]-maps, halves[:, 1]-maps)
            differences = project_gain_direction(geometry-matrices[:, None], K0[:, None])
            expected_geometry = geometry_covariance(differences[:, 0], differences[:, 1])
            self.assertAlmostEqual(points["R_map"][name], expected_map, places=12)
            self.assertAlmostEqual(points["R_geometry"][name], expected_geometry, places=12)
        self.assertEqual(set(points["R_map"]), {"M0b", "M3", *M1_VARIANTS})
        with self.assertRaisesRegex(ValueError, "all four fixed M1 variants"):
            self.run_lopo(predictor=lambda *args: self.predictor(*args)[:-1])

    def test_target_b_cannot_change_its_source_memberships_or_predictions(self):
        before = self.run_lopo().full
        records = list(self.records)
        records[0] = replace(records[0], task_b_partitions=3*records[0].task_b_partitions)
        after = self.run_lopo(records=records).full
        for first, second in zip(before.folds[0].m1_predictions, after.folds[0].m1_predictions):
            np.testing.assert_array_equal(first.maps, second.maps)
            np.testing.assert_array_equal(first.geometry, second.geometry)
            np.testing.assert_array_equal(first.diagnostics["memberships"], second.diagnostics["memberships"])
        self.assertFalse(np.array_equal(before.folds[0].observed_partitions, after.folds[0].observed_partitions))
        self.assertFalse(np.array_equal(before.folds[1].m1_predictions[0].maps, after.folds[1].m1_predictions[0].maps))


if __name__ == "__main__":
    unittest.main()
