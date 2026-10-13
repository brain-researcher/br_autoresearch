"""Focused source/local-model tests; no participant payload or qualification."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, DevelopmentParticipant, FoldContext, fit_development_lopo,
)
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from registered_endpoint_metrics import condition_weights, crossvalidated_geometries, geometry_inner
from spatial_deformation_operators import apply_m3


class LocalFunctionalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xyz = 3*np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        tolerances = BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                                     rank_rtol=1e-8, numerical_atol=1e-10)
        cls.anatomy = build_local_anatomical_basis(xyz, np.arange(500, 708), tolerances=tolerances)
        cls.recipe = LocalFunctionalSource(cls.anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
        cls.B = cls.anatomy.B
        cls.coefficients = np.vstack((np.eye(16), .3*np.eye(13, 16)))
        complement = cls.anatomy.cell_labels.astype(float)
        complement -= complement.mean()
        cls.complement = complement/np.linalg.norm(complement)
        cls.X = cls.coefficients@cls.B.T+.5*cls.complement[None]
        condition_coefficients = np.eye(18, 16)+.1
        cls.Z = condition_coefficients@cls.B.T
        cls.source_partitions = np.stack([
            (1+quarter/4)*cls.Z+(1+quarter**2/5)*np.arange(1., 19.)[:, None]*cls.complement/20
            for quarter in range(4)
        ])
        cls.records = tuple(DevelopmentParticipant(
            participant, TASK_A_CONDITIONS, cls.anatomy.voxel_ids,
            (1+index/40)*cls.X, (1+index/50)*cls.source_partitions,
        ) for index, participant in enumerate(DEVELOPMENT_IDS))
        cls.context = FoldContext(DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[1:], None)

    def rotations(self):
        result = np.zeros((16, 16))
        for block, (start, stop) in enumerate(self.anatomy.blocks):
            theta = .1+.05*block
            result[start:stop, start:stop] = [[np.cos(theta), -np.sin(theta)],
                                             [np.sin(theta), np.cos(theta)]]
        result[:2, :2] = [[1., 0.], [0., -1.]]  # reflection is explicitly allowed
        return result

    def test_known_blocks_reflection_complement_and_actual_locality_certificate(self):
        O = self.rotations()
        target = 2*apply_m3(self.X, self.B, O)+np.arange(29.)[:, None]
        fitted = self.recipe.fit_block_operator(self.X, target, global_gain=1.7)
        np.testing.assert_allclose(fitted["O"], O, atol=1e-12, rtol=0)
        self.assertEqual(fitted["gain"], 1.7)
        self.assertLess(np.linalg.det(fitted["O"][:2, :2]), 0)
        certificate = self.recipe.operator_certificate(fitted["O"])
        for key, value in certificate.items():
            if key.endswith("max_abs"):
                self.assertLessEqual(value, 1e-10)
        np.testing.assert_allclose(apply_m3(self.complement[None], self.B, O),
                                   self.complement[None], atol=1e-12, rtol=0)
        # An orthogonal exchange across different cells preserves geometry but
        # fails the selected locality class. A global Procrustes fallback is invalid.
        crossed = np.eye(16)
        crossed[[0, 2]] = crossed[[2, 0]]
        with self.assertRaisesRegex(ValueError, "locality"):
            self.recipe.operator_certificate(crossed)

    def test_global_source_rank_and_local_uniqueness_are_distinct(self):
        coefficients = np.eye(29, 16)
        X = coefficients@self.B.T
        # Singular global C, but every permitted 2x2 diagonal block is I:
        # the selected local solution is unique and must be admitted.
        joint = np.eye(16)
        joint[:4, :4] = np.block([[np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]])
        fitted = self.recipe.fit_block_operator(X, coefficients@joint@self.B.T, global_gain=1.)
        global_singular = fitted["global_cross_singular_values_diagnostic_only"]
        self.assertLess(global_singular[-1], 1e-8*global_singular[0])
        np.testing.assert_allclose(fitted["O"], np.eye(16), atol=1e-12, rtol=0)
        # Every local source block can have rank2 while the global design has
        # repeated directions; the registered global source rank still refuses.
        deficient = coefficients.copy()
        deficient[:, 2:4] = deficient[:, :2]
        with self.assertRaisesRegex(ValueError, "global Task-A active source design"):
            self.recipe.fit_block_operator(deficient@self.B.T, X, global_gain=1.)
        # A globally full-rank cross matrix can have unidentified local blocks.
        # Roundoff in those zero blocks must not pass relative to its own norm.
        exchange = np.eye(16)
        exchange[:4, :4] = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
        with self.assertRaisesRegex(ValueError, "global objective scale"):
            self.recipe.fit_block_operator(X, coefficients@exchange@self.B.T, global_gain=1.)

    def test_actual_four_source_maps_inverse_orientation_and_scales(self):
        training = []
        for person, record in enumerate(self.records[1:]):
            O = np.zeros((16, 16))
            for block, (start, stop) in enumerate(self.anatomy.blocks):
                theta = (person+1)*(block+1)/80
                O[start:stop, start:stop] = [[np.cos(theta), -np.sin(theta)],
                                             [np.sin(theta), np.cos(theta)]]
            training.append(replace(record,
                task_a_maps=apply_m3(record.task_a_maps, self.B, O),
                task_b_partitions=apply_m3(record.task_b_partitions.reshape(72, 208),
                                           self.B, O).reshape(4, 18, 208)))
        source = self.recipe.fit_shared(training, self.context)
        state = source.geometry_state
        expected_a, expected_b, wrong_forward_b = [], [], []
        self.assertGreater(max(np.max(np.abs(O-np.eye(16))) for O in state.training_rotations), .01)
        self.assertGreater(max(np.max(np.abs(O-O.T)) for O in state.training_rotations), .01)
        for record, rotation in zip(training, state.training_rotations):
            Q = np.eye(208)+self.B@(rotation-np.eye(16))@self.B.T
            expected_a.append(record.task_a_maps@Q.T)
            expected_b.append((record.task_b_partitions.reshape(72, 208)@Q.T).reshape(4, 18, 208))
            wrong_forward_b.append((record.task_b_partitions.reshape(72, 208)@Q).reshape(4, 18, 208))
        np.testing.assert_allclose(source.task_a_maps, np.mean(expected_a, axis=0), atol=1e-12)
        np.testing.assert_allclose(state.source_partitions, np.mean(expected_b, axis=0), atol=1e-12)
        self.assertGreater(np.linalg.norm(state.source_partitions-np.mean(wrong_forward_b, axis=0)), .01)
        np.testing.assert_allclose(source.G, state.source_partitions.mean(axis=0), atol=1e-12)
        actual_k0 = crossvalidated_geometries(state.source_partitions[None])[0].mean(axis=0)
        np.testing.assert_array_equal(source.K0, actual_k0)
        self.assertFalse(np.allclose(source.K0, source.G@source.G.T/208))
        halves = state.source_partitions.reshape(2, 2, 18, 208).mean(axis=1)
        expected_scale = np.sum(condition_weights()*np.mean(halves[0]*halves[1], axis=-1))
        self.assertAlmostEqual(source.S_map, expected_scale, places=14)
        wrong_pairing = np.sum(condition_weights()*np.diag(source.K0))
        self.assertGreater(abs(source.S_map-wrong_pairing), 1e-5)
        self.assertGreater(float(geometry_inner(source.K0, source.K0)), 0)
        self.assertTrue(all(item["gain"] == 1 for item in state.training_diagnostics))

    def test_common_positive_gain_is_fixed_before_local_fit_and_geometry_uses_same_maps(self):
        source = self.recipe.fit_shared(self.records[1:], self.context)
        O = self.rotations()
        target = 2.4*apply_m3(source.task_a_maps, self.B, O)
        expected_gain = np.sum(source.task_a_maps*target)/np.sum(source.task_a_maps**2)
        fitted = self.recipe.calibrate_m3(source, target, self.context)
        np.testing.assert_allclose(fitted["O"], O, atol=1e-12, rtol=0)
        self.assertAlmostEqual(fitted["gain"], expected_gain, places=12)
        self.assertGreater(abs(fitted["gain"]-2.4), .01)
        self.recipe.certify_m3(source, fitted, self.context)
        transformed = fitted["gain"]*apply_m3(
            source.geometry_state.source_partitions.reshape(72, 208), self.B, fitted["O"],
        ).reshape(4, 18, 208)
        actual = self.recipe.predict_m3_geometry(source, fitted, self.context)
        np.testing.assert_array_equal(actual, crossvalidated_geometries(transformed[None])[0].mean(axis=0))
        np.testing.assert_allclose(actual, fitted["gain"]**2*source.K0, atol=1e-12, rtol=1e-12)
        # Gain is a common A-only rule, not rescued by a reflection or a floor.
        with self.assertRaisesRegex(ValueError, "pre-operator common Task-A gain"):
            self.recipe.calibrate_m3(source, -source.task_a_maps, self.context)

    def test_complete_lopo_callback_integration_keeps_integer_support_and_fold_sources(self):
        result = fit_development_lopo(
            self.records, fit_shared=self.recipe.fit_shared, calibrate_m3=self.recipe.calibrate_m3,
            certify_m3=self.recipe.certify_m3, predict_m3_geometry=self.recipe.predict_m3_geometry,
            check=lambda: None, include_participant_deletions=False,
        )
        self.assertEqual(result.full.participant_ids, DEVELOPMENT_IDS)
        self.assertEqual(len(result.full.folds), 12)
        self.assertEqual(result.deletions, ())
        for fold in result.full.folds:
            self.assertNotIn(fold.context.target_id, fold.context.training_ids)
            self.assertEqual(fold.m3_maps.shape, (18, 208))
            self.assertEqual(fold.K0.shape, (18, 18))
            self.assertGreater(fold.S_map, 0)
        self.assertFalse(np.array_equal(result.full.folds[0].K0, result.full.folds[-1].K0))

    def test_exact_task_a_roster_roles_and_no_geometry_state_substitution(self):
        self.assertEqual(len(TASK_A_CONDITIONS), 29)
        self.assertEqual(TASK_A_CONDITIONS[-1], "rest")
        malformed = list(self.records[1:])
        malformed[0] = replace(malformed[0], task_a_condition_ids=TASK_A_CONDITIONS[:-1])
        with self.assertRaisesRegex(ValueError, "condition/support identities"):
            self.recipe.fit_shared(malformed, self.context)
        with self.assertRaisesRegex(ValueError, "training records"):
            self.recipe.fit_shared(self.records[:-1], self.context)
        source = self.recipe.fit_shared(self.records[1:], self.context)
        wrong_context = replace(self.context, deleted_id=DEVELOPMENT_IDS[1])
        with self.assertRaisesRegex(ValueError, "source identities must exactly retain all required participant exclusions"):
            self.recipe.calibrate_m3(source, self.records[0].task_a_maps, wrong_context)
        with self.assertRaisesRegex(ValueError, "actual current-fold fit"):
            self.recipe.calibrate_m3(replace(source, geometry_state=source.K0),
                                      self.records[0].task_a_maps, self.context)


if __name__ == "__main__":
    unittest.main()
