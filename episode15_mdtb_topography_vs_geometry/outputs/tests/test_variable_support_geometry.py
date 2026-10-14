"""Changed-support geometry and complete-panel joins; array fixtures only.

These two checks exercise the adopted support-size change, not empirical
adequacy or a new optimizer qualification. Run only in the coordinator's job.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import test_calibration_falsifiers as existing
from anatomical_group_baseline import AnatomicalGroupBaseline
from calibration_falsifiers import TaskACalibrationPacket, TASK_A_HALF_RUNS
from common_cardinal_source import CommonCardinalSource
from development_lopo import (
    DEVELOPMENT_IDS, DevelopmentParticipant, MatchedWarpPrediction,
    RandomBasisPrediction, fit_development_lopo,
)
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from local_isometry_variants import LocalIsometryVariants
from local_parcel_prediction import LocalParcelPrediction
from matched_random_basis import MatchedRandomLocalBasis
from observed_beta_summary import observed_beta_summary
from paired_stretch_variants import PairedStretchVariants
from smooth_relocation import BoundedPullField, RelocationTaskAObjective, VARIANTS


def reduced_fixture(size):
    """Reuse the existing cheap complete-panel callbacks on an uneven axis."""
    fixture = existing.CalibrationFalsifierTests()
    fixture.setUp()
    fixture.voxels = tuple(range(500, 500 + size))
    x = np.arange(size, dtype=float)
    fixture.u, fixture.v = np.sin(2*np.pi*x/size), np.cos(2*np.pi*x/size)
    fixture.B = np.stack((fixture.u, fixture.v), axis=-1) / np.sqrt(size/2)
    fixture.mean = ((2 + np.arange(29)/30)[:, None]*fixture.u
                    + (np.arange(29)/70)[:, None]*fixture.v)
    fixture.noise = (np.arange(29)/90 + .1)[:, None]*fixture.v
    fixture.packets = tuple(TaskACalibrationPacket(
        person, TASK_A_CONDITIONS, fixture.voxels, (1 + index/25)*fixture.mean,
        np.stack(((1 + index/25)*fixture.mean + fixture.noise,
                  (1 + index/25)*fixture.mean - fixture.noise)), TASK_A_HALF_RUNS,
    ) for index, person in enumerate(DEVELOPMENT_IDS))
    pattern = np.arange(1, 19)[:, None]*fixture.u/18 + fixture.v[None]/3
    fixture.records = tuple(DevelopmentParticipant(
        packet.participant_id, TASK_A_CONDITIONS, fixture.voxels, packet.task_a_maps,
        np.stack([(1 + index/20 + part/7)*pattern + (index + part**2)*fixture.v/40
                  for part in range(4)]),
    ) for index, packet in enumerate(fixture.packets))

    def calibrate(source, task_a, context):
        fixture.assertEqual(task_a.shape, (29, size))
        gain = float(np.sum(source.task_a_maps*task_a)/np.sum(source.task_a_maps**2))
        fixture.calibration_calls.append((id(source), context, gain))
        return {"O": np.eye(2), "gain": gain}

    fixture.calibrate = calibrate
    return fixture


class VariableSupportTests(unittest.TestCase):
    def test_uneven_geometry_rebuilds_all_cells_and_preserves_operator_limits(self):
        original = 3*np.indices((13, 4, 4)).reshape(3, -1).T.astype(float)
        original *= np.array([1., 1.3, 1.7])
        index = np.arange(208)
        original += .005*np.column_stack((np.sin(index), np.cos(index), np.sin(2*index)))
        keep = np.ones(208, dtype=bool)
        keep[[3, 89, 190]] = False
        coordinates, ids = original[keep], tuple((index + 500)[keep])
        size = len(ids)
        tolerances = BasisTolerances(1e-10, 1e-10, 1e-8, 1e-10)
        with self.assertRaises(ValueError):
            build_local_anatomical_basis(coordinates, np.array(ids), tolerances=tolerances)
        anatomy = build_local_anatomical_basis(coordinates, np.array(ids),
            tolerances=tolerances, support_size=size)
        reverse = build_local_anatomical_basis(coordinates[::-1], np.array(ids[::-1]),
            tolerances=tolerances, support_size=size)
        np.testing.assert_allclose(anatomy.B, reverse.B[::-1], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(anatomy.cell_labels, reverse.cell_labels[::-1])
        recipe = LocalFunctionalSource(anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
        variants = LocalIsometryVariants(recipe, anatomy, check=lambda: None)
        self.assertEqual(tuple(op.dimension for op in variants.operators), (16, 16, 16, 24))
        for operator in variants.operators:
            sizes = np.bincount(operator.labels)
            self.assertEqual(int(sizes.sum()), size)
            self.assertLessEqual(int(sizes.max()-sizes.min()), 1)
            self.assertGreater(int(sizes.min()), operator.variant.modes_per_cell)
            operator.operator_certificate(np.eye(operator.dimension))
        parcels = LocalParcelPrediction(recipe, anatomy)
        for count, labels in parcels.cells.items():
            sizes = np.bincount(labels)
            self.assertEqual(len(sizes), count)
            self.assertEqual(set(sizes), {size//count, (size+count-1)//count})
        random = MatchedRandomLocalBasis(recipe, anatomy, check=lambda: None)
        self.assertEqual(random.support_size, size)
        stretches = PairedStretchVariants(recipe, anatomy, check=lambda: None)
        for specification in stretches.specifications:
            stretches.operator_certificate(np.eye(16), .01*specification.modes[0], specification)
        representation = CommonCardinalSource(coordinates, ids, support_size=size)
        weights, derivative = representation.weights_and_derivatives(coordinates)
        np.testing.assert_array_equal(weights, np.eye(size))
        np.testing.assert_allclose(derivative.sum(axis=1), 0, atol=1e-10)
        fields = tuple(BoundedPullField(anatomy, variant) for variant in VARIANTS)
        for field in fields:
            self.assertEqual(field.centers.shape, (8, 3))
            self.assertTrue(field.certificate(np.zeros((8, 3)))["global_diffeomorphism_certified"])
        rng = np.random.default_rng(15061)
        source = rng.normal(size=(29, size))
        source -= source.mean(axis=1, keepdims=True)
        target = 1.3*source + .07*np.roll(source, 1, axis=1)
        gain = float(np.sum(source*target)/np.sum(source**2))
        objective = RelocationTaskAObjective(representation, fields[0], source, target, common_gain=gain)
        value, gradient = objective.value_gradient(np.zeros(24))
        expected = np.mean((source-target/gain)**2)/np.mean(source**2)
        self.assertAlmostEqual(value, expected, places=13)
        delta = np.zeros(24)
        delta[0] = 1e-6
        finite_difference = (objective.value_gradient(delta)[0]-objective.value_gradient(-delta)[0])/(2e-6)
        self.assertAlmostEqual(float(gradient[0]), float(finite_difference), places=7)

    def test_uneven_complete_panel_joins_keep_original_ids_and_mean_voxel_loss(self):
        size = 205
        fixture = reduced_fixture(size)
        baseline = AnatomicalGroupBaseline(fixture.voxels, check=lambda: None, support_size=size)

        def basis(source, task_a, numerical, context):
            base = numerical["gain"]*(source.G-source.G.mean(axis=-1, keepdims=True))
            return tuple(RandomBasisPrediction(draw,
                base + (-1 if draw % 2 else 1)*(draw//2+1)*.03*fixture.u,
                source.K0, {}) for draw in range(8))

        def warp(source, predictions, gain, context):
            return tuple(MatchedWarpPrediction(prediction.name, draw,
                prediction.maps + (-1 if draw % 2 else 1)*(draw//2+1)*.04*fixture.u,
                prediction.geometry, {}) for prediction in predictions for draw in range(8))

        options = dict(fit_shared=fixture.shared, **fixture.callbacks(), fit_m0a=baseline,
            predict_random_bases=basis, predict_matched_warps=warp,
            include_participant_deletions=False)
        with self.assertRaises(ValueError):
            fit_development_lopo(fixture.records, **options)
        ordinary = fit_development_lopo(fixture.records, **options, support_size=size)
        controls = fixture.fit(tuple(fold.calibration_reference for fold in ordinary.full.folds),
                               support_size=size)
        result = observed_beta_summary(ordinary.full, controls, check=lambda: None, support_size=size)
        self.assertEqual(result["support_size"], size)
        self.assertEqual(result["voxel_weight"], 1/size)
        self.assertEqual(len(fixture.source_calls), 24)
        self.assertEqual(len(fixture.calibration_calls), 48)
        expected = []
        for fold in ordinary.full.folds:
            self.assertEqual(fold.calibration_reference.source.voxel_ids, fixture.voxels)
            observed = fold.observed_partitions - fold.observed_partitions.mean(axis=-1, keepdims=True)
            predicted = fold.G-fold.G.mean(axis=-1, keepdims=True)
            expected.append(np.mean((observed-predicted[None])**2, axis=-1))
        np.testing.assert_allclose(result["ordinary"]["losses"]["M0b"]
            ["partition_condition_observed_mse"], expected, atol=1e-14)
        with self.assertRaises(ValueError):
            observed_beta_summary(ordinary.full, controls, check=lambda: None)
        wrong = replace(fixture.records[0], voxel_ids=tuple(reversed(fixture.voxels)))
        with self.assertRaises(ValueError):
            fit_development_lopo((wrong,)+fixture.records[1:], **options, support_size=size)


if __name__ == "__main__":
    unittest.main()
