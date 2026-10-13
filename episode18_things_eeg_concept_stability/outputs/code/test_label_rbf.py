"""Focused kernel-geometry checks; execute only in coordinated Slurm compute."""
import unittest

import numpy as np

from continuous_sequence_model import EventTerm, JointSequenceDesign, SequenceDesign
from label_rbf import fit_label_rbf_basis
from slot_capacity import RIDGE_GRID, slot_capacity_curves
from training_transforms import DIMENSIONS, LABEL, fit_feature_panel


def fixture(coordinates=None):
    points = (np.array([[-1., 0.], [-.5, 1.], [.2, -.4], [1., .6], [2., -1.]])
              if coordinates is None else np.asarray(coordinates))
    ids = tuple(f"concept_{i}" for i in range(len(points)))
    families = {}
    for index, name in enumerate(LABEL):
        values = np.zeros((len(points), DIMENSIONS[name]))
        values[:, :2] = points
        values[:, 2] = points[:, 0] ** (index + 1)
        families[name] = values
    row_ids = {name: ids for name in LABEL}
    panel = fit_feature_panel(families, training_unit_ids=row_ids,
                              requested_ranks={name: 32 for name in LABEL}, panel="label")
    transformed = panel.apply(families, row_ids=row_ids)
    return panel, transformed, ids, families


def fitted(panel, rows, ids, **changes):
    options = dict(training_unit_ids=ids, label_panel_transform=panel, bandwidth_multiplier=1.,
                   eigenvalue_rtol=1e-12, eigenvalue_atol=1e-14,
                   reconstruction_rtol=1e-10, reconstruction_max_atol=1e-10)
    return fit_label_rbf_basis(rows, **dict(options, **changes))


def direct_kernel(left, right, bandwidth):
    distances_squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-distances_squared / (2 * bandwidth ** 2))


def temporal(lags):
    return np.column_stack((np.ones(len(lags)), lags))


class LabelRBFTests(unittest.TestCase):
    def test_sqrt_kernel_coordinates_reproduce_kernel_ridge_predictions(self):
        panel, rows, ids, _ = fixture()
        fit = fitted(panel, rows, ids)
        kernel = direct_kernel(rows, rows, fit.bandwidth)
        np.testing.assert_allclose(fit.training_features @ fit.training_features.T, kernel, atol=1e-12)
        np.testing.assert_allclose(fit.apply(rows), fit.training_features, atol=1e-12)
        evaluation = np.vstack((rows[0] * 1.2, rows[-1] * -.4))
        target = np.column_stack((np.arange(len(rows)), np.arange(len(rows)) ** 2))
        ridge = .1
        phi = fit.training_features
        coefficient = np.linalg.solve(phi.T @ phi + ridge * np.eye(fit.effective_rank), phi.T @ target)
        expected = direct_kernel(evaluation, rows, fit.bandwidth) @ np.linalg.solve(
            kernel + ridge * np.eye(len(rows)), target)
        np.testing.assert_allclose(fit.apply(evaluation) @ coefficient, expected, atol=1e-11)
        self.assertEqual(fit.family_order, LABEL)
        self.assertEqual(fit.family_effective_ranks, tuple(f.effective_rank for f in panel.families))

    def test_heldout_values_never_refit_panel_bandwidth_or_coordinates_and_ids_only_join(self):
        panel, rows, ids, families = fixture()
        fit = fitted(panel, rows, ids, bandwidth_multiplier=.5)
        distances = np.sqrt(np.sum((rows[:, None] - rows[None, :]) ** 2, axis=2))
        upper = distances[np.triu_indices(len(rows), 1)]
        expected_median = np.median(upper[upper > 0])
        self.assertAlmostEqual(fit.median_positive_training_distance, expected_median)
        self.assertAlmostEqual(fit.bandwidth, .5 * expected_median)
        original_features = fit.training_features.copy()
        original_means = tuple(family.mean.copy() for family in panel.families)
        changed = {name: values[:2].copy() for name, values in families.items()}
        for values in changed.values():
            values[:, :3] += 1000.
        transformed = panel.apply(changed, row_ids={name: ("held_a", "held_b") for name in LABEL})
        evaluated = fit.apply(transformed)
        self.assertEqual(evaluated.shape, (2, fit.effective_rank))
        np.testing.assert_array_equal(fit.training_features, original_features)
        for family, original in zip(panel.families, original_means):
            np.testing.assert_array_equal(family.mean, original)
        np.testing.assert_array_equal(fit.event_features((ids[2], ids[0], ids[2])),
                                      original_features[[2, 0, 2]])
        with self.assertRaises(ValueError):
            fit.event_features(("unknown",))
        with self.assertRaises(ValueError):
            fitted(panel, rows, ids[::-1])

    def test_duplicate_values_retain_units_and_zero_distance_or_bad_truncation_refuses(self):
        panel, rows, ids, _ = fixture([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
        fit = fitted(panel, rows, ids)
        self.assertEqual(len(fit.training_unit_ids), 4)
        self.assertEqual(fit.effective_rank, 3)
        self.assertEqual(len(fit.discarded_eigenvalues), 1)
        np.testing.assert_allclose(fit.training_features[0], fit.training_features[1], atol=1e-12)
        np.testing.assert_allclose(fit.apply(rows), fit.training_features, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            fitted(panel, rows, ids, eigenvalue_rtol=.9)
        same_panel, same_rows, same_ids, _ = fixture([[1., 1.], [1., 1.], [1., 1.]])
        with self.assertRaisesRegex(ValueError, "positive training distance"):
            fitted(same_panel, same_rows, same_ids)
        with self.assertRaises(ValueError):
            fitted(panel, rows, ids, bandwidth_multiplier=1.5)

    def test_event_kernel_coordinates_integrate_with_joint_sequence_and_term_capacity(self):
        panel, rows, ids, _ = fixture()
        fit = fitted(panel, rows, ids, bandwidth_multiplier=2.)
        terms = {"label_rbf": EventTerm(np.array([.01, .19, .32]),
                                        fit.event_features((ids[0], ids[2], ids[0])),
                                        temporal, 2, (0., .4))}
        sequence = SequenceDesign("sequence", np.arange(9.) * .05, np.ones((9, 1)), terms)
        operator = JointSequenceDesign([sequence])
        dense = operator @ np.eye(operator.shape[1])
        weights = np.array([1., 1., 0., 1., 2., 1., 1., 1., 1.])
        result = slot_capacity_curves(base_design=dense[:, :1], slot_designs={"label_rbf": dense[:, 1:]},
                                      sample_weights=weights, base_penalties=[0.],
                                      base_rank_relative_tolerance=1e-12)
        curve = result.curve("label_rbf")
        weighted = np.sqrt(weights)[:, None] * dense
        penalties = np.concatenate(([0.], np.ones(dense.shape[1] - 1)))
        augmented = np.vstack((weighted, np.diag(np.sqrt(penalties))))
        influence = np.linalg.pinv(augmented, rcond=1e-12)[:, :len(weights)]
        expected_term = np.trace(weighted[:, 1:] @ influence[1:])
        self.assertAlmostEqual(curve.term_edf[RIDGE_GRID.index(1.)], expected_term, places=10)
        self.assertEqual(curve.slot_columns, fit.effective_rank * 2)


if __name__ == "__main__":
    unittest.main()
