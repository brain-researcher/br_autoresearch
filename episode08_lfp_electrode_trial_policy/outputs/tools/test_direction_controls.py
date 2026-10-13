"""Focused exact wrong-pair and inner-training direction-mean checks."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np

import direction_training as dt
from direction_control_metrics import prediction_scores
from ridge_training import new_progress


class DirectionControlsTest(unittest.TestCase):
    def test_wrong_pair_formula_equals_full_five_trial_derangement_average(self):
        rng = np.random.default_rng(831)
        y, pred = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
        labels = np.repeat(np.arange(8), 5)
        mean, sd = np.array([.2, -.7, 1.]), np.array([.5, 2., 3.])
        permutations = [p for p in itertools.permutations(range(5)) if all(i != j for i, j in enumerate(p))]
        self.assertEqual(len(permutations), 44)
        expected_wrong = 0.
        for d in range(8):
            indices = np.flatnonzero(labels == d)
            expected_wrong += np.mean([np.sum(((y[indices]-pred[indices[list(p)]])/sd)**2)
                                       for p in permutations])
        denom = np.sum(((y-mean)/sd)**2)
        got = prediction_scores(pred, y, labels, mean, sd)
        self.assertAlmostEqual(got["Q_wrong"], 1-expected_wrong/denom, places=13)
        constant = np.repeat(rng.normal(size=(8, 3)), 5, axis=0)
        self.assertAlmostEqual(prediction_scores(constant, y, labels, mean, sd)["pairing_increment"], 0., places=13)

    def test_each_inner_direction_mean_uses_only_its_fitting_rows_and_raw_y_sd(self):
        rng = np.random.default_rng(832)
        labels = np.repeat(np.arange(8), 4)
        x, y = rng.normal(size=(32, 5)), rng.normal(size=(32, 3)) + labels[:, None]*10
        folds = [(np.flatnonzero(np.arange(32)%4 != k), np.flatnonzero(np.arange(32)%4 == k)) for k in range(4)]
        original = dt.residual_statistics
        first, second = [], []
        def capture(store):
            def inner(a, b, d):
                value = original(a, b, d)
                store.append(value)
                return value
            return inner
        with patch.object(dt, "residual_statistics", side_effect=capture(first)):
            dt.fit_residual_candidate(x, y, labels, folds, new_progress())
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[folds[0][1]] += 1000
        changed_y[folds[0][1]] -= 2000
        with patch.object(dt, "residual_statistics", side_effect=capture(second)):
            dt.fit_residual_candidate(changed_x, changed_y, labels, folds, new_progress())
        np.testing.assert_array_equal(first[0].direction_input_mean, second[0].direction_input_mean)
        np.testing.assert_array_equal(first[0].direction_target_mean, second[0].direction_target_mean)
        np.testing.assert_array_equal(first[0].target_sd, second[0].target_sd)
        np.testing.assert_allclose(first[0].target_sd, y[folds[0][0]].std(axis=0), rtol=0, atol=1e-13)
        fit_y = y[folds[0][0]]
        residual_sd = (fit_y - first[0].direction_target_mean[labels[folds[0][0]]]).std(axis=0)
        self.assertTrue(np.all(first[0].target_sd > 5*residual_sd))


if __name__ == "__main__":
    unittest.main()
