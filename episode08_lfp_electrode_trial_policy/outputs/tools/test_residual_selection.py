"""Focused raw-Y weighting, fitting-only selection and permutation boundary."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np

import residual_selection_training as rs


class ResidualSelectionTest(unittest.TestCase):
    def test_training_only_raw_weighting_and_direction_preserving_permutation(self):
        rng = np.random.default_rng(721)
        labels = np.repeat(np.arange(8), 4)
        x = rng.normal(size=(32, 96*3))
        y = rng.normal(size=(32, 3))+10*labels[:, None]
        original_y = y.copy()
        xx, yy = rs.selector_coordinates(x, y, labels)
        means = np.stack([y[labels == d].mean(axis=0) for d in range(8)])
        expected = (y-means[labels])/y.std(axis=0)
        expected -= expected.mean(axis=0)
        np.testing.assert_allclose(yy, expected, atol=1e-14)
        self.assertTrue(np.all(yy.std(axis=0) < .1))
        permutation = rs.within_direction_permutation(labels, "fixture")
        np.testing.assert_array_equal(labels[permutation], labels)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(32))
        # Exact three-row ordinary-permutation population includes fixed points
        # and has zero expected centered cross-covariance.
        group = np.array([-2., -.5, 2.5])
        all_permutations = list(itertools.permutations(range(3)))
        self.assertIn((0, 1, 2), all_permutations)
        self.assertAlmostEqual(np.mean([group @ group[list(p)] for p in all_permutations]), 0.)
        np.testing.assert_array_equal(y, original_y)
        folds = [(np.flatnonzero(np.arange(32)%4 != k), np.flatnonzero(np.arange(32)%4 == k)) for k in range(4)]
        captures = []
        def record(a, b, d, *args):
            captures.append((a.copy(), b.copy(), d.copy()))
            return np.arange(32)
        progress = {"selector_fits": 0, "selection_seconds": 0.}
        with patch.object(rs, "selector_order", side_effect=record):
            rs.order_bank(x, y, labels, folds, "permuted_residual_conditional", "fixture", progress)
        np.testing.assert_array_equal(captures[0][0], x[folds[0][0]])
        np.testing.assert_array_equal(captures[0][1], y[folds[0][0]])
        self.assertEqual(len(captures), 5)
        # The selector never mutates the actual targets used by the decoder.
        rs.selector_order(x, y, labels, "permuted_residual_conditional", 8, "fixture")
        np.testing.assert_array_equal(y, original_y)


if __name__ == "__main__":
    unittest.main()
