"""Known-answer participant algebra; no source data or scientific simulation."""
from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from participant_cross_replicate_covariance import cross_replicate_covariance


class ParticipantCrossReplicateCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.first = np.array([
            [[1., 2., 4.], [2., -1., 0.]],
            [[3., 4., 1.], [5., 0., 2.]],
            [[5., -2., 3.], [-1., 7., 1.]],
            [[7., 0., -1.], [3., 2., -3.]],
        ])
        self.second = np.array([
            [[2., 0., 3.], [1., 4., -1.]],
            [[1., 5., 2.], [-2., 3., 1.]],
            [[6., 1., -2.], [4., 2., 3.]],
            [[3., -1., 4.], [0., 1., 2.]],
        ])

    def test_explicit_unordered_pair_identity_has_participant_variance_scale(self):
        pairs = [(self.first[i]-self.first[j])*(self.second[i]-self.second[j])
                 for i, j in combinations(range(4), 2)]
        expected = sum(pairs)/(4*3)
        actual = cross_replicate_covariance(self.first, self.second)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        np.testing.assert_allclose(2*actual, sum(pairs)/len(pairs), atol=2e-14, rtol=0)
        self.assertEqual(actual.shape, (2, 3))
        # A separate scalar known answer makes the normalization observable.
        self.assertEqual(float(cross_replicate_covariance([1., 3., 5.], [2., 1., 6.])), 4.)

    def test_matched_permutation_and_explicit_nonleading_axis_preserve_coordinates(self):
        original_first, original_second = self.first.copy(), self.second.copy()
        expected = cross_replicate_covariance(self.first, self.second)
        order = [2, 0, 3, 1]
        np.testing.assert_allclose(cross_replicate_covariance(self.first[order], self.second[order]),
                                   expected, atol=2e-14, rtol=0)
        for axis in (1, -1):
            first = np.moveaxis(self.first, 0, axis)
            second = np.moveaxis(self.second, 0, axis)
            np.testing.assert_array_equal(cross_replicate_covariance(first, second, participant_axis=axis), expected)
        np.testing.assert_array_equal(self.first, original_first)
        np.testing.assert_array_equal(self.second, original_second)

    def test_separate_featurewise_offsets_do_not_center_over_feature_axes(self):
        first_offset = np.array([[8., -3., 11.], [-4., 2., 9.]])
        second_offset = np.array([[-5., 12., 1.], [3., -9., 4.]])
        expected = cross_replicate_covariance(self.first, self.second)
        np.testing.assert_array_equal(cross_replicate_covariance(self.first+first_offset,
                                                                self.second+second_offset), expected)
        # One coordinate's participant-constant offset never changes another.
        altered = self.first.copy()
        altered[:, 0, 1] += 100
        np.testing.assert_array_equal(cross_replicate_covariance(altered, self.second), expected)

    def test_negative_covariance_is_preserved_and_n_two_is_valid(self):
        np.testing.assert_array_equal(cross_replicate_covariance(
            [[1., 0.], [3., 1.], [5., 2.]], [[-1., 7.], [-3., 7.], [-5., 7.]]), [-4., 0.])
        self.assertEqual(float(cross_replicate_covariance([1., 5.], [3., -1.])), -8.)

    def test_n_one_misaligned_nonfinite_nonreal_or_invalid_axis_refuses(self):
        bad = self.first.copy()
        bad[2, 1, 0] = np.nan
        cases = [
            (self.first[:1], self.second[:1], 0),
            (self.first, self.second[:, :, :2], 0),
            (bad, self.second, 0),
            (self.first, np.full_like(self.second, np.inf), 0),
            (self.first.astype(complex)+1j, self.second, 0),
            (self.first, self.second, 3),
            (self.first, self.second, True),
            (self.first, self.second, 0.0),
        ]
        for first, second, axis in cases:
            with self.subTest(shape=np.shape(first), axis=axis), self.assertRaises(ValueError):
                cross_replicate_covariance(first, second, participant_axis=axis)


if __name__ == "__main__":
    unittest.main()
