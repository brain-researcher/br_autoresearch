"""Known-answer paired-deletion arithmetic; no source data or simulations."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from participant_cross_replicate_covariance import cross_replicate_covariance
from participant_jackknife_pseudovalues import jackknife_pseudovalues


class ParticipantJackknifePseudovaluesTests(unittest.TestCase):
    def setUp(self):
        self.first = np.array([
            [[1., 2., 4.], [2., -1., 0.]],
            [[3., 4., 1.], [5., 0., 2.]],
            [[5., -2., 3.], [-1., 7., 1.]],
            [[7., 0., -1.], [3., 2., -3.]],
            [[-2., 3., 2.], [0., -4., 6.]],
        ])
        self.second = np.array([
            [[2., 0., 3.], [1., 4., -1.]],
            [[1., 5., 2.], [-2., 3., 1.]],
            [[6., 1., -2.], [4., 2., 3.]],
            [[3., -1., 4.], [0., 1., 2.]],
            [[-1., 2., 0.], [5., -2., 4.]],
        ])

    def test_efficient_arrays_equal_direct_paired_deletions_and_definition(self):
        result = jackknife_pseudovalues(self.first, self.second)
        full = cross_replicate_covariance(self.first, self.second)
        deleted = np.stack([cross_replicate_covariance(
            np.delete(self.first, i, axis=0), np.delete(self.second, i, axis=0)) for i in range(5)])
        np.testing.assert_array_equal(result["statistic"], full)
        np.testing.assert_allclose(result["leave_one_out"], deleted, atol=2e-14, rtol=0)
        np.testing.assert_allclose(result["pseudovalues"], 5*full-4*deleted, atol=3e-14, rtol=0)
        np.testing.assert_allclose(result["pseudovalues"].mean(axis=0), full, atol=2e-14, rtol=0)
        self.assertEqual(result["statistic"].shape, (2, 3))
        self.assertEqual(result["leave_one_out"].shape, self.first.shape)

    def test_three_participants_preserve_negative_estimates_and_signed_pseudovalues(self):
        result = jackknife_pseudovalues([1., 3., 5.], [-1., -3., -5.])
        self.assertEqual(float(result["statistic"]), -4.)
        np.testing.assert_array_equal(result["leave_one_out"], [-2., -8., -2.])
        np.testing.assert_array_equal(result["pseudovalues"], [-8., 4., -8.])
        self.assertEqual(float(result["pseudovalues"].mean()), -4.)
        constant = jackknife_pseudovalues([1., 3., 5.], [7., 7., 7.])
        for value in constant.values():
            np.testing.assert_array_equal(value, np.zeros_like(value))

    def test_nonleading_axes_and_matched_order_preserve_features_and_inputs(self):
        before_first, before_second = self.first.copy(), self.second.copy()
        expected = jackknife_pseudovalues(self.first, self.second)
        self.first.flags.writeable = False
        self.second.flags.writeable = False
        for axis in (1, -1):
            result = jackknife_pseudovalues(np.moveaxis(self.first, 0, axis),
                np.moveaxis(self.second, 0, axis), participant_axis=axis)
            np.testing.assert_array_equal(result["statistic"], expected["statistic"])
            for key in ("leave_one_out", "pseudovalues"):
                np.testing.assert_array_equal(result[key], np.moveaxis(expected[key], 0, axis))
                np.testing.assert_allclose(result["pseudovalues"].mean(axis=axis),
                    expected["statistic"], atol=2e-14, rtol=0)
                self.assertEqual(result[key].dtype, np.dtype("float64"))
        order = [2, 0, 4, 3, 1]
        reordered = jackknife_pseudovalues(self.first[order], self.second[order])
        for key in ("leave_one_out", "pseudovalues"):
            np.testing.assert_allclose(reordered[key], expected[key][order], atol=2e-14, rtol=0)
        np.testing.assert_array_equal(self.first, before_first)
        np.testing.assert_array_equal(self.second, before_second)

    def test_incomplete_nonfinite_nonreal_or_invalid_axis_refuses_without_dropping_rows(self):
        bad = self.first.copy()
        bad[2, 1, 0] = np.nan
        cases = [(self.first[:n], self.second[:n], 0) for n in (0, 1, 2)] + [
            (self.first, self.second[:, :, :2], 0),
            (bad, self.second, 0),
            (self.first, np.full_like(self.second, np.inf), 0),
            (self.first.astype(complex)+1j, self.second, 0),
            (self.first.astype(str), self.second, 0),
            (1., 2., 0),
            (self.first, self.second, 3),
            (self.first, self.second, True),
            (self.first, self.second, 0.0),
        ]
        for first, second, axis in cases:
            with self.subTest(shape=np.shape(first), axis=axis), self.assertRaises(ValueError):
                jackknife_pseudovalues(first, second, participant_axis=axis)


if __name__ == "__main__":
    unittest.main()
