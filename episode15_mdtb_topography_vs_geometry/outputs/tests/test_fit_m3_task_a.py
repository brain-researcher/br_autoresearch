"""Small deterministic fitter checks; no empirical access or model qualification."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from fit_m3_task_a import fit_m3_task_a
from spatial_deformation_operators import apply_m3


class M3TaskAFitTests(unittest.TestCase):
    def setUp(self):
        self.B = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])/np.sqrt(2.)
        self.X = np.array([[2., -1., 4., -5.], [-3., 3., 1., -1.], [2., 0., -3., 1.]])
        self.weights = np.array([1., 3., 2.])

    def fit(self, X, Y, B=None, **changes):
        options = dict(condition_weights=self.weights, fit_global_gain=False,
                       rank_rtol=1e-10, numerical_atol=1e-12)
        options.update(changes)
        return fit_m3_task_a(X, Y, self.B if B is None else B, **options)

    def test_rotation_reflection_and_global_gain_recovered_without_mutation(self):
        originals = [a.copy() for a in (self.X, self.B, self.weights)]
        for O in (np.array([[0., -1.], [1., 0.]]), np.diag([-1., 1.])):
            for gain in (1., 2.5):
                Y = gain*apply_m3(self.X, self.B, O)+np.array([2., -5., 7.])[:, None]
                Y_before = Y.copy()
                result = self.fit(self.X+13., Y, fit_global_gain=gain != 1.)
                np.testing.assert_allclose(result["O"], O, atol=1e-12, rtol=0)
                self.assertAlmostEqual(result["gain"], gain)
                self.assertLess(result["weighted_task_a_squared_error"], 1e-24)
                np.testing.assert_array_equal(Y, Y_before)
        for current, original in zip((self.X, self.B, self.weights), originals):
            np.testing.assert_array_equal(current, original)

    def test_weights_and_rowwise_orientation_match_two_dimensional_optimum(self):
        # Rank-one contributions with different preferred angles give a known
        # full-rank weighted cross-product; changing the weights must change O.
        X = np.array([[1., 0.], [0., 1.], [1., 0.]])@self.B.T
        Y = np.array([[1., 0.], [0., 1.], [0., 1.]])@self.B.T
        result = self.fit(X, Y)
        # C = [[1,2],[0,3]] / 6; polar factor is [[4,2],[-2,4]]/sqrt(20).
        expected = np.array([[4., 2.], [-2., 4.]])/np.sqrt(20.)
        np.testing.assert_allclose(result["O"], expected, atol=1e-12, rtol=0)
        changed = self.fit(X, Y, condition_weights=np.array([1., 1., 4.]))
        self.assertGreater(np.linalg.norm(changed["O"]-expected), .1)
        scaled = self.fit(X, Y, condition_weights=1e200*self.weights)
        np.testing.assert_allclose(scaled["O"], expected, atol=1e-12, rtol=0)
        self.assertEqual(result["gain"], 1.)

    def test_unidentified_source_or_cross_product_refuses_without_completion(self):
        rank_one = np.array([[1., 0.], [2., 0.], [3., 0.]])@self.B.T
        for X, Y, message in ((rank_one, self.X, "source design"),
                              (self.X, np.zeros_like(self.X), "not unique"),
                              (self.X, rank_one, "not unique")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.fit(X, Y)
        # Repeated positive singular values still have one polar factor.
        X = np.vstack((self.B.T, np.zeros(4)))
        repeated = self.fit(X, X, condition_weights=np.ones(3))
        np.testing.assert_allclose(repeated["O"], np.eye(2), atol=1e-12, rtol=0)

    def test_global_gain_cannot_use_negative_or_zero_boundary_optimum(self):
        complement = np.array([1., 1., -1., -1.])
        X = np.vstack((self.B.T, self.B[:, 0]))+10.*complement
        Y = np.vstack((self.B.T, self.B[:, 0]))-10.*complement
        with self.assertRaisesRegex(ValueError, "positive-global-gain"):
            self.fit(X, Y, fit_global_gain=True)
        self.assertEqual(self.fit(X, Y)["gain"], 1.)

    def test_invalid_basis_axes_weights_and_numeric_settings_refuse(self):
        for B in (2.*self.B, self.B+1., np.empty((4, 0)), np.ones((3, 2))):
            with self.subTest(B=B.shape), self.assertRaises(ValueError):
                self.fit(self.X, self.X, B=B)
        for weights in ([1., 0., 1.], [1., -1., 2.], [1., np.inf, 1.],
                        [1., 2.], [1+1j, 1., 1.]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.fit(self.X, self.X, condition_weights=weights)
        for options in ({"rank_rtol": 0}, {"rank_rtol": np.nan},
                        {"numerical_atol": True}, {"numerical_atol": 1j},
                        {"fit_global_gain": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.fit(self.X, self.X, **options)
        for Y in (self.X[:, :3], self.X+1j, np.full_like(self.X, np.nan)):
            with self.subTest(shape=Y.shape), self.assertRaises(ValueError):
                self.fit(self.X, Y)


if __name__ == "__main__":
    unittest.main()
