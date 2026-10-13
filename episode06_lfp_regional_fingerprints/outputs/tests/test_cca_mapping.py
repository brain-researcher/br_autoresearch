"""Two small independent CCA-map checks; execute only in coordinated Slurm."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cca_mapping as core


def check():
    pass


def signs():
    return (np.asarray((-1., -1., -1., -1., 1., 1., 1., 1.)),
            np.asarray((-1., -1., 1., 1., -1., -1., 1., 1.)),
            np.asarray((-1., 1., -1., 1., -1., 1., -1., 1.)))


def diagonal_fixture():
    u, v, w = signs()
    x = np.column_stack((5. + 2. * u, -7. + v))
    y = np.column_stack((11. + 3. * u, -13. + v + 4. * w))
    return x, y


def fit(x, y):
    return core.fit_cca_mapping(x, y, covariance_regularization=.01, prediction_penalty=.01, check=check)


class CCAMappingTests(unittest.TestCase):
    def test_independent_diagonal_covariance_canonical_scale_and_rank_one_ridge_oracle(self):
        x, y = diagonal_fixture()
        fitted = fit(x, y)
        # Uniform-row covariances are diag(4,1), diag(9,17), cross diag(6,1).
        # The leading canonical direction is therefore exactly coordinate1.
        expected_a = 1. / math.sqrt(4.01)
        expected_gamma = (6. / math.sqrt(4.01)) / (4. / 4.01 + .01)
        expected_coefficient = 6. / (4. + .01 * 4.01)
        np.testing.assert_allclose(fitted.x_mean, (5., -7.), rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.y_mean, (11., -13.), rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.x_whitener, np.diag((1. / math.sqrt(4.01), 1. / math.sqrt(1.01))),
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.y_whitener, np.diag((1. / math.sqrt(9.01), 1. / math.sqrt(17.01))),
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.canonical_singular_values,
            (6. / math.sqrt(4.01 * 9.01), 1. / math.sqrt(1.01 * 17.01)), rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.canonical_direction, (expected_a, 0.), rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.score_coefficients, (expected_gamma, 0.), rtol=0, atol=1e-13)
        np.testing.assert_allclose(fitted.coefficients, ((expected_coefficient, 0.), (0., 0.)),
                                   rtol=0, atol=1e-13)
        np.testing.assert_allclose(fitted.intercept, (11. - 5. * expected_coefficient, -13.), rtol=0, atol=1e-13)
        expected = np.column_stack((11. + (x[:, 0] - 5.) * expected_coefficient, np.full(len(x), -13.)))
        np.testing.assert_allclose(core.predict_cca_mapping(fitted, x, check=check), expected, rtol=0, atol=1e-13)
        self.assertEqual((fitted.training_rows, fitted.x_numerical_rank), (8, 2))
        self.assertGreater(fitted.canonical_direction[np.argmax(np.abs(fitted.canonical_direction))], 0)
        self.assertAlmostEqual(float(fitted.canonical_direction @ np.diag((4.01, 1.01))
                                     @ fitted.canonical_direction), 1., places=13)
        # Unit-variance rescaling would change the requested scalar ridge;
        # ordinary full-rank ridge would also recover the second Y direction.
        self.assertGreater(abs(expected_coefficient - 1.5 / 1.01), 1e-5)
        self.assertEqual(fitted.coefficients[1, 1], 0.)
        self.assertGreater(1. / 1.01, .9)

    def test_frozen_deployment_shift_zero_association_semidefinite_y_and_invalid_support(self):
        x, y = diagonal_fixture()
        fitted = fit(x, y)
        source = x.copy()
        baseline = core.predict_cca_mapping(fitted, source, check=check)
        shift = np.asarray((20., -30.))
        shifted = core.predict_cca_mapping(fitted, source + shift, check=check)
        expected_change = np.broadcast_to(shift @ fitted.coefficients, baseline.shape)
        np.testing.assert_allclose(shifted - baseline, expected_change, rtol=0, atol=1e-13)
        x[:] = -999.
        y[:] = 999.
        np.testing.assert_array_equal(core.predict_cca_mapping(fitted, source, check=check), baseline)
        for value in (fitted.x_mean, fitted.coefficients, fitted.canonical_direction, baseline):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.setflags(write=True)
        u, v, w = signs()
        independent_x = np.column_stack((u, v))
        # Y has rank1 but zero cross-covariance with both input coordinates.
        independent_y = np.column_stack((10. + w, -3. + 2. * w))
        zero = fit(independent_x, independent_y)
        np.testing.assert_array_equal(zero.canonical_singular_values, (0., 0.))
        np.testing.assert_array_equal(zero.coefficients, np.zeros((2, 2)))
        np.testing.assert_array_equal(core.predict_cca_mapping(zero, independent_x + shift, check=check),
                                       np.broadcast_to((10., -3.), (8, 2)))
        constant = fit(independent_x, np.broadcast_to((6., -4.), (8, 2)))
        np.testing.assert_array_equal(constant.coefficients, np.zeros((2, 2)))
        # Equal positive canonical singular values use the returned SVD mode;
        # no extra spectral-gap exclusion or alternate-mode search occurs.
        tied = fit(independent_x, independent_x)
        self.assertTrue(np.isfinite(tied.coefficients).all())
        self.assertEqual(tied.canonical_singular_values[0], tied.canonical_singular_values[1])
        for bad_x, bad_y in ((independent_x[:-1], independent_y),
                             (np.column_stack((u, u)), independent_y),
                             (np.zeros((8, 3)), independent_y),
                             (np.full((8, 2), np.nan), independent_y),
                             (independent_x, np.full((8, 2), np.inf)),
                             (independent_x.astype(complex), independent_y)):
            with self.assertRaises(ValueError):
                fit(bad_x, bad_y)
        for covreg, penalty in ((0., .01), (.01, 0.), (True, .01), (.01, math.inf)):
            with self.assertRaises(ValueError):
                core.fit_cca_mapping(independent_x, independent_y,
                    covariance_regularization=covreg, prediction_penalty=penalty, check=check)
        with self.assertRaises(ValueError):
            core.predict_cca_mapping(fitted, np.zeros((2, 3)), check=check)


if __name__ == "__main__":
    unittest.main()
