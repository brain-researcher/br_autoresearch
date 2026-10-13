"""Independent augmented-objective and finite-feature kernel references."""
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from capacity_training import rank_projection, training_kernel
from ridge_training import ridge_coefficients


class CapacityMathTest(unittest.TestCase):
    def test_regularized_reduced_rank_matches_augmented_projection(self):
        rng = np.random.default_rng(483)
        x, y, alpha = rng.normal(size=(13, 5)), rng.normal(size=(13, 4)), .17
        xa = np.vstack((x, np.sqrt(len(x)*alpha)*np.eye(x.shape[1])))
        ya = np.vstack((y, np.zeros((x.shape[1], y.shape[1]))))
        independent = np.linalg.lstsq(xa, ya, rcond=None)[0]
        _, _, vt = np.linalg.svd(xa @ independent, full_matrices=False)
        base = ridge_coefficients(x, y, alpha)
        for rank in (1, 2, 4):
            expected = independent @ vt[:rank].T @ vt[:rank]
            actual = rank_projection(x, y, base, rank)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
            self.assertAlmostEqual(np.sum((ya-xa@actual)**2), np.sum((ya-xa@expected)**2), places=10)
        self.assertIs(rank_projection(x, y, base, "full"), base)

    def test_centered_kernel_matches_explicit_finite_feature_ridge_and_constant_limit(self):
        rng = np.random.default_rng(484)
        x, xv, y, alpha = rng.normal(size=(9, 3)), rng.normal(size=(4, 3)), rng.normal(size=(9, 2)), .23
        kernel, state = training_kernel(x)
        raw = np.exp(-cdist(xv, x, metric="sqeuclidean")/(2*state.sigma_squared))
        cross = raw-state.column_mean[None, :]-raw.mean(axis=1)[:, None]+state.grand_mean
        mean = y.mean(axis=0)
        got = mean+cross @ np.linalg.solve(kernel+len(x)*alpha*np.eye(len(x)), y-mean)
        whole = np.vstack((x, xv))
        gram = np.exp(-cdist(whole, whole, metric="sqeuclidean")/(2*state.sigma_squared))
        values, vectors = np.linalg.eigh(gram)
        phi = vectors*np.sqrt(np.maximum(values, 0))[None, :]
        centered = phi-phi[:len(x)].mean(axis=0)
        xa = np.vstack((centered[:len(x)], np.sqrt(len(x)*alpha)*np.eye(len(whole))))
        ya = np.vstack((y-mean, np.zeros((len(whole), y.shape[1]))))
        expected = mean+centered[len(x):] @ np.linalg.lstsq(xa, ya, rcond=None)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-11)
        constant, constant_state = training_kernel(np.ones((9, 3)))
        np.testing.assert_array_equal(constant, np.zeros((9, 9)))
        self.assertEqual(constant_state.sigma_squared, 0.)


if __name__ == "__main__":
    unittest.main()
