"""Seconds-scale closed-form checks of the registered encoding arithmetic only."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_identical_fold_encoding as encoding


class IdenticalFoldEncodingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [.6, .8]])
        self.test = np.array([[.8, .6], [-.6, .8]])
        self.y = np.array([[2., -1., 4.], [5., 2., 8.], [-2., 1., 6.],
                           [0., -4., 2.], [1., 3., 7.]])

    def test_matches_augmented_normalized_ridge_with_unpenalized_intercept(self):
        design = encoding.prepare_ridge(self.x, self.test)
        prediction, mean = encoding.predict_block(design, self.y)
        augmented = np.column_stack((np.ones(len(self.x)), self.x))
        penalty = np.diag([0., encoding.LAMBDA, encoding.LAMBDA])
        coefficients = np.linalg.solve(augmented.T @ augmented / len(self.x) + penalty,
                                       augmented.T @ self.y / len(self.x))
        expected = np.column_stack((np.ones(len(self.test)), self.test)) @ coefficients
        np.testing.assert_allclose(prediction, expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(mean, self.y.mean(axis=0))
        # An unnormalised penalty is not this registered estimator.
        wrong = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ self.y)
        self.assertGreater(float(np.max(np.abs(expected-np.column_stack(
            (np.ones(len(self.test)), self.test)) @ wrong))), 1e-5)

    def test_training_only_centering_intercept_and_operands_unchanged(self):
        originals = tuple(value.copy() for value in (self.x, self.test, self.y))
        first = encoding.prepare_ridge(self.x, self.test)
        other = encoding.prepare_ridge(self.x, self.test[::-1])
        np.testing.assert_array_equal(first['training_mean'], other['training_mean'])
        np.testing.assert_array_equal(first['training_centered'], other['training_centered'])
        np.testing.assert_array_equal(np.tril(first['factor'][0]), np.tril(other['factor'][0]))
        prediction, _ = encoding.predict_block(first, self.y)
        shifted, shifted_mean = encoding.predict_block(first, self.y+np.array([7., -2., 11.]))
        np.testing.assert_allclose(shifted, prediction+np.array([7., -2., 11.]), atol=1e-12)
        np.testing.assert_allclose(shifted_mean, self.y.mean(axis=0)+[7., -2., 11.])
        for actual, original in zip((self.x, self.test, self.y), originals):
            np.testing.assert_array_equal(actual, original)

    def test_blockwise_sse_uses_training_mean_and_preserves_negative_or_undefined(self):
        design = encoding.prepare_ridge(self.x, self.test)
        evaluation = np.array([[3., 5., 1.], [-2., 7., 9.]])
        prediction, mean = encoding.predict_block(design, self.y)
        full = encoding.squared_errors(evaluation, prediction, mean)
        blocks = []
        for start, stop in ((0, 1), (1, 3)):
            predicted, baseline = encoding.predict_block(design, self.y[:, start:stop])
            blocks.append(encoding.squared_errors(evaluation[:, start:stop], predicted, baseline))
        np.testing.assert_allclose(np.sum(blocks, axis=0), full, rtol=1e-12, atol=1e-12)
        self.assertEqual(full[1], float(np.sum((evaluation-self.y.mean(axis=0))**2)))
        self.assertNotEqual(full[1], float(np.sum((evaluation-evaluation.mean(axis=0))**2)))
        self.assertEqual(encoding.finish_score(8., 2.), -3.)
        self.assertIsNone(encoding.finish_score(0., 0.))
        self.assertIsNone(encoding.finish_score(float('inf'), 1.))
        self.assertEqual(encoding.squared_errors(evaluation, prediction*np.nan, mean), (None, None))

    def test_training_complement_and_evc_missingness_do_not_prune_evaluation(self):
        images = [{'fold': fold} for fold in (0, 1, 2, 0, 1)]
        validity = [dict(primary_eligible=primary, evc_eligible_on_primary_images=evc)
                    for primary, evc in ((True, True), (True, False), (True, True),
                                         (True, False), (False, False))]
        self.assertEqual(encoding.support_indices(images, validity, 0), ([1, 2], [0, 3], []))
        self.assertEqual(encoding.support_indices(images, validity, 0, evc=True), ([2], [0, 3], [3]))
        with self.assertRaises(ValueError):
            encoding.prepare_ridge(self.x[:1], self.test)
        with self.assertRaises(ValueError):
            encoding.predict_block(encoding.prepare_ridge(self.x, self.test), self.y[:-1])


if __name__ == '__main__':
    unittest.main()
