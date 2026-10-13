"""Independent objective check for the EP08 mean-loss ridge convention."""
import unittest
import numpy as np

from ridge_training import ALPHAS, fit_candidate, new_progress, ridge_coefficients
from development_baseline import score_frozen_model


class MeanLossRidgeTest(unittest.TestCase):
    def test_dual_solve_matches_augmented_least_squares_and_row_replication(self):
        rng = np.random.default_rng(802026)
        x = rng.normal(size=(9, 13))
        x[:, 3] = x[:, 2]  # Exact collinearity does not invalidate ridge.
        y = rng.normal(size=(9, 4))
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        for alpha in (0.0001, 1., 10000.):
            # Direct least squares encodes mean loss using X/sqrt(n), not
            # the pilot's n*alpha Gram construction.
            augmented_x = np.vstack((x / np.sqrt(len(x)), np.sqrt(alpha) * np.eye(x.shape[1])))
            augmented_y = np.vstack((y / np.sqrt(len(y)), np.zeros((x.shape[1], y.shape[1]))))
            expected = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
            actual = ridge_coefficients(x, y, alpha)
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)
            repeated = ridge_coefficients(np.repeat(x, 3, axis=0), np.repeat(y, 3, axis=0), alpha)
            np.testing.assert_allclose(repeated, expected, rtol=2e-9, atol=2e-11)


    def test_training_only_cv_and_training_mean_null_against_direct_reference(self):
        rng = np.random.default_rng(802027)
        x = rng.normal(size=(16, 6))
        x[:, -1] = 4.
        y = 2. + x[:, :3] + rng.normal(size=(16, 3)) * .2
        y[:, -1] = -3.
        folds = [(np.flatnonzero(np.arange(16) % 4 != f), np.flatnonzero(np.arange(16) % 4 == f)) for f in range(4)]

        def direct_fit(a, b, alpha):
            xm, ym = a.mean(axis=0), b.mean(axis=0)
            xs, ys = a.std(axis=0), b.std(axis=0)
            xs[xs == 0], ys[ys == 0] = 1., 1.
            a, b = (a - xm) / xs, (b - ym) / ys
            matrix = np.vstack((a / np.sqrt(len(a)), np.sqrt(alpha) * np.eye(a.shape[1])))
            rhs = np.vstack((b / np.sqrt(len(a)), np.zeros((a.shape[1], b.shape[1]))))
            return xm, xs, ym, ys, np.linalg.lstsq(matrix, rhs, rcond=None)[0]

        errors = []
        for alpha in ALPHAS:
            error = 0.
            for fit, valid in folds:
                xm, xs, ym, ys, coefficients = direct_fit(x[fit], y[fit], alpha)
                error += np.sum(((y[valid] - ym) / ys - ((x[valid] - xm) / xs) @ coefficients) ** 2)
            errors.append(error)
        expected_alpha = ALPHAS[min(range(len(ALPHAS)), key=lambda i: (errors[i], -ALPHAS[i]))]
        progress = new_progress()
        model = fit_candidate(x, y, folds, progress)
        self.assertEqual(model.alpha, expected_alpha)
        expected = direct_fit(x, y, expected_alpha)
        np.testing.assert_allclose(model.coefficients, expected[-1], rtol=2e-9, atol=2e-11)
        self.assertEqual(model.constant_input_columns, (5,))
        self.assertEqual(model.constant_target_columns, (2,))
        self.assertEqual(progress["inner_ridge_fits"], 20)
        self.assertEqual(progress["outer_training_refits"], 1)
        # Evaluation is created only after the model is fitted. Deliberately
        # shifted values distinguish the frozen training-mean null from a
        # silently evaluation-centered denominator.
        ex, ey = x[:4] + 1., y[:4] + 5.
        xm, xs, ym, ys, coefficients = expected
        prediction = ((ex - xm) / xs) @ coefficients
        standardized_target = (ey - ym) / ys
        expected_q = 1. - np.sum((standardized_target - prediction) ** 2) / np.sum(standardized_target ** 2)
        self.assertAlmostEqual(score_frozen_model(model, ex, ey), expected_q, places=12)
        before = model.coefficients.copy()
        score_frozen_model(model, ex, ey * -2.)
        np.testing.assert_array_equal(model.coefficients, before)
        with self.assertRaisesRegex(ArithmeticError, "zero"):
            score_frozen_model(model, ex, np.tile(model.target_mean, (4, 1)))

    def test_inner_fold_cannot_smuggle_an_extra_or_duplicated_row(self):
        x, y = np.ones((8, 2)), np.ones((8, 1))
        folds = [(np.flatnonzero(np.arange(8) % 4 != f), np.flatnonzero(np.arange(8) % 4 == f)) for f in range(4)]
        bad = list(folds)
        bad[0] = (bad[0][0], np.append(bad[0][1], 8))
        with self.assertRaisesRegex(ValueError, "partition"):
            fit_candidate(x, y, bad, new_progress())
        duplicate = [folds[0], folds[0], folds[2], folds[3]]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            fit_candidate(x, y, duplicate, new_progress())


if __name__ == "__main__":
    unittest.main()
