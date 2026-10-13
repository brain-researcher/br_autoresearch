"""Small known-answer array fixtures; coordinator executes these in Slurm."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prediction_core as core


def diagonal_features(rank):
    """Distinct known singular values, nonzero means, one constant feature."""
    width = rank + 2
    scales = np.arange(width, 0, -1, dtype=np.float64)
    offset = np.arange(width, dtype=np.float64) / 4
    features = np.vstack((np.diag(scales), -np.diag(scales))) + offset
    return np.column_stack((features, np.full(2 * width, .1))), scales, offset


class PCATests(unittest.TestCase):
    def test_all_three_ranks_known_loadings_and_global_score_scaling(self):
        for rank in core.PCA_RANKS:
            features, scales, offset = diagonal_features(rank)
            with self.subTest(rank=rank):
                fitted = core.fit_pca(features, rank)
                expected_loading = np.eye(features.shape[1], rank)
                expected_mean = np.r_[offset, .1]
                expected_scale = np.sqrt(features.shape[0] / (2 * np.sum(scales[:rank] ** 2)))
                np.testing.assert_allclose(fitted.feature_mean, expected_mean, rtol=0, atol=1e-14)
                np.testing.assert_allclose(fitted.loadings, expected_loading, rtol=0, atol=1e-13)
                self.assertAlmostEqual(fitted.score_scale, expected_scale, places=14)
                scores = fitted.transform(features)
                np.testing.assert_allclose(scores, (features[:, :rank] - offset[:rank]) * expected_scale,
                                           rtol=1e-12, atol=1e-13)
                self.assertAlmostEqual(float(np.sum(scores * scores)), features.shape[0], places=10)
                # One scalar preserves unequal PCA energies; this is not whitening.
                self.assertGreater(float(np.var(scores[:, 0])), float(np.var(scores[:, -1])))
                self.assertEqual(fitted.numerical_rank, rank + 2)
                self.assertFalse(fitted.loadings.flags.writeable)

    def test_test_outlier_uses_only_training_mean_basis_and_scale(self):
        features, _, _ = diagonal_features(32)
        fitted = core.fit_pca(features, 32)
        before = (fitted.feature_mean.copy(), fitted.loadings.copy(), fitted.score_scale)
        heldout = np.vstack((fitted.feature_mean + 1000, fitted.feature_mean - 250))
        expected = (heldout - before[0]) @ before[1] * before[2]
        np.testing.assert_allclose(fitted.transform(heldout), expected, rtol=0, atol=0)
        np.testing.assert_array_equal(fitted.feature_mean, before[0])
        np.testing.assert_array_equal(fitted.loadings, before[1])
        self.assertGreater(float(np.linalg.norm(fitted.transform(heldout).mean(axis=0))), 0)

    def test_positive_global_feature_rescaling_preserves_scores(self):
        features, _, _ = diagonal_features(32)
        heldout = features[:4] + np.arange(features.shape[1])
        first = core.fit_pca(features, 32)
        second = core.fit_pca(features * 8 + 17, 32)
        np.testing.assert_allclose(first.transform(heldout), second.transform(heldout * 8 + 17),
                                   rtol=1e-11, atol=1e-11)

    def test_constant_effective_rank_and_exact_degenerate_cutoff_refuse(self):
        with self.assertRaisesRegex(core.PredictionError, "PCA_rank_unavailable"):
            core.fit_pca(np.full((68, 34), .1), 32)
        scales = np.r_[np.arange(31, 0, -1), 1e-14]
        too_small = np.vstack((np.diag(scales), -np.diag(scales)))
        with self.assertRaisesRegex(core.PredictionError, "PCA_rank_unavailable"):
            core.fit_pca(too_small, 32)
        equal = np.vstack((np.eye(33), -np.eye(33)))
        with self.assertRaisesRegex(core.PredictionError, "PCA_cutoff_exactly_degenerate"):
            core.fit_pca(equal, 32)


class RidgeTests(unittest.TestCase):
    def test_closed_form_n_times_penalty_and_unpenalized_target_mean(self):
        scores = np.array([[10.], [12.]])
        targets = np.array([[7.], [11.]])
        fitted = core.fit_ridge(scores, targets, 1.0)
        np.testing.assert_array_equal(fitted.score_mean, [11.])
        np.testing.assert_array_equal(fitted.target_mean, [9.])
        # Centered X'X=2, X'Y=4; n*lambda=2, so coefficient=1.
        np.testing.assert_allclose(fitted.coefficients, [[1.]], rtol=0, atol=0)
        np.testing.assert_allclose(fitted.predict([[9.], [11.], [13.]]), [[7.], [9.], [11.]], rtol=0, atol=0)
        heavy = core.fit_ridge(scores, targets, 100.0)
        np.testing.assert_array_equal(heavy.predict([[11.]]), [[9.]])
        self.assertFalse(fitted.coefficients.flags.writeable)

    def test_target_affine_rescaling_preserves_predictions_and_R2_relation(self):
        scores = np.array([[-2.], [-1.], [1.], [2.]])
        targets = np.array([[1., 2.], [2., 4.], [4., 7.], [6., 10.]])
        heldout = np.array([[.5], [3.]])
        observed = np.array([[3., 6.], [8., 12.]])
        fit = core.fit_ridge(scores, targets, .1)
        changed = core.fit_ridge(scores, targets * 3 + 100, .1)
        predicted = fit.predict(heldout)
        np.testing.assert_allclose(changed.predict(heldout), predicted * 3 + 100, rtol=1e-13, atol=1e-13)
        original = core.pooled_r2([core.error_sums(observed, predicted, fit.target_mean)])
        rescaled = core.pooled_r2([core.error_sums(observed * 3 + 100, changed.predict(heldout), changed.target_mean)])
        np.testing.assert_allclose(original, rescaled, rtol=1e-12, atol=1e-12)

    def test_fixed_menu_and_exact_tie_order(self):
        options = [("layer2", 32, 100.), ("layer1", 64, 100.),
                   ("layer1", 32, .0001), ("layer1", 32, 100.)]
        self.assertEqual(sorted(options, key=lambda x: core.hyperparameter_tie_key(*x)),
                         [options[3], options[2], options[1], options[0]])
        for option in (("avgpool", 32, 1), ("layer1", 16, 1), ("layer1", 32, 0),
                       ("layer1", 32, .2)):
            with self.subTest(option=option), self.assertRaises(core.PredictionError):
                core.hyperparameter_tie_key(*option)
        self.assertEqual(len(core.LAYERS) * len(core.PCA_RANKS) * len(core.RIDGE_PENALTIES), 84)


class ScoreTests(unittest.TestCase):
    def test_training_baseline_not_test_mean_and_negative_R2_retained(self):
        sums = core.error_sums([[2.], [4.]], [[1.], [3.]], [0.])
        np.testing.assert_array_equal(sums.sse, [2.])
        np.testing.assert_array_equal(sums.sst, [20.])
        np.testing.assert_allclose(core.pooled_r2([sums]), [.9], rtol=0, atol=0)
        negative = core.pooled_r2([core.error_sums([[1.]], [[20.]], [0.])])
        np.testing.assert_array_equal(negative, [-360.])

    def test_pooled_sums_not_fold_mean_and_row_specific_training_baselines(self):
        first = core.error_sums([[0.], [2.]], [[1.], [1.]], [0.])
        second = core.error_sums([[10.]], [[0.]], [0.])
        pooled = core.pooled_r2([first, second])
        np.testing.assert_allclose(pooled, [1 - 102 / 104], rtol=0, atol=0)
        self.assertNotEqual(float(pooled[0]), .25)
        all_rows = core.error_sums([[1.], [4.], [9.]], [[0.], [3.], [7.]], [[0.], [2.], [8.]])
        np.testing.assert_array_equal(all_rows.sst, [6.])
        partitions = [core.error_sums([[1.]], [[0.]], [0.]),
                      core.error_sums([[4.]], [[3.]], [2.]),
                      core.error_sums([[9.]], [[7.]], [8.])]
        np.testing.assert_array_equal(core.pooled_r2([all_rows]), core.pooled_r2(partitions))

    def test_missing_prediction_or_zero_denominator_refuses_whole_score(self):
        with self.assertRaisesRegex(core.PredictionError, "predicted_nonfinite"):
            core.error_sums([[1., 2.], [2., 3.]], [[1., np.nan], [2., 3.]], [0., 0.])
        with self.assertRaisesRegex(core.PredictionError, "pooled_denominator_invalid"):
            core.pooled_r2([core.error_sums([[2., 3.]], [[2., 3.]], [2., 0.])])
        with self.assertRaisesRegex(core.PredictionError, "training_baseline_shape_mismatch"):
            core.error_sums([[1., 2.], [2., 3.]], [[1., 2.], [2., 3.]], [[0., 0.]])

    def test_CKA_known_answer_global_scale_and_cross_voxel_scale_distinction(self):
        # Centered columns are orthogonal and have the same squared norm2.
        x = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        y = x[:, :1]
        self.assertAlmostEqual(core.linear_cka(x, y), 1 / np.sqrt(2), places=14)
        self.assertAlmostEqual(core.linear_cka(x, x), 1, places=14)
        self.assertAlmostEqual(core.linear_cka(x * 5 + 17, y * 3 - 99), core.linear_cka(x, y), places=14)
        # Scaling only one response voxel is not a CKA invariance.
        differently_scaled = x * np.array([1., 4.])
        self.assertLess(core.linear_cka(x, differently_scaled), .8)
        with self.assertRaisesRegex(core.PredictionError, "CKA_denominator_or_numerator_invalid"):
            core.linear_cka(x, np.full((4, 2), .1))


if __name__ == "__main__":
    unittest.main()
