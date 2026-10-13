"""Tiny exact-roster array fixtures; run only in coordinator-owned compute."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import calibration_core as core


def repeated_fixture(unequal=False, width=1):
    responses, orders = [], []
    for image in range(120):
        deviations = np.array([-2., 0., 2.]) if unequal and image % 2 else np.array([-1., 1.])
        responses.append(np.repeat((image + deviations)[:, None], width, axis=1))
        # Gaps and identical parity ensure halves follow retained order, not
        # the parity of a source repetition/acquisition identifier.
        orders.append(np.arange(1, len(deviations) + 1, dtype=np.int64) * 10)
    return responses, orders


class RepeatTests(unittest.TestCase):
    def test_unequal_repeats_equal_image_weight_and_known_ceiling(self):
        responses, orders = repeated_fixture(unequal=True)
        summary = core.summarize_repeats(responses, orders)
        np.testing.assert_array_equal(summary.image_means[:, 0], np.arange(120))
        np.testing.assert_array_equal(summary.repeat_counts, [2, 3] * 60)
        np.testing.assert_array_equal(summary.half_counts, [[1, 1], [2, 1]] * 60)
        np.testing.assert_array_equal(summary.within_image_sample_variances[:, 0], [2., 4.] * 60)
        np.testing.assert_array_equal(summary.half_1_means[:, 0], np.arange(120) - np.tile([1., 0.], 60))
        np.testing.assert_array_equal(summary.half_2_means[:, 0], np.arange(120) + np.tile([1., 0.], 60))
        result = core.calibrate(responses, orders)
        # Var_sample(0,...,119)=120*121/12=1210; equal-image mean
        # of v_i/n_i is mean(1,4/3)=7/6, not a repeat-weighted mean.
        np.testing.assert_allclose(result.image_mean_variance.sample_variance, [1210.], rtol=0, atol=0)
        np.testing.assert_allclose(result.mean_sampling_noise_variance, [7 / 6], rtol=1e-14, atol=0)
        np.testing.assert_allclose(result.noise_ceiling, [1 - 7 / 7260], rtol=1e-14, atol=0)
        self.assertTrue(result.ceiling_valid[0])
        self.assertFalse(result.repeat_counts.flags.writeable)

    def test_acquisition_sorting_retains_all_rows_and_alternates_retained_positions(self):
        responses, orders = repeated_fixture(unequal=True)
        responses[1] = np.array([[1.], [1.], [10.]])
        expected = core.summarize_repeats(responses, orders)
        self.assertEqual(expected.image_means[1, 0], 4.)
        self.assertEqual(expected.half_1_means[1, 0], 5.5)
        self.assertEqual(expected.half_2_means[1, 0], 1.)
        self.assertNotEqual(expected.image_means[1, 0],
                            (expected.half_1_means[1, 0] + expected.half_2_means[1, 0]) / 2)
        shuffled_values, shuffled_order = [], []
        for values, keys in zip(responses, orders):
            permutation = np.array([2, 0, 1]) if len(keys) == 3 else np.array([1, 0])
            shuffled_values.append(values[permutation])
            shuffled_order.append(keys[permutation])
        actual = core.summarize_repeats(shuffled_values, shuffled_order)
        for name in ("repeat_counts", "half_counts", "image_means", "within_image_sample_variances",
                     "half_1_means", "half_2_means"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_roster_repeat_count_voxel_width_and_duplicate_order_errors(self):
        responses, orders = repeated_fixture()
        with self.assertRaisesRegex(core.CalibrationError, "at_least_120"):
            core.calibrate(responses[:119], orders[:119])
        with self.assertRaisesRegex(core.CalibrationError, "too_short"):
            core.calibrate([responses[0][:1], *responses[1:]], [orders[0][:1], *orders[1:]])
        with self.assertRaisesRegex(core.CalibrationError, "axis_width_changed"):
            core.calibrate([np.ones((2, 2)), *responses[1:]], orders)
        for invalid in (np.array([10, 10]), np.array([-1, 2]), np.array([1., 2.]), np.array([1])):
            with self.subTest(order=invalid), self.assertRaisesRegex(core.CalibrationError, "order_invalid"):
                core.calibrate(responses, [invalid, *orders[1:]])


class EligibilityTests(unittest.TestCase):
    def test_nonfinite_voxel_is_flagged_without_dropping_any_images(self):
        for invalid in (np.nan, np.inf, -np.inf):
            responses, orders = repeated_fixture(width=2)
            responses[17][0, 1] = invalid
            summary = core.summarize_repeats(responses, orders)
            self.assertEqual(summary.image_means.shape, (120, 2))
            self.assertTrue(np.isnan(summary.image_means[:, 1]).all())
            result = core.calibrate(responses, orders)
            self.assertEqual(result.image_count, 120)
            np.testing.assert_array_equal(result.response_eligible, [True, False])
            np.testing.assert_array_equal(result.reliability_valid, [True, False])
            np.testing.assert_array_equal(result.ceiling_valid, [True, False])
            for values in (result.pearson_r, result.fisher_z, result.noise_ceiling, result.ceiling_denominator):
                self.assertTrue(np.isfinite(values[0]))
                self.assertTrue(np.isnan(values[1]))

    def test_image_mean_variance_not_pooled_trial_variance_defines_eligibility(self):
        responses = [np.array([[2.], [4.]]) for _ in range(120)]
        orders = [np.array([10, 20]) for _ in range(120)]
        # Every image has nonzero trial variance, but every image mean is3.
        result = core.calibrate(responses, orders)
        self.assertFalse(result.response_eligible[0])
        self.assertFalse(result.reliability_valid[0])
        self.assertFalse(result.ceiling_valid[0])
        self.assertTrue(np.isnan(result.noise_ceiling[0]))

    def test_relative_variance_rule_rescales_and_keeps_degeneracy_explicit(self):
        pattern = np.tile([-1., 0., 1.], 40)
        values = np.column_stack((1e6 + pattern, 1e6 + 10 * pattern,
                                  pattern * 1e-30, np.full(120, .1)))
        first = core.qualify_variance(values)
        second = core.qualify_variance(values * 8)
        np.testing.assert_array_equal(first.valid, [False, True, True, False])
        np.testing.assert_array_equal(second.valid, first.valid)
        self.assertFalse(first.positive_variance[3])
        self.assertTrue(np.isnan(first.population_variance[[0, 3]]).all())
        np.testing.assert_allclose(second.population_variance[1:3], first.population_variance[1:3] * 64)

    def test_degenerate_half_does_not_invalidate_an_otherwise_finite_ceiling(self):
        responses = [np.array([[0.], [2. * image]]) for image in range(120)]
        orders = [np.array([10, 20]) for _ in range(120)]
        result = core.calibrate(responses, orders)
        self.assertTrue(result.response_eligible[0])
        self.assertTrue(result.ceiling_valid[0])
        self.assertFalse(result.half_1_variance_valid[0])
        self.assertTrue(result.half_2_variance_valid[0])
        self.assertFalse(result.reliability_valid[0])
        self.assertTrue(np.isnan(result.pearson_r[0]))


class StatisticTests(unittest.TestCase):
    def test_Pearson_known_orthogonal_construction_and_Fisher_transform(self):
        x = np.tile([-1., -1., 1., 1.], 30)
        orthogonal = np.tile([-1., 1., -1., 1.], 30)
        result = core.repeat_reliability(x[:, None], (x + orthogonal)[:, None]).transformed
        self.assertTrue(result.valid[0])
        self.assertAlmostEqual(result.pearson_r[0], 1 / np.sqrt(2), places=14)
        self.assertAlmostEqual(result.fisher_z[0], np.arctanh(1 / np.sqrt(2)), places=14)
        self.assertFalse(result.fisher_boundary_adjusted[0])

    def test_Fisher_roundoff_tolerance_and_adjustment_do_not_clip_ranking_values(self):
        result = core.fisher_transform([1., -1., 0., 1 + .5e-12, -1 - .5e-12,
                                       1 + 2e-12, np.nan, .99])
        np.testing.assert_array_equal(result.valid, [True, True, True, True, True, False, False, True])
        np.testing.assert_array_equal(result.roundoff_to_boundary, [False, False, False, True, True, False, False, False])
        np.testing.assert_array_equal(result.fisher_boundary_adjusted, [True, True, False, True, True, False, False, False])
        self.assertEqual(result.pearson_r[0], 1.)
        self.assertEqual(result.pearson_r[1], -1.)
        self.assertAlmostEqual(result.fisher_z[0], np.arctanh(1 - 1e-7), places=14)
        self.assertTrue(np.isnan(result.pearson_r[5:7]).all())

    def test_negative_ceiling_is_retained_and_only_denominator_is_floored(self):
        pattern = np.tile([-1., 0., 1.], 40)
        responses = [np.array([[mean - 1], [mean + 1]]) for mean in pattern]
        orders = [np.array([10, 20]) for _ in range(120)]
        result = core.calibrate(responses, orders)
        # Across-image unbiased variance=80/119; mean(v_i/n_i)=1.
        self.assertTrue(result.ceiling_valid[0])
        self.assertAlmostEqual(result.noise_ceiling[0], -39 / 80, places=14)
        self.assertEqual(result.ceiling_denominator[0], .10)
        self.assertEqual(result.fisher_boundary_adjusted_count, 1)
        scaled = core.calibrate([values * 8 for values in responses], orders)
        np.testing.assert_allclose(scaled.noise_ceiling, result.noise_ceiling, rtol=0, atol=1e-14)
        np.testing.assert_allclose(scaled.pearson_r, result.pearson_r, rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
