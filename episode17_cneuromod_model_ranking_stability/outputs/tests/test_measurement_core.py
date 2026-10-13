"""Known-answer measurement arithmetic; no source, neural or image access."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import measurement_core as core


def gain_inputs():
    return np.tile([-1., 1.], 8), np.repeat(np.arange(1, 9), 2), np.arange(16)


class LinearEdges(unittest.TestCase):
    def test_support_change_and_development_forecast_share_one_bank(self):
        d = np.array([-.1, .2])
        edge = core.support_edge(d, [.5, .5], [1., 0.])
        self.assertAlmostEqual(edge.start, .05)
        self.assertAlmostEqual(edge.end, -.1)
        self.assertAlmostEqual(edge.change, -.15)
        frozen = core.freeze_linear_forecast(d, edge.operator_weights)
        self.assertAlmostEqual(frozen.development_change, edge.change)
        d[:] = [2., 3.]
        self.assertAlmostEqual(frozen.development_change, -.15)
        self.assertAlmostEqual(frozen.observed_change([.3, -.1]), .2)
        self.assertFalse(frozen.operator_weights.flags.writeable)

    def test_positive_ceiling_weights_can_reverse_only_the_aggregate(self):
        d = np.array([-.1, .2])
        denominator = np.array([.1, 1.])
        edge = core.ceiling_edge(d, denominator)
        self.assertAlmostEqual(edge.start, .05)
        self.assertAlmostEqual(edge.end, -.4)
        self.assertAlmostEqual(edge.change, -.45)
        np.testing.assert_array_equal(np.sign(d), np.sign(d / denominator))
        self.assertNotAlmostEqual(edge.end, float(d.mean() / denominator.mean()))
        frozen = core.freeze_linear_forecast(d, edge.operator_weights)
        self.assertAlmostEqual(frozen.development_change, edge.change)
        # The weights are not renormalized to sum to one.
        self.assertAlmostEqual(core.ceiling_edge([.2, .4], [.1, .1]).end, 3.)

    def test_scale_or_missing_value_errors_do_not_delete_voxels(self):
        for operation in (
            lambda: core.support_edge([1., np.nan], [.5, .5], [.5, .5]),
            lambda: core.support_edge([1., 2.], [.25, .25], [.5, .5]),
            lambda: core.support_edge([1., 2.], [1.1, -.1], [.5, .5]),
            lambda: core.ceiling_edge([1., 2.], [.09, 1.]),
            lambda: core.ceiling_edge([1., 2.], [.1, np.inf]),
        ):
            with self.assertRaises(core.MeasurementError):
                operation()

    def test_participant_and_ROI_means_have_equal_weights(self):
        values = {(s, r): float(10 * i + j) for i, s in enumerate(core.SUBJECTS)
                  for j, r in enumerate(core.ROIS)}
        self.assertEqual(core.equal_participant_roi_mean(values), 15.5)
        del values["sub-06", "right"]
        with self.assertRaisesRegex(core.MeasurementError, "eight_participant_ROIs_required"):
            core.equal_participant_roi_mean(values)


class BetaEdges(unittest.TestCase):
    def test_calibration_gain_design_and_original_ID_median_ties(self):
        gain, bins, columns = gain_inputs()
        design = core.beta_gain_design(gain, bins, columns)
        np.testing.assert_array_equal(design.standardized_gain, gain)
        np.testing.assert_array_equal(design.low_indices, np.arange(0, 16, 2))
        np.testing.assert_array_equal(design.high_indices, np.arange(1, 16, 2))
        self.assertEqual(design.residual_gain_variance, 1.)
        self.assertEqual(core.beta_high_minus_low(5 + 3 * gain, design), 6.)
        permutation = np.arange(16)[::-1]
        shuffled = core.beta_gain_design(gain[permutation], bins[permutation], columns[permutation])
        np.testing.assert_array_equal(columns[permutation][shuffled.low_indices], columns[design.low_indices])
        np.testing.assert_array_equal(columns[permutation][shuffled.high_indices], columns[design.high_indices])
        median_tie = np.r_[-1., 1., np.zeros(14)]
        tied = core.beta_gain_design(median_tie, bins, columns)
        np.testing.assert_array_equal(tied.low_indices, [0, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(tied.high_indices, [9, 10, 11, 12, 13, 14, 15, 1])
        shuffled_tie = core.beta_gain_design(median_tie[permutation], bins[permutation], columns[permutation])
        np.testing.assert_array_equal(columns[permutation][shuffled_tie.low_indices], columns[tied.low_indices])
        np.testing.assert_array_equal(columns[permutation][shuffled_tie.high_indices], columns[tied.high_indices])

    def test_constant_bin_only_and_missing_bin_gain_are_inapplicable(self):
        gain, bins, columns = gain_inputs()
        with self.assertRaisesRegex(core.MeasurementError, "calibration_gain_variance_zero"):
            core.beta_gain_design(np.full(16, .1), bins, columns)
        with self.assertRaisesRegex(core.MeasurementError, "residual_gain_variance_below_floor"):
            core.beta_gain_design(bins.astype(float), bins, columns)
        with self.assertRaisesRegex(core.MeasurementError, "eight_nonempty_spatial_bins_required"):
            core.beta_gain_design(gain, np.minimum(bins, 7), columns)
        with self.assertRaisesRegex(core.MeasurementError, "gain_identity_invalid"):
            core.beta_gain_design(gain, bins, np.zeros(16, dtype=int))

    def test_twelve_twelve_crossfit_and_pooled_effect_are_distinct(self):
        gain, bins, columns = gain_inputs()
        design = core.beta_gain_design(gain, bins, columns)
        spatial = (bins - 1).astype(float)
        blocks = spatial + 2 * gain + np.arange(24)[:, None]
        fit = core.fit_beta_forecast(blocks, design)
        np.testing.assert_allclose(fit.crossfit_predicted_block_means[:12], 21., atol=1e-12)
        np.testing.assert_allclose(fit.crossfit_predicted_block_means[12:], 9., atol=1e-12)
        np.testing.assert_allclose(fit.crossfit_observed_block_means, np.arange(24) + 3.5, atol=1e-12)
        np.testing.assert_allclose(fit.crossfit_absolute_errors,
                                   np.abs(fit.crossfit_observed_block_means - fit.crossfit_predicted_block_means))
        self.assertAlmostEqual(fit.development_block_mean_forecast, 15., places=12)
        self.assertAlmostEqual(fit.coefficients[-1], 2., places=12)
        # Intercepts make aggregate prediction equal the training target mean.
        self.assertAlmostEqual(fit.development_block_mean_forecast, float(blocks.mean()), places=12)
        self.assertEqual(core.beta_high_minus_low(100 + 3 * gain, design), 6.)
        self.assertNotEqual(fit.development_block_mean_forecast, 100.)

    def test_beta_target_must_keep_all_24_blocks_and_all_voxels(self):
        gain, bins, columns = gain_inputs()
        design = core.beta_gain_design(gain, bins, columns)
        with self.assertRaisesRegex(core.MeasurementError, "24_development_blocks_required"):
            core.fit_beta_forecast(np.zeros((12, 16)), design)
        blocks = np.zeros((24, 16))
        blocks[23, 15] = np.nan
        with self.assertRaisesRegex(core.MeasurementError, "block_stage_changes_nonfinite"):
            core.fit_beta_forecast(blocks, design)


if __name__ == "__main__":
    unittest.main()
