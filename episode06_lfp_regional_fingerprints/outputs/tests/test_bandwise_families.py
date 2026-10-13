"""Small arithmetic/role integration checks; run only in coordinated Slurm.

These fabricated arrays test a frozen interface, not empirical performance,
source qualification, candidate selection or an outcome-opening permission.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bandwise_prediction as core
import primary_profile as profile
from component_scoring import BoundActivity, FrozenDesign
from test_bandwise_prediction import fixture, recipe, ForbiddenArray


def check():
    pass


def cca_recipe():
    return core.BandwiseRecipe(2, .01,
        tuple(core.FeatureBlock(f"band_{index}", (band,)) for index, band in enumerate(core.BANDS)),
        feature_family="pca2_cca", covariance_regularization=.01)


def cca_fixture(*, evaluation=False, region="M1"):
    # All four orthogonal sign combinations occur equally within each direction
    # and at every bin. Population covariance is diag(9,1), with identity PCs.
    directions = np.repeat(np.arange(8), 4)
    u = np.tile((-1., -1., 1., 1.), 8)[:, None] * np.ones((1, 10))
    v = np.tile((-1., 1., -1., 1.), 8)[:, None] * np.ones((1, 10))
    if evaluation:
        u, v = 2 * u + 1, 3 * v - 2
    lfp = np.empty((32, 10, 15, 9))
    for band in range(9):
        for electrode in range(15):
            signal = u if electrode < 8 else (v if electrode < 14 else np.zeros_like(u))
            lfp[:, :, electrode, band] = 100 * band + 5 * electrode + (band + 1) * (electrode + 1) * signal
    spikes = np.stack((100 + 3 * u + (7 if evaluation else 0),
                       200 + v + (11 if evaluation else 0)), axis=2)
    start = 1001 if evaluation else 1
    design = FrozenDesign("Mihili_CO_20140217.mat", "evaluation" if evaluation else "training",
        tuple(range(8)), tuple(f"trial_{index:06d}" for index in range(start, start + 32)),
        tuple(int(value) for value in directions), tuple(range(5, 15)), ("neuron_a", "neuron_b"))
    return core.BandwiseLFP(region, design, tuple(range(1, 16)), core.BANDS, lfp), BoundActivity(design, spikes), u, v


class BandwiseFamilyTests(unittest.TestCase):
    def test_median_grouped_ridge_rejects_seven_electrode_outliers_with_known_shrinkage(self):
        training, observed, _, _ = fixture()
        evaluation, outcome, signal, _ = fixture(role="evaluation", id_start=101,
                                                 signal_multiplier=3, observation_shift=7)
        def contaminated(lfp):
            values = lfp.values.copy()
            # Eight clean offsets[-7..0] and seven large positive electrodes
            # give the original clean band signal as the exact median.
            direction = np.asarray(lfp.design.trial_directions)
            values[:, :, 8:, :] += (1000 + 100 * direction)[:, None, None, None]
            return replace(lfp, values=values)
        training, evaluation = contaminated(training), contaminated(evaluation)
        median = core.fit_bandwise(training, observed,
            recipe=replace(recipe(), feature_family="median_ridge"), check=check)
        mean = core.fit_bandwise(training, observed, recipe=recipe(), check=check)
        self.assertIsInstance(median, core.FrozenPopulationContext)
        self.assertIsInstance(median, core.FittedBandwise)
        np.testing.assert_array_equal(median.predictor_mean, np.arange(9) * 10)
        predictions = core.predict_bandwise(median, evaluation, check=check)
        actual = profile.score_band_profile(median, outcome, predictions, check=check)
        truth = math.sqrt(5) * (3 * signal + 7)
        energy = math.fsum(float(value)**2 for value in truth.flat)
        for result, prediction, k in zip(actual.blocks, predictions.predictions, (1, 2, 0, 8)):
            expected = 3 * math.sqrt(5) * signal * k / (k + .5)
            np.testing.assert_allclose(prediction.values[:, :, 0], expected, rtol=0, atol=6e-12)
            error = math.fsum((float(y) - float(p))**2 for y, p in zip(truth.flat, expected.flat))
            self.assertAlmostEqual(result.score, 1 - error / energy, places=12)
        mean_prediction = core.predict_bandwise(mean, evaluation, check=check)
        self.assertGreater(np.max(np.abs(mean_prediction.predictions[0].values
                                         - predictions.predictions[0].values)), 1.)
        self.assertEqual(mean.recipe.feature_family, "mean_ridge")
        self.assertIsNone(mean.recipe.covariance_regularization)

    def test_pca2_cca_independent_diagonal_oracle_frozen_shift_and_paired_profile(self):
        training, observed, _, _ = cca_fixture()
        fitted = core.fit_bandwise(training, observed, recipe=cca_recipe(), check=check)
        self.assertIsInstance(fitted, core.FrozenPopulationContext)
        self.assertIsInstance(fitted, core.FittedCCABandwise)
        self.assertFalse(hasattr(fitted, "predictor_mean"))  # no artificial ridge state
        np.testing.assert_allclose(fitted.loadings, np.eye(2), rtol=0, atol=2e-14)
        np.testing.assert_array_equal(fitted.training_template, np.tile((100., 200.), (8, 10, 1)))
        expected_electrode_axes = np.zeros((15, 2))
        expected_electrode_axes[:8, 0] = 1 / math.sqrt(8)
        expected_electrode_axes[8:14, 1] = 1 / math.sqrt(6)
        gain = 8. / (8. + .01 * 8.01)
        for index, block in enumerate(fitted.blocks):
            np.testing.assert_allclose(block.electrode_mean, 100 * index + 5 * np.arange(15), rtol=0, atol=1e-12)
            np.testing.assert_allclose(block.electrode_scale[:14], (index + 1) * np.arange(1, 15), rtol=0, atol=1e-12)
            self.assertEqual(block.electrode_scale[14], 1.)
            np.testing.assert_allclose(block.loadings, expected_electrode_axes, rtol=0, atol=5e-14)
            np.testing.assert_allclose(block.mapping.canonical_direction, (1 / math.sqrt(8.01), 0.),
                                       rtol=0, atol=5e-13)
            np.testing.assert_allclose(block.mapping.canonical_singular_values,
                (3 * math.sqrt(8) / math.sqrt(8.01 * 9.01),
                 math.sqrt(6) / math.sqrt(6.01 * 1.01)), rtol=0, atol=5e-13)
        evaluation, outcome, u, v = cca_fixture(evaluation=True)
        expected = np.stack((3 * gain * u, np.zeros_like(v)), axis=2)
        predictions = core.predict_bandwise(fitted, evaluation, check=check)
        truth = np.stack((3 * u + 7, v + 11), axis=2)
        energy = math.fsum(float(value)**2 for value in truth.flat)
        error = math.fsum((float(y) - float(p))**2 for y, p in zip(truth.flat, expected.flat))
        scored = profile.score_band_profile(fitted, outcome, predictions, check=check)
        self.assertTrue(scored.available)
        for prediction, score in zip(predictions.predictions, scored.blocks):
            np.testing.assert_allclose(prediction.values, expected, rtol=0, atol=1e-11)
            self.assertAlmostEqual(score.score, 1 - error / energy, places=12)
        # A storage reorder retains physical singleton identities. Refit only
        # the same training fixture; evaluation moments must never be fitted.
        order = (8, 2, 0, 1, 3, 4, 5, 6, 7)
        reordered = replace(training, region="PMd", bands=tuple(training.bands[i] for i in order),
                            values=training.values[:, :, :, order])
        other_fit = core.fit_bandwise(reordered, observed, recipe=cca_recipe(), check=check)
        other_features = replace(evaluation, region="PMd", bands=reordered.bands,
                                 values=evaluation.values[:, :, :, order])
        other_predictions = core.predict_bandwise(other_fit, other_features, check=check)
        other_score = profile.score_band_profile(other_fit, outcome, other_predictions, check=check)
        np.testing.assert_allclose(profile.paired_profile(scored, other_score, check=check).m1_minus_pmd,
                                   np.zeros(9), rtol=0, atol=1e-13)
        for array in (fitted.blocks[0].electrode_mean, fitted.blocks[0].loadings,
                      fitted.blocks[0].mapping.coefficients, predictions.predictions[0].values):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_cca_role_axis_and_rank_refusals_before_values_with_rank_zero_residual_allowed(self):
        training, observed, _, _ = cca_fixture()
        guarded = replace(training, values=ForbiddenArray())
        guarded_y = replace(observed, values=ForbiddenArray())
        for bad in (replace(cca_recipe(), rank=1), replace(cca_recipe(), covariance_regularization=None),
                    replace(cca_recipe(), blocks=cca_recipe().blocks[:-1]),
                    replace(cca_recipe(), feature_family="unregistered")):
            with self.assertRaises(ValueError):
                core.fit_bandwise(guarded, guarded_y, recipe=bad, check=check)
        for role in ("calibration", "evaluation"):
            design = replace(training.design, role=role)
            with self.assertRaises(PermissionError):
                core.fit_bandwise(replace(guarded, design=design), replace(guarded_y, design=design),
                                  recipe=cca_recipe(), check=check)
        bad = training.values.copy()
        bad[:, :, 8:14, 0] = 0  # one entire declared band now supports only PC1
        with self.assertRaisesRegex(ValueError, "electrode rank 2 exceeds numerical training rank"):
            core.fit_bandwise(replace(training, values=bad), observed, recipe=cca_recipe(), check=check)
        # Unresidualized spikes have two distinct direction-dependent axes,
        # while each training direction/bin residual is exactly zero.
        direction = np.asarray(training.design.trial_directions)
        counts = np.stack((100 + 3 * (direction % 2), 200 + direction // 2), axis=1)
        counts = np.broadcast_to(counts[:, None, :], observed.design.shape).copy()
        zero = core.fit_bandwise(training, replace(observed, values=counts), recipe=cca_recipe(), check=check)
        for prediction in core.predict_bandwise(zero, training, check=check).predictions:
            np.testing.assert_array_equal(prediction.values, np.zeros(prediction.design.shape))
        evaluation, _, _, _ = cca_fixture(evaluation=True)
        evaluation = replace(evaluation, values=ForbiddenArray())
        with self.assertRaises(ValueError):
            core.predict_bandwise(zero, replace(evaluation, electrode_ids=evaluation.electrode_ids[::-1]), check=check)
        with self.assertRaises(PermissionError):
            core.predict_bandwise(zero, replace(evaluation,
                design=replace(evaluation.design, trial_ids=training.design.trial_ids)), check=check)
        def cancelled():
            raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            core.predict_bandwise(zero, evaluation, check=cancelled)


if __name__ == "__main__":
    unittest.main()
