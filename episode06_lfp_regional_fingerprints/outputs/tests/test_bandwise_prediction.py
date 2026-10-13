"""Three independent arithmetic/access checks; execute only in coordinated Slurm.

Tiny fabricated arrays test this numerical branch, not source eligibility,
matched regional support, an empirical profile or scientific recovery.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from component_scoring import FrozenDesign, BoundActivity
import bandwise_prediction as core


def check():
    pass


def fixture(*, role="training", id_start=1, signal_multiplier=1, observation_shift=0):
    directions = np.repeat(np.arange(8), 2)
    bins = np.arange(1, 11)
    signal = np.tile([-1, 1], 8)[:, None] * bins[None, :]
    base = 100 + 10 * directions[:, None] + bins[None, :]
    z = base + signal_multiplier * signal + observation_shift
    spikes = np.stack((20 + z, 30 + 2 * z), axis=2)
    values = np.empty((16, 10, 15, 9))
    electrodes = np.arange(-7, 8)
    for band in range(9):
        x = 10 * band + (band + 1) * signal_multiplier * signal if band < 8 else np.full_like(signal, 80)
        values[:, :, :, band] = x[:, :, None] + electrodes[None, None, :]
    design = FrozenDesign("Mihili_CO_20140217.mat", role, tuple(range(8)),
        tuple(f"trial_{index:06d}" for index in range(id_start, id_start + 16)),
        tuple(int(value) for value in directions), tuple(range(5, 15)), ("unit_a", "unit_b"))
    return (core.BandwiseLFP("M1", design, tuple(range(1, 16)), core.BANDS, values),
            BoundActivity(design, spikes), signal, base)


def recipe(rank=1):
    return core.BandwiseRecipe(rank, .5, (
        core.FeatureBlock("lmp", core.BANDS[:1]),
        core.FeatureBlock("low_group", core.BANDS[1:3]),
        core.FeatureBlock("constant_high", core.BANDS[-1:]),
        core.FeatureBlock("full", core.BANDS)))


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("values accessed before role/identity binding")


class BandwisePredictionTests(unittest.TestCase):
    def test_analytic_pca_training_template_and_correlated_multiband_ridge(self):
        lfp, observed, signal, base = fixture()
        fitted = core.fit_bandwise(lfp, observed, recipe=recipe(), check=check)
        # The entire two-neuron population lies on the known [1,2] axis.
        expected_axis = np.asarray([[1.], [2.]]) / math.sqrt(5)
        np.testing.assert_allclose(fitted.loadings, expected_axis, rtol=0, atol=2e-15)
        np.testing.assert_allclose(fitted.loadings.T @ fitted.loadings, [[1.]], rtol=0, atol=2e-15)
        expected_template = np.stack((20 + base[::2], 30 + 2 * base[::2]), axis=2)
        np.testing.assert_array_equal(fitted.training_template, expected_template)
        np.testing.assert_allclose(fitted.population_mean,
            [20 + float(np.mean(base)), 30 + 2 * float(np.mean(base))], rtol=0, atol=1e-13)
        self.assertEqual(fitted.numerical_rank, 1)
        np.testing.assert_array_equal(fitted.predictor_mean, np.arange(9) * 10)
        np.testing.assert_allclose(fitted.predictor_scale[:8],
            np.arange(1, 9) * math.sqrt(38.5), rtol=1e-14, atol=0)
        self.assertEqual(fitted.predictor_scale[8], 1.)
        residual = core.transform_observations(fitted, observed, region="M1", check=check)
        truth = math.sqrt(5) * signal[:, :, None]
        np.testing.assert_allclose(residual.values, truth, rtol=0, atol=3e-14)
        predicted = core.predict_bandwise(fitted, lfp, check=check)
        # k identical standardized columns give shrinkage k/(k+lambda).
        # The ninth predictor is constant and remains present with coefficient0.
        for block, prediction, k in zip(fitted.blocks, predicted.predictions, (1, 2, 0, 8)):
            np.testing.assert_allclose(prediction.values, truth * k / (k + .5), rtol=0, atol=2e-12)
            np.testing.assert_allclose(block.intercept, [0.], rtol=0, atol=2e-14)
            self.assertEqual(prediction.design, residual.design)
        self.assertEqual(fitted.blocks[-1].coefficients.shape, (9, 1))
        self.assertAlmostEqual(fitted.blocks[-1].coefficients[-1, 0], 0., places=13)
        # Supplied storage order need not be frequency order. Move the constant
        # high band to position0 so positional selection would change LMP/group.
        order = (8, 2, 0, 1, 3, 4, 5, 6, 7)
        reordered = replace(lfp, bands=tuple(lfp.bands[index] for index in order),
                            values=lfp.values[:, :, :, order])
        reordered_fit = core.fit_bandwise(reordered, observed, recipe=recipe(), check=check)
        reordered_prediction = core.predict_bandwise(reordered_fit, reordered, check=check)
        for left, right in zip(predicted.predictions, reordered_prediction.predictions):
            np.testing.assert_allclose(left.values, right.values, rtol=0, atol=2e-12)
        for array in (fitted.loadings, fitted.training_template, fitted.predictor_scale,
                      fitted.blocks[0].coefficients, predicted.predictions[0].values, residual.values):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_feature_only_prediction_and_training_origin_preserve_new_cell_mean_shifts(self):
        lfp, observed, _, _ = fixture()
        fitted = core.fit_bandwise(lfp, observed, recipe=recipe(), check=check)
        evaluation, outcome, signal, _ = fixture(role="evaluation", id_start=101,
                                               signal_multiplier=3, observation_shift=7)
        before = tuple(array.copy() for array in (fitted.loadings, fitted.training_template,
                                                  fitted.predictor_mean, fitted.predictor_scale))
        prediction = core.predict_bandwise(fitted, evaluation, check=check)
        self.assertFalse(hasattr(evaluation, "spikes"))
        self.assertFalse(hasattr(prediction, "scores"))
        for value, k in zip(prediction.predictions, (1, 2, 0, 8)):
            np.testing.assert_allclose(value.values,
                (3 * math.sqrt(5) * signal[:, :, None]) * k / (k + .5), rtol=0, atol=6e-12)
        transformed = core.transform_observations(fitted, outcome, region="M1", check=check)
        np.testing.assert_allclose(transformed.values,
            math.sqrt(5) * (3 * signal[:, :, None] + 7), rtol=0, atol=5e-14)
        # Evaluation-centered component E would remove this genuine shift.
        self.assertAlmostEqual(float(np.mean(transformed.values)), 7 * math.sqrt(5), places=12)
        changed_outcome = replace(outcome, values=outcome.values + np.asarray([11, 22]))
        core.transform_observations(fitted, changed_outcome, region="M1", check=check)
        again = core.predict_bandwise(fitted, evaluation, check=check)
        for left, right in zip(prediction.predictions, again.predictions):
            np.testing.assert_array_equal(left.values, right.values)
        for left, right in zip(before, (fitted.loadings, fitted.training_template,
                                      fitted.predictor_mean, fitted.predictor_scale)):
            np.testing.assert_array_equal(left, right)

    def test_roles_complete_support_and_identity_refusals_precede_values_without_rank_fallback(self):
        lfp, observed, _, _ = fixture()
        fitted = core.fit_bandwise(lfp, observed, recipe=recipe(), check=check)
        unread = replace(lfp, values=ForbiddenArray())
        unread_y = replace(observed, values=ForbiddenArray())
        for role in ("calibration", "evaluation"):
            design = replace(lfp.design, role=role)
            with self.assertRaises(PermissionError):
                core.fit_bandwise(replace(unread, design=design), replace(unread_y, design=design),
                                  recipe=recipe(), check=check)
        with self.assertRaises(ValueError):
            core.fit_bandwise(unread, replace(unread_y, design=replace(lfp.design, axis_ids=("b", "a"))),
                              recipe=recipe(), check=check)
        for bands in ((core.BANDS[0], core.BANDS[1]), (core.BANDS[1], core.BANDS[3]),
                      ((1., 4.),)):
            with self.assertRaises(ValueError):
                core.fit_bandwise(unread, unread_y, recipe=core.BandwiseRecipe(1, .5,
                    (core.FeatureBlock("invalid", bands),)), check=check)
        with self.assertRaisesRegex(ValueError, "numerical training rank"):
            core.fit_bandwise(lfp, observed, recipe=recipe(rank=2), check=check)
        evaluation, _, _, _ = fixture(role="evaluation", id_start=101)
        evaluation = replace(evaluation, values=ForbiddenArray())
        overlap = replace(evaluation.design,
            trial_ids=(lfp.design.trial_ids[0],) + evaluation.design.trial_ids[1:])
        with self.assertRaises(PermissionError):
            core.predict_bandwise(fitted, replace(evaluation, design=overlap), check=check)
        with self.assertRaises(PermissionError):
            core.transform_observations(fitted, BoundActivity(overlap, ForbiddenArray()),
                                        region="M1", check=check)
        with self.assertRaises(PermissionError):
            core.transform_observations(fitted, BoundActivity(evaluation.design, ForbiddenArray()),
                                        region="PMd", check=check)
        for wrong in (replace(evaluation, electrode_ids=evaluation.electrode_ids[::-1]),
                      replace(evaluation, electrode_ids=evaluation.electrode_ids[:-1]),
                      replace(evaluation, bands=evaluation.bands[::-1]),
                      replace(evaluation, bands=evaluation.bands[:-1]),
                      replace(evaluation, design=replace(evaluation.design, time_ids=evaluation.design.time_ids[::-1])),
                      replace(evaluation, design=replace(evaluation.design, axis_ids=("unit_b", "unit_a")))):
            with self.assertRaises(ValueError):
                core.predict_bandwise(fitted, wrong, check=check)
        for wrong in (replace(evaluation, region="PMd"),
                      replace(evaluation, design=replace(evaluation.design, session_id="another_session"))):
            with self.assertRaises(PermissionError):
                core.predict_bandwise(fitted, wrong, check=check)
        bad = lfp.values.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            core.fit_bandwise(replace(lfp, values=bad), observed, recipe=recipe(), check=check)
        def cancelled():
            raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            core.predict_bandwise(fitted, evaluation, check=cancelled)


if __name__ == "__main__":
    unittest.main()
