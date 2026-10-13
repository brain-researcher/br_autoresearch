"""Two fixed interface/arithmetic checks; execute only in coordinated Slurm.

The complete tiny native fixture is inherited from the existing CCA check.
Its orthogonal signs make both ridge shrinkages independently calculable;
these arrays are not an empirical comparison or recovery study.
"""
from dataclasses import replace
import math
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bandwise_prediction as core
import primary_profile as profile
from test_bandwise_families import cca_fixture, cca_recipe
from test_bandwise_prediction import ForbiddenArray


def check():
    pass


def recipe():
    return replace(cca_recipe(), feature_family="pca2_ridge", covariance_regularization=None)


class PCARidgeTests(unittest.TestCase):
    def test_both_mode_ridge_oracle_frozen_shift_profile_and_self_pickle_roundtrip(self):
        training, observed, _, _ = cca_fixture()
        fitted = core.fit_bandwise(training, observed, recipe=recipe(), check=check)
        self.assertIsInstance(fitted, core.FittedPCARidgeBandwise)
        self.assertIsInstance(fitted, core.FrozenPopulationContext)
        self.assertFalse(hasattr(fitted, "predictor_mean"))
        self.assertIsNone(fitted.recipe.covariance_regularization)
        np.testing.assert_allclose(fitted.loadings, np.eye(2), rtol=0, atol=2e-14)
        # Standardized electrode covariance has eigenvalues8 and6, while
        # native population responses are3*u andv. Ridge fits BOTH signals.
        gains = np.array((8. / (8. + .01), 6. / (6. + .01)))
        coefficients = np.diag((3 * math.sqrt(8) / (8 + .01),
                                math.sqrt(6) / (6 + .01)))
        for block in fitted.blocks:
            self.assertFalse(hasattr(block, "mapping"))
            np.testing.assert_allclose(block.coefficients, coefficients, rtol=0, atol=2e-13)
            np.testing.assert_allclose(block.intercept, np.zeros(2), rtol=0, atol=2e-13)
            self.assertEqual(block.electrode_scale[-1], 1.)
            for value in (block.coefficients, block.intercept, block.electrode_scale, block.loadings):
                with self.assertRaises(ValueError):
                    value.setflags(write=True)
        evaluation, outcome, u, v = cca_fixture(evaluation=True)
        expected = np.stack((3 * gains[0] * u, gains[1] * v), axis=2)
        truth = np.stack((3 * u + 7, v + 11), axis=2)
        energy = math.fsum(float(value)**2 for value in truth.flat)
        error = math.fsum((float(y) - float(p))**2 for y, p in zip(truth.flat, expected.flat))
        predictions = core.predict_bandwise(fitted, evaluation, check=check)
        scored = profile.score_band_profile(fitted, outcome, predictions, check=check)
        self.assertTrue(scored.available)
        for prediction, score in zip(predictions.predictions, scored.blocks):
            np.testing.assert_allclose(prediction.values, expected, rtol=0, atol=1e-11)
            self.assertAlmostEqual(score.score, 1 - error / energy, places=12)
        # This uses only freshly self-produced bytes. There is no external
        # pickle loader or empirical serialized-artifact read in the test.
        restored = pickle.loads(pickle.dumps(fitted, protocol=5))
        self.assertIs(type(restored), core.FittedPCARidgeBandwise)
        self.assertEqual(restored.recipe, fitted.recipe)
        for before, after in zip(predictions.predictions,
                core.predict_bandwise(restored, evaluation, check=check).predictions):
            np.testing.assert_array_equal(after.values, before.values)
        # The old CCA class remains its own state and its one-mode prediction
        # remains the original closed-form reference after the common branch.
        cca = core.fit_bandwise(training, observed, recipe=cca_recipe(), check=check)
        cca = pickle.loads(pickle.dumps(cca, protocol=5))
        self.assertIs(type(cca), core.FittedCCABandwise)
        cca_gain = 8. / (8. + .01 * 8.01)
        for prediction in core.predict_bandwise(cca, evaluation, check=check).predictions:
            np.testing.assert_allclose(prediction.values,
                np.stack((3 * cca_gain * u, np.zeros_like(v)), axis=2), rtol=0, atol=1e-11)

    def test_complete_frozen_roles_axes_and_rank_without_residual_rank_requirement(self):
        training, observed, _, _ = cca_fixture()
        unread = replace(training, values=ForbiddenArray())
        unread_y = replace(observed, values=ForbiddenArray())
        for bad in (replace(recipe(), rank=1), replace(recipe(), covariance_regularization=.01),
                    replace(recipe(), blocks=recipe().blocks[:-1]),
                    replace(recipe(), penalty=0), replace(recipe(), penalty=True)):
            with self.assertRaises(ValueError):
                core.fit_bandwise(unread, unread_y, recipe=bad, check=check)
        for role in ("calibration", "evaluation"):
            design = replace(training.design, role=role)
            with self.assertRaises(PermissionError):
                core.fit_bandwise(replace(unread, design=design), replace(unread_y, design=design),
                                  recipe=recipe(), check=check)
        with self.assertRaises(ValueError):
            core.fit_bandwise(unread, replace(unread_y,
                design=replace(observed.design, axis_ids=observed.design.axis_ids[::-1])),
                recipe=recipe(), check=check)
        deficient = training.values.copy()
        deficient[:, :, 8:14, 0] = 0
        with self.assertRaisesRegex(ValueError, "electrode rank 2 exceeds numerical training rank"):
            core.fit_bandwise(replace(training, values=deficient), observed, recipe=recipe(), check=check)
        # Population covariance is rank2, but direction/bin means explain all
        # training counts: zero residuals are a supported zero mapping.
        directions = np.asarray(training.design.trial_directions)
        counts = np.stack((100 + 3 * (directions % 2), 200 + directions // 2), axis=1)
        counts = np.broadcast_to(counts[:, None, :], observed.design.shape).copy()
        zero = core.fit_bandwise(training, replace(observed, values=counts), recipe=recipe(), check=check)
        for prediction in core.predict_bandwise(zero, training, check=check).predictions:
            np.testing.assert_array_equal(prediction.values, np.zeros(prediction.design.shape))
        evaluation, _, _, _ = cca_fixture(evaluation=True)
        evaluation = replace(evaluation, values=ForbiddenArray())
        for changed in (replace(evaluation, electrode_ids=evaluation.electrode_ids[::-1]),
                        replace(evaluation, bands=evaluation.bands[::-1])):
            with self.assertRaises(ValueError):
                core.predict_bandwise(zero, changed, check=check)
        with self.assertRaises(PermissionError):
            core.predict_bandwise(zero, replace(evaluation,
                design=replace(evaluation.design, trial_ids=training.design.trial_ids)), check=check)
        with self.assertRaises(ValueError):
            core.predict_bandwise(replace(zero, recipe=cca_recipe()), evaluation, check=check)
        def stopped():
            raise TimeoutError("caller cancellation")
        with self.assertRaises(TimeoutError):
            core.predict_bandwise(zero, evaluation, check=stopped)


if __name__ == "__main__":
    unittest.main()
