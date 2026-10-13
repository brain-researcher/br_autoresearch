"""Three bounded wrapper checks; run only in coordinator-owned compute.

Complete fabricated role/axis fixtures check assembly and numerical identity,
not empirical benefit or model qualification. The learned-mass check mocks only
the downstream primary fit so independent source-weight fixtures may retain
their smaller native neuron axes. Other methods run the real small assembly.
"""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from lfp_features import FeatureRecipe, fit_lfp_features
from primary_model import PrimaryRecipe, fit_primary_model, predict_primary_development
from regression import RegressionRecipe
from source_weighting import SourceWeightRecipe
from test_population_bridge import full_roster
from test_source_weighting import fixture as weight_fixture, ForbiddenArray
import weighted_primary_model as core


def check():
    pass


def recipe():
    return PrimaryRecipe(FeatureRecipe(quantiles=(.5,)), 4, "orthogonal", None,
                         "explicit", RegressionRecipe("ridge", .1))


def development(target):
    directions = np.repeat(np.arange(8, dtype=np.int64), 6)
    return replace(target, view_role="development_features", role="development",
        trial_ids=tuple(f"trial_{1001 + i:06d}" for i in range(48)),
        directions=directions, lfp=np.zeros((48, 10, 9)), spikes=None, unit_guide=None)


class WeightedPrimaryTests(unittest.TestCase):
    def test_equal_weights_reproduce_ordinary_fit_and_frozen_native_predictions(self):
        days = full_roster()[0]
        chosen = recipe()
        weights = SourceWeightRecipe("equal", target_mass=1., source_total_mass=3.)
        ordinary = fit_primary_model(days, recipe=chosen, check=check)
        # Input arrival order cannot move a source's mass to another day.
        with patch.object(core, "fit_lfp_features", side_effect=AssertionError("unneeded extra feature fit")):
            fitted = core.fit_weighted_primary_model(days[::-1], recipe=chosen,
                                                    weight_recipe=weights, check=check)
        self.assertEqual(fitted.base_recipe, chosen)
        self.assertEqual(fitted.effective_recipe, chosen)
        self.assertEqual(fitted.source_weights.fit_trial_ids, tuple((day.day, day.trial_ids) for day in days))
        self.assertEqual(fitted.primary.day_masses, tuple((day.day, .25) for day in days))
        np.testing.assert_allclose(fitted.primary.regression.coefficients,
                                   ordinary.regression.coefficients, rtol=0, atol=1e-12)
        dev = development(days[-1])
        expected = predict_primary_development(ordinary, dev, check=check)
        with patch.object(core, "fit_source_weights", side_effect=AssertionError("prediction refit weights")), \
                patch.object(core, "fit_primary_model", side_effect=AssertionError("prediction refit model")):
            prediction = core.predict_weighted_primary_development(fitted, dev, check=check)
        self.assertEqual(prediction.trial_ids, expected.trial_ids)
        self.assertEqual(prediction.calibration_trial_ids, days[-1].trial_ids)
        np.testing.assert_array_equal(prediction.unit_guide, expected.unit_guide)
        np.testing.assert_allclose(prediction.values, expected.values, rtol=0, atol=1e-10)
        with self.assertRaises(ValueError):
            prediction.values.setflags(write=True)
        with self.assertRaises(PermissionError):
            core.predict_weighted_primary_development(fitted,
                replace(dev, stream="today_only"), check=check)
        with self.assertRaises(TypeError):
            core.predict_weighted_primary_development(ordinary, dev, check=check)

    def test_learned_masses_bind_source_days_and_refuse_conflicting_or_closed_scope(self):
        days = weight_fixture()
        chosen = recipe()
        weight_recipe = SourceWeightRecipe("reliability", target_mass=2., source_total_mass=5.,
                                           weight_floor=.25, reliability_power=2.)
        for selected in (weight_recipe,
                         replace(weight_recipe, family="hierarchical_shrinkage", shrinkage_strength=120.)):
            sentinel = object()
            with patch.object(core, "fit_primary_model", return_value=sentinel) as fit, \
                    patch.object(core, "fit_lfp_features", side_effect=AssertionError("unexpected feature fit")):
                fitted = core.fit_weighted_primary_model(days[::-1], recipe=chosen,
                                                        weight_recipe=selected, check=check)
            self.assertIs(fitted.primary, sentinel)
            self.assertTrue(all(actual is expected for actual, expected in zip(fit.call_args.args[0], days)))
            r = [1., 0., 4 / 5]
            if selected.family == "hierarchical_shrinkage":
                mean_r = math.fsum(r) / 3
                r = [(n * value + 120 * mean_r) / (n + 120)
                     for n, value in zip((120, 128, 136), r)]
            raw = [(.25 + value)**2 for value in r]
            expected = tuple(5 * value / math.fsum(raw) for value in raw)
            np.testing.assert_allclose(fitted.effective_recipe.source_masses, expected, rtol=0, atol=1e-14)
            self.assertEqual(fitted.effective_recipe.calibration_mass, 2.)
            self.assertEqual(fit.call_args.kwargs["recipe"], fitted.effective_recipe)
            self.assertEqual(fitted.source_weights.fit_trial_ids, tuple((day.day, day.trial_ids) for day in days))
            self.assertEqual(fitted.base_recipe.source_masses, (1., 1., 1.))
        unread = tuple(replace(day, directions=ForbiddenArray(), lfp=ForbiddenArray(),
                               spikes=ForbiddenArray()) for day in days)
        with patch.object(core, "fit_primary_model", side_effect=AssertionError("unexpected primary fit")), \
                patch.object(core, "fit_lfp_features", side_effect=AssertionError("unexpected feature fit")):
            for changed in (replace(chosen, source_masses=(1., 2., 1.)),
                            replace(chosen, calibration_mass=2.),
                            replace(chosen, source_masses=()),
                            replace(chosen, source_masses=(True, 1., 1.))):
                with self.assertRaises(ValueError):
                    core.fit_weighted_primary_model(unread, recipe=changed, weight_recipe=weight_recipe, check=check)
            for changes in (dict(stream="today_only"),
                            dict(view_role="development_features", role="development"),
                            dict(view_role="held_out", role="held_out"), dict(target_day="20140307")):
                invalid = unread[:-1] + (replace(unread[-1], **changes),)
                with self.assertRaises(PermissionError):
                    core.fit_weighted_primary_model(invalid, recipe=chosen, weight_recipe=weight_recipe, check=check)
            with self.assertRaises(ValueError):
                core.fit_weighted_primary_model(unread[-1:], recipe=chosen, weight_recipe=weight_recipe, check=check)
            def cancelled():
                raise RuntimeError("cancel before fitting")
            with self.assertRaisesRegex(RuntimeError, "cancel before fitting"):
                core.fit_weighted_primary_model(unread, recipe=chosen, weight_recipe=weight_recipe, check=cancelled)

    def test_similarity_refits_identical_features_on_exact_history_calibration_scope(self):
        days = []
        for index, day in enumerate(full_roster()[0]):
            repeats = len(day.trial_ids) // 8
            repetition = np.tile(np.arange(repeats), 8) - (repeats - 1) / 2
            delta = index if index < 3 else 0
            lfp = (day.directions[:, None, None] + np.arange(10)[None, :, None]
                   + repetition[:, None, None] + delta + np.zeros((1, 1, 9)))
            days.append(replace(day, lfp=lfp))
        days = tuple(days)
        chosen = recipe()
        weights = SourceWeightRecipe("calibration_similarity", target_mass=1., source_total_mass=3.,
            weight_floor=.1, similarity_temperature=2., feature_scale_floor=1.)
        weight_features, primary_features = [], []
        def fit_for_weights(packs, **kwargs):
            self.assertTrue(all(a is b for a, b in zip(packs, days)))
            value = fit_lfp_features(packs, **kwargs)
            weight_features.append(value)
            return value
        def fit_for_primary(packs, **kwargs):
            self.assertTrue(all(a is b for a, b in zip(packs, days)))
            value = fit_lfp_features(packs, **kwargs)
            primary_features.append(value)
            return value
        with patch.object(core, "fit_lfp_features", side_effect=fit_for_weights), \
                patch("primary_model.fit_lfp_features", side_effect=fit_for_primary):
            fitted = core.fit_weighted_primary_model(days, recipe=chosen, weight_recipe=weights, check=check)
        self.assertEqual((len(weight_features), len(primary_features)), (1, 1))
        before, after = weight_features[0], primary_features[0]
        self.assertIsNot(before, after)
        self.assertIs(fitted.primary.features, after)
        self.assertEqual(before.recipe, after.recipe)
        self.assertEqual(before.fit_trial_ids, after.fit_trial_ids)
        self.assertEqual(before.fit_trial_ids, fitted.source_weights.fit_trial_ids)
        self.assertEqual(before.axes, after.axes)
        np.testing.assert_array_equal(before.center, after.center)
        np.testing.assert_array_equal(before.scale, after.scale)
        distances = [2 / 3 * delta**2 / (85 / 6) for delta in (0, 1, 2)]
        raw = [.1 + math.exp(-distance / 2) for distance in distances]
        np.testing.assert_allclose(fitted.source_weights.prototype_distances, distances, rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.effective_recipe.source_masses,
                                   [3 * value / math.fsum(raw) for value in raw], rtol=0, atol=1e-14)
        np.testing.assert_allclose([mass for _, mass in fitted.primary.day_masses],
            [value / 4 for value in fitted.effective_recipe.source_masses] + [.25], rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
