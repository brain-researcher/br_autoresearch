"""Four focused control-refit checks; execute only in coordinated Slurm.

Complete fabricated role/axis packs exercise arithmetic and access boundaries,
not empirical control success, known-answer scientific recoveries or authority
to open held-out outcomes. The existing native_day/latent_cells fixture provides
four orthogonal count directions with independent day identities. No real
payload is read; these methods remain unrun until the coordinator executes them.
"""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from lfp_features import FeatureRecipe, fit_lfp_features, transform_lfp_features
from native_trials import DayTrials
from primary_model import PrimaryRecipe, fit_primary_model
from regression import RegressionRecipe
from source_weighting import SourceWeightRecipe, fit_source_weights
from test_population_bridge import native_day, latent_cells
from weighted_primary_model import fit_weighted_primary_model
import primary_control as core


def check():
    pass


class UnreadDay(DayTrials):
    def __getattribute__(self, name):
        if name in ("lfp", "lfp_guide", "spikes", "unit_guide", "directions", "bin_offsets"):
            raise AssertionError("an omitted or refused pack's arrays were accessed")
        return super().__getattribute__(name)


def fixture(*, split_slopes=False):
    days = []
    for index in range(4):
        day = native_day(index, (index + 1) * np.diag([12., 6., 3., 1.]))
        constant = (0., 1., 2., 0.)[index] if split_slopes else 10. * index
        day = replace(day, lfp=np.full_like(day.lfp, constant, dtype=np.float64))
        if split_slopes and index in (1, 2):
            repeat = np.tile(np.arange(15), 8)
            slope = np.where(repeat % 2 == 0, 1., -1. if index == 1 else 2.)
            center = 100. + 20. * index
            day = replace(day, spikes=center + (day.spikes - center) * slope[:, None, None])
        days.append(day)
    return tuple(days)


def recipe(*, histogram=False, stream="history_assisted", adjustment="explicit"):
    feature = (FeatureRecipe(family="robust_histogram", histogram_bins=4,
                            histogram_quantiles=(0., 1.), scaling="robust")
               if histogram else FeatureRecipe(quantiles=(.5,)))
    return PrimaryRecipe(feature, 4, "orthogonal" if stream == "history_assisted" else "none",
        None, adjustment, RegressionRecipe("ridge", 0.),
        source_masses=(1., 1., 1.) if stream == "history_assisted" else ())


def development(target):
    directions = np.repeat(np.arange(8, dtype=np.int64), 6)
    return replace(target, role="development", view_role="development_features",
        trial_ids=tuple(f"trial_{1001 + index:06d}" for index in range(48)),
        directions=directions, lfp=np.zeros((48, 10, 9)), spikes=None, unit_guide=None)


def expected_native(dev, gain):
    # PCA axes and orthogonal maps are the identity for the distinct positive
    # diagonal loadings. Each balanced day contributes its equal-cell mean;
    # the target's own raw-count origin is 160 regardless of source origins.
    cells = latent_cells().reshape(8, 10, 4)[dev.directions]
    return 160. + gain * cells * np.asarray([12., 6., 3., 1.])


class PrimaryControlTests(unittest.TestCase):
    def test_each_chronological_omission_refits_all_state_and_preserves_mass_ratio(self):
        days = fixture()
        chosen = replace(recipe(histogram=True), source_masses=(1., 2., 4.), calibration_mass=3.)
        for omitted in range(3):
            unread = tuple(UnreadDay(**vars(day)) if index == omitted else day
                           for index, day in enumerate(days))
            fitted = core.fit_primary_control(unread[::-1], recipe=chosen,
                control=core.PrimaryControl("omit_source", omitted + 1), check=check)
            kept = [index for index in range(3) if index != omitted]
            retained_total = math.fsum(chosen.source_masses[index] for index in kept)
            masses = [7 * chosen.source_masses[index] / retained_total for index in kept]
            np.testing.assert_allclose(fitted.effective_recipe.source_masses, masses, rtol=0, atol=1e-14)
            self.assertEqual(fitted.effective_recipe.calibration_mass, 3.)
            expected_days = tuple(days[index].day for index in kept + [3])
            self.assertEqual(fitted.retained_source_days, expected_days[:-1])
            self.assertEqual(tuple(day for day, _ in fitted.primary.day_masses), expected_days)
            np.testing.assert_allclose([mass for _, mass in fitted.primary.day_masses],
                                       [mass / 10 for mass in masses] + [.3], rtol=0, atol=1e-14)
            self.assertEqual(tuple(basis.day for basis in fitted.primary.population.bases), expected_days)
            self.assertEqual(len(fitted.primary.population.source_maps), 2)
            self.assertEqual(tuple(means.day for means in fitted.primary.design.day_means), expected_days)
            self.assertEqual(tuple(day for day, _ in fitted.primary.features.fit_trial_ids), expected_days)
            self.assertEqual(tuple(day for day, _ in fitted.primary.rows_per_day), expected_days)
            # Histogram edges must actually refit after removing the minimum
            # source; no original feature bounds or fitted state are inherited.
            for edge in fitted.primary.features.histogram_edges:
                np.testing.assert_array_equal(edge, np.linspace(10 * min(kept), 30., 5))
            dev = development(days[-1])
            # Use target LFP so no between-day feature effect is extrapolated.
            dev = replace(dev, lfp=np.full_like(dev.lfp, 30.))
            gain = (math.fsum(mass * (index + 1) for mass, index in zip(masses, kept)) + 3 * 4) / 10
            actual = core.predict_primary_control_development(fitted, dev, check=check)
            np.testing.assert_allclose(actual.values, expected_native(dev, gain), rtol=0, atol=2e-8)
            with self.assertRaises(PermissionError):
                transform_lfp_features(fitted.primary.features, unread[omitted], check=check)

    def test_remove_history_refits_own_calibration_architecture_and_skips_source_weights(self):
        days = fixture()
        unread = tuple(UnreadDay(**vars(day)) for day in days[:-1]) + days[-1:]
        chosen = recipe(histogram=True)
        weights = SourceWeightRecipe("reliability", target_mass=2., source_total_mass=5.,
                                     weight_floor=.1, reliability_power=1.)
        with patch("weighted_primary_model.fit_source_weights",
                   side_effect=AssertionError("zero-source control fitted weights")):
            fitted = core.fit_primary_control(unread, recipe=chosen, weight_recipe=weights,
                control=core.PrimaryControl("remove_history"), check=check)
        self.assertEqual(fitted.original_recipe, chosen)
        self.assertEqual(fitted.weight_recipe, weights)
        self.assertEqual(fitted.effective_recipe.source_masses, ())
        self.assertEqual(fitted.effective_recipe.calibration_mass, 2.)
        self.assertIsNone(fitted.source_weights)
        self.assertEqual(fitted.retained_source_days, ())
        self.assertEqual(fitted.primary.day_masses, ((days[-1].day, 1.),))
        self.assertEqual(fitted.primary.features.stream, "history_assisted")
        self.assertEqual(fitted.primary.population.alignment, "orthogonal")
        self.assertEqual(fitted.primary.population.source_maps, ())
        self.assertEqual(len(fitted.primary.population.bases), 1)
        self.assertEqual(fitted.primary.features.fit_trial_ids, ((days[-1].day, days[-1].trial_ids),))
        for edge in fitted.primary.features.histogram_edges:
            np.testing.assert_array_equal(edge, np.full(5, 30.))
        dev = replace(development(days[-1]), lfp=np.full((48, 10, 9), 30.))
        prediction = core.predict_primary_control_development(fitted, dev, check=check)
        np.testing.assert_allclose(prediction.values, expected_native(dev, 4.), rtol=0, atol=2e-8)
        for invalid in (replace(dev, stream="today_only"),
                        replace(dev, role="held_out", view_role="held_out"),
                        replace(dev, trial_ids=days[-1].trial_ids + dev.trial_ids[:24])):
            with self.assertRaises(PermissionError):
                core.predict_primary_control_development(fitted, invalid, check=check)
        with self.assertRaises(TypeError):
            core.fit_primary_control(unread, recipe=fitted.primary,
                control=core.PrimaryControl("remove_history"), check=check)

    def test_remove_nuisance_keeps_raw_origin_without_cell_effects_and_ordinary_grammar_stays_strict(self):
        days = fixture()
        target = replace(days[-1], stream="today_only")
        for original in ("explicit", "residualize", "both"):
            chosen = recipe(stream="today_only", adjustment=original)
            fitted = core.fit_primary_control((target,), recipe=chosen,
                control=core.PrimaryControl("remove_nuisance"), check=check)
            self.assertEqual(fitted.effective_recipe.adjustment, "none")
            self.assertEqual(len(fitted.primary.design.axes), len(fitted.primary.features.axes))
            self.assertIsNone(fitted.primary.design.day_means[0].feature_means)
            self.assertIsNone(fitted.primary.design.day_means[0].response_means)
            dev = development(target)
            # Constant features carry no labels. The fitted unpenalized
            # intercept predicts the raw grand mean, not zero or a cell mean.
            for input_day in (dev, replace(dev, directions=(dev.directions + 1) % 8)):
                actual = core.predict_primary_control_development(fitted, input_day, check=check)
                np.testing.assert_allclose(actual.values, np.full((48, 10, 4), 160.),
                                           rtol=0, atol=2e-9)
        unread = tuple(UnreadDay(**vars(day)) for day in days)
        with self.assertRaises(ValueError):
            fit_primary_model(unread[1:], recipe=replace(recipe(), source_masses=(1., 1.)), check=check)
        with self.assertRaises(ValueError):
            fit_primary_model(unread[-1:], recipe=replace(recipe(), source_masses=()), check=check)
        for fit in (fit_primary_model, fit_weighted_primary_model):
            kwargs = {} if fit is fit_primary_model else {
                "weight_recipe": SourceWeightRecipe("equal", 1., 3.)}
            with self.assertRaises(ValueError):
                fit(unread, recipe=replace(recipe(), adjustment="none"), check=check, **kwargs)
        for control in (core.PrimaryControl("omit_source", 0), core.PrimaryControl("omit_source", True),
                        core.PrimaryControl("remove_history", 1), core.PrimaryControl("arbitrary_subset")):
            with self.assertRaises(ValueError):
                core.fit_primary_control(unread, recipe=recipe(), control=control, check=check)
        with self.assertRaises(PermissionError):
            core.fit_primary_control((UnreadDay(**vars(target)),), recipe=recipe(stream="today_only"),
                control=core.PrimaryControl("remove_history"), check=check)
        with self.assertRaises(PermissionError):
            core.fit_primary_control(unread[:-1] + (UnreadDay(**vars(replace(days[-1], role="held_out", view_role="held_out"))),),
                recipe=recipe(), control=core.PrimaryControl("remove_history"), check=check)
        def cancelled():
            raise RuntimeError("cancel before values")
        with self.assertRaisesRegex(RuntimeError, "cancel before values"):
            core.fit_primary_control(unread, recipe=recipe(),
                control=core.PrimaryControl("remove_history"), check=cancelled)

    def test_two_source_learned_weights_refit_reliability_shrinkage_and_similarity(self):
        days = fixture(split_slopes=True)
        unread = (UnreadDay(**vars(days[0])),) + days[1:]
        chosen = recipe()
        control = core.PrimaryControl("omit_source", 1)
        recipes = (
            SourceWeightRecipe("equal", 2., 5.),
            SourceWeightRecipe("reliability", 2., 5., weight_floor=.25, reliability_power=2.),
            SourceWeightRecipe("hierarchical_shrinkage", 2., 5., weight_floor=.25,
                               reliability_power=2., shrinkage_strength=120.),
            SourceWeightRecipe("calibration_similarity", 2., 5., weight_floor=.1,
                               similarity_temperature=2., feature_scale_floor=1.))
        for weight_recipe in recipes:
            fitted = core.fit_primary_control(unread, recipe=chosen, control=control,
                                             weight_recipe=weight_recipe, check=check)
            if weight_recipe.family == "equal":
                raw = [1., 1.]
            elif weight_recipe.family == "calibration_similarity":
                distances = [2 / 3, 8 / 3]
                raw = [.1 + math.exp(-value / 2) for value in distances]
                np.testing.assert_allclose(fitted.source_weights.prototype_distances, distances,
                                           rtol=0, atol=1e-14)
            else:
                values = [0., .8]
                np.testing.assert_allclose(fitted.source_weights.reliability, values, rtol=0, atol=1e-14)
                if weight_recipe.family == "hierarchical_shrinkage":
                    # The refitted prior is the mean of TWO retained sources.
                    values = [.2, .6]  # n_eff=kappa=120, retained mean=.4.
                    np.testing.assert_allclose(fitted.source_weights.shrunk_reliability, values,
                                               rtol=0, atol=1e-14)
                raw = [(.25 + value)**2 for value in values]
            masses = [5 * value / math.fsum(raw) for value in raw]
            np.testing.assert_allclose(fitted.effective_recipe.source_masses, masses, rtol=0, atol=1e-13)
            self.assertEqual(fitted.effective_recipe.calibration_mass, 2.)
            expected_ids = tuple((day.day, day.trial_ids) for day in days[1:])
            self.assertEqual(fitted.source_weights.fit_trial_ids, expected_ids)
            self.assertEqual(fitted.primary.features.fit_trial_ids, expected_ids)
            dev = development(days[-1])
            gain = (masses[0] * 2 / 15 + masses[1] * 3 * 22 / 15 + 2 * 4) / 7
            actual = core.predict_primary_control_development(fitted, dev, check=check)
            np.testing.assert_allclose(actual.values, expected_native(dev, gain), rtol=0, atol=2e-8)
        with self.assertRaises(ValueError):
            core.fit_primary_control(unread, recipe=replace(chosen, source_masses=(1., 2., 1.)),
                control=control, weight_recipe=recipes[0], check=check)
        # Ordinary feature/weight APIs cannot infer a removal from a short list.
        with self.assertRaises(ValueError):
            fit_lfp_features(unread[1:], check=check)
        with self.assertRaises(ValueError):
            fit_source_weights(unread[1:], recipe=recipes[0], check=check)


if __name__ == "__main__":
    unittest.main()
