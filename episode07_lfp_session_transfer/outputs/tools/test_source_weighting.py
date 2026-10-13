"""Four small arithmetic/access checks; run only in coordinator-owned compute.

The fabricated native-axis fixtures check candidate formulas, not experimental
recovery, fit quality or any empirical source/day benefit. No tests or imports
were run while writing this file on the login node.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

from lfp_features import FeatureRecipe, fit_lfp_features
from native_trials import BANDS, DayTrials, EARLIER, _snapshot
from source_weighting import SourceWeightRecipe, fit_source_weights


def check():
    pass


def fixture():
    result = []
    dates = EARLIER["Mihili"] + ("20140304",)
    for index, (date, repeats, slopes, neurons) in enumerate(zip(
            dates, (15, 16, 17, 3), ((1, 1), (1, -1), (1, 2), (1, 1)), (1, 3, 2, 4))):
        directions = np.repeat(np.arange(8), repeats)
        repetition = np.tile(np.arange(repeats), 8)
        offsets = np.arange(5, 15)
        cell = directions[:, None] * 10 + np.arange(10)[None, :]
        slope = np.asarray(slopes)[repetition % 2]
        spikes = (10000 + slope[:, None, None] * cell[:, :, None]
                  + 11 * np.arange(neurons)[None, None, :]).astype(np.float64)
        # One electrode per band: means and medians equal this value; SD=0.
        # Source offsets 0,1,2; target offset0. Centered repeats make prototype
        # comparisons independent of their different complete trial counts.
        delta = index if index < 3 else 0
        lfp = (directions[:, None, None] + np.arange(10)[None, :, None]
               + (repetition - (repeats - 1) / 2)[:, None, None] + delta
               + np.zeros((1, 1, 9))).astype(np.float64)
        # Crossing six/seven digits exposes accidental lexical ID sorting.
        ids = tuple(f"trial_{999990 + row:06d}" for row in range(len(directions)))
        result.append(DayTrials("Mihili", date, "20140304", "M1", "history_assisted",
            "fit", "calibration" if index == 3 else "training", ids,
            _snapshot(directions), _snapshot(offsets), _snapshot(lfp),
            _snapshot(np.asarray([(100 + index, low, high) for low, high in BANDS])),
            _snapshot(spikes), _snapshot(np.asarray([(1 + unit, 1) for unit in range(neurons)]))))
    return tuple(result)


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("forbidden array was interpreted")


class SourceWeightTests(unittest.TestCase):
    def test_equal_mass_is_per_day_not_trial_or_neuron_count_and_needs_no_values(self):
        days = tuple(replace(day, spikes=ForbiddenArray(), unit_guide=ForbiddenArray(),
                             lfp=ForbiddenArray(), lfp_guide=ForbiddenArray()) for day in fixture())
        recipe = SourceWeightRecipe("equal", target_mass=1, source_total_mass=3)
        result = fit_source_weights(days[::-1], recipe=recipe, check=check)
        self.assertEqual(result.source_days, EARLIER["Mihili"])
        np.testing.assert_array_equal(result.relative_weights, np.full(3, 1 / 3))
        self.assertEqual(result.source_masses, (1., 1., 1.))
        self.assertEqual(result.target_mass, 1.)
        self.assertEqual(result.fit_trial_ids, tuple((day.day, day.trial_ids) for day in days))
        self.assertIsNone(result.reliability)
        self.assertIsNone(result.feature_scale)
        changed = fit_source_weights(days, recipe=replace(recipe, target_mass=7, source_total_mass=6),
                                     check=check)
        np.testing.assert_array_equal(changed.relative_weights, result.relative_weights)
        self.assertEqual(changed.source_masses, (2., 2., 2.))
        self.assertEqual(changed.target_mass, 7.)
        for value in (result.relative_weights, result.bin_offsets):
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_split_half_native_anchor_formula_and_group_mean_shrinkage(self):
        days = fixture()
        days = days[:-1] + (replace(days[-1], spikes=ForbiddenArray(), unit_guide=ForbiddenArray()),)
        recipe = SourceWeightRecipe("reliability", target_mass=2, source_total_mass=5,
                                    weight_floor=.25, reliability_power=2)
        result = fit_source_weights(days, recipe=recipe, check=check)
        # Own-axis centered anchors are respectively B=A, B=-A and B=2A.
        # Pooling any number of native neurons leaves r at 1,0,4/5.
        expected_r = np.array([1., 0., 4 / 5])
        q = [(1 / 4 + value) ** 2 for value in expected_r]
        expected_weights = np.asarray([value / math.fsum(q) for value in q])
        np.testing.assert_allclose(result.reliability, expected_r, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.relative_weights, expected_weights, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.source_masses, 5 * expected_weights, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result.effective_trials, [120., 128., 136.])

        reordered = []
        for day in days[:-1]:
            order = np.arange(len(day.trial_ids))[::-1]
            reordered.append(replace(day, trial_ids=tuple(day.trial_ids[i] for i in order),
                directions=_snapshot(day.directions[order]), lfp=_snapshot(day.lfp[order]),
                spikes=_snapshot(day.spikes[order])))
        reversed_result = fit_source_weights(tuple(reordered) + days[-1:], recipe=recipe, check=check)
        np.testing.assert_allclose(reversed_result.reliability, expected_r, rtol=0, atol=1e-14)

        hierarchical = fit_source_weights(days, recipe=replace(recipe,
            family="hierarchical_shrinkage", shrinkage_strength=120.), check=check)
        mean_r = math.fsum(expected_r) / 3
        expected_shrunk = [(n * r + 120 * mean_r) / (n + 120)
                           for n, r in zip((120, 128, 136), expected_r)]
        q = [(1 / 4 + value) ** 2 for value in expected_shrunk]
        np.testing.assert_allclose(hierarchical.shrunk_reliability, expected_shrunk,
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(hierarchical.relative_weights,
            [value / math.fsum(q) for value in q], rtol=0, atol=1e-14)
        constant = tuple(replace(day, spikes=_snapshot(np.full_like(day.spikes, 10)))
                         for day in fixture())
        np.testing.assert_array_equal(fit_source_weights(constant, recipe=recipe,
                                                       check=check).reliability, np.zeros(3))
        with self.assertRaises(ValueError):
            hierarchical.shrunk_reliability.setflags(write=True)

    def test_similarity_uses_target_calibration_scale_and_one_bound_feature_transform(self):
        days = fixture()
        model = fit_lfp_features(days, recipe=FeatureRecipe(quantiles=(.5,), scaling="none"), check=check)
        # Similarity never requires any neural counts, including calibration.
        without_counts = tuple(replace(day, spikes=ForbiddenArray(), unit_guide=ForbiddenArray())
                               for day in days)
        recipe = SourceWeightRecipe("calibration_similarity", target_mass=1, source_total_mass=3,
            weight_floor=.1, similarity_temperature=2, feature_scale_floor=1)
        result = fit_source_weights(without_counts, recipe=recipe, feature_model=model, check=check)
        # Target values are direction(0..7)+bin(0..9)+repeat(-1,0,1).
        # Population variance is 63/12 + 99/12 + 2/3 = 85/6.
        # Two of three statistics per band shift; the retained SD feature is0.
        variance = 85 / 6
        distances = [2 / 3 * delta * delta / variance for delta in (0, 1, 2)]
        q = [.1 + math.exp(-distance / 2) for distance in distances]
        np.testing.assert_allclose(result.prototype_distances, distances, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.feature_scale,
            [value for _ in BANDS for value in (math.sqrt(variance), 1., math.sqrt(variance))],
            rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.relative_weights, [value / math.fsum(q) for value in q],
                                   rtol=0, atol=1e-14)
        self.assertIsNone(result.reliability)
        forbidden_lfp = tuple(replace(day, lfp=ForbiddenArray()) for day in without_counts)
        wrong_ids = replace(model, fit_trial_ids=model.fit_trial_ids[:-1])
        for wrong_model in (wrong_ids, replace(model, stream="today_only"),
                            replace(model, region="PMd")):
            with self.assertRaises(PermissionError):
                fit_source_weights(forbidden_lfp, recipe=recipe, feature_model=wrong_model, check=check)
        with self.assertRaises(ValueError):
            result.feature_scale.setflags(write=True)

    def test_rejects_scope_role_axis_invalid_parameters_and_cancellation(self):
        days = fixture()
        recipe = SourceWeightRecipe("reliability", target_mass=1, source_total_mass=3,
                                    weight_floor=.1, reliability_power=1)
        for changes in (dict(view_role="development_features", role="development"),
                        dict(view_role="held_out", role="held_out"), dict(stream="today_only"),
                        dict(animal="Chewie-L"), dict(region="PMd")):
            broken = replace(days[2], **changes, directions=ForbiddenArray(), spikes=ForbiddenArray())
            with self.assertRaises(PermissionError):
                fit_source_weights(days[:2] + (broken,) + days[3:], recipe=recipe, check=check)
        with self.assertRaises(ValueError):
            fit_source_weights(days[:-1], recipe=recipe, check=check)
        for changed in (replace(days[0], bin_offsets=_snapshot(np.arange(10))),
                        replace(days[0], unit_guide=_snapshot(np.array([[1, 1], [1, 1]]))),
                        replace(days[0], spikes=_snapshot(np.full_like(days[0].spikes, .5)))):
            with self.assertRaises(ValueError):
                fit_source_weights((changed,) + days[1:], recipe=recipe, check=check)
        for wrong in (replace(recipe, weight_floor=0), replace(recipe, reliability_power=True),
                      replace(recipe, source_total_mass=float("inf")),
                      replace(recipe, target_mass=0), replace(recipe, shrinkage_strength=1),
                      replace(recipe, family="unknown")):
            with self.assertRaises(ValueError):
                fit_source_weights(days, recipe=wrong, check=check)
        with self.assertRaises(NotImplementedError):
            fit_source_weights(days, recipe=replace(recipe, family="leave_one_day_stacking"), check=check)
        def cancelled():
            raise RuntimeError("cancelled before source access")
        with self.assertRaisesRegex(RuntimeError, "cancelled before source access"):
            fit_source_weights(days, recipe=recipe, check=cancelled)


if __name__ == "__main__":
    unittest.main()
