"""Focused fabricated-array feature checks, not real-data model qualification.

Run only in the coordinator's compute allocation. Small DayTrials fixtures
isolate feature arithmetic; complete-role pack support is the native adapter's
separate responsibility. No project tests or imports ran while writing this.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

import lfp_features as core
from native_trials import BANDS, DayTrials, EARLIER


def check():
    pass


def day(*, date="20140304", stream="today_only", role="calibration", trials=2,
        electrodes=3, constant=None):
    guide = np.array([(e + 11, low, high) for low, high in BANDS
                      for e in range(electrodes)], dtype=float)
    values = np.empty((trials, 10, len(guide)))
    for band in range(9):
        for electrode in range(electrodes):
            values[:, :, band * electrodes + electrode] = (
                100 * band + constant if constant is not None else
                100 * band + np.arange(trials)[:, None] * 2 + np.arange(10)[None, :] / 8
                + (electrode - 1) ** 3)
    return DayTrials(animal="Mihili", day=date, target_day="20140304", region="M1",
        stream=stream, view_role="development_features" if role == "development" else "fit",
        role=role, trial_ids=tuple(f"trial_{i + 1 + (1000 if role == 'development' else 0):06d}"
                                  for i in range(trials)),
        directions=np.arange(trials, dtype=np.int64) % 8, bin_offsets=np.arange(5, 15),
        lfp=values, lfp_guide=guide, spikes=None, unit_guide=None)


def history():
    # Deliberately unequal sample and electrode counts: fitted bounds/scaling
    # give each permitted day 1/4 weight, not each flattened sample 1/N.
    return tuple(day(date=date, stream="history_assisted",
        role="training" if index < 3 else "calibration", trials=(7, 2, 4, 3)[index],
        electrodes=(9, 2, 4, 3)[index], constant=index * 10.)
        for index, date in enumerate(EARLIER["Mihili"] + ("20140304",)))


def literal_moments(record, quantiles):
    answer = np.empty((len(record.trial_ids), 10, 9 * (2 + len(quantiles))))
    for trial in range(len(record.trial_ids)):
        for time in range(10):
            row = []
            for band in BANDS:
                values = sorted(float(record.lfp[trial, time, column])
                    for column, guide in enumerate(record.lfp_guide)
                    if tuple(guide[1:]) == band)
                mean = math.fsum(values) / len(values)
                sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
                row.extend((mean, sd))
                row.extend(values[max(0, math.ceil(q * len(values)) - 1)] for q in quantiles)
            answer[trial, time] = row
    return answer


class UnreadDay(DayTrials):
    def __getattribute__(self, name):
        if name in ("lfp", "spikes", "unit_guide"):
            raise AssertionError("forbidden roles must fail before numerical array access")
        return super().__getattribute__(name)


class LFPFeatureTests(unittest.TestCase):
    def test_moment_quantile_reference_and_electrode_identity_invariance(self):
        record = day(electrodes=4)
        recipe = core.FeatureRecipe(quantiles=(0., .25, .5, .75, 1.))
        fitted = core.fit_lfp_features((record,), recipe=recipe, check=check)
        result = core.transform_lfp_features(fitted, record, check=check)
        np.testing.assert_allclose(result.values, literal_moments(record, recipe.quantiles),
                                   rtol=1e-14, atol=1e-14)
        self.assertEqual(result.values.shape, (2, 10, 63))
        self.assertEqual(tuple(axis.band for axis in result.axes[::7]), BANDS)
        self.assertEqual(tuple(axis.statistic for axis in result.axes[:7]),
                         ("mean", "population_sd", "quantile_0.0", "quantile_0.25",
                          "quantile_0.5", "quantile_0.75", "quantile_1.0"))
        order = np.arange(record.lfp.shape[2])[::-1]
        guide = record.lfp_guide[order].copy()
        guide[:, 0] += 1000
        permuted = replace(record, lfp=record.lfp[:, :, order], lfp_guide=guide)
        np.testing.assert_array_equal(core.transform_lfp_features(fitted, permuted, check=check).values,
                                      result.values)
        duplicated_guide = np.repeat(record.lfp_guide, 2, axis=0)
        duplicated_guide[:, 0] = np.arange(len(duplicated_guide)) + 100
        duplicate = replace(record, lfp=np.repeat(record.lfp, 2, axis=2), lfp_guide=duplicated_guide)
        np.testing.assert_allclose(core.transform_lfp_features(fitted, duplicate, check=check).values,
                                   result.values, rtol=1e-14, atol=1e-14)
        self.assertEqual(result.trial_ids, record.trial_ids)
        np.testing.assert_array_equal(result.bin_offsets, record.bin_offsets)
        np.testing.assert_array_equal(result.directions, record.directions)
        self.assertEqual(result.values.dtype, np.float64)
        with self.assertRaises(ValueError):
            result.values.setflags(write=True)

    def test_equal_day_histogram_bounds_full_tails_and_collapsed_bounds(self):
        recipe = core.FeatureRecipe(family="robust_histogram", histogram_bins=2,
                                    histogram_quantiles=(.25, .75))
        fitted = core.fit_lfp_features(history()[::-1], recipe=recipe, check=check)
        for band, edge in enumerate(fitted.histogram_edges):
            np.testing.assert_array_equal(edge, np.array([0., 10., 20.]) + 100 * band)
        observed = day(stream="history_assisted", role="development", electrodes=6)
        raw = observed.lfp.copy()
        for band in range(9):
            raw[:, :, 6 * band:6 * (band + 1)] = np.array([-1., 0., 5., 10., 20., 21.]) + 100 * band
        observed = replace(observed, lfp=raw)
        actual = core.transform_lfp_features(fitted, observed, check=check).values.reshape(2, 10, 9, 4)
        # Literal histogram with both tails and endpoint conventions.
        expected = np.array([1., 2., 2., 1.]) / 6
        np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))
        np.testing.assert_allclose(actual.sum(axis=3), 1., rtol=0, atol=2e-16)
        calibration = day(constant=30.)
        today = core.fit_lfp_features((calibration,), recipe=recipe, check=check)
        observed = day(role="development", electrodes=3)
        raw = observed.lfp.copy()
        for band in range(9):
            raw[:, :, 3 * band:3 * (band + 1)] = np.array([29., 30., 31.]) + 100 * band
        collapsed = core.transform_lfp_features(today, replace(observed, lfp=raw), check=check)
        np.testing.assert_array_equal(collapsed.values.reshape(2, 10, 9, 4),
            np.broadcast_to([1/3, 0., 1/3, 1/3], (2, 10, 9, 4)))
        for band, edge in enumerate(today.histogram_edges):
            np.testing.assert_array_equal(edge, np.full(3, 30. + 100 * band))

    def test_frozen_robust_scaling_and_fit_roles_do_not_leak_between_streams(self):
        recipe = core.FeatureRecipe(quantiles=(.5,), scaling="robust")
        fitted = core.fit_lfp_features(history(), recipe=recipe, check=check)
        # Equal-day CDF of 0,10,20,30 has q25=0, median=10, q75=20.
        expected_center = np.array([[100 * band + 10, 0., 100 * band + 10] for band in range(9)]).ravel()
        expected_scale = np.tile([20., 1., 20.], 9)
        np.testing.assert_array_equal(fitted.center, expected_center)
        np.testing.assert_array_equal(fitted.scale, expected_scale)
        development = day(stream="history_assisted", role="development", constant=100000.)
        first = core.transform_lfp_features(fitted, development, check=check)
        reference = (literal_moments(development, (.5,)) - expected_center) / expected_scale
        np.testing.assert_array_equal(first.values, reference)
        np.testing.assert_array_equal(fitted.center, expected_center)
        with self.assertRaises(ValueError):
            fitted.scale.setflags(write=True)
        today_day = day(constant=30.)
        today = core.fit_lfp_features((today_day,), recipe=recipe, check=check)
        np.testing.assert_array_equal(today.scale, np.ones(27))
        today_values = core.transform_lfp_features(today, today_day, check=check).values
        np.testing.assert_array_equal(today_values, np.zeros_like(today_values))
        bad = UnreadDay(**vars(development))
        with self.assertRaises(PermissionError):
            core.fit_lfp_features((bad,), recipe=recipe, check=check)
        unread_source = UnreadDay(**vars(history()[0]))
        with self.assertRaises(PermissionError):
            core.fit_lfp_features((today_day, unread_source), recipe=recipe, check=check)
        with self.assertRaises(PermissionError):
            core.transform_lfp_features(fitted, UnreadDay(**vars(today_day)), check=check)
        with self.assertRaises(ValueError):
            core.fit_lfp_features(history()[:-1], recipe=recipe, check=check)

    def test_no_dropped_bands_columns_or_native_bins_and_cancellation(self):
        record = day()
        fitted = core.fit_lfp_features((record,), check=check)
        overlapping = replace(day(role="development"), trial_ids=record.trial_ids)
        with self.assertRaises(PermissionError):
            core.transform_lfp_features(fitted, UnreadDay(**vars(overlapping)), check=check)
        missing = replace(record, lfp=record.lfp[:, :, 3:], lfp_guide=record.lfp_guide[3:])
        nonfinite = record.lfp.copy()
        nonfinite[1, 9, -1] = np.nan
        unknown = record.lfp_guide.copy()
        unknown[-1, 1:] = (300, 450)
        duplicate = record.lfp_guide.copy()
        duplicate[1, 0] = duplicate[0, 0]
        for invalid in (missing, replace(record, lfp=nonfinite),
                        replace(record, lfp_guide=unknown), replace(record, lfp_guide=duplicate),
                        replace(record, bin_offsets=np.arange(6, 16)),
                        replace(record, lfp=record.lfp[:, :-1, :])):
            with self.subTest(invalid_shape=invalid.lfp.shape):
                with self.assertRaises(ValueError):
                    core.transform_lfp_features(fitted, invalid, check=check)
        calls = []
        def cancel():
            calls.append(None)
            if len(calls) == 4:
                raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            core.fit_lfp_features(history(), recipe=core.FeatureRecipe(family="robust_histogram"), check=cancel)
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
