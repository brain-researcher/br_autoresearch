"""Four array-only primary-design checks; no empirical model qualification.

Run only in coordinated compute. Fabricated complete direction/role counts
exercise the existing native adapter checks; two artificial FeatureAxis columns
isolate design arithmetic, not qualification of an actual nine-band transform.
No tests/imports were executed while writing these sources.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

from lfp_features import FeatureAxis, TrialFeatures
from native_trials import BANDS, DayTrials, EARLIER, _snapshot
import primary_design as core


def check():
    pass


FEATURES = (FeatureAxis(BANDS[0], "fixture_a"), FeatureAxis(BANDS[1], "fixture_b"))


def pack(date="20140304", *, index=3, stream="today_only", role="calibration", region="M1"):
    repeats = {"training": 15, "calibration": 3, "development": 6}[role]
    directions = np.repeat(np.arange(8), repeats)
    within = np.tile(np.arange(repeats), 8)
    offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
    ids = tuple(f"trial_{i + 1 + (1000 if role == 'development' else 0):06d}" for i in range(len(directions)))
    bins, neurons = len(offsets), index + 2
    x = np.empty((len(ids), bins, 2))
    y = np.empty((len(ids), bins, neurons), dtype=np.int64)
    for trial, (direction, repeated) in enumerate(zip(directions, within)):
        for time in range(bins):
            x[trial, time] = (index * 20 + direction + time / 8 + repeated / 2,
                             index * 6 + time * direction - repeated ** 2 / 4)
            for neuron in range(neurons):
                y[trial, time, neuron] = 10 * (index + 1) + 2 * direction + (time + repeated) * (neuron + 1)
    native = DayTrials("Mihili", date, "20140304", region, stream,
        "development_features" if role == "development" else "fit", role, ids,
        _snapshot(directions), _snapshot(offsets), _snapshot(np.ones((len(ids), bins, 9))),
        _snapshot(np.array([(10, *band) for band in BANDS])),
        None if role == "development" else _snapshot(y),
        None if role == "development" else _snapshot(np.array([(index * 100 + n + 1, 1) for n in range(neurons)])))
    features = TrialFeatures(native.animal, native.day, native.target_day, native.region,
        native.stream, native.view_role, native.role, ids, native.directions, native.bin_offsets,
        FEATURES, _snapshot(x))
    return native, features


def history():
    pairs = tuple(pack(date, index=index, stream="history_assisted",
                       role="training" if index < 3 else "calibration")
                  for index, date in enumerate(EARLIER["Mihili"] + ("20140304",)))
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


def literal_means(values, directions):
    result = np.empty((8, values.shape[1], values.shape[2]))
    for direction in range(8):
        indices = [index for index, label in enumerate(directions) if label == direction]
        for time in range(values.shape[1]):
            for column in range(values.shape[2]):
                result[direction, time, column] = math.fsum(float(values[index, time, column])
                    for index in indices) / len(indices)
    return result


class UnreadNative(DayTrials):
    def __getattribute__(self, name):
        if name in ("lfp", "spikes", "unit_guide"):
            raise AssertionError("forbidden scope must fail before native numerical arrays")
        return super().__getattribute__(name)


class UnreadFeatures(TrialFeatures):
    def __getattribute__(self, name):
        if name == "values":
            raise AssertionError("scope/identity mismatch must fail before feature values")
        return super().__getattribute__(name)


class PrimaryDesignTests(unittest.TestCase):
    def test_per_day_cell_mean_reference_and_independent_native_response_axes(self):
        days, features = history()
        fitted = core.fit_primary_design(days[::-1], features[::-1], adjustment="residualize", check=check)
        self.assertEqual(tuple(value.day for value in fitted.day_means), EARLIER["Mihili"] + ("20140304",))
        for day, feature, means in zip(days, features, fitted.day_means):
            mean_x = literal_means(feature.values, day.directions)
            mean_y = literal_means(day.spikes, day.directions)
            np.testing.assert_allclose(means.feature_means, mean_x, rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(means.response_means, mean_y)
            result = core.make_primary_fit_design(fitted, day, feature, check=check)
            np.testing.assert_allclose(result.design.values, feature.values - mean_x[day.directions],
                                       rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(result.responses, day.spikes - mean_y[day.directions])
            self.assertEqual(result.Y.shape, (len(day.trial_ids) * 10, len(day.unit_guide)))
            self.assertEqual(result.design.X.shape, (len(day.trial_ids) * 10, 2))
            np.testing.assert_array_equal(result.unit_guide, day.unit_guide)
            for trial, time in ((0, 0), (1, 7), (len(day.trial_ids) - 1, 9)):
                np.testing.assert_array_equal(result.design.X[trial * 10 + time], result.design.values[trial, time])
                np.testing.assert_array_equal(result.Y[trial * 10 + time], result.responses[trial, time])
            for direction in range(8):
                np.testing.assert_allclose(result.responses[day.directions == direction].mean(axis=0), 0., atol=1e-14)
            with self.assertRaises(ValueError):
                result.design.X.setflags(write=True)
        self.assertEqual(tuple(len(value.unit_guide) for value in fitted.day_means), (2, 3, 4, 5))
        # The last-day means are its own calibration estimates, not pooled source estimates.
        self.assertEqual(fitted.day_means[-1].response_means[0, 0, 0], 41.)

    def test_full_cell_encoding_both_modes_and_primary_native_mean_restoration(self):
        for region, bins in (("M1", 10), ("PMd", 15)):
            day, feature = pack(region=region)
            _, development = pack(region=region, role="development")
            expected_cells = np.zeros((len(development.trial_ids), bins, 8 * bins - 1))
            for trial, direction in enumerate(development.directions):
                for time in range(bins):
                    cell = int(direction) * bins + time
                    if cell:
                        expected_cells[trial, time, cell - 1] = 1.
            for adjustment in core.ADJUSTMENTS:
                fitted = core.fit_primary_design((day,), (feature,), adjustment=adjustment, check=check)
                design = core.transform_primary_features(fitted, development, check=check)
                centered = adjustment != "explicit"
                expected_x = development.values - literal_means(feature.values, day.directions)[development.directions] if centered else development.values
                np.testing.assert_array_equal(design.values[:, :, :2], expected_x)
                if adjustment in ("explicit", "both"):
                    np.testing.assert_array_equal(design.values[:, :, 2:], expected_cells)
                    self.assertEqual(len(design.axes), 2 + 8 * bins - 1)
                    self.assertEqual((design.axes[-1].direction, design.axes[-1].bin_offset),
                                     (7, int(day.bin_offsets[-1])))
                else:
                    self.assertEqual(len(design.axes), 2)
                native = np.full((len(development.trial_ids), bins, len(day.unit_guide)), -.25)
                restored = core.restore_primary_prediction(fitted, design, native, check=check)
                expected_y = native + literal_means(day.spikes, day.directions)[development.directions] if centered else native
                np.testing.assert_array_equal(restored.values, expected_y)
                self.assertEqual(restored.trial_ids, development.trial_ids)
                self.assertEqual(restored.calibration_trial_ids, day.trial_ids)
                np.testing.assert_array_equal(restored.bin_offsets, development.bin_offsets)
                np.testing.assert_array_equal(restored.unit_guide, day.unit_guide)
                np.testing.assert_array_equal(native, -.25 * np.ones_like(native))

    def test_fit_scope_stream_and_known_development_overlap_refuse_before_values(self):
        day, feature = pack()
        fitted = core.fit_primary_design((day,), (feature,), adjustment="both", check=check)
        development_day, development = pack(role="development")
        blocked_day, blocked_features = UnreadNative(**vars(development_day)), UnreadFeatures(**vars(development))
        with self.assertRaises(PermissionError):
            core.fit_primary_design((blocked_day,), (blocked_features,), adjustment="both", check=check)
        with self.assertRaises(PermissionError):
            core.make_primary_fit_design(fitted, blocked_day, blocked_features, check=check)
        overlap = replace(development, trial_ids=(day.trial_ids[0],) + development.trial_ids[1:])
        with self.assertRaises(PermissionError):
            core.transform_primary_features(fitted, UnreadFeatures(**vars(overlap)), check=check)
        wrong_stream = replace(development, stream="history_assisted")
        with self.assertRaises(PermissionError):
            core.transform_primary_features(fitted, UnreadFeatures(**vars(wrong_stream)), check=check)
        wrong_rows = replace(feature, trial_ids=feature.trial_ids[::-1])
        with self.assertRaises(ValueError):
            core.fit_primary_design((UnreadNative(**vars(day)),), (UnreadFeatures(**vars(wrong_rows)),),
                                    adjustment="explicit", check=check)
        days, features = history()
        with self.assertRaises(ValueError):
            core.fit_primary_design(days[:-1], features[:-1], adjustment="residualize", check=check)
        # Reused trial numbers in DIFFERENT source days remain legitimate.
        valid = core.fit_primary_design(days, features, adjustment="explicit", check=check)
        self.assertEqual(len(valid.day_means), 4)

    def test_axis_nonfinite_missing_cell_and_cancellation_refusals(self):
        day, feature = pack()
        fitted = core.fit_primary_design((day,), (feature,), adjustment="residualize", check=check)
        _, development = pack(role="development")
        bad_values = development.values.copy()
        bad_values[-1, -1, -1] = np.nan
        for bad in (replace(development, axes=FEATURES[::-1]),
                    replace(development, values=_snapshot(bad_values)),
                    replace(development, bin_offsets=_snapshot(np.arange(6, 16))),
                    replace(development, values=_snapshot(development.values[:, :-1]))):
            with self.assertRaises(ValueError):
                core.transform_primary_features(fitted, bad, check=check)
        wrong_units = replace(day, unit_guide=_snapshot(day.unit_guide[::-1]))
        with self.assertRaises(ValueError):
            core.make_primary_fit_design(fitted, wrong_units, feature, check=check)
        missing_direction = _snapshot(np.where(day.directions == 7, 6, day.directions))
        with self.assertRaises(ValueError):
            core.fit_primary_design((replace(day, directions=missing_direction),),
                (replace(feature, directions=missing_direction),), adjustment="explicit", check=check)
        design = core.transform_primary_features(fitted, development, check=check)
        with self.assertRaises(ValueError):
            core.restore_primary_prediction(fitted, design,
                np.zeros((len(development.trial_ids), 10, len(day.unit_guide) - 1)), check=check)
        calls = []
        def cancel():
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            core.fit_primary_design((day,), (feature,), adjustment="explicit", check=cancel)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
