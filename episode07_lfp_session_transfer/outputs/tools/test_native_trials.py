"""Tiny native-axis interface fixtures; run only in coordinated Slurm compute."""
from dataclasses import replace
import unittest

import numpy as np

from native_trials import (
    BANDS, EARLIER, adapt_records, fit_calibration_mean,
    load_native_trials, predict_calibration_mean,
)


def allowed_scope(view_role="fit", stream="today_only", region="M1"):
    return dict(view_role=view_role, animal="Mihili", target_day="20140304",
                region=region, stream=stream, check=lambda: None)


def records_for(day="20140304", role="calibration", region="M1", neurons=2,
                first_trial=1, spike_offset=0, electrode=7):
    per_direction = {"training": 15, "calibration": 3, "development": 6}[role]
    offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
    guide = np.asarray([(electrode, low, high) for low, high in BANDS], dtype=np.float64)
    # Deliberately overlapping numeric neuron labels on different days have no
    # cross-day identity meaning. Day-specific widths need not agree.
    units = np.asarray([(index + 1, 1) for index in range(neurons)], dtype=np.float64)
    result = []
    for direction in range(8):
        for repeat in range(per_direction):
            trial = first_trial + len(result)
            row = dict(animal="Mihili", day=day, region=region,
                       trial_id=f"trial_{trial:06d}", role=role, direction=direction,
                       bin_offsets=offsets.copy(), lfp_guide=guide.copy(),
                       lfp=np.full((len(offsets), len(BANDS)), direction - repeat / 2,
                                   dtype=np.float32))
            if role != "development":
                row["unit_guide"] = units.copy()
                row["spikes"] = (spike_offset + 10 * direction + repeat
                                  + 2 * np.arange(len(offsets))[:, None]
                                  + np.arange(neurons)[None, :]).astype(np.float64)
            result.append(row)
    return result


class FakeViews:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def iter_records(self, role, **scope):
        self.calls.append((role, scope))
        return iter(self.records)


class NativeTrialTests(unittest.TestCase):
    def test_complete_days_preserve_independent_axes_order_and_immutable_values(self):
        records = []
        widths = (1, 3, 2, 2)
        for index, day in enumerate(EARLIER["Mihili"] + ("20140304",)):
            role = "calibration" if day == "20140304" else "training"
            records.extend(reversed(records_for(day, role, neurons=widths[index],
                                                electrode=index + 7,
                                                spike_offset=1000 if role == "training" else 0)))
        views = FakeViews(records)
        packs = load_native_trials(views, **allowed_scope(stream="history_assisted"))
        self.assertEqual(tuple(pack.day for pack in packs), EARLIER["Mihili"] + ("20140304",))
        self.assertEqual(tuple(pack.spikes.shape[2] for pack in packs), widths)
        self.assertEqual(packs[0].trial_ids[0], "trial_000120")
        self.assertEqual(packs[-1].trial_ids[-1], "trial_000001")
        self.assertEqual(views.calls, [("fit", dict(animal="Mihili", target_day="20140304",
                                                  region="M1", stream="history_assisted"))])
        np.testing.assert_array_equal(packs[0].spikes[0], records[0]["spikes"])
        np.testing.assert_array_equal(packs[0].bin_offsets, np.arange(5, 15))
        self.assertEqual(packs[0].lfp.dtype, np.dtype("float32"))
        before = packs[0].lfp[0, 0, 0]
        records[0]["lfp"][0, 0] += 99
        self.assertEqual(packs[0].lfp[0, 0, 0], before)
        for array in (packs[0].lfp, packs[0].spikes, packs[0].lfp_guide,
                      packs[0].unit_guide, packs[0].directions, packs[0].bin_offsets):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        pmd = adapt_records(records_for(region="PMd"), **allowed_scope(region="PMd"))[0]
        self.assertEqual(pmd.spikes.shape, (24, 15, 2))
        np.testing.assert_array_equal(pmd.bin_offsets, np.arange(-15, 0))

    def test_baseline_is_target_calibration_mean_in_native_time_and_neuron_axes(self):
        calibration = adapt_records(records_for(), **allowed_scope())[0]
        development = adapt_records(records_for(role="development", first_trial=101),
                                    **allowed_scope("development_features"))[0]
        model = fit_calibration_mean(calibration, check=lambda: None)
        prediction = predict_calibration_mean(model, development, check=lambda: None)
        expected_means = np.empty((8, 10, 2), dtype=np.float64)
        for direction in range(8):
            for time_bin in range(10):
                for neuron in range(2):
                    expected_means[direction, time_bin, neuron] = (
                        10 * direction + 2 * time_bin + neuron + (0 + 1 + 2) / 3)
        np.testing.assert_array_equal(model.means, expected_means)
        np.testing.assert_array_equal(model.trials_per_direction, np.full(8, 3))
        np.testing.assert_array_equal(prediction.values, expected_means[development.directions])
        np.testing.assert_array_equal(prediction.unit_guide, calibration.unit_guide)
        self.assertEqual(prediction.trial_ids, development.trial_ids)
        self.assertEqual(prediction.calibration_trial_ids, calibration.trial_ids)
        self.assertIsNone(development.spikes)
        self.assertIsNone(development.unit_guide)
        with self.assertRaises(ValueError):
            prediction.values.setflags(write=True)
        # History values cannot enter this fitting function: it accepts only
        # the target calibration day, even for the history-assisted stream.
        history = []
        for day in EARLIER["Mihili"]:
            history.extend(records_for(day, "training", spike_offset=100000))
        history.extend(records_for())
        packs = adapt_records(history, **allowed_scope(stream="history_assisted"))
        np.testing.assert_array_equal(fit_calibration_mean(packs[-1], check=lambda: None).means,
                                      model.means)
        for forbidden in (packs[0], development):
            with self.assertRaises(PermissionError):
                fit_calibration_mean(forbidden, check=lambda: None)
        with self.assertRaises(ValueError):
            predict_calibration_mean(model, replace(development, day="20140307", target_day="20140307"),
                                     check=lambda: None)
        overlap = replace(development, trial_ids=(calibration.trial_ids[0],) + development.trial_ids[1:])
        with self.assertRaises(ValueError):
            predict_calibration_mean(model, overlap, check=lambda: None)

    def test_access_scope_is_rejected_before_reader_or_supplied_array_access(self):
        views = FakeViews([])
        for role in ("heldout", "development_targets", "scorer"):
            with self.assertRaises(PermissionError):
                load_native_trials(views, **allowed_scope(role))
        with self.assertRaises(ValueError):
            load_native_trials(views, **dict(allowed_scope(), target_day="20140306"))
        self.assertEqual(views.calls, [])

        class ForbiddenArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError("array interpreted before role/scope rejection")

        for changes in (dict(animal="Chewie-L"), dict(region="PMd"),
                        dict(day="20140217", role="training"), dict(role="held_out")):
            row = records_for()[0]
            row.update(changes)
            row["lfp"] = ForbiddenArray()
            with self.assertRaises(ValueError):
                adapt_records([row], **allowed_scope())
        leaked = records_for(role="development", first_trial=101)
        leaked[0]["spikes"] = ForbiddenArray()
        with self.assertRaises(ValueError):
            adapt_records(leaked, **allowed_scope("development_features"))

    def test_wrong_axes_counts_support_and_cancellation_refuse_without_dropping(self):
        def changed(field, value, index=0):
            rows = records_for()
            rows[index][field] = value
            return rows

        rows = records_for()
        duplicate_trial = rows + [dict(rows[0])]
        cases = [
            duplicate_trial,
            rows[:-1],  # one direction has only two calibration trials
            changed("bin_offsets", np.arange(10)),
            changed("lfp_guide", rows[0]["lfp_guide"][::-1], index=1),
            changed("unit_guide", rows[0]["unit_guide"][::-1], index=1),
            changed("spikes", np.full((10, 2), 0.5)),
            changed("spikes", np.full((10, 2), -1)),
            changed("lfp", np.full((10, 9), np.nan)),
            changed("spikes", np.ones((10, 1))),
        ]
        for index, malformed in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                adapt_records(malformed, **allowed_scope())
        with self.assertRaises(ValueError):
            adapt_records(rows, **allowed_scope(stream="history_assisted"))

        closed = []
        def iterator():
            try:
                yield from rows
            finally:
                closed.append(True)
        checks = []
        def cancel():
            checks.append(True)
            if len(checks) == 4:
                raise RuntimeError("fixture cancellation")
        with self.assertRaisesRegex(RuntimeError, "fixture cancellation"):
            adapt_records(iterator(), **dict(allowed_scope(), check=cancel))
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
