"""Fabricated scorer fixtures; create temporary HDF only during Slurm tests.

No real response or held-out file is used. These checks establish arithmetic
and the procedural development interface, not OS isolation or a model result.
"""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import development_score as core
from native_trials import NativePredictions, NEW


def check():
    pass


def fixture_panel():
    models, baselines, targets = [], [], {}
    day_index = 0
    for animal, dates in NEW.items():
        for date in dates:
            day_index += 1
            for region, bins, multiplier in (("M1", 10, 1), ("PMd", 15, 2)):
                ids = tuple(f"trial_{101 + trial:06d}" for trial in range(48))
                calibration = tuple(f"trial_{trial + 1:06d}" for trial in range(24))
                directions = np.repeat(np.arange(8), 6)
                offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
                units = np.array([(1, 1), (3, 2), (8, 1)], dtype=np.int64)
                y = np.empty((48, bins, 3), dtype=np.float64)
                y[:, :, 0] = 10 * day_index + (np.arange(48) % 6)[:, None]
                y[:, :, 1] = 10 * day_index + 2 * np.arange(bins)[None, :]
                y[:, :, 2] = 100 + 10 * day_index
                predicted, baseline = y.copy(), y.copy()
                # A constant neuron's error must remain in the joint SSE even
                # though that neuron's own SST is zero.
                predicted[:, :, 2] += multiplier * day_index
                baseline[:, :, 0] += multiplier * 7
                metadata = dict(animal=animal, target_day=date, region=region,
                    trial_ids=ids, directions=directions, bin_offsets=offsets,
                    unit_guide=units, calibration_trial_ids=calibration)
                models.append(NativePredictions(values=predicted, **metadata))
                baselines.append(NativePredictions(values=baseline, **metadata))
                targets[(animal, date, region)] = y
    return tuple(models), tuple(baselines), targets


def write_targets(directory, predictions, targets):
    path = Path(directory) / "development_targets.h5"
    with h5py.File(path, "w") as handle:
        for prediction in predictions:
            key = prediction.animal, prediction.target_day, prediction.region
            group = handle.create_group("/".join(key))
            for index, trial in enumerate(prediction.trial_ids):
                record = group.create_group(trial)
                record.attrs["role"] = "development"
                record.attrs["direction"] = int(prediction.directions[index])
                record.attrs["native_bin_seconds"] = .03
                record.create_dataset("spikes", data=targets[key][index])
                record.create_dataset("unit_guide", data=prediction.unit_guide)
    return path


class DevelopmentScoreTests(unittest.TestCase):
    def test_joint_score_uses_per_neuron_means_and_keeps_constant_neuron_errors(self):
        y = np.array([[[0, 100, 7], [2, 102, 7]],
                      [[4, 104, 7], [6, 106, 7]]], dtype=float)
        predicted = y.copy()
        predicted[:, :, 2] += 2
        sse = math.fsum((float(y[t, b, n]) - float(predicted[t, b, n])) ** 2
                       for t in range(2) for b in range(2) for n in range(3))
        means = [math.fsum(float(y[t, b, n]) for t in range(2) for b in range(2)) / 4
                 for n in range(3)]
        sst = math.fsum((float(y[t, b, n]) - means[n]) ** 2
                       for t in range(2) for b in range(2) for n in range(3))
        self.assertEqual((sse, sst), (16., 40.))
        self.assertEqual(core.joint_r2(y, predicted, check=check), 1 - 16 / 40)
        self.assertLess(core.joint_r2(y, y + 100, check=check), 0)
        self.assertIsNone(core.joint_r2(np.ones_like(y), np.zeros_like(y), check=check))
        with self.assertRaises(ValueError):
            core.joint_r2(y, predicted[:, :-1], check=check)
        invalid = predicted.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            core.joint_r2(y, invalid, check=check)

    def test_complete_ten_slot_panel_uses_two_three_day_animal_balance(self):
        models, baselines, targets = fixture_panel()
        with tempfile.TemporaryDirectory() as directory:
            write_targets(directory, models, targets)
            feedback = core.score_development_panel(directory, models[::-1], baselines,
                                                     check=check)
        self.assertEqual(tuple(value.region for value in feedback), ("M1", "PMd"))
        for result, bins, multiplier in zip(feedback, (10, 15), (1, 2)):
            # Native neuron0 has uniform 0..5 repeats; neuron1 has 2*b;
            # neuron2 is constant. This computes SST without scorer arithmetic.
            sst = 48 * bins * (35 / 12 + (bins * bins - 1) / 3)
            day_scores = [1 - 48 * bins * (multiplier * day) ** 2 / sst
                          for day in range(1, 6)]
            expected_model = (day_scores[0] + day_scores[1]) / 4 + math.fsum(day_scores[2:]) / 6
            expected_baseline = 1 - 48 * bins * (multiplier * 7) ** 2 / sst
            self.assertTrue(result.available)
            self.assertIsNone(result.unavailable_reason)
            self.assertEqual(result.complete_new_days, 5)
            self.assertEqual(result.independent_animals, 2)
            self.assertAlmostEqual(result.model_r2, expected_model, places=13)
            self.assertAlmostEqual(result.calibration_mean_r2, expected_baseline, places=13)
            self.assertAlmostEqual(result.improvement_over_mean,
                                   expected_model - expected_baseline, places=13)
            self.assertGreater(abs(expected_model - math.fsum(day_scores) / 5), 1e-4)

    def test_incomplete_role_or_paired_axis_fail_before_open_and_no_closed_reader(self):
        models, baselines, _ = fixture_panel()
        invalid_panels = [
            models[:-1], models + (models[0],),
            (replace(models[0], target_day="20140306"),) + models[1:],
            (replace(models[0], calibration_trial_ids=models[0].trial_ids[:1]),) + models[1:],
            (replace(models[0], unit_guide=models[0].unit_guide[::-1]),) + models[1:],
        ]
        with patch.object(core.h5py, "File", side_effect=AssertionError("opened before binding")):
            for predictions in invalid_panels:
                with self.assertRaises(ValueError):
                    core.score_development_panel("unused", predictions, baselines, check=check)
            with self.assertRaises(TypeError):
                core.score_development_panel("unused", ("held_out",) + models[1:],
                                             baselines, check=check)
            def cancel():
                raise RuntimeError("fixture cancellation")
            with self.assertRaisesRegex(RuntimeError, "fixture cancellation"):
                core.score_development_panel("unused", models, baselines, check=cancel)
        with tempfile.TemporaryDirectory() as directory:
            # A caller cannot request this file through the scorer API.
            closed = Path(directory) / "heldout.h5"
            closed.write_bytes(b"closed fixture must not be opened")
            with patch.object(core.h5py, "File", wraps=h5py.File) as opened:
                with self.assertRaises(FileNotFoundError):
                    core.score_development_panel(directory, models, baselines, check=check)
            self.assertEqual(len(opened.call_args_list), 1)
            self.assertEqual(Path(opened.call_args.args[0]), Path(directory) / "development_targets.h5")
            self.assertEqual(closed.read_bytes(), b"closed fixture must not be opened")

    def test_zero_variance_and_misbound_source_records_never_drop_required_data(self):
        models, baselines, targets = fixture_panel()
        first = models[0]
        key = first.animal, first.target_day, first.region
        targets[key] = np.full_like(targets[key], 11.)
        with tempfile.TemporaryDirectory() as directory:
            path = write_targets(directory, models, targets)
            feedback = core.score_development_panel(directory, models, baselines, check=check)
            self.assertFalse(feedback[0].available)
            self.assertIsNone(feedback[0].model_r2)
            self.assertIsNone(feedback[0].improvement_over_mean)
            self.assertEqual(feedback[0].unavailable_reason,
                             "zero_joint_native_response_variance_in_required_day")
            self.assertTrue(feedback[1].available)
            record_path = "/".join(key + (first.trial_ids[0],))
            with h5py.File(path, "r+") as handle:
                handle[record_path].attrs["role"] = "held_out"
            with self.assertRaises(PermissionError):
                core.score_development_panel(directory, models, baselines, check=check)
            with h5py.File(path, "r+") as handle:
                handle[record_path].attrs["role"] = "development"
                handle[record_path]["unit_guide"][...] = first.unit_guide[::-1]
            with self.assertRaisesRegex(ValueError, "neuron axes differ"):
                core.score_development_panel(directory, models, baselines, check=check)
            with h5py.File(path, "r+") as handle:
                handle[record_path]["unit_guide"][...] = first.unit_guide
                del handle[record_path]
            with self.assertRaisesRegex(ValueError, "complete scorer day"):
                core.score_development_panel(directory, models, baselines, check=check)


if __name__ == "__main__":
    unittest.main()
