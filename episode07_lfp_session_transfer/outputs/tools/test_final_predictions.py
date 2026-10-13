"""Paired publication checks using fabricated arrays in compute-local storage."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

import final_predictions as core
from test_candidate_predictions import pair


def check():
    pass


def writer(path):
    return core.HeldOutPredictionWriter(path, history_candidate_id="adaptive_03",
                                       today_candidate_id="adaptive_07", check=check)


def load(path):
    return core.load_held_out_predictions(path, history_candidate_id="adaptive_03",
                                          today_candidate_id="adaptive_07", check=check)


class FinalPublicationTests(unittest.TestCase):
    def test_complete_paired_native_panel_roundtrip_and_stream_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.h5"
            with writer(path) as output:
                for key in core.PANEL_KEYS[::-1]:
                    history, baseline = pair(key)
                    output.write(history, replace(history, values=history.values - 1), baseline)
            actual = load(path)
            self.assertEqual(actual.history_candidate_id, "adaptive_03")
            self.assertEqual(actual.today_candidate_id, "adaptive_07")
            self.assertEqual(len(actual.history), 10)
            for key, history, today, baseline in zip(core.PANEL_KEYS, actual.history,
                    actual.today, actual.calibration_means):
                expected, mean = pair(key)
                self.assertEqual(history.trial_ids, expected.trial_ids)
                self.assertEqual(today.calibration_trial_ids, mean.calibration_trial_ids)
                np.testing.assert_array_equal(history.values, expected.values)
                np.testing.assert_array_equal(today.values, expected.values - 1)
                np.testing.assert_array_equal(baseline.values, mean.values)
                with self.assertRaises(ValueError):
                    history.values.setflags(write=True)
            with self.assertRaises(OSError):
                with writer(path):
                    pass
            with self.assertRaises(ValueError):
                core.load_held_out_predictions(path, history_candidate_id="adaptive_07",
                    today_candidate_id="adaptive_03", check=check)
            with h5py.File(path, "r+") as handle:
                self.assertEqual(handle.attrs["role"], "held_out")
                handle.attrs["role"] = "development"
            with self.assertRaises(ValueError):
                load(path)

    def test_missing_or_misaligned_stream_and_interrupted_publication_stay_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            history, baseline = pair(core.PANEL_KEYS[0])
            mismatch = replace(history, trial_ids=history.trial_ids[::-1])
            for index, today in enumerate((None, mismatch)):
                path = Path(directory) / f"bad{index}.h5"
                with self.assertRaises((TypeError, ValueError)):
                    with writer(path) as output:
                        output.write(history, today, baseline)
                with self.assertRaises(ValueError):
                    load(path)
            missing = Path(directory) / "partial.h5"
            with self.assertRaises(ValueError):
                with writer(missing) as output:
                    output.write(history, history, baseline)
            with self.assertRaises(ValueError):
                load(missing)
            interrupted = Path(directory) / "interrupted.h5"
            with self.assertRaisesRegex(RuntimeError, "before complete"):
                with writer(interrupted) as output:
                    for key in core.PANEL_KEYS:
                        prediction, mean = pair(key)
                        output.write(prediction, prediction, mean)
                    raise RuntimeError("before complete")
            with self.assertRaises(ValueError):
                load(interrupted)
            with h5py.File(interrupted, "r") as handle:
                self.assertFalse(handle.attrs["completed"])


if __name__ == "__main__":
    unittest.main()
