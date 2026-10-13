"""Four publication checks using temporary fabricated prediction artifacts.

These filesystem checks must run only in coordinator-owned compute. They test
publication/axis integrity, not model quality or empirical role qualification.
No file I/O, project imports or tests ran while writing this source.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

import candidate_predictions as core
from native_trials import NativePredictions, _snapshot


def check():
    pass


def pair(key):
    animal, date, region = key
    directions = np.repeat(np.arange(8), 6)
    offsets = np.arange(5, 15) if region == "M1" else np.arange(-15, 0)
    neurons = 2 if region == "M1" else 3
    ids = tuple(f"trial_{1001 + i:06d}" for i in range(48))
    calibration = tuple(f"trial_{i + 1:06d}" for i in range(24))
    units = np.array([(10 + i, 1) for i in range(neurons)])
    baseline = (directions[:, None, None] + np.arange(len(offsets))[None, :, None] / 10
                + np.arange(neurons)[None, None, :] / 5)
    values = baseline - .25 + np.arange(48)[:, None, None] / 100
    common = (animal, date, region, ids, _snapshot(directions), _snapshot(offsets), _snapshot(units))
    return (NativePredictions(*common, _snapshot(values), calibration),
            NativePredictions(*common, _snapshot(baseline), calibration))


def publish(path, *, candidate="starting_00", stream="history_assisted"):
    with core.CandidatePredictionWriter(path, candidate_id=candidate, stream=stream, check=check) as writer:
        for key in core.PANEL_KEYS[::-1]:
            writer.write(*pair(key))


def load(path, *, candidate="starting_00", stream="history_assisted", callback=check):
    return core.load_candidate_predictions(path, expected_candidate_id=candidate,
                                           expected_stream=stream, check=callback)


class CandidatePublicationTests(unittest.TestCase):
    def test_incremental_complete_panel_roundtrip_native_axes_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.h5"
            publish(path)
            result = load(path)
            self.assertEqual((result.candidate_id, result.stream), ("starting_00", "history_assisted"))
            self.assertEqual(result.check_record, core.PublicationCheck(core.CHECK_KIND, 10))
            self.assertEqual(tuple((value.animal, value.target_day, value.region)
                                  for value in result.predictions), core.PANEL_KEYS)
            for key, prediction, baseline in zip(core.PANEL_KEYS, result.predictions, result.calibration_means):
                expected_prediction, expected_baseline = pair(key)
                self.assertEqual(prediction.trial_ids, expected_prediction.trial_ids)
                self.assertEqual(prediction.calibration_trial_ids, expected_prediction.calibration_trial_ids)
                for name in ("directions", "bin_offsets", "unit_guide", "values"):
                    np.testing.assert_array_equal(getattr(prediction, name), getattr(expected_prediction, name))
                    np.testing.assert_array_equal(getattr(baseline, name), getattr(expected_baseline, name))
                self.assertEqual(prediction.values.dtype, np.float64)
                self.assertLess(prediction.values[0, 0, 0], 0.)
                with self.assertRaises(ValueError):
                    prediction.values.setflags(write=True)
            with self.assertRaises(OSError):
                publish(path)
            # Failed overwrite does not modify the accepted original artifact.
            np.testing.assert_array_equal(load(path).predictions[0].values, result.predictions[0].values)
            for candidate, stream in (("starting_01", "history_assisted"), ("starting_00", "today_only")):
                with self.assertRaises(ValueError):
                    load(path, candidate=candidate, stream=stream)

    def test_incomplete_exception_and_cancellation_never_publish_a_complete_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            incomplete = Path(directory) / "incomplete.h5"
            with self.assertRaises(ValueError):
                with core.CandidatePredictionWriter(incomplete, candidate_id="starting_00",
                        stream="history_assisted", check=check) as writer:
                    writer.write(*pair(core.PANEL_KEYS[0]))
            failure = Path(directory) / "failure.h5"
            with self.assertRaisesRegex(RuntimeError, "candidate failed"):
                with core.CandidatePredictionWriter(failure, candidate_id="starting_00",
                        stream="history_assisted", check=check) as writer:
                    for key in core.PANEL_KEYS:
                        writer.write(*pair(key))
                    raise RuntimeError("candidate failed")
            cancelled = Path(directory) / "cancelled.h5"
            calls = []
            def cancel():
                calls.append(None)
                if len(calls) == 4:
                    raise RuntimeError("cancelled after cell write")
            with self.assertRaisesRegex(RuntimeError, "cancelled after cell write"):
                with core.CandidatePredictionWriter(cancelled, candidate_id="starting_00",
                        stream="history_assisted", check=cancel) as writer:
                    writer.write(*pair(core.PANEL_KEYS[0]))
            self.assertEqual(len(calls), 4)
            for path in (incomplete, failure, cancelled):
                self.assertTrue(path.exists())
                with h5py.File(path, "r") as handle:
                    self.assertFalse(handle.attrs["completed"])
                with self.assertRaises(ValueError):
                    load(path)

    def test_duplicate_mismatched_or_closed_role_records_poison_publication(self):
        candidate, baseline = pair(core.PANEL_KEYS[0])
        not_finite = candidate.values.copy()
        not_finite[-1, -1, -1] = np.nan
        nonconstant = baseline.values.copy()
        nonconstant[1, 0, 0] += 1
        cases = (
            (replace(candidate, trial_ids=(candidate.calibration_trial_ids[0],) + candidate.trial_ids[1:]), baseline),
            (candidate, replace(baseline, unit_guide=_snapshot(baseline.unit_guide[::-1]))),
            (replace(candidate, values=_snapshot(not_finite)), baseline),
            (candidate, replace(baseline, values=_snapshot(nonconstant))),
            (replace(candidate, target_day="20140306"), baseline),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, values in enumerate(cases):
                path = Path(directory) / f"invalid_{index}.h5"
                with self.assertRaises(ValueError):
                    with core.CandidatePredictionWriter(path, candidate_id="starting_00",
                            stream="history_assisted", check=check) as writer:
                        writer.write(*values)
                with self.assertRaises(ValueError):
                    load(path)
            duplicate = Path(directory) / "duplicate.h5"
            with self.assertRaisesRegex(ValueError, "publication failed"):
                with core.CandidatePredictionWriter(duplicate, candidate_id="starting_00",
                        stream="history_assisted", check=check) as writer:
                    writer.write(candidate, baseline)
                    with self.assertRaises(ValueError):
                        writer.write(candidate, baseline)
                    with self.assertRaises(RuntimeError):
                        writer.write(*pair(core.PANEL_KEYS[1]))
            with self.assertRaises(ValueError):
                load(duplicate)
            mismatch = Path(directory) / "regions.h5"
            other, other_baseline = pair(core.PANEL_KEYS[1])
            changed_ids = tuple(f"trial_{2001 + i:06d}" for i in range(48))
            with self.assertRaisesRegex(ValueError, "M1 and PMd"):
                with core.CandidatePredictionWriter(mismatch, candidate_id="starting_00",
                        stream="history_assisted", check=check) as writer:
                    writer.write(candidate, baseline)
                    writer.write(replace(other, trial_ids=changed_ids), replace(other_baseline, trial_ids=changed_ids))

    def test_completed_marker_alone_cannot_hide_missing_cells_or_redirect_value_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.h5"
            publish(missing)
            with h5py.File(missing, "r+") as handle:
                del handle["predictions/Mihili/20140304/PMd"]
            with self.assertRaises(ValueError):
                load(missing)
            redirected = Path(directory) / "redirected.h5"
            publish(redirected)
            # A nonexistent external file guarantees the loader must reject the
            # link itself, before HDF5 could try to open its referenced values.
            with h5py.File(redirected, "r+") as handle:
                group = handle["predictions/Mihili/20140304/M1"]
                del group["prediction"]
                group["prediction"] = h5py.ExternalLink(str(Path(directory) / "closed_source.h5"), "/response")
            with self.assertRaisesRegex(ValueError, "external HDF5 links"):
                load(redirected)
            complete = Path(directory) / "complete.h5"
            publish(complete)
            def cancel():
                raise RuntimeError("stop before opening")
            with self.assertRaisesRegex(RuntimeError, "stop before opening"):
                load(complete, callback=cancel)
            self.assertEqual(len(load(complete).predictions), 10)


if __name__ == "__main__":
    unittest.main()
