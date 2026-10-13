"""Two fabricated transport checks; run only in coordinator-owned compute.

These tests mock the already separately tested primary arithmetic engine. They
exercise literal HDF target/identity transport and durable caller-owned opening
order, not empirical qualification or permission to open real held-out data.
All files are temporary fabricated fixtures; no test was executed on login.
"""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import final_score as core
from final_predictions import HeldOutPredictionWriter
from test_candidate_predictions import pair


HISTORY, TODAY = "adaptive_03", "adaptive_07"


def check():
    pass


def fixture(directory):
    prediction_path, heldout_path = Path(directory) / "predictions.h5", Path(directory) / "heldout.h5"
    expected = {}
    keys = tuple((animal, day, region) for animal, day in core.DAY_ORDER for region in core.REGIONS)
    with HeldOutPredictionWriter(prediction_path, history_candidate_id=HISTORY,
            today_candidate_id=TODAY, check=check) as writer, h5py.File(heldout_path, "x") as handle:
        for cell_index, key in reversed(tuple(enumerate(keys))):
            history, baseline = pair(key)
            # Reverse prediction order independently of the HDF5 lexical order.
            def reverse(value):
                return replace(value, trial_ids=value.trial_ids[::-1],
                    directions=value.directions[::-1], values=value.values[::-1])
            history, baseline = reverse(history), reverse(baseline)
            today = replace(history, values=history.values + .125)
            writer.write(history, today, baseline)
            counts = []
            for row, trial in enumerate(history.trial_ids):
                number = int(trial[6:]) - 1001
                y = (10000 * cell_index + 100 * number
                     + 5 * np.arange(len(history.bin_offsets))[:, None]
                     + np.arange(len(history.unit_guide))[None, :]).astype(np.int64)
                record = handle.create_group("/".join((*key, trial)))
                record.attrs.update(role="held_out", direction=int(history.directions[row]),
                                    native_bin_seconds=.03)
                record.create_dataset("spikes", data=y)
                record.create_dataset("unit_guide", data=history.unit_guide)
                record.create_dataset("bin_offsets", data=history.bin_offsets)
                # Names only: the scorer must never dereference these objects.
                record.create_dataset("lfp", data=[-999.])
                record.create_dataset("lfp_guide", data=[-999.])
                counts.append(y)
            expected[key] = history, today, baseline, np.stack(counts)
    return prediction_path, heldout_path, expected


def score(prediction_path, heldout_path, begin):
    return core.score_primary_held_out(heldout_path, prediction_path,
        history_candidate_id=HISTORY, today_candidate_id=TODAY,
        begin_opening=begin, check=check)


class FinalScoreTests(unittest.TestCase):
    def test_complete_paired_alignment_no_lfp_access_and_one_durable_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, heldout, expected = fixture(directory)
            claim = Path(directory) / "primary_opening.json"
            events = []
            def begin():
                with claim.open("x") as handle:
                    handle.write('{"held_out_opened":true}\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                events.append("claim")
            real_file, real_group = h5py.File, h5py.Group.__getitem__
            def guarded_file(path, *args, **kwargs):
                if Path(path) == heldout:
                    self.assertTrue(claim.exists())
                    self.assertEqual(events, ["claim"])
                    events.append("held_file")
                return real_file(path, *args, **kwargs)
            def guarded_group(group, key):
                if str(key).rsplit("/", 1)[-1] in ("lfp", "lfp_guide"):
                    raise AssertionError("scorer dereferenced LFP features")
                return real_group(group, key)
            sentinel = object()
            with patch("final_score.h5py.File", side_effect=guarded_file), \
                    patch.object(h5py.Group, "__getitem__", guarded_group), \
                    patch("final_score.infer_primary", return_value=sentinel) as inference:
                self.assertIs(score(predictions, heldout, begin), sentinel)
                cells = inference.call_args.args[0]
                self.assertEqual(len(cells), 10)
                self.assertEqual(tuple((c.animal, c.day, c.region) for c in cells), tuple(
                    (animal, day, region) for animal, day in core.DAY_ORDER for region in core.REGIONS))
                for cell in cells:
                    h, t, baseline, observed = expected[cell.animal, cell.day, cell.region]
                    self.assertEqual(cell.role, "held_out")
                    self.assertEqual(cell.trial_ids, h.trial_ids)
                    np.testing.assert_array_equal(cell.observed, observed)
                    np.testing.assert_array_equal(cell.history.values, h.values)
                    np.testing.assert_array_equal(cell.today.values, t.values)
                    np.testing.assert_array_equal(cell.calibration_mean.values, baseline.values)
                    np.testing.assert_array_equal(cell.unit_guide, h.unit_guide)
                    np.testing.assert_array_equal(cell.bin_offsets, h.bin_offsets)
                    np.testing.assert_array_equal(cell.directions, h.directions)
                    with self.assertRaises(ValueError):
                        cell.observed.setflags(write=True)
                # Caller refuses an existing opening; no second held-file read
                # or inference invocation occurs, even though predictions load.
                with self.assertRaises(FileExistsError):
                    score(predictions, heldout, begin)
                inference.assert_called_once()
            self.assertEqual(events, ["claim", "held_file"])
            self.assertIn('"held_out_opened":true', claim.read_text())

    def test_invalid_panel_or_late_target_binding_never_reads_partial_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, heldout, _ = fixture(directory)
            claims = []
            begin = lambda: claims.append("claimed")
            last = "Chewie-L/20161021/PMd/trial_001001"
            real_file, real_dataset = h5py.File, h5py.Dataset.__getitem__
            def no_held_file(path, *args, **kwargs):
                if Path(path) == heldout:
                    raise AssertionError("held file opened before complete prediction admission")
                return real_file(path, *args, **kwargs)
            with h5py.File(predictions, "r+") as handle:
                handle.attrs["completed"] = False
            with patch("final_score.h5py.File", side_effect=no_held_file):
                with self.assertRaises(ValueError):
                    score(predictions, heldout, begin)
            self.assertEqual(claims, [])
            with h5py.File(predictions, "r+") as handle:
                handle.attrs["completed"] = True
            def no_counts(dataset, key):
                if dataset.name.rsplit("/", 1)[-1] == "spikes":
                    raise AssertionError("read counts before complete held metadata binding")
                return real_dataset(dataset, key)
            # Each defect is in the final cell/trial; none may permit reading
            # earlier valid counts or silently reduce/reorder the fixed panel.
            for kind in ("extra_trial", "missing_trial", "wrong_role", "swapped_units"):
                with self.subTest(kind=kind):
                    with h5py.File(heldout, "r+") as handle:
                        if kind == "extra_trial":
                            handle["Chewie-L/20161021/PMd"].create_group("trial_999999")
                        elif kind == "missing_trial":
                            saved_attrs = dict(handle[last].attrs)
                            saved_fields = {name: dataset[...] for name, dataset in handle[last].items()}
                            del handle[last]
                        elif kind == "wrong_role":
                            handle[last].attrs["role"] = "development"
                        else:
                            guide = handle[last + "/unit_guide"][...]
                            handle[last + "/unit_guide"][...] = guide[::-1]
                    with patch.object(h5py.Dataset, "__getitem__", no_counts), \
                            patch("final_score.infer_primary", side_effect=AssertionError("invalid panel scored")):
                        with self.assertRaises((ValueError, PermissionError)):
                            score(predictions, heldout, begin)
                    with h5py.File(heldout, "r+") as handle:
                        if kind == "extra_trial":
                            del handle["Chewie-L/20161021/PMd/trial_999999"]
                        elif kind == "missing_trial":
                            restored = handle.create_group(last)
                            restored.attrs.update(saved_attrs)
                            for field, values in saved_fields.items():
                                restored.create_dataset(field, data=values)
                        elif kind == "wrong_role":
                            handle[last].attrs["role"] = "held_out"
                        else:
                            handle[last + "/unit_guide"][...] = guide
            self.assertEqual(claims, ["claimed"] * 4)


if __name__ == "__main__":
    unittest.main()
