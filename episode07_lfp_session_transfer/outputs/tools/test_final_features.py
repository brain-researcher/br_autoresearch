"""Four source-bound held-feature transport tests; run only in coordinated Slurm.

Temporary HDF5 files and native arrays here are fabricated fixtures. They test
closed response access, role preservation and frozen arithmetic, not empirical
qualification, finalist eligibility or permission to open any real held-out file.
No imports or tests were executed while authoring this file.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from final_view_access import HeldOutFeatureViews
from lfp_features import transform_lfp_features, transform_lfp_held_out
from native_trials import (adapt_records, adapt_held_out_records, load_native_trials,
    load_held_out_trials, fit_calibration_mean, predict_calibration_mean,
    predict_calibration_mean_held_out)
from primary_design import transform_primary_features
from primary_model import (fit_primary_model, predict_primary_development, predict_primary_held_out)
from test_population_bridge import full_roster
from test_primary_model import recipe, development
from view_access import CandidateViews


SCOPE = dict(animal="Mihili", target_day="20140304", region="M1", stream="today_only")


def check():
    pass


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("refused feature or response values were accessed")


def records(count=48):
    target = full_roster()[0][-1]
    # Cross the six/seven-digit boundary to distinguish numeric from lexical
    # whole-trial ordering. Complete support is six per direction when count48.
    return [dict(animal="Mihili", day="20140304", region="M1",
        trial_id=f"trial_{999990 + index:06d}", role="held_out", direction=index // 6,
        lfp=np.full((10, 9), index / 10.), lfp_guide=target.lfp_guide,
        bin_offsets=np.arange(5, 15, dtype=np.int64)) for index in range(count)]


def write_fixture(path, *, count=48):
    with h5py.File(path, "x") as handle:
        for record in records(count):
            group = handle.create_group(f"Mihili/20140304/M1/{record['trial_id']}")
            group.attrs["role"], group.attrs["direction"] = record["role"], record["direction"]
            for name in ("lfp", "lfp_guide", "bin_offsets"):
                group.create_dataset(name, data=record[name])
            group.create_dataset("spikes", data=np.full((10, 7), 999.))
            group.create_dataset("unit_guide", data=np.column_stack((np.arange(7), np.ones(7))))


class FinalFeatureTests(unittest.TestCase):
    def test_reader_never_dereferences_responses_and_adapter_preserves_held_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "heldout.h5"
            write_fixture(path)
            fetched = []
            original = h5py.Group.__getitem__
            def guarded(group, key):
                name = str(key).rsplit("/", 1)[-1]
                if name in ("spikes", "unit_guide"):
                    raise AssertionError("literal response dataset dereference")
                fetched.append(name)
                return original(group, key)
            with patch.object(h5py.Group, "__getitem__", guarded):
                result = load_held_out_trials(HeldOutFeatureViews(path), **SCOPE, check=check)
            self.assertEqual(len(result), 1)
            day = result[0]
            self.assertEqual((day.view_role, day.role), ("held_out_features", "held_out"))
            self.assertEqual(day.trial_ids, tuple(record["trial_id"] for record in records()))
            self.assertIsNone(day.spikes)
            self.assertIsNone(day.unit_guide)
            np.testing.assert_array_equal(day.directions, np.repeat(np.arange(8), 6))
            np.testing.assert_allclose(day.lfp[:, 0, 0], np.arange(48) / 10., rtol=0, atol=0)
            self.assertIn("lfp", fetched)
            self.assertNotIn("spikes", fetched)
            for value in (day.lfp, day.lfp_guide, day.bin_offsets, day.directions):
                with self.assertRaises(ValueError):
                    value.setflags(write=True)

    def test_candidate_interfaces_stay_closed_and_final_adapter_rejects_mixed_roles_before_values(self):
        sample = records()
        with patch("view_access.h5py.File", side_effect=AssertionError("closed role opened a file")):
            with self.assertRaises(PermissionError):
                list(CandidateViews("unused").iter_records("held_out_features", **SCOPE))
            with self.assertRaises(PermissionError):
                load_native_trials(CandidateViews("unused"), view_role="held_out_features", **SCOPE, check=check)
        with self.assertRaises(PermissionError):
            adapt_records(sample, view_role="held_out_features", **SCOPE, check=check)
        day = adapt_held_out_records(sample, **SCOPE, check=check)[0]
        with self.assertRaises(PermissionError):
            fit_primary_model((day,), recipe=recipe("today_only"), check=check)
        for changes in (dict(role="development"), dict(day="20140303"),
                        dict(spikes=ForbiddenArray(), unit_guide=ForbiddenArray())):
            bad = dict(sample[0], lfp=ForbiddenArray(), **changes)
            with self.assertRaises(ValueError):
                adapt_held_out_records([bad], **SCOPE, check=check)
        incomplete = sample[:-1]
        with self.assertRaises(ValueError):
            adapt_held_out_records(incomplete, **SCOPE, check=check)
        def cancelled():
            raise RuntimeError("cancel before final read")
        with patch("final_view_access.h5py.File", side_effect=AssertionError("read after cancel")):
            with self.assertRaisesRegex(RuntimeError, "cancel before final read"):
                load_held_out_trials(HeldOutFeatureViews("unused"), **SCOPE, check=cancelled)

    def test_frozen_prediction_matches_native_oracle_and_development_without_relabeling(self):
        target = replace(full_roster()[0][-1], stream="today_only")
        fitted = fit_primary_model((target,), recipe=recipe("today_only"), check=check)
        baseline = fit_calibration_mean(target, check=check)
        dev = development(target)
        held = replace(dev, view_role="held_out_features", role="held_out",
                       trial_ids=tuple(f"trial_{2001 + index:06d}" for index in range(48)))
        expected = np.stack([target.spikes[np.flatnonzero(target.directions == direction)[0]]
                             for direction in held.directions])
        development_prediction = predict_primary_development(fitted, dev, check=check)
        # Spy on the actual numerical kernels, not just the outer wrapper: a
        # hidden conversion to development would fail at either stage here.
        import lfp_features
        import primary_design
        original_data, original_features = lfp_features._data, primary_design._features
        seen = []
        def native_values(day):
            self.assertEqual((day.view_role, day.role), ("held_out_features", "held_out"))
            seen.append("native_held")
            return original_data(day)
        def feature_values(day):
            self.assertEqual((day.view_role, day.role), ("held_out_features", "held_out"))
            seen.append("features_held")
            return original_features(day)
        with patch("lfp_features._data", side_effect=native_values), \
                patch("primary_design._features", side_effect=feature_values), \
                patch("primary_model.fit_regression", side_effect=AssertionError("prediction refit")), \
                patch("primary_model.fit_lfp_features", side_effect=AssertionError("prediction refit")), \
                patch("primary_model.fit_primary_population", side_effect=AssertionError("prediction refit")), \
                patch("primary_model.fit_primary_design", side_effect=AssertionError("prediction refit")):
            actual = predict_primary_held_out(fitted, held, check=check)
        self.assertEqual(seen, ["native_held", "features_held"])
        self.assertEqual(actual.trial_ids, held.trial_ids)
        self.assertEqual(actual.calibration_trial_ids, target.trial_ids)
        np.testing.assert_array_equal(actual.unit_guide, target.unit_guide)
        np.testing.assert_allclose(actual.values, expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(actual.values, development_prediction.values, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(predict_calibration_mean_held_out(baseline, held, check=check).values,
                                      expected)
        with self.assertRaises(PermissionError):
            predict_primary_development(fitted, held, check=check)
        with self.assertRaises(PermissionError):
            predict_primary_held_out(fitted, dev, check=check)
        with self.assertRaises(PermissionError):
            predict_calibration_mean(baseline, held, check=check)
        with self.assertRaises(PermissionError):
            transform_lfp_features(fitted.features, held, check=check)
        features = transform_lfp_held_out(fitted.features, held, check=check)
        with self.assertRaises(PermissionError):
            transform_primary_features(fitted.design, features, check=check)
        unread = replace(held, lfp=ForbiddenArray())
        for bad in (replace(unread, trial_ids=(target.trial_ids[0],) + held.trial_ids[1:]),
                    replace(unread, day="20140307", target_day="20140307"),
                    replace(unread, spikes=ForbiddenArray()),
                    replace(unread, role="development")):
            with self.assertRaises(PermissionError):
                predict_primary_held_out(fitted, bad, check=check)
        with self.assertRaises(ValueError):
            predict_primary_held_out(fitted, replace(unread, directions=np.full(48, 8)), check=check)

    def test_linked_or_external_feature_storage_is_refused_before_any_feature_values(self):
        for kind in ("soft", "external_link", "virtual", "external_storage"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "heldout.h5"
                write_fixture(path, count=1)
                with h5py.File(path, "r+") as handle:
                    group = handle["Mihili/20140304/M1/trial_999990"]
                    del group["lfp"]
                    if kind == "soft":
                        group["lfp"] = h5py.SoftLink("spikes")
                    elif kind == "external_link":
                        group["lfp"] = h5py.ExternalLink(str(Path(temporary) / "absent.h5"), "spikes")
                    elif kind == "virtual":
                        layout = h5py.VirtualLayout(shape=(10, 9), dtype=np.float64)
                        layout[:] = h5py.VirtualSource(str(Path(temporary) / "absent.h5"),
                                                      "spikes", shape=(10, 9))
                        group.create_virtual_dataset("lfp", layout)
                    else:
                        group.create_dataset("lfp", shape=(10, 9), dtype=np.float64,
                            external=[(str(Path(temporary) / "absent.raw"), 0, h5py.h5f.UNLIMITED)])
                with patch.object(h5py.Dataset, "__array__",
                                  side_effect=AssertionError("read feature values before refusing storage")):
                    with self.assertRaises(ValueError):
                        list(HeldOutFeatureViews(path).iter_records(**SCOPE, check=check))


if __name__ == "__main__":
    unittest.main()
