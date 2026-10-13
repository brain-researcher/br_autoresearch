"""Two role/identity checks for the reader; execute only in coordinated Slurm.

Fabricated HDF metadata has all seven required dates and whole-trial support.
Most neural datasets deliberately contain NaNs: neither constructing a reader
nor reading features may decode unopened outcomes. This is not empirical
eligibility, scientific recovery or permission to open the actual release.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import development_views as reader
from prepare_development_views import BANDS, KIND, OFFSETS, REGIONS, ROLES, SESSIONS


def check():
    pass


def fixture(path):
    """Source-order identities plus an independently constructed role split."""
    with h5py.File(path, "w") as handle:
        handle.attrs.update(kind=KIND, completed=True, expected_sessions=7)
        sessions = handle.create_group("sessions")
        for spec in SESSIONS:
            session = sessions.create_group(spec.session_id)
            session.attrs.update(animal=spec.animal, source_filename=spec.filename,
                                 original_position=spec.original_position, complete=True)
            session.create_dataset("bands", data=np.asarray(BANDS))
            session.create_dataset("bin_offsets", data=np.asarray(OFFSETS))
            assignments = {role: [] for role in ROLES}
            for d in range(8):
                rows = list(range(d, spec.trials, 8))
                train = len(rows) // 2
                cal = (len(rows) - train) // 2
                for role, indices in zip(ROLES, (rows[:train], rows[train:train + cal], rows[train + cal:])):
                    assignments[role].extend(indices)
            for region_number, region in enumerate(REGIONS):
                group = session.create_group(region)
                group.create_dataset("electrode_ids", data=np.arange(1, 97))
                group.create_dataset("training_finite_electrodes", data=np.ones(96, dtype=bool))
                group.create_dataset("selected_electrode_positions", data=np.arange(0, 30, 2))
                # Unequal native unit counts stay intact; a paired model owns
                # its later equal latent-rank requirement, not this reader.
                units = 2 + region_number
                group.create_dataset("unit_guide", data=np.column_stack((np.arange(1, units + 1), np.ones(units))))
                for role in ROLES:
                    rg = group.create_group(role)
                    indices = np.asarray(sorted(assignments[role]))
                    direction = indices % 8
                    rg.create_dataset("source_trial_indices", data=indices)
                    rg.create_dataset("trial_ids", data=[f"{spec.session_id}:row_{index + 1:06d}" for index in indices],
                                      dtype=h5py.string_dtype("utf-8"))
                    rg.create_dataset("direction", data=direction)
                    rg.create_dataset("direction_raw", data=direction * np.pi / 4)
                    halves = np.full(len(indices), -1)
                    if role == "calibration":
                        for d in range(8):
                            rows = np.flatnonzero(direction == d)
                            halves[rows] = np.arange(len(rows)) % 2
                    rg.create_dataset("calibration_half", data=halves)
                    rg.create_dataset("window_start", data=np.full(len(indices), 25))
                    rg.create_dataset("window_stop", data=np.full(len(indices), 35))
                    lfp = rg.create_dataset("lfp", shape=(len(indices), 10, 96, 9), dtype="f8", fillvalue=np.nan)
                    spikes = rg.create_dataset("spikes", shape=(len(indices), 10, units), dtype="f8", fillvalue=np.nan)
                    if spec == SESSIONS[0]:
                        # Distinct source, bin, electrode, band and region
                        # values detect transposition or a substituted subset.
                        lfp[:] = (indices[:, None, None, None] * 10000
                            + np.arange(10)[None, :, None, None] * 1000
                            + np.arange(96)[None, None, :, None] * 10
                            + np.arange(9)[None, None, None, :] + region_number * 1000000)
                        spikes[:] = (indices[:, None, None] * 100
                            + np.arange(10)[None, :, None] * 10 + np.arange(units)[None, None, :])


@contextmanager
def guard_values(deny):
    old_get, old_array = h5py.Dataset.__getitem__, h5py.Dataset.__array__

    def get(node, *args, **kwargs):
        if deny(node.name):
            raise AssertionError(f"forbidden neural value read: {node.name}")
        return old_get(node, *args, **kwargs)

    def array(node, *args, **kwargs):
        if deny(node.name):
            raise AssertionError(f"forbidden neural array coercion: {node.name}")
        return old_array(node, *args, **kwargs)

    with patch.object(h5py.Dataset, "__getitem__", get), patch.object(h5py.Dataset, "__array__", array):
        yield


class DevelopmentViewTests(unittest.TestCase):
    def test_candidate_feature_only_access_and_role_pinned_outcomes_preserve_native_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "development_views.h5"
            fixture(path)
            session = SESSIONS[0].filename
            with guard_values(lambda name: name.endswith("/spikes") and "/training/" not in name):
                candidate = reader.CandidateDevelopmentViews(path, check=check)
                lfp, observed = candidate.training(session, "M1")
                self.assertEqual(lfp.design, observed.design)
                self.assertEqual(lfp.electrode_ids, tuple(float(x) for x in range(1, 30, 2)))
                self.assertEqual(observed.values.shape[-1], 2)
                meta = candidate.metadata(session, "M1", "training")
                self.assertEqual(len(meta.native_electrode_ids), 96)
                expected = (np.asarray(meta.source_trial_indices)[:, None, None, None] * 10000
                    + np.arange(10)[None, :, None, None] * 1000
                    + np.arange(0, 30, 2)[None, None, :, None] * 10 + np.arange(9)[None, None, None, :])
                np.testing.assert_array_equal(lfp.values, expected)
                expected_counts = (np.asarray(meta.source_trial_indices)[:, None, None] * 100
                    + np.arange(10)[None, :, None] * 10 + np.arange(2)[None, None, :])
                np.testing.assert_array_equal(observed.values, expected_counts)
                for role in ("calibration", "evaluation"):
                    features = candidate.features(session, "PMd", role)
                    self.assertFalse(hasattr(features, "spikes"))
                    self.assertEqual(features.design.role, role)
                    self.assertEqual(len(features.design.axis_ids), 3)
                    self.assertTrue(set(features.design.trial_ids).isdisjoint(lfp.design.trial_ids))
                with self.assertRaises(PermissionError):
                    candidate.features(session, "M1", "training")
            # Outcome access never decodes LFP or the other trial role.
            for role in ("calibration", "evaluation"):
                with guard_values(lambda name: name.endswith("/lfp") or
                                  (name.endswith("/spikes") and f"/{role}/" not in name)):
                    scorer = reader.DevelopmentOutcomeViews(path, role=role, check=check)
                    outcome = scorer.targets(session, "PMd")
                    meta = scorer.metadata(session, "PMd")
                    self.assertEqual(outcome.design, meta.design)
                    expected = (np.asarray(meta.source_trial_indices)[:, None, None] * 100
                        + np.arange(10)[None, :, None] * 10 + np.arange(3)[None, None, :])
                    np.testing.assert_array_equal(outcome.values, expected)
                    with self.assertRaises(AttributeError):
                        scorer.role = "training"
                    with self.assertRaises(ValueError):
                        outcome.values.setflags(write=True)
            with self.assertRaises(PermissionError):
                reader.DevelopmentOutcomeViews(path, role="training", check=check)

    def test_partial_reserved_mismatched_and_dropped_support_fail_before_any_neural_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "development_views.h5"
            fixture(path)
            deny = lambda name: name.endswith(("/lfp", "/spikes"))
            with guard_values(deny):
                candidate = reader.CandidateDevelopmentViews(path, check=check)
                with self.assertRaises(PermissionError):
                    candidate.features("Mihili_CO_20140303.mat", "M1", "evaluation")
                def cancelled():
                    raise RuntimeError("cancelled")
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    reader.CandidateDevelopmentViews(path, check=cancelled)
            session = f"sessions/{SESSIONS[0].session_id}"
            mutations = (
                ("completion", lambda h: h.attrs.__setitem__("completed", False)),
                ("pairing", lambda h: (
                    h[session + "/PMd/evaluation/window_start"].__setitem__(slice(None), 26),
                    h[session + "/PMd/evaluation/window_stop"].__setitem__(slice(None), 36))),
                ("trial_identity", lambda h: h[session + "/M1/evaluation/trial_ids"].__setitem__(0, "wrong:row_000001")),
                ("fixed_subset", lambda h: h[session + "/M1/selected_electrode_positions"].__setitem__(0, 2)),
                ("native_units", lambda h: h[session + "/PMd/evaluation"].__delitem__("spikes")),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    fixture(path)
                    with h5py.File(path, "r+") as handle:
                        mutate(handle)
                    with guard_values(deny), self.assertRaises((ValueError, PermissionError)):
                        reader.CandidateDevelopmentViews(path, check=check)


if __name__ == "__main__":
    unittest.main()
