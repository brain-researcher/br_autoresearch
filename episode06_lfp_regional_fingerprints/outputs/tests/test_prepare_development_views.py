"""Two bounded exporter checks; execute only in coordinated Slurm.

Fabricated metadata uses the fixed seven session identities, not empirical
neural values. A small HDF fixture tests the newly implemented window/export
logic with a stub for the independently checked MATLAB metadata dereferencer.
These are access/indexing checks, not a synthetic research or recovery study.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_development_views as export


def check():
    pass


def metadata_fixture(spec):
    # Original columns are reversed physical IDs inside reversed band order;
    # this deliberately differs from the exported electrode-major grid.
    guide = [[float(e), *band] for band in reversed(export.BANDS) for e in range(96, 0, -1)]
    guides = {"M1_lfp_guide": [guide], "PMd_lfp_guide": [guide],
              "M1_unit_guide": [[[8., 1.], [4., 2.], [11., 1.]]],
              "PMd_unit_guide": [[[22., 1.], [6., 2.]] ]}
    rows = []
    for index in range(spec.trials):
        direction = index % 8
        angle = direction * np.pi / 4
        if direction >= 4:
            angle -= 2 * np.pi
        angle += .002 if direction == 3 else 0.
        rows.append({"source_trial_index_zero_based": index,
            "structural_trial_key": f"{spec.session_id}:row_{index + 1:06d}",
            "metadata_and_shape_checks_pass": True,
            "metadata": {"bin_size": .03, "result": "R", "tgtDir": float(angle),
                         "idx_movement_on": 11., "idx_endTime": 40.},
            "shared_time_bins": 40, "array_headers": {},
            "guide_catalog_indices": {name: 0 for name in guides}})
    return {"filename": spec.filename, "contract_animal": spec.animal,
            "declared_day": spec.day, "status": "metadata_inventory_complete",
            "problem_counts": {}, "source_trial_count": spec.trials,
            "trials": rows, "guide_catalog": guides}


def handoff_fixture(reports):
    return {"source_episode": "episode07_lfp_session_transfer",
        "analysis_frozen_utc": "2020-01-01T00:00:00+00:00",
        "primary_opening_started_utc": "2020-01-01T00:01:00+00:00",
        "primary_opening_ended_utc": "2020-01-01T00:02:00+00:00",
        "primary_opening_complete": True, "ep07_views": "source_support_attempt002",
        "excluded_session": "Mihili_CO_20140306.mat",
        "records": [{"animal": spec.animal, "session_name": spec.filename,
                     "source_trial_index_zero_based": row["source_trial_index_zero_based"],
                     "ep07_role": "fixture_explicit_role"}
                    for spec, report in zip(export.SESSIONS, reports) for row in report["trials"]]}


def source_fixture(path, spec, report, plan):
    """Real HDF numerical arrays, with a metadata stub at the old reader API."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("trial_data")
        group.create_dataset("monkey", data=np.zeros(spec.trials, dtype=np.uint8))
        for row in plan:
            index = row["source_trial_index_zero_based"]
            trial = group.create_group(f"trial_{index:06d}")
            for region_index, region in enumerate(export.REGIONS):
                guide = report["guide_catalog"][region + "_lfp_guide"][0]
                values = np.asarray([[index * 1000000 + region_index * 1000000000
                                      + e * 1000 + export.BANDS.index(tuple(band)) * 100 + time
                                      for time in range(40)] for e, *band in guide], dtype=np.float64)
                # Electrode1 is nonfinite throughout training and evaluation
                # in M1. The training rule excludes it; no row is discarded.
                if region == "M1":
                    bad_column = next(i for i, value in enumerate(guide) if value == [1., *export.BANDS[0]])
                    values[bad_column, 17] = np.nan
                trial.create_dataset(region + "_lfp", data=values)
                units = len(report["guide_catalog"][region + "_unit_guide"][0])
                counts = (index * 100 + np.arange(units)[:, None] * 10 + np.arange(40)[None, :])
                trial.create_dataset(region + "_spikes", data=counts.astype(np.float64))

    def inspect_trial(handle, group, index, animal, day):
        row = deepcopy(report["trials"][index])
        row["guides"] = {name: values[0] for name, values in report["guide_catalog"].items()}
        row["problems"] = []
        return row

    def field(handle, group, index, name):
        return group[f"trial_{index:06d}/{name}"]

    return SimpleNamespace(inspect_trial=inspect_trial, field=field)


class PrepareDevelopmentViewsTests(unittest.TestCase):
    def test_fixed_complete_role_halves_and_handoff_precede_scientific_import_or_source_access(self):
        spec = export.SESSIONS[0]
        report = metadata_fixture(spec)
        before = np.random.get_state()
        plan = export.plan_session_roles(spec, report, check=check)
        self.assertEqual(plan, export.plan_session_roles(spec, report, check=check))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(tuple(row["source_trial_index_zero_based"] for row in plan), tuple(range(208)))
        self.assertEqual(tuple(row["direction_raw"] for row in plan),
                         tuple(row["metadata"]["tgtDir"] for row in report["trials"]))
        for direction in range(8):
            rows = [i for i in range(208) if i % 8 == direction]
            # Reproduce only the declared random schedule, not exporter output
            # or its role assignment helper, to bind seed/order conventions.
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([6006101, 0, 1, direction])))
            shuffled = rng.permutation(rows).tolist()
            expected = {"training": sorted(shuffled[:13]), "calibration": sorted(shuffled[13:19]),
                        "evaluation": sorted(shuffled[19:])}
            for role, indices in expected.items():
                actual = [row["source_trial_index_zero_based"] for row in plan
                          if row["direction"] == direction and row["role"] == role]
                self.assertEqual(actual, indices)
                for ordinal, index in enumerate(indices):
                    self.assertEqual(plan[index]["calibration_half"], ordinal % 2 if role == "calibration" else -1)
        self.assertTrue(all((row["window_start"], row["window_stop"]) == (15, 25) for row in plan))
        for mutate in (lambda r: r["trials"][0]["metadata"].__setitem__("tgtDir", .006),
                       lambda r: r["trials"][0]["metadata"].__setitem__("idx_endTime", 24.),
                       lambda r: r["trials"][0].__setitem__("source_trial_index_zero_based", 1)):
            changed = deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                export.plan_session_roles(spec, changed, check=check)
        reports = tuple(metadata_fixture(value) for value in export.SESSIONS)
        with tempfile.TemporaryDirectory() as temporary:
            handoff_path = Path(temporary) / "ep07_primary_handoff.json"
            output = Path(temporary) / "export"
            handoff = handoff_fixture(reports)
            with patch.object(export, "HANDOFF_PATH", handoff_path):
                handoff_path.write_text(json.dumps(handoff))
                loaded, roles = export._handoff(handoff_path, reports)
                self.assertTrue(loaded["primary_opening_complete"])
                self.assertEqual(len(roles), 1954)
                for bad in (dict(handoff, primary_opening_complete=False),
                            dict(handoff, records=handoff["records"][:-1]),
                            dict(handoff, analysis_frozen_utc="2020-01-01T00:03:00+00:00")):
                    handoff_path.write_text(json.dumps(bad))
                    with patch.object(export, "_compute_only"), \
                         patch.object(export, "_metadata", return_value=({}, reports)), \
                         patch.object(export, "_runtime", side_effect=AssertionError("scientific import reached")):
                        with self.assertRaises((PermissionError, ValueError)):
                            export.prepare_development_views(opening_handoff=handoff_path,
                                                             output_directory=output, check=check)
                    self.assertFalse(output.exists())
        def cancelled():
            raise RuntimeError("fixture cancelled")
        with patch.object(export, "_runtime", side_effect=AssertionError("runtime reached")), \
             self.assertRaisesRegex(RuntimeError, "fixture cancelled"):
            export.plan_session_roles(spec, report, check=cancelled)

    def test_native_window_guides_counts_and_training_only_subset_with_no_evaluation_redraw(self):
        spec = export.SESSIONS[0]
        report = metadata_fixture(spec)
        plan = export.plan_session_roles(spec, report, check=check)
        with tempfile.TemporaryDirectory() as temporary:
            source_path, out_path = Path(temporary) / "source.h5", Path(temporary) / "views.h5"
            source = source_fixture(source_path, spec, report, plan)
            with h5py.File(source_path, "r") as handle, h5py.File(out_path, "w") as output:
                export._export_session(handle, spec, report, plan, output, source, np, h5py, check)
            selected_by_region = {}
            with h5py.File(out_path, "r") as output:
                session = output[spec.session_id]
                self.assertTrue(session.attrs["complete"])
                for region_index, region in enumerate(export.REGIONS):
                    group = session[region]
                    eligible = np.arange(1, 96) if region == "M1" else np.arange(96)
                    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([6006102, 0, 1, region_index])))
                    expected_positions = np.sort(rng.choice(eligible, 15, replace=False))
                    np.testing.assert_array_equal(group["selected_electrode_positions"], expected_positions)
                    selected_by_region[region] = expected_positions
                    np.testing.assert_array_equal(group["unit_guide"], report["guide_catalog"][region + "_unit_guide"][0])
                    seen = []
                    for role in export.ROLES:
                        rg = group[role]
                        indices = [row["source_trial_index_zero_based"] for row in plan if row["role"] == role]
                        seen.extend(indices)
                        np.testing.assert_array_equal(rg["source_trial_indices"], indices)
                        for ordinal, index in enumerate(indices):
                            expected_lfp = (index * 1000000 + region_index * 1000000000
                                + np.arange(15, 25)[:, None, None]
                                + (expected_positions + 1)[None, :, None] * 1000
                                + np.arange(9)[None, None, :] * 100)
                            np.testing.assert_array_equal(rg["lfp"][ordinal][:, expected_positions, :], expected_lfp)
                            units = rg["spikes"].shape[-1]
                            expected_counts = index * 100 + np.arange(15, 25)[:, None] + np.arange(units)[None, :] * 10
                            np.testing.assert_array_equal(rg["spikes"][ordinal], expected_counts)
                    self.assertEqual(sorted(seen), list(range(spec.trials)))
            evaluation = next(row for row in plan if row["role"] == "evaluation")
            guide = report["guide_catalog"]["M1_lfp_guide"][0]
            selected_id = float(selected_by_region["M1"][0] + 1)
            column = guide.index([selected_id, *export.BANDS[0]])
            location = f"trial_data/trial_{evaluation['source_trial_index_zero_based']:06d}"
            with h5py.File(source_path, "r+") as handle:
                handle[location + "/M1_lfp"][column, 17] = np.nan
            failed = Path(temporary) / "failed.h5"
            with h5py.File(source_path, "r") as handle, h5py.File(failed, "w") as output:
                with self.assertRaisesRegex(ValueError, "no trial drop or redraw"):
                    export._export_session(handle, spec, report, plan, output, source, np, h5py, check)
                np.testing.assert_array_equal(output[spec.session_id + "/M1/selected_electrode_positions"],
                                              selected_by_region["M1"])
                self.assertFalse(output[spec.session_id].attrs["complete"])
            # A fractional native count must also fail without conversion or
            # silently removing that neuron/trial. Repair the fixture LFP only.
            index = evaluation["source_trial_index_zero_based"]
            with h5py.File(source_path, "r+") as handle:
                handle[location + "/M1_lfp"][column, 17] = index * 1000000 + selected_id * 1000 + 17
                handle[location + "/M1_spikes"][0, 17] = .5
            with h5py.File(source_path, "r") as handle, h5py.File(Path(temporary) / "bad_count.h5", "w") as output:
                with self.assertRaisesRegex(ValueError, "no count conversion or unit dropping"):
                    export._export_session(handle, spec, report, plan, output, source, np, h5py, check)


if __name__ == "__main__":
    unittest.main()
