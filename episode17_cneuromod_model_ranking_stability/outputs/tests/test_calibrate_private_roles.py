"""Source-free calibration adapter checks; execute only in coordinated Slurm."""
from contextlib import ExitStack
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import calibrate_private_roles as method


def rows_fixture():
    rows = []
    for image in range(120):
        for repeat in range(2 + image % 2):
            row = {name: "" for name in method.original.MAP_FIELDS}
            row.update(participant="sub-01", role=method.ROLE, eligible_image="True", ordinal_status="valid",
                       expected_provider_image_path=f"image{image:03d}",
                       retained_acquisition_rank=str(10000 + image * 100 + repeat * 10),
                       retained_repeat_rank=str(repeat), repeat_half=str(1 + repeat % 2),
                       glmsingle_trial_index_zero_based=str(len(rows)))
            rows.append(row)
    # Storage order differs from chronology; the valid retained ranks do not.
    return list(reversed(rows))


def counters():
    return {"calibration_values_read": False, "private_read_seconds": 0., "numerical_seconds": 0.,
            "completed_physical_stripes": 0, "completed_decoded_bound_bytes": 0}


def private_inputs(base, width=513, nrows=300, bad_last=False):
    entries = {}
    base.mkdir(mode=0o700)
    role = base / method.ROLE
    role.mkdir(mode=0o700)
    for subject in method.support.PARTICIPANT_COLUMNS:
        directory = role / subject
        directory.mkdir(mode=0o700)
        for stage in method.STAGES:
            path = directory / f"{stage}.h5"
            with h5py.File(path, "w") as handle:
                invalid = bad_last and subject == "sub-06" and stage == "D"
                handle.create_dataset("modelmd", shape=(nrows, 1, 1, width), dtype="<f4",
                                      chunks=(min(1024, nrows), 1, 1, min(512, width)),
                                      compression="gzip" if invalid else None)
            path.chmod(0o600)
            entries[(subject, stage)] = {"rows": nrows, "shape": [nrows, 1, 1, width], "dtype_str": "<f4",
                                         "array_file_bytes": path.stat().st_size}
    return entries


class PrivateCalibrationTests(unittest.TestCase):
    def test_streamed_stripes_equal_complete_roster_core_without_finite_row_salvage(self):
        rows = rows_fixture()
        groups = method.image_groups(rows)
        values = np.zeros((len(rows), 1, 1, 513), dtype="<f4")
        for local, row in enumerate(rows):
            image = int(row["expected_provider_image_path"][5:])
            rank = int(row["retained_repeat_rank"])
            values[local, 0, 0, 0] = image + rank - 1
            values[local, 0, 0, 511] = 2 * image - rank
            values[local, 0, 0, 512] = image if rank % 2 else 0
        values[groups[10]["rows"][0], 0, 0, 511] = np.nan
        ids = (0, 511, 512)
        plan = {"shape": list(values.shape), "dtype_str": "<f4", "column_ids": list(ids),
                "stripes": method.stripe_plan(ids, values.shape)}
        reads = []
        class Reader:
            def __getitem__(self, key):
                reads.append(key)
                return values[key]
        report = counters()
        actual = method.calibrate_product(Reader(), plan, groups, report, lambda: None)
        expected = method.numerical.calibrate(
            [values[item["rows"], 0, 0][:, ids] for item in groups],
            [np.asarray(item["order"], dtype=np.int64) for item in groups])
        for name in method.VECTOR_FIELDS:
            np.testing.assert_array_equal(actual[name], getattr(expected, name))
        self.assertEqual([(item[3].start, item[3].stop) for item in reads], [(0, 512), (512, 513)])
        self.assertEqual(report["completed_physical_stripes"], 2)
        self.assertEqual(report["completed_decoded_bound_bytes"], 2 * len(rows) * 512 * 4)
        self.assertFalse(actual["response_eligible"][1])
        self.assertTrue(actual["response_eligible"][2])
        self.assertFalse(actual["reliability_valid"][2])

    def test_exact_private_map_and_retained_chronology_are_required_before_values(self):
        rows = rows_fixture()
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=method.original.MAP_FIELDS)
        writer.writeheader()
        for local, row in enumerate(rows):
            writer.writerow({**row, "local_row_index_zero_based": local})
        correct = stream.getvalue().encode()
        entries = {("sub-01", stage): {"rows": len(rows)} for stage in method.STAGES}
        with patch.object(method.copied, "load_full_plan", return_value=("never-open-this-mixed-root", {"sub-01": rows})), \
             patch.object(method.copied, "ordinary_private_file", return_value=len(correct)), \
             patch.object(method.original.stage, "bounded_read", return_value=correct), \
             patch.object(method.h5py, "File", side_effect=AssertionError("no payload in metadata admission")):
            groups, _ = method.metadata_plan(entries)
            self.assertEqual(len(groups["sub-01"]), 120)
            with patch.object(method.original.stage, "bounded_read", return_value=correct.replace(b"support_calibration", b"development", 1)):
                with self.assertRaisesRegex(method.core.Refusal, "private_calibration_index_map_changed"):
                    method.metadata_plan(entries)
        changed = copy.deepcopy(rows)
        changed[0]["repeat_half"] = "1" if changed[0]["repeat_half"] == "2" else "2"
        with self.assertRaisesRegex(method.core.Refusal, "calibration_repeat_chronology_changed"):
            method.image_groups(changed)
        missing = [row for row in rows if row["expected_provider_image_path"] != "image000"]
        with self.assertRaisesRegex(method.core.Refusal, "fixed_calibration_image_roster_too_small"):
            method.image_groups(missing)

    def test_support_intersects_stages_without_reliability_or_positive_ceiling_filter(self):
        columns = [method.support.StructuralColumn("sub-01", "left", ident, f"BIN{ident // 40 + 1:02d}", 1007)
                   for ident in range(320)]
        products = {}
        for stage in method.STAGES:
            products[("sub-01", stage)] = {"response_eligible": np.ones(320, dtype=bool),
                                           "reliability_valid": np.ones(320, dtype=bool),
                                           "pearson_r": np.linspace(.9, .1, 320),
                                           "ceiling_valid": np.ones(320, dtype=bool),
                                           "noise_ceiling": np.full(320, -.25)}
        products[("sub-01", "B")]["response_eligible"][0] = False
        products[("sub-01", "D")]["reliability_valid"][1] = False
        products[("sub-01", "D")]["ceiling_valid"][2] = False
        selected = {"sub-01": tuple(range(320))}
        groups = {("sub-01", "left"): columns}
        actual = method.construct_supports(groups, selected, products)
        prefix = "sub-01_left_"
        self.assertNotIn(0, actual[prefix + "A_all"])
        self.assertIn(1, actual[prefix + "A_all"])
        self.assertNotIn(1, actual[prefix + "R_N"])
        self.assertIn(1, actual[prefix + "V_NC"])
        self.assertNotIn(2, actual[prefix + "V_NC"])
        self.assertEqual(actual[prefix + "A_N_banks"].shape, (32, 256))
        baseline = actual[prefix + "A_N_banks"].copy()
        products[("sub-01", "D")]["pearson_r"] *= -1
        np.testing.assert_array_equal(method.construct_supports(groups, selected, products)[prefix + "A_N_banks"], baseline)
        products[("sub-01", "D")]["reliability_valid"][:10] = False
        with self.assertRaises(method.support.SupportError):
            method.construct_supports(groups, selected, products)

    def test_all_private_headers_precede_values_and_late_changed_layout_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "attempt002"
            entries = private_inputs(base, bad_last=True)
            selected = {subject: (0, 512) for subject in method.support.PARTICIPANT_COLUMNS}
            with patch.object(method, "COPY_PRIVATE", base), \
                 patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("preflight read values")):
                with self.assertRaisesRegex(method.core.Refusal, "private_output_representation_changed"):
                    method.preflight(entries, selected, lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "attempt002"
            entries = private_inputs(base)
            with patch.object(method, "COPY_PRIVATE", base), \
                 patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("preflight read values")):
                plans = method.preflight(entries, selected, lambda: None)
            self.assertEqual(len(plans), 12)
            self.assertEqual(sum(len(p["stripes"]) for p in plans), 24)
            self.assertEqual(sum(s["row_chunks"] for p in plans for s in p["stripes"]), 24)

    def test_completed_copy_and_matching_cost_pilot_are_mandatory(self):
        with self.assertRaisesRegex(method.core.Refusal, "completed_private_full_attempt002_required"):
            method.validate_copy({"status": "running", "full_copy_complete": False})
        with tempfile.TemporaryDirectory() as directory:
            private, public = Path(directory) / "private", Path(directory) / "public"
            private.mkdir(mode=0o700)
            public.mkdir()
            binding = {"pilot_column_ids": list(range(128)), "eligible_images": {"fixed": True}}
            for name, data in (("read_plan.json", json.dumps({"binding": binding}).encode()), ("calibration_maps.npz", b"fixture")):
                path = private / name
                path.write_bytes(data)
                path.chmod(0o600)
            report = {"status": "completed", "pilot_complete": True, "mode": "pilot", "errors": [],
                      "private_attempt": str(private), "full_copy_attempt": str(method.COPY_PRIVATE),
                      "selected_original_columns": 128, "planned_physical_stripes": 3,
                      "completed_physical_stripes": 3, "derived_arrays_verified": True,
                      "elapsed_seconds": 2., "peak_rss_kib": 500000, "private_read_seconds": .3, "numerical_seconds": .2}
            (public / "calibration_report.json").write_text(json.dumps(report))
            with patch.object(method, "paths", return_value=(private, public)):
                self.assertGreater(method.validate_cost_pilot(binding), 0)
                with self.assertRaisesRegex(method.core.Refusal, "calibration_cost_pilot_binding_changed"):
                    method.validate_cost_pilot({**binding, "pilot_column_ids": list(range(1, 129))})

    def test_private_readback_deadline_preserves_partial_and_public_failure_hides_values(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "maps.npz"
            budget = method.copied.OutputBudget(0, cap=4 * 1024**2)
            def deadline():
                raise method.core.Refusal("calibration_deadline")
            with self.assertRaisesRegex(method.core.Refusal, "calibration_deadline"):
                method.save_arrays(target, {"private_response": np.array([987654.321])}, budget, deadline)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_suffix(".npz.partial").exists())
            self.assertEqual(target.with_suffix(".npz.partial").stat().st_mode & 0o7777, 0o600)
            private, public = base / "private", base / "public"
            with patch.object(method, "paths", return_value=(private, public)), \
                 patch.object(method.original, "load_json", side_effect=ValueError("secret_response_987654.321")):
                self.assertEqual(method.run("pilot"), 2)
            payload = (public / "calibration_report.json").read_text()
            self.assertNotIn("987654", payload)
            self.assertNotIn("secret_response", payload)
            report = json.loads(payload)
            self.assertFalse(report["calibration_values_read"])
            self.assertFalse(report["reader_handoffs_activated"])
            self.assertFalse(report["candidate_scoring_authorized"])
            self.assertFalse(report["raw_audit_bytes_accessed"])
            self.assertEqual(report["scientific_audit_evaluation_count"], 0)
            self.assertEqual(report["errors"], [{"code": "private_calibration_processing_failure"}])


if __name__ == "__main__":
    unittest.main()
