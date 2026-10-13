"""Small source-free fixtures for the fixed EP17 structural bin rule."""

from contextlib import contextmanager
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qualify_structural_support as stage


def table_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(stage.FIELDS)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


@contextmanager
def fixture(left=129, right=128, mutate=None, report_change=None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "input"
        source.mkdir()
        count = left + right + 1
        rows, participants = [], []
        counts = {subject: (count, left, right) for subject in stage.SUBJECTS}
        for subject in stage.SUBJECTS:
            for column in range(count):
                label, roi = (1007, "left") if column < left else (2007, "right") if column < left + right else (0, "")
                rows.append([subject, str(column), str(column), "0", str(column), "0",
                             "0", str(column), "0", str(label), roi])
            participants.append({"participant": subject, "original_model_columns": count,
                                 "axis_rows_written": count, "BCD_header_column_counts_agree": True,
                                 "mask_geometry": {"shape": [1, count, 1],
                                     "selected_affine_mm": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
                                 "ROIs": {"left": {"label": 1007, "original_axis_voxels": left, "refusal": None},
                                          "right": {"label": 2007, "original_axis_voxels": right, "refusal": None}}})
        if mutate:
            mutate(rows)
        payload = table_bytes(rows)
        (source / "voxel_axis.csv.gz").write_bytes(gzip.compress(payload, mtime=0))
        report = {"analysis_id": "anatomical_axis", "status": "completed",
                  "structural_voxel_axis_complete": True, "anatomical_fusiform_mapping_complete": True,
                  "errors": [], "refusals": [], "mask_column_rule": "original_union_boolean_C_order",
                  "modelmd_coordinate_rule": "modelmd[p,0,0,j]", "table_rows_written": count * 4,
                  "table_uncompressed_bytes_written": len(payload), "participants": participants}
        if report_change:
            report_change(report)
        (source / "anatomical_axis_report.json").write_text(json.dumps(report))
        with patch.object(stage, "INPUT", source), patch.object(stage, "EXPECTED_COUNTS", counts), \
             patch.object(stage, "EXPECTED_COLUMNS", count * 4):
            yield root, rows


class StructuralSupportTests(unittest.TestCase):
    def test_y_then_z_ties_and_earlier_remainder_are_order_invariant(self):
        rows = [stage.Row((), column, 0, -column) for column in range(19)]
        counts = stage.assign_bins(rows[::-1])
        self.assertEqual(counts, [3, 2, 3, 2, 3, 2, 2, 2])
        self.assertEqual([row.bin for row in rows],
                         [2, 2, 1, 1, 1, 4, 4, 3, 3, 3, 6, 6, 5, 5, 5, 8, 8, 7, 7])
        tied = [stage.Row((), column, 0, 0) for column in range(19)]
        stage.assign_bins(tied[::-1])
        self.assertEqual([row.bin for row in tied],
                         [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8])

    def test_global_minimum_floor_cap_and_refusal_boundaries(self):
        self.assertEqual(stage.select_quota([16] * 64), (8, 64))
        self.assertEqual(stage.select_quota([17] * 64), (8, 64))
        self.assertEqual(stage.select_quota([100] * 64), (32, 256))
        counts = [100] * 64
        counts[63] = 33
        self.assertEqual(stage.select_quota(counts), (16, 128))
        counts[63] = 15
        with self.assertRaisesRegex(stage.Refusal, "structural_quota_below_minimum"):
            stage.select_quota(counts)
        with self.assertRaisesRegex(stage.Refusal, "bin_count_contract_changed"):
            stage.select_quota([16] * 63)

    def test_full_pass_preserves_all_nominated_rows_and_original_columns(self):
        with fixture() as (root, original):
            report = stage.load_report()
            telemetry = {}
            rows, bins = stage.read_axis(report, time.monotonic() + 10, telemetry)
            self.assertEqual(telemetry["axis_rows_read"], 1032)
            self.assertEqual(telemetry["nominated_rows_retained"], 1028)
            self.assertEqual([row.fields for row in rows], [tuple(row) for row in original if row[10]])
            self.assertEqual(len(bins), 8)
            self.assertTrue(all(len(bins[(s, "left")]) == 129 and len(bins[(s, "right")]) == 128 for s in stage.SUBJECTS))

    def test_original_axis_identity_coordinate_label_and_truncation_refusals(self):
        cases = (
            (lambda rows: rows[2].__setitem__(1, "1"), "original_axis_identity_changed"),
            (lambda rows: rows[2].__setitem__(2, "1"), "original_axis_identity_changed"),
            (lambda rows: rows[2].__setitem__(0, "sub-02"), "axis_participant_order_changed"),
            (lambda rows: rows[2].__setitem__(6, "nan"), "nonfinite_world_coordinate"),
            (lambda rows: rows[2].__setitem__(7, "999"), "axis_world_coordinate_changed"),
            (lambda rows: rows[2].__setitem__(10, "right"), "axis_hemisphere_label_changed"),
            (lambda rows: rows.pop(), "incomplete_original_axis"),
        )
        for change, reason in cases:
            with self.subTest(reason=reason), fixture(mutate=change):
                report = stage.load_report()
                with self.assertRaisesRegex(stage.Refusal, reason):
                    stage.read_axis(report, time.monotonic() + 10, {})

    def test_completed_report_and_exact_recorded_counts_are_required(self):
        cases = (
            lambda report: report.update(status="stopped"),
            lambda report: report.update(structural_voxel_axis_complete=False),
            lambda report: report["participants"][0].update(original_model_columns=257),
            lambda report: report["participants"][0]["ROIs"]["left"].update(original_axis_voxels=128),
            lambda report: report.update(mask_column_rule="F_order"),
        )
        for change in cases:
            with self.subTest(change=change), fixture(report_change=change):
                with self.assertRaises(stage.Refusal):
                    stage.load_report()

    def test_successful_output_has_all_roi_rows_fixed_bins_and_no_neural_claim(self):
        with fixture() as (root, original):
            out = root / "output"
            out.mkdir()
            code = stage.run_attempt(out)
            report = json.loads((out / "structural_support_report.json").read_text())
            with gzip.open(out / "bin_columns.csv.gz", "rt", newline="") as stream:
                got = list(csv.reader(stream))
            self.assertEqual(code, 0)
            self.assertEqual((report["q_per_bin"], report["N_per_participant_ROI"]), (8, 64))
            self.assertTrue(report["structural_bins_complete"])
            self.assertEqual(got[0], [*stage.FIELDS, "spatial_bin"])
            self.assertEqual([row[:-1] for row in got[1:]], [row for row in original if row[10]])
            self.assertEqual({row[-1] for row in got[1:]}, {f"BIN{i:02d}" for i in range(1, 9)})
            self.assertEqual(report["source_axis_complete_passes"], 1)
            for key in ("neural_values_read", "source_components_pruned", "A_all_finite_support_complete",
                        "reliable_support_complete", "anatomical_bank_constructed", "audit_opened"):
                self.assertFalse(report[key])
            self.assertLessEqual(sum(p.stat().st_size for p in out.iterdir()), stage.OUTPUT_CAP)

    def test_infeasible_quota_preserves_all_rois_without_publishing_success(self):
        with fixture(left=120, right=128) as (root, original):
            out = root / "output"
            out.mkdir()
            self.assertEqual(stage.run_attempt(out), 2)
            report = json.loads((out / "structural_support_report.json").read_text())
            self.assertFalse(report["N_and_quotas_frozen"])
            self.assertFalse(report["structural_bins_complete"])
            self.assertEqual(len(report["participants"]), 4)
            self.assertIn("structural_quota_below_minimum", report["errors"])
            self.assertFalse((out / "bin_columns.csv.gz").exists())

    def test_output_bound_and_late_deadline_never_leave_completion_true(self):
        with patch.object(stage, "OUTPUT_CAP", stage.REPORT_CAP + 2):
            with self.assertRaisesRegex(stage.Refusal, "combined_output_byte_cap"):
                stage.encode_rows([], time.monotonic() + 10)
        def late(report, deadline):
            report.update(status="completed", structural_bins_complete=True, N_and_quotas_frozen=True)
            raise stage.Refusal("fixture_late_deadline")
        with fixture() as (root, original):
            out = root / "output"
            out.mkdir()
            with patch.object(stage, "finish", side_effect=late):
                self.assertEqual(stage.run_attempt(out), 2)
            report = json.loads((out / "structural_support_report.json").read_text())
            self.assertFalse(report["structural_bins_complete"])
            self.assertFalse(report["N_and_quotas_frozen"])
            self.assertEqual(report["errors"], ["fixture_late_deadline"])


if __name__ == "__main__":
    unittest.main()
