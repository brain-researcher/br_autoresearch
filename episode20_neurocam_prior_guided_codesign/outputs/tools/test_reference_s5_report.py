"""Focused report-boundary checks; execute only in the coordinated allocation."""
import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from reference_s5_family import CONTACT_NAMES, STRUCTURES
from reference_s5_operator import nearest_summary
from reference_s5_prediction import prepare
from reference_s5_report import build_result, spatial_curves, write_report


class FakeSummary:
    def __init__(self, amplitudes):
        self.values = amplitudes

    def amplitudes(self, sites):
        return self.values[np.asarray(sites)]


class TestS5Report(unittest.TestCase):
    def frozen(self):
        with TemporaryDirectory() as temporary:
            manifest = prepare(Path(temporary) / "manifest.json")
        rows = []
        for structure in STRUCTURES:
            for point, parameters in enumerate(manifest["points"]):
                rows.append(dict(structure=structure, point=point,
                    parameters=parameters, record_dir=f"/unopened/{structure}/point_{point:02d}",
                    named_templates=[f"{name}/{structure}" for name in CONTACT_NAMES],
                    nearest=dict(driven_amplitude_volts=1.,
                        nearest_source=dict(db=-60., status="finite"),
                        nearest_gate=dict(db=-70., status="finite")),
                    compatible=False, raw_clipped_count=0,
                    binding=json.dumps(dict(manifest=manifest, structure=structure, parameters=parameters))))
        return dict(manifest=manifest, calibration=rows, compatible=[],
                    status="fit_frozen_before_distant_evaluation")

    def test_partial_or_selected_compatible_subset_stops_before_record_access(self):
        def forbidden(*args):
            self.fail("record opened before the full grid/frozen set was checked")
        partial = self.frozen()
        partial["calibration"].pop()
        with self.assertRaisesRegex(ValueError, "all 36"):
            build_result(partial, record_loader=forbidden)
        selected = self.frozen()
        selected["calibration"][0]["nearest"]["nearest_source"]["db"] = -12.6
        selected["calibration"][0]["nearest"]["nearest_gate"]["db"] = -37.2
        selected["calibration"][0]["compatible"] = True
        with self.assertRaisesRegex(ValueError, "complete nearest-only compatible set"):
            build_result(selected, record_loader=forbidden)

    def test_empty_sample_has_three_line_report_without_distant_access(self):
        def forbidden(*args):
            self.fail("empty calibration sample accessed a response record")
        class Unreadable:
            def __eq__(self, other):
                raise AssertionError("empty calibration sample inspected distant paper/check evidence")
            def __ne__(self, other):
                raise AssertionError("empty calibration sample inspected distant paper/check evidence")
        result, amplitudes = build_result(self.frozen(), record_loader=forbidden,
            paper_far_db=Unreadable(), check_max_db=Unreadable())
        self.assertIsNone(amplitudes)
        self.assertFalse(result["distant_evidence_accessed"])
        self.assertEqual(len(result["calibration"]), 36)
        self.assertEqual(result["grid_records_by_structure"], dict(R1=12, R2=12, R3=12))
        self.assertNotIn("paper_far_db", result)
        self.assertIn("does not reject", result["interpretation"][2])
        with TemporaryDirectory() as temporary:
            write_report(temporary, result, amplitudes)
            self.assertEqual((Path(temporary) / "summary.md").read_text().splitlines(), result["interpretation"])
            self.assertEqual(len(result["interpretation"]), 3)
            self.assertTrue((Path(temporary) / "conditional_s5_prediction.png").is_file())

    def test_off_axis_failure_is_visible_and_matches_the_held_aside_statistic(self):
        amplitude = np.full(4096, 1e-6)
        amplitude[31 * 64 + 31] = 1.
        amplitude[[30 * 64 + 31, 32 * 64 + 31]] = 10 ** (-12.6 / 20.)
        amplitude[[31 * 64 + 30, 31 * 64 + 32]] = 10 ** (-37.2 / 20.)
        amplitude[32 * 64 + 32] = .1  # off both plotted axis lines, Manhattan shell 2
        curves = spatial_curves(amplitude)
        self.assertEqual(len(curves["all_pad_manhattan_curve"]), 65)
        self.assertAlmostEqual(curves["all_pad_manhattan_curve"][2]["db"], -20.)
        self.assertAlmostEqual(curves["far_maximum"]["db"], -20.)
        self.assertLess(curves["source_curve"][2]["db"], -42.)
        self.assertLess(curves["gate_curve"][2]["db"], -42.)
        frozen = self.frozen()
        row = frozen["calibration"][0]
        summary = FakeSummary(amplitude)
        row["nearest"], row["compatible"] = nearest_summary(summary), True
        frozen["compatible"] = [{key: row[key] for key in ("structure", "point", "record_dir")}]
        def load(*args):
            return summary, 0, row["binding"]
        result, full = build_result(frozen, record_loader=load, paper_far_db=-45., check_max_db=-42.)
        self.assertFalse(result["entries"][0]["distant_check_passes"])
        self.assertEqual(full.shape, (1, 4096))
        with TemporaryDirectory() as temporary:
            write_report(temporary, result, full)
            self.assertEqual(len((Path(temporary) / "summary.md").read_text().splitlines()), 3)
        changed = copy.deepcopy(amplitude)
        changed[30 * 64 + 31] *= 2
        with self.assertRaisesRegex(ValueError, "differs from its frozen calibration"):
            build_result(frozen, paper_far_db=-45., check_max_db=-42.,
                         record_loader=lambda *args: (FakeSummary(changed), 0, row["binding"]))


if __name__ == "__main__":
    unittest.main()
