"""Focused complete-history/report-boundary tests; run only on compute."""
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from reference_s5_family import CONTACT_NAMES, STRUCTURES
from reference_s5_operator import nearest_summary
from reference_s5_prediction import prepare
from reference_s5_refinement import refinement_manifest
from reference_s5_centering import centering_manifest
from reference_s5_r2_centering import centering_manifest as r2_centering_manifest
from reference_s5_refinement_report import (
    CENTER, INITIAL, INTERIOR, R2_CENTER, _calibration_panels, build_result, compatible, distant_interpretation,
    freeze_history, main, require_active_centering, validate_frozen, write_report)


class Summary:
    def __init__(self, accepted=False):
        self.values = np.full(4096, 1e-6)
        self.values[31 * 64 + 31] = 1.
        self.values[[30 * 64 + 31, 32 * 64 + 31]] = 10 ** ((-12.6 if accepted else -2.) / 20.)
        self.values[[31 * 64 + 30, 31 * 64 + 32]] = 10 ** ((-37.2 if accepted else -397.) / 20.)

    def amplitudes(self, sites):
        return self.values[np.asarray(sites)]


class TestRefinementReport(unittest.TestCase):
    def test_distant_interpretation_distinguishes_ceiling_from_agreement_without_selecting_points(self):
        def entry(db):
            return dict(far_maximum=dict(status="finite", db=db), distant_check_passes=db <= -42.)
        cases = (([-50., -45.], "all_compatible_points_pass_ceiling"),
                 ([-45., -40.], "nearest_calibration_leaves_distant_check_unresolved"),
                 ([-40., -39.], "all_compatible_points_fail_ceiling"))
        for values, category in cases:
            entries = [entry(value) for value in values]
            original_entries = copy.deepcopy(entries)
            interval, text, meaning = distant_interpretation(entries)
            self.assertEqual(meaning["category"], category)
            self.assertEqual(interval["minimum"]["db"], min(values))
            self.assertEqual(interval["maximum"]["db"], max(values))
            self.assertTrue(interval["complete"])
            self.assertEqual(entries, original_entries)
            self.assertIn("dB", text)
        _, _, passing = distant_interpretation([entry(-200.), entry(-45.)])
        self.assertIn("does not establish agreement", passing["meaning"])
        zero = dict(far_maximum=dict(status="negative_infinity_zero_numerator", db=None),
                    distant_check_passes=True)
        interval, text, _ = distant_interpretation([zero, entry(-45.)])
        self.assertEqual(interval["minimum"]["status"], "negative_infinity_zero_numerator")
        self.assertEqual(interval["zero_amplitude_points"], 1)
        self.assertIn("zero fitted amplitude", text)
        undefined = dict(far_maximum=dict(status="undefined_zero_denominator", db=None),
                         distant_check_passes=False)
        interval, text, meaning = distant_interpretation([entry(-45.), undefined])
        self.assertFalse(interval["complete"])
        self.assertIn("unavailable", text)
        self.assertEqual(meaning["category"], "undefined_distant_values")

    def fixture(self, accepted=False):
        with TemporaryDirectory() as temporary:
            manifest = prepare(Path(temporary) / "manifest.json")
        summary = Summary(accepted)
        def loader(directory, bound_manifest, structure, point):
            binding = json.dumps(dict(manifest=bound_manifest, structure=structure,
                                      parameters=bound_manifest["points"][point]))
            return summary, 0, binding
        rows = []
        for structure in STRUCTURES:
            for point, parameters in enumerate(manifest["points"]):
                directory = f"/original/{structure}/point_{point:02d}"
                _, clips, binding = loader(directory, manifest, structure, point)
                nearest = nearest_summary(summary)
                rows.append(dict(structure=structure, point=point, parameters=parameters,
                    record_dir=directory, nearest=nearest, raw_clipped_count=clips, binding=binding,
                    named_templates=[f"{name}/{structure}" for name in CONTACT_NAMES],
                    compatible=bool(compatible(nearest, manifest["calibration"]))))
        initial = dict(status="fit_frozen_before_distant_evaluation", manifest=manifest,
            calibration=rows, compatible=[{key: row[key] for key in ("structure", "point", "record_dir")}
                                         for row in rows if row["compatible"]])
        return initial, refinement_manifest(), loader

    def test_complete_union_reuses_original_bindings_and_requires_every_record(self):
        initial, manifest, loader = self.fixture(accepted=True)
        calls = []
        def observed(*args):
            calls.append(args)
            return loader(*args)
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=observed)
        self.assertEqual(len(calls), 63)
        self.assertEqual(len(frozen["calibration"]), 63)
        self.assertEqual(len(frozen["compatible"]), 63)
        self.assertEqual(frozen["round_counts"], {INITIAL: 36, INTERIOR: 27})
        for old, new in zip(initial["calibration"], frozen["calibration"][:36]):
            self.assertEqual({key: value for key, value in new.items() if key != "round_id"}, old)
        validate_frozen(frozen)
        result, amplitudes = build_result(frozen, record_loader=loader,
                                          paper_far_db=-45., check_max_db=-42.)
        self.assertEqual(len(result["entries"]), 63)
        self.assertEqual(amplitudes.shape, (63, 4096))
        self.assertEqual({row["round_id"] for row in result["entries"]}, {INITIAL, INTERIOR})
        selected = copy.deepcopy(frozen)
        selected["compatible"].pop()
        with self.assertRaisesRegex(ValueError, "every accepted"):
            validate_frozen(selected)
        def incomplete(directory, *args):
            if str(directory).endswith("R3/point_08"):
                raise ValueError("incomplete full record")
            return loader(directory, *args)
        with self.assertRaisesRegex(ValueError, "incomplete full record"):
            freeze_history(initial, manifest, "/refinement", record_loader=incomplete)

    def test_missing_duplicate_or_changed_binding_stops_before_distant_access(self):
        initial, manifest, loader = self.fixture()
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=loader)
        def forbidden(*args):
            self.fail("invalid combined freeze opened a response record")
        missing = copy.deepcopy(frozen)
        missing["calibration"].pop()
        duplicate = copy.deepcopy(frozen)
        duplicate["calibration"][-1] = copy.deepcopy(duplicate["calibration"][-2])
        changed = copy.deepcopy(frozen)
        binding = json.loads(changed["calibration"][-1]["binding"])
        binding["manifest"] = initial["manifest"]
        changed["calibration"][-1]["binding"] = json.dumps(binding)
        for invalid in (missing, duplicate, changed):
            with self.subTest(invalid=invalid["calibration"][-1]["point"]):
                with self.assertRaises(ValueError):
                    build_result(invalid, record_loader=forbidden)

    def test_empty_union_keeps_far_closed_and_writes_three_lines_without_overwrite(self):
        initial, manifest, loader = self.fixture()
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=loader)
        def forbidden(*args):
            self.fail("empty compatible union opened a response record")
        class Unreadable:
            def __eq__(self, other):
                raise AssertionError("distant threshold was inspected")
            def __ne__(self, other):
                raise AssertionError("distant threshold was inspected")
        result, amplitudes = build_result(frozen, record_loader=forbidden,
            paper_far_db=Unreadable(), check_max_db=Unreadable())
        self.assertIsNone(amplitudes)
        self.assertFalse(result["distant_evidence_accessed"])
        self.assertNotIn("paper_far_db", result)
        self.assertEqual(result["grid_records_by_structure"], dict(R1=21, R2=21, R3=21))
        self.assertEqual(len(result["calibration"]), 63)
        with TemporaryDirectory() as temporary:
            write_report(temporary, result, amplitudes)
            self.assertEqual((Path(temporary) / "summary.md").read_text().splitlines(), result["interpretation"])
            self.assertEqual(len(result["interpretation"]), 3)
            self.assertTrue((Path(temporary) / "conditional_s5_refinement_prediction.png").is_file())
            with self.assertRaisesRegex(ValueError, "fresh and empty"):
                write_report(temporary, result, amplitudes)

    def test_centering_requires_complete_66_history_and_keeps_the_compatible_union(self):
        initial, manifest, loader = self.fixture(accepted=True)
        prior = freeze_history(initial, manifest, "/refinement", record_loader=loader)
        center = centering_manifest()
        calls = []
        def observed(*args):
            calls.append(args)
            return loader(*args)
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=observed,
            centering_manifest=center, centering_records_root="/centering")
        self.assertEqual(len(calls), 66)
        self.assertEqual(frozen["calibration"][:63], prior["calibration"])
        self.assertEqual(frozen["round_counts"], {INITIAL: 36, INTERIOR: 27, CENTER: 3})
        self.assertEqual(frozen["required_grid_records"], 66)
        result, amplitudes = build_result(frozen, record_loader=loader,
                                          paper_far_db=-45., check_max_db=-42.)
        self.assertEqual(amplitudes.shape, (66, 4096))
        self.assertEqual(result["grid_records_by_structure"], dict(R1=22, R2=22, R3=22))
        self.assertEqual(len(result["entries"]), 66)
        self.assertEqual({row["round_id"] for row in result["entries"]}, {INITIAL, INTERIOR, CENTER})
        for row in result["entries"]:
            self.assertEqual(row["named_templates"],
                             [f"{name}/{row['structure']}" for name in CONTACT_NAMES])
        self.assertIn("R1 nearest-only interpolation", result["interpretation"][0])
        selected = copy.deepcopy(frozen)
        selected["compatible"].pop()
        with self.assertRaisesRegex(ValueError, "every accepted"):
            validate_frozen(selected)
        def incomplete(directory, *args):
            if str(directory) == "/centering/R3/point_00":
                raise ValueError("incomplete full centering record")
            return loader(directory, *args)
        with self.assertRaisesRegex(ValueError, "incomplete full centering record"):
            freeze_history(initial, manifest, "/refinement", record_loader=incomplete,
                centering_manifest=center, centering_records_root="/centering")

    def test_active_cli_rejects_historical_63_and_66_before_distant_access(self):
        initial, manifest, loader = self.fixture(accepted=True)
        prior63 = freeze_history(initial, manifest, "/refinement", record_loader=loader)
        prior66 = freeze_history(initial, manifest, "/refinement", record_loader=loader,
            centering_manifest=centering_manifest(), centering_records_root="/centering")
        for historical in (prior63, prior66):
            with self.subTest(total=historical["required_grid_records"]), TemporaryDirectory() as output:
                with mock.patch("sys.argv", ["reporter", "report", "--freeze", "/historical.json",
                        "--output-dir", output, "--paper-far-db", "-45", "--check-max-db", "-42"]), \
                        mock.patch("fast_solver_continuous.compute_context"), \
                        mock.patch("reference_s5_refinement_report.load_json", return_value=historical), \
                        mock.patch("reference_s5_refinement_report.build_result") as distant:
                    with self.assertRaisesRegex(ValueError, "requires all 69"):
                        main()
                    distant.assert_not_called()

    def test_active_freeze_cli_requires_both_center002_inputs_before_record_access(self):
        arguments = ["reporter", "freeze", "--initial-freeze", "/initial.json", "--manifest",
            "/interior.json", "--records-root", "/interior", "--centering-manifest", "/center.json",
            "--centering-records-root", "/center", "--output", "/new.json"]
        for extra, missing in (([], "--r2-centering-manifest"),
                (["--r2-centering-manifest", "/r2.json"], "--r2-centering-records-root"),
                (["--r2-centering-records-root", "/r2"], "--r2-centering-manifest")):
            with self.subTest(extra=extra), mock.patch("sys.argv", arguments + extra), \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                    mock.patch("reference_s5_refinement_report.load_json") as records:
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(missing, stderr.getvalue())
                records.assert_not_called()

    def test_center002_requires_complete_69_and_retains_every_compatible_history_point(self):
        initial, manifest, loader = self.fixture(accepted=True)
        options = dict(centering_manifest=centering_manifest(), centering_records_root="/centering")
        prior = freeze_history(initial, manifest, "/refinement", record_loader=loader, **options)
        options.update(r2_centering_manifest=r2_centering_manifest(),
                       r2_centering_records_root="/r2_centering")
        calls = []
        def observed(*args):
            calls.append(args)
            return loader(*args)
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=observed, **options)
        self.assertEqual(len(calls), 69)
        self.assertEqual(frozen["calibration"][:66], prior["calibration"])
        self.assertEqual(frozen["required_grid_records"], 69)
        self.assertEqual(frozen["round_counts"], {INITIAL: 36, INTERIOR: 27, CENTER: 3, R2_CENTER: 3})
        self.assertIs(require_active_centering(frozen), frozen)
        result, amplitudes = build_result(frozen, record_loader=loader,
                                          paper_far_db=-45., check_max_db=-42.)
        self.assertEqual(amplitudes.shape, (69, 4096))
        self.assertEqual(result["grid_records_by_structure"], dict(R1=23, R2=23, R3=23))
        self.assertEqual(len(result["entries"]), 69)
        self.assertEqual({row["round_id"] for row in result["entries"]},
                         {INITIAL, INTERIOR, CENTER, R2_CENTER})
        for row in result["entries"]:
            self.assertEqual(row["named_templates"],
                             [f"{name}/{row['structure']}" for name in CONTACT_NAMES])
        self.assertIn("3 center001", result["interpretation"][0])
        self.assertIn("3 center002", result["interpretation"][0])
        self.assertIn("R2-guided nearest-only", result["interpretation"][0])
        selected = copy.deepcopy(frozen)
        selected["compatible"].pop(36)  # Old compatible records cannot be displaced by the final triplet.
        with self.assertRaisesRegex(ValueError, "every accepted"):
            validate_frozen(selected)
        def incomplete(directory, *args):
            if str(directory) == "/r2_centering/R3/point_00":
                raise ValueError("incomplete full center002 record")
            return loader(directory, *args)
        with self.assertRaisesRegex(ValueError, "incomplete full center002 record"):
            freeze_history(initial, manifest, "/refinement", record_loader=incomplete, **options)
        with TemporaryDirectory() as temporary:
            write_report(temporary, result, amplitudes)
            self.assertEqual(len((Path(temporary) / "summary.md").read_text().splitlines()), 3)
            with np.load(Path(temporary) / "per_pad_amplitudes.npz") as saved:
                self.assertEqual(saved["amplitudes_volts"].shape, (69, 4096))
                self.assertEqual(set(saved["round_ids"]), {INITIAL, INTERIOR, CENTER, R2_CENTER})
            self.assertTrue((Path(temporary) / "conditional_s5_refinement_prediction.png").is_file())

    def test_center002_missing_prior_or_record_rejected_before_distant_access(self):
        initial, manifest, loader = self.fixture(accepted=True)
        center2 = r2_centering_manifest()
        def forbidden(*args):
            self.fail("invalid center002 history opened a response record")
        for options in ({"r2_centering_manifest": center2},
                        {"r2_centering_records_root": "/r2_centering"}):
            with self.assertRaisesRegex(ValueError, "supplied together"):
                freeze_history(initial, manifest, "/refinement", record_loader=forbidden, **options)
        with self.assertRaisesRegex(ValueError, "requires the retained center001"):
            freeze_history(initial, manifest, "/refinement", record_loader=forbidden,
                r2_centering_manifest=center2, r2_centering_records_root="/r2_centering")
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=loader,
            centering_manifest=centering_manifest(), centering_records_root="/centering",
            r2_centering_manifest=center2, r2_centering_records_root="/r2_centering")
        missing = copy.deepcopy(frozen)
        missing["calibration"].pop()
        duplicate = copy.deepcopy(frozen)
        duplicate["calibration"][-1] = copy.deepcopy(duplicate["calibration"][-2])
        old_total = copy.deepcopy(frozen)
        old_total["required_grid_records"] = 66
        for name, invalid in (("missing", missing), ("duplicate", duplicate), ("total", old_total)):
            with self.subTest(case=name), self.assertRaises(ValueError):
                build_result(invalid, record_loader=forbidden, paper_far_db=-45., check_max_db=-42.)

    def test_centering_missing_duplicate_or_wrong_binding_stops_before_distant_access(self):
        initial, manifest, loader = self.fixture()
        center = centering_manifest()
        def forbidden(*args):
            self.fail("invalid centering input opened a response record")
        for options in ({"centering_manifest": center}, {"centering_records_root": "/centering"}):
            with self.assertRaisesRegex(ValueError, "supplied together"):
                freeze_history(initial, manifest, "/refinement", record_loader=forbidden, **options)
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=loader,
            centering_manifest=center, centering_records_root="/centering")
        missing = copy.deepcopy(frozen)
        missing["calibration"].pop()
        duplicate = copy.deepcopy(frozen)
        duplicate["calibration"][-1] = copy.deepcopy(duplicate["calibration"][-2])
        changed = copy.deepcopy(frozen)
        binding = json.loads(changed["calibration"][-1]["binding"])
        binding["manifest"] = manifest
        changed["calibration"][-1]["binding"] = json.dumps(binding)
        old_total = copy.deepcopy(frozen)
        old_total["required_grid_records"] = 63
        for name, invalid in (("missing", missing), ("duplicate", duplicate),
                              ("binding", changed), ("total", old_total)):
            with self.subTest(case=name), self.assertRaises(ValueError):
                build_result(invalid, record_loader=forbidden)

    def test_empty_69_union_keeps_far_closed_and_places_centers_after_history(self):
        initial, manifest, loader = self.fixture()
        frozen = freeze_history(initial, manifest, "/refinement", record_loader=loader,
            centering_manifest=centering_manifest(), centering_records_root="/centering",
            r2_centering_manifest=r2_centering_manifest(), r2_centering_records_root="/r2_centering")
        def forbidden(*args):
            self.fail("empty centered union opened a response record")
        class Unreadable:
            def __eq__(self, other):
                raise AssertionError("distant threshold was inspected")
            def __ne__(self, other):
                raise AssertionError("distant threshold was inspected")
        result, amplitudes = build_result(frozen, record_loader=forbidden,
            paper_far_db=Unreadable(), check_max_db=Unreadable())
        self.assertIsNone(amplitudes)
        self.assertFalse(result["distant_evidence_accessed"])
        self.assertNotIn("paper_far_db", result)
        self.assertEqual(result["status"], "no_compatible_point_in_complete69_point_exploration")
        self.assertEqual(len(result["calibration"]), 69)
        with TemporaryDirectory() as temporary:
            write_report(temporary, result, amplitudes)
            self.assertEqual(len((Path(temporary) / "summary.md").read_text().splitlines()), 3)
            stored = json.loads((Path(temporary) / "prediction.json").read_text())
            self.assertEqual(stored["grid_records"], 69)
            self.assertTrue((Path(temporary) / "conditional_s5_refinement_prediction.png").is_file())
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2)
        try:
            _calibration_panels(axes, result)
            for axis in axes:
                self.assertIn("interior points 12–20 | center point 21 | R2 center point 22",
                              axis.get_xlabel())
                self.assertIn("all 69 records", axis.get_title())
                self.assertEqual(len(axis.collections), 69)
                # The final point in each structural series is center002,
                # after all twelve initial, nine interior and one center001 points.
                for index, collection_index in enumerate((22, 45, 68)):
                    self.assertAlmostEqual(axis.collections[collection_index].get_offsets()[0, 0],
                                           22 + (index - 1) * .2)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
