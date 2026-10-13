"""Focused indexed-null source, completion and unknown-count checks; use compute."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_null_baseline_wave as wave
import test_mz_null_rival as fixtures


FIRST_ARRAY = 46579924
REMAINING_ARRAY = 60000000


class NullBaselineWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def report(self, index=0, *, technical=False):
        report = deepcopy(self.fixture.source())
        report["nested_comparison"] = wave.fixed.nested_contrast(report["participant_scores"], "MZ", "M0")
        if index:
            report.update(phase=wave.indexed.PHASE, replicate_index=index, seed=wave.indexed.null_seed(index),
                          job_id=str(FIRST_ARRAY if index <= 4 else REMAINING_ARRAY),
                          wave_selected=True, wave_scope=wave.indexed.WAVE_SCOPE,
                          fixed_cohort_count=200, selected_replicates=[index],
                          full_menu_wave_selected=False, retry_selected=False)
        if technical:
            report.update(status="technical_representative_failure", pairwise_diagnostic_complete=False,
                          elapsed_seconds=20., simulation_seconds=2., scored_primary_directions=[],
                          error="RuntimeError: Nonconverged fit is technical; it is not a recovery failure")
            report["fits"] = report["fits"][:2]
            report["fits"][-1]["converged"] = False
            report["fits"][-1]["elapsed_seconds"] = 12.
            report.pop("participant_scores")
            report.pop("nested_comparison")
        return report

    def arguments(self, output, selected="0,1", *, remaining=None):
        args = ["--first-array-job-id", str(FIRST_ARRAY), "--terminal-indices", selected,
                "--implementation", str(wave.IMPLEMENTATION), "--output", str(output)]
        if remaining is not None:
            args.extend(("--remaining-array-job-id", str(remaining)))
        return args

    def invoke(self, arguments, reports, *, missing=()):
        allowed = {wave.IMPLEMENTATION: json.dumps(self.spec)}
        cohorts = {}
        for index, report in reports.items():
            path = wave.report_path(index, FIRST_ARRAY, REMAINING_ARRAY)
            allowed[path] = json.dumps(report)
            if report.get("status") == "pairwise_diagnostic_completed":
                allowed[path.parent / "generating_parameters.json"] = json.dumps(self.truth)
                cohorts[path.parent / "simulated_events.csv.gz"] = self.people
        absent = {wave.report_path(index, FIRST_ARRAY, REMAINING_ARRAY) for index in missing}
        reads, loads = [], []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in absent:
                raise FileNotFoundError(path)
            if path not in allowed:
                raise AssertionError(f"Unexpected source read: {path}")
            return allowed[path]

        def loader(path):
            loads.append(path)
            if path not in cohorts:
                raise AssertionError(f"Unexpected simulation read: {path}")
            return cohorts[path]

        with patch.object(wave, "require_compute"), patch.object(Path, "read_text", reader), \
                patch.object(wave.null.shared, "load_cohort", side_effect=loader), redirect_stdout(io.StringIO()):
            result = wave.main(arguments)
        return result, reads, loads, allowed, cohorts

    def test_exact_source_mapping_preserves_pilot_and_two_array_ranges(self):
        self.assertEqual(wave.report_path(0, FIRST_ARRAY, None), wave.indexed.PILOT_DIRECTORY / "mz_null_representative.json")
        for index in (1, 4, 5, 199):
            expected_array = FIRST_ARRAY if index <= 4 else REMAINING_ARRAY
            self.assertEqual(wave.report_path(index, FIRST_ARRAY, REMAINING_ARRAY),
                             wave.RUNS / f"20261005-M0-null-baseline-array-{expected_array}" /
                             f"replicate{index}/mz_null_replicate.json")
        with self.assertRaises(ValueError):
            wave.report_path(5, FIRST_ARRAY, None)
        with self.assertRaises(ValueError):
            wave.report_path(200, FIRST_ARRAY, REMAINING_ARRAY)

    def test_completed_technical_missing_and_unselected_remain_distinct_with_fixed200(self):
        reports = {0: self.report(), 1: self.report(1), 2: self.report(2, technical=True), 5: self.report(5)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            code, reads, loads, allowed, cohorts = self.invoke(
                self.arguments(output, "5,3,2,1,0", remaining=REMAINING_ARRAY), reports, missing=(3,))
            result = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(set(reads), set(allowed) | {wave.report_path(3, FIRST_ARRAY, REMAINING_ARRAY)})
        self.assertEqual(set(loads), set(cohorts))
        self.assertEqual(result["fixed_cohort_count"], 200)
        self.assertEqual(result["completed_indices"], [0, 1, 5])
        self.assertEqual(len(result["unknown_indices"]), 197)
        self.assertEqual(result["diagnostic_counts"], {"completed_pairwise_diagnostics": 3,
                         "unknown_cohorts": 197, "MZ_M0_nested_margin_qualifications": 3})
        self.assertEqual(result["cohorts"][2]["reason"], "technical_baseline_failure")
        self.assertEqual(result["cohorts"][2]["error"], reports[2]["error"])
        self.assertEqual(result["cohorts"][2]["process_elapsed_seconds"], 20.)
        self.assertEqual(result["cohorts"][2]["returned_fits"][1]["elapsed_seconds"], 12.)
        self.assertFalse(result["cohorts"][2]["returned_fits"][1]["converged"])
        self.assertNotIn("nested_comparison", result["cohorts"][2])
        self.assertEqual(result["cohorts"][3]["reason"], "selected_terminal_report_missing")
        self.assertEqual(result["cohorts"][4]["reason"], "not_selected_as_terminal_by_coordinator")
        for field in wave.null.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertEqual(result["early_kill"], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        self.assertFalse(result["retry_selected"])
        self.assertFalse(result["full_menu_wave_selected"])

    def test_pilot_and_indexed_source_validation_preserve_original_identity_without_mutation(self):
        original_validator = wave.null.validate_source
        with patch.object(wave.null, "validate_source", wraps=original_validator) as validate:
            for index in (0, 1, 199):
                report = self.report(index)
                before = deepcopy(report)
                diagnostic = wave.completed_diagnostic(report, self.truth, self.people, index, self.spec)
                self.assertEqual(diagnostic, report["nested_comparison"])
                self.assertEqual(report, before)
        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0]["replicate_index"], 0)
        self.assertEqual(validate.call_args.args[0]["phase"], "M0_null_MZ_pairwise_representative")

    def test_changed_identity_scope_or_claim_cannot_enter_as_success_or_unknown(self):
        changes = (("phase", "M0_null_MZ_pairwise_representative"), ("generating_model", "MZ"),
                   ("replicate_index", 2), ("seed", 1300037), ("implementation", {}),
                   ("design_source", "/different/design"), ("focal_generating_truth", {"eta": 0., "rho": .1, "criterion": 0.}),
                   ("wave_selected", False), ("wave_scope", "full_menu"), ("fixed_cohort_count", 199),
                   ("selected_replicates", [1, 2]), ("retry_selected", True), ("full_recovery_pass", True),
                   ("counts_as_curtailment_failure", False)) + tuple((field, "evaluated") for field in wave.null.CLAIM_FIELDS)
        for technical in (False, True):
            for field, value in changes:
                report = self.report(1, technical=technical)
                report[field] = value
                with self.subTest(technical=technical, field=field), self.assertRaises(ValueError):
                    wave.validate_identity(report, 1, self.spec)
        report = self.report(0)
        report["job_id"] = str(FIRST_ARRAY)
        with self.assertRaises(ValueError):
            wave.validate_identity(report, 0, self.spec)

    def test_malformed_success_missing_fits_wrong_truth_or_score_diagnostic_aborts(self):
        for field, value in (("fits", []), ("scored_primary_directions", []), ("participants", 107),
                             ("nested_comparison", {}), ("simulated_deadline_misses", 1)):
            report = self.report(1)
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                wave.completed_diagnostic(report, self.truth, self.people, 1, self.spec)
        report = self.report(1)
        report["fits"][0]["converged"] = False
        with self.assertRaises(RuntimeError):
            wave.completed_diagnostic(report, self.truth, self.people, 1, self.spec)
        report = self.report(1)
        report["fits"][0]["objective_history"] = [100.] * 501
        with self.assertRaises(ValueError):
            wave.completed_diagnostic(report, self.truth, self.people, 1, self.spec)
        truth = deepcopy(self.truth)
        truth["p000"]["rho"] = .05
        with self.assertRaises(ValueError):
            wave.completed_diagnostic(self.report(1), truth, self.people, 1, self.spec)
        report = self.report(1)
        report["participant_scores"]["p000"]["MZ"] += .1
        with self.assertRaises(ValueError):
            wave.completed_diagnostic(report, self.truth, self.people, 1, self.spec)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist.json"
            report = self.report(1)
            report["status"] = "running"
            with self.assertRaises(ValueError):
                self.invoke(self.arguments(output, "1"), {1: report})
            self.assertFalse(output.exists())

    def test_inconsistent_technical_report_is_not_silently_accepted(self):
        for field, value in (("error", ""), ("pairwise_diagnostic_complete", True), ("participant_scores", {}),
                             ("nested_comparison", {}), ("fits", None)):
            report = self.report(1, technical=True)
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                wave.technical_metadata(report)

    def test_guards_and_optional_remaining_array_prevent_unselected_source_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            with patch.object(wave, "require_compute"), \
                    patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
                with self.assertRaises(ValueError):
                    wave.main(self.arguments(output, "5"))
                changed = self.arguments(output)
                changed[5] = str(Path(directory) / "wrong-implementation.json")
                with self.assertRaises(ValueError):
                    wave.main(changed)
                output.write_text("preserved")
                with self.assertRaises(FileExistsError):
                    wave.main(self.arguments(output))
            fresh = Path(directory) / "first-only.json"
            self.invoke(self.arguments(fresh, "0,1"), {0: self.report(), 1: self.report(1)})
            result = json.loads(fresh.read_text())
            self.assertIsNone(result["source_arrays"]["indices5_through199"])
            self.assertEqual(result["completed_indices"], [0, 1])
        with redirect_stderr(io.StringIO()):
            for value in ("1,1", "200", "-1", ""):
                with self.subTest(indices=value), self.assertRaises(SystemExit):
                    wave.main(self.arguments("/unused/output.json", value))


if __name__ == "__main__":
    unittest.main()
