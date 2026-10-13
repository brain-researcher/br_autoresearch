"""Focused half-cell summary checks; run only in task-owned compute."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_half_wave as summary
import test_mz_half_wave as fixtures


class HalfWavePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.HalfWaveTests.setUpClass()
        cls.fixture = fixtures.HalfWaveTests(
            "test_both_endpoints_preserve_seed_truth_four_fits_scoring_and_every_checkpoint_scope")
        cls.spec = cls.fixture.spec
        cls.templates = {}
        for name, options in (("positive", {}), ("below_margin", {"margin": .005}),
                              ("negative", {"margin": -.005}), ("technical", {"nonconverged": True})):
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "new", 1, **options)
            cls.templates[name] = report
        with tempfile.TemporaryDirectory() as directory:
            report, *_ = cls.fixture.fixture.exercise(Path(directory) / "new", "half", 0, margin=.005)
        report["job_id"] = str(summary.REPRESENTATIVE_JOB)
        cls.templates["representative"] = report

    def report(self, index, kind="below_margin"):
        result = deepcopy(self.templates["representative" if index == 0 and kind == "below_margin" else kind])
        result.update(replicate_index=index, selected_replicates=[index], seed=summary.levels.level_seed("half", index),
                      job_id=str(summary.REPRESENTATIVE_JOB if index == 0 else 72000000 + index))
        if index == 0 and kind != "below_margin":
            result["wave_selected"] = False
            for field in ("dispatch_scope", "externally_selected_index_range", "subsequent_wave_selected"):
                result.pop(field, None)
        return result

    def invoke(self, reports, selected=None, *, output=None):
        selected = sorted(reports) if selected is None else selected
        paths = {index: summary.report_path(index, summary.FIRST_ARRAY, summary.REMAINING_ARRAY) for index in selected}
        allowed = {paths[index]: json.dumps(report, allow_nan=True) for index, report in reports.items() if index in paths}
        allowed[summary.IMPLEMENTATION] = json.dumps(self.spec)
        before, reads = deepcopy(reports), []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"No unselected report, truth or events read: {path}")
            return allowed[path]

        with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
            if output is None:
                result = summary.pool(selected, summary.FIRST_ARRAY, summary.REMAINING_ARRAY, self.spec)
            else:
                code = summary.main(["--terminal-indices", ",".join(map(str, selected)),
                    "--first-array-job-id", str(summary.FIRST_ARRAY), "--remaining-array-job-id", str(summary.REMAINING_ARRAY),
                    "--implementation", str(summary.IMPLEMENTATION), "--output", str(output)])
                self.assertEqual(code, 0)
                result = None
        self.assertEqual(reports, before)
        return json.loads(output.read_text()) if output is not None else result, reads

    def assert_scope(self, result):
        self.assertEqual((result["phase"], result["effect_level"], result["SESOI_level"], result["generating_z"]),
                         (summary.PHASE, "half", .5, .475))
        self.assertEqual(result["focal_generating_truth"], summary.levels.focal_truth("half"))
        self.assertEqual(result["fixed_cohort_count"], 200)
        self.assertEqual(result["valid_components"] + result["unknown_cohorts"], 200)
        self.assertEqual(len(result["cohorts"]), 200)
        self.assertIs(result["recovery_90pct_criterion_applies"], False)
        self.assertIsNone(result["required_recovery_fraction"])
        self.assertIs(result["counts_as_curtailment_failure"], False)
        for field in summary.levels.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertEqual(result["full_menu_selection"], "not_evaluated")
        self.assertEqual(result["full_behavioral_gate"], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        for field in ("episode_terminal", "full_menu_wave_selected", "retry_selected", "automatic_retry"):
            self.assertIs(result[field], False)
        for field in ("required_necessary_screen_successes", "necessary_screen_successes", "valid_core_failures",
                      "best_possible_necessary_screen_successes", "necessary_90pct_component_count_met",
                      "necessary_90pct_component_unreachable", "recorded_richer_diagnostic_successes",
                      "recorded_richer_diagnostic_failures", "pass_rate"):
            self.assertNotIn(field, result)
        for row in result["cohorts"]:
            if row["status"] == "continuous_diagnostic_completed":
                self.assertEqual(set(row["continuous_diagnostics"]), {*summary.CONTINUOUS_FIELDS,
                    "recorded_participant_median_95CI", "recorded_bootstrap"})
                self.assertNotIn("necessary_nested_MZ_vs_M0_pass", row)
                self.assertNotIn("counts_as_curtailment_failure", row)

    def test_cli_reads_exact_half_representative_first_and_remaining_routes_only(self):
        reports = {index: self.report(index) for index in (0, 1, 199)}
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        paths = [summary.report_path(index, summary.FIRST_ARRAY, summary.REMAINING_ARRAY) for index in (0, 1, 199)]
        for path, fragment in zip(paths, ("20261005-MZ-effect-level-representatives-46603942/half/replicate0",
                "20261005-MZ-half-SESOI-wave-46607094/replicate1", "20261005-MZ-half-SESOI-wave-46607282/replicate199")):
            self.assertIn(fragment, str(path))
        self.assertEqual(set(reads), {*paths, summary.IMPLEMENTATION})
        self.assertEqual(result["completed_indices"], [0, 1, 199])
        self.assertEqual((result["valid_components"], result["unknown_cohorts"]), (3, 197))
        self.assert_scope(result)

    def test_positive_below_margin_and_negative_are_all_valid_continuous_diagnostics(self):
        reports = {1: self.report(1, "positive"), 2: self.report(2), 3: self.report(3, "negative")}
        result, _ = self.invoke(reports)
        self.assertEqual((result["valid_components"], result["unknown_cohorts"]), (3, 197))
        for index, report in reports.items():
            diagnostic = result["cohorts"][index]["continuous_diagnostics"]
            for field in summary.CONTINUOUS_FIELDS:
                self.assertEqual(diagnostic[field], report["comparison"][field])
            self.assertEqual(diagnostic["recorded_participant_median_95CI"], report["comparison"]["participant_median_95CI"])
        self.assertLess(result["cohorts"][3]["continuous_diagnostics"]["mean_gain_nats_per_trial"], 0)
        self.assert_scope(result)

    def test_all200_below_margin_do_not_invoke_any90pct_or21_failure_boundary(self):
        reports = {index: self.report(index) for index in range(200)}
        self.assertTrue(all(report["comparison"]["necessary_nested_MZ_vs_M0_pass"] is False for report in reports.values()))
        result, _ = self.invoke(reports)
        self.assertEqual((result["valid_components"], result["unknown_cohorts"]), (200, 0))
        self.assertEqual(result["unknown_indices"], [])
        self.assert_scope(result)

    def test_real_technical_prefix_and_missing_remain_unknown_with_cost_and_error(self):
        technical = self.report(1, "technical")
        result, reads = self.invoke({1: technical, 3: self.report(3)}, [1, 2, 3])
        self.assertEqual((result["valid_components"], result["unknown_cohorts"]), (1, 199))
        self.assertEqual(result["cohorts"][1]["reason"], "technical_component_failure")
        self.assertEqual(result["cohorts"][1]["elapsed_seconds"], technical["elapsed_seconds"])
        self.assertEqual(result["cohorts"][1]["error"], technical["error"])
        self.assertEqual(len(result["cohorts"][1]["returned_fits"]), 2)
        self.assertFalse(result["cohorts"][1]["returned_fits"][-1]["converged"])
        self.assertEqual(result["cohorts"][2]["reason"], "selected_terminal_report_missing")
        self.assertEqual(len(reads), 3)
        self.assert_scope(result)

    def test_wrong_level_seed_truth_wave_criterion_or_false_completed_claim_aborts(self):
        changes = (
            lambda report: report.update(phase="MZ_twice_SESOI_baseline_component"),
            lambda report: report.update(effect_level="twice", SESOI_level=2.),
            lambda report: report.update(seed=1900057 + 10007),
            lambda report: report.update(generating_z=.40),
            lambda report: report["focal_generating_truth"].update(rho=0.),
            lambda report: report.update(wave_selected=False),
            lambda report: report.update(subsequent_wave_selected=True),
            lambda report: report.update(recovery_90pct_criterion_applies=True),
            lambda report: report.update(required_recovery_fraction=.9),
            lambda report: report.update(counts_as_curtailment_failure=True),
            lambda report: report["comparison"].update(counts_as_curtailment_failure=True),
            lambda report: report["comparison"].update(recovery_90pct_criterion_applies=True),
            lambda report: report["comparison"].update(participant_SE=99.),
            lambda report: report["comparison"]["requirements"].update(mean_gain_at_least_0p01=True),
            lambda report: report.update(baseline_diagnostic_complete=False),
            lambda report: report["fits"][0].update(converged=False),
            lambda report: report["fits"][1].update(training_runs=[1, 2]),
            lambda report: report["participant_scores"]["p000"].update(MZ=float("nan")),
            lambda report: report["implementation"]["optimizer"].update(maximum_sweeps=1000),
            lambda report: report.update(full_gate="pass"),
        )
        for number, change in enumerate(changes):
            report = self.report(1)
            change(report)
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke({1: report}, [0, 1])  # Missing0 cannot conceal malformed completed1.
        representative = self.report(0)
        representative["wave_selected"] = True
        with self.assertRaises(ValueError):
            self.invoke({0: representative})
        technical = self.report(1, "technical")
        technical["participant_scores"] = {}
        with self.assertRaises(ValueError):
            self.invoke({1: technical})

    def test_exact_array_and_unique_terminal_selection_guards_precede_reads(self):
        for value in ("1,1", "0,200", "-1", "all", ""):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                summary.terminal_indices(value)
        self.assertEqual(summary.terminal_indices("199,0,1"), [0, 1, 199])
        for value in (True, 0, -1, "1,2"):
            with self.subTest(array=value), self.assertRaises(argparse.ArgumentTypeError):
                summary.array_id(value)
        with patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for indices, first, rest in (([1, 1], summary.FIRST_ARRAY, summary.REMAINING_ARRAY),
                                        ([1], 46606307, summary.REMAINING_ARRAY),
                                        ([2], summary.FIRST_ARRAY, 46606413),
                                        ([2], summary.FIRST_ARRAY, None)):
                with self.subTest(indices=indices, first=first, rest=rest), self.assertRaises(ValueError):
                    summary.pool(indices, first, rest, self.spec)

    def test_compute_fixed_implementation_and_fresh_output_guards_precede_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            args = ["--terminal-indices", "0", "--first-array-job-id", str(summary.FIRST_ARRAY),
                    "--remaining-array-job-id", str(summary.REMAINING_ARRAY),
                    "--implementation", str(summary.IMPLEMENTATION), "--output", str(output)]
            with patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
                with patch.object(summary, "require_compute", side_effect=RuntimeError("compute required")), \
                        self.assertRaisesRegex(RuntimeError, "compute required"):
                    summary.main(args)
                with patch.object(summary, "require_compute"), self.assertRaises(ValueError):
                    summary.main([*args[:7], str(Path(directory) / "alternate-spec.json"), *args[8:]])
                output.write_text("preserved")
                with patch.object(summary, "require_compute"), self.assertRaises(FileExistsError):
                    summary.main(args)
            self.assertEqual(output.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
