"""Focused fixed200 twice-cell summary checks; task-owned compute only."""
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

import summarize_mz_twice_wave as summary
import test_mz_twice_wave as fixtures


class TwiceWavePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TwiceWaveTests.setUpClass()
        cls.fixture = fixtures.TwiceWaveTests(
            "test_both_endpoints_preserve_seed_truth_four_fits_scoring_and_every_checkpoint_scope")
        cls.spec = cls.fixture.spec
        cls.templates = {}
        for name, options in (("pass", {}), ("fail", {"margin": 0.}), ("technical", {"nonconverged": True})):
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "new", 1, **options)
            cls.templates[name] = report
        with tempfile.TemporaryDirectory() as directory:
            report, *_ = cls.fixture.fixture.exercise(Path(directory) / "new", "twice", 0)
        report["job_id"] = str(summary.REPRESENTATIVE_JOB)
        cls.templates["representative"] = report

    def report(self, index, kind="pass"):
        result = deepcopy(self.templates["representative" if index == 0 and kind == "pass" else kind])
        result.update(replicate_index=index, selected_replicates=[index], seed=summary.levels.level_seed("twice", index),
                      job_id=str(summary.REPRESENTATIVE_JOB if index == 0 else 71000000 + index))
        if index == 0 and kind != "pass":
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
                         (summary.PHASE, "twice", 2., .40))
        self.assertEqual(result["focal_generating_truth"], summary.levels.focal_truth("twice"))
        self.assertEqual((result["fixed_cohort_count"], result["required_necessary_screen_successes"]), (200, 180))
        for field in summary.levels.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertEqual(result["full_menu_selection"], "not_evaluated")
        self.assertEqual(result["full_behavioral_gate"], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        for field in ("episode_terminal", "full_menu_wave_selected", "retry_selected", "automatic_retry"):
            self.assertIs(result[field], False)
        self.assertEqual(result["valid_components"] + result["unknown_cohorts"], 200)

    def test_cli_uses_exact_three_source_routes_and_never_reads_unselected_live_reports(self):
        reports = {index: self.report(index) for index in (0, 1, 199)}
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        paths = [summary.report_path(index, summary.FIRST_ARRAY, summary.REMAINING_ARRAY) for index in (0, 1, 199)]
        self.assertIn("20261005-MZ-effect-level-representatives-46603942/twice/replicate0", str(paths[0]))
        self.assertIn("20261005-MZ-twice-SESOI-wave-46606307/replicate1", str(paths[1]))
        self.assertIn("20261005-MZ-twice-SESOI-wave-46606413/replicate199", str(paths[2]))
        self.assertEqual(set(reads), {*paths, summary.IMPLEMENTATION})
        self.assertEqual(result["completed_indices"], [0, 1, 199])
        self.assertEqual((result["necessary_screen_successes"], result["valid_core_failures"], result["unknown_cohorts"]), (3, 0, 197))
        self.assertFalse(result["necessary_90pct_component_count_met"])
        self.assertFalse(result["necessary_90pct_component_unreachable"])
        self.assert_scope(result)

    def test_twenty_vs_twenty_one_core_failures_define_unreachable_fixed180_bound(self):
        reports = {index: self.report(index, "fail") for index in range(1, 21)}
        reports[21] = self.report(21, "technical")
        result, _ = self.invoke(reports)
        self.assertEqual((result["valid_core_failures"], result["unknown_cohorts"], result["best_possible_necessary_screen_successes"]), (20, 180, 180))
        self.assertFalse(result["necessary_90pct_component_unreachable"])
        reports[21] = self.report(21, "fail")
        result, _ = self.invoke(reports)
        self.assertEqual((result["valid_core_failures"], result["unknown_cohorts"], result["best_possible_necessary_screen_successes"]), (21, 179, 179))
        self.assertTrue(result["necessary_90pct_component_unreachable"])
        self.assert_scope(result)

    def test_180_valid_necessary_successes_establish_only_component_count_with_unknown_remainder(self):
        reports = {index: self.report(index) for index in range(180)}
        result, _ = self.invoke(reports)
        self.assertEqual((result["necessary_screen_successes"], result["unknown_cohorts"]), (180, 20))
        self.assertTrue(result["necessary_90pct_component_count_met"])
        self.assertFalse(result["necessary_90pct_component_unreachable"])
        self.assert_scope(result)

    def test_richer_failure_alone_never_counts_as_a_core_failure(self):
        report = self.report(1)
        for row in report["participant_scores"].values():
            row["MZ"] = row["M0"] + (-.001 if row["version"] == "EI" else .061)
        comparison = summary.core.summarize_paired_scores(report["participant_scores"], "MZ", report["seed"] + 4099)
        necessary = comparison.pop("core_family_selection_pass")
        comparison.update(necessary_nested_MZ_vs_M0_pass=necessary, **summary.levels.criterion_scope("twice"),
            counts_as_curtailment_failure=not necessary,
            curtailment_rule="only_valid_at_or_above_SESOI_mean_0p01_and_strict_2SE_failures; no_automatic_curtailment",
            family_selection="not_evaluated", family_recovery_rate="not_evaluated")
        report.update(comparison=comparison, counts_as_curtailment_failure=not necessary)
        result, _ = self.invoke({1: report})
        self.assertEqual((result["necessary_screen_successes"], result["valid_core_failures"],
                          result["recorded_richer_diagnostic_failures"]), (1, 0, 1))
        self.assertFalse(result["cohorts"][1]["richer_diagnostics"]["requirements"]["positive_in_both_task_versions"])
        self.assert_scope(result)

    def test_missing_and_real_technical_prefix_stay_unknown_with_no_scientific_failure(self):
        technical = self.report(1, "technical")
        result, reads = self.invoke({1: technical}, [1, 2])
        self.assertEqual((result["valid_components"], result["unknown_cohorts"], result["valid_core_failures"]), (0, 200, 0))
        self.assertEqual(result["cohorts"][1]["reason"], "technical_component_failure")
        self.assertEqual(result["cohorts"][1]["elapsed_seconds"], technical["elapsed_seconds"])
        self.assertEqual(len(result["cohorts"][1]["returned_fits"]), 2)
        self.assertEqual(result["cohorts"][2]["reason"], "selected_terminal_report_missing")
        self.assertEqual(len(reads), 2)
        self.assert_scope(result)

    def test_wrong_level_seed_truth_wave_context_or_false_complete_is_rejected(self):
        changes = (
            lambda report: report.update(phase="MZ_SESOI_fitted_recovery_component"),
            lambda report: report.update(effect_level="half", SESOI_level=.5),
            lambda report: report.update(seed=910031 + 10007),
            lambda report: report.update(generating_z=.45),
            lambda report: report["focal_generating_truth"].update(rho=0.),
            lambda report: report.update(wave_selected=False),
            lambda report: report.update(subsequent_wave_selected=True),
            lambda report: report.update(baseline_diagnostic_complete=False),
            lambda report: report.update(counts_as_curtailment_failure=True),
            lambda report: report["comparison"]["requirements"].update(mean_gain_greater_than_two_participant_SE=False),
            lambda report: report["comparison"].update(participant_SE=99.),
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
                self.invoke({1: report}, [1, 2])
        representative = self.report(0)
        representative["wave_selected"] = True
        with self.assertRaises(ValueError):
            self.invoke({0: representative})
        technical = self.report(1, "technical")
        technical["participant_scores"] = {}
        with self.assertRaises(ValueError):
            self.invoke({1: technical})

    def test_duplicate_indices_or_alternate_arrays_cannot_choose_favorable_attempts(self):
        for value in ("1,1", "0,200", "-1", "all", ""):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                summary.terminal_indices(value)
        self.assertEqual(summary.terminal_indices("199,0,1"), [0, 1, 199])
        for value in (True, 0, -1, "1,2"):
            with self.subTest(array=value), self.assertRaises(argparse.ArgumentTypeError):
                summary.array_id(value)
        with patch.object(Path, "read_text", side_effect=AssertionError("No selected source read")):
            for indices, first, rest in (([1, 1], summary.FIRST_ARRAY, summary.REMAINING_ARRAY),
                                        ([1], 1, summary.REMAINING_ARRAY), ([2], summary.FIRST_ARRAY, 2),
                                        ([2], summary.FIRST_ARRAY, None)):
                with self.subTest(indices=indices, first=first, rest=rest), self.assertRaises(ValueError):
                    summary.pool(indices, first, rest, self.spec)


if __name__ == "__main__":
    unittest.main()
