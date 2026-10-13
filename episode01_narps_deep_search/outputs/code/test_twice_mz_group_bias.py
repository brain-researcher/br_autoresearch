"""Twice-cell group-bias checks using authored reports; coordinated compute only."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import run_twice_mz_group_bias as bias
import test_mz_twice_wave_pool as previous_tests


class TwiceMZGroupBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_tests.TwiceWavePoolTests.setUpClass()
        cls.previous = previous_tests.TwiceWavePoolTests(
            "test_cli_uses_exact_three_source_routes_and_never_reads_unselected_live_reports")
        cls.spec = cls.previous.spec

    def report(self, index, kind="pass", *, ei=.10, er=.10):
        report = self.previous.report(index, kind)
        # The legacy authored fixture omits these fields; the production fitter
        # explicitly returns False for both point-fit qualification claims.
        for fit in report["fits"]:
            fit["group_interval_coverage_qualified"] = False
            fit["individual_rank_recovery_qualified"] = False
        if kind != "technical":
            for fit in report["fits"]:
                if fit["model"] == "MZ":
                    column = fit["parameter_names"].index("rho")
                    for person, theta in zip(fit["participants"], fit["parameters"]):
                        effect = ei if report["participant_scores"][person]["version"] == "EI" else er
                        z = .5 - effect
                        theta[column] = math.log(z / (1. - z))
        return report

    def invoke(self, reports, *, output=None, deadline_stage=None):
        paths = {index: bias.twice.report_path(index, bias.twice.FIRST_ARRAY, bias.twice.REMAINING_ARRAY)
                 for index in range(200)}
        encoded = {paths[index]: json.dumps(report, allow_nan=True) for index, report in reports.items()}
        before, reads, now = deepcopy(reports), [], [100.]
        real_reader, real_writer, real_alarm = Path.read_text, bias.write_report, signal.alarm
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path == bias.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path not in paths.values():
                raise AssertionError(f"No source events/truth or other report may be read: {path}")
            if deadline_stage == "read" and path == paths[1]:
                now[0] += 241.
            if path not in encoded:
                raise FileNotFoundError(str(path))
            return encoded[path]

        def writer(path, result):
            real_writer(path, result)
            if deadline_stage == "publication" and result["group_point_bias_complete"]:
                now[0] += 241.

        def alarm(seconds):
            result = real_alarm(seconds)
            if seconds == 0 and deadline_stage == "cleanup":
                now[0] += 241.
            return result

        with ExitStack() as stack:
            for context in (patch.object(Path, "read_text", reader), patch.object(bias, "require_compute"),
                    patch.dict(os.environ, {key: "1" for key in bias.THREAD_ENV}),
                    patch.object(bias.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(bias, "write_report", side_effect=writer),
                    patch.object(bias.signal, "alarm", side_effect=alarm), redirect_stdout(io.StringIO())):
                stack.enter_context(context)
            if output is None:
                result = bias.pool(list(range(200)), bias.twice.FIRST_ARRAY, bias.twice.REMAINING_ARRAY, self.spec)
                exit_code = None
            else:
                exit_code = bias.main(["--terminal-indices", ",".join(map(str, range(200))),
                    "--first-array-job-id", str(bias.twice.FIRST_ARRAY),
                    "--remaining-array-job-id", str(bias.twice.REMAINING_ARRAY),
                    "--implementation", str(bias.IMPLEMENTATION), "--output", str(output), "--max-seconds", "240"])
                result = json.loads(real_reader(output))
        self.assertEqual(reports, before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        return result, reads, exit_code

    def assert_scope(self, result):
        self.assertEqual((result["phase"], result["effect_level"], result["SESOI_level"], result["generating_z"]),
                         (bias.PHASE, "twice", 2., .40))
        self.assertEqual((result["true_rejection_displacement"], result["relative_bias_limit"]), (.10, .10))
        self.assertEqual(result["fixed_cohort_count"], 200)
        self.assertEqual(result["valid_replicates"] + result["unknown_replicates"], 200)
        for field in ("group_recovery", "family_selection", "family_recovery_rate", "full_recovery_gate"):
            self.assertEqual(result[field], "not_evaluated")
        for field in ("interval_coverage_evaluated", "group_interval_coverage_qualified", "new_simulation",
                "new_model_fits", "bootstrap_performed", "automatic_retry", "observed_choices_or_RTs_used"):
            self.assertFalse(result[field])
        self.assertIsNone(result["full_recovery_pass"])

    def test_fixed_source_mapping_and_equal_person_half_physical_transform_without_relabeling(self):
        reports = {index: self.report(index) for index in (0, 1, 199)}
        for fit in reports[1]["fits"]:
            if fit["model"] == "MZ":
                column = fit["parameter_names"].index("rho")
                counters = {"EI": 0, "ER": 0}
                for person, theta in zip(fit["participants"], fit["parameters"]):
                    version = reports[1]["participant_scores"][person]["version"]
                    count = counters[version]
                    counters[version] += 1
                    z = (.2 if count % 2 == 0 else .6) if version == "EI" and fit["training_runs"] == [1, 4] else .4
                    theta[column] = math.log(z / (1-z))
                fit["global_mean"][column] = 1.
                fit["version_means"][0][column] = 1.
                fit["version_means"][1][column] = 1.
        with tempfile.TemporaryDirectory() as directory:
            result, reads, code = self.invoke(reports, output=Path(directory) / "bias.json")
        self.assertEqual(code, 0)
        self.assertEqual(len(reads), 201)
        self.assertIn("46603942/twice/replicate0", result["cohorts"][0]["report"])
        self.assertIn("wave-46606307/replicate1", result["cohorts"][1]["report"])
        self.assertIn("wave-46606413/replicate199", result["cohorts"][199]["report"])
        for version in ("EI", "ER"):
            self.assertAlmostEqual(result["cohorts"][1]["rejection_displacement"][version], .10, places=14)
            self.assertIsNone(result["version_statistics"][version]["relative_bias_pass"])
        self.assertEqual((result["valid_replicates"], result["unknown_replicates"]), (3, 197))
        self.assertTrue(result["statistics_provisional"])
        self.assert_scope(result)

    def test_all200_bias_uses_mean_signed_effect_and_point10_denominator_including_failed_screens(self):
        reports = {index: self.report(index, "fail" if index == 17 else "pass",
            ei=.08 if index % 2 == 0 else .12, er=.112) for index in range(200)}
        result, _, _ = self.invoke(reports)
        self.assertTrue(result["group_point_bias_complete"])
        self.assertEqual(result["valid_core_failures_included"], 1)
        self.assertEqual((result["valid_replicates"], result["unknown_replicates"]), (200, 0))
        self.assertAlmostEqual(result["version_statistics"]["EI"]["mean_rejection_displacement"], .10, places=14)
        self.assertAlmostEqual(result["version_statistics"]["EI"]["absolute_relative_bias"], 0., places=13)
        self.assertTrue(result["version_statistics"]["EI"]["relative_bias_pass"])
        self.assertAlmostEqual(result["version_statistics"]["ER"]["absolute_relative_bias"], .12, places=13)
        self.assertFalse(result["version_statistics"]["ER"]["relative_bias_pass"])
        self.assertFalse(result["both_versions_relative_bias_pass"])
        self.assert_scope(result)

    def test_missing_or_technical200th_remains_unknown_and_preserves_consumed_prefix(self):
        reports = {index: self.report(index) for index in range(199)}
        result, _, _ = self.invoke(reports)
        self.assertEqual((result["valid_replicates"], result["unknown_replicates"]), (199, 1))
        self.assertEqual(result["cohorts"][199]["reason"], "selected_terminal_report_missing")
        reports[199] = self.report(199, "technical")
        result, _, _ = self.invoke(reports)
        technical = result["cohorts"][199]
        self.assertEqual(technical["reason"], "technical_component_failure")
        self.assertEqual(technical["elapsed_seconds"], reports[199]["elapsed_seconds"])
        self.assertEqual(len(technical["returned_fits"]), 2)
        self.assertNotIn("rejection_displacement", technical)
        self.assertFalse(result["group_point_bias_complete"])
        self.assertIsNone(result["both_versions_relative_bias_pass"])
        self.assertIsNone(result["version_statistics"]["EI"]["relative_bias_pass"])
        self.assert_scope(result)

    def test_changed_identity_or_incomplete_success_aborts_and_no_other_index_set_is_selected(self):
        changes = (lambda r: r.update(seed=910031), lambda r: r.update(generating_z=.45),
            lambda r: r.update(phase="MZ_SESOI_fitted_recovery_component"),
            lambda r: r["fits"].pop(), lambda r: r["fits"][1].update(converged=False),
            lambda r: r["fits"][1]["parameters"].pop(),
            lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500))
        for mutation in changes:
            report = self.report(1)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                self.invoke({1: report})
        for field in ("group_interval_coverage_qualified", "individual_rank_recovery_qualified"):
            for fit_index in range(4):
                for missing in (False, True):
                    report = self.report(1)
                    if missing:
                        report["fits"][fit_index].pop(field)
                    else:
                        report["fits"][fit_index][field] = True
                    with self.subTest(field=field, fit_index=fit_index, missing=missing), self.assertRaisesRegex(
                            ValueError, "Original point-fit report cannot claim interval or individual qualification"):
                        self.invoke({1: report})
        with patch.object(Path, "read_text", side_effect=AssertionError("No input read before selection validation")):
            for indices, first, rest in ((list(range(199)), bias.twice.FIRST_ARRAY, bias.twice.REMAINING_ARRAY),
                    (list(range(199)) + [198], bias.twice.FIRST_ARRAY, bias.twice.REMAINING_ARRAY),
                    (list(range(200)), 1, bias.twice.REMAINING_ARRAY),
                    (list(range(200)), bias.twice.FIRST_ARRAY, 2)):
                with self.subTest(first=first, rest=rest, count=len(indices)), self.assertRaises(ValueError):
                    bias.pool(indices, first, rest, self.spec)

    def test_changed_version_roster_is_rejected_after_individually_valid_reports(self):
        report = self.report(1)
        people = report["participant_scores"]
        ei = next(person for person, row in people.items() if row["version"] == "EI")
        er = next(person for person, row in people.items() if row["version"] == "ER")
        people[ei]["version"], people[er]["version"] = "ER", "EI"
        # Score fixtures have identical per-person contrasts, so their version
        # summaries remain valid; only the cross-cohort identity rule should fail.
        with self.assertRaisesRegex(ValueError, "Participant identities or version assignments"):
            self.invoke({0: self.report(0), 1: report})

    def test_process_deadline_retains_partial_rows_and_clears_late_final_verdict(self):
        reports = {index: self.report(index) for index in range(200)}
        for stage, valid in (("read", 1), ("publication", 200), ("cleanup", 200)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                result, _, code = self.invoke(reports, output=Path(directory) / "bias.json", deadline_stage=stage)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "technical_group_bias_summary_failure")
            self.assertEqual(result["valid_replicates"], valid)
            self.assertFalse(result["group_point_bias_complete"])
            self.assertTrue(result["statistics_provisional"])
            self.assertIsNone(result["both_versions_relative_bias_pass"])
            self.assert_scope(result)


if __name__ == "__main__":
    unittest.main()
