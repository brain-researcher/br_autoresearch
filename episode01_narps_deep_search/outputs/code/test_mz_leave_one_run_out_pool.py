"""Focused four-report LOO pooling checks; execute only in task-owned compute."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_leave_one_run_out as summary
import test_mz_leave_one_run_out as fixtures


class LeaveOneRunOutPoolTests(unittest.TestCase):
    JOB = 70012345

    @classmethod
    def setUpClass(cls):
        fixtures.LeaveOneRunOutTests.setUpClass()
        cls.fixture = fixtures.LeaveOneRunOutTests(
            "test_all_four_complements_use_only192_training_rows_and64_heldout_rows_after_primary_replay")
        cls.spec = cls.fixture.spec
        cls.source_root = summary.loo.SOURCE.parent / f"20261005-MZ-LOO-{cls.JOB}"
        cls.templates = {}
        for held in summary.HELD_RUNS:
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "new", held)
            report["job_id"] = str(cls.JOB)
            cls.templates[held] = report

    def reports(self):
        reports = deepcopy(self.templates)
        # Different run means expose accidental sum/256 reuse or primary-score
        # mixing; participant variation exposes treating folds as432 subjects.
        multipliers = {1: 4., 2: 2., 3: 1., 4: -1.}
        for held, report in reports.items():
            for index, person in enumerate(sorted(report["participant_scores"])):
                row = report["participant_scores"][person]
                row.update(M0=-2. * held, MZ=-2. * held + multipliers[held] * (.02 + index * .0001))
            report["nested_diagnostic"] = summary.loo.fixed.nested_contrast(report["participant_scores"], "MZ", "M0")
        return reports

    def invoke(self, reports, *, output=None):
        paths = summary.report_paths(self.source_root, self.JOB)
        allowed = {paths[held]: json.dumps(report, allow_nan=True) for held, report in reports.items()}
        allowed[summary.loo.IMPLEMENTATION] = json.dumps(self.spec)
        before, reads = deepcopy(reports), []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"No original source or unselected payload read: {path}")
            return allowed[path]

        with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
            if output is None:
                result = summary.pool(self.source_root, self.JOB, self.spec)
            else:
                code = summary.main(["--source-root", str(self.source_root), "--job-id", str(self.JOB),
                                     "--implementation", str(summary.loo.IMPLEMENTATION), "--output", str(output)])
                self.assertEqual(code, 0 if len(reports) == 4 and all(report["loo_diagnostic_complete"] for report in reports.values()) else 1)
                result = None
        self.assertEqual(reports, before)
        return json.loads(output.read_text()) if output is not None else result, reads

    def assert_scope(self, result):
        self.assertEqual(result["selected_replicates"], [0])
        for field in summary.loo.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "automatic_retry", "LOO_runs_are_independent_replications", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertNotIn("counts_as_curtailment_failure", result)

    def test_cli_averages_four_run_means_within108_people_and_retains_descriptive_signs(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_held_out_runs"], [1, 2, 3, 4])
        self.assertEqual(result["unknown_held_out_runs"], [])
        self.assertEqual((result["participants"], result["combined_heldout_trials_per_participant"]), (108, 256))
        differences = []
        for person, pooled in result["participant_scores"].items():
            self.assertEqual(pooled["M0"], -5.)
            expected = sum(reports[held]["participant_scores"][person]["MZ"] for held in summary.HELD_RUNS) / 4
            self.assertEqual(pooled["MZ"], expected)
            differences.append(expected + 5.)
        diagnostic = result["pooled_nested_diagnostic"]
        self.assertEqual(diagnostic["participants"], 108)
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences) / math.sqrt(108), places=14)
        self.assertEqual(result["positive_run_mean_count"], 3)
        self.assertTrue(result["at_least_three_of_four_run_means_positive"])
        self.assertEqual([row["mean_advantage_sign"] for row in result["per_run"]], [1, 1, 1, -1])
        self.assertEqual(result["per_run"][-1]["version_advantage_signs"], {"EI": -1, "ER": -1})
        self.assertEqual(set(reads), {*summary.report_paths(self.source_root, self.JOB).values(), summary.loo.IMPLEMENTATION})
        self.assert_scope(result)

    def test_exact_zero_run_is_not_a_positive_sign_or_a_scientific_failure(self):
        reports = self.reports()
        for row in reports[3]["participant_scores"].values():
            row["MZ"] = row["M0"]
        reports[3]["nested_diagnostic"] = summary.loo.fixed.nested_contrast(reports[3]["participant_scores"], "MZ", "M0")
        result, _ = self.invoke(reports)
        self.assertEqual(result["positive_run_mean_count"], 2)
        self.assertFalse(result["at_least_three_of_four_run_means_positive"])
        self.assertEqual(result["per_run"][2]["mean_advantage_sign"], 0)
        self.assertEqual(result["per_run"][2]["version_advantage_signs"], {"EI": 0, "ER": 0})
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assert_scope(result)

    def test_missing_or_any_real_technical_prefix_prevents_partial_renormalization(self):
        reports = self.reports()
        del reports[4]
        result, _ = self.invoke(reports)
        self.assertEqual(result["unknown_held_out_runs"], [4])
        self.assertEqual(result["per_run"][3]["status"], "missing_unknown")
        self.assertNotIn("participant_scores", result)
        for options in ({"bad_source": True}, {"replay_error": True}, {"score_error": True},
                        {"fail_second": True}, {"deadline_stage": "finalization"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "new", 4, **options)
            technical["job_id"] = str(self.JOB)
            reports[4] = technical
            result, _ = self.invoke(reports)
            self.assertFalse(result["pooled_diagnostic_complete"])
            self.assertEqual(result["complete_held_out_runs"], [1, 2, 3])
            self.assertEqual(result["unknown_held_out_runs"], [4])
            self.assertEqual(result["per_run"][3]["status"], "technical_unknown")
            self.assertNotIn("participant_scores", result)
            self.assertNotIn("pooled_nested_diagnostic", result)
            self.assertNotIn("positive_run_mean_count", result)
            self.assert_scope(result)

    def test_duplicate_run_mixed_parent_source_or_primary_score_last_bits_are_rejected(self):
        base = self.reports()
        changes = (
            lambda reports: reports.__setitem__(4, deepcopy(reports[1])),
            lambda reports: reports[4].update(job_id="70012346"),
            lambda reports: reports[4].update(source_dir="different"),
            lambda reports: reports[4].update(source_job_id="70000000"),
            lambda reports: reports[4].update(source_finished_utc="2026-10-05T00:00:00+00:00"),
            lambda reports: reports[4]["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda reports: reports[4].update(saved_score_replay_max_absolute_discrepancy=2e-12),
        )
        for number, change in enumerate(changes):
            reports = deepcopy(base)
            change(reports)
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_false_complete_changed_fits_nonfinite_scores_and_claims_abort_even_with_another_missing_fold(self):
        base = self.reports()
        changes = (
            lambda reports: reports[1].update(loo_diagnostic_complete=False),
            lambda reports: reports[1].update(training_runs=[1, 2, 3]),
            lambda reports: reports[1].update(score_normalization="sum_divided_by256"),
            lambda reports: reports[1].update(heldout_trials_per_participant=128),
            lambda reports: reports[1].update(fits=reports[1]["fits"][:1]),
            lambda reports: reports[1]["fits"][0].update(converged=False),
            lambda reports: reports[1]["fits"][1].update(training_runs=[1, 2, 3]),
            lambda reports: reports[1]["fits"][0].update(objective_history=[1.] * 501),
            lambda reports: reports[1].update(scored_fits=[]),
            lambda reports: reports[1]["participant_scores"]["p000"].update(MZ=float("nan")),
            lambda reports: reports[1]["participant_scores"]["p000"].update(version="ER"),
            lambda reports: reports[1]["nested_diagnostic"].update(mean_gain_nats_per_trial=99.),
            lambda reports: reports[1]["implementation"]["optimizer"].update(maximum_sweeps=1000),
            lambda reports: reports[1].update(full_gate="pass"),
        )
        for number, change in enumerate(changes):
            reports = deepcopy(base)
            del reports[4]
            change(reports)
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_explicit_parent_paths_and_malformed_technical_reports_cannot_supply_a_pool(self):
        paths = summary.report_paths(self.source_root, self.JOB)
        self.assertEqual([path.parent.name for path in paths.values()], ["heldout1", "heldout2", "heldout3", "heldout4"])
        with self.assertRaises(ValueError):
            summary.report_paths(self.source_root, self.JOB + 1)
        for value in (True, 0, -1, "all", "70012345,70012346"):
            with self.subTest(job=value), self.assertRaises(argparse.ArgumentTypeError):
                summary.parent_job(value)
        with tempfile.TemporaryDirectory() as directory:
            technical, *_ = self.fixture.exercise(Path(directory) / "new", 4, fail_second=True)
        technical["job_id"] = str(self.JOB)
        for change in (lambda report: report.update(participant_scores={}),
                       lambda report: report.update(loo_diagnostic_complete=True),
                       lambda report: report.update(scored_fits=[{"model": model, "training_runs": [1, 2, 3], "scoring_runs": [4]}
                                                                for model in ("M0", "MZ")])):
            reports = self.reports()
            reports[4] = deepcopy(technical)
            change(reports[4])
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)


if __name__ == "__main__":
    unittest.main()
