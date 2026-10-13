"""Focused complementary-support/sum-count pool checks; task-owned compute only."""
from __future__ import annotations

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

import summarize_mz_common_gain_range as summary
import test_mz_common_gain_range as fixtures


class CommonGainRangePoolTests(unittest.TestCase):
    JOB = 70012345

    @classmethod
    def setUpClass(cls):
        fixtures.CommonGainRangeTests.setUpClass()
        cls.fixture = fixtures.CommonGainRangeTests()
        cls.spec = cls.fixture.spec
        cls.source, _ = cls.fixture.source()
        cls.root = summary.common.SOURCE.parent / f"20261005-MZ-common-gain-range-{cls.JOB}"
        cls.templates = {}
        for direction in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "new", direction)
            report["job_id"] = str(cls.JOB)
            cls.templates[direction] = report

    def reports(self):
        reports = deepcopy(self.templates)
        for direction, report in reports.items():
            for index, person in enumerate(sorted(report["participant_scores"])):
                n = report["participant_scores"][person]["heldout_trials"]
                for model, mean in (("M0", -3. + 2 * direction),
                                    ("MZ", -3. + 2 * direction + .02 + .04 * direction + index * .0001)):
                    value = mean * n
                    report["participant_scores"][person][model + "_log_sum"] = value
                    report["completed_model_scores"][model][person]["log_sum"] = value
        return reports

    def invoke(self, reports, *, output=None):
        paths = summary.report_paths(self.root, self.JOB)
        allowed = {paths[direction]: json.dumps(report, allow_nan=True) for direction, report in reports.items()}
        allowed[summary.common.IMPLEMENTATION] = json.dumps(self.spec)
        before, reads = deepcopy((reports, self.spec)), []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"Only explicit reports/spec may be read: {path}")
            return allowed[path]

        try:
            with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
                if output is None:
                    result = summary.pool(self.root, self.JOB, self.spec)
                else:
                    code = summary.main(["--source-root", str(self.root), "--job-id", str(self.JOB),
                        "--implementation", str(summary.common.IMPLEMENTATION), "--output", str(output)])
                    complete = len(reports) == 2 and all(report["common_gain_component_complete"] for report in reports.values())
                    self.assertEqual(code, 0 if complete else 1)
                    result = None
        finally:
            self.assertEqual((reports, self.spec), before)
        return (json.loads(output.read_text()) if output is not None else result), reads

    def assert_scope(self, result):
        for field in summary.common.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "automatic_retry", "task_versions_are_independent_replications", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertNotIn("participant_scores", result)
        self.assertNotIn("counts_as_curtailment_failure", result)
        self.assertEqual(result["gain_range_dollars"], [10, 20])
        self.assertTrue(result["filter_boundary_inclusive"])
        self.assertEqual(result["fixed_primary_participant_scores"], self.source["participant_scores"])

    def test_cli_uses_logsum_totals_and_actual_counts_not_half_means_or256_and_equal_people(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_directions"], [0, 1])
        self.assertEqual(result["unknown_directions"], [])
        differences = []
        for person, row in result["common_gain_participant_scores"].items():
            count = sum(reports[d]["participant_scores"][person]["heldout_trials"] for d in (0, 1))
            self.assertEqual(count, 96 if row["version"] == "EI" else 176)
            self.assertEqual(result["retained_trials_per_participant"][person], count)
            for model in ("M0", "MZ"):
                total = sum(reports[d]["participant_scores"][person][model + "_log_sum"] for d in (0, 1))
                expected = total / count
                self.assertEqual(row[model], expected)
                self.assertNotEqual(row[model], total / 256)
            differences.append(row["MZ"] - row["M0"])
        person = "p000"
        n = [reports[d]["participant_scores"][person]["heldout_trials"] for d in (0, 1)]
        self.assertNotEqual(n[0], n[1])
        mean_of_means = sum(reports[d]["participant_scores"][person]["M0_log_sum"] / n[d] for d in (0, 1)) / 2
        self.assertNotEqual(result["common_gain_participant_scores"][person]["M0"], mean_of_means)
        diagnostic = result["common_gain_nested_diagnostic"]
        self.assertEqual(diagnostic["participants"], 108)
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences) / math.sqrt(108), places=14)
        self.assertEqual(set(reads), {*summary.report_paths(self.root, self.JOB).values(), summary.common.IMPLEMENTATION})
        self.assert_scope(result)

    def test_missing_or_real_technical_prefix_never_supplies_partial_direction_or_model_means(self):
        for missing in (0, 1):
            reports = self.reports()
            del reports[missing]
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_directions"], [missing])
            self.assertNotIn("common_gain_nested_diagnostic", result)
        for options in ({"bad_source": True}, {"replay_error": True}, {"fail_second": True}, {"score_error": True},
                        {"deadline_stage": "source"}, {"deadline_stage": "fit"},
                        {"deadline_stage": "scoring"}, {"deadline_stage": "finalization"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "new", 1, **options)
            technical["job_id"] = str(self.JOB)
            reports = self.reports()
            reports[1] = technical
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_directions"], [1])
            self.assertEqual(result["components"][1]["status"], "technical_unknown")
            self.assertEqual(result["components"][1]["completed_model_scores"], technical["completed_model_scores"])
            self.assertNotIn("common_gain_nested_diagnostic", result)
            self.assertNotIn("common_gain_participant_scores", result)
            self.assert_scope(result)

    def test_duplicate_leaking_and_noncomplementary_trial_slots_cannot_qualify(self):
        changes = (lambda r: r[0]["retained_support"]["p000"]["heldout_slots"][0].__setitem__(0, 1),
            lambda r: r[0]["retained_support"]["p000"]["heldout_slots"].__setitem__(1, r[0]["retained_support"]["p000"]["heldout_slots"][0]),
            lambda r: r[0]["retained_support"]["p000"].update(training_trials=128),
            lambda r: r[0]["retained_support"]["p000"].update(full_session_trials=256))
        for change in changes:
            reports = self.reports()
            change(reports)
            with self.assertRaises(ValueError):
                self.invoke(reports)
        reports = self.reports()
        support = reports[0]["retained_support"]["p000"]
        current = {tuple(slot) for slot in support["heldout_slots"]}
        replacement = next((run, trial) for run in (2, 3) for trial in range(1, 65) if (run, trial) not in current)
        support["heldout_slots"][0] = list(replacement)
        # The single direction still has legal nonduplicate run slots/counts;
        # its unchanged partner exposes the selected-trial mismatch.
        with self.assertRaisesRegex(ValueError, "complementary"):
            self.invoke(reports)

    def test_malformed_completed_fit_score_normalization_or_claim_aborts_when_partner_missing(self):
        changes = (
            lambda r: r.update(common_gain_component_complete=False),
            lambda r: r.update(score_normalization="divide_by256"),
            lambda r: r.update(gain_range_dollars=[10, 18]),
            lambda r: r.update(filter_boundary_inclusive=False),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r["fits"][1].update(training_runs=[1, 2]),
            lambda r: r["fits"][0].update(objective_history=[7000.] * 501),
            lambda r: r.update(scored_fits=[]),
            lambda r: r["participant_scores"]["p000"].update(heldout_trials=128),
            lambda r: r["participant_scores"]["p000"].update(M0_log_sum=float("nan")),
            lambda r: r["participant_scores"]["p000"].update(M0_log_sum=-123.),
            lambda r: r["completed_model_scores"]["M0"]["p000"].update(trials=256),
            lambda r: r["participant_scores"]["p000"].update(version="ER"),
            lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r.update(saved_primary_fits_reused_for_common_gain=True),
            lambda r: r.update(full_gate="pass"),
        )
        for number, change in enumerate(changes):
            reports = {0: self.reports()[0]}
            change(reports[0])
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_exact_parent_direction_source_completion_and_primary_last_bits_are_required(self):
        changes = (lambda r: r.update(job_id="70012346"), lambda r: r.update(direction=0),
            lambda r: r.update(source_dir="different"), lambda r: r.update(source_job_id="70000000"),
            lambda r: r.update(source_finished_utc="different"),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=2e-12))
        for change in changes:
            reports = self.reports()
            change(reports[1])
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_negative_gain_remains_a_completed_sensitivity_without_a_gate_verdict(self):
        reports = self.reports()
        for report in reports.values():
            for person, row in report["participant_scores"].items():
                value = row["M0_log_sum"] - .03 * row["heldout_trials"]
                row["MZ_log_sum"] = value
                report["completed_model_scores"]["MZ"][person]["log_sum"] = value
        result, _ = self.invoke(reports)
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertAlmostEqual(result["common_gain_nested_diagnostic"]["mean_gain_nats_per_trial"], -.03)
        self.assert_scope(result)

    def test_explicit_parent_spec_and_fresh_output_guards_precede_reads(self):
        with self.assertRaises(ValueError):
            summary.report_paths(self.root, self.JOB + 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No payload read")):
            args = ["--source-root", str(self.root), "--job-id", str(self.JOB),
                    "--implementation", str(summary.common.IMPLEMENTATION), "--output", str(Path(directory) / "fresh.json")]
            wrong = list(args)
            wrong[5] = str(Path(directory) / "another_spec.json")
            with self.assertRaises(ValueError):
                summary.main(wrong)
            args[-1] = directory
            with self.assertRaises(FileExistsError):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
