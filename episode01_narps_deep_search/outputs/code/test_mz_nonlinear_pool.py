"""Focused two-report nonlinear comparison checks; task-owned compute only."""
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

import summarize_mz_nonlinear as summary
import test_mz_nonlinear as fixtures


class NonlinearPoolTests(unittest.TestCase):
    JOB = 70012345

    @classmethod
    def setUpClass(cls):
        fixtures.NonlinearTests.setUpClass()
        cls.fixture = fixtures.NonlinearTests(
            "test_each_model_replays_fixed_primary_then_fits_only_its_two_nonlinear_halves")
        cls.spec = cls.fixture.spec
        cls.source_root = summary.nonlinear.SOURCE.parent / f"20261005-MZ-nonlinear-{cls.JOB}"
        cls.templates = {}
        for model in summary.MODELS:
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "new", model)
            report["job_id"] = str(cls.JOB)
            cls.templates[model] = report

    def reports(self):
        reports = deepcopy(self.templates)
        # Distinct fixed/nonlinear magnitudes expose accidental score mixing;
        # participant variation exposes treating two halves as216 people.
        for index, person in enumerate(sorted(reports["M0"]["participant_scores"])):
            reports["M0"]["participant_scores"][person]["nonlinear_score"] = -3.
            reports["MZ"]["participant_scores"][person]["nonlinear_score"] = -3. + .02 + index * .0001
        return reports

    def invoke(self, reports, *, output=None):
        paths = summary.report_paths(self.source_root, self.JOB)
        allowed = {paths[model]: json.dumps(report, allow_nan=True) for model, report in reports.items()}
        allowed[summary.nonlinear.IMPLEMENTATION] = json.dumps(self.spec)
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
                                     "--implementation", str(summary.nonlinear.IMPLEMENTATION), "--output", str(output)])
                self.assertEqual(code, 0 if len(reports) == 2 and all(report["nonlinear_component_complete"]
                                                                    for report in reports.values()) else 1)
                result = None
        self.assertEqual(reports, before)
        return json.loads(output.read_text()) if output is not None else result, reads

    def assert_scope(self, result):
        self.assertEqual(result["selected_replicates"], [0])
        for field in summary.nonlinear.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertEqual(result["fixed_vs_nonlinear_arbitration"], "not_evaluated")
        self.assertNotIn("counts_as_curtailment_failure", result)
        self.assertNotIn("participant_scores", result)
        self.assertIn("curvature-adjusted reference", result["interpretation"])

    def test_cli_compares_only_nonlinear108_person_means_and_keeps_fixed_last_bits_separate(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_models"], ["M0", "MZ"])
        self.assertEqual(result["unknown_models"], [])
        self.assertEqual((result["participants"], result["combined_heldout_trials_per_participant"]), (108, 256))
        self.assertEqual(result["fixed_primary_participant_scores"], reports["M0"]["saved_primary_participant_scores"])
        differences = []
        for person, row in result["nonlinear_participant_scores"].items():
            self.assertEqual(row["M0"], -3.)
            self.assertEqual(row["MZ"], reports["MZ"]["participant_scores"][person]["nonlinear_score"])
            differences.append(row["MZ"] - row["M0"])
        diagnostic = result["nonlinear_nested_diagnostic"]
        self.assertEqual(diagnostic["participants"], 108)
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences) / math.sqrt(108), places=14)
        self.assertEqual(set(reads), {*summary.report_paths(self.source_root, self.JOB).values(), summary.nonlinear.IMPLEMENTATION})
        self.assert_scope(result)

    def test_zero_or_negative_nested_gain_remains_a_completed_diagnostic_without_gate_claim(self):
        for change in (0., -.03):
            reports = self.reports()
            for person, row in reports["MZ"]["participant_scores"].items():
                row["nonlinear_score"] = reports["M0"]["participant_scores"][person]["nonlinear_score"] + change
            result, _ = self.invoke(reports)
            self.assertTrue(result["pooled_diagnostic_complete"])
            self.assertAlmostEqual(result["nonlinear_nested_diagnostic"]["mean_gain_nats_per_trial"], change)
            self.assert_scope(result)

    def test_missing_or_real_technical_prefix_never_supplies_partial_comparison(self):
        reports = self.reports()
        del reports["MZ"]
        result, _ = self.invoke(reports)
        self.assertEqual(result["unknown_models"], ["MZ"])
        self.assertEqual(result["components"][1]["status"], "missing_unknown")
        self.assertNotIn("nonlinear_nested_diagnostic", result)
        for options in ({"bad_source": True}, {"replay_error": True}, {"score_error": True},
                        {"fail_second": True}, {"deadline_stage": "source"}, {"deadline_stage": "finalization"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "new", "MZ", **options)
            technical["job_id"] = str(self.JOB)
            reports["MZ"] = technical
            result, _ = self.invoke(reports)
            self.assertFalse(result["pooled_diagnostic_complete"])
            self.assertEqual(result["complete_models"], ["M0"])
            self.assertEqual(result["unknown_models"], ["MZ"])
            self.assertEqual(result["components"][1]["status"], "technical_unknown")
            self.assertEqual(result["components"][1]["elapsed_seconds"], technical["elapsed_seconds"])
            self.assertNotIn("nonlinear_participant_scores", result)
            self.assertNotIn("nonlinear_nested_diagnostic", result)
            self.assert_scope(result)

    def test_duplicate_model_mixed_parent_source_primary_last_bits_or_replay_are_rejected(self):
        base = self.reports()
        changes = (
            lambda reports: reports.__setitem__("MZ", deepcopy(reports["M0"])),
            lambda reports: reports["MZ"].update(job_id="70012346"),
            lambda reports: reports["MZ"].update(source_dir="different"),
            lambda reports: reports["MZ"].update(source_job_id="70000000"),
            lambda reports: reports["MZ"].update(source_finished_utc="2026-10-05T00:00:00+00:00"),
            lambda reports: reports["MZ"]["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda reports: reports["MZ"].update(saved_score_replay_max_absolute_discrepancy=2e-12),
        )
        for number, change in enumerate(changes):
            reports = deepcopy(base)
            change(reports)
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_malformed_completed_nonlinear_fits_scores_or_claims_abort_even_when_other_model_missing(self):
        base = self.reports()
        changes = (
            lambda reports: reports["M0"].update(nonlinear_component_complete=False),
            lambda reports: reports["M0"].update(score_normalization="sum_divided_by128"),
            lambda reports: reports["M0"].update(combined_scored_trials_per_participant=128),
            lambda reports: reports["M0"].update(fits=reports["M0"]["fits"][:1]),
            lambda reports: reports["M0"]["fits"][0].update(rival="fixed"),
            lambda reports: reports["M0"]["fits"][0].update(converged=False),
            lambda reports: reports["M0"]["fits"][1].update(training_runs=[1, 2]),
            lambda reports: reports["M0"]["fits"][0].update(parameter_names=["kappa", "log_a", "nondecision_logit"]),
            lambda reports: reports["M0"]["fits"][0]["parameters"][0].__setitem__(-1, 1.50001),
            lambda reports: reports["M0"]["fits"][0].update(objective_history=[1.] * 501),
            lambda reports: reports["M0"].update(scored_primary_directions=[]),
            lambda reports: reports["M0"]["participant_scores"]["p000"].update(nonlinear_score=float("nan")),
            lambda reports: reports["M0"]["participant_scores"]["p000"].update(version="ER"),
            lambda reports: reports["M0"]["implementation"]["optimizer"].update(maximum_sweeps=1000),
            lambda reports: reports["M0"].update(full_gate="pass"),
        )
        for number, change in enumerate(changes):
            reports = deepcopy(base)
            del reports["MZ"]
            change(reports)
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_explicit_parent_paths_and_malformed_technical_report_cannot_supply_a_comparison(self):
        self.assertEqual([path.parent.name for path in summary.report_paths(self.source_root, self.JOB).values()], ["M0", "MZ"])
        with self.assertRaises(ValueError):
            summary.report_paths(self.source_root, self.JOB + 1)
        for value in (True, 0, -1, "all", "70012345,70012346"):
            with self.subTest(job=value), self.assertRaises(argparse.ArgumentTypeError):
                summary.parent_job(value)
        with tempfile.TemporaryDirectory() as directory:
            technical, *_ = self.fixture.exercise(Path(directory) / "new", "MZ", fail_second=True)
        technical["job_id"] = str(self.JOB)
        for change in (lambda report: report.update(participant_scores={}),
                       lambda report: report.update(nonlinear_component_complete=True),
                       lambda report: report.update(scored_primary_directions=[
                           {"training_runs": list(train), "scoring_runs": list(held)} for train, held in summary.nonlinear.HALVES])):
            reports = self.reports()
            reports["MZ"] = deepcopy(technical)
            change(reports["MZ"])
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)


if __name__ == "__main__":
    unittest.main()
