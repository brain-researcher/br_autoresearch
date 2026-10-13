"""Focused mixed500/1500 nonlinear admission checks; task-owned compute only."""
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

import summarize_mz_nonlinear_repaired as summary
import test_mz_nonlinear_m0_repair as fixtures


class NonlinearRepairedPoolTests(unittest.TestCase):
    JOB = 70012345

    @classmethod
    def setUpClass(cls):
        fixtures.NonlinearM0RepairTests.setUpClass()
        cls.fixture = fixtures.NonlinearM0RepairTests(
            "test_exactly_two_fresh_runtime1500_fits_follow_original_source_replay_with_cost_once")
        cls.spec = cls.fixture.spec
        cls.source, _ = cls.fixture.fixture.source()
        cls.failed = cls.fixture.failed
        cls.repair_dir = summary.repair.SOURCE.parent / f"20261005-MZ-nonlinear-M0-repair-{cls.JOB}"
        with tempfile.TemporaryDirectory() as directory:
            m0, *_ = cls.fixture.exercise(Path(directory) / "repair")
            mz, *_ = cls.fixture.fixture.exercise(Path(directory) / "original", "MZ")
        m0["job_id"], mz["job_id"] = str(cls.JOB), summary.repair.FAILED_JOB
        cls.templates = {"M0": m0, "MZ": mz}

    def reports(self):
        reports = deepcopy(self.templates)
        for index, person in enumerate(sorted(reports["M0"]["participant_scores"])):
            reports["M0"]["participant_scores"][person]["nonlinear_score"] = -3.
            reports["MZ"]["participant_scores"][person]["nonlinear_score"] = -3. + .02 + index * .0001
        return reports

    def bindings(self):
        core = summary.nonlinear
        return core.PHASE, core.MAX_SECONDS, core.validate_nonlinear_fit

    def invoke(self, reports, *, source=None, failed=None, output=None):
        paths = summary.report_paths(self.repair_dir, self.JOB)
        source = deepcopy(self.source) if source is None else source
        failed = deepcopy(self.failed) if failed is None else failed
        before = deepcopy((reports, source, failed, self.spec))
        bindings = self.bindings()
        allowed = {paths[model]: json.dumps(report, allow_nan=True) for model, report in reports.items()}
        allowed.update({summary.SOURCE_REPORT: json.dumps(source), summary.repair.FAILED_REPORT: json.dumps(failed),
                        summary.repair.IMPLEMENTATION: json.dumps(self.spec)})
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"No events/truth or unselected report read: {path}")
            return allowed[path]

        try:
            with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
                if output is None:
                    result = summary.pool(self.repair_dir, self.JOB, self.spec)
                else:
                    code = summary.main(["--repair-dir", str(self.repair_dir), "--repair-job-id", str(self.JOB),
                        "--implementation", str(summary.repair.IMPLEMENTATION), "--output", str(output)])
                    complete = len(reports) == 2 and all(report["nonlinear_component_complete"] for report in reports.values())
                    self.assertEqual(code, 0 if complete else 1)
                    result = None
        finally:
            self.assertEqual(self.bindings(), bindings)
            self.assertEqual((reports, source, failed, self.spec), before)
        return (json.loads(output.read_text()) if output is not None else result), reads

    def assert_scope(self, result):
        for field in summary.nonlinear.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertEqual(result["fixed_vs_nonlinear_arbitration"], "not_evaluated")
        self.assertNotIn("participant_scores", result)
        self.assertNotIn("counts_as_curtailment_failure", result)
        self.assertIn("curvature-adjusted reference", result["interpretation"])
        self.assertEqual(result["prior_M0_failure"]["previous_support_contacts"], self.failed["fits"][0]["support_contacts"])
        self.assertEqual(result["fixed_primary_participant_scores"], self.source["participant_scores"])

    def test_cli_mixes_actual1500_and500_with108_person_contrast_and_prior_contacts_cost_separate(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_models"], ["M0", "MZ"])
        self.assertEqual(result["unknown_models"], [])
        self.assertEqual((result["participants"], result["combined_heldout_trials_per_participant"]), (108, 256))
        self.assertEqual(result["original_implementation"], self.spec)
        self.assertEqual(result["model_fit_implementations"],
                         {"M0": summary.repair.runtime_specification(self.spec), "MZ": self.spec})
        self.assertEqual([row["phase"] for row in result["components"]], [summary.repair.PHASE, summary.nonlinear.PHASE])
        self.assertTrue(all(len(fit["objective_history"]) > 500 for fit in reports["M0"]["fits"]))
        self.assertEqual(result["components"][0]["cumulative_process_elapsed_seconds"],
                         self.failed["elapsed_seconds"] + reports["M0"]["elapsed_seconds"])
        self.assertEqual(result["components"][0]["current_fit_support_contacts"], [fit["support_contacts"] for fit in reports["M0"]["fits"]])
        differences = []
        for person, row in result["nonlinear_participant_scores"].items():
            self.assertEqual(row["M0"], -3.)
            self.assertEqual(row["MZ"], reports["MZ"]["participant_scores"][person]["nonlinear_score"])
            differences.append(row["MZ"] - row["M0"])
        diagnostic = result["nonlinear_nested_diagnostic"]
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences) / math.sqrt(108), places=14)
        self.assertEqual(set(reads), {*summary.report_paths(self.repair_dir, self.JOB).values(),
            summary.SOURCE_REPORT, summary.repair.FAILED_REPORT, summary.repair.IMPLEMENTATION})
        self.assert_scope(result)

    def test_missing_and_actual_repair_technical_prefixes_remain_unknown_without_partial_comparison(self):
        for missing in ("M0", "MZ"):
            reports = self.reports()
            del reports[missing]
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_models"], [missing])
            self.assertNotIn("nonlinear_nested_diagnostic", result)
            self.assert_scope(result)
        for options in ({"bad_source": True}, {"replay_error": True}, {"fail_fit": 1}, {"fail_fit": 2},
                        {"deadline_stage": "failure_read"}, {"deadline_stage": "fit"}, {"deadline_stage": "finalization"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "repair", **options)
            technical["job_id"] = str(self.JOB)
            reports = self.reports()
            reports["M0"] = technical
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_models"], ["M0"])
            self.assertEqual(result["components"][0]["status"], "technical_unknown")
            self.assertEqual(result["components"][0]["elapsed_seconds"], technical["elapsed_seconds"])
            self.assertNotIn("nonlinear_nested_diagnostic", result)
            self.assertNotIn("nonlinear_participant_scores", result)
            self.assert_scope(result)

    def test_malformed_repair_success_is_rejected_even_if_original_mz_missing_and_bindings_restore(self):
        mutations = (
            lambda r: r.update(phase=summary.nonlinear.PHASE),
            lambda r: r.update(job_id="70012346"),
            lambda r: r.update(source_finished_utc="wrong"),
            lambda r: r.update(failed_report="another_attempt"),
            lambda r: r["runtime_implementation"]["optimizer"].update(maximum_sweeps=500),
            lambda r: r["original_implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r.update(repair_context_validated=False),
            lambda r: r["numerical_repair"]["previous_support_contacts"].pop(),
            lambda r: r.update(cumulative_process_elapsed_seconds=r["cumulative_process_elapsed_seconds"] + 1),
            lambda r: r.update(fresh_fit_calls=1),
            lambda r: r["fits"][0].update(objective_history=[7000.] * 1501),
            lambda r: r["fits"][1].update(training_runs=[1, 2]),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r["fits"][0]["parameters"][0].__setitem__(-1, 1.50001),
            lambda r: r.update(scored_primary_directions=[]),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda r: r["participant_scores"]["p000"].update(nonlinear_score=float("nan")),
            lambda r: r.update(full_gate="pass"),
        )
        for number, mutate in enumerate(mutations):
            reports = {"M0": self.reports()["M0"]}
            mutate(reports["M0"])
            with self.subTest(mutation=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_original_mz_retains500_identity_and_technical_mz_never_becomes_a_comparison(self):
        mutations = (lambda r: r.update(job_id="46606083"), lambda r: r.update(fitted_model="M0"),
            lambda r: r["fits"][0].update(objective_history=[7000.] * 501),
            lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r.update(source_job_id="70000000"))
        for number, mutate in enumerate(mutations):
            reports = self.reports()
            mutate(reports["MZ"])
            with self.subTest(mutation=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)
        with tempfile.TemporaryDirectory() as directory:
            technical, *_ = self.fixture.fixture.exercise(Path(directory) / "original", "MZ", fail_second=True)
        technical["job_id"] = summary.repair.FAILED_JOB
        reports = self.reports()
        reports["MZ"] = technical
        result, _ = self.invoke(reports)
        self.assertEqual(result["unknown_models"], ["MZ"])
        self.assertNotIn("nonlinear_nested_diagnostic", result)
        self.assert_scope(result)

    def test_changed_original_source_or_failed_context_cannot_qualify_a_new_menu(self):
        for mutate in (lambda r: r.update(seed=r["seed"] + 1),
                       lambda r: r["fits"][0].update(converged=False)):
            source = deepcopy(self.source)
            mutate(source)
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(self.reports(), source=source)
        for mutate in (lambda r: r.update(finished_utc="2026-10-05T11:10:26.656204+00:00"),
                       lambda r: r.update(error="RuntimeError: deadline"),
                       lambda r: r["fits"][0].update(subject_optimizer_failures=[{}]),
                       lambda r: r["fits"][0]["support_contacts"].pop()):
            failed = deepcopy(self.failed)
            mutate(failed)
            with self.assertRaises(ValueError):
                self.invoke(self.reports(), failed=failed)

    def test_negative_nonlinear_gain_is_completed_diagnostic_with_all_claims_closed(self):
        reports = self.reports()
        for person, row in reports["MZ"]["participant_scores"].items():
            row["nonlinear_score"] = reports["M0"]["participant_scores"][person]["nonlinear_score"] - .03
        result, _ = self.invoke(reports)
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertAlmostEqual(result["nonlinear_nested_diagnostic"]["mean_gain_nats_per_trial"], -.03)
        self.assert_scope(result)

    def test_explicit_path_spec_and_fresh_output_guards_do_not_read_sources(self):
        with self.assertRaises(ValueError):
            summary.report_paths(self.repair_dir, self.JOB + 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No payload read")):
            args = ["--repair-dir", str(self.repair_dir), "--repair-job-id", str(self.JOB),
                    "--implementation", str(summary.repair.IMPLEMENTATION), "--output", str(Path(directory) / "fresh.json")]
            wrong = list(args)
            wrong[5] = str(Path(directory) / "another_spec.json")
            with self.assertRaises(ValueError):
                summary.main(wrong)
            args[-1] = directory
            with self.assertRaises(FileExistsError):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
