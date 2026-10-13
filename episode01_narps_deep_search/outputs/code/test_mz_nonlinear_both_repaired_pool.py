"""Focused both-repair nonlinear pool checks; task-owned compute only."""
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

import summarize_mz_nonlinear_both_repaired as summary
import test_mz_nonlinear_m0_repair as m0_fixtures
import test_mz_nonlinear_mz_reverse_repair as mz_fixtures


class NonlinearBothRepairedPoolTests(unittest.TestCase):
    JOB = 70012346

    @classmethod
    def setUpClass(cls):
        m0_fixtures.NonlinearM0RepairTests.setUpClass()
        mz_fixtures.NonlinearMZReverseRepairTests.setUpClass()
        cls.m0 = m0_fixtures.NonlinearM0RepairTests()
        cls.mz = mz_fixtures.NonlinearMZReverseRepairTests()
        cls.spec = cls.m0.spec
        cls.source, _ = cls.m0.fixture.source()
        cls.failed = {"M0": cls.m0.failed, "MZ": cls.mz.failed}
        cls.repair_dir = summary.reverse.SOURCE.parent / f"20261005-MZ-nonlinear-MZ-reverse-repair-{cls.JOB}"
        with tempfile.TemporaryDirectory() as directory:
            m0, *_ = cls.m0.exercise(Path(directory) / "m0")
            mz, *_ = cls.mz.exercise(Path(directory) / "mz")
        m0["job_id"], mz["job_id"] = summary.M0_REPAIR_JOB, str(cls.JOB)
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
        before, bindings = deepcopy((reports, source, failed, self.spec)), self.bindings()
        allowed = {paths[model]: json.dumps(report, allow_nan=True) for model, report in reports.items()}
        allowed.update({summary.m0_pool.SOURCE_REPORT: json.dumps(source),
            summary.m0_pool.repair.FAILED_REPORT: json.dumps(failed["M0"]),
            summary.reverse.FAILED_REPORT: json.dumps(failed["MZ"]),
            summary.reverse.IMPLEMENTATION: json.dumps(self.spec)})
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"No events/truth or unselected report reads: {path}")
            return allowed[path]

        try:
            with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
                if output is None:
                    result = summary.pool(self.repair_dir, self.JOB, self.spec)
                else:
                    code = summary.main(["--repair-dir", str(self.repair_dir), "--repair-job-id", str(self.JOB),
                        "--implementation", str(summary.reverse.IMPLEMENTATION), "--output", str(output)])
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
        self.assertEqual(result["per_model_half_maximum_sweeps"], {"M0": [1500, 1500], "MZ": [500, 1500]})
        self.assertEqual(result["prior_failures"]["M0"]["previous_support_contacts"], self.failed["M0"]["fits"][0]["support_contacts"])
        self.assertEqual(result["prior_failures"]["MZ"]["previous_support_contacts"], [fit["support_contacts"] for fit in self.failed["MZ"]["fits"]])
        self.assertEqual(result["fixed_primary_participant_scores"], self.source["participant_scores"])
        self.assertIn("curvature-adjusted reference", result["interpretation"])

    def test_cli_compares_two_complete_repairs_with_mixed_half_settings_and_separate_cost_contacts(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_models"], ["M0", "MZ"])
        self.assertEqual(result["unknown_models"], [])
        self.assertEqual((result["participants"], result["combined_heldout_trials_per_participant"]), (108, 256))
        self.assertEqual([row["phase"] for row in result["components"]], [summary.m0_pool.repair.PHASE, summary.reverse.PHASE])
        for row in result["components"]:
            model = row["fitted_model"]
            self.assertEqual(row["cumulative_process_elapsed_seconds"],
                self.failed[model]["elapsed_seconds"] + reports[model]["elapsed_seconds"])
            self.assertEqual(row["current_fit_support_contacts"], [fit["support_contacts"] for fit in reports[model]["fits"]])
        self.assertEqual(result["components"][1]["per_half_provenance"], summary.reverse.per_half_provenance())
        differences = []
        for person, row in result["nonlinear_participant_scores"].items():
            self.assertEqual(row["M0"], -3.)
            self.assertEqual(row["MZ"], reports["MZ"]["participant_scores"][person]["nonlinear_score"])
            differences.append(row["MZ"] - row["M0"])
        diagnostic = result["nonlinear_nested_diagnostic"]
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences) / math.sqrt(108), places=14)
        self.assertEqual(set(reads), {*summary.report_paths(self.repair_dir, self.JOB).values(),
            summary.m0_pool.SOURCE_REPORT, summary.m0_pool.repair.FAILED_REPORT,
            summary.reverse.FAILED_REPORT, summary.reverse.IMPLEMENTATION})
        self.assert_scope(result)

    def test_missing_or_actual_mz_technical_prefix_never_produces_partial_comparison(self):
        for missing in ("M0", "MZ"):
            reports = self.reports()
            del reports[missing]
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_models"], [missing])
            self.assertNotIn("nonlinear_nested_diagnostic", result)
        for options in ({"bad_source": True}, {"replay_error": True}, {"rescore_error": True},
                        {"nonconverged": True}, {"deadline_stage": "failure_read"},
                        {"deadline_stage": "fit"}, {"deadline_stage": "finalization"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.mz.exercise(Path(directory) / "mz", **options)
            technical["job_id"] = str(self.JOB)
            reports = self.reports()
            reports["MZ"] = technical
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_models"], ["MZ"])
            self.assertEqual(result["components"][1]["status"], "technical_unknown")
            self.assertEqual(result["components"][1]["elapsed_seconds"], technical["elapsed_seconds"])
            self.assertNotIn("nonlinear_nested_diagnostic", result)
            self.assertNotIn("nonlinear_participant_scores", result)
            self.assert_scope(result)

    def test_mz_first_fit_contents_rescore_provenance_and_cost_are_required_even_if_m0_missing(self):
        changes = (
            lambda r: r.update(phase=summary.nonlinear.PHASE),
            lambda r: r.update(job_id="70012347"),
            lambda r: r.update(source_finished_utc="wrong"),
            lambda r: r["fits"][0]["parameters"][0].__setitem__(0, 1.5),
            lambda r: r["fits"][0].update(objective_history=[7000.] * 153),
            lambda r: r["fits"][0]["support_contacts"].append({"changed": True}),
            lambda r: r.update(first_half_copy_elapsed_seconds=r["first_half_copy_elapsed_seconds"] + 1),
            lambda r: r.update(first_half_rescored=False),
            lambda r: r.update(first_half_rescore_elapsed_seconds=-1),
            lambda r: r.update(fresh_fit_calls=0),
            lambda r: r.update(per_half_provenance=[]),
            lambda r: r.update(preserved_first_validation_maximum_sweeps=1500),
            lambda r: r.update(cumulative_process_elapsed_seconds=r["cumulative_process_elapsed_seconds"] + 1),
            lambda r: r["runtime_implementation"]["optimizer"].update(maximum_sweeps=500),
            lambda r: r["fits"][1].update(objective_history=[7000.] * 1501),
            lambda r: r["fits"][1].update(converged=False),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda r: r["participant_scores"]["p000"].update(nonlinear_score=float("nan")),
            lambda r: r.update(full_gate="pass"),
        )
        for number, change in enumerate(changes):
            reports = {"MZ": self.reports()["MZ"]}
            change(reports["MZ"])
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)

    def test_fixed_m0_repair_identity_and_frozen_validation_remain_required(self):
        for change in (lambda r: r.update(job_id="46611339"),
                       lambda r: r.update(phase=summary.nonlinear.PHASE),
                       lambda r: r.update(repair_context_validated=False),
                       lambda r: r["fits"][0].update(objective_history=[7000.] * 1501),
                       lambda r: r["numerical_repair"]["previous_support_contacts"].pop()):
            reports = self.reports()
            change(reports["M0"])
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)
        with tempfile.TemporaryDirectory() as directory:
            technical, *_ = self.m0.exercise(Path(directory) / "m0", fail_fit=2)
        technical["job_id"] = summary.M0_REPAIR_JOB
        reports = self.reports()
        reports["M0"] = technical
        result, _ = self.invoke(reports)
        self.assertEqual(result["unknown_models"], ["M0"])
        self.assertNotIn("nonlinear_nested_diagnostic", result)
        self.assert_scope(result)

    def test_exact_source_and_both_failed_attempts_must_reconstruct_before_admission(self):
        source = deepcopy(self.source)
        source["seed"] += 1
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke(self.reports(), source=source)
        for model in ("M0", "MZ"):
            for mutate in (lambda r: r.update(finished_utc="wrong"),
                           lambda r: r.update(error="RuntimeError: deadline"),
                           lambda r: r["fits"][-1]["support_contacts"].pop()):
                failed = deepcopy(self.failed)
                mutate(failed[model])
                with self.subTest(model=model), self.assertRaises((ValueError, RuntimeError)):
                    self.invoke(self.reports(), failed=failed)

    def test_negative_gain_remains_a_complete_descriptive_diagnostic_with_closed_claims(self):
        reports = self.reports()
        for person, row in reports["MZ"]["participant_scores"].items():
            row["nonlinear_score"] = reports["M0"]["participant_scores"][person]["nonlinear_score"] - .03
        result, _ = self.invoke(reports)
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertAlmostEqual(result["nonlinear_nested_diagnostic"]["mean_gain_nats_per_trial"], -.03)
        self.assert_scope(result)

    def test_explicit_job_path_spec_and_fresh_output_guards_precede_reads(self):
        with self.assertRaises(ValueError):
            summary.report_paths(self.repair_dir, self.JOB + 1)
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No payload read")):
            args = ["--repair-dir", str(self.repair_dir), "--repair-job-id", str(self.JOB),
                    "--implementation", str(summary.reverse.IMPLEMENTATION), "--output", str(Path(directory) / "fresh.json")]
            wrong = list(args)
            wrong[5] = str(Path(directory) / "another_spec.json")
            with self.assertRaises(ValueError):
                summary.main(wrong)
            args[-1] = directory
            with self.assertRaises(FileExistsError):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
