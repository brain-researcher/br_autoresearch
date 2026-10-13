"""Focused mixed nonlinear direction pooling checks; run only in Slurm."""
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

import summarize_mz_nonlinear_m0_directions_repaired as summary
import test_mz_nonlinear_m0_direction_repair as fixtures
import test_mz_nonlinear_mz_reverse_repair as mz_fixtures


class NonlinearM0DirectionsPoolTests(unittest.TestCase):
    JOB = 70012347

    @classmethod
    def setUpClass(cls):
        fixtures.NonlinearM0DirectionRepairTests.setUpClass()
        mz_fixtures.NonlinearMZReverseRepairTests.setUpClass()
        cls.fixture = fixtures.NonlinearM0DirectionRepairTests()
        cls.mz_fixture = mz_fixtures.NonlinearMZReverseRepairTests()
        cls.spec = cls.fixture.spec
        cls.source, _ = cls.fixture.source()
        cls.original_m0, cls.failed_m0 = cls.fixture.original_failed, cls.fixture.failed1500
        cls.original_mz = cls.mz_fixture.failed
        cls.root = summary.nonlinear.SOURCE.parent / f"20261005-MZ-nonlinear-M0-directions-repair-{cls.JOB}"
        cls.templates = {}
        for direction in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "new", direction=direction)
            report["job_id"] = str(cls.JOB)
            cls.templates[direction] = report
        with tempfile.TemporaryDirectory() as directory:
            cls.mz_report, *_ = cls.mz_fixture.exercise(Path(directory) / "mz")
        cls.mz_report["job_id"] = summary.MZ_JOB
        cls.mz_report["fits"][1]["objective_history"] = [7000.]*736

    def reports(self, *, negative=False):
        directions, mz = deepcopy(self.templates), deepcopy(self.mz_report)
        for index, person in enumerate(sorted(self.source["participant_scores"])):
            for direction, value in ((0, -1.8), (1, -.6)):
                directions[direction]["participant_scores"][person]["nonlinear_score"] = value
                directions[direction]["completed_direction_scores"][person]["nonlinear_score"] = value
            mz["participant_scores"][person]["nonlinear_score"] = -2.4 + (-.03 if negative else .02+index*.0001)
        return directions, mz

    def invoke(self, directions, mz, *, source=None, original_m0=None, failed_m0=None, original_mz=None, output=None):
        source = deepcopy(self.source) if source is None else source
        original_m0 = deepcopy(self.original_m0) if original_m0 is None else original_m0
        failed_m0 = deepcopy(self.failed_m0) if failed_m0 is None else failed_m0
        original_mz = deepcopy(self.original_mz) if original_mz is None else original_mz
        paths = summary.report_paths(self.root, self.JOB)
        allowed = {paths[d]: json.dumps(report, allow_nan=True) for d, report in directions.items()}
        allowed.update({summary.SOURCE_REPORT: json.dumps(source),
            summary.ORIGINAL_M0_REPORT: json.dumps(original_m0), summary.FAILED_M0_REPAIR_REPORT: json.dumps(failed_m0),
            summary.previous.reverse.FAILED_REPORT: json.dumps(original_mz), summary.MZ_REPORT: json.dumps(mz),
            summary.nonlinear.IMPLEMENTATION: json.dumps(self.spec)})
        before = deepcopy((directions, mz, source, original_m0, failed_m0, original_mz, self.spec))
        core = summary.nonlinear
        bindings = core.PHASE, core.MAX_SECONDS, core.validate_nonlinear_fit
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"No event/truth or unselected report read: {path}")
            return allowed[path]

        try:
            with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
                if output is None:
                    result = summary.pool(self.root, self.JOB, self.spec)
                else:
                    code = summary.main(["--repair-dir", str(self.root), "--repair-job-id", str(self.JOB),
                        "--implementation", str(core.IMPLEMENTATION), "--output", str(output)])
                    complete = len(directions) == 2 and all(r["nonlinear_direction_complete"] for r in directions.values())
                    self.assertEqual(code, 0 if complete else 1)
                    result = None
        finally:
            self.assertEqual((core.PHASE, core.MAX_SECONDS, core.validate_nonlinear_fit), bindings)
            self.assertEqual((directions, mz, source, original_m0, failed_m0, original_mz, self.spec), before)
        return (json.loads(output.read_text()) if output is not None else result), reads

    def assert_scope_and_cost(self, result, directions, mz):
        self.assertEqual(result["per_model_half_maximum_sweeps"], {"M0": [2000, 1500], "MZ": [500, 1500]})
        self.assertEqual(result["MZ_repair_job_id"], "46612046")
        self.assertEqual(result["fixed_primary_participant_scores"], self.source["participant_scores"])
        self.assertEqual(len(result["historical_processes"]), 4)
        self.assertEqual(len({row["report"] for row in result["historical_processes"]}), 4)
        expected = {str(path): report["elapsed_seconds"] for path, report in (
            (summary.ORIGINAL_M0_REPORT, self.original_m0), (summary.FAILED_M0_REPAIR_REPORT, self.failed_m0),
            (summary.previous.reverse.FAILED_REPORT, self.original_mz), (summary.MZ_REPORT, mz))}
        self.assertEqual({row["report"]: row["elapsed_seconds"] for row in result["historical_processes"]}, expected)
        self.assertEqual(result["historical_process_elapsed_seconds"], sum(expected.values()))
        self.assertEqual(result["new_M0_direction_process_elapsed_seconds"], sum(r["elapsed_seconds"] for r in directions.values()))
        self.assertNotIn("cumulative_process_elapsed_seconds", result)
        self.assertEqual(result["prior_failures"]["M0_1500"]["previous_support_contacts"],
                         self.failed_m0["fits"][0]["support_contacts"])
        self.assertEqual(result["prior_failures"]["M0_original500"]["previous_support_contacts"],
                         self.original_m0["fits"][0]["support_contacts"])
        for field in summary.nonlinear.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertEqual(result["fixed_vs_nonlinear_arbitration"], "not_evaluated")
        self.assertNotIn("counts_as_curtailment_failure", result)
        self.assertIn("curvature-adjusted reference", result["interpretation"])

    def test_complete_mixed_directions_sum256_contributions_and_count_each_process_once(self):
        directions, mz = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(directions, mz, output=Path(directory) / "summary.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_M0_directions"], [0, 1])
        self.assertEqual(result["unknown_M0_directions"], [])
        self.assertEqual((result["participants"], result["combined_heldout_trials_per_participant"]), (108, 256))
        self.assertEqual(result["M0_runtime_implementations"],
                         {str(d): summary.repair.runtime_specification(self.spec, d) for d in (0, 1)})
        differences = []
        for person, row in result["nonlinear_participant_scores"].items():
            expected = sum(directions[d]["participant_scores"][person]["nonlinear_score"] for d in (0, 1))
            self.assertEqual(row["M0"], expected)
            self.assertNotEqual(row["M0"], expected/2)
            self.assertNotEqual(row["M0"], expected/256)
            self.assertEqual(row["MZ"], mz["participant_scores"][person]["nonlinear_score"])
            differences.append(row["MZ"]-row["M0"])
        diagnostic = result["nonlinear_nested_diagnostic"]
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences)/math.sqrt(108), places=14)
        self.assertEqual(set(reads), {*summary.report_paths(self.root, self.JOB).values(), summary.SOURCE_REPORT,
            summary.ORIGINAL_M0_REPORT, summary.FAILED_M0_REPAIR_REPORT, summary.previous.reverse.FAILED_REPORT,
            summary.MZ_REPORT, summary.nonlinear.IMPLEMENTATION})
        self.assert_scope_and_cost(result, directions, mz)
        directions, mz = self.reports(negative=True)
        result, _ = self.invoke(directions, mz)
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertAlmostEqual(result["nonlinear_nested_diagnostic"]["mean_gain_nats_per_trial"], -.03)
        self.assert_scope_and_cost(result, directions, mz)

    def test_missing_or_actual_technical_direction_never_yields_partial_contrast(self):
        for missing in (0, 1):
            directions, mz = self.reports()
            del directions[missing]
            result, _ = self.invoke(directions, mz)
            self.assertEqual(result["unknown_M0_directions"], [missing])
            self.assertNotIn("nonlinear_nested_diagnostic", result)
            self.assertNotIn("nonlinear_participant_scores", result)
            self.assert_scope_and_cost(result, directions, mz)
        options = ({"bad_source": True}, {"replay_error": True},
            {"failed_change": lambda r: r.update(finished_utc="wrong")}, {"nonconverged": True},
            {"score_error": True}, {"parallel_error": True}, {"deadline_stage": "finalization"})
        saw_precontext_replay = False
        for index, option in enumerate(options):
            direction = index % 2
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "new", direction=direction, **option)
            technical["job_id"] = str(self.JOB)
            saw_precontext_replay |= technical["saved_primary_scores_replayed"] and not technical["repair_context_validated"]
            directions, mz = self.reports()
            directions[direction] = technical
            result, _ = self.invoke(directions, mz)
            self.assertFalse(result["pooled_diagnostic_complete"])
            self.assertEqual(result["unknown_M0_directions"], [direction])
            self.assertEqual(result["components"][direction]["status"], "technical_unknown")
            self.assertNotIn("nonlinear_nested_diagnostic", result)
            self.assertNotIn("nonlinear_participant_scores", result)
            self.assert_scope_and_cost(result, directions, mz)
        self.assertTrue(saw_precontext_replay)

    def test_malformed_direction_identity_runtime_support_scores_and_cost_rejected(self):
        changes = (lambda r: r.update(direction=1), lambda r: r.update(job_id="another"),
            lambda r: r.update(phase=summary.nonlinear.PHASE), lambda r: r.update(failed_report="wrong"),
            lambda r: r.update(fitted_maximum_sweeps=1500),
            lambda r: r["runtime_implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r.update(score_normalization=summary.nonlinear.NORMALIZATION),
            lambda r: r.update(heldout_trials_per_direction_per_participant=256),
            lambda r: r.update(direction_contribution_only=False), lambda r: r.update(workers=1),
            lambda r: r.update(parallel_workers_cleaned_up=False),
            lambda r: r["fits"][0].update(objective_history=[7000.]*2001),
            lambda r: r["fits"][0].update(training_runs=[2, 3]),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r.update(source_finished_utc="wrong"),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda r: r["prior_1500_failed_context"].update(previous_process_elapsed_seconds=1.),
            lambda r: r["original_500_failed_context"]["previous_support_contacts"].pop(),
            lambda r: r.update(cumulative_process_elapsed_seconds=1.),
            lambda r: r["completed_direction_scores"]["p000"].update(nonlinear_score=float("nan")),
            lambda r: r.update(participant_scores={**r["participant_scores"],
                "p000": {**r["participant_scores"]["p000"], "nonlinear_score": -999.}}),
            lambda r: r.update(full_gate="pass"))
        for number, change in enumerate(changes):
            directions, mz = self.reports()
            del directions[1]
            change(directions[0])
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(directions, mz)
        directions, mz = self.reports()
        directions[1]["fits"][0]["objective_history"] = [7000.]*1501
        with self.assertRaises(ValueError):
            self.invoke(directions, mz)

    def test_preserved_mz_first_fit_and_both_m0_failures_remain_mandatory(self):
        for change in (lambda r: r.update(job_id="46612047"),
            lambda r: r["fits"][0]["parameters"][0].__setitem__(0, 1.5),
            lambda r: r.update(preserved_first_validation_maximum_sweeps=1500),
            lambda r: r.update(first_half_rescored=False),
            lambda r: r.update(cumulative_process_elapsed_seconds=r["cumulative_process_elapsed_seconds"]+1),
            lambda r: r.update(source_finished_utc="wrong")):
            directions, mz = self.reports()
            change(mz)
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(directions, mz)
        for key, template in (("original_m0", self.original_m0), ("failed_m0", self.failed_m0),
                              ("original_mz", self.original_mz)):
            failed = deepcopy(template)
            failed["finished_utc"] = "wrong"
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(*self.reports(), **{key: failed})
        source = deepcopy(self.source)
        source["seed"] += 1
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke(*self.reports(), source=source)

    def test_exact_parent_original_spec_and_fresh_output_guards(self):
        with self.assertRaises(ValueError):
            summary.report_paths(self.root, self.JOB+1)
        with self.assertRaises(ValueError):
            summary.report_paths(summary.FAILED_M0_REPAIR_REPORT.parent, self.JOB)
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No report read")):
            args = ["--repair-dir", str(self.root), "--repair-job-id", str(self.JOB),
                    "--implementation", str(summary.nonlinear.IMPLEMENTATION), "--output", str(Path(directory) / "new.json")]
            changed = list(args)
            changed[5] = str(Path(directory) / "another.json")
            with self.assertRaises(ValueError):
                summary.main(changed)
            args[-1] = directory
            with self.assertRaises(FileExistsError):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
