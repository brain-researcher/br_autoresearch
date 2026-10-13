"""Focused four-component repair admission checks; task-owned compute only."""
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

import summarize_mz_common_gain_range_repaired as summary
import test_mz_common_gain_range_repair as fixtures


class CommonGainRangeRepairedPoolTests(unittest.TestCase):
    JOB = 70012345

    @classmethod
    def setUpClass(cls):
        fixtures.CommonGainRangeRepairTests.setUpClass()
        cls.fixture = fixtures.CommonGainRangeRepairTests()
        cls.spec = cls.fixture.spec
        cls.source, _ = cls.fixture.source()
        cls.failed = cls.fixture.failed
        cls.root = summary.common.SOURCE.parent / f"20261005-MZ-common-gain-repair-{cls.JOB}"
        cls.templates = {}
        for direction in (0, 1):
            for model in summary.MODELS:
                with tempfile.TemporaryDirectory() as directory:
                    report, *_ = cls.fixture.exercise(Path(directory) / "new", direction, model)
                report["job_id"] = str(cls.JOB)
                cls.templates[direction, model] = report

    def reports(self):
        reports = deepcopy(self.templates)
        for (direction, model), report in reports.items():
            for index, person in enumerate(sorted(report["participant_scores"])):
                n = report["participant_scores"][person]["heldout_trials"]
                mean = -3. + 2 * direction + (.02 + .04 * direction + index * .0001 if model == "MZ" else 0.)
                report["participant_scores"][person][model + "_log_sum"] = mean * n
                report["completed_model_scores"][model][person]["log_sum"] = mean * n
        return reports

    def invoke(self, reports, *, source=None, failed=None, output=None):
        paths = summary.report_paths(self.root, self.JOB)
        source = deepcopy(self.source) if source is None else source
        failed = deepcopy(self.failed) if failed is None else failed
        before = deepcopy((reports, source, failed, self.spec))
        bindings = (summary.common.PHASE, summary.common.MODELS, summary.common.MAX_SECONDS, summary.common.validate_fit)
        allowed = {paths[key]: json.dumps(report, allow_nan=True) for key, report in reports.items()}
        allowed.update({summary.SOURCE_REPORT: json.dumps(source), summary.common.IMPLEMENTATION: json.dumps(self.spec),
            **{summary.repair.failed_path(direction): json.dumps(failed[direction]) for direction in (0, 1)}})
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in paths.values() and path not in allowed:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"Only explicit component/baseline/failure reports/spec may be read: {path}")
            return allowed[path]

        try:
            with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
                if output is None:
                    result = summary.pool(self.root, self.JOB, self.spec)
                else:
                    code = summary.main(["--source-root", str(self.root), "--job-id", str(self.JOB),
                        "--implementation", str(summary.common.IMPLEMENTATION), "--output", str(output)])
                    complete = len(reports) == 4 and all(report["common_gain_component_complete"] for report in reports.values())
                    self.assertEqual(code, 0 if complete else 1)
                    result = None
        finally:
            self.assertEqual((reports, source, failed, self.spec), before)
            self.assertEqual((summary.common.PHASE, summary.common.MODELS, summary.common.MAX_SECONDS, summary.common.validate_fit), bindings)
        return (json.loads(output.read_text()) if output is not None else result), reads

    def assert_scope(self, result):
        for field in summary.common.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "retry_wave_selected", "automatic_retry", "task_versions_are_independent_replications", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertNotIn("participant_scores", result)
        self.assertNotIn("counts_as_curtailment_failure", result)
        self.assertNotIn("cumulative_process_elapsed_seconds", result)
        self.assertEqual(result["per_model_direction_maximum_sweeps"], {"M0": [1500, 1500], "MZ": [500, 500]})
        self.assertEqual(result["fixed_primary_participant_scores"], self.source["participant_scores"])
        self.assertEqual(result["unique_prior_process_elapsed_seconds"], sum(report["elapsed_seconds"] for report in self.failed.values()))
        self.assertEqual(len(result["prior_direction_failures"]), 2)
        for context in result["prior_direction_failures"].values():
            self.assertEqual(context["previous_support_contacts"], self.failed[context["direction"]]["fits"][0]["support_contacts"])

    def test_four_complete_components_keep_actual_sweeps_and_pool_log_sums_counts_with_prior_cost_once(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(reports, output=Path(directory) / "pool.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["complete_components"], [{"direction": d, "model": m} for d in (0, 1) for m in summary.MODELS])
        self.assertEqual(result["unknown_components"], [])
        self.assertEqual(result["new_component_process_elapsed_seconds"], sum(r["elapsed_seconds"] for r in reports.values()))
        self.assertEqual(result["runtime_implementations"], {m: summary.repair.runtime_specification(self.spec, m) for m in summary.MODELS})
        self.assertGreater(len(reports[0, "M0"]["fits"][0]["objective_history"]), 500)
        self.assertLessEqual(len(reports[0, "MZ"]["fits"][0]["objective_history"]), 500)
        differences = []
        for person, row in result["common_gain_participant_scores"].items():
            expected_n = 96 if row["version"] == "EI" else 176
            self.assertEqual(result["retained_trials_per_participant"][person], expected_n)
            for model in summary.MODELS:
                total = sum(reports[d, model]["participant_scores"][person][model + "_log_sum"] for d in (0, 1))
                self.assertEqual(row[model], total / expected_n)
                self.assertNotEqual(row[model], total / 256)
            differences.append(row["MZ"] - row["M0"])
        n = [reports[d, "M0"]["participant_scores"]["p000"]["heldout_trials"] for d in (0, 1)]
        self.assertNotEqual(n[0], n[1])
        wrong = sum(reports[d, "M0"]["participant_scores"]["p000"]["M0_log_sum"] / n[d] for d in (0, 1)) / 2
        self.assertNotEqual(result["common_gain_participant_scores"]["p000"]["M0"], wrong)
        diagnostic = result["common_gain_nested_diagnostic"]
        self.assertEqual(diagnostic["participants"], 108)
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences) / math.sqrt(108), places=14)
        self.assertEqual(set(reads), {*summary.report_paths(self.root, self.JOB).values(), summary.SOURCE_REPORT,
            summary.repair.failed_path(0), summary.repair.failed_path(1), summary.common.IMPLEMENTATION})
        self.assert_scope(result)

    def test_any_missing_or_real_technical_component_prevents_every_joint_contrast(self):
        for missing in self.templates:
            reports = self.reports()
            del reports[missing]
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_components"], [{"direction": missing[0], "model": missing[1]}])
            self.assertNotIn("common_gain_nested_diagnostic", result)
        options = ({"bad_source": True}, {"replay_error": True}, {"nonconverged": True}, {"score_error": True},
                   {"deadline_stage": "failed_read"}, {"deadline_stage": "fit"},
                   {"deadline_stage": "scoring"}, {"deadline_stage": "finalization"})
        for index, option in enumerate(options):
            direction, model = (0, "M0") if index % 2 == 0 else (1, "MZ")
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "new", direction, model, **option)
            technical["job_id"] = str(self.JOB)
            reports = self.reports()
            reports[direction, model] = technical
            result, _ = self.invoke(reports)
            self.assertEqual(result["unknown_components"], [{"direction": direction, "model": model}])
            row = next(row for row in result["components"] if (row["direction"], row["fitted_model"]) == (direction, model))
            self.assertEqual(row["status"], "technical_unknown")
            self.assertEqual(row["completed_model_scores"], technical["completed_model_scores"])
            self.assertNotIn("common_gain_nested_diagnostic", result)
            self.assertNotIn("common_gain_participant_scores", result)
            self.assert_scope(result)

    def test_runtime_model_identity_prior_reference_and_completed_score_mutations_abort(self):
        changes = (
            lambda r: r.update(fitted_model="MZ"), lambda r: r.update(fitted_models=["M0", "MZ"]),
            lambda r: r.update(phase=summary.common.PHASE), lambda r: r.update(job_id="70012346"),
            lambda r: r.update(source_context_validated=False),
            lambda r: r["runtime_implementation"]["optimizer"].update(maximum_sweeps=500),
            lambda r: r.update(fitted_maximum_sweeps=500),
            lambda r: r.update(execution_kind="previously_unattempted_MZ_component"),
            lambda r: r["prior_failed_context"]["previous_support_contacts"].pop(),
            lambda r: r.update(prior_direction_process_elapsed_seconds=1.),
            lambda r: r.update(cumulative_process_elapsed_seconds=1.),
            lambda r: r.update(fresh_fit_calls=0),
            lambda r: r["fits"][0].update(objective_history=[7000.] * 1501),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r["fits"][0].update(training_runs=[1, 2]),
            lambda r: r["participant_scores"]["p000"].update(heldout_trials=128),
            lambda r: r["participant_scores"]["p000"].update(M0_log_sum=float("nan")),
            lambda r: r["participant_scores"]["p000"].update(M0_log_sum=-123.),
            lambda r: r.update(full_gate="pass"),
        )
        for number, change in enumerate(changes):
            reports = {(0, "M0"): self.reports()[0, "M0"]}
            change(reports[0, "M0"])
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(reports)
        reports = self.reports()
        reports[0, "MZ"]["fits"][0]["objective_history"] = [7000.] * 501
        with self.assertRaises(ValueError):
            self.invoke(reports)

    def test_retained_slots_must_match_exact_failure_and_models_cannot_use_different_trial_sets(self):
        for model in summary.MODELS:
            reports = self.reports()
            support = reports[0, model]["retained_support"]["p000"]
            current = {tuple(slot) for slot in support["heldout_slots"]}
            replacement = next((run, trial) for run in (2, 3) for trial in range(1, 65) if (run, trial) not in current)
            support["heldout_slots"][0] = list(replacement)
            with self.assertRaisesRegex(ValueError, "previously validated filtered"):
                self.invoke(reports)
        for change in (lambda r: r["retained_support"]["p000"].update(full_session_trials=256),
                       lambda r: r["retained_support"]["p000"]["heldout_slots"][0].__setitem__(0, 1),
                       lambda r: r["retained_support"]["p000"].update(training_trials=128)):
            reports = self.reports()
            change(reports[0, "M0"])
            with self.assertRaises(ValueError):
                self.invoke(reports)

    def test_actual_source_primary_replay_and_both_exact_failed_attempts_are_required(self):
        changes = (lambda r: r.update(source_finished_utc="wrong"), lambda r: r.update(source_job_id="70000000"),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=2e-12))
        for change in changes:
            reports = self.reports()
            change(reports[1, "MZ"])
            with self.assertRaises(ValueError):
                self.invoke(reports)
        for direction in (0, 1):
            failed = deepcopy(self.failed)
            failed[direction]["finished_utc"] = "wrong"
            with self.assertRaises(ValueError):
                self.invoke(self.reports(), failed=failed)
        source = deepcopy(self.source)
        source["seed"] += 1
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke(self.reports(), source=source)

    def test_negative_nested_gain_is_a_complete_sensitivity_without_gate_or_recovery_claim(self):
        reports = self.reports()
        for direction in (0, 1):
            for person, row in reports[direction, "MZ"]["participant_scores"].items():
                value = reports[direction, "M0"]["participant_scores"][person]["M0_log_sum"] - .03 * row["heldout_trials"]
                row["MZ_log_sum"] = value
                reports[direction, "MZ"]["completed_model_scores"]["MZ"][person]["log_sum"] = value
        result, _ = self.invoke(reports)
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertAlmostEqual(result["common_gain_nested_diagnostic"]["mean_gain_nats_per_trial"], -.03)
        self.assert_scope(result)

    def test_explicit_parent_original_spec_and_fresh_output_guards_precede_reads(self):
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
