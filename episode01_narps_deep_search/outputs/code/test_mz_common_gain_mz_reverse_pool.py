"""Focused mixed-runtime common-gain pooling checks; compute allocation only."""
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

import summarize_mz_common_gain_mz_reverse_repaired as summary
import test_mz_common_gain_mz_reverse_repair as fixtures


class CommonGainMZReversePoolTests(unittest.TestCase):
    JOB = 70012346

    @classmethod
    def setUpClass(cls):
        fixtures.CommonGainMZReverseRepairTests.setUpClass()
        cls.fixture = fixtures.CommonGainMZReverseRepairTests()
        cls.spec = cls.fixture.spec
        cls.source, _ = cls.fixture.source()
        cls.old_components = cls.fixture.old_components
        cls.original_failures = cls.fixture.prior.failed
        cls.repair_dir = summary.common.SOURCE.parent / f"20261005-MZ-common-gain-MZ-reverse-repair-{cls.JOB}" / "dir1/MZ"
        with tempfile.TemporaryDirectory() as directory:
            cls.template, *_ = cls.fixture.exercise(Path(directory) / "new")
        cls.template["job_id"] = str(cls.JOB)

    def panels(self, *, negative=False):
        old, new = deepcopy(self.old_components), deepcopy(self.template)
        for direction, model, report in ((0, "M0", old[0, "M0"]), (0, "MZ", old[0, "MZ"]),
                                          (1, "M0", old[1, "M0"]), (1, "MZ", new)):
            for index, person in enumerate(sorted(report["participant_scores"])):
                count = report["participant_scores"][person]["heldout_trials"]
                advantage = -.03 if negative else .02+.04*direction+index*.0001
                mean = -3.+2*direction + (advantage if model == "MZ" else 0.)
                report["participant_scores"][person][model+"_log_sum"] = count*mean
                report["completed_model_scores"][model][person]["log_sum"] = count*mean
        return old, new

    def invoke(self, old, new, *, source=None, original_failures=None, output=None):
        source = deepcopy(self.source) if source is None else source
        failures = deepcopy(self.original_failures) if original_failures is None else original_failures
        old_paths = summary.old.report_paths(summary.OLD_ROOT, summary.OLD_JOB)
        new_path = summary.repair_path(self.repair_dir, self.JOB)
        allowed = {old_paths[key]: json.dumps(report, allow_nan=True) for key, report in old.items()}
        allowed.update({summary.SOURCE_REPORT: json.dumps(source), summary.common.IMPLEMENTATION: json.dumps(self.spec),
            **{summary.old.repair.failed_path(direction): json.dumps(failures[direction]) for direction in (0, 1)}})
        if new is not None:
            allowed[new_path] = json.dumps(new, allow_nan=True)
        before = deepcopy((old, new, source, failures, self.spec))
        bindings = (summary.common.PHASE, summary.common.MODELS, summary.common.MAX_SECONDS, summary.common.validate_fit)
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path == new_path and new is None:
                raise FileNotFoundError(str(path))
            if path not in allowed:
                raise AssertionError(f"No event/truth or unselected report reads: {path}")
            return allowed[path]

        try:
            with patch.object(Path, "read_text", reader), patch.object(summary, "require_compute"), redirect_stdout(io.StringIO()):
                if output is None:
                    result = summary.pool(self.repair_dir, self.JOB, self.spec)
                else:
                    code = summary.main(["--repair-dir", str(self.repair_dir), "--repair-job-id", str(self.JOB),
                        "--implementation", str(summary.common.IMPLEMENTATION), "--output", str(output)])
                    self.assertEqual(code, 0 if new is not None and new["common_gain_component_complete"] else 1)
                    result = None
        finally:
            self.assertEqual((old, new, source, failures, self.spec), before)
            self.assertEqual((summary.common.PHASE, summary.common.MODELS, summary.common.MAX_SECONDS, summary.common.validate_fit), bindings)
        return (json.loads(output.read_text()) if output is not None else result), reads

    def assert_scope_and_cost(self, result, old, new):
        self.assertEqual(result["per_model_direction_maximum_sweeps"], {"M0": [1500, 1500], "MZ": [500, 1500]})
        self.assertEqual(result["historical_job_id"], "46615238")
        self.assertEqual(result["fixed_primary_participant_scores"], self.source["participant_scores"])
        self.assertEqual(result["unique_prior_process_elapsed_seconds"],
                         sum(self.original_failures[d]["elapsed_seconds"] for d in (0, 1)))
        self.assertEqual(result["historical_component_process_elapsed_seconds"],
                         sum(old[d, m]["elapsed_seconds"] for d in (0, 1) for m in summary.MODELS))
        self.assertEqual(result["new_repair_process_elapsed_seconds"], None if new is None else new["elapsed_seconds"])
        self.assertEqual(len(result["historical_components"]), 4)
        self.assertEqual(len(result["components"]), 4)
        self.assertEqual(result["prior_MZ_failed_context"]["previous_process_elapsed_seconds"], old[1, "MZ"]["elapsed_seconds"])
        self.assertEqual(result["prior_MZ_failed_context"]["previous_support_contacts"], old[1, "MZ"]["fits"][0]["support_contacts"])
        self.assertNotIn("cumulative_process_elapsed_seconds", result)
        for field in summary.common.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "retry_wave_selected", "automatic_retry",
                      "task_versions_are_independent_replications", "episode_terminal"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["full_recovery_pass"])
        self.assertNotIn("counts_as_curtailment_failure", result)

    def test_four_actual_panels_mixed_sweeps_count_weighting_and_all_prior_processes_once(self):
        old, new = self.panels()
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke(old, new, output=Path(directory) / "summary.json")
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertEqual(result["unknown_components"], [])
        self.assert_scope_and_cost(result, old, new)
        self.assertGreater(len(new["fits"][0]["objective_history"]), 500)
        self.assertLessEqual(len(old[0, "MZ"]["fits"][0]["objective_history"]), 500)
        differences = []
        for person, row in result["common_gain_participant_scores"].items():
            count = 96 if row["version"] == "EI" else 176
            self.assertEqual(result["retained_trials_per_participant"][person], count)
            for model in summary.MODELS:
                second = new if model == "MZ" else old[1, model]
                total = old[0, model]["participant_scores"][person][model+"_log_sum"] + second["participant_scores"][person][model+"_log_sum"]
                self.assertEqual(row[model], total/count)
                self.assertNotEqual(row[model], total/256)
            differences.append(row["MZ"]-row["M0"])
        n = [old[d, "M0"]["participant_scores"]["p000"]["heldout_trials"] for d in (0, 1)]
        self.assertNotEqual(n[0], n[1])
        wrong = sum(old[d, "M0"]["participant_scores"]["p000"]["M0_log_sum"]/n[d] for d in (0, 1))/2
        self.assertNotEqual(result["common_gain_participant_scores"]["p000"]["M0"], wrong)
        diagnostic = result["common_gain_nested_diagnostic"]
        self.assertEqual(diagnostic["participants"], 108)
        self.assertAlmostEqual(diagnostic["mean_gain_nats_per_trial"], statistics.mean(differences), places=14)
        self.assertAlmostEqual(diagnostic["participant_SE"], statistics.stdev(differences)/math.sqrt(108), places=14)
        self.assertEqual(set(reads), {summary.SOURCE_REPORT, summary.common.IMPLEMENTATION,
            summary.old.repair.failed_path(0), summary.old.repair.failed_path(1),
            *summary.old.report_paths(summary.OLD_ROOT, summary.OLD_JOB).values(),
            summary.repair_path(self.repair_dir, self.JOB)})
        old, new = self.panels(negative=True)
        result, _ = self.invoke(old, new)
        self.assertTrue(result["pooled_diagnostic_complete"])
        self.assertAlmostEqual(result["common_gain_nested_diagnostic"]["mean_gain_nats_per_trial"], -.03)
        self.assert_scope_and_cost(result, old, new)

    def test_missing_and_actual_technical_repair_prefixes_leave_joint_contrast_unknown(self):
        old, _ = self.panels()
        result, _ = self.invoke(old, None)
        self.assertEqual(result["unknown_components"], [{"direction": 1, "model": "MZ"}])
        self.assertNotIn("common_gain_nested_diagnostic", result)
        self.assert_scope_and_cost(result, old, None)
        options = ({"bad_source": True}, {"replay_error": True},
            {"failed_change": lambda report: report.update(finished_utc="wrong")},
            {"nonconverged": True}, {"score_error": True}, {"parallel_error": True},
            {"deadline_stage": "failed_read"}, {"deadline_stage": "finalization"})
        saw_replay_before_context = False
        for case in options:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                technical, *_ = self.fixture.exercise(Path(directory) / "new", **case)
            technical["job_id"] = str(self.JOB)
            self.assertFalse(technical["common_gain_component_complete"])
            saw_replay_before_context |= technical["saved_primary_scores_replayed"] and not technical["source_context_validated"]
            result, _ = self.invoke(old, technical)
            self.assertFalse(result["pooled_diagnostic_complete"])
            self.assertEqual(result["unknown_components"], [{"direction": 1, "model": "MZ"}])
            self.assertEqual(result["components"][-1]["status"], "technical_unknown")
            self.assertNotIn("common_gain_nested_diagnostic", result)
            self.assertNotIn("common_gain_participant_scores", result)
            self.assert_scope_and_cost(result, old, technical)
        self.assertTrue(saw_replay_before_context)

    def test_invalid_new_identity_runtime_fit_source_replay_or_cost_never_counts_complete(self):
        changes = (lambda r: r.update(job_id="another"), lambda r: r.update(phase=summary.old.repair.PHASE),
            lambda r: r.update(failed_report="another"), lambda r: r.update(fitted_maximum_sweeps=500),
            lambda r: r["runtime_implementation"]["optimizer"].update(maximum_sweeps=500),
            lambda r: r.update(source_context_validated=False), lambda r: r.update(workers=1),
            lambda r: r["fits"][0].update(objective_history=[7000.]*1501),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r["fits"][0].update(training_runs=[1, 4]),
            lambda r: r.update(source_finished_utc="wrong"),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda r: r["prior_MZ_failed_context"].update(previous_process_elapsed_seconds=1.),
            lambda r: r["original_M0_failed_context"].update(previous_process_elapsed_seconds=1.),
            lambda r: r.update(cumulative_process_elapsed_seconds=1.),
            lambda r: r["participant_scores"]["p000"].update(MZ_log_sum=float("nan")),
            lambda r: r["participant_scores"]["p000"].update(MZ_log_sum=-123.),
            lambda r: r.update(full_gate="pass"))
        for number, change in enumerate(changes):
            old, new = self.panels()
            change(new)
            with self.subTest(change=number), self.assertRaises((ValueError, RuntimeError)):
                self.invoke(old, new)

    def test_failure_lineages_preserved_components_and_actual_slots_counts_are_required(self):
        for change in (lambda r: r.update(finished_utc="wrong"),
                       lambda r: r["fits"][0]["support_contacts"].pop(),
                       lambda r: r["fits"][0]["objective_history"].__setitem__(-1, 0.)):
            old, new = self.panels()
            change(old[1, "MZ"])
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(old, new)
        for key in ((0, "M0"), (0, "MZ"), (1, "M0")):
            old, new = self.panels()
            old[key]["source_finished_utc"] = "wrong"
            with self.assertRaises(ValueError):
                self.invoke(old, new)
        for target in ("new", "old"):
            old, new = self.panels()
            report = new if target == "new" else old[0, "MZ"]
            report["participant_scores"]["p000"]["heldout_trials"] += 1
            with self.assertRaises(ValueError):
                self.invoke(old, new)
            old, new = self.panels()
            report = new if target == "new" else old[0, "MZ"]
            report["retained_support"]["p000"]["heldout_slots"][0][1] += 1
            with self.assertRaises(ValueError):
                self.invoke(old, new)
        old, new = self.panels()
        source = deepcopy(self.source)
        source["seed"] += 1
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke(old, new, source=source)
        failures = deepcopy(self.original_failures)
        failures[0]["finished_utc"] = "wrong"
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke(old, new, original_failures=failures)

    def test_explicit_repair_path_original_spec_and_fresh_cli_output_guards(self):
        with self.assertRaises(ValueError):
            summary.repair_path(self.repair_dir, self.JOB+1)
        with self.assertRaises(ValueError):
            summary.repair_path(summary.OLD_ROOT / "dir1/MZ", self.JOB)
        with tempfile.TemporaryDirectory() as directory, patch.object(summary, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No payload reads")):
            args = ["--repair-dir", str(self.repair_dir), "--repair-job-id", str(self.JOB),
                    "--implementation", str(summary.common.IMPLEMENTATION), "--output", str(Path(directory) / "new.json")]
            wrong = list(args)
            wrong[5] = str(Path(directory) / "another.json")
            with self.assertRaises(ValueError):
                summary.main(wrong)
            args[-1] = directory
            with self.assertRaises(FileExistsError):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
