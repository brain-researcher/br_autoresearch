"""Focused completed-null35 rival lineage/replay checks; run only on compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_mz_null_completed_baseline_rival as rival
import test_mz_null_baseline_completion as fixtures
from test_mz_null_representative import THREADS


class CompletedNullBaselineRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullBaselineCompletionTests.setUpClass()
        cls.source_fixture = fixtures.NullBaselineCompletionTests(
            "test_context_validates_original_three_fit_prefix_without_changing_history_or_cost")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def completed(self, failed):
        fits, metadata = rival.completion.completion_context(failed, self.truth, self.people, self.spec)
        fresh = self.fixture.fit("MZ", (2, 3), (1, 4))
        fresh["elapsed_seconds"] = 900.
        fits.append(fresh)
        scores = deepcopy(self.fixture.source()["participant_scores"])
        return {
            "analysis_id": "behavioral_mechanism_kill_test", "phase": rival.completion.PHASE,
            "job_id": rival.COMPLETION_JOB_ID, "generating_model": "M0", "fitted_models": ["M0", "MZ"],
            "replicate_index": 35, "seed": 1650282, "SESOI_level": 0.,
            "focal_generating_truth": {"eta": 0., "rho": 0., "criterion": 0.}, "implementation": deepcopy(self.spec),
            "source_dir": str(rival.completion.SOURCE), "source_report": rival.completion.SOURCE_REPORT,
            "source_job_id": rival.completion.SOURCE_JOB_ID,
            "generating_parameters": str(rival.completion.SOURCE / "generating_parameters.json"),
            "simulated_events": str(rival.completion.SOURCE / "simulated_events.csv.gz"),
            "started_utc": "2026-10-05T06:50:00+00:00", "finished_utc": "2026-10-05T07:06:40+00:00",
            "status": "pairwise_diagnostic_completed", "pairwise_diagnostic_complete": True,
            "preserved_scores_recomputed": True, "participants": 108, "runs": 432, "trials": 27648,
            "scored_primary_directions": [{"training_runs": list(train), "scoring_runs": list(held)}
                                          for train, held in rival.null.HALVES],
            "scored_fit_prefix": [{"model": model, "training_runs": list(train), "scoring_runs": list(held)}
                                  for model, train, held in rival.completion.EXPECTED],
            "selected_replicates": [35], "fixed_cohort_count": 200,
            "component_scope": "single_M0_MZ_baseline_completion",
            "saved_score_agreement": "not_evaluated_original_failure_has_no_saved_scores",
            **{field: False for field in ("simulation_regenerated", "simulation_copied", "saved_completed_fits_refitted",
                "unfinished_fit_state_reused", "observed_choices_or_RTs_used", "old_synthetic_responses_used",
                "wave_selected", "retry_wave_selected", "full_menu_wave_selected", "episode_terminal")},
            **{field: "not_evaluated" for field in rival.null.CLAIM_FIELDS}, "full_recovery_pass": None,
            "fits": fits, "participant_scores": scores,
            "nested_comparison": rival.completion.fixed.nested_contrast(scores, "MZ", "M0"),
            "internal_seconds": 2700, "elapsed_seconds": 1000., "fresh_fit_elapsed_seconds": 900.,
            "source_validation_and_copy_seconds": .02, "preserved_fit_rescore_seconds": .03, "fresh_fit_score_seconds": .01,
            "cumulative_process_elapsed_seconds": failed["elapsed_seconds"] + 1000., **metadata,
        }

    def arguments(self, output, model="MC", **changes):
        values = dict(model=model, implementation=rival.null.IMPLEMENTATION, output_dir=Path(output),
                      max_seconds=rival.MAX_SECONDS.get(model, 2400))
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return tuple(getattr(rival.null, name) for name in
                     ("SOURCE", "SOURCE_REPORT", "SOURCE_JOB_ID", "SEED", "REPLICATE_INDEX",
                      "REPORT_NAME", "validate_source", "write_report"))

    def test_exact_completion_and_preserved_original_fit_lineage_are_required(self):
        failed = self.source_fixture.source()
        report = self.completed(failed)
        before = deepcopy((report, failed))
        lineage = rival.validate_completed_baseline(report, failed, self.truth, self.people, self.spec)
        self.assertEqual((report, failed), before)
        self.assertEqual(lineage["original_worker_job_id"], "46580478")
        self.assertEqual(lineage["completion_job_id"], "46586603")
        self.assertEqual(lineage["cumulative_baseline_process_elapsed_seconds"], failed["elapsed_seconds"] + report["elapsed_seconds"])
        changes = (("phase", rival.completion.indexed.PHASE), ("job_id", "46586604"),
                   ("replicate_index", 36), ("seed", 1650283), ("generating_model", "MZ"),
                   ("source_job_id", "46580479"), ("source_finished_utc", "different"),
                   ("source_report", "another.json"), ("source_dir", "/wrong"),
                   ("simulated_events", "/copied/events.csv.gz"), ("implementation", {}),
                   ("status", "running"), ("pairwise_diagnostic_complete", False),
                   ("source_context_validated", False), ("preserved_scores_recomputed", False),
                   ("scored_primary_directions", []), ("scored_fit_prefix", []),
                   ("wave_selected", True), ("simulation_regenerated", True), ("full_gate", "pass"),
                   ("cumulative_process_elapsed_seconds", 1000.), ("fresh_fit_elapsed_seconds", 901.))
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                rival.validate_completed_baseline({**report, field: value}, failed, self.truth, self.people, self.spec)
        for position in range(3):
            for field in ("elapsed_seconds", "parameters", "objective_history"):
                altered = deepcopy(report)
                if field == "elapsed_seconds":
                    altered["fits"][position][field] += .1
                elif field == "parameters":
                    altered["fits"][position][field][0][0] += .001
                else:
                    altered["fits"][position][field][0] += .001
                with self.subTest(position=position, field=field), self.assertRaises(ValueError):
                    rival.validate_completed_baseline(altered, failed, self.truth, self.people, self.spec)

    def test_failed_source_actual_people_truth_and_original500_cannot_be_substituted(self):
        failed = self.source_fixture.source()
        report = self.completed(failed)
        for field, value in (("job_id", "46580479"), ("error", "another cause"),
                             ("status", "pairwise_diagnostic_completed"), ("fits", failed["fits"][:2])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                rival.validate_completed_baseline(report, {**failed, field: value}, self.truth, self.people, self.spec)
        for position in range(4):
            altered = deepcopy(report)
            altered["fits"][position]["participants"][0] = "changed_person"
            with self.subTest(position=position), self.assertRaises(ValueError):
                rival.validate_completed_baseline(altered, failed, self.truth, self.people, self.spec)
        altered = deepcopy(report)
        altered["fits"][3]["objective_history"] = [1.] * 501
        with self.assertRaises(ValueError):
            rival.validate_completed_baseline(altered, failed, self.truth, self.people, self.spec)
        truth = deepcopy(self.truth)
        truth["p000"]["rho"] = .1
        with self.assertRaises(ValueError):
            rival.validate_completed_baseline(report, failed, truth, self.people, self.spec)

    def exercise(self, output, model="MC", *, bad_completion=False, bad_failed=False, replay_error=False, failed_half=None):
        module = rival.null
        failed = self.source_fixture.source()
        source = self.completed(failed)
        if bad_completion:
            source["pairwise_diagnostic_complete"] = False
        if bad_failed:
            failed["job_id"] = "46580479"
        before = deepcopy((source, failed, self.truth, self.spec))
        allowed = {module.IMPLEMENTATION: json.dumps(self.spec), rival.COMPLETION_REPORT: json.dumps(source),
                   rival.completion.SOURCE / "generating_parameters.json": json.dumps(self.truth),
                   rival.FAILED_REPORT: json.dumps(failed)}
        calls, reads, scores, checkpoints = [], [], [], []
        original_bindings = self.bindings()
        actual_writer = module.write_report

        def reader(path, *args, **kwargs):
            reads.append(path)
            return allowed[path]

        def split(events, train, held):
            self.assertEqual(events, rival.completion.SOURCE / "simulated_events.csv.gz")
            return tuple(train), tuple(held)

        def fitter(training, selected, bound, spec):
            calls.append((training, selected, bound, deepcopy(spec)))
            return self.fixture.fit(selected, converged=len(calls) != failed_half)

        def scorer(fit, scoring, spec, check):
            check()
            self.assertEqual(tuple(fit["scoring_runs"]), scoring)
            self.assertEqual(spec, self.spec)
            scores.append((fit["model"], fit["training_runs"]))
            value = {"M0": -1., "MZ": -.99, "MC": -.98, "MV": -.97, "MVZ": -.96}[fit["model"]]
            if replay_error and fit["model"] == "M0":
                value += 1e-9
            return {person: value for person in self.people}

        def writer(destination, report):
            checkpoints.append(deepcopy(report))
            return actual_writer(destination, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(rival, "require_compute"))
            stack.enter_context(patch.object(module, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            loader = stack.enter_context(patch.object(module.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(module, "load_split", side_effect=split))
            stack.enter_context(patch.object(module, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(module.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(module, "write_report", side_effect=writer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = rival.run(self.arguments(output, model))
            self.assertEqual(self.bindings(), installed)
            loader.assert_called_once()
            self.assertEqual(loader.call_args.args[0], rival.completion.SOURCE / "simulated_events.csv.gz")
        self.assertEqual(self.bindings(), original_bindings)
        self.assertEqual((source, failed, self.truth, self.spec), before)
        self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
        self.assertEqual(sorted(path.name for path in output.iterdir()), [rival.REPORT_NAME])
        return report, calls, reads, scores, checkpoints, source, failed

    def assert_scope(self, report, model):
        self.assertEqual((report["phase"], report["generating_model"], report["fitted_model"], report["replicate_index"], report["seed"]),
                         (rival.PHASE, "M0", model, 35, 1650282))
        self.assertEqual(report["source_dir"], str(rival.completion.SOURCE))
        self.assertEqual(report["source_report"], str(rival.COMPLETION_REPORT))
        self.assertEqual(report["selected_models"], [model])
        self.assertEqual(report["selected_replicates"], [35])
        self.assertEqual(report["component_scope"], f"{model}_only")
        self.assertEqual(report["fixed_cohort_count"], 200)
        for field in ("wave_selected", "retry_wave_selected", "full_menu_wave_selected", "simulation_regenerated",
                      "simulation_copied", "saved_M0_MZ_refitted", "episode_terminal"):
            self.assertFalse(report[field])
        for field in rival.null.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])

    def test_each_model_replays_completed_baseline_last_bits_then_fits_only_its_two_original_halves(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in rival.MODELS:
                with self.subTest(model=model):
                    report, calls, reads, scores, checkpoints, source, failed = self.exercise(Path(directory) / model, model)
                    self.assertEqual(calls, [((1, 4), model, "fixed", self.spec), ((2, 3), model, "fixed", self.spec)])
                    self.assertTrue(report["rival_diagnostics_complete"])
                    self.assertTrue(report["source_baseline_validated"])
                    self.assertTrue(report["saved_scores_replayed"])
                    self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
                    self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
                    self.assertEqual(report["source_job_id"], "46586603")
                    self.assertEqual(report["source_finished_utc"], source["finished_utc"])
                    self.assertEqual(report["baseline_lineage"]["original_worker_job_id"], "46580478")
                    self.assertEqual(report["baseline_lineage"]["cumulative_baseline_process_elapsed_seconds"], source["cumulative_process_elapsed_seconds"])
                    self.assertNotIn("cumulative_process_elapsed_seconds", report)
                    self.assertLess(report["elapsed_seconds"], source["elapsed_seconds"])
                    self.assertEqual(scores[:4], [(model0, list(train)) for model0, train, held in rival.completion.EXPECTED])
                    self.assertEqual(report["completed_rivals"], [model])
                    for person, row in source["participant_scores"].items():
                        self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
                    self.assertEqual(reads, [rival.null.IMPLEMENTATION, rival.COMPLETION_REPORT,
                                             rival.completion.SOURCE / "generating_parameters.json", rival.FAILED_REPORT])
                    self.assertFalse(checkpoints[0]["source_baseline_validated"])
                    self.assertIsNone(checkpoints[0]["source_job_id"])
                    for checkpoint in checkpoints:
                        self.assert_scope(checkpoint, model)

    def test_incomplete_completion_or_wrong_failed_lineage_aborts_before_replay_and_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            for failure in ("bad_completion", "bad_failed"):
                report, calls, reads, scores, checkpoints, source, failed = self.exercise(Path(directory) / failure, **{failure: True})
                self.assertEqual(calls, [])
                self.assertEqual(scores, [])
                self.assertFalse(report["source_baseline_validated"])
                self.assertIsNone(report["source_job_id"])
                self.assertFalse(report["saved_scores_replayed"])
                self.assertEqual(report["status"], "technical_rival_failure")
                self.assertNotIn("participant_scores", report)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, "MC")

    def test_replay_failure_and_both_rival_half_failures_preserve_technical_claim_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            for number, scenario in enumerate(({"replay_error": True}, {"failed_half": 1}, {"failed_half": 2})):
                report, calls, reads, scores, checkpoints, source, failed = self.exercise(Path(directory) / str(number), "MVZ", **scenario)
                self.assertEqual(len(calls), scenario.get("failed_half", 0))
                self.assertFalse(report["rival_diagnostics_complete"])
                self.assertTrue(report["source_baseline_validated"])
                self.assertEqual(report["source_job_id"], "46586603")
                self.assertEqual(report["saved_scores_replayed"], "replay_error" not in scenario)
                self.assertEqual(report["completed_rivals"], [])
                self.assertEqual(report["diagnostics"], {})
                self.assertNotIn("participant_scores", report)
                self.assertEqual(len(report["fits"]), len(calls))
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, "MVZ")

    def test_scope_caps_and_inherited_guards_restore_bindings_without_source_reads(self):
        before = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(rival, "require_compute"), \
                patch.object(rival.null, "require_compute"), \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read before guards")):
            for model in ("M0", "MZ", "MVC"):
                with self.subTest(model=model), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh", model))
            for model, cap in rival.MAX_SECONDS.items():
                for seconds in (0, cap + 1, True):
                    with self.subTest(model=model, seconds=seconds), self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "fresh", model, max_seconds=seconds))
            with self.assertRaises(ValueError):
                rival.run(self.arguments(Path(directory) / "fresh", implementation=Path(directory) / "wrong.json"))
            for key in ("SLURM_CPUS_PER_TASK", *THREADS):
                with self.subTest(key=key), patch.dict(os.environ, {key: "2"}), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh"))
            for output in (Path(directory), *(path / "child" for path in rival.PRESERVED_DIRECTORIES)):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    rival.run(self.arguments(output))
        self.assertEqual(self.bindings(), before)


if __name__ == "__main__":
    unittest.main()
