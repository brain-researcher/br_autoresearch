"""Focused saved indexed-null MVC runtime/source/deadline tests; compute only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import run_mz_null_indexed_mvc as rival
import test_mz_null_baseline_wave as fixtures
import test_mz_null_completed_baseline_rival as completion_fixtures
from test_mz_null_representative import THREADS


SOURCE_WORKER = "70000013"


class IndexedNullMVCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullBaselineWaveTests.setUpClass()
        cls.source_fixture = fixtures.NullBaselineWaveTests("test_exact_source_mapping_preserves_pilot_and_two_array_ranges")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth
        completion_fixtures.CompletedNullBaselineRivalTests.setUpClass()
        cls.completion_fixture = completion_fixtures.CompletedNullBaselineRivalTests(
            "test_exact_completion_and_preserved_original_fit_lineage_are_required")

    def arguments(self, output, index=1, **changes):
        values = dict(replicate_index=index, implementation=rival.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=8400)
        values.update(changes)
        return argparse.Namespace(**values)

    def reports(self, index):
        if index == 35:
            failed = self.completion_fixture.source_fixture.source()
            return self.completion_fixture.completed(failed), failed
        report = self.source_fixture.report(index)
        report["job_id"] = SOURCE_WORKER
        return report, None

    def exercise(self, output, index=1, *, source_change=None, truth_change=None,
                 failed_change=None, failure=None, failed_half=None):
        source, failed = self.reports(index)
        spec, truth = deepcopy(self.spec), deepcopy(self.truth)
        if source_change:
            source_change(source)
        if truth_change:
            truth_change(truth)
        if failed_change:
            failed_change(failed)
        preserved = deepcopy((source, failed, spec, truth))
        directory, source_path = rival.source_paths(index)
        allowed = {rival.IMPLEMENTATION: json.dumps(spec), source_path: json.dumps(source),
                   directory / "generating_parameters.json": json.dumps(truth)}
        if failed is not None:
            allowed[rival.completed35.FAILED_REPORT] = json.dumps(failed)
        calls, scores, validations, checkpoints, reads, source_validations, alarms = [], [], [], [], [], [], []
        clock, alarm_active = 0., False
        actual_writer, actual_validator, actual_source = rival.write_report, rival.validate_fit, rival.validate_source
        original_references = (rival.null.run, rival.null.validate_source, rival.null.fit_partially_pooled,
                               rival.null.REPORT_NAME, rival.null.SOURCE)
        original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def alarm(seconds):
            nonlocal alarm_active
            alarms.append(seconds)
            alarm_active = seconds > 0

        def reader(path, *args, **kwargs):
            nonlocal clock
            self.assertTrue(alarm_active, "Every scientific/specification/source read must be inside the process timer")
            reads.append(path)
            if path == source_path and failure == "missing":
                raise FileNotFoundError(path)
            if path == source_path and failure == "source_deadline":
                clock = 8400.
            return allowed[path]

        def loader(path, check):
            self.assertEqual(path, directory / "simulated_events.csv.gz")
            check()
            return self.people

        def validate_source(report, source_truth, people, selected, implementation):
            source_validations.append((selected, deepcopy(implementation)))
            return actual_source(report, source_truth, people, selected, implementation)

        def fitter(training, model, bound, implementation):
            self.assertTrue(alarm_active)
            calls.append((training, model, bound, deepcopy(implementation)))
            if failure == "fit_signal" and len(calls) == failed_half:
                rival.interrupted(signal.SIGTERM, None)
            converged = not (failure == "nonconvergence" and len(calls) == failed_half)
            fit = self.fixture.fit(model, converged=converged)
            count = 650 if converged else 1000
            fit["objective_history"] = [float(value) for value in range(count, 1, -1)] + [2. if converged else 1.]
            return fit

        def validate(fit, model, train, held, implementation, people):
            validations.append((model, tuple(train), implementation["optimizer"]["maximum_sweeps"]))
            return actual_validator(fit, model, train, held, implementation, people)

        def score(fit, scoring, implementation, check):
            check()
            self.assertTrue(alarm_active)
            self.assertEqual(implementation, self.spec)
            self.assertEqual(tuple(fit["scoring_runs"]), scoring)
            scores.append((fit["model"], tuple(fit["training_runs"])))
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            if failure == "replay" and fit["model"] == "M0":
                value += 1e-9
            return {person: value for person in self.people}

        def writer(destination, report):
            nonlocal clock
            if report["status"] == "rival_diagnostics_completed":
                self.assertTrue(alarm_active, "Successful finalization remains inside the whole-process timer")
                if failure == "finalization_deadline":
                    clock = 8400.
            checkpoints.append(deepcopy(report))
            return actual_writer(destination, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(rival, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(rival.time, "monotonic", side_effect=lambda: clock))
            stack.enter_context(patch.object(rival.signal, "alarm", side_effect=alarm))
            stack.enter_context(patch.object(rival.null.shared, "load_cohort", side_effect=loader))
            stack.enter_context(patch.object(rival, "load_split", side_effect=lambda events, train, held: (tuple(train), tuple(held))))
            stack.enter_context(patch.object(rival, "validate_source", side_effect=validate_source))
            stack.enter_context(patch.object(rival, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(rival, "validate_fit", side_effect=validate))
            stack.enter_context(patch.object(rival.null.shared, "score_fit", side_effect=score))
            stack.enter_context(patch.object(rival, "write_report", side_effect=writer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = rival.run(self.arguments(output, index))
        self.assertEqual(original_references, (rival.null.run, rival.null.validate_source, rival.null.fit_partially_pooled,
                                               rival.null.REPORT_NAME, rival.null.SOURCE))
        self.assertEqual({sig: signal.getsignal(sig) for sig in original_handlers}, original_handlers)
        self.assertEqual(alarms, [8400, 0])
        self.assertFalse(alarm_active)
        self.assertEqual((source, failed, spec, truth), preserved)
        self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
        self.assertEqual([path.name for path in output.iterdir()], [rival.REPORT_NAME])
        return report, calls, validations, checkpoints, reads, source, scores, source_validations

    def assert_scope(self, report, index):
        self.assertEqual((report["phase"], report["generating_model"], report["fitted_model"]), (rival.PHASE, "M0", "MVC"))
        self.assertEqual((report["replicate_index"], report["seed"]), (index, 1300037 + 10007 * index))
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual(report["selected_models"], ["MVC"])
        self.assertEqual(report["fixed_cohort_count"], 200)
        self.assertEqual(report["component_scope"], "MVC_only")
        self.assertEqual(report["wave_scope"], "MVC_component_only")
        self.assertTrue(report["wave_selected"])
        for field in ("full_menu_wave_selected", "retry_selected", "automatic_retry", "simulation_regenerated",
                      "simulation_copied", "saved_M0_MZ_refitted", "observed_choices_or_RTs_used", "episode_terminal"):
            self.assertFalse(report[field])
        for field in rival.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertNotIn("counts_as_curtailment_failure", report)
        self.assertNotIn("core_family_selection_pass", report)
        self.assertNotIn("cumulative_process_elapsed_seconds", report)
        self.assertFalse(report["numerical_override"]["nonconverged_checkpoint_state_reused"])
        if "implementation" in report:
            runtime = deepcopy(report["runtime_implementation"])
            self.assertEqual(runtime["optimizer"]["maximum_sweeps"], 1000)
            runtime["optimizer"]["maximum_sweeps"] = 500
            self.assertEqual(runtime, report["implementation"])
            self.assertEqual(report["implementation"], self.spec)

    def test_exact_fixed_sources_preserve_pilot0_and_index35_uses_its_completion(self):
        for index in (1, 4, 5, 35, 199):
            directory, report = rival.source_paths(index)
            array = 46579924 if index <= 4 else 46580387
            self.assertEqual(directory, rival.indexed_source.baseline.RUNS /
                             f"20261005-M0-null-baseline-array-{array}/replicate{index}")
            self.assertEqual(report, rival.completed35.COMPLETION_REPORT if index == 35
                             else directory / "mz_null_replicate.json")
        for invalid in (0, -1, 200, True):
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                rival.source_paths(invalid)
        original = deepcopy(self.spec)
        runtime = rival.runtime_specification(original)
        self.assertEqual(original, self.spec)
        runtime["optimizer"]["maximum_sweeps"] = 500
        self.assertEqual(runtime, original)
        original["optimizer"]["maximum_sweeps"] = 1000
        with self.assertRaises(ValueError):
            rival.runtime_specification(original)

    def test_both_array_ranges_keep_original_baselines_and_fit_only_two_fresh_runtime1000_halves(self):
        for index in (4, 5):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, reads, source, scores, source_validations = self.exercise(Path(directory) / "new", index)
            self.assertTrue(report["rival_diagnostics_complete"])
            self.assertEqual([(train, model, bound) for train, model, bound, spec in calls],
                             [((1, 4), "MVC", "fixed"), ((2, 3), "MVC", "fixed")])
            self.assertEqual(validations, [("MVC", (1, 4), 1000), ("MVC", (2, 3), 1000)])
            self.assertEqual(source_validations, [(index, self.spec)])
            for train, model, bound, spec in calls:
                self.assertEqual(spec, report["runtime_implementation"])
            self.assertTrue(report["saved_scores_replayed"])
            self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
            self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
            for person, row in source["participant_scores"].items():
                self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
            self.assertEqual(scores[:4], [(model, train) for train, held in rival.HALVES for model in ("M0", "MZ")])
            self.assertEqual(report["source_job_id"], SOURCE_WORKER)
            self.assertIsNone(report["baseline_lineage"])
            self.assertNotIn(rival.QUALIFYING_NULL0_REPORT, reads)
            self.assertFalse(checkpoints[0]["source_baseline_validated"])
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, index)

    def test_index35_reuses_validated_completed_baseline_and_original_simulation_with_separate_cost_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, validations, checkpoints, reads, source, scores, source_validations = self.exercise(Path(directory) / "completed35", 35)
        self.assertTrue(report["rival_diagnostics_complete"])
        self.assertEqual(report["source_job_id"], "46586603")
        self.assertEqual(report["source_report"], str(rival.completed35.COMPLETION_REPORT))
        self.assertEqual(report["source_dir"], str(rival.completed35.completion.SOURCE))
        self.assertEqual(report["baseline_lineage"]["original_worker_job_id"], "46580478")
        self.assertEqual(report["baseline_lineage"]["cumulative_baseline_process_elapsed_seconds"], source["cumulative_process_elapsed_seconds"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(reads, [rival.IMPLEMENTATION, rival.completed35.COMPLETION_REPORT,
                               rival.completed35.completion.SOURCE / "generating_parameters.json", rival.completed35.FAILED_REPORT])
        for person, row in source["participant_scores"].items():
            self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
        for checkpoint in checkpoints:
            self.assert_scope(checkpoint, 35)

    def test_bad_original_or_completed_source_truth_and_replay_prevent_any_new_fit(self):
        scenarios = (
            (1, {"source_change": lambda source: source.update(replicate_index=2)}),
            (5, {"source_change": lambda source: source.update(pairwise_diagnostic_complete=False)}),
            (1, {"source_change": lambda source: source["fits"][0].update(objective_history=[1.] * 501)}),
            (1, {"truth_change": lambda truth: truth["p000"].update(rho=.1)}),
            (1, {"failure": "missing"}),
            (35, {"source_change": lambda source: source.update(job_id="46586604")}),
            (35, {"failed_change": lambda failed: failed.update(job_id="46580479")}),
            (35, {"source_change": lambda source: source["fits"][0].update(elapsed_seconds=1.)}),
            (1, {"failure": "replay"}), (35, {"failure": "replay"}),
        )
        for number, (index, scenario) in enumerate(scenarios):
            with self.subTest(scenario=number), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, reads, source, scores, source_validations = self.exercise(Path(directory) / "bad", index, **scenario)
            self.assertEqual(calls, [])
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertEqual(report["source_baseline_validated"], scenario.get("failure") == "replay")
            self.assertFalse(report["saved_scores_replayed"])
            self.assertEqual(report["fits"], [])
            self.assertNotIn("participant_scores", report)
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, index)

    def test_first_or_second_fit_failure_preserves_returned_checkpoints_without_scores_or_retry(self):
        for failure in ("nonconvergence", "fit_signal"):
            for half in (1, 2):
                with self.subTest(failure=failure, half=half), tempfile.TemporaryDirectory() as directory:
                    report, calls, validations, checkpoints, reads, source, scores, source_validations = self.exercise(
                        Path(directory) / "failed", failure=failure, failed_half=half)
                self.assertEqual(len(calls), half)
                self.assertEqual(len(report["fits"]), half if failure == "nonconvergence" else half - 1)
                self.assertEqual(len(report["scored_primary_directions"]), half - 1)
                self.assertEqual(report["status"], "technical_rival_failure")
                self.assertFalse(report["rival_diagnostics_complete"])
                self.assertTrue(report["saved_scores_replayed"])
                self.assertNotIn("participant_scores", report)
                self.assertEqual(report["completed_rivals"], [])
                self.assertEqual(report["diagnostics"], {})
                if failure == "nonconvergence":
                    self.assertEqual(len(report["fits"][-1]["objective_history"]), 1000)
                    self.assertFalse(report["fits"][-1]["converged"])
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, 1)

    def test_whole_process_deadline_covers_source_reads_and_successful_finalization(self):
        for failure in ("source_deadline", "finalization_deadline"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, reads, source, scores, source_validations = self.exercise(Path(directory) / "deadline", failure=failure)
            self.assertEqual(len(calls), 0 if failure == "source_deadline" else 2)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertEqual(checkpoints[-1], report)

    def test_compute_original_spec_one_cpu_threads_cap_and_fresh_output_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read before guards")):
            with patch.object(rival, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                rival.run(self.arguments(Path(directory) / "new"))
            with patch.object(rival, "require_compute"):
                for changes in ({"replicate_index": 0}, {"replicate_index": 200}, {"replicate_index": True},
                                {"implementation": Path(directory) / "wrong.json"}, {"max_seconds": 0},
                                {"max_seconds": 8401}, {"max_seconds": True}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "new", **changes))
                for name in ("SLURM_CPUS_PER_TASK", *THREADS):
                    with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "new"))
                for output in (Path(directory), *(path / "new" for path in rival.PRESERVED_DIRECTORIES)):
                    with self.subTest(output=output), self.assertRaises(FileExistsError):
                        rival.run(self.arguments(output))


if __name__ == "__main__":
    unittest.main()
