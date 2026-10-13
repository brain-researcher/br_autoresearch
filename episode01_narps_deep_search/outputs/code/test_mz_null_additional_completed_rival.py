"""Focused completed-null108/176 lineage and model-routing checks; compute only."""
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

import run_mz_null_additional_completed_rival as rival
import test_mz_null_completed_baseline_rival as fixtures
from test_mz_null_representative import THREADS


class AdditionalCompletedNullRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CompletedNullBaselineRivalTests.setUpClass()
        cls.completed_fixture = fixtures.CompletedNullBaselineRivalTests(
            "test_exact_completion_and_preserved_original_fit_lineage_are_required")
        cls.fixture = cls.completed_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def sources(self, index):
        failed = self.completed_fixture.source_fixture.source()
        worker, finished = rival.additional.SOURCES[index]
        failed.update(replicate_index=index, seed=1300037 + 10007 * index,
                      job_id=worker, finished_utc=finished, selected_replicates=[index])
        with rival.selected_baseline(index):
            report = self.completed_fixture.completed(failed)
        report.update(replicate_index=index, seed=1300037 + 10007 * index, selected_replicates=[index],
                      automatic_retry=False, early_kill="not_evaluated",
                      started_utc="2026-10-05T07:40:00+00:00", finished_utc="2026-10-05T07:56:40+00:00")
        return report, failed

    def arguments(self, output, index=108, model="MC", **changes):
        values = dict(replicate_index=index, model=model, implementation=rival.null.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=rival.MAX_SECONDS.get(model, 2400))
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        panels = (
            (rival.null, ("SOURCE", "SOURCE_REPORT", "SOURCE_JOB_ID", "REPLICATE_INDEX", "SEED",
                          "REPORT_NAME", "validate_source", "write_report")),
            (rival.mvc, ("source_paths", "validate_source", "PHASE", "REPORT_NAME", "PRESERVED_DIRECTORIES", "write_report")),
            (rival.completed, ("COMPLETION_JOB_ID", "COMPLETION_DIRECTORY", "COMPLETION_REPORT", "FAILED_REPORT")),
            (rival.additional.base, ("REPLICATE_INDEX", "SEED", "SOURCE", "SOURCE_JOB_ID",
                                    "SOURCE_FINISHED_UTC", "PHASE", "REPORT_NAME", "completion_context")),
        )
        return tuple(tuple(getattr(module, name) for name in names) for module, names in panels)

    def test_exact_two_completed_identities_and_shared_cost_lineage_are_preserved(self):
        bindings = self.bindings()
        for index in rival.additional.SOURCES:
            report, failed = self.sources(index)
            before = deepcopy((report, failed, self.truth, self.spec))
            lineage = rival.validate_completed_baseline(index, report, failed, self.truth, self.people, self.spec)
            directory, path = rival.source_paths(index)
            self.assertEqual(lineage["original_failed_report"], str(directory / "mz_null_replicate.json"))
            self.assertEqual(lineage["original_worker_job_id"], rival.additional.SOURCES[index][0])
            self.assertEqual(lineage["completion_report"], str(path))
            self.assertEqual(lineage["completion_job_id"], "46591918")
            self.assertEqual(lineage["cumulative_baseline_process_elapsed_seconds"],
                             failed["elapsed_seconds"] + report["elapsed_seconds"])
            self.assertEqual((report, failed, self.truth, self.spec), before)
            self.assertEqual(self.bindings(), bindings)
        for index in (0, 35, 107, 109, 175, 177, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                rival.source_paths(index)

    def test_incomplete_wrong_source_claims_and_changed_preserved_fits_are_rejected(self):
        before = self.bindings()
        for index in rival.additional.SOURCES:
            report, failed = self.sources(index)
            changes = (("job_id", "46586603"), ("phase", rival.additional.base.PHASE),
                       ("replicate_index", 35), ("seed", 1650282), ("source_job_id", "46580478"),
                       ("source_dir", "/wrong"), ("source_finished_utc", "different"),
                       ("status", "running"), ("pairwise_diagnostic_complete", False),
                       ("source_context_validated", False), ("preserved_scores_recomputed", False),
                       ("full_gate", "pass"), ("wave_selected", True), ("automatic_retry", True),
                       ("simulation_copied", True), ("cumulative_process_elapsed_seconds", 1000.))
            for field, value in changes:
                with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                    rival.validate_completed_baseline(index, {**report, field: value}, failed,
                                                      self.truth, self.people, self.spec)
                self.assertEqual(self.bindings(), before)
            for position in range(3):
                for field in ("parameters", "objective_history", "elapsed_seconds"):
                    altered = deepcopy(report)
                    if field == "parameters":
                        altered["fits"][position][field][0][0] += .001
                    elif field == "objective_history":
                        altered["fits"][position][field][0] += .001
                    else:
                        altered["fits"][position][field] += .001
                    with self.subTest(index=index, position=position, field=field), self.assertRaises(ValueError):
                        rival.validate_completed_baseline(index, altered, failed, self.truth, self.people, self.spec)
            for field, value in (("job_id", "46580478"), ("error", "another cause"),
                                 ("finished_utc", "different"), ("fits", failed["fits"][:2])):
                with self.subTest(index=index, failed_field=field), self.assertRaises(ValueError):
                    rival.validate_completed_baseline(index, report, {**failed, field: value},
                                                      self.truth, self.people, self.spec)

    def test_actual_people_null_truth_and_all_four_original500_fits_are_required(self):
        for index in rival.additional.SOURCES:
            report, failed = self.sources(index)
            for position in range(4):
                changed = deepcopy(report)
                changed["fits"][position]["participants"][0] = "different_person"
                with self.subTest(index=index, position=position), self.assertRaises(ValueError):
                    rival.validate_completed_baseline(index, changed, failed, self.truth, self.people, self.spec)
            changed = deepcopy(report)
            changed["fits"][3]["objective_history"] = [1.] * 501
            with self.assertRaises(ValueError):
                rival.validate_completed_baseline(index, changed, failed, self.truth, self.people, self.spec)
            truth = deepcopy(self.truth)
            truth["p000"]["eta"] = .1
            with self.assertRaises(ValueError):
                rival.validate_completed_baseline(index, report, failed, truth, self.people, self.spec)

    def exercise(self, output, index=108, model="MC", *, bad_completion=False,
                 bad_failed=False, replay_error=False, failed_half=None):
        source, failed = self.sources(index)
        if bad_completion:
            source["pairwise_diagnostic_complete"] = False
        if bad_failed:
            failed["job_id"] = "46580478"
        preserved = deepcopy((source, failed, self.truth, self.spec))
        directory, path = rival.source_paths(index)
        failed_path = directory / "mz_null_replicate.json"
        allowed = {rival.null.IMPLEMENTATION: json.dumps(self.spec), path: json.dumps(source),
                   directory / "generating_parameters.json": json.dumps(self.truth), failed_path: json.dumps(failed)}
        engine = rival.mvc if model == "MVC" else rival.null
        original_writer = engine.write_report
        bindings = self.bindings()
        handlers = tuple(signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM))
        calls, reads, scores, checkpoints, alarms = [], [], [], [], []

        def reader(selected, *args, **kwargs):
            self.assertEqual(alarms[-1], rival.MAX_SECONDS[model])
            reads.append(selected)
            return allowed[selected]

        def split(events, train, held):
            self.assertEqual(events, directory / "simulated_events.csv.gz")
            return tuple(train), tuple(held)

        def fitter(training, selected, bound, spec):
            self.assertEqual(alarms[-1], rival.MAX_SECONDS[model])
            calls.append((training, selected, bound, deepcopy(spec)))
            return self.fixture.fit(selected, converged=len(calls) != failed_half)

        def scorer(fit, scoring, spec, check):
            check()
            self.assertEqual(spec, self.spec)
            self.assertEqual(tuple(fit["scoring_runs"]), scoring)
            scores.append((fit["model"], fit["training_runs"]))
            value = {"M0": -1., "MZ": -.99, "MC": -.98, "MV": -.97, "MVZ": -.96, "MVC": -.95}[fit["model"]]
            if replay_error and fit["model"] == "M0":
                value += 1e-9
            return {person: value for person in self.people}

        def writer(destination, report):
            checkpoints.append(deepcopy(report))
            original_writer(destination, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(rival, "require_compute"))
            stack.enter_context(patch.object(engine, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            loader = stack.enter_context(patch.object(rival.null.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(engine, "load_split", side_effect=split))
            stack.enter_context(patch.object(engine, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(rival.null.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(engine, "write_report", side_effect=writer))
            stack.enter_context(patch.object(signal, "alarm", side_effect=lambda seconds: alarms.append(seconds) or 0))
            simulator = stack.enter_context(patch.object(rival.additional.base.indexed.representative, "simulate_null",
                                                        side_effect=AssertionError("No regeneration")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = rival.run(self.arguments(output, index, model))
            self.assertEqual(self.bindings(), installed)
            simulator.assert_not_called()
            loader.assert_called_once()
            self.assertEqual(loader.call_args.args[0], directory / "simulated_events.csv.gz")
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual(tuple(signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)), handlers)
        self.assertEqual(alarms, [rival.MAX_SECONDS[model], 0])
        self.assertEqual((source, failed, self.truth, self.spec), preserved)
        self.assertEqual(reads, [rival.null.IMPLEMENTATION, path, directory / "generating_parameters.json", failed_path])
        self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
        self.assertEqual([entry.name for entry in output.iterdir()], [rival.REPORT_NAME])
        for checkpoint in checkpoints:
            self.assert_scope(checkpoint, index, model)
        return report, calls, scores, checkpoints, source

    def assert_scope(self, report, index, model):
        directory, path = rival.source_paths(index)
        self.assertEqual((report["phase"], report["generating_model"], report["fitted_model"],
                          report["replicate_index"], report["seed"]),
                         (rival.PHASE, "M0", model, index, 1300037 + 10007 * index))
        self.assertEqual(report["source_dir"], str(directory))
        self.assertEqual(report["source_report"], str(path))
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual(report["selected_models"], [model])
        self.assertEqual(report["fixed_cohort_count"], 200)
        self.assertEqual(report["component_scope"], f"{model}_only")
        self.assertEqual(report["baseline_validation_maximum_sweeps"], 500)
        self.assertEqual(report["fitted_maximum_sweeps"], 1000 if model == "MVC" else 500)
        for field in ("wave_selected", "retry_selected", "retry_wave_selected", "full_menu_wave_selected",
                      "automatic_retry", "simulation_regenerated", "simulation_copied", "saved_M0_MZ_refitted", "episode_terminal"):
            self.assertFalse(report[field])
        for field in (*rival.null.CLAIM_FIELDS, "early_kill"):
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])

    def test_all_models_on_both_sources_replay_exact_baseline_then_fit_only_two_fresh_halves(self):
        for index in rival.additional.SOURCES:
            for model in rival.MODELS:
                with self.subTest(index=index, model=model), tempfile.TemporaryDirectory() as directory:
                    report, calls, scores, checkpoints, source = self.exercise(Path(directory) / "new", index, model)
                fit_spec = deepcopy(self.spec)
                if model == "MVC":
                    fit_spec["optimizer"]["maximum_sweeps"] = 1000
                    self.assertEqual(report["runtime_implementation"], fit_spec)
                    self.assertEqual(report["numerical_override"]["runtime_maximum_sweeps"], 1000)
                self.assertEqual(calls, [((1, 4), model, "fixed", fit_spec), ((2, 3), model, "fixed", fit_spec)])
                self.assertEqual(report["implementation"], self.spec)
                self.assertTrue(report["rival_diagnostics_complete"])
                self.assertTrue(report["saved_scores_replayed"])
                self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
                self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
                self.assertEqual(report["source_job_id"], "46591918")
                self.assertEqual(report["source_finished_utc"], source["finished_utc"])
                self.assertEqual(report["baseline_lineage"]["original_worker_job_id"], rival.additional.SOURCES[index][0])
                self.assertNotIn("cumulative_process_elapsed_seconds", report)
                self.assertEqual(report["baseline_lineage"]["cumulative_baseline_process_elapsed_seconds"],
                                 source["cumulative_process_elapsed_seconds"])
                for person, row in source["participant_scores"].items():
                    self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
                self.assertEqual(scores[:4], [(model0, list(train)) for model0, train, held in rival.additional.base.EXPECTED])
                self.assertFalse(checkpoints[0]["source_baseline_validated"])
                self.assertIsNone(checkpoints[0]["source_job_id"])

    def test_invalid_sources_abort_before_replay_or_either_engine_fitting(self):
        for model in ("MC", "MVC"):
            for failure in ("bad_completion", "bad_failed"):
                with self.subTest(model=model, failure=failure), tempfile.TemporaryDirectory() as directory:
                    report, calls, scores, checkpoints, source = self.exercise(Path(directory) / "bad", 176, model, **{failure: True})
                self.assertEqual(calls, [])
                self.assertEqual(scores, [])
                self.assertFalse(report["source_baseline_validated"])
                self.assertFalse(report["saved_scores_replayed"])
                self.assertIsNone(report["source_job_id"])
                self.assertEqual(report["status"], "technical_rival_failure")
                self.assertNotIn("participant_scores", report)

    def test_replay_failure_and_each_half_failure_stay_technical_in_both_engines(self):
        for model in ("MVZ", "MVC"):
            for scenario in ({"replay_error": True}, {"failed_half": 1}, {"failed_half": 2}):
                with self.subTest(model=model, scenario=scenario), tempfile.TemporaryDirectory() as directory:
                    report, calls, scores, checkpoints, source = self.exercise(Path(directory) / "bad", 108, model, **scenario)
                self.assertEqual(len(calls), scenario.get("failed_half", 0))
                self.assertEqual(len(report["fits"]), len(calls))
                self.assertTrue(report["source_baseline_validated"])
                self.assertFalse(report["rival_diagnostics_complete"])
                self.assertEqual(report["status"], "technical_rival_failure")
                self.assertEqual(report["completed_rivals"], [])
                self.assertEqual(report["diagnostics"], {})
                self.assertNotIn("participant_scores", report)

    def test_compute_caps_protected_outputs_and_inherited_guards_restore_all_bindings(self):
        bindings = self.bindings()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No reads before guards")):
            with patch.object(rival, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                rival.run(self.arguments(Path(directory) / "new"))
            with patch.object(rival, "require_compute"), patch.object(rival.null, "require_compute"), patch.object(rival.mvc, "require_compute"):
                for changes in ({"replicate_index": 35}, {"replicate_index": True}, {"model": "MZ"}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "new", **changes))
                for model, cap in rival.MAX_SECONDS.items():
                    for seconds in (0, True, cap + 1):
                        with self.subTest(model=model, seconds=seconds), self.assertRaises(ValueError):
                            rival.run(self.arguments(Path(directory) / "new", model=model, max_seconds=seconds))
                    with self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "new", model=model, implementation=Path(directory) / "wrong.json"))
                    for name in ("SLURM_CPUS_PER_TASK", *THREADS):
                        with self.subTest(model=model, name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                            rival.run(self.arguments(Path(directory) / "new", model=model))
                    for output in (Path(directory), *(path / "new" for path in rival.PRESERVED_DIRECTORIES)):
                        with self.subTest(model=model, output=output), self.assertRaises(FileExistsError):
                            rival.run(self.arguments(output, model=model))
                    self.assertEqual(self.bindings(), bindings)


if __name__ == "__main__":
    unittest.main()
