"""Focused saved-null MC/MV adapter checks; run only on coordinated compute."""
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

import run_mz_null_indexed_simple_rival as rival
import test_mz_null_baseline_wave as fixtures
from test_mz_null_representative import THREADS


SOURCE_WORKER = "70000013"


class IndexedSimpleNullRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullBaselineWaveTests.setUpClass()
        cls.source_fixture = fixtures.NullBaselineWaveTests("test_exact_source_mapping_preserves_pilot_and_two_array_ranges")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def arguments(self, output, model="MC", index=1, **changes):
        values = dict(model=model, replicate_index=index, source_dir=rival.source_directory(index),
                      implementation=rival.null.IMPLEMENTATION, output_dir=Path(output), max_seconds=2400)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return tuple(getattr(rival.null, name) for name in
                     ("SOURCE", "SOURCE_REPORT", "SOURCE_JOB_ID", "SEED", "REPLICATE_INDEX",
                      "REPORT_NAME", "validate_source", "write_report"))

    def exercise(self, output, model="MC", index=1, *, source_change=None, truth_change=None,
                 replay_error=False, failed_half=None, missing_source=False, changed_ceiling=False):
        module = rival.null
        source = self.source_fixture.report(index)
        source["job_id"] = SOURCE_WORKER
        truth, spec = deepcopy(self.truth), deepcopy(self.spec)
        if source_change:
            source_change(source)
        if truth_change:
            truth_change(truth)
        if changed_ceiling:
            spec["optimizer"]["maximum_sweeps"] = 1000
            source["implementation"] = spec
        preserved = deepcopy((source, truth, spec))
        directory = rival.source_directory(index)
        source_path = directory / rival.indexed.REPORT_NAME
        allowed = {module.IMPLEMENTATION: json.dumps(spec), source_path: json.dumps(source),
                   directory / "generating_parameters.json": json.dumps(truth)}
        calls, checkpoints, reads, decoded = [], [], [], []
        original_bindings = self.bindings()
        actual_writer = module.write_report
        actual_validator = rival.baseline.completed_diagnostic

        def reader(path, *args, **kwargs):
            reads.append(path)
            if missing_source and path == source_path:
                raise FileNotFoundError(path)
            return allowed[path]

        def validate(report, source_truth, people, source_index, implementation):
            decoded.append((report, source_truth, implementation))
            return actual_validator(report, source_truth, people, source_index, implementation)

        def fitter(training, selected_model, bound, implementation):
            calls.append((training, selected_model, bound, deepcopy(implementation)))
            return self.fixture.fit(selected_model, converged=len(calls) != failed_half)

        def scorer(fit, scoring, implementation, check):
            check()
            self.assertEqual(implementation, spec)
            value = {"M0": -1., "MZ": -.99, "MC": -.98, "MV": -.97}[fit["model"]]
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
            stack.enter_context(patch.object(module.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(module, "load_split", side_effect=lambda events, train, held: (tuple(train), tuple(held))))
            stack.enter_context(patch.object(module, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(module.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(module, "write_report", side_effect=writer))
            validator = stack.enter_context(patch.object(rival.baseline, "completed_diagnostic", side_effect=validate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = rival.run(self.arguments(output, model, index))
            self.assertEqual(self.bindings(), installed)
        self.assertEqual(self.bindings(), original_bindings)
        self.assertEqual((source, truth, spec), preserved)
        for inputs in decoded:
            self.assertEqual(inputs, preserved)
        self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
        return report, calls, checkpoints, source, reads, validator

    def assert_scope(self, report, model, index):
        self.assertEqual((report["phase"], report["generating_model"], report["fitted_model"]),
                         (rival.PHASE, "M0", model))
        self.assertEqual((report["replicate_index"], report["seed"]), (index, 1300037 + 10007 * index))
        self.assertEqual(report["source_dir"], str(rival.source_directory(index)))
        self.assertEqual(report["source_report"], "mz_null_replicate.json")
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual(report["selected_models"], [model])
        self.assertEqual(report["component_scope"], f"{model}_only")
        self.assertEqual(report["wave_scope"], f"{model}_component_only")
        self.assertTrue(report["wave_selected"])
        self.assertEqual(report["fixed_cohort_count"], 200)
        for field in ("full_menu_wave_selected", "retry_selected", "simulation_regenerated",
                      "saved_M0_MZ_refitted", "episode_terminal"):
            self.assertFalse(report[field])
        for field in rival.null.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertNotIn("counts_as_curtailment_failure", report)
        self.assertNotIn("core_family_selection_pass", report)

    def test_exact_source_mapping_model_scope_and_original_representatives_are_protected(self):
        for index in (1, 4, 5, 199):
            array = 46579924 if index <= 4 else 46580387
            self.assertEqual(rival.source_directory(index), rival.baseline.RUNS /
                             f"20261005-M0-null-baseline-array-{array}" / f"replicate{index}")
        for invalid in (0, -1, 200, True):
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                rival.source_directory(invalid)
        with tempfile.TemporaryDirectory() as directory, patch.object(rival, "require_compute"), \
                patch.object(rival.null, "run") as core:
            for model in ("M0", "MZ", "MVZ", "MVC"):
                with self.subTest(model=model), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh", model))
            for source in (rival.source_directory(5), rival.indexed.PILOT_DIRECTORY):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh", source_dir=source))
            for preserved in rival.PRESERVED_DIRECTORIES:
                with self.subTest(path=preserved), self.assertRaises(FileExistsError):
                    rival.run(self.arguments(preserved / "new-child"))
            core.assert_not_called()

    def test_both_models_in_both_array_ranges_replay_exact_baselines_then_fit_two_original_halves(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in rival.MODELS:
                for index in (4, 5):
                    with self.subTest(model=model, index=index):
                        output = Path(directory) / f"{model}-{index}"
                        report, calls, checkpoints, source, reads, validator = self.exercise(output, model, index)
                        validator.assert_called_once()
                        self.assertEqual(validator.call_args.args[3], index)
                        self.assertEqual(validator.call_args.args[0]["phase"], rival.indexed.PHASE)
                        self.assertEqual([(train, fitted, bound) for train, fitted, bound, spec in calls],
                                         [((1, 4), model, "fixed"), ((2, 3), model, "fixed")])
                        self.assertTrue(all(spec == self.spec and spec["optimizer"]["maximum_sweeps"] == 500
                                            for train, fitted, bound, spec in calls))
                        self.assertTrue(report["rival_diagnostics_complete"])
                        self.assertTrue(report["saved_scores_replayed"])
                        self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
                        self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
                        self.assertEqual(report["source_finished_utc"], source["finished_utc"])
                        self.assertEqual(report["source_job_id"], SOURCE_WORKER)
                        self.assertNotIn(int(report["source_job_id"]), (rival.FIRST_ARRAY, rival.REMAINING_ARRAY))
                        self.assertEqual(report["completed_rivals"], [model])
                        self.assertEqual([(fit["training_runs"], fit["scoring_runs"]) for fit in report["fits"]],
                                         [(list(train), list(held)) for train, held in rival.null.HALVES])
                        for person, row in source["participant_scores"].items():
                            self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
                        self.assertIsNone(checkpoints[0]["source_job_id"])
                        self.assertFalse(checkpoints[0]["source_baseline_validated"])
                        self.assertTrue(report["source_baseline_validated"])
                        for checkpoint in checkpoints:
                            self.assert_scope(checkpoint, model, index)
                        self.assertEqual(reads, [rival.null.IMPLEMENTATION,
                            rival.source_directory(index) / rival.indexed.REPORT_NAME,
                            rival.source_directory(index) / "generating_parameters.json"])
                        self.assertFalse((output / "simulated_events.csv.gz").exists())
                        self.assertFalse((output / "generating_parameters.json").exists())

    def test_wrong_or_incomplete_source_and_changed_truth_or_spec_preserve_technical_failure(self):
        scenarios = (
            {"source_change": lambda source: source.update(replicate_index=2)},
            {"source_change": lambda source: source.update(seed=1300037)},
            {"source_change": lambda source: source.update(phase="M0_null_MZ_pairwise_representative")},
            {"source_change": lambda source: source.update(status="technical_representative_failure", pairwise_diagnostic_complete=False)},
            {"source_change": lambda source: source.update(fits=[])},
            {"truth_change": lambda truth: truth["p000"].update(rho=.1)},
            {"changed_ceiling": True}, {"missing_source": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            for number, scenario in enumerate(scenarios):
                model = rival.MODELS[number % 2]
                with self.subTest(model=model, scenario=number):
                    report, calls, checkpoints, source, reads, validator = self.exercise(
                        Path(directory) / f"source{number}", model, **scenario)
                    self.assertEqual(calls, [])
                    self.assertEqual(report["status"], "technical_rival_failure")
                    self.assertFalse(report["source_baseline_validated"])
                    self.assertIsNone(report["source_job_id"])
                    self.assertFalse(report["saved_scores_replayed"])
                    self.assertEqual(report["fits"], [])
                    self.assertNotIn("participant_scores", report)
                    for checkpoint in checkpoints:
                        self.assert_scope(checkpoint, model, 1)

    def test_each_models_replay_failure_keeps_source_worker_but_prevents_new_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in rival.MODELS:
                report, calls, checkpoints, source, reads, validator = self.exercise(
                    Path(directory) / model, model, replay_error=True)
                self.assertEqual(calls, [])
                self.assertEqual(report["status"], "technical_rival_failure")
                self.assertTrue(report["source_baseline_validated"])
                self.assertEqual(report["source_job_id"], SOURCE_WORKER)
                self.assertFalse(report["saved_scores_replayed"])
                self.assertEqual(report["fits"], [])
                self.assertEqual(report["completed_rivals"], [])
                self.assertEqual(report["diagnostics"], {})
                self.assertNotIn("participant_scores", report)
                self.assertIn("replay tolerance", report["error"])
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, model, 1)

    def test_each_models_first_or_second_half_failure_keeps_checkpoints_without_final_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in rival.MODELS:
                for half in (1, 2):
                    with self.subTest(model=model, failed_half=half):
                        report, calls, checkpoints, source, reads, validator = self.exercise(
                            Path(directory) / f"{model}-{half}", model, failed_half=half)
                        self.assertEqual(len(calls), half)
                        self.assertEqual(report["status"], "technical_rival_failure")
                        self.assertTrue(report["saved_scores_replayed"])
                        self.assertFalse(report["rival_diagnostics_complete"])
                        self.assertEqual(len(report["fits"]), half)
                        self.assertFalse(report["fits"][-1]["converged"])
                        self.assertEqual(len(report["scored_primary_directions"]), half - 1)
                        self.assertEqual(report["completed_rivals"], [])
                        self.assertEqual(report["diagnostics"], {})
                        self.assertNotIn("participant_scores", report)
                        for checkpoint in checkpoints:
                            self.assert_scope(checkpoint, model, 1)

    def test_compute_and_inherited_cpu_threads_implementation_time_fresh_guards_restore_bindings(self):
        original = self.bindings()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No input read")):
            with patch.object(rival, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                rival.run(self.arguments(Path(directory) / "fresh"))
            with patch.object(rival, "require_compute"), patch.object(rival.null, "require_compute"):
                for changes in ({"implementation": Path(directory) / "changed.json"}, {"max_seconds": 0},
                                {"max_seconds": 5401}, {"max_seconds": True}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "fresh", **changes))
                    self.assertEqual(self.bindings(), original)
                for name in ("SLURM_CPUS_PER_TASK", *THREADS):
                    with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                        rival.run(self.arguments(Path(directory) / "fresh"))
                    self.assertEqual(self.bindings(), original)
                with self.assertRaises(FileExistsError):
                    rival.run(self.arguments(Path(directory)))
        self.assertEqual(self.bindings(), original)


if __name__ == "__main__":
    unittest.main()
