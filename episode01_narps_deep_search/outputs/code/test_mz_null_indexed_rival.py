"""Focused indexed MVZ source/replay/checkpoint tests; execute on compute only."""
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

import run_mz_null_indexed_rival as rival
import test_mz_null_baseline_wave as fixtures


THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
SOURCE_WORKER = "70000013"


class IndexedNullRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullBaselineWaveTests.setUpClass()
        cls.source_fixture = fixtures.NullBaselineWaveTests("test_exact_source_mapping_preserves_pilot_and_two_array_ranges")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def arguments(self, output, index=1, **changes):
        values = dict(model="MVZ", replicate_index=index, source_dir=rival.source_directory(index),
                      implementation=rival.null.IMPLEMENTATION, output_dir=Path(output), max_seconds=3300)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        module = rival.null
        return tuple(getattr(module, name) for name in ("SOURCE", "SOURCE_REPORT", "SOURCE_JOB_ID", "SEED",
                     "REPLICATE_INDEX", "REPORT_NAME", "validate_source", "write_report"))

    def test_fixed_array_sources_model_and_preserved_pilots_are_enforced(self):
        for index in (1, 4, 5, 199):
            array = 46579924 if index <= 4 else 46580387
            self.assertEqual(rival.source_directory(index), rival.baseline.RUNS /
                             f"20261005-M0-null-baseline-array-{array}" / f"replicate{index}")
        for invalid in (0, -1, 200, True):
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                rival.source_directory(invalid)
        with tempfile.TemporaryDirectory() as directory, patch.object(rival, "require_compute"), \
                patch.object(rival.null, "run") as core:
            for changes in ({"model": "MC"}, {"model": "MVC"}, {"source_dir": rival.source_directory(5)},
                            {"source_dir": rival.indexed.PILOT_DIRECTORY}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh", **changes))
            for preserved in rival.PRESERVED_DIRECTORIES:
                with self.subTest(path=preserved), self.assertRaises(FileExistsError):
                    rival.run(self.arguments(preserved / "new-child"))
            core.assert_not_called()

    def test_inherited_implementation_cpu_thread_cap_and_fresh_guards_restore_bindings(self):
        original = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(rival, "require_compute"), \
                patch.object(rival.null, "require_compute"), \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No input read")):
            for changes in ({"implementation": Path(directory) / "changed.json"}, {"max_seconds": 0},
                            {"max_seconds": 5401}, {"max_seconds": True}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh", **changes))
                self.assertEqual(self.bindings(), original)
            for name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
                with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "fresh"))
            with self.assertRaises(FileExistsError):
                rival.run(self.arguments(Path(directory)))
        self.assertEqual(self.bindings(), original)

    def exercise(self, output, index=1, *, source_change=None, truth_change=None,
                 replay_error=False, fail_second=False, missing_source=False, changed_ceiling=False):
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
        directory = rival.source_directory(index)
        source_path = directory / rival.indexed.REPORT_NAME
        allowed = {module.IMPLEMENTATION: json.dumps(spec), source_path: json.dumps(source),
                   directory / "generating_parameters.json": json.dumps(truth)}
        calls, checkpoints, reads = [], [], []
        original = self.bindings()
        actual_writer = module.write_report

        def reader(path, *args, **kwargs):
            reads.append(path)
            if missing_source and path == source_path:
                raise FileNotFoundError(path)
            return allowed[path]

        def fitter(training, model, bound, implementation):
            calls.append((model, bound, deepcopy(implementation)))
            return self.fixture.fit(model, converged=not (fail_second and len(calls) == 2))

        def scorer(fit, scoring, implementation, check):
            check()
            value = {"M0": -1., "MZ": -.99, "MVZ": -.98}[fit["model"]]
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
            stack.enter_context(patch.object(module, "load_split", return_value=({}, {})))
            stack.enter_context(patch.object(module, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(module.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(module, "write_report", side_effect=writer))
            validator = stack.enter_context(patch.object(rival.baseline, "completed_diagnostic",
                                                          wraps=rival.baseline.completed_diagnostic))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = rival.run(self.arguments(output, index))
            self.assertEqual(self.bindings(), installed)
        self.assertEqual(self.bindings(), original)
        return report, calls, checkpoints, source, reads, validator

    def assert_scope(self, report, index):
        self.assertEqual((report["phase"], report["generating_model"], report["fitted_model"]),
                         (rival.PHASE, "M0", "MVZ"))
        self.assertEqual((report["replicate_index"], report["seed"]), (index, rival.indexed.null_seed(index)))
        self.assertEqual(report["source_dir"], str(rival.source_directory(index)))
        self.assertEqual(report["source_report"], "mz_null_replicate.json")
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual(report["selected_models"], ["MVZ"])
        self.assertEqual(report["component_scope"], "MVZ_only")
        self.assertEqual(report["fixed_cohort_count"], 200)
        self.assertFalse(report["simulation_regenerated"])
        self.assertFalse(report["saved_M0_MZ_refitted"])
        self.assertTrue(report["wave_selected"])
        self.assertEqual(report["wave_scope"], "MVZ_component_only")
        self.assertFalse(report["full_menu_wave_selected"])
        self.assertFalse(report["retry_selected"])
        for field in rival.null.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertFalse(report["episode_terminal"])
        self.assertNotIn("counts_as_curtailment_failure", report)
        self.assertNotIn("core_family_selection_pass", report)

    def test_both_array_ranges_validate_real_index_then_replay_and_fit_only_two_mvz_halves(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (4, 5):
                output = Path(directory) / f"index{index}"
                report, calls, checkpoints, source, reads, validator = self.exercise(output, index)
                validator.assert_called_once()
                self.assertEqual(validator.call_args.args[3], index)
                self.assertEqual(validator.call_args.args[0]["phase"], rival.indexed.PHASE)
                self.assertEqual([model for model, bound, spec in calls], ["MVZ", "MVZ"])
                self.assertTrue(all(bound == "fixed" and spec == self.spec for model, bound, spec in calls))
                self.assertTrue(report["rival_diagnostics_complete"])
                self.assertTrue(report["saved_scores_replayed"])
                self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
                self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
                self.assertEqual(report["source_finished_utc"], source["finished_utc"])
                self.assertEqual(report["source_job_id"], SOURCE_WORKER)
                self.assertNotIn(int(report["source_job_id"]), (rival.FIRST_ARRAY, rival.REMAINING_ARRAY))
                self.assertEqual(report["completed_rivals"], ["MVZ"])
                self.assertEqual([(fit["training_runs"], fit["scoring_runs"]) for fit in report["fits"]],
                                 [(list(train), list(held)) for train, held in rival.null.HALVES])
                for person, row in source["participant_scores"].items():
                    self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
                self.assertIsNone(checkpoints[0]["source_job_id"])
                self.assertFalse(checkpoints[0]["source_baseline_validated"])
                self.assertTrue(report["source_baseline_validated"])
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, index)
                self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
                self.assertFalse((output / "simulated_events.csv.gz").exists())
                self.assertFalse((output / "generating_parameters.json").exists())

    def test_incomplete_wrong_index_or_nonnull_source_retains_failure_before_fitting(self):
        scenarios = (
            {"source_change": lambda source: source.update(replicate_index=2)},
            {"source_change": lambda source: source.update(phase="M0_null_MZ_pairwise_representative")},
            {"source_change": lambda source: source.update(status="technical_representative_failure", pairwise_diagnostic_complete=False)},
            {"source_change": lambda source: source.update(fits=[])},
            {"truth_change": lambda truth: truth["p000"].update(rho=.1)},
            {"changed_ceiling": True}, {"missing_source": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            for number, scenario in enumerate(scenarios):
                output = Path(directory) / f"failure{number}"
                report, calls, checkpoints, source, reads, validator = self.exercise(output, **scenario)
                self.assertEqual(calls, [])
                self.assertEqual(report["status"], "technical_rival_failure")
                self.assertFalse(report["source_baseline_validated"])
                self.assertIsNone(report["source_job_id"])
                self.assertFalse(report["saved_scores_replayed"])
                self.assertFalse(report["rival_diagnostics_complete"])
                self.assertEqual(report["fits"], [])
                self.assertNotIn("participant_scores", report)
                self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, 1)

    def test_replay_failure_retains_validated_worker_id_but_stops_before_new_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checkpoints, source, reads, validator = self.exercise(Path(directory) / "replay", replay_error=True)
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

    def test_second_half_failure_preserves_returned_fits_without_completed_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fit-failure"
            report, calls, checkpoints, source, reads, validator = self.exercise(output, fail_second=True)
            self.assertEqual(len(calls), 2)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertTrue(report["saved_scores_replayed"])
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertTrue(report["fits"][0]["converged"])
            self.assertFalse(report["fits"][1]["converged"])
            self.assertEqual(len(report["scored_primary_directions"]), 1)
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertNotIn("participant_scores", report)
            self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, 1)


if __name__ == "__main__":
    unittest.main()
