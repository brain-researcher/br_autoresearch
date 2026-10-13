"""Focused indexed-null adapter checks; execute only in a compute allocation."""
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

import numpy as np

import run_mz_null_replicate as indexed
import test_mz_null_rival as fixtures


class NullReplicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.source = cls.fixture.spec, cls.fixture.people

    def arguments(self, output, index=1, **changes):
        values = dict(replicate_index=index, design_events=indexed.representative.DESIGN_SOURCE,
                      implementation=indexed.representative.IMPLEMENTATION, output_dir=Path(output), max_seconds=2700)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        module = indexed.representative
        return module.SEED, module.REPLICATE_INDEX, module.REPORT_NAME, module.write_report

    def test_fixed_seed_schedule_is_unique_and_preserves_pilot_zero(self):
        seeds = [indexed.null_seed(index) for index in range(200)]
        self.assertEqual(len(set(seeds)), 200)
        self.assertEqual((seeds[0], seeds[1], seeds[199]), (1300037, 1310044, 3291430))
        self.assertTrue(all(right - left == 10007 for left, right in zip(seeds, seeds[1:])))
        self.assertEqual(indexed.representative.SEED, seeds[0])
        self.assertEqual(indexed.representative.REPLICATE_INDEX, 0)
        for invalid in (-1, 200, True, 1.0, "1"):
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                indexed.null_seed(invalid)

    def test_index_cpu_and_preserved_pilot_guards_precede_core_execution(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(indexed, "require_compute"), \
                patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(indexed.representative, "run") as core:
            for invalid in (0, -1, 200, True):
                with self.subTest(index=invalid), self.assertRaises(ValueError):
                    indexed.run(self.arguments(Path(directory) / "fresh", invalid))
            for cpus in ("", "2"):
                with self.subTest(cpus=cpus), patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": cpus}), self.assertRaises(ValueError):
                    indexed.run(self.arguments(Path(directory) / "fresh"))
            with self.assertRaises(FileExistsError):
                indexed.run(self.arguments(indexed.PILOT_DIRECTORY / "new-child"))
            core.assert_not_called()

    def test_original_path_cap_thread_and_fresh_output_guards_remain_and_restore_bindings(self):
        module = indexed.representative
        original = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(indexed, "require_compute"), \
                patch.object(module, "require_compute"), patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(module, "load_design", side_effect=AssertionError("No design payload read")) as loader:
            for change in ({"design_events": Path(directory) / "wrong.csv.gz"},
                           {"implementation": Path(directory) / "wrong.json"},
                           {"max_seconds": 0}, {"max_seconds": 5401}, {"max_seconds": True}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    indexed.run(self.arguments(Path(directory) / "fresh", **change))
                self.assertEqual(self.bindings(), original)
            for name in fixtures.THREADS:
                with self.subTest(thread=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                    indexed.run(self.arguments(Path(directory) / "fresh"))
            with self.assertRaises(FileExistsError):
                indexed.run(self.arguments(Path(directory)))
            with self.assertRaises(FileExistsError):
                indexed.run(self.arguments(module.DESIGN_SOURCE.parent / "forbidden-new-child"))
            loader.assert_not_called()
        self.assertEqual(self.bindings(), original)

    def exercise(self, output, index, *, fail_second=False):
        module = indexed.representative
        fit_calls, checkpoints = [], []
        original = self.bindings()
        actual_writer, actual_rng = module.write_report, np.random.default_rng

        def fitter(training, model, rival, spec):
            fit_calls.append((model, rival, deepcopy(spec)))
            return self.fixture.fit(model, converged=not (fail_second and len(fit_calls) == 2))

        def simulator(rng, drift, *args, **kwargs):
            return np.ones(len(drift), dtype=int), np.full(len(drift), .5)

        def scorer(fit, scoring, spec, check):
            check()
            return {person: -1. if fit["model"] == "M0" else -.99 for person in self.source}

        def writer(directory, report):
            checkpoints.append(deepcopy(report))
            return actual_writer(directory, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(indexed, "require_compute"))
            stack.enter_context(patch.object(module, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(module, "load_design", return_value=self.source))
            stack.enter_context(patch.object(module.deadline_ddm, "simulate", side_effect=simulator))
            stack.enter_context(patch.object(module, "load_split", return_value=(self.source, self.source)))
            stack.enter_context(patch.object(module, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(module, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(module, "write_report", side_effect=writer))
            rng = stack.enter_context(patch.object(np.random, "default_rng", wraps=actual_rng))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = indexed.run(self.arguments(output, index))
            self.assertEqual(self.bindings(), installed)
            rng.assert_called_once_with(indexed.null_seed(index))
        self.assertEqual(self.bindings(), original)
        return report, fit_calls, checkpoints

    def assert_scope(self, report, index):
        self.assertEqual((report["phase"], report["replicate_index"], report["seed"]),
                         (indexed.PHASE, index, indexed.null_seed(index)))
        self.assertEqual(report["generating_model"], "M0")
        self.assertEqual(report["focal_generating_truth"], {"eta": 0., "rho": 0., "criterion": 0.})
        self.assertEqual(report["implementation"], self.spec)
        self.assertEqual(report["implementation"]["optimizer"]["maximum_sweeps"], 500)
        self.assertTrue(report["wave_selected"])
        self.assertEqual(report["wave_scope"], "M0_MZ_baseline_only")
        self.assertEqual(report["fixed_cohort_count"], 200)
        self.assertEqual(report["selected_replicates"], [index])
        self.assertFalse(report["full_menu_wave_selected"])
        self.assertFalse(report["retry_selected"])
        for field in ("family_selection", "full_menu_false_selection", "false_selection_rate",
                      "confusion_rate", "group_recovery", "interval_coverage", "full_gate"):
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertFalse(report["episode_terminal"])
        self.assertNotIn("counts_as_curtailment_failure", report)
        self.assertNotIn("core_family_selection_pass", report)

    def test_original_simulator_uses_each_index_seed_and_preserves_four_fit_spec_and_every_checkpoint(self):
        truths = []
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 199):
                output = Path(directory) / f"replicate{index}"
                report, calls, checkpoints = self.exercise(output, index)
                self.assertEqual([model for model, rival, spec in calls], ["M0", "MZ", "M0", "MZ"])
                self.assertTrue(all(rival == "fixed" and spec == self.spec for model, rival, spec in calls))
                self.assertTrue(report["pairwise_diagnostic_complete"])
                self.assertTrue(report["nested_comparison"]["qualifies_nested_comparison"])
                self.assertEqual([(fit["model"], fit["training_runs"], fit["scoring_runs"]) for fit in report["fits"]],
                                 [(model, list(train), list(held)) for train, held in indexed.representative.HALVES
                                  for model in ("M0", "MZ")])
                self.assertGreater(len(checkpoints), 4)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, index)
                self.assertEqual(json.loads((output / indexed.REPORT_NAME).read_text()), report)
                self.assertFalse((output / "mz_null_representative.json").exists())
                truth = json.loads((output / "generating_parameters.json").read_text())
                self.assertTrue(all(params[key] == 0. for params in truth.values() for key in ("eta", "rho", "criterion")))
                truths.append(truth)
        self.assertNotEqual(truths[0], truths[1])

    def test_technical_failure_keeps_simulation_and_returned_fits_without_retry_or_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            report, calls, checkpoints = self.exercise(output, 42, fail_second=True)
            self.assertEqual([model for model, rival, spec in calls], ["M0", "MZ"])
            self.assertEqual(report["status"], "technical_representative_failure")
            self.assertFalse(report["pairwise_diagnostic_complete"])
            self.assertEqual(len(report["fits"]), 2)
            self.assertFalse(report["fits"][-1]["converged"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_comparison", report)
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, 42)
            self.assertTrue((output / "simulated_events.csv.gz").exists())
            self.assertTrue((output / "generating_parameters.json").exists())
            self.assertEqual(json.loads((output / indexed.REPORT_NAME).read_text()), report)


if __name__ == "__main__":
    unittest.main()
