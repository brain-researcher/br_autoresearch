"""Focused ceiling-only adapter and failure-accounting tests; run on compute."""
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

import run_mz_mvc_convergence_pilot as pilot
import test_mz_null_rival as fixtures


class ConvergencePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def failed(self):
        return {
            "analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_saved_replicate_rival_diagnostics",
            "generating_model": "MZ", "fitted_model": "MVC", "replicate_index": 42,
            "seed": pilot.saved.cell_seed("MZ", 42), "source_dir": str(pilot.saved.source_directory(42)),
            "implementation": self.spec, "status": "technical_rival_failure", "rival_diagnostics_complete": False,
            "scored_primary_directions": [], "completed_rivals": [], "elapsed_seconds": 1661.5854664351791,
            "fits": [{"model": "MVC", "rival": "fixed", "training_runs": [1, 4], "scoring_runs": [2, 3],
                      "converged": False, "subject_optimizer_failures": [], "objective_history": list(range(500, 0, -1)),
                      "elapsed_seconds": 1659.960657726042, "parameters": "MUST_NOT_BE_REUSED"}],
        }

    def args(self, output, **changes):
        values = dict(implementation=pilot.saved.IMPLEMENTATION, output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_runtime_copy_changes_only_the_recorded_sweep_ceiling(self):
        original = deepcopy(self.spec)
        runtime, metadata = pilot.repair_context(original, self.failed())
        self.assertEqual(original, self.spec)
        self.assertEqual(runtime["optimizer"]["maximum_sweeps"], 1000)
        runtime["optimizer"]["maximum_sweeps"] = 500
        self.assertEqual(runtime, original)
        self.assertFalse(metadata["nonconverged_checkpoint_state_reused"])
        self.assertEqual(metadata["previous_process_elapsed_seconds"], self.failed()["elapsed_seconds"])

    def test_wrong_attempt_or_different_failure_is_rejected(self):
        for field, value in (("replicate_index", 41), ("seed", 910031), ("fitted_model", "MVZ"),
                             ("source_dir", "/different/source"), ("status", "running")):
            failed = self.failed()
            failed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pilot.repair_context(self.spec, failed)
        for field, value in (("converged", True), ("objective_history", [1.] * 499),
                             ("subject_optimizer_failures", [{"participant": "p000", "message": "failure"}])):
            failed = self.failed()
            failed["fits"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pilot.repair_context(self.spec, failed)

    def exercise(self, output, *, fail_first=False):
        saved = pilot.saved
        base = self.fixture.source()
        allowed = {
            saved.IMPLEMENTATION: json.dumps(self.spec), pilot.FAILED_REPORT: json.dumps(self.failed()),
            saved.source_directory(42) / "single_process_recovery.json": json.dumps(base),
            saved.source_directory(42) / "generating_parameters.json": json.dumps(self.fixture.truth),
        }
        fit_calls, validations, checkpoints = [], [], []
        actual_validator, actual_writer = saved.validate_fit, saved.write_report
        original_references = (saved.fit_partially_pooled, saved.validate_fit, saved.write_report, saved.REPORT_NAME)

        def validate_source(source, truth, people, spec, index):
            self.assertEqual(index, 42)
            self.assertEqual(spec, self.spec)
            for fit in source["fits"]:
                saved.validate_fit(fit, fit["model"], fit["training_runs"], fit["scoring_runs"], spec, people)

        def fitter(training, model, rival, spec):
            fit_calls.append((model, rival, deepcopy(spec)))
            fit = self.fixture.fit(model, converged=not fail_first)
            count = 1000 if fail_first else 600
            fit["objective_history"] = [float(value) for value in range(count, 1, -1)] + [1. if fail_first else 2.]
            return fit

        def validator(fit, model, train, held, spec, people):
            validations.append((model, spec["optimizer"]["maximum_sweeps"]))
            return actual_validator(fit, model, train, held, spec, people)

        def writer(directory, report):
            checkpoints.append(deepcopy(report))
            return actual_writer(directory, report)

        def scorer(fit, scoring, spec, check):
            check()
            self.assertEqual(spec, self.spec)
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            return {person: value for person in self.people}

        with ExitStack() as stack:
            stack.enter_context(patch.object(pilot, "require_compute"))
            stack.enter_context(patch.object(saved, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", lambda path, *args, **kwargs: allowed[path]))
            stack.enter_context(patch.object(saved.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(saved, "validate_source", side_effect=validate_source))
            stack.enter_context(patch.object(saved, "load_split", return_value=({}, {})))
            stack.enter_context(patch.object(saved, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(saved, "validate_fit", side_effect=validator))
            stack.enter_context(patch.object(saved, "write_report", side_effect=writer))
            stack.enter_context(patch.object(saved.shared, "score_fit", side_effect=scorer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = (saved.fit_partially_pooled, saved.validate_fit, saved.write_report, saved.REPORT_NAME)
            report = pilot.run(self.args(output))
            self.assertEqual((saved.fit_partially_pooled, saved.validate_fit, saved.write_report, saved.REPORT_NAME), installed)
        self.assertEqual((saved.fit_partially_pooled, saved.validate_fit, saved.write_report, saved.REPORT_NAME), original_references)
        return report, fit_calls, validations, checkpoints, base

    def test_original_baseline_validation_and_replay_runtime_mvc_and_every_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "success"
            report, calls, validations, checkpoints, base = self.exercise(output)
            self.assertEqual([model for model, bound, spec in calls], ["MVC", "MVC"])
            for model, bound, spec in calls:
                self.assertEqual(bound, "fixed")
                self.assertEqual(spec["optimizer"]["maximum_sweeps"], 1000)
                expected = deepcopy(self.spec)
                expected["optimizer"]["maximum_sweeps"] = 1000
                self.assertEqual(spec, expected)
            self.assertEqual(validations, [("M0", 500), ("MZ", 500), ("M0", 500), ("MZ", 500),
                                          ("MVC", 1000), ("MVC", 1000)])
            self.assertTrue(report["rival_diagnostics_complete"])
            self.assertEqual(report["phase"], pilot.PHASE)
            self.assertGreater(len(checkpoints), 4)
            for checkpoint in checkpoints:
                self.assertEqual(checkpoint["implementation"], self.spec)
                self.assertEqual(checkpoint["runtime_implementation"]["optimizer"]["maximum_sweeps"], 1000)
                self.assertEqual(checkpoint["numerical_repair"]["original_maximum_sweeps"], 500)
                self.assertFalse(checkpoint["retry_wave_selected"])
            for person, row in base["participant_scores"].items():
                self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
            self.assertEqual(report["cumulative_process_elapsed_seconds"],
                             self.failed()["elapsed_seconds"] + report["elapsed_seconds"])
            self.assertEqual(json.loads((output / pilot.REPORT_NAME).read_text()), report)
            self.assertIsNone(report["full_recovery_pass"])
            self.assertEqual(report["family_selection"], "not_evaluated")

    def test_first_half_nonconvergence_preserves_failure_and_prevents_second_fit_or_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            report, calls, validations, checkpoints, base = self.exercise(output, fail_first=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertEqual(len(report["fits"]), 1)
            self.assertEqual(len(report["fits"][0]["objective_history"]), 1000)
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertEqual(report["scored_primary_directions"], [])
            self.assertEqual(json.loads((output / pilot.REPORT_NAME).read_text()), report)
            self.assertTrue(all("numerical_repair" in item for item in checkpoints))

    def test_scope_and_cpu_guards_run_before_metadata_reads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "require_compute"), \
                patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read is allowed")):
            for changes in ({"implementation": Path(directory) / "changed.json"}, {"max_seconds": 5401}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    pilot.run(self.args(Path(directory) / "fresh", **changes))
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), self.assertRaises(ValueError):
                pilot.run(self.args(Path(directory) / "fresh"))
            with self.assertRaises(FileExistsError):
                pilot.run(self.args(Path(directory)))


if __name__ == "__main__":
    unittest.main()
