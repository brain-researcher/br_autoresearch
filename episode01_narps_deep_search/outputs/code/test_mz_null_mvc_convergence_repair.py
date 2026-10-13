"""Focused single null0 MVC half-reuse checks; execute only on compute."""
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

import run_mz_null_mvc_convergence_repair as repair
import test_mz_null_rival as fixtures


class NullMVCRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def failed(self, source=None):
        source = self.fixture.source() if source is None else source
        fits = [self.fixture.fit("MVC", train, held) for train, held in repair.null.HALVES]
        fits[0].update(elapsed_seconds=1400., objective_history=[float(i) for i in range(364, 0, -1)] + [1.])
        fits[1].update(elapsed_seconds=2100., converged=False, objective_history=list(range(500, 0, -1)))
        return {
            "analysis_id": "behavioral_mechanism_kill_test", "phase": "M0_null_saved_representative_rival_diagnostics",
            "generating_model": "M0", "fitted_model": "MVC", "replicate_index": 0, "seed": 1300037,
            "job_id": repair.FAILED_JOB_ID, "source_job_id": repair.null.SOURCE_JOB_ID,
            "source_report": repair.null.SOURCE_REPORT, "source_dir": str(repair.null.SOURCE),
            "source_finished_utc": source["finished_utc"], "implementation": deepcopy(self.spec),
            "started_utc": "2026-10-05T05:00:00+00:00", "finished_utc": "2026-10-05T06:00:00+00:00",
            "status": "technical_rival_failure", "rival_diagnostics_complete": False,
            "error": "RuntimeError: Nonconverged fit is technical; it is not a recovery failure",
            "fits": fits, "elapsed_seconds": 3550., "participants": 108, "runs": 432, "trials": 27648,
            "completed_rivals": [], "scored_primary_directions": [{"training_runs": [1, 4], "scoring_runs": [2, 3]}],
            "saved_scores_replayed": True, "saved_score_replay_absolute_tolerance": 1e-12,
            "saved_score_replay_max_absolute_discrepancy": 0.,
            **{field: False for field in ("wave_selected", "simulation_regenerated", "saved_M0_MZ_refitted",
                "observed_choices_or_RTs_used", "old_synthetic_responses_used", "episode_terminal")},
            **{field: "not_evaluated" for field in repair.null.CLAIM_FIELDS}, "full_recovery_pass": None,
        }

    def arguments(self, output, **changes):
        values = dict(implementation=repair.null.IMPLEMENTATION, output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return tuple(getattr(repair.null, name) for name in
                     ("validate_source", "validate_fit", "fit_partially_pooled", "write_report", "REPORT_NAME"))

    def test_context_preserves_original_fit_and_changes_only_second_half_ceiling(self):
        failed = self.failed()
        before = deepcopy(failed)
        runtime, first, metadata = repair.repair_context(self.spec, failed, self.fixture.source(), self.people)
        self.assertEqual(failed, before)
        self.assertEqual(first, failed["fits"][0])
        self.assertIsNot(first, failed["fits"][0])
        self.assertEqual(runtime["optimizer"]["maximum_sweeps"], 1000)
        runtime["optimizer"]["maximum_sweeps"] = 500
        self.assertEqual(runtime, self.spec)
        self.assertEqual(metadata["previous_process_elapsed_seconds"], 3550.)
        self.assertEqual(metadata["first_half_original_fit_elapsed_seconds"], 1400.)
        self.assertFalse(metadata["nonconverged_checkpoint_state_reused"])

    def test_wrong_attempt_source_scope_and_unresolved_failure_are_rejected(self):
        changes = (("job_id", "46576235"), ("replicate_index", 1), ("seed", 1310044),
                   ("generating_model", "MZ"), ("fitted_model", "MVZ"),
                   ("phase", "MZ_SESOI_saved_replicate_rival_diagnostics"),
                   ("source_job_id", "46574267"), ("source_dir", "/different/source"),
                   ("source_finished_utc", "different"), ("implementation", {}),
                   ("wave_selected", True), ("full_recovery_pass", True),
                   ("saved_scores_replayed", False), ("scored_primary_directions", []),
                   ("error", "RuntimeError: Internal whole-process deadline exceeded"))
        for field, value in changes:
            failed = self.failed()
            failed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.repair_context(self.spec, failed, self.fixture.source(), self.people)
        for field, value in (("objective_history", [1.] * 500), ("objective_history", [1.] * 499),
                             ("subject_optimizer_failures", [{"participant": "p000"}])):
            failed = self.failed()
            failed["fits"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.repair_context(self.spec, failed, self.fixture.source(), self.people)

    def test_preserved_first_fit_must_validate_under500_and_against_actual_people(self):
        for field, value in (("converged", False), ("objective_history", [1.] * 501),
                             ("parameter_names", [])):
            failed = self.failed()
            failed["fits"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.repair_context(self.spec, failed, self.fixture.source(), self.people)
        failed = self.failed()
        # The two failed-report fits agree with each other, but not the source.
        for fit in failed["fits"]:
            fit["participants"][0] = "different"
            fit["participants"].sort()
        with self.assertRaises(ValueError):
            repair.repair_context(self.spec, failed, self.fixture.source(), self.people)

    def exercise(self, output, *, source_bad=False, failed_bad=False, replay_bad=False, second_fails=False):
        module = repair.null
        source = self.fixture.source()
        failed = self.failed(source)
        if source_bad:
            source["fits"] = source["fits"][:3]
        if failed_bad:
            failed["source_finished_utc"] = "different"
        preserved = deepcopy(failed)
        allowed = {module.IMPLEMENTATION: json.dumps(self.spec),
                   module.SOURCE / module.SOURCE_REPORT: json.dumps(source),
                   module.SOURCE / "generating_parameters.json": json.dumps(self.truth),
                   repair.FAILED_REPORT: json.dumps(failed)}
        reads, calls, validations, scores, checkpoints = [], [], [], [], []
        original = self.bindings()
        original_writer, original_validator = module.write_report, module.validate_fit

        def reader(path, *args, **kwargs):
            reads.append(path)
            return allowed[path]

        def fitter(training, model, rival, spec):
            calls.append((model, rival, deepcopy(spec)))
            fit = self.fixture.fit(model, converged=not second_fails)
            fit["objective_history"] = ([float(i) for i in range(1000, 0, -1)] if second_fails
                                        else [float(i) for i in range(599, 0, -1)] + [1.])
            return fit

        def validate(fit, model, train, held, spec, people):
            validations.append((model, list(train), spec["optimizer"]["maximum_sweeps"]))
            return original_validator(fit, model, train, held, spec, people)

        def scorer(fit, scoring, spec, check):
            check()
            scores.append((fit["model"], fit["training_runs"], deepcopy(spec)))
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            if replay_bad and fit["model"] == "M0":
                value += 1e-9
            return {person: value for person in self.people}

        def writer(directory, report):
            checkpoints.append(deepcopy(report))
            return original_writer(directory, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(module, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(module.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(module, "load_split", return_value=({}, {})))
            stack.enter_context(patch.object(module, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(module, "validate_fit", side_effect=validate))
            stack.enter_context(patch.object(module.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(module, "write_report", side_effect=writer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = repair.run(self.arguments(output))
            self.assertEqual(self.bindings(), installed)
        self.assertEqual(self.bindings(), original)
        self.assertEqual(failed, preserved)
        return report, calls, validations, scores, checkpoints, source, failed, reads

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["generating_model"], report["replicate_index"], report["seed"]),
                         (repair.PHASE, "M0", 0, 1300037))
        self.assertEqual(report["selected_replicates"], [0])
        self.assertEqual(report["selected_models"], ["MVC"])
        for field in ("wave_selected", "retry_wave_selected", "full_menu_wave_selected", "episode_terminal",
                      "simulation_regenerated", "saved_M0_MZ_refitted"):
            self.assertFalse(report[field])
        for field in repair.null.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertEqual([half["validation_maximum_sweeps"] for half in report["per_half_provenance"]], [500, 1000])

    def test_exact_first_fit_is_reused_rescored_and_only_second_fit_runs_with_correct_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "repair"
            report, calls, validations, scores, checkpoints, source, failed, reads = self.exercise(output)
            self.assertTrue(report["rival_diagnostics_complete"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:2], ("MVC", "fixed"))
            expected = deepcopy(self.spec)
            expected["optimizer"]["maximum_sweeps"] = 1000
            self.assertEqual(calls[0][2], expected)
            self.assertEqual(validations, [("M0", [1, 4], 500), ("MZ", [1, 4], 500),
                ("M0", [2, 3], 500), ("MZ", [2, 3], 500), ("MVC", [1, 4], 500), ("MVC", [2, 3], 1000)])
            self.assertEqual([(model, train) for model, train, spec in scores if model == "MVC"],
                             [("MVC", [1, 4]), ("MVC", [2, 3])])
            self.assertTrue(all(spec == self.spec for model, train, spec in scores))
            self.assertEqual({key: value for key, value in report["fits"][0].items() if key != "elapsed_seconds"},
                             {key: value for key, value in failed["fits"][0].items() if key != "elapsed_seconds"})
            self.assertEqual(report["numerical_repair"]["first_half_original_fit_elapsed_seconds"], 1400.)
            self.assertEqual(report["cumulative_process_elapsed_seconds"], 3550. + report["elapsed_seconds"])
            for person, saved in source["participant_scores"].items():
                self.assertEqual({key: report["participant_scores"][person][key] for key in saved}, saved)
            self.assertFalse(checkpoints[0]["repair_context_validated"])
            self.assertTrue(report["repair_context_validated"])
            self.assertTrue(report["first_half_fit_reused"])
            self.assertTrue(report["fresh_second_half_fit_called"])
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint)
            self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
            self.assertFalse((output / "simulated_events.csv.gz").exists())

    def test_source_context_and_replay_failures_stop_before_reuse_or_fresh_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            for scenario in ("source_bad", "failed_bad", "replay_bad"):
                report, calls, validations, scores, checkpoints, source, failed, reads = self.exercise(
                    Path(directory) / scenario, **{scenario: True})
                self.assertEqual(calls, [])
                self.assertFalse(report["first_half_fit_reused"])
                self.assertFalse(report["fresh_second_half_fit_called"])
                self.assertEqual(report["fits"], [])
                self.assertFalse(report["rival_diagnostics_complete"])
                self.assertNotIn("participant_scores", report)
                if scenario == "source_bad":
                    self.assertNotIn(repair.FAILED_REPORT, reads)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint)

    def test_second_half_failure_keeps_both_fits_and_original_cost_without_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, validations, scores, checkpoints, source, failed, reads = self.exercise(
                Path(directory) / "failed", second_fails=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertEqual(len(report["fits"]), 2)
            self.assertTrue(report["fits"][0]["converged"])
            self.assertFalse(report["fits"][1]["converged"])
            self.assertEqual(len(report["scored_primary_directions"]), 1)
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["cumulative_process_elapsed_seconds"], 3550. + report["elapsed_seconds"])

    def test_inherited_guards_protected_outputs_and_restored_bindings(self):
        original = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.object(repair.null, "require_compute"), \
                patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No payload read")):
            for change in ({"implementation": Path(directory) / "wrong.json"}, {"max_seconds": 0},
                           {"max_seconds": 5401}, {"max_seconds": True}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    repair.run(self.arguments(Path(directory) / "fresh", **change))
                self.assertEqual(self.bindings(), original)
            for name in ("SLURM_CPUS_PER_TASK", *fixtures.THREADS):
                with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                    repair.run(self.arguments(Path(directory) / "fresh"))
            for output in (Path(directory), repair.null.SOURCE / "new", repair.FAILED_REPORT.parent / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    repair.run(self.arguments(output))
        self.assertEqual(self.bindings(), original)


if __name__ == "__main__":
    unittest.main()
