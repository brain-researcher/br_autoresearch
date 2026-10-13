"""Focused MZ42 reverse completion tests; run only in coordinated compute."""
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

import run_mz_mvc_reverse_completion as repair
import test_mz_mvc_convergence_repair as fixtures
from test_mz_null_representative import THREADS


class ReverseCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ConvergenceRepairTests.setUpClass()
        cls.prior_fixture = fixtures.ConvergenceRepairTests(
            "test_first_half_exhaustion_freshly_fits_both_halves_with_original_baseline_replay")
        cls.fixture = cls.prior_fixture.fixture
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def reports(self):
        source = self.prior_fixture.source(42)
        source["finished_utc"] = repair.SOURCE_FINISHED_UTC
        original = self.prior_fixture.failed(42)
        original.update(source_finished_utc=source["finished_utc"], elapsed_seconds=1661.5854664351791,
                        job_id=repair.ORIGINAL_JOB_ID, finished_utc=repair.ORIGINAL_FINISHED_UTC)
        original["fits"][0]["elapsed_seconds"] = 1659.960657726042
        runtime, metadata = repair.pilot.repair_context(self.spec, original)
        previous = deepcopy(original)
        first = self.fixture.fit("MVC", *repair.saved.HALVES[0])
        first.update(objective_history=[float(value) for value in range(666, 1, -1)] + [2.],
                     elapsed_seconds=2156.1116262879223)
        reverse = self.fixture.fit("MVC", *repair.saved.HALVES[1], converged=False)
        reverse.update(objective_history=[float(value) for value in range(1000, 0, -1)],
                       elapsed_seconds=3234.350768652046)
        previous.update(phase=repair.pilot.PHASE, job_id=repair.PILOT_JOB_ID,
                        started_utc="2026-10-05T05:21:42.783354+00:00", finished_utc=repair.PILOT_FINISHED_UTC,
                        runtime_implementation=runtime, numerical_repair=metadata, internal_seconds=5400,
                        selected_replicates=[42], retry_wave_selected=False,
                        full_menu_false_selection="not_evaluated", false_selection_rate="not_evaluated",
                        fits=[first, reverse], elapsed_seconds=5392.607597094029,
                        cumulative_process_elapsed_seconds=original["elapsed_seconds"] + 5392.607597094029,
                        scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
        return source, original, previous

    def args(self, output, **changes):
        values = dict(implementation=repair.saved.IMPLEMENTATION, output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def references(self):
        saved = repair.saved
        return (saved.validate_source, saved.fit_partially_pooled, saved.validate_fit,
                saved.shared.score_fit, saved.write_report, saved.REPORT_NAME)

    def exercise(self, output, *, reports=None, failure=None):
        saved = repair.saved
        source, original, previous = self.reports() if reports is None else reports
        preserved = deepcopy((source, original, previous))
        truth = deepcopy(self.fixture.truth)
        for params in truth.values():
            params.update(saved.focal_truth("MZ"))
        allowed = {saved.IMPLEMENTATION: json.dumps(self.spec), repair.ORIGINAL_REPORT: json.dumps(original),
                   repair.PILOT_REPORT: json.dumps(previous),
                   saved.source_directory(42) / "single_process_recovery.json": json.dumps(source),
                   saved.source_directory(42) / "generating_parameters.json": json.dumps(truth)}
        calls, validations, checkpoints, scored, reads = [], [], [], [], []
        validator, writer = saved.validate_fit, saved.write_report
        references = self.references()
        alarm_active = False

        def alarm(seconds):
            nonlocal alarm_active
            alarm_active = seconds > 0

        def read(path, *args, **kwargs):
            self.assertTrue(alarm_active, "Source/spec/prior-attempt reads belong inside the existing timer")
            reads.append(path)
            return allowed[path]

        def fitter(training, model, rival, spec):
            self.assertTrue(alarm_active)
            calls.append((training, model, rival, deepcopy(spec)))
            if failure == "deadline":
                raise RuntimeError("Saved-replicate rival interrupted by signal15; unfinished work is technical")
            fit = self.fixture.fit(model, converged=failure != "fit")
            count = 1500 if failure == "fit" else 1200
            fit["objective_history"] = [float(value) for value in range(count, 1, -1)] + [1. if failure == "fit" else 2.]
            return fit

        def validate(fit, model, train, held, spec, people):
            validations.append((model, tuple(train), spec["optimizer"]["maximum_sweeps"]))
            return validator(fit, model, train, held, spec, people)

        def score(fit, scoring, spec, check):
            check()
            scored.append((fit["model"], tuple(fit["training_runs"]), scoring))
            self.assertEqual(spec, self.spec)
            if failure == "rescore" and fit["model"] == "MVC" and fit["training_runs"] == [1, 4]:
                raise RuntimeError("Synthetic preserved first-fit rescore failure")
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            if failure == "replay" and fit["model"] == "M0":
                value += .001
            return {person: value for person in self.people}

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            return writer(directory, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(saved, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", read))
            stack.enter_context(patch.object(saved.signal, "alarm", side_effect=alarm))
            stack.enter_context(patch.object(saved.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(saved, "load_split", side_effect=lambda events, train, held: (tuple(train), tuple(held))))
            stack.enter_context(patch.object(saved, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(saved, "validate_fit", side_effect=validate))
            stack.enter_context(patch.object(saved.shared, "score_fit", side_effect=score))
            stack.enter_context(patch.object(saved, "write_report", side_effect=checkpoint))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.references()
            result = repair.run(self.args(output))
            self.assertEqual(self.references(), installed)
            self.assertFalse(alarm_active)
        self.assertEqual(self.references(), references)
        self.assertEqual((source, original, previous), preserved)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), result)
        self.assertIn(repair.ORIGINAL_REPORT, reads)
        self.assertIn(repair.PILOT_REPORT, reads)
        return result, calls, validations, checkpoints, source, scored

    def assert_boundary(self, result, checkpoints):
        self.assertFalse(checkpoints[0]["repair_context_validated"])
        for report in checkpoints:
            self.assertEqual(report["phase"], repair.PHASE)
            self.assertEqual(report["selected_replicates"], [42])
            self.assertEqual(report["baseline_validation_maximum_sweeps"], 500)
            self.assertEqual([half["validation_maximum_sweeps"] for half in report["per_half_provenance"]], [1000, 1500])
            for field in repair.CLAIM_FIELDS:
                self.assertEqual(report[field], "not_evaluated")
            for field in ("wave_selected", "retry_wave_selected", "automatic_retry", "full_menu_wave_selected", "episode_terminal"):
                self.assertFalse(report[field])
            self.assertIsNone(report["full_recovery_pass"])
            if report["repair_context_validated"]:
                self.assertEqual(report["implementation"], self.spec)
                for key, ceiling in (("first_half_runtime_implementation", 1000), ("runtime_implementation", 1500)):
                    runtime = deepcopy(report[key])
                    self.assertEqual(runtime["optimizer"]["maximum_sweeps"], ceiling)
                    runtime["optimizer"]["maximum_sweeps"] = 500
                    self.assertEqual(runtime, self.spec)
                self.assertFalse(report["numerical_repair"]["nonconverged_checkpoint_state_reused"])
        prior = self.reports()[2]["cumulative_process_elapsed_seconds"]
        self.assertEqual(result["cumulative_process_elapsed_seconds"], prior + result["elapsed_seconds"])

    def test_exactly_one_fresh_reverse_fit_with_mixed_specs_replay_and_preserved_first_fit(self):
        previous = self.reports()[2]
        with tempfile.TemporaryDirectory() as directory:
            result, calls, validations, checkpoints, source, scored = self.exercise(Path(directory) / "complete")
        self.assertTrue(result["rival_diagnostics_complete"])
        self.assertEqual([(train, model, rival) for train, model, rival, spec in calls], [((2, 3), "MVC", "fixed")])
        runtime = deepcopy(self.spec)
        runtime["optimizer"]["maximum_sweeps"] = 1500
        self.assertEqual(calls[0][3], runtime)
        self.assertEqual([ceiling for model, train, ceiling in validations], [500] * 4 + [1000, 1500])
        self.assertEqual({key: value for key, value in result["fits"][0].items() if key != "elapsed_seconds"},
                         {key: value for key, value in previous["fits"][0].items() if key != "elapsed_seconds"})
        self.assertEqual([(model, train, held) for model, train, held in scored if model == "MVC"],
                         [("MVC", (1, 4), (2, 3)), ("MVC", (2, 3), (1, 4))])
        for person, row in source["participant_scores"].items():
            self.assertEqual({key: result["participant_scores"][person][key] for key in row}, row)
        metadata = result["numerical_repair"]
        self.assertEqual(metadata["first_half_original_fit_elapsed_seconds"], 2156.1116262879223)
        self.assertEqual(metadata["first_half_materialization_elapsed_seconds"], result["fits"][0]["elapsed_seconds"])
        self.assertTrue(metadata["first_half_rescored"])
        self.assertGreaterEqual(metadata["first_half_rescore_elapsed_seconds"], 0.)
        self.assert_boundary(result, checkpoints)

    def test_source_identity_terminal_failure_and_claim_boundaries_are_exact(self):
        mutations = (("phase", "MZ_SESOI_saved_replicate_rival_diagnostics"), ("job_id", "46577370"),
                     ("replicate_index", 43), ("seed", 1), ("source_dir", "/other/source"),
                     ("source_finished_utc", "2026-10-05T00:00:00+00:00"),
                     ("finished_utc", "2026-10-05T06:51:35+00:00"), ("family_selection", "evaluated"),
                     ("error", "RuntimeError: Internal whole-process deadline exceeded"),
                     ("saved_score_replay_max_absolute_discrepancy", 2e-12), ("scored_primary_directions", []),
                     ("cumulative_process_elapsed_seconds", 5392.607597094029))
        for field, value in mutations:
            source, original, previous = self.reports()
            previous[field] = value
            preserved = deepcopy((source, original, previous))
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.repair_context(self.spec, original, previous, source, self.people)
            self.assertEqual((source, original, previous), preserved)
        for field, value in (("error", "RuntimeError: interrupted"), ("job_id", "46572753"),
                             ("finished_utc", "2026-10-05T05:03:24+00:00")):
            source, original, previous = self.reports()
            original[field] = value
            with self.subTest(original_field=field), self.assertRaises(ValueError):
                repair.repair_context(self.spec, original, previous, source, self.people)

    def test_runtime_only_change_and_actual_people_and_exhaustion_are_required(self):
        for mismatch in ("runtime", "metadata", "first_people", "reverse_people", "first_subject_failure",
                         "reverse_subject_failure", "reverse_converged", "reverse_tolerance", "first_sweeps"):
            source, original, previous = self.reports()
            if mismatch == "runtime":
                previous["runtime_implementation"]["optimizer"]["joint_relative_tolerance"] *= 2
            elif mismatch == "metadata":
                previous["numerical_repair"]["previous_process_elapsed_seconds"] += 1
            elif mismatch.endswith("people"):
                previous["fits"][0 if mismatch.startswith("first") else 1]["participants"][0] = "other"
            elif mismatch.endswith("subject_failure"):
                previous["fits"][0 if mismatch.startswith("first") else 1]["subject_optimizer_failures"] = ["failed"]
            elif mismatch == "reverse_converged":
                previous["fits"][1]["converged"] = True
            elif mismatch == "reverse_tolerance":
                previous["fits"][1]["objective_history"] = [1.] * 1000
            else:
                previous["fits"][0]["objective_history"] = [1.] * 667
            with self.subTest(mismatch=mismatch), self.assertRaises((ValueError, RuntimeError)):
                repair.repair_context(self.spec, original, previous, source, self.people)

    def test_context_rejection_is_timed_and_retains_a_technical_checkpoint_without_fitting(self):
        reports = self.reports()
        reports[2]["source_finished_utc"] = "2026-10-05T00:00:00+00:00"
        with tempfile.TemporaryDirectory() as directory:
            result, calls, validations, checkpoints, source, scored = self.exercise(Path(directory) / "bad", reports=reports)
        self.assertEqual(result["status"], "technical_rival_failure")
        self.assertFalse(result["repair_context_validated"])
        self.assertEqual(calls, [])
        self.assertEqual(result["fits"], [])
        self.assertNotIn("participant_scores", result)
        self.assertNotIn("cumulative_process_elapsed_seconds", result)

    def test_replay_and_preserved_rescore_failures_stop_before_new_optimization(self):
        for failure in ("replay", "rescore"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                result, calls, validations, checkpoints, source, scored = self.exercise(Path(directory) / failure, failure=failure)
            self.assertEqual(calls, [])
            self.assertEqual(result["status"], "technical_rival_failure")
            self.assertEqual(len(result["fits"]), 0 if failure == "replay" else 1)
            self.assertFalse(result["numerical_repair"]["first_half_rescored"])
            self.assertNotIn("participant_scores", result)
            self.assertEqual(result["completed_rivals"], [])
            self.assertEqual(result["diagnostics"], {})
            self.assert_boundary(result, checkpoints)

    def test_reverse_failure_preserves_both_fits_and_costs_without_scores_or_retry(self):
        for failure in ("fit", "deadline"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                result, calls, validations, checkpoints, source, scored = self.exercise(Path(directory) / "failed", failure=failure)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["status"], "technical_rival_failure")
            self.assertEqual(len(result["fits"]), 2 if failure == "fit" else 1)
            self.assertTrue(result["fits"][0]["converged"])
            if failure == "fit":
                self.assertFalse(result["fits"][1]["converged"])
                self.assertEqual(len(result["fits"][1]["objective_history"]), 1500)
            self.assertEqual(result["scored_primary_directions"], [{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
            self.assertNotIn("participant_scores", result)
            self.assertEqual(result["diagnostics"], {})
            self.assertEqual(result["completed_rivals"], [])
            self.assert_boundary(result, checkpoints)

    def test_compute_cpu_cap_and_fresh_protected_output_guards_precede_source_reads(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No read before guards")):
            with patch.object(repair, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                repair.run(self.args(Path(directory) / "fresh"))
            with patch.object(repair, "require_compute"):
                for changes in ({"max_seconds": 0}, {"max_seconds": 5401}, {"implementation": Path(directory) / "other.json"}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        repair.run(self.args(Path(directory) / "fresh", **changes))
                for name in ("SLURM_CPUS_PER_TASK", *THREADS):
                    with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                        repair.run(self.args(Path(directory) / "fresh"))
                for output in (Path(directory), repair.saved.source_directory(42) / "new",
                               repair.ORIGINAL_REPORT.parent / "new", repair.PILOT_REPORT.parent / "new"):
                    with self.subTest(output=output), self.assertRaises(FileExistsError):
                        repair.run(self.args(output))


if __name__ == "__main__":
    unittest.main()
