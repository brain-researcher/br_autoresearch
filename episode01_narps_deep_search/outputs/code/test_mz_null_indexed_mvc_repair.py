"""Focused exact null43/61/168 first-half exhaustion adapter tests; compute only."""
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

import run_mz_null_indexed_mvc_repair as repair
import test_mz_null_indexed_menu as fixtures
from test_mz_null_representative import THREADS


class IndexedNullMVCRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IndexedNullMenuTests.setUpClass()
        cls.source_fixture = fixtures.IndexedNullMenuTests(
            "test_indexed_builder_reuses_nine_contrasts_and_original500_rivals_runtime1000_MVC")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def reports(self, index=43):
        panels = self.source_fixture.reports(index)
        source, failed = panels["baseline"], panels["MVC"]
        identity = repair.IDENTITIES[index]
        source.update(job_id=identity["source_job_id"], finished_utc=identity["source_finished_utc"])
        failed.update(job_id=identity["failed_job_id"], source_job_id=identity["source_job_id"],
                      source_finished_utc=identity["source_finished_utc"],
                      started_utc="2026-10-05T07:42:28.167570+00:00", finished_utc=identity["failed_finished_utc"],
                      status="technical_rival_failure", rival_diagnostics_complete=False, error=repair.NONCONVERGENCE,
                      elapsed_seconds=3413.08795671002, internal_seconds=8400, fits=failed["fits"][:1],
                      scored_primary_directions=[], completed_rivals=[], diagnostics={})
        failed.pop("participant_scores")
        failed["fits"][0].update(converged=False, objective_history=[float(value) for value in range(1000, 0, -1)],
                                 elapsed_seconds=3411.043164890958, subject_optimizer_failures=[])
        return source, failed

    def arguments(self, output, index=43, **changes):
        args = dict(replicate_index=index, implementation=repair.indexed.IMPLEMENTATION,
                    output_dir=Path(output), max_seconds=12600)
        args.update(changes)
        return argparse.Namespace(**args)

    def bindings(self):
        return tuple(getattr(repair.indexed, name) for name in
                     ("RUNTIME_SWEEPS", "MAX_SECONDS", "REPORT_NAME", "validate_source", "write_report", "PHASE"))

    def exercise(self, output, *, index=43, source_change=None, failed_change=None, failure=None):
        engine = repair.indexed
        source, failed = self.reports(index)
        if source_change:
            source_change(source)
        if failed_change:
            failed_change(failed)
        original = deepcopy((source, failed, self.spec, self.truth))
        directory, path = engine.source_paths(index)
        allowed = {engine.IMPLEMENTATION: json.dumps(self.spec), path: json.dumps(source),
                   directory / "generating_parameters.json": json.dumps(self.truth), repair.failed_path(index): json.dumps(failed)}
        reads, calls, scored, checkpoints, alarms = [], [], [], [], []
        writer = engine.write_report
        clock = [0.]
        bindings = self.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def reader(selected, *args, **kwargs):
            self.assertEqual(alarms[-1], 12600, "All source and failed-attempt reads stay inside the checked timer")
            reads.append(selected)
            if failure == "source_timeout" and selected == repair.failed_path(index):
                engine.interrupted(signal.SIGALRM, None)
            return allowed[selected]

        def fitter(training, model, bound, spec):
            self.assertEqual(alarms[-1], 12600)
            self.assertEqual(engine.RUNTIME_SWEEPS, 1500, "Prior validation must restore the new runtime before fitting")
            calls.append((training, model, bound, deepcopy(spec)))
            if failure == "fit_timeout":
                engine.interrupted(signal.SIGTERM, None)
            converged = not ((failure == "first_fit" and len(calls) == 1) or (failure == "second_fit" and len(calls) == 2))
            fit = self.fixture.fit(model, converged=converged)
            count = 1100 if converged else 1500
            fit["objective_history"] = [float(value) for value in range(count, 1, -1)] + [2. if converged else 1.]
            return fit

        def score(fit, scoring, spec, check):
            check()
            self.assertEqual(spec, self.spec)
            scored.append((fit["model"], tuple(fit["training_runs"])))
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            if failure == "replay" and fit["model"] == "M0":
                value += 1e-9
            return {person: value for person in self.people}

        def checkpoint(destination, report):
            checkpoints.append(deepcopy(report))
            if report["rival_diagnostics_complete"]:
                self.assertEqual(alarms[-1], 12600)
                if failure == "finalization_timeout":
                    clock[0] = 12600.
            return writer(destination, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(engine, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(engine.signal, "alarm", side_effect=alarms.append))
            stack.enter_context(patch.object(engine.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(engine.null.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(engine, "load_split", side_effect=lambda events, train, held: (tuple(train), tuple(held))))
            stack.enter_context(patch.object(engine, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(engine.null.shared, "score_fit", side_effect=score))
            stack.enter_context(patch.object(engine, "write_report", side_effect=checkpoint))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = repair.run(self.arguments(output, index))
            self.assertEqual(self.bindings(), installed)
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(alarms, [12600, 0])
        self.assertEqual((source, failed, self.spec, self.truth), original)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        return report, calls, scored, checkpoints, reads, source, failed

    def assert_scope(self, report, checkpoints, failed):
        for checkpoint in checkpoints:
            self.assertEqual(checkpoint["phase"], repair.PHASE)
            index = failed["replicate_index"]
            self.assertEqual((checkpoint["replicate_index"], checkpoint["seed"]), (index, 1300037 + 10007 * index))
            self.assertEqual(checkpoint["fitted_maximum_sweeps"], 1500)
            self.assertEqual(checkpoint["previous_runtime_maximum_sweeps"], 1000)
            self.assertEqual(checkpoint["original_maximum_sweeps"], 500)
            self.assertEqual(checkpoint["numerical_override"]["runtime_maximum_sweeps"], 1500)
            for field in ("wave_selected", "retry_wave_selected", "retry_selected", "full_menu_wave_selected", "automatic_retry"):
                self.assertFalse(checkpoint[field])
            for field in repair.indexed.CLAIM_FIELDS:
                self.assertEqual(checkpoint[field], "not_evaluated")
            if "runtime_implementation" in checkpoint:
                spec = deepcopy(checkpoint["runtime_implementation"])
                self.assertEqual(spec["optimizer"]["maximum_sweeps"], 1500)
                spec["optimizer"]["maximum_sweeps"] = 500
                self.assertEqual(spec, self.spec)
        if report["repair_context_validated"]:
            self.assertEqual(report["cumulative_process_elapsed_seconds"], failed["elapsed_seconds"] + report["elapsed_seconds"])
            self.assertEqual(report["numerical_repair"]["previous_runtime_implementation"]["optimizer"]["maximum_sweeps"], 1000)
            self.assertFalse(report["numerical_repair"]["nonconverged_checkpoint_state_reused"])
            self.assertFalse(report["numerical_repair"]["completed_first_fit_available"])

    def test_exact_prior1000_failure_preserves_original_spec_and_restores_runtime_binding(self):
        for index in repair.IDENTITIES:
            source, failed = self.reports(index)
            before = deepcopy((source, failed, self.spec))
            with patch.object(repair.indexed, "RUNTIME_SWEEPS", 1500):
                metadata = repair.validate_failure(index, failed, source, self.people, self.spec)
                self.assertEqual(repair.indexed.RUNTIME_SWEEPS, 1500)
            self.assertEqual((source, failed, self.spec), before)
            self.assertEqual(metadata["failed_report"], str(repair.failed_path(index)))
            self.assertEqual(metadata["failed_job_id"], repair.IDENTITIES[index]["failed_job_id"])
            self.assertEqual(metadata["previous_process_elapsed_seconds"], failed["elapsed_seconds"])
            self.assertGreater(metadata["final_objective_change"], metadata["unchanged_convergence_threshold"])
            self.assertEqual(metadata["previous_fit_elapsed_seconds"], [failed["fits"][0]["elapsed_seconds"]])

    def test_other_source_identity_second_half_or_unsupported_failure_is_rejected(self):
        changes = (("job_id", "other"), ("finished_utc", "different"), ("replicate_index", 44),
                   ("seed", 1), ("source_finished_utc", "different"), ("source_job_id", "other"),
                   ("error", "RuntimeError: deadline"), ("saved_score_replay_max_absolute_discrepancy", 2e-12),
                   ("scored_primary_directions", [{}]), ("family_selection", "selected"))
        source, failed = self.reports()
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.validate_failure(43, {**failed, field: value}, source, self.people, self.spec)
        for field, value in (("converged", True), ("objective_history", [1.] * 1000),
                             ("objective_history", list(range(500, 0, -1))), ("subject_optimizer_failures", ["failed"])):
            changed = deepcopy(failed)
            changed["fits"][0][field] = value
            with self.subTest(fit_field=field), self.assertRaises(ValueError):
                repair.validate_failure(43, changed, source, self.people, self.spec)
        with self.assertRaises(ValueError):
            repair.validate_failure(43, {**failed, "fits": failed["fits"] * 2}, source, self.people, self.spec)
        with self.assertRaises(ValueError):
            repair.validate_failure(168, failed, source, self.people, self.spec)

    def test_two_fresh1500_fits_follow_original500_replay_preserve_last_bits_and_cost_once(self):
        for index in repair.IDENTITIES:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                report, calls, scored, checkpoints, reads, source, failed = self.exercise(Path(directory) / "new", index=index)
            self.assertTrue(report["rival_diagnostics_complete"])
            self.assertEqual([(train, model, bound) for train, model, bound, spec in calls],
                             [((1, 4), "MVC", "fixed"), ((2, 3), "MVC", "fixed")])
            self.assertTrue(all(spec == report["runtime_implementation"] for train, model, bound, spec in calls))
            self.assertEqual(scored[:4], [(model, train) for train, held in repair.indexed.HALVES for model in ("M0", "MZ")])
            self.assertIn(repair.failed_path(index), reads)
            for other_index in repair.IDENTITIES:
                if other_index != index:
                    self.assertNotIn(repair.failed_path(other_index), reads)
            self.assertNotIn(repair.QUALIFYING_REPORT, reads)
            for person, row in source["participant_scores"].items():
                self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
            self.assert_scope(report, checkpoints, failed)

    def test_bad_baseline_or_failed_attempt_stops_before_new_fit_with_technical_checkpoint(self):
        for kind in ("baseline", "failed"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                report, calls, scored, checkpoints, reads, source, failed = self.exercise(Path(directory) / kind,
                    source_change=(lambda value: value.update(job_id="other")) if kind == "baseline" else None,
                    failed_change=(lambda value: value.update(job_id="other")) if kind == "failed" else None)
            self.assertEqual(calls, [])
            self.assertFalse(report["repair_context_validated"])
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertNotIn("participant_scores", report)
            self.assert_scope(report, checkpoints, failed)

    def test_replay_failure_cannot_start_optimization_or_release_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, scored, checkpoints, reads, source, failed = self.exercise(Path(directory) / "replay", failure="replay")
        self.assertEqual(calls, [])
        self.assertTrue(report["repair_context_validated"])
        self.assertFalse(report["saved_scores_replayed"])
        self.assertEqual(report["fits"], [])
        self.assertNotIn("participant_scores", report)
        self.assert_scope(report, checkpoints, failed)

    def test_fresh_half_failure_retains_prefix_and_never_reports_scientific_failure(self):
        for failure, count in (("first_fit", 1), ("second_fit", 2)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, scored, checkpoints, reads, source, failed = self.exercise(Path(directory) / failure, failure=failure)
            self.assertEqual(len(calls), count)
            self.assertEqual(len(report["fits"]), count)
            self.assertEqual(len(report["scored_primary_directions"]), count - 1)
            self.assertFalse(report["fits"][-1]["converged"])
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertNotIn("participant_scores", report)
            self.assert_scope(report, checkpoints, failed)

    def test_timed_failed_reads_fit_interruption_and_finalization_restore_all_bindings(self):
        for failure in ("source_timeout", "fit_timeout", "finalization_timeout"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, scored, checkpoints, reads, source, failed = self.exercise(Path(directory) / failure, failure=failure)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["completed_rivals"], [])
            self.assert_scope(report, checkpoints, failed)

    def test_compute_exact_index_cap_original_path_cpu_and_output_guards(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            with patch.object(repair, "require_compute", side_effect=RuntimeError("compute")), self.assertRaises(RuntimeError):
                repair.run(self.arguments(Path(directory) / "new"))
            with patch.object(repair, "require_compute"), patch.object(repair.indexed, "require_compute"):
                for changes in ({"replicate_index": 0}, {"replicate_index": 42}, {"replicate_index": 44},
                                {"replicate_index": True}, {"max_seconds": 0}, {"max_seconds": 12601},
                                {"implementation": Path(directory) / "wrong.json"}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        repair.run(self.arguments(Path(directory) / "new", **changes))
                with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), self.assertRaises(ValueError):
                    repair.run(self.arguments(Path(directory) / "new"))
                for output in (Path(directory), repair.failed_path(43).parent / "new", repair.failed_path(168).parent / "new",
                               repair.indexed.source_paths(43)[0] / "new"):
                    with self.subTest(output=output), self.assertRaises(FileExistsError):
                        repair.run(self.arguments(output))


if __name__ == "__main__":
    unittest.main()
