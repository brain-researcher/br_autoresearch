"""Focused exact selected null preserved1000/fresh1500 completion checks; compute only."""
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

import run_mz_null_indexed_mvc_reverse_completion as repair
import test_mz_null_indexed_menu as fixtures
from test_mz_null_representative import THREADS


class NullMVCReverseCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IndexedNullMenuTests.setUpClass()
        cls.source_fixture = fixtures.IndexedNullMenuTests(
            "test_indexed_builder_reuses_nine_contrasts_and_original500_rivals_runtime1000_MVC")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def reports(self, index=183):
        panels = self.source_fixture.reports(index)
        source, failed = panels["baseline"], panels["MVC"]
        identity = repair.IDENTITIES[index]
        source.update(job_id=identity["source_job_id"], finished_utc=identity["source_finished_utc"])
        failed.update(job_id=identity["failed_job_id"], source_job_id=identity["source_job_id"],
            source_finished_utc=identity["source_finished_utc"], started_utc="2026-10-05T07:44:50.102175+00:00",
            finished_utc=identity["failed_finished_utc"], status="technical_rival_failure", rival_diagnostics_complete=False,
            error=repair.NONCONVERGENCE, elapsed_seconds=4413.057063470595, internal_seconds=8400,
            scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}], completed_rivals=[], diagnostics={})
        failed.pop("participant_scores")
        failed["fits"][0].update(objective_history=[float(value) for value in range(269, 1, -1)] + [2.],
                                  elapsed_seconds=929.43805972673)
        failed["fits"][1].update(converged=False, objective_history=list(range(1000, 0, -1)),
                                  elapsed_seconds=3481.6794653348625, subject_optimizer_failures=[])
        return source, failed

    def arguments(self, output, index=183, **changes):
        values = dict(replicate_index=index, implementation=repair.indexed.IMPLEMENTATION, output_dir=Path(output), max_seconds=6600)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return (tuple(getattr(repair.indexed, name) for name in ("RUNTIME_SWEEPS", "MAX_SECONDS", "REPORT_NAME",
            "validate_source", "fit_partially_pooled", "validate_fit", "write_report", "PHASE")),
            repair.indexed.null.shared.score_fit)

    def exercise(self, output, *, index=183, source_change=None, failed_change=None, failure=None):
        engine = repair.indexed
        source, failed = self.reports(index)
        if source_change:
            source_change(source)
        if failed_change:
            failed_change(failed)
        before = deepcopy((source, failed, self.spec, self.truth))
        directory, source_path = engine.source_paths(index)
        allowed = {engine.IMPLEMENTATION: json.dumps(self.spec), source_path: json.dumps(source),
                   directory / "generating_parameters.json": json.dumps(self.truth), repair.failed_path(index): json.dumps(failed)}
        reads, calls, scores, validations, checkpoints, alarms = [], [], [], [], [], []
        clock = [0.]
        original_write, original_validate = engine.write_report, engine.validate_fit
        bindings = self.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def reader(path, *args, **kwargs):
            self.assertEqual(alarms[-1], 6600, "All original source and failed-attempt reads are timed")
            reads.append(path)
            if failure == "source_timeout" and path == repair.failed_path(index):
                engine.interrupted(signal.SIGALRM, None)
            return allowed[path]

        def fitter(training, model, bound, spec):
            self.assertEqual(alarms[-1], 6600)
            self.assertEqual(engine.RUNTIME_SWEEPS, 1500)
            calls.append((training, model, bound, deepcopy(spec)))
            if failure == "fit_timeout":
                engine.interrupted(signal.SIGTERM, None)
            converged = failure != "nonconvergence"
            fit = self.fixture.fit(model, converged=converged)
            count = 1100 if converged else 1500
            fit["objective_history"] = [float(value) for value in range(count, 1, -1)] + [2. if converged else 1.]
            return fit

        def validator(fit, model, train, held, spec, people):
            validations.append((model, tuple(train), spec["optimizer"]["maximum_sweeps"]))
            return original_validate(fit, model, train, held, spec, people)

        def scorer(fit, scoring, spec, check):
            check()
            self.assertEqual(spec, self.spec)
            self.assertEqual(alarms[-1], 6600)
            scores.append(deepcopy(fit))
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            if failure == "replay" and fit["model"] == "M0":
                value += 1e-9
            if failure == "rescore" and fit["model"] == "MVC":
                raise RuntimeError("Selected preserved-first rescoring failure")
            return {person: value for person in self.people}

        def writer(destination, report):
            checkpoints.append(deepcopy(report))
            if report["rival_diagnostics_complete"]:
                self.assertEqual(alarms[-1], 6600)
                if failure == "finalization_timeout":
                    clock[0] = 6600.
            return original_write(destination, report)

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
            stack.enter_context(patch.object(engine, "validate_fit", side_effect=validator))
            stack.enter_context(patch.object(engine.null.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(engine, "write_report", side_effect=writer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = repair.run(self.arguments(output, index))
            self.assertEqual(self.bindings(), installed)
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(alarms, [6600, 0])
        self.assertEqual((source, failed, self.spec, self.truth), before)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        return report, calls, scores, validations, checkpoints, reads, source, failed

    def assert_scope(self, report, checkpoints, failed):
        index = failed["replicate_index"]
        for checkpoint in checkpoints:
            self.assertEqual((checkpoint["phase"], checkpoint["replicate_index"], checkpoint["seed"]),
                             (repair.PHASE, index, 1300037 + 10007 * index))
            self.assertEqual([row["validation_maximum_sweeps"] for row in checkpoint["per_half_provenance"]], [1000, 1500])
            self.assertEqual(checkpoint["numerical_override"]["fresh_fit_directions"],
                             [{"training_runs": [2, 3], "scoring_runs": [1, 4]}])
            for field in repair.indexed.CLAIM_FIELDS:
                self.assertEqual(checkpoint[field], "not_evaluated")
            for field in ("wave_selected", "retry_selected", "retry_wave_selected", "automatic_retry", "full_menu_wave_selected"):
                self.assertFalse(checkpoint[field])
            if "runtime_implementation" in checkpoint:
                runtime = deepcopy(checkpoint["runtime_implementation"])
                self.assertEqual(runtime["optimizer"]["maximum_sweeps"], 1500)
                runtime["optimizer"]["maximum_sweeps"] = 500
                self.assertEqual(runtime, self.spec)
                self.assertEqual(checkpoint["implementation"], self.spec)
        if report["repair_context_validated"]:
            self.assertEqual(report["cumulative_process_elapsed_seconds"], failed["elapsed_seconds"] + report["elapsed_seconds"])
            self.assertEqual(report["numerical_repair"]["first_half_original_fit_elapsed_seconds"], failed["fits"][0]["elapsed_seconds"])
        if report["fits"]:
            self.assertEqual(report["first_half_copy_elapsed_seconds"], report["fits"][0]["elapsed_seconds"])
            self.assertEqual({key: value for key, value in report["fits"][0].items() if key != "elapsed_seconds"},
                             {key: value for key, value in failed["fits"][0].items() if key != "elapsed_seconds"})

    def test_context_requires_exact_source_and_returns_copy_of_valid_first1000_fit(self):
        for index in repair.IDENTITIES:
            source, failed = self.reports(index)
            before = deepcopy((source, failed, self.spec))
            with patch.object(repair.indexed, "RUNTIME_SWEEPS", 1500):
                first, metadata = repair.repair_context(index, self.spec, failed, source, self.people)
                self.assertEqual(repair.indexed.RUNTIME_SWEEPS, 1500)
            self.assertEqual(first, failed["fits"][0])
            self.assertIsNot(first, failed["fits"][0])
            self.assertEqual(metadata["failed_job_id"], repair.IDENTITIES[index]["failed_job_id"])
            self.assertEqual(metadata["previous_runtime_implementation"]["optimizer"]["maximum_sweeps"], 1000)
            self.assertGreater(metadata["final_reverse_objective_change"], metadata["unchanged_convergence_threshold"])
            self.assertEqual((source, failed, self.spec), before)

    def test_wrong_identity_cause_first_fit_actual_people_and_specs_are_rejected(self):
        source, failed = self.reports()
        for field, value in (("job_id", "other"), ("finished_utc", "different"), ("replicate_index", 43),
                             ("seed", 1), ("source_finished_utc", "different"), ("source_job_id", "other"),
                             ("error", "RuntimeError: deadline"), ("scored_primary_directions", []),
                             ("saved_score_replay_max_absolute_discrepancy", 2e-12), ("full_gate", "pass")):
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                repair.repair_context(183, self.spec, {**failed, field: value}, source, self.people)
        for position, field, value in ((0, "converged", False), (0, "objective_history", [1.] * 1001),
                                      (0, "parameter_names", []), (1, "converged", True),
                                      (1, "objective_history", [1.] * 1000), (1, "subject_optimizer_failures", ["failed"])):
            changed = deepcopy(failed)
            changed["fits"][position][field] = value
            with self.subTest(position=position, field=field), self.assertRaises((ValueError, RuntimeError)):
                repair.repair_context(183, self.spec, changed, source, self.people)
        with self.assertRaises(ValueError):
            repair.repair_context(183, self.spec, failed, {**source, "job_id": "other"}, self.people)
        with self.assertRaises(ValueError):
            repair.repair_context(183, self.spec, failed, source, {key: value for key, value in self.people.items() if key != "p000"})
        with self.assertRaises(ValueError):
            repair.repair_context(31, self.spec, failed, source, self.people)

    def test_one_fresh_reverse_after_baseline_replay_and_preserved_first_rescore_with_cost_once(self):
        for index in repair.IDENTITIES:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                report, calls, scores, validations, checkpoints, reads, source, failed = self.exercise(Path(directory) / "new", index=index)
            self.assertTrue(report["rival_diagnostics_complete"])
            self.assertEqual([(train, model, bound) for train, model, bound, spec in calls], [((2, 3), "MVC", "fixed")])
            self.assertEqual(calls[0][3], report["runtime_implementation"])
            self.assertEqual(validations, [("MVC", (1, 4), 1000), ("MVC", (2, 3), 1500)])
            self.assertEqual([(fit["model"], fit["training_runs"]) for fit in scores[:4]],
                             [(model, list(train)) for train, held in repair.indexed.HALVES for model in ("M0", "MZ")])
            self.assertTrue(report["first_half_fit_reused"] and report["fresh_second_half_fit_called"] and report["first_half_rescored"])
            self.assertGreaterEqual(report["first_half_rescore_elapsed_seconds"], 0.)
            for person, row in source["participant_scores"].items():
                self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
            self.assertIn(repair.failed_path(index), reads)
            for other_index in repair.IDENTITIES:
                if other_index != index:
                    self.assertNotIn(repair.failed_path(other_index), reads)
            self.assertNotIn(repair.QUALIFYING_REPORT, reads)
            self.assert_scope(report, checkpoints, failed)

    def test_bad_source_or_failed_identity_and_replay_failure_stop_before_any_new_fit(self):
        variants = ({"source_change": lambda source: source.update(fits=source["fits"][:3])},
                    {"failed_change": lambda failed: failed.update(source_finished_utc="different")}, {"failure": "replay"})
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                report, calls, scores, validations, checkpoints, reads, source, failed = self.exercise(Path(directory) / "new", **variant)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertEqual(calls, [])
            self.assertEqual(report["fits"], [])
            self.assertNotIn("participant_scores", report)
            self.assert_scope(report, checkpoints, failed)

    def test_preserved_rescore_and_fresh_reverse_failure_preserve_cost_prefix_without_scores(self):
        for failure, expected_calls, expected_fits in (("rescore", 0, 1), ("nonconvergence", 1, 2)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, scores, validations, checkpoints, reads, source, failed = self.exercise(Path(directory) / "new", failure=failure)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertEqual((len(calls), len(report["fits"])), (expected_calls, expected_fits))
            self.assertEqual(report["completed_rivals"], [])
            self.assertNotIn("participant_scores", report)
            self.assert_scope(report, checkpoints, failed)

    def test_source_fit_and_successful_finalization_deadlines_preserve_technical_closure(self):
        for failure, expected_calls in (("source_timeout", 0), ("fit_timeout", 1), ("finalization_timeout", 1)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, scores, validations, checkpoints, reads, source, failed = self.exercise(Path(directory) / "new", failure=failure)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertEqual(len(calls), expected_calls)
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertNotIn("participant_scores", report)
            self.assert_scope(report, checkpoints, failed)

    def test_compute_cap_original_path_cpu_and_fresh_source_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch.object(repair, "require_compute", side_effect=RuntimeError("compute")), self.assertRaises(RuntimeError):
                repair.run(self.arguments(output))
            with patch.object(repair, "require_compute"), patch.object(repair.indexed, "require_compute"), \
                    patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}):
                for cap in (0, 6601, True):
                    with self.subTest(cap=cap), self.assertRaises(ValueError):
                        repair.run(self.arguments(output, max_seconds=cap))
                for invalid in (0, 43, 61, 168, 184, True):
                    with self.subTest(index=invalid), self.assertRaises(ValueError):
                        repair.run(self.arguments(output, index=invalid))
                with self.assertRaises(ValueError):
                    repair.run(self.arguments(output, implementation=Path("different.json")))
                with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), self.assertRaises(ValueError):
                    repair.run(self.arguments(output))
                for protected in (Path(directory), repair.failed_path(183).parent / "new", repair.failed_path(31).parent / "new",
                                  repair.indexed.source_paths(183)[0] / "new"):
                    with self.subTest(protected=protected), self.assertRaises(FileExistsError):
                        repair.run(self.arguments(protected))


if __name__ == "__main__":
    unittest.main()
