"""Focused null108/176 completion identity/binding tests; coordinated compute only."""
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

import run_mz_null_additional_baseline_completion as completion
import test_mz_null_baseline_completion as fixtures
from test_mz_null_representative import THREADS


class AdditionalNullBaselineCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullBaselineCompletionTests.setUpClass()
        cls.source_fixture = fixtures.NullBaselineCompletionTests(
            "test_context_validates_original_three_fit_prefix_without_changing_history_or_cost")
        cls.fixture = cls.source_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def source(self, index):
        source = self.source_fixture.source()
        worker, finished = completion.SOURCES[index]
        source.update(replicate_index=index, seed=completion.base.indexed.null_seed(index),
                      job_id=worker, finished_utc=finished, selected_replicates=[index])
        return source

    def arguments(self, output, index=108, **changes):
        values = dict(replicate_index=index, implementation=completion.base.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=2700)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return tuple(getattr(completion.base, name) for name in ("REPLICATE_INDEX", "SEED", "SOURCE",
                     "SOURCE_JOB_ID", "SOURCE_FINISHED_UTC", "PHASE", "REPORT_NAME", "write_report"))

    def exercise(self, output, index, *, source_change=None, failure=None):
        base = completion.base
        source = self.source(index)
        if source_change:
            source_change(source)
        preserved = deepcopy((source, self.spec, self.truth))
        source_dir = completion.source_directory(index)
        allowed = {base.IMPLEMENTATION: json.dumps(self.spec),
                   source_dir / base.SOURCE_REPORT: json.dumps(source),
                   source_dir / "generating_parameters.json": json.dumps(self.truth)}
        calls, scores, checkpoints, reads = [], [], [], []
        writer = base.write_report
        original_bindings = self.bindings()

        def reader(path, *args, **kwargs):
            reads.append(path)
            return allowed[path]

        def split(events, train, held):
            self.assertEqual(events, source_dir / "simulated_events.csv.gz")
            return tuple(train), tuple(held)

        def fitter(training, model, bound, spec):
            calls.append((training, model, bound, deepcopy(spec)))
            if failure == "interrupt":
                raise RuntimeError("Synthetic interruption of the single fresh reverse fit")
            return self.fixture.fit(model, converged=failure != "nonconvergence")

        def scorer(fit, scoring, spec, check):
            check()
            self.assertEqual(scoring, tuple(fit["scoring_runs"]))
            self.assertEqual(spec, self.spec)
            scores.append(deepcopy(fit))
            if failure == "rescore":
                raise ValueError("Synthetic preserved-fit scoring failure")
            return {person: (-1. if fit["model"] == "M0" else -.99) for person in self.people}

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(completion, "require_compute"))
            stack.enter_context(patch.object(base, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            loader = stack.enter_context(patch.object(base.null.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(base, "load_split", side_effect=split))
            stack.enter_context(patch.object(base, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(base.null.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(base, "write_report", side_effect=checkpoint))
            simulator = stack.enter_context(patch.object(base.indexed.representative, "simulate_null",
                                                        side_effect=AssertionError("No new simulation")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            result = completion.run(self.arguments(output, index))
            self.assertEqual(self.bindings(), installed)
            simulator.assert_not_called()
            self.assertEqual(loader.call_args.args[0], source_dir / "simulated_events.csv.gz")
        self.assertEqual(self.bindings(), original_bindings)
        self.assertEqual((source, self.spec, self.truth), preserved)
        self.assertEqual(json.loads((output / completion.REPORT_NAME).read_text()), result)
        self.assertEqual([path.name for path in output.iterdir()], [completion.REPORT_NAME])
        self.assertEqual(reads, [base.IMPLEMENTATION, source_dir / base.SOURCE_REPORT,
                                 source_dir / "generating_parameters.json"])
        return result, calls, scores, checkpoints, source

    def assert_scope(self, report, index):
        worker, finished = completion.SOURCES[index]
        self.assertEqual((report["phase"], report["generating_model"], report["replicate_index"], report["seed"]),
                         (completion.PHASE, "M0", index, 1300037 + 10007 * index))
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual(report["source_job_id"], worker)
        self.assertEqual(report["source_dir"], str(completion.source_directory(index)))
        self.assertIn(f"null{index}'s", report["interpretation"])
        self.assertNotIn("null35", report["interpretation"])
        self.assertEqual(report["fixed_cohort_count"], 200)
        for field in ("wave_selected", "retry_wave_selected", "full_menu_wave_selected", "automatic_retry",
                      "simulation_regenerated", "simulation_copied", "saved_completed_fits_refitted",
                      "unfinished_fit_state_reused", "episode_terminal"):
            self.assertFalse(report[field])
        for field in (*completion.base.null.CLAIM_FIELDS, "early_kill"):
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        if report["source_context_validated"]:
            self.assertEqual(report["source_finished_utc"], finished)
            self.assertEqual(report["implementation"], self.spec)
            self.assertTrue(all(row["validation_maximum_sweeps"] == 500 for row in report["per_fit_provenance"]))
            self.assertTrue(all(row["source_job_id"] == worker for row in report["per_fit_provenance"][:3]))

    def test_only_exact_two_source_identities_are_accepted_and_old_bindings_restore(self):
        original_bindings = self.bindings()
        for index in completion.SOURCES:
            source = self.source(index)
            before = deepcopy(source)
            fits, metadata = completion.completion_context(index, source, self.truth, self.people, self.spec)
            self.assertEqual(fits, source["fits"])
            self.assertEqual(source, before)
            self.assertEqual(metadata["source_finished_utc"], completion.SOURCES[index][1])
            self.assertEqual(self.bindings(), original_bindings)
            for field, value in (("job_id", "46580478"), ("finished_utc", "2026-10-05T06:46:38.259015+00:00"),
                                 ("replicate_index", 35), ("seed", 1650282), ("error", "Different failure")):
                with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                    completion.completion_context(index, {**source, field: value}, self.truth, self.people, self.spec)
                self.assertEqual(self.bindings(), original_bindings)
        for index in (0, 35, 107, 109, 175, 177, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                completion.source_directory(index)

    def test_each_cohort_reuses_exact_three_fits_and_fits_only_reverse_MZ_under_original500(self):
        for index in completion.SOURCES:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                report, calls, scores, checkpoints, source = self.exercise(Path(directory) / "new", index)
            self.assertTrue(report["pairwise_diagnostic_complete"])
            self.assertEqual(calls, [((2, 3), "MZ", "fixed", self.spec)])
            self.assertEqual(report["fits"][:3], source["fits"])
            self.assertEqual(scores[:3], source["fits"])
            self.assertTrue(report["preserved_scores_recomputed"])
            self.assertEqual(report["cumulative_process_elapsed_seconds"], source["elapsed_seconds"] + report["elapsed_seconds"])
            self.assertEqual(report["participant_scores"]["p000"]["M0"], -2.)
            self.assertEqual(report["participant_scores"]["p000"]["MZ"], -1.98)
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, index)
            with self.assertRaises(ValueError):
                completion.base.baseline.validate_identity(report, index, self.spec)

    def test_actual_participants_truth_and_original_fit_ceiling_remain_required(self):
        for index in completion.SOURCES:
            for mismatch in ("people", "truth", "ceiling"):
                source = self.source(index)
                truth = deepcopy(self.truth)
                if mismatch == "people":
                    source["fits"][0]["participants"][0] = "other"
                elif mismatch == "truth":
                    truth["p000"]["rho"] = .1
                else:
                    source["fits"][0]["objective_history"] = [1.] * 501
                with self.subTest(index=index, mismatch=mismatch), self.assertRaises(ValueError):
                    completion.completion_context(index, source, truth, self.people, self.spec)

    def test_source_and_rescore_failure_preserve_technical_attempt_without_new_fit(self):
        for index in completion.SOURCES:
            for failure in ("source", "rescore"):
                with self.subTest(index=index, failure=failure), tempfile.TemporaryDirectory() as directory:
                    report, calls, scores, checkpoints, source = self.exercise(Path(directory) / "bad", index,
                        source_change=(lambda source: source.update(job_id="46580478")) if failure == "source" else None,
                        failure="rescore" if failure == "rescore" else None)
                self.assertEqual(calls, [])
                self.assertEqual(report["status"], "technical_baseline_completion_failure")
                self.assertFalse(report["pairwise_diagnostic_complete"])
                self.assertNotIn("participant_scores", report)
                self.assertNotIn("nested_comparison", report)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, index)

    def test_fresh_reverse_failure_keeps_original_prefix_and_consumed_cost_without_diagnostics(self):
        for index in completion.SOURCES:
            for failure in ("interrupt", "nonconvergence"):
                with self.subTest(index=index, failure=failure), tempfile.TemporaryDirectory() as directory:
                    report, calls, scores, checkpoints, source = self.exercise(Path(directory) / "failed", index, failure=failure)
                self.assertEqual(len(calls), 1)
                self.assertEqual(report["fits"][:3], source["fits"])
                self.assertEqual(len(report["fits"]), 3 if failure == "interrupt" else 4)
                self.assertEqual(report["status"], "technical_baseline_completion_failure")
                self.assertFalse(report["pairwise_diagnostic_complete"])
                self.assertNotIn("participant_scores", report)
                self.assertNotIn("nested_comparison", report)
                self.assertEqual(report["cumulative_process_elapsed_seconds"], source["elapsed_seconds"] + report["elapsed_seconds"])
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint, index)

    def test_compute_cpu_cap_and_protected_outputs_keep_old35_and_restore_bindings_on_guard_failure(self):
        original_bindings = self.bindings()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read before guards")):
            with patch.object(completion, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                completion.run(self.arguments(Path(directory) / "new"))
            with patch.object(completion, "require_compute"), patch.object(completion.base, "require_compute"):
                for changes in ({"replicate_index": 35}, {"replicate_index": 107}, {"replicate_index": True},
                                {"implementation": Path(directory) / "wrong.json"}, {"max_seconds": 0},
                                {"max_seconds": 2701}, {"max_seconds": True}):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        completion.run(self.arguments(Path(directory) / "new", **changes))
                    self.assertEqual(self.bindings(), original_bindings)
                for name in ("SLURM_CPUS_PER_TASK", *THREADS):
                    with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                        completion.run(self.arguments(Path(directory) / "new"))
                    self.assertEqual(self.bindings(), original_bindings)
                for output in (Path(directory), *(path / "new" for path in completion.PRESERVED_DIRECTORIES),
                               completion.base.indexed.PILOT_DIRECTORY / "new", completion.base.DESIGN_SOURCE.parent / "new"):
                    with self.subTest(output=output), self.assertRaises(FileExistsError):
                        completion.run(self.arguments(output))
                    self.assertEqual(self.bindings(), original_bindings)


if __name__ == "__main__":
    unittest.main()
