"""Focused saved-scalar benchmark checks; execute only in task-owned compute."""
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

import numpy as np

import run_mz_urgency_batched_benchmark as benchmark
import test_mz_urgency_cost_precision_pilot as fixtures


class UrgencyBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.UrgencyPilotTests.setUpClass()
        cls.fixture = fixtures.UrgencyPilotTests(
            "test_full_source_replay_precedes_exact24_training_probe_pairs_without_any_fit")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        with tempfile.TemporaryDirectory() as directory:
            cls.reference, *_ = cls.fixture.exercise(Path(directory) / "scalar")
        cls.reference["job_id"] = benchmark.REFERENCE_JOB

    def args(self, output, **changes):
        values = dict(source_dir=benchmark.SOURCE, implementation=benchmark.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=600)
        values.update(changes)
        return argparse.Namespace(**values)

    def exercise(self, output, *, reference_change=None, bad_source=False, replay_error=False,
                 fail_call=None, nonfinite=False, difference=0., deadline_stage=None):
        source, truth = self.fixture.fixture.source()
        reference = deepcopy(self.reference)
        if bad_source:
            source["replicate_index"] = 1
        if reference_change is not None:
            reference_change(reference)
        before = deepcopy((source, truth, reference, self.spec, self.people))
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        read_text, writer, now = Path.read_text, benchmark.write_report, [100.]
        fixed_calls, calls, checkpoints, reads = [], [], [], []

        def reader(path, *args, **kwargs):
            if path == benchmark.IMPLEMENTATION:
                reads.append(path)
                return json.dumps(self.spec)
            if path == benchmark.REFERENCE:
                reads.append(path)
                return json.dumps(reference, allow_nan=True)
            if path == benchmark.SOURCE / benchmark.scalar.SOURCE_REPORT:
                reads.append(path)
                if deadline_stage == "source":
                    now[0] += 601
                return json.dumps(source)
            if path == benchmark.SOURCE / "generating_parameters.json":
                reads.append(path)
                return json.dumps(truth)
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, benchmark.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(train), panel(held)

        def fixed_likelihood(theta, names, data, spec, rival):
            self.assertEqual(rival, "fixed", "Never rerun the scalar urgency likelihood")
            self.assertEqual(spec, self.spec)
            self.assertEqual(calls, [])
            fixed_calls.append(tuple(names))
            return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))

        def candidate(theta, names, data, spec):
            self.assertEqual(len(fixed_calls), 432)
            probe_index, resolution = divmod(len(calls), 2)
            mesh = "refined" if resolution else "coarse"
            expected = reference["probes"][probe_index]
            np.testing.assert_array_equal(theta, expected["parameters"])
            self.assertEqual(names, expected["parameter_names"])
            self.assertEqual(spec, reference["refined_numerical_implementation"] if resolution else self.spec)
            rows = [row for row in self.people[expected["participant"]] if int(row["run"]) in expected["training_runs"]]
            actual_data = benchmark.scalar.shared.make_data(rows)
            for key in actual_data:
                np.testing.assert_array_equal(data[key], actual_data[key])
            calls.append((probe_index, mesh))
            if len(calls) == fail_call:
                if nonfinite:
                    return np.full(128, np.nan)
                raise RuntimeError("Candidate call interrupted")
            values = np.asarray(expected["meshes"][mesh]["trial_log_scores"]).copy()
            values[0] += difference
            if deadline_stage == "likelihood" and len(calls) == 1:
                now[0] += 601
            return values

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["benchmark_complete"]:
                now[0] += 601

        with ExitStack() as stack:
            stack.enter_context(patch.object(benchmark, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(benchmark.scalar.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(benchmark.scalar.shared, "load_split", side_effect=split))
            stack.enter_context(patch.object(benchmark.scalar.shared, "trial_likelihood", side_effect=fixed_likelihood))
            stack.enter_context(patch.object(benchmark.batched, "trial_likelihood", side_effect=candidate))
            stack.enter_context(patch.object(benchmark.scalar.shared, "fit_partially_pooled", side_effect=AssertionError("No model fit")))
            stack.enter_context(patch.object(benchmark.scalar.behavior, "minimize", side_effect=AssertionError("No optimizer")))
            stack.enter_context(patch.object(benchmark, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(benchmark.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = benchmark.run(self.args(output))
        self.assertEqual((source, truth, reference, self.spec, self.people), before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / benchmark.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        self.assertFalse((output / "generating_parameters.json").exists())
        for record in checkpoints:
            self.assert_scope(record)
        return report, fixed_calls, calls, checkpoints, reads

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["replicate_index"], report["seed"], report["rival"]),
                         (benchmark.PHASE, 0, 910031, "urgency"))
        self.assertEqual(report["reference_report"], str(benchmark.REFERENCE))
        self.assertEqual(report["reference_job_id"], "46607960")
        self.assertEqual(report["engineering_absolute_tolerance"], 1e-12)
        for field in benchmark.scalar.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "observed_choices_or_RTs_used",
                      "scalar_urgency_calls_repeated", "optimizer_called", "model_fitted", "actual_fitted_support_evaluated",
                      "solver_adopted", "wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        for field in ("participant_scores", "comparison", "nested_diagnostic", "valid_recovery_component"):
            self.assertNotIn(field, report)

    def test_full_source_replay_and_saved48_vectors_supply_exact_mapping_and_timing_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fixed, calls, checkpoints, reads = self.exercise(Path(directory) / "new")
        self.assertTrue(report["benchmark_complete"])
        self.assertTrue(report["scalar_reference_validated"])
        self.assertTrue(report["engineering_equivalent_at_1e_minus12"])
        self.assertEqual((len(fixed), len(calls), report["completed_likelihood_calls"]), (432, 48, 48))
        self.assertEqual(report["maximum_absolute_trial_log_score_discrepancy"], 0.)
        for record in report["calls"]:
            saved = self.reference["probes"][record["probe_index"]]["meshes"][record["mesh"]]
            self.assertEqual(record["trial_log_scores"], saved["trial_log_scores"])
            self.assertEqual(record["scalar_reference_elapsed_seconds"], saved["elapsed_seconds"])
            self.assertEqual(record["measured_scalar_to_candidate_time_ratio"], saved["elapsed_seconds"] / record["candidate_elapsed_seconds"])
            self.assertEqual(record["mean_log_score_difference"], 0.)
        self.assertEqual(set(reads), {benchmark.REFERENCE, benchmark.IMPLEMENTATION,
            benchmark.SOURCE / benchmark.scalar.SOURCE_REPORT, benchmark.SOURCE / "generating_parameters.json"})
        self.assertEqual(report["scalar_process_elapsed_seconds"], self.reference["elapsed_seconds"])
        self.assertNotIn("active_call", report)

    def test_engineering_mismatch_is_measured_without_adoption_or_an_inference_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            report, *_ = self.exercise(Path(directory) / "new", difference=1e-10)
        self.assertTrue(report["benchmark_complete"])
        self.assertFalse(report["engineering_equivalent_at_1e_minus12"])
        for record in report["calls"]:
            self.assertGreater(record["maximum_absolute_trial_log_score_discrepancy"], 1e-12)
            self.assertAlmostEqual(record["mean_log_score_difference"] * 128,
                                   record["maximum_absolute_trial_log_score_discrepancy"], places=15)
        self.assert_scope(report)

    def test_incomplete_wrong_source_or_malformed_scalar_panel_prevents_candidate_calls(self):
        changes = (
            lambda report: report.update(pilot_complete=False),
            lambda report: report.update(job_id="46607961"),
            lambda report: report.update(source_finished_utc="2026-10-05T00:00:00+00:00"),
            lambda report: report["saved_primary_participant_scores"]["p000"].update(M0=-2.),
            lambda report: report["probes"][0].update(training_runs=[1, 2]),
            lambda report: report["probes"][0]["parameters"].__setitem__(0, 0.),
            lambda report: report["probes"][0]["meshes"]["refined"].update(trial_log_scores=[0.] * 127),
            lambda report: report["probes"][0]["meshes"]["coarse"].update(elapsed_seconds=0.),
            lambda report: report["probes"][0]["mesh_difference"].update(mean_log_score_difference=9.),
        )
        for number, change in enumerate(changes):
            with self.subTest(change=number), tempfile.TemporaryDirectory() as directory:
                report, fixed, calls, checkpoints, reads = self.exercise(Path(directory) / "new", reference_change=change)
            self.assertFalse(report["benchmark_complete"])
            self.assertFalse(report["scalar_reference_validated"])
            self.assertEqual(calls, [])
            self.assertEqual(report["completed_likelihood_calls"], 0)
            self.assertTrue(report["incomplete_reason"])

    def test_actual_source_or_fixed_replay_failure_precedes_every_candidate_call(self):
        for options in ({"bad_source": True}, {"replay_error": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                report, fixed, calls, checkpoints, reads = self.exercise(Path(directory) / "new", **options)
            self.assertFalse(report["benchmark_complete"])
            self.assertFalse(report["scalar_reference_validated"])
            self.assertEqual(calls, [])
            self.assertIsNone(report["engineering_equivalent_at_1e_minus12"])

    def test_candidate_exception_or_nonfinite_retains_completed_prefix_and_active_identity(self):
        for nonfinite in (False, True):
            with self.subTest(nonfinite=nonfinite), tempfile.TemporaryDirectory() as directory:
                report, *_ = self.exercise(Path(directory) / "new", fail_call=2, nonfinite=nonfinite)
            self.assertFalse(report["benchmark_complete"])
            self.assertEqual(report["completed_likelihood_calls"], 1)
            self.assertEqual(report["active_call"], {"probe_index": 0, "mesh": "refined"})
            self.assertEqual(len(report["calls"][0]["trial_log_scores"]), 128)
            self.assertIsNone(report["engineering_equivalent_at_1e_minus12"])

    def test_whole_cap_includes_source_call_and_finalization_and_preserves_finished_calls(self):
        for stage, count in (("source", 0), ("likelihood", 1), ("finalization", 48)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, *_ = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertFalse(report["benchmark_complete"])
            self.assertIn("whole-process deadline", report["incomplete_reason"])
            self.assertEqual(report["completed_likelihood_calls"], count)
            self.assertIsNone(report["engineering_equivalent_at_1e_minus12"])
            self.assertTrue(report["finished_utc"])

    def test_fixed_paths_cpu_cap_fresh_output_guards_and_cli(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for change in ({"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"},
                           {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": 601}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    benchmark.run(self.args(Path(directory) / "new", **change))
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), self.assertRaises(ValueError):
                benchmark.run(self.args(Path(directory) / "new"))
            for output in (Path(directory), benchmark.SOURCE / "new", benchmark.REFERENCE.parent / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    benchmark.run(self.args(output))
        with patch.object(benchmark, "run", return_value={"benchmark_complete": True}) as runner, redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark.main(["--source-dir", str(benchmark.SOURCE), "--implementation", str(benchmark.IMPLEMENTATION),
                "--output-dir", "/tmp/batched-benchmark-fresh", "--max-seconds", "600"]), 0)
        self.assertEqual(vars(runner.call_args.args[0]), vars(self.args("/tmp/batched-benchmark-fresh")))


if __name__ == "__main__":
    unittest.main()
