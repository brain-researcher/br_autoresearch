"""Trace transparency checks; only cheap likelihoods, to run in coordinated Slurm."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_mz_urgency_optimizer_trace as trace
import test_mz_urgency_endpoint_diagnostic as fixtures


class UrgencyOptimizerTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.UrgencyEndpointTests.setUpClass()
        cls.fixture = fixtures.UrgencyEndpointTests()
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        cls.names, cls.starts = cls.fixture.names, cls.fixture.starts
        cls.mean = np.array([cls.spec["parameters"][name]["population_center"] for name in cls.names])
        cls.sd = np.exp(np.log([cls.spec["parameters"][name]["individual_scale_center"] for name in cls.names]))
        cls.rows = [row for row in cls.people[cls.fixture.selected["ER"]] if int(row["run"]) in (1, 4)]
        cls.data = trace.optimization.scalar.shared.make_data(cls.rows)

    def values(self, theta):
        target = self.mean + np.array([.03, -.02, .01, .04])
        return np.full(128, -1. - float(np.sum((theta-target)**2))/128.)

    def exercise(self, output, *, bad_source=False, replay_error=False, bad_probe=False,
                 bad_benchmark=False, fail_call=None, deadline_stage=None):
        source, truth = self.fixture.fixture.source()
        probe, rows = self.fixture.probe(source)
        if bad_source:
            source["replicate_index"] = 1
        if bad_probe:
            probe["best_start"]["start_index"] = 0
        reference, benchmark = {}, {"finished_utc": probe["benchmark_finished_utc"]}
        before = deepcopy((source, truth, probe, self.spec, self.people))
        shared = trace.optimization.scalar.shared
        read_text, writer, now = Path.read_text, trace.write_report, [100.]
        fixed_calls, calls, checkpoints = [], [], []
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def reader(path, *args, **kwargs):
            objects = {trace.IMPLEMENTATION: self.spec, trace.PROBE: probe,
                trace.SOURCE / trace.optimization.scalar.SOURCE_REPORT: source,
                trace.SOURCE / "generating_parameters.json": truth,
                trace.optimization.benchmark.REFERENCE: reference, trace.optimization.BENCHMARK: benchmark}
            if path in objects:
                if ((deadline_stage == "source" and objects[path] is source)
                        or (deadline_stage == "probe" and objects[path] is probe)):
                    now[0] += 1201
                return json.dumps(objects[path])
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, trace.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": original[0]["task_version"],
                    "rows": [row for row in original if int(row["run"]) in runs]}
                    for person, original in self.people.items()}
            return panel(train), panel(held)

        def fixed(theta, names, data, spec, rival):
            self.assertEqual(rival, "fixed", "The trace must not call scalar urgency")
            self.assertEqual(calls, [])
            fixed_calls.append(list(names))
            return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))

        def likelihood(theta, names, data, spec):
            self.assertEqual(len(fixed_calls), 432)
            self.assertEqual(names, self.names)
            self.assertEqual(spec, self.spec)
            for key in self.data:
                np.testing.assert_array_equal(data[key], self.data[key])
            calls.append(theta.copy())
            if len(calls) == fail_call:
                raise RuntimeError("Interrupted actual requested likelihood call")
            if deadline_stage == "likelihood" and len(calls) == 1:
                now[0] += 1201
            return self.values(theta)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["trace_complete"]:
                now[0] += 1201

        args = argparse.Namespace(source_dir=trace.SOURCE, implementation=trace.IMPLEMENTATION,
                                  output_dir=output, max_seconds=1200)
        with ExitStack() as stack:
            stack.enter_context(patch.object(trace, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {key: "1" for key in
                ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(shared, "load_split", side_effect=split))
            stack.enter_context(patch.object(shared, "trial_likelihood", side_effect=fixed))
            stack.enter_context(patch.object(trace.optimization.batched, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(trace.optimization, "validate_benchmark",
                side_effect=ValueError("Invalid completed benchmark") if bad_benchmark else None))
            stack.enter_context(patch.object(shared, "fit_partially_pooled", side_effect=AssertionError("No full fit")))
            stack.enter_context(patch.object(trace, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(trace.time, "monotonic", side_effect=lambda: now[0]))
            report = trace.run(args)
        self.assertEqual((source, truth, probe, self.spec, self.people), before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / trace.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assert_scope(record)
        return report, fixed_calls, calls, checkpoints

    def assert_scope(self, report):
        self.assertEqual(report["phase"], trace.PHASE)
        self.assertEqual((report["original_start_index"], report["local_start_index"], report["expected_starts"]), (2, 0, 1))
        self.assertEqual(report["added_likelihood_evaluations"], 0)
        self.assertEqual(report["batched_solver_adoption_scope"], "this_original_single_start_trace_only")
        for field in trace.optimization.scalar.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "observed_choices_or_RTs_used", "population_hyperparameters_updated",
                "full_cohort_model_fitted", "heldout_urgency_scores_computed", "refined_mesh_called",
                "original_48_call_benchmark_repeated", "optimizer_settings_changed",
                "endpoint_reevaluated_after_optimizer", "full_fitter_solver_adopted", "automatic_retry"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])

    def test_installed_optimizer_result_and_every_candidate_are_unchanged_by_trace(self):
        start = self.starts[2].copy()
        data, spec, names = deepcopy(self.data), deepcopy(self.spec), list(self.names)
        mean, sd = self.mean.copy(), self.sd.copy()
        originals = deepcopy((start, data, spec, names, mean, sd))
        direct_calls, traced_calls, direct_attempts, traced_attempts, evaluations, returned_arrays = [], [], [], [], [], []

        def plain(theta, actual_names, actual_data, actual_spec):
            direct_calls.append(theta.copy())
            return self.values(theta)

        def underlying(theta, actual_names, actual_data, actual_spec):
            traced_calls.append(theta.copy())
            values = self.values(theta)
            returned_arrays.append(values)
            return values

        adapter = trace.make_trace_likelihood(underlying, mean, sd, evaluations)
        def checked_adapter(*args):
            values = adapter(*args)
            self.assertIs(values, returned_arrays[-1])
            return values

        direct = trace.optimization.optimize_subject_block(data, names, spec, mean, sd, [start],
            likelihood=plain, attempts=direct_attempts)
        traced = trace.optimization.optimize_subject_block(data, names, spec, mean, sd, [start],
            likelihood=checked_adapter, attempts=traced_attempts)
        self.assertEqual(direct, traced)
        np.testing.assert_array_equal(direct_calls, traced_calls)
        self.assertGreater(len(evaluations), 1)
        self.assertEqual(len(evaluations), len(direct_calls))
        ignored = {"elapsed_seconds", "likelihood_seconds"}
        self.assertEqual({key: value for key, value in direct_attempts[0].items() if key not in ignored},
                         {key: value for key, value in traced_attempts[0].items() if key not in ignored})
        for index, (row, theta, values) in enumerate(zip(evaluations, direct_calls, returned_arrays)):
            self.assertEqual(row["evaluation_index"], index)
            np.testing.assert_array_equal(row["parameters"], theta)
            np.testing.assert_array_equal(row["trial_log_scores"], values)
            self.assertEqual(row["conditional_objective"], float(-values.sum()+.5*np.sum(((theta-mean)/sd)**2)))
            original_geometry = trace.endpoint.grid_metadata(theta, names, data, spec)
            for key in ("active", "raw_bin_index", "density_bin_index", "safe_lookup_bin_index", "response_present"):
                self.assertEqual(row["geometry"][key], [item[key] for item in original_geometry["rows"]])
        np.testing.assert_array_equal(start, originals[0])
        for key in data:
            np.testing.assert_array_equal(data[key], originals[1][key])
        self.assertEqual((spec, names), (originals[2], originals[3]))
        np.testing.assert_array_equal(mean, originals[4])
        np.testing.assert_array_equal(sd, originals[5])

    def test_adapter_preserves_nonfinite_array_identity_and_original_exception(self):
        theta = self.starts[2].copy()
        values = np.full(128, -2.)
        values[0] = np.nan
        evaluations, calls = [], []
        def likelihood(*args):
            calls.append(1)
            return values
        adapter = trace.make_trace_likelihood(likelihood, self.mean, self.sd, evaluations)
        result = adapter(theta, self.names, self.data, self.spec)
        self.assertIs(result, values)
        self.assertEqual(calls, [1])
        self.assertTrue(evaluations[0]["likelihood_returned"])
        self.assertFalse(evaluations[0]["trial_log_scores_finite"])
        self.assertIsNone(evaluations[0]["trial_log_scores"][0])
        self.assertIsNone(evaluations[0]["conditional_objective"])
        json.dumps(evaluations, allow_nan=False)
        error = ValueError("Original numerical likelihood failure")
        def failing(*args):
            raise error
        adapter = trace.make_trace_likelihood(failing, self.mean, self.sd, evaluations)
        with self.assertRaises(ValueError) as caught:
            adapter(theta, self.names, self.data, self.spec)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(evaluations), 2)
        self.assertFalse(evaluations[1]["likelihood_returned"])
        self.assertIn("Original numerical likelihood failure", evaluations[1]["likelihood_error"])

    def test_trace_matching_is_exact_retains_duplicates_and_does_not_reconcile_x_with_fun(self):
        attempt = {"parameters_finite": True, "objective_finite": True, "parameters": [1., 2.],
                   "objective": 5., "success": False}
        historical = {**attempt, "parameters": [1.25, 2.], "objective": 4.}
        evaluations = [{"evaluation_index": index, "parameters_finite": True, "parameters": theta,
            "conditional_objective_finite": objective is not None, "conditional_objective": objective}
            for index, (theta, objective) in enumerate((([1., 2.], 7.), ([1., 2.], 8.),
                ([np.nextafter(1., 2.), 2.], 5.), ([1., 2.], None)))]
        result = trace.summarize_trace(attempt, evaluations, historical)
        self.assertEqual(result["returned_parameters_matching_evaluation_indices"], [0, 1, 3])
        self.assertEqual(result["reported_objective_matching_evaluation_indices"], [2])
        self.assertEqual(result["returned_parameters_and_reported_objective_matching_evaluation_indices"], [])
        self.assertEqual([row["objective_minus_reported_fun"] for row in result["recorded_objectives_at_returned_parameters"]],
                         [2., 3., None])
        self.assertEqual(result["historical_parameter_differences"], [-.25, 0.])
        self.assertEqual(result["historical_objective_difference"], 1.)
        self.assertFalse(attempt["success"])

    def test_full_source_replay_then_only_original_start2_as_local0(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fixed, calls, checkpoints = self.exercise(Path(directory) / "new")
        self.assertTrue(report["trace_complete"])
        self.assertEqual(len(fixed), 432)
        self.assertEqual((len(report["attempts"]), report["completed_starts"]), (1, 1))
        self.assertEqual(report["attempts"][0]["start_index"], 0)
        self.assertEqual(report["historical_start"]["start_index"], 2)
        np.testing.assert_array_equal(calls[0], self.starts[2])
        self.assertEqual(report["original_start_parameters"], self.starts[2].tolist())
        self.assertEqual(report["likelihood_calls_attempted"], len(calls))
        self.assertEqual(report["recorded_evaluations"], len(calls))
        self.assertEqual(report["recorded_likelihood_returns"], len(calls))
        self.assertEqual(report["attempts"][0]["nfev"], len(calls))
        self.assertEqual([row["parameters"] for row in report["evaluations"]], [theta.tolist() for theta in calls])
        # Only helper checkpoints publish: the logger does not write once per evaluation.
        self.assertLess(len(checkpoints), len(calls))

    def test_source_replay_benchmark_and_terminal_probe_rejection_precede_optimizer(self):
        for options in ({"bad_source": True}, {"replay_error": True}, {"bad_benchmark": True}, {"bad_probe": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                report, fixed, calls, _ = self.exercise(Path(directory) / "new", **options)
            self.assertFalse(report["trace_complete"])
            self.assertEqual(calls, [])
            self.assertEqual(report["attempts"], [])
            self.assertEqual(report["evaluations"], [])
            self.assertFalse(report["optimizer_called"])

    def test_interruption_and_whole_deadline_retain_only_actual_evaluation_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, calls, _ = self.exercise(Path(directory) / "new", fail_call=2)
        self.assertFalse(report["trace_complete"])
        self.assertEqual((len(calls), report["recorded_evaluations"], report["recorded_likelihood_returns"]), (2, 2, 1))
        self.assertFalse(report["attempts"][0]["returned"])
        self.assertTrue(report["evaluations"][0]["objective_arithmetic_recorded"])
        self.assertFalse(report["evaluations"][1]["likelihood_returned"])
        self.assertIn("interrupted", report["evaluations"][1]["likelihood_error"].lower())
        self.assertIsNone(report["comparison"])
        for stage in ("source", "probe", "likelihood", "finalization"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, _, calls, _ = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertFalse(report["trace_complete"])
            self.assertIn("whole-process deadline", report["incomplete_reason"])
            self.assertEqual(report["recorded_evaluations"], len(calls))
            self.assertEqual(report["recorded_likelihood_returns"], len(calls))
            if stage in ("source", "probe"):
                self.assertEqual(calls, [])
            elif stage == "likelihood":
                self.assertEqual(len(calls), 1)
            else:
                self.assertGreater(len(calls), 1)
                self.assertTrue(report["attempts"][0]["returned"])


if __name__ == "__main__":
    unittest.main()
