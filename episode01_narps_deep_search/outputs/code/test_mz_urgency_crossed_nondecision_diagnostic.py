"""Focused five-call crossed diagnostics; mocked likelihoods, Slurm only."""
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

import run_mz_urgency_crossed_nondecision_diagnostic as diagnostic
import test_mz_urgency_endpoint_diagnostic as fixtures


class UrgencyCrossedNondecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.UrgencyEndpointTests.setUpClass()
        cls.fixture = fixtures.UrgencyEndpointTests()
        cls.spec, cls.people, cls.names = cls.fixture.spec, cls.fixture.people, cls.fixture.names
        cls.mean = np.array([cls.spec["parameters"][name]["population_center"] for name in cls.names])
        cls.sd = np.exp(np.log([cls.spec["parameters"][name]["individual_scale_center"] for name in cls.names]))
        cls.rows = [row for row in cls.people[cls.fixture.selected["ER"]] if int(row["run"]) in (1, 4)]
        cls.data = diagnostic.optimization.scalar.shared.make_data(cls.rows)
        cls.a = np.array(diagnostic.endpoint.ENDPOINT)
        cls.b = cls.a + np.array([.03, -.02, .015, .04])

    def values(self, theta):
        # An explicit background-by-nondecision interaction makes the crossing
        # signs testable without any DDM solver or optimizer evaluation.
        return np.arange(128, dtype=float)/512 - 2 + .125*theta[0] + .25*theta[2] + .5*theta[0]*theta[2]

    def anchors(self, source, probe, rows):
        identity, _ = diagnostic.optimization.probe_identity(2, self.fixture.selected)
        common = {**identity, "source_dir": str(diagnostic.SOURCE), "source_job_id": source["job_id"],
            "source_finished_utc": source["finished_utc"], "implementation": deepcopy(self.spec),
            "parameter_names": list(self.names), "conditional_prior_mean": self.mean.tolist(),
            "conditional_prior_sd": self.sd.tolist(),
            "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
            "probe_report": str(diagnostic.PROBE), "benchmark_report": str(diagnostic.optimization.BENCHMARK),
            "scalar_reference_report": str(diagnostic.optimization.benchmark.REFERENCE)}
        geometry = diagnostic.endpoint.grid_metadata(self.a, self.names, self.data, self.spec)
        evaluations = [None]*490
        for index, theta, offset in ((270, self.a, 0.), (485, self.b, .0002)):
            values = self.values(theta)+offset
            evaluations[index] = {"evaluation_index": index, "parameters": theta.tolist(),
                "geometry": diagnostic.trace_producer.compact_geometry(theta, self.names, self.data, self.spec),
                "trial_log_scores": values.tolist(),
                "nll": float(-values.sum()), "prior_penalty": float(.5*np.sum(((theta-self.mean)/self.sd)**2)),
                "conditional_objective": float(-values.sum()+.5*np.sum(((theta-self.mean)/self.sd)**2))}
        trace = {**deepcopy(common), "job_id": "46617879", "historical_start": deepcopy(probe["attempts"][2]),
            "finished_utc": "2026-10-05T13:25:00+00:00", "evaluations": evaluations}
        a = evaluations[270]
        reference = {**deepcopy(common), "phase": diagnostic.endpoint.PHASE, "job_id": "46615238",
            "status": "urgency_endpoint_diagnostic_completed", "diagnostic_complete": True,
            "source_baseline_validated": True, "saved_primary_scores_replayed": True,
            "benchmark_validated": True, "terminal_probe_validated": True,
            "attempted_likelihood_calls": 6, "completed_likelihood_calls": 6,
            "scientific_optimizer_called": False, "refined_mesh_called": False,
            "started_utc": "2026-10-05T12:50:00+00:00", "finished_utc": "2026-10-05T12:51:00+00:00",
            "points": [{"name": "endpoint", "parameters": self.a.tolist(), "coordinate_delta": 0.,
                "geometry": geometry, "calls": {solver: {"returned": True, "finite128_vector": True,
                    "trial_log_scores": deepcopy(a["trial_log_scores"]),
                    "conditional_objective": a["conditional_objective"]} for solver in ("scalar", "batched")}},
                {"name": "optimizer_perturbation"}, {"name": "opposite_diagnostic"}]}
        return trace, reference

    def exercise(self, output, *, bad_source=False, replay_error=False, anchor_change=None,
                 fail_call=None, deadline_stage=None):
        source, truth = self.fixture.fixture.source()
        probe, rows = self.fixture.probe(source)
        trace, reference = self.anchors(source, probe, rows)
        if bad_source:
            source["replicate_index"] = 1
        if anchor_change:
            anchor_change(trace, reference)
        before = deepcopy((source, truth, probe, trace, reference, self.spec, self.people))
        shared = diagnostic.optimization.scalar.shared
        writer, now = diagnostic.write_report, [100.]
        calls, fixed_calls, checkpoints, reads = [], [], [], []
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        c, d = diagnostic.crossed_points(self.a.tolist(), self.b.tolist(), self.names)
        expected_points = {"B": self.b.tolist(), "C": c, "D": d}
        objects = {diagnostic.IMPLEMENTATION: self.spec, diagnostic.PROBE: probe,
            diagnostic.SOURCE / diagnostic.optimization.scalar.SOURCE_REPORT: source,
            diagnostic.SOURCE / "generating_parameters.json": truth,
            diagnostic.optimization.benchmark.REFERENCE: {},
            diagnostic.optimization.BENCHMARK: {"finished_utc": probe["benchmark_finished_utc"]},
            diagnostic.TRACE: trace, diagnostic.A_REFERENCE: reference}

        def reader(path, *args, **kwargs):
            if path not in objects:
                raise AssertionError(f"No other input read: {path}")
            reads.append(path)
            if ((deadline_stage == "source" and path == diagnostic.SOURCE / diagnostic.optimization.scalar.SOURCE_REPORT)
                    or (deadline_stage == "anchors" and path == diagnostic.A_REFERENCE)):
                now[0] += 181
            return json.dumps(objects[path])

        def split(path, train, held):
            self.assertEqual(path, diagnostic.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": original[0]["task_version"],
                    "rows": [row for row in original if int(row["run"]) in runs]}
                    for person, original in self.people.items()}
            return panel(train), panel(held)

        def candidate(solver, theta, names, data, spec):
            self.assertEqual(len(fixed_calls), 432)
            self.assertEqual((names, spec), (self.names, self.spec))
            for key in self.data:
                np.testing.assert_array_equal(data[key], self.data[key])
            name, expected_solver = diagnostic.CALLS[len(calls)]
            self.assertEqual(solver, expected_solver)
            np.testing.assert_array_equal(theta, expected_points[name])
            self.assertFalse(np.array_equal(theta, self.a), "Never reevaluate saved A")
            calls.append({"point": name, "solver": solver, "parameters": theta.tolist()})
            if len(calls) == fail_call:
                raise RuntimeError("Selected new call interrupted")
            if deadline_stage == "likelihood" and len(calls) == 2:
                now[0] += 181
            return self.values(theta)+(.001 if solver == "batched" else 0.)

        def likelihood(theta, names, data, spec, rival):
            if rival == "fixed":
                self.assertEqual(calls, [])
                fixed_calls.append(list(names))
                return np.full(128, (-1.98 if "rho" in names else -2.)+(1e-8 if replay_error else 0.))
            self.assertEqual(rival, "urgency")
            return candidate("scalar", theta, names, data, spec)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if (deadline_stage == "staged_call" and report["attempted_new_likelihood_calls"] == 1
                    and not report["points"]["B"]["calls"]["scalar"]["returned"]):
                now[0] += 181
            if deadline_stage == "finalization" and report["diagnostic_complete"]:
                now[0] += 181

        def saved_trace_validator(actual, check):
            # Already-qualified490-call identity/packet validation is outside
            # this new test scope; retain the new anchor integration itself.
            check()
            return actual["evaluations"], actual["trial_slots"]

        args = argparse.Namespace(source_dir=diagnostic.SOURCE, implementation=diagnostic.IMPLEMENTATION,
                                  output_dir=output, max_seconds=180)
        with ExitStack() as stack:
            stack.enter_context(patch.object(diagnostic, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {key: "1" for key in
                (*diagnostic.trace_analysis.THREAD_ENV, "SLURM_CPUS_PER_TASK")}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(shared, "load_split", side_effect=split))
            stack.enter_context(patch.object(shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(diagnostic.optimization.batched, "trial_likelihood",
                side_effect=lambda *args: candidate("batched", *args)))
            stack.enter_context(patch.object(diagnostic.optimization, "validate_benchmark"))
            stack.enter_context(patch.object(diagnostic.trace_analysis, "validate_trace", side_effect=saved_trace_validator))
            stack.enter_context(patch.object(shared, "fit_partially_pooled", side_effect=AssertionError("No model fit")))
            stack.enter_context(patch.object(diagnostic.optimization, "optimize_subject_block",
                side_effect=AssertionError("No optimizer call")))
            stack.enter_context(patch.object(diagnostic, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(diagnostic.time, "monotonic", side_effect=lambda: now[0]))
            report = diagnostic.run(args)
        self.assertEqual((source, truth, probe, trace, reference, self.spec, self.people), before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / diagnostic.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assert_scope(record)
        return report, calls, fixed_calls, trace, reference, reads

    def assert_scope(self, report):
        self.assertEqual(report["phase"], diagnostic.PHASE)
        self.assertEqual(report["expected_new_likelihood_calls"], 5)
        for field in ("A_reevaluated", "optimizer_called", "optimizer_settings_changed", "mesh_changed",
                "simulation_regenerated", "observed_choices_or_RTs_used", "population_hyperparameters_updated",
                "heldout_urgency_scores_computed", "full_fitter_solver_adopted", "automatic_retry"):
            self.assertIs(report[field], False)
        for field in diagnostic.optimization.scalar.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        for field in ("participant_scores", "nested_diagnostic", "numerical_precision_passed", "optimizer_qualified"):
            self.assertNotIn(field, report)

    def test_crossing_changes_only_nondecision_and_signed_interaction_keeps_prior_separate(self):
        a, b = [.1, .2, -2., .5], [.3, .4, -1., .6]
        before = deepcopy((a, b, self.names))
        c, d = diagnostic.crossed_points(a, b, self.names)
        self.assertEqual(c, [.1, .2, -1., .5])
        self.assertEqual(d, [.3, .4, -2., .6])
        self.assertEqual((a, b, self.names), before)
        with self.assertRaises(ValueError):
            diagnostic.crossed_points(a, d, self.names)
        points = {}
        scalar_values = {"A": -2., "B": -1., "C": -1.75, "D": -1.5}
        offsets = {"A": 0., "B": .0625, "C": .03125, "D": -.03125}
        for name, theta in zip(("A", "B", "C", "D"), (a, b, c, d)):
            calls = {}
            prior = float(.5*np.sum(((np.asarray(theta)-self.mean)/self.sd)**2))
            for solver in ("scalar", "batched"):
                values = np.full(128, scalar_values[name]+(offsets[name] if solver == "batched" else 0.))
                calls[solver] = {"trial_log_scores": values.tolist(), "nll": float(-values.sum()),
                    "prior_penalty": prior, "conditional_objective": float(-values.sum()+prior)}
            points[name] = {"parameters": theta, "calls": calls,
                "geometry": diagnostic.trace_producer.compact_geometry(np.asarray(theta), self.names, self.data, self.spec)}
        slots = [[int(row["run"]), int(row["trial_order"])] for row in self.rows]
        summary = diagnostic.summarize_crossed(points, self.mean, self.sd, slots)
        self.assertEqual(summary["interaction"]["scalar"]["trial_log_score_interaction"], [.25]*128)
        self.assertEqual(summary["interaction"]["scalar"]["negative_sum_trial_log_score_interaction"], -32.)
        for solver in ("scalar", "batched"):
            comparisons = summary["within_background_t0_comparisons"][solver]
            self.assertEqual([(row["left_point"], row["right_point"]) for row in comparisons], [("A", "C"), ("D", "B")])
            for key in ("nll", "prior_penalty", "conditional_objective"):
                expected = ((points["B"]["calls"][solver][key]-points["D"]["calls"][solver][key])
                    -(points["C"]["calls"][solver][key]-points["A"]["calls"][solver][key]))
                self.assertEqual(summary["interaction"][solver][key], expected)
        for name in ("B", "C", "D"):
            self.assertEqual(summary["scalar_batched_comparisons"][name]["batched_minus_scalar_trial_log_scores"],
                             [offsets[name]]*128)
        self.assertIn("do not isolate the bin lookup operator", summary["interpretation"])

    def test_original_source_replay_then_exact_five_calls_with_saved_anchor_and_prior_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, fixed, trace, reference, reads = self.exercise(Path(directory)/"new")
        self.assertTrue(report["diagnostic_complete"])
        self.assertEqual(len(fixed), 432)
        self.assertEqual([(row["point"], row["solver"]) for row in calls], list(diagnostic.CALLS))
        self.assertEqual(report["new_call_order"], [{"point": name, "solver": solver} for name, solver in diagnostic.CALLS])
        self.assertEqual((report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]), (5, 5))
        self.assertEqual(report["conditional_prior_mean"], self.mean.tolist())
        self.assertEqual(report["conditional_prior_sd"], self.sd.tolist())
        self.assertEqual(report["points"]["A"]["calls"]["scalar"]["trial_log_scores"],
                         reference["points"][0]["calls"]["scalar"]["trial_log_scores"])
        for name, index in (("A", 270), ("B", 485)):
            saved = report["points"][name]["calls"]["batched"]
            self.assertFalse(saved["new_call"])
            self.assertEqual(saved["source_evaluation_index"], index)
            self.assertEqual(saved["trial_log_scores"], trace["evaluations"][index]["trial_log_scores"])
        for name, point in report["points"].items():
            theta = np.asarray(point["parameters"])
            for solver, call in point["calls"].items():
                values = np.asarray(call["trial_log_scores"])
                self.assertEqual(call["conditional_objective"], float(-values.sum()+.5*np.sum(((theta-self.mean)/self.sd)**2)))
                self.assertEqual(call["new_call"], (name, solver) in diagnostic.CALLS)
        self.assertEqual(reads.count(diagnostic.TRACE), 1)
        self.assertEqual(reads.count(diagnostic.A_REFERENCE), 1)
        self.assertGreater(report["summary"]["scalar_batched_comparisons"]["B"]["maximum_absolute_trial_log_score_difference"], 0.)

    def test_source_replay_and_new_anchor_mismatch_reject_before_new_likelihoods(self):
        options = ({"bad_source": True}, {"replay_error": True},
            {"anchor_change": lambda t, r: r.update(job_id="wrong")},
            {"anchor_change": lambda t, r: r["points"][0]["calls"]["scalar"]["trial_log_scores"].__setitem__(0, -999.)},
            {"anchor_change": lambda t, r: t["conditional_prior_sd"].__setitem__(0, 99.)},
            {"anchor_change": lambda t, r: r["points"][0]["geometry"].update(n_steps=1)})
        for option in options:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                report, calls, *_ = self.exercise(Path(directory)/"new", **option)
            self.assertFalse(report["diagnostic_complete"])
            self.assertFalse(report["saved_anchors_validated"])
            self.assertEqual(calls, [])
            self.assertEqual(report["attempted_new_likelihood_calls"], 0)
            self.assertIsNone(report["summary"])

    def test_failure_and_deadline_preserve_only_actual_call_prefix_without_extra_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, *_ = self.exercise(Path(directory)/"new", fail_call=3)
        self.assertFalse(report["diagnostic_complete"])
        self.assertEqual((len(calls), report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]), (3, 3, 2))
        self.assertFalse(report["points"]["C"]["calls"]["batched"]["returned"])
        self.assertIn("interrupted", report["points"]["C"]["calls"]["batched"]["error"])
        self.assertEqual(report["points"]["D"]["calls"], {})
        self.assertIsNone(report["summary"])
        with tempfile.TemporaryDirectory() as directory:
            report, calls, *_ = self.exercise(Path(directory)/"new", deadline_stage="staged_call")
        self.assertFalse(report["diagnostic_complete"])
        self.assertEqual(calls, [])
        self.assertEqual((report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]), (1, 0))
        self.assertFalse(report["points"]["B"]["calls"]["scalar"]["returned"])
        self.assertFalse(report["points"]["A"]["calls"]["scalar"]["new_call"])
        self.assertIsNone(report["summary"])
        for stage, expected in (("source", 0), ("anchors", 0), ("likelihood", 2), ("finalization", 5)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, calls, *_ = self.exercise(Path(directory)/"new", deadline_stage=stage)
            self.assertFalse(report["diagnostic_complete"])
            self.assertEqual((len(calls), report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]),
                             (expected, expected, expected))
            self.assertIn("whole-process deadline", report["error"])
            self.assertIsNone(report["summary"])


if __name__ == "__main__":
    unittest.main()
