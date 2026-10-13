"""Two refined-call checks using preserved coarse fixtures; Slurm only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_mz_urgency_refined_crossed_diagnostic as diagnostic
import test_mz_urgency_crossed_nondecision_diagnostic as fixtures


class UrgencyRefinedCrossedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.UrgencyCrossedNondecisionTests.setUpClass()
        cls.fixture = fixtures.UrgencyCrossedNondecisionTests()
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        cls.names, cls.mean, cls.sd = cls.fixture.names, cls.fixture.mean, cls.fixture.sd
        cls.rows, cls.data = cls.fixture.rows, cls.fixture.data
        with tempfile.TemporaryDirectory() as directory:
            cls.coarse, *_ = cls.fixture.exercise(Path(directory)/"coarse")
        cls.coarse["job_id"] = "46618614"

    def exercise(self, output, *, fail_call=None, deadline_stage=None, coarse_change=None):
        source, truth = self.fixture.fixture.fixture.source()
        coarse = deepcopy(self.coarse)
        if coarse_change:
            coarse_change(coarse)
        before = deepcopy((source, truth, coarse, self.spec, self.people))
        shared = diagnostic.optimization.scalar.shared
        writer, now = diagnostic.write_report, [100.]
        calls, fixed_calls, checkpoints, reads = [], [], [], []
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        objects = {diagnostic.IMPLEMENTATION: self.spec, diagnostic.COARSE_REPORT: coarse,
            diagnostic.SOURCE / diagnostic.optimization.scalar.SOURCE_REPORT: source,
            diagnostic.SOURCE / "generating_parameters.json": truth}
        refined = deepcopy(self.spec)
        refined["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)

        def reader(path, *args, **kwargs):
            if path not in objects:
                raise AssertionError(f"No old trace/probe/reference or unrelated read: {path}")
            reads.append(path)
            if deadline_stage == "coarse_read" and path == diagnostic.COARSE_REPORT:
                now[0] += diagnostic.MAX_SECONDS+1
            return json.dumps(objects[path])

        def split(path, train, held):
            self.assertEqual(path, diagnostic.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": original[0]["task_version"],
                    "rows": [row for row in original if int(row["run"]) in runs]}
                    for person, original in self.people.items()}
            return panel(train), panel(held)

        def fixed(theta, names, data, spec, rival):
            self.assertEqual(rival, "fixed", "No new scalar urgency call")
            self.assertEqual(spec, self.spec)
            self.assertEqual(calls, [])
            fixed_calls.append(list(names))
            return np.full(128, -1.98 if "rho" in names else -2.)

        def candidate(theta, names, data, spec):
            self.assertEqual(len(fixed_calls), 432)
            self.assertEqual((names, spec), (self.names, refined))
            name = diagnostic.POINTS[len(calls)]
            np.testing.assert_array_equal(theta, coarse["points"][name]["parameters"])
            for key in self.data:
                np.testing.assert_array_equal(data[key], self.data[key])
            calls.append({"point": name, "parameters": theta.tolist(), "implementation": deepcopy(spec)})
            if len(calls) == fail_call:
                raise RuntimeError("Selected refined call interrupted")
            if deadline_stage == "likelihood" and len(calls) == 1:
                now[0] += diagnostic.MAX_SECONDS+1
            return np.asarray(coarse["points"][name]["calls"]["batched"]["trial_log_scores"])+(.125 if name == "A" else -.25)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if (deadline_stage == "staged_call" and report["attempted_new_likelihood_calls"] == 1
                    and not report["points"]["A"]["refined"]["call"]["returned"]):
                now[0] += diagnostic.MAX_SECONDS+1
            if deadline_stage == "finalization" and report["diagnostic_complete"]:
                now[0] += diagnostic.MAX_SECONDS+1

        args = argparse.Namespace(source_dir=diagnostic.SOURCE, implementation=diagnostic.IMPLEMENTATION,
                                  output_dir=output, max_seconds=diagnostic.MAX_SECONDS)
        with ExitStack() as stack:
            stack.enter_context(patch.object(diagnostic, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {key: "1" for key in
                (*diagnostic.crossed.trace_analysis.THREAD_ENV, "SLURM_CPUS_PER_TASK")}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(shared, "load_split", side_effect=split))
            stack.enter_context(patch.object(shared, "trial_likelihood", side_effect=fixed))
            stack.enter_context(patch.object(diagnostic.optimization.batched, "trial_likelihood", side_effect=candidate))
            stack.enter_context(patch.object(shared, "fit_partially_pooled", side_effect=AssertionError("No fit")))
            stack.enter_context(patch.object(diagnostic.optimization, "optimize_subject_block", side_effect=AssertionError("No optimizer")))
            stack.enter_context(patch.object(diagnostic, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(diagnostic.time, "monotonic", side_effect=lambda: now[0]))
            report = diagnostic.run(args)
        self.assertEqual((source, truth, coarse, self.spec, self.people), before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / diagnostic.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assertEqual(record["expected_new_likelihood_calls"], 2)
            self.assertEqual((record["new_coarse_urgency_calls"], record["new_scalar_urgency_calls"]), (0, 0))
            for field in ("optimizer_called", "optimizer_settings_changed", "simulation_regenerated",
                    "observed_choices_or_RTs_used", "population_hyperparameters_updated", "heldout_urgency_scores_computed",
                    "full_fitter_solver_adopted", "fitted_support_precision_qualified", "mesh_adopted",
                    "acceptance_threshold_selected", "automatic_retry"):
                self.assertIs(record[field], False)
            for field in diagnostic.optimization.scalar.CLAIM_FIELDS:
                self.assertEqual(record[field], "not_evaluated")
            self.assertIsNone(record["full_recovery_pass"])
        return report, calls, fixed_calls, coarse, reads

    def test_only_two_refined_A_C_calls_preserve_coarse_anchors_rows_prior_and_original_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, fixed, coarse, reads = self.exercise(Path(directory)/"new")
        self.assertTrue(report["diagnostic_complete"])
        self.assertEqual(len(fixed), 432)
        self.assertEqual([row["point"] for row in calls], ["A", "C"])
        self.assertEqual(report["new_call_order"], ["A", "C"])
        self.assertEqual((report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]), (2, 2))
        self.assertEqual(report["implementation"], self.spec)
        original = deepcopy(report["refined_numerical_implementation"])
        original["urgency_solver"].update(dt_seconds=.002, spatial_intervals=200)
        self.assertEqual(original, self.spec)
        self.assertEqual(report["conditional_prior_mean"], self.mean.tolist())
        self.assertEqual(report["conditional_prior_sd"], self.sd.tolist())
        for name, point in report["points"].items():
            self.assertEqual(point["coarse"], {"geometry": coarse["points"][name]["geometry"],
                "call": coarse["points"][name]["calls"]["batched"]})
            self.assertEqual(point["parameters"], coarse["points"][name]["parameters"])
            a, b = point["coarse"]["geometry"], point["refined"]["geometry"]
            self.assertEqual((a["nominal_dt_seconds"], a["spatial_intervals"]), (.002, 200))
            self.assertEqual((b["nominal_dt_seconds"], b["spatial_intervals"]), (.001, 400))
            self.assertEqual(b["decision_horizon_seconds"], a["decision_horizon_seconds"])
            self.assertEqual(b["n_steps"], math.ceil(b["decision_horizon_seconds"]/.001))
            self.assertEqual(b["actual_dt_seconds"], b["decision_horizon_seconds"]/b["n_steps"])
            call = point["refined"]["call"]
            values, theta = np.asarray(call["trial_log_scores"]), np.asarray(point["parameters"])
            self.assertEqual(call["conditional_objective"], float(-values.sum()+.5*np.sum(((theta-self.mean)/self.sd)**2)))
        self.assertEqual(set(reads), {diagnostic.IMPLEMENTATION, diagnostic.COARSE_REPORT,
            diagnostic.SOURCE / diagnostic.optimization.scalar.SOURCE_REPORT,
            diagnostic.SOURCE / "generating_parameters.json"})

    def test_signed_resolution_and_A_to_C_arithmetic_keep_prior_separate_without_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            report, *_ = self.exercise(Path(directory)/"new")
        points = deepcopy(report["points"])
        values = {("A", "coarse"): -2., ("A", "refined"): -1.75,
                  ("C", "coarse"): -1.5, ("C", "refined"): -1.625}
        for (name, mesh), value in values.items():
            theta = np.asarray(points[name]["parameters"])
            prior = float(.5*np.sum(((theta-self.mean)/self.sd)**2))
            points[name][mesh]["call"].update(trial_log_scores=[value]*128,
                nll=-128*value, prior_penalty=prior, conditional_objective=float(-128*value+prior))
        summary = diagnostic.summarize_refinement(points, self.mean, self.sd, report["trial_slots"])
        for name, delta, nll in (("A", .25, -32.), ("C", -.125, 16.)):
            row = summary["refined_minus_coarse"][name]
            self.assertEqual([item["right_minus_left_trial_log_score"] for item in row["rows"]], [delta]*128)
            self.assertEqual(row["delta_nll"], nll)
            self.assertEqual(row["delta_prior_penalty"], 0.)
            self.assertEqual(row["negative_sum_trial_log_score_differences"], nll)
            self.assertEqual(sum(group["count"] for group in row["groups"].values()), 128)
        for mesh, delta in (("coarse", .5), ("refined", .125)):
            row = summary["A_to_C"][mesh]
            self.assertEqual((row["left_index"], row["right_index"]), ("A_"+mesh, "C_"+mesh))
            self.assertEqual([item["right_minus_left_trial_log_score"] for item in row["rows"]], [delta]*128)
            self.assertEqual(row["delta_conditional_objective"],
                points["C"][mesh]["call"]["conditional_objective"]-points["A"][mesh]["call"]["conditional_objective"])
        self.assertIn("do not qualify fitted-support precision", summary["interpretation"])
        self.assertNotIn("precision_passed", summary)

    def test_partial_failure_and_deadlines_retain_coarse_anchors_and_actual_refined_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, *_ = self.exercise(Path(directory)/"new", fail_call=2)
        self.assertFalse(report["diagnostic_complete"])
        self.assertEqual((len(calls), report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]), (2, 2, 1))
        self.assertTrue(report["points"]["A"]["refined"]["call"]["returned"])
        self.assertFalse(report["points"]["C"]["refined"]["call"]["returned"])
        self.assertIsNone(report["summary"])
        for stage, expected, attempted in (("coarse_read", 0, 0), ("staged_call", 0, 1),
                                           ("likelihood", 1, 1), ("finalization", 2, 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, calls, _, coarse, _ = self.exercise(Path(directory)/"new", deadline_stage=stage)
            self.assertFalse(report["diagnostic_complete"])
            self.assertEqual((len(calls), report["attempted_new_likelihood_calls"], report["completed_new_likelihood_calls"]),
                             (expected, attempted, expected))
            self.assertIn("whole-process deadline", report["error"])
            self.assertIsNone(report["summary"])
            for name, point in report["points"].items():
                self.assertEqual(point["coarse"]["call"], coarse["points"][name]["calls"]["batched"])
        with tempfile.TemporaryDirectory() as directory:
            report, calls, *_ = self.exercise(Path(directory)/"new", coarse_change=lambda r:
                r["points"]["C"]["geometry"].update(n_steps=1))
        self.assertFalse(report["coarse_anchors_validated"])
        self.assertFalse(report["diagnostic_complete"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
