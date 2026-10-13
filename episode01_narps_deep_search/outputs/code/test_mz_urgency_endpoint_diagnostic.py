"""Focused endpoint diagnostics; DDM calls are mocked, execution belongs in Slurm."""
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
from scipy.optimize import minimize as installed_minimize

import collapsing_ddm
import run_mz_urgency_endpoint_diagnostic as diagnostic
import test_mz_urgency_cost_precision_pilot as fixtures


class UrgencyEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.UrgencyPilotTests.setUpClass()
        cls.fixture = fixtures.UrgencyPilotTests.fixture
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        cls.names, cls.starts = diagnostic.optimization.scalar.initialization_vectors("M0", cls.spec)
        cls.selected = {version: min(person for person, rows in cls.people.items()
                                    if rows[0]["task_version"] == version) for version in ("EI", "ER")}

    def probe(self, source):
        identity, _ = diagnostic.optimization.probe_identity(2, self.selected)
        rows = [row for row in self.people[self.selected["ER"]] if int(row["run"]) in (1, 4)]
        priors = [self.spec["parameters"][name] for name in self.names]
        initial = np.array([prior["individual_scale_center"] for prior in priors])
        report = {**identity, "analysis_id": "behavioral_mechanism_kill_test",
            "phase": diagnostic.optimization.PHASE, "status": "urgency_conditional_cost_probe_completed",
            "generating_model": "MZ", "replicate_index": 0, "seed": 910031, "rival": "urgency",
            "implementation": self.spec, "job_id": diagnostic.PROBE_JOB,
            "source_dir": str(diagnostic.SOURCE),
            "source_report": str(diagnostic.SOURCE / diagnostic.optimization.scalar.SOURCE_REPORT),
            "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
            "source_participants": 108, "source_runs": 432, "source_trials": 27648,
            "benchmark_report": str(diagnostic.optimization.BENCHMARK),
            "scalar_reference_report": str(diagnostic.optimization.benchmark.REFERENCE),
            "benchmark_finished_utc": "2026-10-05T11:31:00+00:00",
            "started_utc": "2026-10-05T11:40:00+00:00", "finished_utc": diagnostic.PROBE_FINISHED,
            "internal_seconds": 3000, "elapsed_seconds": 2100., "parameter_names": self.names,
            "conditional_prior_mean": [prior["population_center"] for prior in priors],
            "original_initial_sd": initial.tolist(), "conditional_prior_sd": np.exp(np.log(initial)).tolist(),
            "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
            "selected_probe_participants": self.selected, "expected_starts": 3, "completed_starts": 3,
            "likelihood_calls_attempted": 1290, "likelihood_calls_returned": 1290,
            "batched_solver_adoption_scope": "this_conditional_cost_pilot_only",
            "saved_score_replay_absolute_tolerance": 1e-12,
            "saved_score_replay_max_absolute_discrepancy": 0., "full_recovery_pass": None,
            **{key: True for key in ("probe_complete", "conditional_optimization_complete",
                "source_baseline_validated", "saved_primary_scores_replayed", "benchmark_validated", "optimizer_called")},
            **{key: False for key in ("simulation_regenerated", "simulation_copied", "observed_choices_or_RTs_used",
                "population_hyperparameters_updated", "full_cohort_model_fitted", "heldout_urgency_scores_computed",
                "actual_fitted_support_precision_evaluated", "full_fitter_solver_adopted", "wave_selected",
                "automatic_retry", "episode_terminal")},
            **{key: "not_evaluated" for key in diagnostic.optimization.scalar.CLAIM_FIELDS}, "attempts": []}
        for index, start in enumerate(self.starts):
            report["attempts"].append({"start_index": index, "start_parameters": start.tolist(),
                "optimizer_entered": True, "returned": True, "success": False, "status": 2,
                "message": "ABNORMAL_TERMINATION_IN_LNSRCH", "objective_finite": True,
                "parameters_finite": True, "parameters": list(diagnostic.ENDPOINT),
                "objective": diagnostic.REPORTED_OBJECTIVE + (2-index),
                "nfev": (350, 450, 490)[index], "numerical_error_calls": 0,
                "likelihood_calls_attempted": (350, 450, 490)[index],
                "likelihood_calls_returned": (350, 450, 490)[index]})
        report["best_start"] = {key: deepcopy(report["attempts"][2][key]) for key in
            ("start_index", "parameters", "parameters_finite", "objective", "objective_finite", "success", "message")}
        return report, rows

    def exercise(self, output, *, bad_source=False, replay_error=False, probe_change=None,
                 fail_call=None, nonfinite=False, deadline_stage=None):
        source, truth = self.fixture.source()
        probe, rows = self.probe(source)
        if bad_source:
            source["replicate_index"] = 1
        if probe_change is not None:
            probe_change(probe)
        reference, benchmark = {}, {"finished_utc": probe["benchmark_finished_utc"]}
        before = deepcopy((source, truth, probe, self.spec, self.people))
        read_text, writer, now = Path.read_text, diagnostic.write_report, [100.]
        fixed_calls, calls, checkpoints = [], [], []
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        shared = diagnostic.optimization.scalar.shared

        def reader(path, *args, **kwargs):
            objects = {diagnostic.IMPLEMENTATION: self.spec, diagnostic.PROBE: probe,
                diagnostic.SOURCE / diagnostic.optimization.scalar.SOURCE_REPORT: source,
                diagnostic.SOURCE / "generating_parameters.json": truth,
                diagnostic.optimization.benchmark.REFERENCE: reference,
                diagnostic.optimization.BENCHMARK: benchmark}
            if path in objects:
                if ((deadline_stage == "source" and objects[path] is source)
                        or (deadline_stage == "probe" and objects[path] is probe)):
                    now[0] += 301
                return json.dumps(objects[path])
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, diagnostic.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": original[0]["task_version"],
                    "rows": [row for row in original if int(row["run"]) in runs]}
                    for person, original in self.people.items()}
            return panel(train), panel(held)

        def likelihood(theta, names, data, spec, rival):
            if rival == "fixed":
                self.assertEqual(calls, [])
                fixed_calls.append(list(names))
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            self.assertEqual(rival, "urgency")
            return candidate("scalar", theta, names, data, spec)

        def candidate(solver, theta, names, data, spec):
            self.assertEqual(len(fixed_calls), 432)
            self.assertEqual(names, self.names)
            self.assertEqual(spec, self.spec)
            expected = shared.make_data(rows)
            for key in expected:
                np.testing.assert_array_equal(data[key], expected[key])
            calls.append((solver, theta.copy()))
            if len(calls) == fail_call:
                if nonfinite:
                    return np.full(128, np.nan)
                raise RuntimeError("Selected endpoint call interrupted")
            if deadline_stage == "likelihood" and len(calls) == 1:
                now[0] += 301
            # Depend on the selected coordinate; retain a deliberate scalar/batched
            # difference so comparison signs and summed objectives are exercised.
            return np.full(128, -2. + .25*(theta[2]-diagnostic.ENDPOINT[2])
                           + (1e-10 if solver == "batched" else 0.))

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["diagnostic_complete"]:
                now[0] += 301

        args = argparse.Namespace(source_dir=diagnostic.SOURCE, implementation=diagnostic.IMPLEMENTATION,
                                  output_dir=output, max_seconds=300)
        with ExitStack() as stack:
            stack.enter_context(patch.object(diagnostic, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {key: "1" for key in
                ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(shared, "load_split", side_effect=split))
            stack.enter_context(patch.object(shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(diagnostic.optimization.batched, "trial_likelihood",
                side_effect=lambda *args: candidate("batched", *args)))
            # The existing 48-call validator is already qualified. Retain actual
            # source validation/replay and the new exact probe validator here.
            stack.enter_context(patch.object(diagnostic.optimization, "validate_benchmark"))
            stack.enter_context(patch.object(shared, "fit_partially_pooled", side_effect=AssertionError("No model fit")))
            stack.enter_context(patch.object(diagnostic.optimization, "optimize_subject_block",
                side_effect=AssertionError("No scientific optimization")))
            stack.enter_context(patch.object(diagnostic, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(diagnostic.time, "monotonic", side_effect=lambda: now[0]))
            report = diagnostic.run(args)
        self.assertEqual((source, truth, probe, self.spec, self.people), before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / diagnostic.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assert_scope(record)
        return report, fixed_calls, calls, checkpoints

    def assert_scope(self, report):
        self.assertEqual(report["phase"], diagnostic.PHASE)
        self.assertEqual(report["expected_likelihood_calls"], 6)
        self.assertEqual(report["probe_report"], str(diagnostic.PROBE))
        for field in diagnostic.optimization.scalar.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("scientific_optimizer_called", "simulation_regenerated", "observed_choices_or_RTs_used",
                "population_hyperparameters_updated", "heldout_urgency_scores_computed", "refined_mesh_called",
                "original_48_call_benchmark_repeated", "optimizer_settings_changed", "full_fitter_solver_adopted",
                "automatic_retry"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        for field in ("participant_scores", "nested_diagnostic", "numerical_precision_passed", "solver_adopted"):
            self.assertNotIn(field, report)

    def test_installed_stencil_uses_exact_options_axis_and_observed_bound_aware_step(self):
        observed, supplied = [], []
        def recorder(objective, start, **kwargs):
            supplied.append(kwargs)
            def recording(candidate):
                observed.append(candidate.copy())
                return objective(candidate)
            return installed_minimize(recording, start, **kwargs)
        before = deepcopy(self.spec)
        with patch.object(diagnostic, "minimize", side_effect=recorder):
            stencil = diagnostic.record_stencil(diagnostic.ENDPOINT, self.names, self.spec)
        self.assertEqual(stencil["recording_vectors"], [row.tolist() for row in observed])
        self.assertEqual(supplied, [{"method": "L-BFGS-B",
            "bounds": [self.spec["parameters"][name]["bounds"] for name in self.names],
            "options": {"maxiter": self.spec["optimizer"]["subject_maximum_iterations"],
                "ftol": self.spec["optimizer"]["subject_ftol"], "gtol": self.spec["optimizer"]["subject_gtol"]}}])
        axis = self.names.index("nondecision_logit")
        theta = np.array(diagnostic.ENDPOINT)
        points = stencil["points"]
        self.assertEqual([row["name"] for row in points], ["endpoint", "optimizer_perturbation", "opposite_diagnostic"])
        np.testing.assert_array_equal(points[0]["parameters"], theta)
        for point in points[1:]:
            self.assertEqual(np.flatnonzero(np.asarray(point["parameters"]) != theta).tolist(), [axis])
        self.assertIn(points[1]["parameters"], stencil["recording_vectors"])
        self.assertNotEqual(stencil["effective_step"], 0.)
        self.assertEqual(points[1]["coordinate_delta"], stencil["effective_step"])
        self.assertEqual(points[2]["coordinate_delta"], -stencil["effective_step"])
        for bound_index in (0, 1):
            observed.clear()
            bounded = theta.copy()
            bounded[axis] = self.spec["parameters"]["nondecision_logit"]["bounds"][bound_index]
            with patch.object(diagnostic, "minimize", side_effect=recorder):
                with self.assertRaisesRegex(ValueError, "opposite-side"):
                    diagnostic.record_stencil(bounded, self.names, self.spec)
            steps = [row[axis]-bounded[axis] for row in observed
                     if np.flatnonzero(row != bounded).tolist() == [axis]]
            self.assertEqual(len(steps), 1)
            self.assertGreater(steps[0] if bound_index == 0 else -steps[0], 0.)
        self.assertEqual(self.spec, before)

    def test_grid_geometry_matches_scalar_density_bins_ceil_and_spatial_interpolation(self):
        names = self.names + ["rho"]
        theta = np.array(diagnostic.ENDPOINT + [math.log(.503/.497)])
        theta[2] = math.log(.25/.75)
        t0 = 4*diagnostic.expit(theta[2])
        horizon = 4-t0
        n = math.ceil(horizon/.002)
        edges = np.linspace(0., horizon, n+1)
        data = {"gain": np.full(7, 10.), "loss": np.full(7, 12.),
            "rt": np.array([t0-.1, t0, t0+edges[1], 4., 4.1, t0+.5*edges[1], np.nan]),
            "choice": np.array([0, 1, 0, 1, 1, 0, -1])}
        geometry = diagnostic.grid_metadata(theta, names, data, self.spec)
        self.assertEqual(geometry["n_steps"], n)
        self.assertEqual(geometry["actual_dt_seconds"], horizon/n)
        self.assertEqual(geometry["decision_horizon_seconds"], horizon)
        self.assertEqual(geometry["actual_unique_drift_count"], 1)
        dist = collapsing_ddm.FirstPassageDistribution(edges, np.arange(1, n+1, dtype=float),
                                                      np.arange(1, n+1, dtype=float)*2, .1)
        densities = dist.response_density(data["rt"][:-1]-t0, data["choice"][:-1])
        for row, density, choice in zip(geometry["rows"][:-1], densities, data["choice"][:-1]):
            expected = 0. if not row["active"] else ((row["safe_lookup_bin_index"]+1)
                * (2 if choice else 1) / np.diff(edges)[row["safe_lookup_bin_index"]])
            self.assertEqual(density, expected)
        self.assertFalse(geometry["rows"][0]["active"])
        self.assertFalse(geometry["rows"][1]["active"])
        self.assertTrue(geometry["rows"][3]["active"])
        self.assertEqual(geometry["rows"][3]["density_bin_index"], n-1)
        self.assertFalse(geometry["rows"][4]["active"])
        self.assertIsNone(geometry["rows"][-1]["active"])
        self.assertIsNone(geometry["rows"][-1]["decision_time"])
        self.assertTrue(geometry["spatial_upper_mass_added"])
        self.assertEqual(geometry["spatial_lower_weight"]+geometry["spatial_upper_weight"], 1.)
        self.assertAlmostEqual((geometry["spatial_low"]*geometry["spatial_lower_weight"]
            +(geometry["spatial_low"]+1)*geometry["spatial_upper_weight"])*geometry["spatial_h"], geometry["z"], places=15)
        lower, upper = theta.copy(), theta.copy()
        lower[2] -= 1e-8
        upper[2] += 1e-8
        self.assertEqual(diagnostic.grid_metadata(lower, names, data, self.spec)["n_steps"], 1501)
        self.assertEqual(diagnostic.grid_metadata(upper, names, data, self.spec)["n_steps"], 1500)

    def test_objective_and_signed_slopes_preserve_optimizer_fun_mismatch(self):
        theta, mean, sd = np.array([1., -2.]), np.array([.5, -1.]), np.array([.2, .3])
        values = np.array([-1., -2., -3.])
        self.assertEqual(diagnostic.conditional_objective(values, theta, mean, sd),
                         float(-values.sum()+.5*np.sum(((theta-mean)/sd)**2)))
        points = []
        for name, delta, objective in (("endpoint", 0., 10.), ("optimizer_perturbation", -.25, 9.),
                                       ("opposite_diagnostic", .25, 12.)):
            points.append({"name": name, "coordinate_delta": delta, "calls": {
                solver: {"conditional_objective": objective + offset, "trial_log_scores": [-2.+offset+delta]*128}
                for solver, offset in (("scalar", 0.), ("batched", .5))}})
        summary = diagnostic.summarize_points(points, 7.)
        expected = {"recomputed_endpoint_objective": 10.,
            "endpoint_objective_minus_optimizer_report": 3., "forward_slope": 8., "backward_slope": 4.}
        self.assertEqual({key: summary["solvers"]["scalar"][key] for key in expected}, expected)
        self.assertEqual(summary["solvers"]["batched"]["endpoint_objective_minus_optimizer_report"], 3.5)
        self.assertTrue(all(row["mean_trial_log_score_difference"] == .5 for row in summary["scalar_batched_comparisons"]))
        for solver in ("scalar", "batched"):
            self.assertEqual(summary["solvers"][solver]["local_comparisons"], [
                {"point": name, "coordinate_delta": delta,
                 "point_minus_endpoint_trial_log_scores": [delta]*128}
                for name, delta in (("optimizer_perturbation", -.25), ("opposite_diagnostic", .25))])

    def test_full_source_replay_precedes_exact_six_local_calls_without_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fixed, calls, checkpoints = self.exercise(Path(directory) / "new")
        self.assertTrue(report["diagnostic_complete"])
        self.assertTrue(report["terminal_probe_validated"])
        self.assertEqual(report["historical_probe_elapsed_seconds"], 2100.)
        self.assertEqual((len(fixed), len(calls), report["completed_likelihood_calls"]), (432, 6, 6))
        self.assertEqual([solver for solver, _ in calls], ["scalar", "batched"]*3)
        for index, (_, theta) in enumerate(calls):
            np.testing.assert_array_equal(theta, report["stencil"]["points"][index//2]["parameters"])
        self.assertNotEqual(report["summary"]["solvers"]["scalar"]["endpoint_objective_minus_optimizer_report"], 0.)
        self.assertTrue(all(row["mean_trial_log_score_difference"] > 0.
                            for row in report["summary"]["scalar_batched_comparisons"]))
        self.assertTrue(all(len(point["geometry"]["rows"]) == 128 for point in report["points"]))

    def test_wrong_source_replay_or_terminal_probe_context_prevents_all_endpoint_calls(self):
        changes = (lambda r: r.update(job_id="another"), lambda r: r.update(completed_starts=2),
            lambda r: r.update(source_finished_utc="another"), lambda r: r["attempts"][0].update(success=True),
            lambda r: r["attempts"][1]["start_parameters"].__setitem__(0, 9.),
            lambda r: r["attempts"][2].update(objective=diagnostic.REPORTED_OBJECTIVE+.1),
            lambda r: r["best_start"].update(start_index=0), lambda r: r["trial_slots"].pop(),
            lambda r: r["attempts"][0].update(numerical_error_calls=1),
            lambda r: r.update(full_fitter_solver_adopted=True))
        options = [{"bad_source": True}, {"replay_error": True}] + [{"probe_change": change} for change in changes]
        for case in options:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                report, fixed, calls, _ = self.exercise(Path(directory) / "new", **case)
            self.assertFalse(report["diagnostic_complete"])
            self.assertFalse(report["terminal_probe_validated"])
            self.assertEqual(calls, [])
            self.assertFalse(report["constant_objective_recording_called"])
            self.assertIsNone(report["summary"])

    def test_interrupted_or_nonfinite_call_preserves_completed_prefix_and_actual_partial_identity(self):
        for nonfinite in (False, True):
            with self.subTest(nonfinite=nonfinite), tempfile.TemporaryDirectory() as directory:
                report, _, calls, _ = self.exercise(Path(directory) / "new", fail_call=2, nonfinite=nonfinite)
            self.assertFalse(report["diagnostic_complete"])
            self.assertEqual((len(calls), report["attempted_likelihood_calls"], report["completed_likelihood_calls"]), (2, 2, 1))
            self.assertEqual(len(report["points"]), 1)
            self.assertEqual(report["points"][0]["name"], "endpoint")
            scalar, batched = (report["points"][0]["calls"][key] for key in ("scalar", "batched"))
            self.assertTrue(scalar["finite128_vector"])
            self.assertEqual(len(scalar["trial_log_scores"]), 128)
            self.assertEqual(batched["returned"], nonfinite)
            self.assertFalse(batched["finite128_vector"])
            self.assertIn("error", batched)
            self.assertIsNone(report["summary"])

    def test_deadline_covers_reads_calls_and_final_publication_without_discarding_finished_vectors(self):
        for stage, count in (("source", 0), ("probe", 0), ("likelihood", 1), ("finalization", 6)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, _, calls, _ = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertFalse(report["diagnostic_complete"])
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(report["completed_likelihood_calls"], count)
            self.assertEqual(len(calls), count)
            self.assertIsNone(report["summary"])
            self.assertTrue(report["finished_utc"])


if __name__ == "__main__":
    unittest.main()
