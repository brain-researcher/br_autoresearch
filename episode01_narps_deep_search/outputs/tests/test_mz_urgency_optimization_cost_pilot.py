"""Focused cost-pilot fixtures; never read source data or solve a DDM."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import urgency_conditional_optimization as core
import run_mz_urgency_optimization_cost_pilot as pilot


def specification():
    return {"optimizer": {"maximum_sweeps": 500,
            "initial_standardized_offsets": [[-.5]*8, [.5]*8],
            "subject_maximum_iterations": 17, "subject_ftol": 1e-9, "subject_gtol": 1e-6},
        "parameters": {name: {"population_center": center, "individual_scale_center": scale,
                              "bounds": bounds}
            for name, center, scale, bounds in (
                ("kappa", -1., .13, [-4., 3.]), ("log_a", 0., .7, [-3., 2.]),
                ("nondecision_logit", -2., .31, [-6., 0.]),
                ("rho", 0., .2, [-3., 3.]), ("urgency", .1, .4, [0., 2.]))},
        "urgency_solver": {"dt_seconds": .002, "spatial_intervals": 200,
            "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}}


def result(x, fun, success=True):
    return SimpleNamespace(x=np.asarray(x), fun=fun, success=success, message="fixture",
                           nfev=1, njev=1, nit=1, status=0 if success else 1)


class ConditionalCostTests(unittest.TestCase):
    def setUp(self):
        self.spec = specification()
        self.names, self.starts = pilot.scalar.initialization_vectors("MZ", self.spec)
        self.mean = np.array([self.spec["parameters"][name]["population_center"] for name in self.names])
        self.initial_sd = np.array([self.spec["parameters"][name]["individual_scale_center"] for name in self.names])
        self.sd = np.exp(np.log(self.initial_sd))

    def block(self, minimize, likelihood=lambda *args: np.array([-2., -3.]), **kwargs):
        attempts = kwargs.pop("attempts", [])
        with patch.object(core, "minimize", side_effect=minimize):
            best = core.optimize_subject_block({}, self.names, self.spec, self.mean, self.sd,
                self.starts, likelihood=likelihood, attempts=attempts, **kwargs)
        return best, attempts

    def test_exact_objective_starts_options_and_literal_sd(self):
        observed = []
        def minimize(objective, start, **kwargs):
            index = len(observed)
            np.testing.assert_array_equal(start, self.starts[index])
            expected = float(5. + .5*np.sum(((start-self.mean)/np.exp(np.log(self.initial_sd)))**2))
            self.assertEqual(objective(start), expected)
            self.assertEqual(kwargs, {"method": "L-BFGS-B",
                "bounds": [self.spec["parameters"][name]["bounds"] for name in self.names],
                "options": {"maxiter": 17, "ftol": 1e-9, "gtol": 1e-6}})
            observed.append(index)
            return result(start, expected)
        best, attempts = self.block(minimize)
        self.assertEqual(best["start_index"], 0)
        self.assertEqual(len(observed), 3)
        self.assertEqual([row["objective_calls"] for row in attempts], [1, 1, 1])
        self.assertEqual([row["likelihood_calls_returned"] for row in attempts], [1, 1, 1])
        self.assertTrue(all(row["likelihood_seconds"] >= 0 for row in attempts))

    def test_original_first_minimum_does_not_prefer_success(self):
        returns = iter([result(self.starts[0], 3.), result(self.starts[1], 1., False), result(self.starts[2], 1., True)])
        best, attempts = self.block(lambda *args, **kwargs: next(returns))
        self.assertEqual(best["start_index"], 1)
        self.assertFalse(best["success"])
        self.assertEqual([row["returned"] for row in attempts], [True]*3)

    def test_nonfinite_returns_are_explicit_json_and_finite_filter(self):
        returns = iter([result(np.full(5, np.nan), np.nan), result(self.starts[1], np.inf), result(self.starts[2], 4.)])
        best, attempts = self.block(lambda *args, **kwargs: next(returns))
        self.assertEqual(best["start_index"], 2)
        self.assertIsNone(attempts[0]["objective"])
        self.assertFalse(attempts[0]["parameters_finite"])
        self.assertEqual(attempts[0]["parameters"], [None]*5)
        self.assertEqual(attempts[1]["nonfinite_reason"], "nonfinite_optimizer_return")
        json.dumps(attempts, allow_nan=False)
        attempts = []
        with self.assertRaisesRegex(RuntimeError, "No finite"):
            self.block(lambda *args, **kwargs: result(self.mean, np.inf), attempts=attempts)
        self.assertEqual(sum(row["returned"] for row in attempts), 3)

    def test_numerical_errors_map_to_infinity_but_timeout_propagates(self):
        calls = []
        def numerical(*args):
            calls.append(1)
            raise ValueError("numerical support")
        def minimize(objective, start, **kwargs):
            self.assertTrue(np.isinf(objective(start)))
            return result(start, 7.)
        _, attempts = self.block(minimize, numerical)
        self.assertEqual([row["numerical_error_calls"] for row in attempts], [1]*3)
        attempts = []
        def interrupted(*args):
            raise RuntimeError("deadline")
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.block(minimize, interrupted, attempts=attempts)
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["returned"])
        self.assertEqual(attempts[0]["likelihood_calls_attempted"], 1)
        self.assertEqual(attempts[0]["likelihood_calls_returned"], 0)
        self.assertEqual(attempts[0]["numerical_error_calls"], 0)
        self.assertIn("deadline", attempts[0]["interruption"])

    def test_partial_second_start_retains_first_and_negative_upper_contact(self):
        self.spec["parameters"]["kappa"]["bounds"] = [-5., -2.]
        seen = []
        def minimize(objective, start, **kwargs):
            seen.append(1)
            if len(seen) == 2:
                objective(start)
                raise RuntimeError("stop second start")
            x = start.copy()
            x[0] = -2.
            return result(x, 1.)
        attempts = []
        with self.assertRaisesRegex(RuntimeError, "second start"):
            self.block(minimize, attempts=attempts)
        self.assertEqual([row["returned"] for row in attempts], [True, False])
        self.assertEqual(attempts[0]["support_contacts"][0]["value"], -2.)
        self.assertEqual(attempts[1]["likelihood_calls_returned"], 1)

    def test_deadline_before_first_optimizer_entry_is_not_a_call(self):
        expired = [False]
        def check():
            if expired[0]:
                raise RuntimeError("deadline before optimizer")
        def checkpoint():
            expired[0] = True
        attempts = []
        with patch.object(core, "minimize") as minimize:
            with self.assertRaisesRegex(RuntimeError, "before optimizer"):
                core.optimize_subject_block({}, self.names, self.spec, self.mean, self.sd, self.starts,
                    likelihood=lambda *args: np.zeros(2), attempts=attempts, check=check, checkpoint=checkpoint)
        minimize.assert_not_called()
        self.assertFalse(attempts[0]["optimizer_entered"])
        self.assertEqual(attempts[0]["likelihood_calls_attempted"], 0)

    def test_eight_probe_order_and_no_other_index(self):
        selected = {"EI": "p01", "ER": "p09"}
        actual = [pilot.probe_identity(index, selected) for index in range(8)]
        expected = [(half, version, model) for half in range(2) for version in ("EI", "ER") for model in ("M0", "MZ")]
        self.assertEqual([(half, row["task_version"], row["model"]) for row, half in actual], expected)
        self.assertEqual(actual[0][0]["training_runs"], [1, 4])
        self.assertEqual(actual[4][0]["training_runs"], [2, 3])
        self.assertTrue(all(row["participant"] == selected[row["task_version"]] for row, half in actual))
        for invalid in (-1, 8, True, 1.5):
            with self.assertRaises(ValueError):
                pilot.probe_identity(invalid, selected)

    def benchmark_fixture(self):
        source = {"job_id": "source", "finished_utc": "2026-10-05T00:00:00+00:00"}
        selected, primary = {"EI": "p01", "ER": "p09"}, {"fixture": "scores"}
        plan = [({"participant": "p01", "start_index": index % 3}, None, None, None) for index in range(24)]
        reference = {"finished_utc": "2026-10-05T01:00:00+00:00", "probes": [
            {"meshes": {mesh: {"trial_log_scores": [-2.]*128} for mesh, _, _ in pilot.scalar.MESHES}} for _ in plan]}
        refined = deepcopy(self.spec)
        refined["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)
        report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": pilot.benchmark.PHASE,
            "generating_model": "MZ", "replicate_index": 0, "seed": pilot.scalar.shared.SEED, "rival": "urgency",
            "source_dir": str(pilot.SOURCE), "source_report": str(pilot.SOURCE / pilot.scalar.SOURCE_REPORT),
            "reference_report": str(pilot.benchmark.REFERENCE), "reference_job_id": pilot.benchmark.REFERENCE_JOB,
            "job_id": "46607960", "status": "urgency_batched_benchmark_completed", "benchmark_complete": True,
            "expected_likelihood_calls": 48, "completed_likelihood_calls": 48,
            "engineering_absolute_tolerance": 1e-12, "engineering_equivalent_at_1e_minus12": True,
            "maximum_absolute_trial_log_score_discrepancy": 0., "source_baseline_validated": True,
            "saved_primary_scores_replayed": True, "scalar_reference_validated": True,
            "source_participants": 108, "source_runs": 432, "source_trials": 27648,
            "implementation": self.spec, "refined_numerical_implementation": refined,
            "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
            "saved_primary_participant_scores": primary, "selected_probe_participants": selected,
            "scalar_reference_finished_utc": reference["finished_utc"],
            "saved_score_replay_absolute_tolerance": 1e-12, "saved_score_replay_max_absolute_discrepancy": 0.,
            "started_utc": "2026-10-05T02:00:00+00:00", "finished_utc": "2026-10-05T02:01:00+00:00",
            **{key: False for key in ("optimizer_called", "model_fitted", "observed_choices_or_RTs_used",
                                     "simulation_regenerated", "simulation_copied", "solver_adopted")}, "calls": []}
        for index, (identity, *_rest) in enumerate(plan):
            for mesh, dt, intervals in pilot.scalar.MESHES:
                report["calls"].append({**identity, "probe_index": index, "mesh": mesh, "dt_seconds": dt,
                    "spatial_intervals": intervals, "engineering_equivalent_at_1e_minus12": True,
                    "maximum_absolute_trial_log_score_discrepancy": 0., "mean_log_score_difference": 0.,
                    "trial_log_scores": [-2.]*128})
        return report, reference, source, primary, selected, plan

    def test_benchmark_identity_complete_panel_and_exact_vector_binding(self):
        report, reference, source, primary, selected, plan = self.benchmark_fixture()
        with patch.object(pilot.benchmark, "validate_reference") as validate_reference:
            pilot.validate_benchmark(report, reference, self.spec, source, primary, selected, plan)
            validate_reference.assert_called_once()
            for mutate in (
                lambda r: r.update(job_id="different"), lambda r: r.update(completed_likelihood_calls=47),
                lambda r: r.update(source_finished_utc="different"), lambda r: r.update(solver_adopted=True),
                lambda r: r["calls"][47]["trial_log_scores"].__setitem__(3, -2.0001),
                lambda r: r["calls"][2].update(start_index=9)):
                changed = deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    pilot.validate_benchmark(changed, reference, self.spec, source, primary, selected, plan)

    def run_fixture(self, directory, *, expire_on_final=False, reject_benchmark=False):
        source = directory / "source"
        source.mkdir()
        implementation = directory / "implementation.json"
        implementation.write_text(json.dumps(self.spec))
        (source / pilot.scalar.SOURCE_REPORT).write_text(json.dumps({"job_id": "source", "finished_utc": "done"}))
        (source / "generating_parameters.json").write_text("{}")
        reference = directory / "scalar_reference" / "reference.json"
        benchmark = directory / "batched_benchmark" / "benchmark.json"
        reference.parent.mkdir()
        benchmark.parent.mkdir()
        reference.write_text("{}")
        benchmark.write_text(json.dumps({"finished_utc": "done"}))
        selected = {"EI": "p01", "ER": "p09"}
        rows = [{"run": run, "trial_order": trial} for run in (1, 4) for trial in range(1, 65)]
        training = {person: {"version": version, "rows": rows} for version, person in selected.items()}
        clock, observed, publications = [0.], [], []
        original_write = pilot.write_report
        def write(output, report):
            original_write(output, report)
            publications.append(report["status"])
            if expire_on_final and report["status"] == "urgency_conditional_cost_probe_completed":
                clock[0] = 10.
        def optimize(data, names, spec, mean, sd, starts, **kwargs):
            observed.append((mean.copy(), sd.copy(), [start.copy() for start in starts]))
            for index in range(3):
                kwargs["attempts"].append({"start_index": index, "optimizer_entered": True, "returned": True,
                    "likelihood_calls_attempted": 1, "likelihood_calls_returned": 1, "likelihood_seconds": .01})
            return {"start_index": 0, "objective": 1., "success": False}
        args = SimpleNamespace(source_dir=source, implementation=implementation,
                               output_dir=directory / "output", probe_index=1, max_seconds=10)
        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "1", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}))
            for name, value in (("SOURCE", source), ("IMPLEMENTATION", implementation), ("BENCHMARK", benchmark)):
                stack.enter_context(patch.object(pilot, name, value))
            stack.enter_context(patch.object(pilot.benchmark, "REFERENCE", reference))
            stack.enter_context(patch.object(pilot, "require_compute"))
            for name in ("signal", "alarm"):
                stack.enter_context(patch.object(pilot.signal, name))
            stack.enter_context(patch.object(pilot.time, "monotonic", side_effect=lambda: clock[0]))
            cohort = stack.enter_context(patch.object(pilot.scalar.shared, "load_cohort", return_value={"full": "cohort"}))
            validate = stack.enter_context(patch.object(pilot.scalar.shared, "validate_source"))
            stack.enter_context(patch.object(pilot.scalar.shared, "load_split", return_value=(training, {})))
            replay = stack.enter_context(patch.object(pilot.scalar.shared, "replay_saved_scores", return_value=({}, 0.)))
            stack.enter_context(patch.object(pilot.scalar.shared, "make_data", return_value={}))
            stack.enter_context(patch.object(pilot.benchmark, "probe_plan", return_value=(selected, [])))
            stack.enter_context(patch.object(pilot, "validate_benchmark",
                side_effect=ValueError("benchmark mismatch") if reject_benchmark else None))
            stack.enter_context(patch.object(pilot, "optimize_subject_block", side_effect=optimize))
            stack.enter_context(patch.object(pilot, "write_report", side_effect=write))
            report = pilot.run(args)
            cohort.assert_called_once()
            validate.assert_called_once()
            replay.assert_called_once()
        saved = json.loads((args.output_dir / pilot.REPORT_NAME).read_text())
        self.assertEqual(report, saved)
        return report, observed, publications

    def test_runner_full_source_replay_precedes_fixed_prior_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            report, observed, _ = self.run_fixture(Path(directory))
        self.assertTrue(report["probe_complete"])
        self.assertTrue(report["conditional_optimization_complete"])
        self.assertFalse(report["best_start"]["success"])
        self.assertTrue(report["optimizer_called"])
        np.testing.assert_array_equal(observed[0][0], self.mean)
        np.testing.assert_array_equal(observed[0][1], np.exp(np.log(self.initial_sd)))
        for left, right in zip(observed[0][2], self.starts):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(report["population_hyperparameters_updated"])
        self.assertFalse(report["heldout_urgency_scores_computed"])
        with tempfile.TemporaryDirectory() as directory:
            refused, observed, _ = self.run_fixture(Path(directory), reject_benchmark=True)
        self.assertFalse(refused["optimizer_called"])
        self.assertEqual(observed, [])

    def test_final_publication_deadline_preserves_returned_facts_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, publications = self.run_fixture(Path(directory), expire_on_final=True)
        self.assertEqual(publications[-2:], ["urgency_conditional_cost_probe_completed", "urgency_conditional_cost_probe_incomplete"])
        self.assertTrue(report["probe_complete"])
        self.assertEqual(report["completed_starts"], 3)
        self.assertFalse(report["conditional_optimization_complete"])
        self.assertIsNone(report["best_start"])
        self.assertIn("during final publication", report["incomplete_reason"])


if __name__ == "__main__":
    unittest.main()
