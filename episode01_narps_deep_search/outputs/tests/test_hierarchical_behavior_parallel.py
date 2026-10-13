"""Source-free arithmetic/worker-lifecycle comparisons to the original fitter.

Run only in the coordinator's allocated compute step with at least two CPUs.
Top-level likelihoods/minimizers are deliberately picklable for spawn workers.
"""
from copy import deepcopy
import multiprocessing
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import hierarchical_behavior as original
import hierarchical_behavior_parallel as parallel


THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def training_fixture():
    # Deliberately unsorted insertion, with task versions interleaved by ID.
    return {person: {"version": version, "rows": [
        {"gain": str(marker), "loss": str(trial + 1), "response_time": "1.1", "choice": "1"}
        for trial in range(3)]}
        for person, version, marker in (("p04", "ER", 4), ("p02", "ER", 2),
                                      ("p03", "EI", 3), ("p01", "EI", 1))}


def specification():
    return {"parameters": {name: {
            "population_center": center, "population_scale": 1.5,
            "version_mean_scale": .8, "individual_scale_center": scale,
            "individual_log_scale_sd": .4, "bounds": bounds,
            "individual_log_scale_bounds": [-4., 1.]}
        for name, center, scale, bounds in (
            ("kappa", 0., .13, [-3., 3.]), ("log_a", 0., .31, [-3., 3.]),
            ("nondecision_logit", -.2, .7, [-3., 3.]), ("rho", 0., .2, [-3., 3.]),
            ("urgency", .1, .4, [0., 2.]))},
        "optimizer": {"maximum_sweeps": 4,
            "initial_standardized_offsets": [[-.5]*8, [.5]*8],
            "subject_maximum_iterations": 80, "subject_ftol": 1e-12,
            "subject_gtol": 1e-8, "scale_xatol": 1e-8, "joint_relative_tolerance": 0.}}


def quadratic_likelihood(theta, names, data, spec, rival):
    marker = data["gain"][0]
    target = np.array([.1*marker, -.15+.03*marker, -.3+.02*marker,
                       .1-.01*marker, .04+.01*marker])[:len(names)]
    return np.full(len(data["gain"]), -.5*np.sum((theta-target)**2) - .01*marker)


def cancellation_likelihood(theta, names, data, spec, rival):
    # Ordered NLLs1e16,1,-1e16,1 expose a changed summation order.
    value = (-1e16, -1., 1e16, -1.)[int(data["gain"][0])-1]
    values = np.zeros(len(data["gain"]))
    values[0] = value
    return values


def stationary_minimizer(objective, start, method, bounds, options):
    return SimpleNamespace(x=np.asarray(start).copy(), fun=objective(start), success=True,
                           message="stationary fixture", nfev=1, njev=0, nit=0, status=0)


def failing_likelihood(theta, names, data, spec, rival):
    raise RuntimeError("deliberate worker likelihood failure")


def slow_likelihood(theta, names, data, spec, rival):
    time.sleep(5.)
    return quadratic_likelihood(theta, names, data, spec, rival)


def entered_blocking_likelihood(theta, names, data, spec, rival):
    # A source-free temporary marker proves that the deadline interrupts an
    # entered likelihood, rather than merely expiring during spawn/import.
    Path(spec["_test_entry_marker"]).write_text("objective entered\n")
    time.sleep(20.)
    return quadratic_likelihood(theta, names, data, spec, rival)


class ParallelFitterTests(unittest.TestCase):
    def setUp(self):
        self.training, self.spec = training_fixture(), specification()
        self.environment = patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "2", **{name: "1" for name in THREADS}})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.guard = patch.object(parallel, "require_compute")
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def candidate(self, *, likelihood=quadratic_likelihood, **kwargs):
        return parallel.fit_partially_pooled_parallel(self.training, "MZ", "urgency", self.spec,
            workers=kwargs.pop("workers", 2), max_seconds=kwargs.pop("max_seconds", 30.),
            likelihood=likelihood, **kwargs)

    def assert_no_new_children(self, before):
        self.assertEqual({child.pid for child in multiprocessing.active_children()} - before, set())

    def test_actual_original_multisweep_fit_matches_two_spawn_workers_exactly(self):
        untouched_training, untouched_spec = deepcopy(self.training), deepcopy(self.spec)
        with patch.object(original, "trial_likelihood", quadratic_likelihood):
            expected = original.fit_partially_pooled(self.training, "MZ", "urgency", self.spec)
        before = {child.pid for child in multiprocessing.active_children()}
        events = []
        actual = self.candidate(progress=events.append)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["participants"], ["p01", "p02", "p03", "p04"])
        self.assertEqual(len(actual["objective_history"]), 4)
        self.assertNotEqual(actual["version_means"][0], actual["version_means"][1])
        self.assertEqual(self.training, untouched_training)
        self.assertEqual(self.spec, untouched_spec)
        conditional = [event for event in events if event["stage"] == "conditional"]
        self.assertEqual(len(conditional), 16)
        for event in conditional:
            self.assertEqual(event["result"]["attempt_count"], 3 if event["sweep"] == 1 else 1)
        for sweep in range(1, 5):
            self.assertEqual({event["participant"] for event in conditional if event["sweep"] == sweep},
                             set(self.training))
        self.assert_no_new_children(before)

    def conditional_payload(self):
        names = original.parameter_names("MZ", "urgency")
        priors = [self.spec["parameters"][name] for name in names]
        centers = np.array([prior["population_center"] for prior in priors])
        initial_sd = np.array([prior["individual_scale_center"] for prior in priors])
        bounds = [prior["bounds"] for prior in priors]
        starts = [centers.copy()] + [np.clip(centers+np.asarray(offset[:len(names)])*initial_sd,
            [bound[0] for bound in bounds], [bound[1] for bound in bounds])
            for offset in self.spec["optimizer"]["initial_standardized_offsets"]]
        return {"index": 0, "participant": "p01", "data": original.make_data(self.training["p01"]["rows"]),
            "names": names, "spec": self.spec, "rival": "urgency", "mean": centers,
            "sd": np.exp(np.log(initial_sd)), "starts": starts, "deadline": time.monotonic()+30.}

    def test_conditional_objective_options_and_first_failed_minimum_tie(self):
        payload, calls = self.conditional_payload(), []
        def minimize(objective, start, method, bounds, options):
            index = len(calls)
            np.testing.assert_array_equal(start, payload["starts"][index])
            expected = float(-quadratic_likelihood(start, payload["names"], payload["data"],
                self.spec, "urgency").sum() + .5*np.sum(((start-payload["mean"])/payload["sd"])**2))
            self.assertEqual(objective(start), expected)
            self.assertEqual(method, "L-BFGS-B")
            self.assertEqual(bounds, [self.spec["parameters"][name]["bounds"] for name in payload["names"]])
            self.assertEqual(options, {"maxiter": 80, "ftol": 1e-12, "gtol": 1e-8})
            calls.append(index)
            return SimpleNamespace(x=start.copy(), fun=(3., 1., 1.)[index],
                                   success=index != 1, message="selected failure" if index == 1 else "success")
        actual = parallel._conditional_task(payload, quadratic_likelihood, minimize)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(actual["attempt_count"], 3)
        self.assertEqual(actual["selected_start_index"], 1)
        self.assertFalse(actual["success"])
        self.assertEqual(actual["message"], "selected failure")
        np.testing.assert_array_equal(actual["parameters"], payload["starts"][1])

    def test_no_finite_conditional_start_is_a_technical_error(self):
        payload = self.conditional_payload()
        def minimize(objective, start, **kwargs):
            return SimpleNamespace(x=start.copy(), fun=np.inf, success=False, message="nonfinite")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            parallel._conditional_task(payload, quadratic_likelihood, minimize)

    def test_cancellation_sensitive_objective_keeps_original_order(self):
        with patch.object(original, "trial_likelihood", cancellation_likelihood), \
                patch.object(original, "minimize", stationary_minimizer):
            expected = original.fit_partially_pooled(self.training, "MZ", "urgency", self.spec)
        actual = self.candidate(likelihood=cancellation_likelihood, subject_minimizer=stationary_minimizer)
        self.assertEqual(actual, expected)
        self.assertGreaterEqual(len(actual["objective_history"]), 2)

    def test_real_worker_failure_propagates_and_all_children_are_joined(self):
        before = {child.pid for child in multiprocessing.active_children()}
        handler = signal.getsignal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError, "deliberate worker likelihood failure") as caught:
            self.candidate(likelihood=failing_likelihood)
        self.assertTrue(caught.exception.parallel_progress["workers_cleaned_up"])
        self.assertEqual(caught.exception.parallel_progress["stage"], "conditional")
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)
        self.assert_no_new_children(before)

    def test_short_deadline_stops_children_without_returning_partial_fit(self):
        before = {child.pid for child in multiprocessing.active_children()}
        handler = signal.getsignal(signal.SIGTERM)
        with self.assertRaises((RuntimeError, TimeoutError)) as caught:
            self.candidate(likelihood=slow_likelihood, max_seconds=.2)
        self.assertTrue(caught.exception.parallel_progress["workers_cleaned_up"])
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)
        self.assert_no_new_children(before)

    def test_deadline_interrupts_an_entered_objective_and_joins_workers(self):
        before = {child.pid for child in multiprocessing.active_children()}
        handler = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "objective_entered.txt"
            self.spec["_test_entry_marker"] = str(marker)
            with self.assertRaises((RuntimeError, TimeoutError)) as caught:
                self.candidate(likelihood=entered_blocking_likelihood, max_seconds=8.)
            self.assertTrue(marker.is_file(), "Fixture must enter an objective before its deadline")
            self.assertEqual(marker.read_text(), "objective entered\n")
        self.assertTrue(caught.exception.parallel_progress["workers_cleaned_up"])
        self.assertEqual(caught.exception.parallel_progress["stage"], "conditional")
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)
        self.assert_no_new_children(before)

    def test_explicit_cpu_thread_and_duration_guards(self):
        before = {child.pid for child in multiprocessing.active_children()}
        for workers in (0, -1, 3, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.candidate(workers=workers)
        for seconds in (0., -1., float("inf"), float("nan")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.candidate(max_seconds=seconds)
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "2"}), self.assertRaises(ValueError):
            self.candidate()
        self.assert_no_new_children(before)


if __name__ == "__main__":
    unittest.main()
