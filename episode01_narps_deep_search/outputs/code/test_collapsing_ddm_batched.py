"""Candidate/scalar numerical equivalence checks; task-owned compute only.

The1e-12 absolute bound is an implementation-equivalence check, not evidence
that either fixed mesh is accurate enough for fitted-support inference.
"""
from __future__ import annotations

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

import collapsing_ddm_batched as candidate
import hierarchical_behavior as original


MESHES = ((.002, 200), (.001, 400))
EQUIVALENCE_ATOL = 1e-12


class BatchedCollapsingTests(unittest.TestCase):
    def assert_distribution(self, actual, expected):
        np.testing.assert_array_equal(actual.time_edges, expected.time_edges)
        np.testing.assert_allclose(actual.reject_mass, expected.reject_mass, rtol=0, atol=EQUIVALENCE_ATOL)
        np.testing.assert_allclose(actual.accept_mass, expected.accept_mass, rtol=0, atol=EQUIVALENCE_ATOL)
        self.assertLessEqual(abs(actual.survival_mass - expected.survival_mass), EQUIVALENCE_ATOL)
        self.assertLessEqual(abs(actual.reject_mass.sum() + actual.accept_mass.sum() + actual.survival_mass - 1), 1e-8)
        self.assertGreaterEqual(actual.reject_mass.min(), -1e-13)
        self.assertGreaterEqual(actual.accept_mass.min(), -1e-13)

    def test_both_meshes_match_scalar_flux_survival_and_density_over_full_decision_horizon(self):
        drifts = np.array([-1.3, 0., 2.1])
        before = drifts.copy()
        for dt, intervals in MESHES:
            with self.subTest(dt=dt, intervals=intervals):
                actual = candidate.solve_many(drifts, 1.5, .431, .2, 3.75, dt=dt, intervals=intervals)
                for drift, distribution in zip(drifts, actual):
                    expected = candidate.scalar.solve(float(drift), 1.5, .431, .2, 3.75, dt=dt, intervals=intervals)
                    self.assert_distribution(distribution, expected)
                    times = np.array([-.1, 0., dt / 2, dt, .137, 1.23, 3.75, 3.76])
                    for choice in (0, 1):
                        np.testing.assert_allclose(distribution.response_density(times, choice),
                            expected.response_density(times, choice), rtol=0, atol=EQUIVALENCE_ATOL)
        np.testing.assert_array_equal(drifts, before)

    def test_single_block_and_zero_urgency_preserve_original_initial_mass_interpolation(self):
        for dt, intervals in MESHES:
            for z in (.5, .317):
                with self.subTest(dt=dt, z=z):
                    actual, = candidate.solve_many([.7], 1.2, z, 0., .4, dt=dt, intervals=intervals)
                    expected = candidate.scalar.solve(.7, 1.2, z, 0., .4, dt=dt, intervals=intervals)
                    self.assert_distribution(actual, expected)

    def test_order_duplicate_and_changed_neighbor_leave_each_block_independent(self):
        for dt, intervals in MESHES:
            initial = candidate.solve_many([-.8, .2, 1.6], 1.1, .37, .1, .3, dt=dt, intervals=intervals)
            reordered = candidate.solve_many([1.6, -.8, .2, .2], 1.1, .37, .1, .3, dt=dt, intervals=intervals)
            changed = candidate.solve_many([-.8, -4., 1.6], 1.1, .37, .1, .3, dt=dt, intervals=intervals)
            for actual, expected in zip(reordered, (initial[2], initial[0], initial[1], initial[1])):
                self.assert_distribution(actual, expected)
            self.assert_distribution(changed[0], initial[0])
            self.assert_distribution(changed[2], initial[2])
            self.assertFalse(np.shares_memory(reordered[2].accept_mass, reordered[3].accept_mass))
            self.assertFalse(np.shares_memory(reordered[2].time_edges, reordered[3].time_edges))

    def test_every_step_has_zero_cross_block_couplings_and_one_original_banded_solve(self):
        real_solve, calls = candidate.solve_banded, []
        count, intervals, horizon, dt = 3, 200, .02, .002
        block_size = intervals - 1

        def inspect(bandwidth, matrix, rhs, **kwargs):
            self.assertEqual(bandwidth, (1, 1))
            self.assertEqual(matrix.shape, (3, count * block_size))
            self.assertEqual(rhs.shape, (count * block_size,))
            self.assertEqual(kwargs, {"check_finite": False})
            np.testing.assert_array_equal(matrix[0, np.arange(count) * block_size], np.zeros(count))
            np.testing.assert_array_equal(matrix[2, (np.arange(count) + 1) * block_size - 1], np.zeros(count))
            calls.append(1)
            return real_solve(bandwidth, matrix, rhs, **kwargs)

        with patch.object(candidate, "solve_banded", side_effect=inspect):
            candidate.solve_many([-1., 0., 1.], 1.5, .5, .1, horizon, dt=dt, intervals=intervals)
        self.assertEqual(len(calls), math.ceil(horizon / dt))

    def test_urgency_likelihood_matches_scalar_for_responses_fast_contaminants_and_misses(self):
        data = {"gain": np.array([10., 20., 10., 20., 10., 20., 15., 15.]),
                "loss": np.array([10., 10., 15., 15., 10., 10., 15., 15.]),
                "rt": np.array([np.nan, .1, .4, 1.2, 4., np.nan, .252, 2.]),
                "choice": np.array([-1, 0, 1, 0, 1, -1, 1, 0])}
        frozen = {key: value.copy() for key, value in data.items()}
        for model, urgency in (("M0", 0.), ("MZ", .2)):
            names = original.parameter_names(model, "urgency")
            params = {"kappa": math.log(.8), "log_a": math.log(1.5), "nondecision_logit": math.log(1 / 15),
                      "rho": -.3, "urgency": urgency}
            theta = np.array([params[name] for name in names])
            for dt, intervals in MESHES:
                spec = {"contaminant": {"fraction": .02, "miss_probability": .01},
                        "urgency_solver": {"dt_seconds": dt, "spatial_intervals": intervals}}
                before = deepcopy(spec)
                with self.subTest(model=model, dt=dt):
                    expected = original.trial_likelihood(theta, names, data, spec, "urgency")
                    actual = candidate.trial_likelihood(theta, names, data, spec)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=EQUIVALENCE_ATOL)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertEqual(actual[1], np.log(.02 * .99 / 8))
                    self.assertEqual(spec, before)
        for key in data:
            np.testing.assert_array_equal(data[key], frozen[key])

    def test_likelihood_batches_exact_unique_drifts_without_rounding_and_restores_row_order(self):
        data = {"gain": np.array([10., np.nextafter(10., np.inf), 10., np.nextafter(10., np.inf)]),
                "loss": np.full(4, 10.), "rt": np.array([1., 1.2, np.nan, .5]), "choice": np.array([0, 1, -1, 1])}
        names = original.parameter_names("M0", "urgency")
        theta = np.array([0., 0., math.log(1 / 15), .1])
        spec = {"contaminant": {"fraction": .02, "miss_probability": .01},
                "urgency_solver": {"dt_seconds": .002, "spatial_intervals": 200}}
        drift = candidate.deadline_ddm.drift_rate(data["gain"], data["loss"], 0.)
        self.assertEqual(len(np.unique(drift)), 2)
        real_solve = candidate.solve_many
        with patch.object(candidate, "solve_many", wraps=real_solve) as solve:
            actual = candidate.trial_likelihood(theta, names, data, spec)
        self.assertEqual(solve.call_count, 1)
        np.testing.assert_array_equal(solve.call_args.args[0], np.unique(drift))
        expected = original.trial_likelihood(theta, names, data, spec, "urgency")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=EQUIVALENCE_ATOL)

    def test_existing_support_and_mass_checks_are_retained_per_drift(self):
        for kwargs in ({"z": 0.}, {"z": .999}, {"a0": 0.}, {"urgency": -.1}, {"decision_deadline": 0.},
                       {"dt": 0.}, {"intervals": 19}, {"intervals": 200.5}, {"urgency": 301.}):
            args = dict(a0=1., z=.5, urgency=.1, decision_deadline=1., dt=.002, intervals=200)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                candidate.solve_many([0.], **args)
        for drifts in ([np.nan], [[0.]]):
            with self.subTest(drifts=drifts), self.assertRaises(ValueError):
                candidate.solve_many(drifts, 1., .5, .1, 1.)
        real_solve = candidate.solve_banded

        def corrupt_one_block(*args, **kwargs):
            values = real_solve(*args, **kwargs)
            values[:199] = 0.  # Only one drift loses its initial probability.
            return values

        with patch.object(candidate, "solve_banded", side_effect=corrupt_one_block), \
                self.assertRaisesRegex(FloatingPointError, "probability mass"):
            candidate.solve_many([0., 1.], 1., .5, .1, .002)


if __name__ == "__main__":
    unittest.main()
