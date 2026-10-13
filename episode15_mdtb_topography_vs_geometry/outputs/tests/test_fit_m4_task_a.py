"""Focused paired-stretch arithmetic checks; no MDTB data or qualification sweep."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from fit_m4_task_a import PairedM4TaskAObjective, fit_m4_task_a
from spatial_deformation_operators import apply_m3, apply_m4


class PairedM4FitTests(unittest.TestCase):
    def setUp(self):
        self.B = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                           [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])/np.sqrt(2.)
        self.O = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
        self.X = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 1.], [1., -1., 2.]])@self.B.T
        self.modes = np.array([np.diag([1., -1., 0.]), [[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]])/np.sqrt(2.)
        self.theta = np.array([.12, -.08])
        self.C = np.einsum("p,pij->ij", self.theta, self.modes)
        self.gain = 1.7
        self.Y = self.gain*apply_m4(self.X, self.B, self.O, self.C)
        self.settings = dict(global_gain=self.gain, condition_weights=np.array([1., 2., 3., 1.]),
                             mode_matrices=self.modes, rank_bound=2, dof_bound=2,
                             rank_rtol=1e-10, numerical_atol=1e-9)

    def objective(self, **changes):
        settings = {**self.settings, **changes}
        return PairedM4TaskAObjective(self.X, self.Y, self.B, self.O, **settings)

    def fit(self, **changes):
        settings = {**self.settings, "spectral_norm_bound": .4,
                    "optimizer_ftol": 1e-13, "max_iterations": 150, **changes}
        return fit_m4_task_a(self.X, self.Y, self.B, self.O, **settings)

    def test_frechet_gradient_matches_independent_centered_differences(self):
        objective = self.objective()
        theta = np.array([-.09, .18])
        loss, gradient = objective.value_gradient(theta)
        h = 1e-6
        finite_difference = np.array([(objective.value_gradient(theta+h*row)[0]-
                                      objective.value_gradient(theta-h*row)[0])/(2*h)
                                     for row in np.eye(2)])
        np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-8)
        predictions = objective.predict(self.X, theta)
        weights = self.settings["condition_weights"]/self.settings["condition_weights"].sum()
        self.assertAlmostEqual(loss, np.sum(weights[:, None]*(predictions-self.Y)**2), places=12)

    def test_noncommuting_stretch_recovered_with_paired_anchor_unchanged(self):
        originals = [a.copy() for a in (self.X, self.Y, self.B, self.O, self.modes)]
        result = self.fit()
        np.testing.assert_allclose(result["C"], self.C, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(result["O"], self.O)
        self.assertEqual(result["gain"], self.gain)
        self.assertLess(result["weighted_task_a_squared_error"], 1e-11)
        self.assertLess(result["zero_stretch_replay_max_abs"], 1e-12)
        for name in ("unique_solution_certified", "global_optimum_certified", "locality_certified", "empirical_recipe_selected"):
            self.assertFalse(result[name])
        for current, original in zip((self.X, self.Y, self.B, self.O, self.modes), originals):
            np.testing.assert_array_equal(current, original)

    def test_spectral_bound_is_enforced_without_postfit_clipping(self):
        result = self.fit(spectral_norm_bound=.025)
        self.assertLessEqual(np.max(np.abs(result["C_eigenvalues"])), .025+1e-9)
        self.assertGreater(result["weighted_task_a_squared_error"], 1e-5)
        self.assertLess(result["weighted_task_a_squared_error"], result["paired_zero_stretch_loss"])
        self.assertAlmostEqual(np.trace(result["C"]), 0., places=13)
        self.assertEqual(result["joint_mode_rank"], 2)

    def test_zero_mode_replays_m3_and_common_scaling_of_weights_is_invariant(self):
        objective = self.objective()
        np.testing.assert_allclose(objective.predict(self.X, np.zeros(2)),
                                   self.gain*apply_m3(self.X, self.B, self.O), atol=1e-14, rtol=0)
        scaled = self.objective(condition_weights=1e200*self.settings["condition_weights"])
        for a, b in zip(objective.value_gradient(self.theta), scaled.value_gradient(self.theta)):
            np.testing.assert_allclose(a, b, atol=1e-14, rtol=0)

    def test_invalid_constraints_and_unidentified_mode_parameterizations_refuse(self):
        bad_modes = self.modes.copy()
        bad_modes[0, 0, 0] += .1
        for settings in ({"mode_matrices": bad_modes}, {"mode_matrices": np.stack([self.modes[0]]*2)},
                         {"rank_bound": 1}, {"dof_bound": 1}, {"global_gain": 0.},
                         {"condition_weights": [1., 0., 1., 1.]}, {"rank_rtol": True}):
            with self.subTest(settings=list(settings)), self.assertRaises(ValueError):
                self.objective(**settings)
        with self.assertRaisesRegex(ValueError, "source design"):
            PairedM4TaskAObjective(np.zeros_like(self.X), self.Y, self.B, self.O, **self.settings)

    def test_optimizer_failure_is_not_returned_as_a_fit(self):
        with self.assertRaisesRegex(ArithmeticError, "did not converge"):
            self.fit(max_iterations=1)

    def test_small_modes_cannot_hide_coefficient_amplified_constraint_failures(self):
        # Absolute-only input checks would accept this trace defect, then
        # theta=1e11 would yield trace(C)=.05 within a spectral bound of .4.
        modes = self.modes.copy()*1e-12
        modes[0] = 1e-12*np.diag([1., -.5, 0.])
        with self.assertRaisesRegex(ValueError, "trace free"):
            self.objective(mode_matrices=modes)
        # A true trace-free small parameterization remains allowed.
        objective = self.objective(mode_matrices=1e-12*self.modes)
        metrics = objective.final_stretch_diagnostics(1e12*self.theta, .4)
        self.assertLess(abs(metrics["C_trace"]), 1e-14)
        self.assertEqual(metrics["C_numerical_rank"], 2)
        with self.assertRaisesRegex(ArithmeticError, "spectral norm"):
            objective.final_stretch_diagnostics(1e12*self.theta, .01)


if __name__ == "__main__":
    unittest.main()
