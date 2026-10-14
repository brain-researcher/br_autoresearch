"""Scope checks for a failed local trial; no empirical data or fitting."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import fit_m4_task_a as fitting


class OptimizerFailureScopeTests(unittest.TestCase):
    def _objective(self):
        return SimpleNamespace(modes=np.zeros((1, 2, 2)),
            matrix=Mock(return_value=np.zeros((2, 2))),
            value_gradient=Mock(return_value=(1., np.zeros(1))),
            final_stretch_diagnostics=Mock(return_value={}))

    def _call(self, objective, minimizer):
        with patch.object(fitting, "PairedM4TaskAObjective", return_value=objective), \
                patch.object(fitting, "minimize", side_effect=minimizer):
            return fitting.fit_m4_task_a(None, None, None, None, global_gain=1.,
                condition_weights=np.ones(2), mode_matrices=objective.modes,
                rank_bound=2, dof_bound=1, spectral_norm_bound=.2,
                rank_rtol=1e-8, numerical_atol=1e-10, optimizer_ftol=1e-10,
                max_iterations=10)

    def test_nonfinite_objective_guard_has_specific_refusal_type(self):
        objective = object.__new__(fitting.PairedM4TaskAObjective)
        objective.matrix = Mock(return_value=np.zeros((2, 2)))
        objective.rotated_source = np.eye(2)
        objective.active_target = np.zeros((2, 2))
        objective.weights, objective.gain = np.full(2, .5), 1.
        objective.modes, objective.complement_loss = np.zeros((1, 2, 2)), 0.
        # Exercise the real finite-value/gradient guard with a failed matrix
        # exponential, avoiding an intentionally enormous scientific solve.
        with patch.object(fitting, "expm", return_value=np.full((2, 2), np.inf)), \
                patch.object(fitting, "expm_frechet", return_value=np.zeros((2, 2))), \
                np.errstate(invalid="ignore", over="ignore"):
            with self.assertRaises(fitting.M4ObjectiveNonfinite):
                objective.value_gradient(np.zeros(1))

    def test_nonfinite_trial_refuses_local_solve_and_retains_original_cause(self):
        objective = self._objective()
        failure = fitting.M4ObjectiveNonfinite("trial guard")
        objective.value_gradient.side_effect = [(1., np.zeros(1)), failure]
        def evaluate_trial(fun, zero, **kwargs):
            return fun(np.ones_like(zero))
        with self.assertRaises(fitting.M4OptimizerTrialNonfinite) as raised:
            self._call(objective, evaluate_trial)
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(objective.value_gradient.call_count, 2)
        objective.final_stretch_diagnostics.assert_not_called()

    def test_initial_and_final_nonfinite_values_do_not_become_optimizer_retries(self):
        for stage in ("initial", "final"):
            with self.subTest(stage=stage):
                objective = self._objective()
                failure = fitting.M4ObjectiveNonfinite(stage+" guard")
                objective.value_gradient.side_effect = ([failure] if stage == "initial"
                    else [(1., np.zeros(1)), failure])
                minimizer = Mock(return_value=SimpleNamespace(success=True, x=np.zeros(1), nit=1))
                with self.assertRaises(fitting.M4ObjectiveNonfinite) as raised:
                    self._call(objective, minimizer)
                self.assertIs(raised.exception, failure)
                if stage == "initial":
                    minimizer.assert_not_called()
                else:
                    minimizer.assert_called_once()

    def test_unrelated_trial_arithmetic_failure_is_not_reclassified(self):
        objective = self._objective()
        failure = ArithmeticError("different arithmetic defect")
        with self.assertRaises(ArithmeticError) as raised:
            self._call(objective, Mock(side_effect=failure))
        self.assertIs(raised.exception, failure)
        objective.final_stretch_diagnostics.assert_not_called()


if __name__ == "__main__":
    unittest.main()
