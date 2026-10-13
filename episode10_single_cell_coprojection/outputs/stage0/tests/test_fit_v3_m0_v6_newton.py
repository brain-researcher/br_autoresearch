from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


TOOL = Path(__file__).parents[1] / "tools" / "fit_v3_m0_v6.py"
SPEC = importlib.util.spec_from_file_location("fit_v3_m0_v6_for_tests", TOOL)
assert SPEC and SPEC.loader
v6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v6
SPEC.loader.exec_module(v6)


class ExactHessianNewtonTests(unittest.TestCase):
    def test_pair_moments_match_exhaustive_subsets_for_every_K(self) -> None:
        result = v6._exhaustive_pair_moment_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["K_values_checked"], list(range(7)))
        self.assertLess(result["maximum_covariance_absolute_error"], 2.0e-12)

    def test_K1_covariance_is_multinomial_covariance(self) -> None:
        result = v6._multinomial_K1_covariance_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertLess(result["maximum_absolute_error"], 1.0e-13)

    def test_Hessian_matches_gradient_and_variance_identity(self) -> None:
        result = v6._hessian_derivative_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertLess(
            result["finite_difference_max_absolute_error"], 2.0e-8
        )
        self.assertGreaterEqual(
            result["minimum_eigenvalue"], result["ridge_lower_bound"] - 2.0e-12
        )

    def test_full_255_parameter_Hessian_is_SPD_and_solvable(self) -> None:
        result = v6._full_dimension_SPD_self_test()
        self.assertEqual(result["parameter_dimension"], 255)
        self.assertGreater(result["minimum_eigenvalue"], 0.0)
        self.assertLessEqual(
            result["solve_scaled_backward_error"],
            v6.SOLVE_BACKWARD_ERROR_TOLERANCE,
        )
        self.assertLess(result["directional_derivative"], 0.0)

    def test_solve_checks_reject_nonfinite_asymmetric_and_indefinite(self) -> None:
        result = v6._solve_fail_closed_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["negative_controls"],
            [
                "nonfinite",
                "asymmetric",
                "indefinite_beyond_damping_bound",
            ],
        )

    def test_strong_grid_is_bitwise_V5(self) -> None:
        result = v6._strong_grid_v5_parity_self_test()
        self.assertTrue(result["coefficient_arrays_bitwise_equal"])
        self.assertTrue(result["result_scalars_bitwise_equal"])
        self.assertTrue(result["telemetry_equal"])

    def test_weak_grid_Newton_is_deterministic_and_exact_Wolfe(self) -> None:
        result = v6._weak_newton_recovery_self_test()
        self.assertEqual(result["repeat_coefficient_max_absolute_error"], 0.0)
        self.assertLessEqual(
            result["gradient_infinity_norm"], v6.GRADIENT_TOLERANCE
        )
        self.assertLessEqual(
            result["relative_objective_change"],
            v6.RELATIVE_OBJECTIVE_TOLERANCE,
        )
        self.assertTrue(
            result["all_accepted_steps_satisfied_exact_strong_Wolfe"]
        )


if __name__ == "__main__":
    unittest.main()
