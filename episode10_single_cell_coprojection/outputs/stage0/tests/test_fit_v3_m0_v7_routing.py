from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


TOOL = Path(__file__).parents[1] / "tools" / "fit_v3_m0_v7.py"
SPEC = importlib.util.spec_from_file_location("fit_v3_m0_v7_for_tests", TOOL)
assert SPEC and SPEC.loader
v7 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v7
SPEC.loader.exec_module(v7)


class V7RoutingTests(unittest.TestCase):
    def test_exactly_four_frozen_lambdas_use_Newton(self) -> None:
        result = v7._routing_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["Newton_lambda_count"], 4)
        self.assertEqual(result["V5_forward_lambda_count"], 8)
        self.assertTrue(result["lambda_1e5_uses_Newton"])

    def test_other_eight_lambdas_are_bitwise_V5(self) -> None:
        result = v7._eight_lambda_v5_parity_self_test()
        self.assertEqual(result["fits_compared"], 8)
        self.assertTrue(result["coefficient_arrays_bitwise_equal"])
        self.assertTrue(result["result_scalars_bitwise_equal"])
        self.assertTrue(result["telemetry_equal"])

    def test_lambda_1e5_Newton_warm_zero_equivalence(self) -> None:
        result = v7._lambda_1e5_newton_equivalence_self_test()
        self.assertEqual(result["ridge_lambda"], 1.0e-5)
        self.assertTrue(result["deterministic_zero_start_repeat"])
        self.assertLessEqual(
            result["objective_absolute_difference"],
            result["objective_absolute_tolerance"],
        )
        self.assertLessEqual(
            result["marginal_max_absolute_difference"],
            result["marginal_max_absolute_tolerance"],
        )
        self.assertLessEqual(
            result["log_probability_max_absolute_difference"],
            result["log_probability_max_absolute_tolerance"],
        )
        self.assertTrue(result["pass"])

    def test_inherited_exact_Hessian_255_solve_gate_remains_active(self) -> None:
        result = v7.v6._full_dimension_SPD_self_test()
        self.assertEqual(result["parameter_dimension"], 255)
        self.assertGreater(result["minimum_eigenvalue"], 0.0)
        self.assertLessEqual(
            result["solve_scaled_backward_error"],
            v7.v6.SOLVE_BACKWARD_ERROR_TOLERANCE,
        )
        self.assertLess(result["directional_derivative"], 0.0)


if __name__ == "__main__":
    unittest.main()
