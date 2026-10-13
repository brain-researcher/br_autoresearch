from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


TOOL = Path(__file__).parents[1] / "tools" / "fit_v3_m0_v5.py"
SPEC = importlib.util.spec_from_file_location("fit_v3_m0_v5_for_tests", TOOL)
assert SPEC and SPEC.loader
v5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v5
SPEC.loader.exec_module(v5)


class StrongConvexityCertificateTests(unittest.TestCase):
    def test_comprehensive_arithmetic_and_fail_closed_controls(self) -> None:
        result = v5._certificate_arithmetic_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["traced_state"]["certificate_pass"])
        self.assertFalse(
            result["traced_state"]["prior_successive_relative_change_pass"]
        )

    def test_exact_l2_is_reported_but_conservative_bound_is_gated(self) -> None:
        gradient = np.asarray([3.0e-8, 4.0e-8])
        decision = v5._strong_convexity_certificate(
            failure_message=(
                "strong-Wolfe zoom bracket reached floating-point resolution"
            ),
            objective=2.0,
            gradient=gradient,
            ridge_lambda=1.0,
            parameter_dimension=2,
            prior_relative_change=1.0e-4,
        )
        self.assertTrue(decision["certificate_pass"])
        self.assertEqual(
            decision["eligible_failure_reason"], "bracket_floating_resolution"
        )
        self.assertEqual(decision["exact_gradient_l2_norm"], math.hypot(*gradient))
        raw_conservative = (
            gradient.size * float(np.max(np.abs(gradient))) ** 2 / 2.0 / 2.0
        )
        self.assertGreaterEqual(
            decision["upward_rounded_conservative_relative_gap_bound"],
            raw_conservative,
        )
        self.assertFalse(decision["gate_used_exact_l2_bound"])
        self.assertTrue(
            decision["gate_used_upward_rounded_dimension_times_gradient_inf_bound"]
        )

    def test_known_strongly_convex_quadratic_bounds_do_not_understate(self) -> None:
        result = v5._known_quadratic_bound_self_test()
        self.assertEqual(result["status"], "pass")
        for case in result["known_optimum_quadratics"]:
            self.assertGreaterEqual(
                case["conservative_gap_bound"], case["actual_gap"]
            )

    def test_rounded_evaluator_stalls_then_certifies_without_a_step(self) -> None:
        result = v5._rounded_evaluator_stall_self_test()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["current_iterate_unchanged"])
        self.assertTrue(result["no_Wolfe_step_fabricated"])
        self.assertTrue(result["certificate"]["certificate_pass"])

    def test_full_255_parameter_ridge_coverage(self) -> None:
        result = v5._full_ridge_coverage_self_test()
        self.assertEqual(result["parameter_dimension"], 255)
        self.assertEqual(result["ridge_nonzero_gradient_coordinates"], 255)

    def test_ordinary_v4_path_is_bitwise_identical(self) -> None:
        result = v5._ordinary_v4_parity_self_test()
        self.assertGreater(result["accepted_steps"], 0)
        self.assertTrue(result["coefficient_arrays_bitwise_equal"])
        self.assertTrue(result["ordinary_result_scalars_bitwise_equal"])
        self.assertTrue(result["shared_Wolfe_telemetry_equal"])

    def test_certificate_branch_keeps_iterate_steps_and_history_unchanged(self) -> None:
        result = v5._synthetic_certificate_termination_self_test()
        self.assertEqual(result["accepted_steps"], 2)
        self.assertTrue(result["actual_prior_relative_change_still_failed"])
        self.assertTrue(result["returned_iterate_equals_prefailure_iterate"])
        self.assertTrue(result["accepted_step_and_history_counts_unchanged"])
        self.assertTrue(result["no_pseudo_step"])

    def test_no_move_or_ulp_proximity_is_not_an_eligible_reason(self) -> None:
        for message in ("exact no-move", "objective within one ULP"):
            decision = v5._strong_convexity_certificate(
                failure_message=message,
                objective=1.0,
                gradient=np.asarray([0.0]),
                ridge_lambda=1.0,
                parameter_dimension=1,
                prior_relative_change=0.0,
            )
            self.assertFalse(decision["attempted"])
            self.assertFalse(decision["failure_reason_eligible"])
            self.assertFalse(decision["certificate_pass"])


if __name__ == "__main__":
    unittest.main()
