"""Small analytical cases for the diagonal dual certificate; compute only."""
import math
import unittest

import numpy as np

from diagonal_m4_dual import _growth, _selected, scalar_dual_certificate
from paired_stretch_variants import M4CertificateUnverified
from proposed_m4_certificate import _i


class DiagonalDualTests(unittest.TestCase):
    def test_interior_trace_root_and_global_growth(self):
        rho = math.log(1.25)
        p = np.linspace(.02, .04, 16)
        # lambda=0 has the unique scalar minimizers log(d/p), with zero trace.
        center = np.linspace(-.12, .12, 16)
        d = p*np.exp(center)
        result = scalar_dual_certificate(p, d, np.full(8, .125), rho, check=lambda: None)
        self.assertLessEqual(result["lambda_interval"][0], 0)
        self.assertGreaterEqual(result["lambda_interval"][1], 0)
        self.assertEqual(result["coordinate_branches"], ("interior",)*16)
        for expected, (lo, hi) in zip(center, result["coordinate_minimizer_intervals"]):
            self.assertLessEqual(lo, expected+2e-15)
            self.assertGreaterEqual(hi, expected-2e-15)
        self.assertTrue(result["unique_global_minimum_sufficient_dual"])
        self.assertFalse(result["full_domain_positive_curvature_asserted"])

    def test_unique_boundary_optimum_with_negative_curvature(self):
        rho = math.log(1.25)
        p = np.full(16, .01)
        # At lambda=.02, d=.01 favors the lower bound and d=.08 the upper.
        # For the latter f'' < 0 over the ENTIRE box. Uniqueness comes from
        # the one-sided value gap, not a Hessian/convex-sublevel assertion.
        d = np.r_[np.full(8, .01), np.full(8, .08)]
        result = scalar_dual_certificate(p, d, np.full(8, .125), rho, check=lambda: None)
        self.assertEqual(result["coordinate_branches"], ("lower",)*8+("upper",)*8)
        self.assertLess(4*.01*math.exp(2*rho)-2*.08*math.exp(rho), 0)
        self.assertTrue(all(value > 1e-10*.125 for value in result["scalar_growth_lower_bounds"]))

    def test_dual_jump_is_unresolved_not_nonuniqueness_claim(self):
        # Sixteen identical strictly concave scalar functions switch together.
        # The constraint has many equally good 8-lower/8-upper vertices. No
        # unique scalar product witness exists and no optimizer is accepted.
        with self.assertRaises(M4CertificateUnverified) as stopped:
            scalar_dual_certificate(np.full(16, .01), np.full(16, .08),
                                    np.full(8, .125), math.log(1.25), check=lambda: None)
        self.assertIn("branch switch", str(stopped.exception))
        diagnostics = stopped.exception.diagnostics
        self.assertEqual(diagnostics["left_nonstrict_coordinate_count"], 0)
        self.assertEqual(diagnostics["right_nonstrict_coordinate_count"], 0)
        self.assertGreater(diagnostics["left_trace_interval"][0], 0)
        self.assertLess(diagnostics["right_trace_interval"][1], 0)
        self.assertEqual(diagnostics["endpoint_to_endpoint_switch_count"], 16)

    def test_growth_checks_whole_interval_and_respects_caller_stop(self):
        rho = math.log(1.25)
        p, d = np.full(16, .01), np.r_[np.full(8, .01), np.full(8, .08)]
        # This dual value is just below the two-endpoint crossover. The
        # whole-interval Taylor lower bound is negative; subdivision must
        # establish endpoint separation away from the unique upper minimizer.
        optimum, strict = _selected(.01, .08, .14, rho)
        self.assertTrue(strict)
        self.assertEqual(optimum[0], "upper")
        value, cells = _growth(.01, .08, _i(.14), rho, optimum, 1e-11, lambda: None, 4095)
        self.assertGreater(value, 1e-11)
        self.assertGreater(cells, 1)
        def stop():
            raise TimeoutError("caller stopped this exact computation")
        with self.assertRaises(TimeoutError):
            scalar_dual_certificate(p, d, np.full(8, .125), rho, check=stop)


if __name__ == "__main__":
    unittest.main()
