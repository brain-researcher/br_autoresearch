"""One analytical action-changing case: a unique primal optimum with dual jump."""
import math
import unittest

import numpy as np

from diagonal_m4_global import _diagonal_first_order_gap, _exterior_slabs, _uniform_branches, moment_global_search, restricted_dual_bound
from paired_stretch_variants import CURVATURE_MARGIN, GAP_RTOL, M4CertificateUnverified, stretch_specifications
from proposed_m4_certificate import _I


class GlobalProfileTests(unittest.TestCase):
    def test_linear_gap_bounds_stationary_and_known_nonstationary_cases(self):
        rho = math.log(1.25)
        p, c = np.ones(16), np.zeros(16)
        stationary = _diagonal_first_order_gap(p, p, c, rho)
        self.assertGreaterEqual(stationary, 0.)
        self.assertLess(stationary, GAP_RTOL)
        # At c=0 these coefficients give eight +1 and eight -1 derivatives.
        # The exact trace-zero-box linear minimum is -16*rho, so the gap is
        # 16*rho. The upper bound must not hide this nonstationary point.
        d = np.r_[np.full(8, .5), np.full(8, 1.5)]
        gap = _diagonal_first_order_gap(p, d, c, rho)
        self.assertGreaterEqual(gap, 16*rho)
        self.assertLess(gap-16*rho, 1e-10)

    def test_uniform_convex_minimum_keeps_shared_multiplier_dependence(self):
        # f(c)=exp(2c)-2exp(c) is strictly convex on this interval. For every
        # lambda in [-.001,.001], its unique minimum is the positive-curvature
        # root log((1+sqrt(1-2*lambda))/2), strictly inside the box. Comparing
        # independently enclosed function values loses the shared lambda and
        # needlessly refuses this known unique branch.
        lower, upper = np.array([-.01]), np.array([.01])
        args = (np.array([1.]), np.array([1.]), .01, _I(-.001, .001))
        rows = _uniform_branches(*args, ("interior",), lower, upper)
        for lam in (-.001, 0., .001):
            exact = math.log((1+math.sqrt(1-2*lam))/2)
            self.assertLessEqual(rows[0][1].lo, exact)
            self.assertGreaterEqual(rows[0][1].hi, exact)
        with self.assertRaises(M4CertificateUnverified):
            _uniform_branches(*args, ("lower",), lower, upper)

    def test_exterior_slabs_use_every_final_coordinate_bound(self):
        lower, upper = np.array([-2., -3.]), np.array([2., 3.])
        inner_lower, inner_upper = np.array([-.5, -1.]), np.array([.75, 1.25])
        slabs = tuple(_exterior_slabs(lower, upper, inner_lower, inner_upper))
        expected = (
            ([-2., -3.], [-.5, 3.]), ([.75, -3.], [2., 3.]),
            ([-.5, -3.], [.75, -1.]), ([-.5, 1.25], [.75, 3.]),
        )
        self.assertEqual(len(slabs), len(expected))
        for (a, b), (ea, eb) in zip(slabs, expected):
            np.testing.assert_array_equal(a, ea)
            np.testing.assert_array_equal(b, eb)
        # Endpoints and adjoining regions are covered after BOTH coordinates
        # shrink. The open inner box remains reserved for its local proof.
        for x in (-2., -.5, 0., .75, 2.):
            for y in (-3., -1., 0., 1.25, 3.):
                point = np.array([x, y])
                exterior = any(np.all(a <= point) and np.all(point <= b) for a, b in slabs)
                inside = bool(np.all(inner_lower < point) and np.all(point < inner_upper))
                self.assertEqual(exterior, not inside)

    def test_unique_coupled_minimum_despite_one_concave_scalar_dual_jump(self):
        rho = math.log(1.25)
        # First coordinate is concave on the entire box. The other 15 are
        # identical and strongly convex, hence their constrained minimum at
        # fixed t is exactly c_i=-t/15. The resulting profile has q'(0)=0 and
        # q'' >= -.1375 + (8*exp(-2*rho/15)-4.14*exp(rho/15))/15 > 0.
        # Therefore c=0 is the UNIQUE global primal optimum, although its first
        # scalar Lagrangian prefers an endpoint and the unrestricted dual jumps.
        p, d = np.full(16, 2.), np.full(16, 2.07)
        p[0], d[0] = .01, .08
        self.assertGreater(-.1375+(8*math.exp(-2*rho/15)-4.14*math.exp(rho/15))/15, 0)
        H = np.diagonal(stretch_specifications()[0].modes, axis1=1, axis2=2).T
        exact = float(np.sum(p-2*d))
        whole = restricted_dual_bound(p, d, np.full(16, -rho), np.full(16, rho), check=lambda: None)
        self.assertLess(whole["lower_bound"], exact-1e-5)
        result = moment_global_search(p, d, rho, H, check=lambda: None, seconds=120., max_cells=4096)
        self.assertTrue(result["all_other_regions_excluded"])
        self.assertTrue(result["unique_global_minimum_sufficient_profile"])
        self.assertEqual(result["profile_axis_rule"], "lowest_curvature_primal_free")
        self.assertTrue(result["profile_axis_is_primal_free"])
        self.assertEqual(result["primal_free_nonpositive_curvature_count"], 1)
        self.assertLessEqual(result["global_lower_bound"], exact+2e-12)
        self.assertGreaterEqual(result["feasible_upper_bound"], exact-2e-12)
        self.assertLessEqual(result["global_gap_upper_bound"], GAP_RTOL)
        self.assertGreater(result["global_coordinate_growth_lower_bound"], CURVATURE_MARGIN)
        for interval in result["coordinate_intervals"]:
            self.assertLess(abs((interval.lo+interval.hi)/2), 1e-7)


if __name__ == "__main__":
    unittest.main()
