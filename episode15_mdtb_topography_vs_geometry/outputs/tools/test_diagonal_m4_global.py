"""Action-changing analytical cases for coupled profiles and balanced vertices."""
import math
import unittest

import numpy as np

from diagonal_m4_global import (
    SearchBudget, SearchStop, _balanced_vertex_certificate, _constrained_vertex_certificate,
    _diagonal_first_order_gap, _exterior_slabs, _local_balanced_vertex_region,
    _uniform_branches, moment_global_search, restricted_dual_bound,
)
from paired_stretch_variants import CURVATURE_MARGIN, GAP_RTOL, M4CertificateUnverified, stretch_specifications
from proposed_m4_certificate import _I


class GlobalProfileTests(unittest.TestCase):
    def test_constrained_vertex_with_scalar_dual_gap_has_complete_global_proof(self):
        rho = math.log(1.25)
        # The first concave scalar prefers +rho; its paired convex coordinate
        # prefers -rho under the trace constraint. At that endpoint the pair
        # has strict inward slope .00875. Yet no scalar-dual multiplier makes
        # both endpoints global scalar minima: the first requires lambda below
        # about .14066 and the second requires lambda >= .16. Other coordinates
        # have strong endpoint preferences. Full exterior exclusion is needed.
        p = np.r_[.01, 2., np.ones(14)]
        d = np.r_[.08, 1.7, np.full(7, 1.4), np.zeros(7)]
        vertex = np.r_[rho, -rho, np.full(7, rho), np.full(7, -rho)]
        H = np.diagonal(stretch_specifications()[0].modes, axis1=1, axis2=2).T
        exact = float(np.sum(p*np.exp(2*vertex)-2*d*np.exp(vertex)))
        whole = restricted_dual_bound(p, d, np.full(16, -rho), np.full(16, rho), check=lambda: None)
        self.assertLess(whole["lower_bound"], exact-1e-6)
        result = moment_global_search(p, d, rho, H, check=lambda: None,
                                      seconds=60., max_cells=1024)
        self.assertEqual(result["certificate_kind"], "balanced_constrained_vertex_with_global_exclusion")
        self.assertEqual(result["primal_free_coordinate_count"], 0)
        self.assertTrue(result["unique_global_minimum_sufficient_constrained_vertex"])
        self.assertFalse(result["unique_global_minimum_sufficient_dual"])
        self.assertFalse(result["unique_global_minimum_sufficient_profile"])
        self.assertTrue(result["all_other_regions_excluded"])
        self.assertGreater(result["initial_exterior_slabs"], 0)
        self.assertEqual(result["remaining_cells"], 0)
        self.assertLessEqual(result["global_lower_bound"], exact+2e-12)
        self.assertGreaterEqual(result["feasible_upper_bound"], exact-2e-12)
        self.assertLessEqual(result["global_gap_upper_bound"], GAP_RTOL)
        growth = result["global_coordinate_growth_lower_bound"]
        self.assertGreater(growth, CURVATURE_MARGIN)
        for interval, expected in zip(result["coordinate_intervals"], vertex):
            self.assertEqual((interval.lo, interval.hi), (expected, expected))
        for fraction in (.001, .1, 1., 2.):
            point = (1-fraction)*vertex
            difference = float(np.sum(p*np.exp(2*point)-2*d*np.exp(point)))-exact
            self.assertGreaterEqual(difference+1e-12, growth*np.sum((point-vertex)**2)/2)

    def test_local_strict_vertex_does_not_certify_tied_or_better_exterior(self):
        rho, p = math.log(1.25), np.full(16, .01)
        vertex = np.r_[np.full(8, -rho), np.full(8, rho)]
        for d in (np.full(16, .08), np.r_[np.full(8, .081), np.full(8, .08)]):
            with self.subTest(tied=bool(d[0] == d[-1])):
                # Concavity makes every balanced vertex a strict local minimum;
                # an equal or better opposite vertex still prevents the global
                # uniqueness certificate. Exhausted exterior work must refuse.
                local = _local_balanced_vertex_region(p, d, rho, vertex,
                    SearchBudget(lambda: None, 15., 64))
                self.assertGreater(local["full_growth"], CURVATURE_MARGIN)
                own = float(np.sum(p*np.exp(2*vertex)-2*d*np.exp(vertex)))
                other = float(np.sum(p*np.exp(-2*vertex)-2*d*np.exp(-vertex)))
                self.assertLessEqual(other, own+1e-14)
                with self.assertRaises(SearchStop):
                    _constrained_vertex_certificate(p, d, rho, vertex,
                        SearchBudget(lambda: None, 15., 32))

    def test_balanced_vertex_is_certified_without_free_profile_coordinates(self):
        rho = math.log(1.25)
        p, d = np.full(16, .01), np.r_[np.full(8, .08), np.full(8, -.08)]
        # The first eight scalars are strictly decreasing and concave on the
        # full box; the other eight are strictly increasing. Their unique
        # independent endpoint minima already have trace zero. No smooth
        # conditional rest coordinate exists at this constrained optimizer.
        self.assertLess(2*.01*math.exp(rho)-.08, 0)
        vertex = np.r_[np.full(8, rho), np.full(8, -rho)]
        H = np.diagonal(stretch_specifications()[0].modes, axis1=1, axis2=2).T
        result = moment_global_search(p, d, rho, H, check=lambda: None,
                                      seconds=30., max_cells=256)
        self.assertEqual(result["certificate_kind"], "balanced_strict_scalar_dual_vertex")
        self.assertEqual(result["primal_free_coordinate_count"], 0)
        self.assertTrue(result["unique_global_minimum_sufficient_dual"])
        self.assertFalse(result["unique_global_minimum_sufficient_profile"])
        self.assertFalse(result["full_domain_positive_curvature_asserted"])
        self.assertEqual(result["exact_balanced_endpoint_count"], 8)
        for interval, expected in zip(result["coordinate_intervals"], vertex):
            self.assertEqual((interval.lo, interval.hi), (expected, expected))
        def value(point):
            z = np.exp(point)
            return float(np.sum(p*z*z-2*d*z))
        exact = value(vertex)
        self.assertLessEqual(result["global_lower_bound"], exact+2e-12)
        self.assertGreaterEqual(result["feasible_upper_bound"], exact-2e-12)
        self.assertLessEqual(result["global_gap_upper_bound"], GAP_RTOL)
        growth = result["global_coordinate_growth_lower_bound"]
        self.assertGreater(growth, CURVATURE_MARGIN)
        for point in (np.zeros(16), -vertex, vertex/2):
            self.assertGreaterEqual(value(point)-exact+1e-12,
                                    .5*growth*float(np.sum((point-vertex)**2)))

    def test_vertex_route_refuses_genuine_ties_and_unbalanced_endpoints(self):
        rho, p, d = math.log(1.25), np.full(16, .01), np.full(16, .08)
        # Every scalar is strictly concave. This multiplier gives equal values
        # at its two distinct endpoint minima, so many balanced vertices tie.
        lower = .01*math.exp(-2*rho)-.16*math.exp(-rho)
        upper = .01*math.exp(2*rho)-.16*math.exp(rho)
        tied_lambda = (lower-upper)/(2*rho)
        with self.assertRaisesRegex(M4CertificateUnverified, "strict scalar global endpoint"):
            _balanced_vertex_certificate(p, d, rho, {"lambda": tied_lambda},
                SearchBudget(lambda: None, 15., 128))
        # At lambda=0 all sixteen upper endpoints are strict minima, but that
        # unconstrained optimizer is not trace zero and cannot be accepted.
        with self.assertRaisesRegex(M4CertificateUnverified, "balanced trace-zero"):
            _balanced_vertex_certificate(p, d, rho, {"lambda": 0.},
                SearchBudget(lambda: None, 15., 128))

    def test_vertex_route_does_not_waive_growth_margin_for_small_dual_gap(self):
        rho = math.log(1.25)
        p, d = np.full(16, .01e-12), np.r_[np.full(8, .08e-12), np.full(8, -.08e-12)]
        # The balanced vertex is still uniquely optimal, but scaling all
        # objective differences down makes the retained growth requirement
        # impossible. A tiny primal/dual value gap must not bypass it.
        with self.assertRaises(M4CertificateUnverified):
            _balanced_vertex_certificate(p, d, rho, {"lambda": 0.},
                SearchBudget(lambda: None, 15., 128))

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
