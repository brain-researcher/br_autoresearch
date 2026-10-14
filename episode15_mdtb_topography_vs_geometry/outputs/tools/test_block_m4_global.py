"""Focused analytical checks; execute only in coordinator-owned compute.

Polynomial candidates are independent test oracles, never production proofs.
These inputs are mathematical fixtures and contain no participant data.
"""
import math
import unittest

import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import expm

from block_m4_global import BlockLoss, certify_block
from paired_stretch_variants import CURVATURE_MARGIN, GAP_RTOL, M4CertificateUnverified
from proposed_m4_certificate import _i


def diagonal_candidates(P, D, rho):
    # F'(t)/2 times exp(2t): P00 z^4-D00 z^3+D11 z-P11.
    roots = np.roots([P[0, 0], -D[0, 0], 0., D[1, 1], -P[1, 1]])
    interior = [math.log(root.real) for root in roots
                if abs(root.imag) < 1e-9 and math.exp(-rho) < root.real < math.exp(rho)]
    return [-rho, rho, *interior]


def matrix_value(P, D, x):
    C = np.array([[x[0], x[1]], [x[1], -x[0]]])
    return float(np.trace(P@expm(2*C))-2*np.trace(D@expm(C)))


def shear_candidates(P, D, rho):
    a, b = np.trace(P), np.trace(D)
    p, d = np.array([P[0, 0]-P[1, 1], 2*P[0, 1]]), np.array([D[0, 0]-D[1, 1], 2*D[0, 1]])
    z = Polynomial([0., 1.])
    W = (z*z-1)*sum(((pi*z-di)**2 for pi, di in zip(p, d)), Polynomial([0.]))
    stationary = (4*a*z-2*b)**2*W-W.deriv()**2
    radii = [0., rho]
    for root in stationary.roots():
        if abs(root.imag) > 1e-8 or not 1 < root.real < math.cosh(rho):
            continue
        zz = float(root.real)
        if W(zz) > 0 and abs(4*a*zz-2*b-W.deriv()(zz)/math.sqrt(W(zz))) < 1e-6:
            radii.append(math.acosh(zz))
    # A norm cusp can be a candidate even though the stationary equation was
    # derived only at W>0. Include any common real zero of both components.
    for pi, di in zip(p, d):
        if pi != 0 and 1 < di/pi < math.cosh(rho) and np.linalg.norm(p*(di/pi)-d) < 1e-12:
            radii.append(math.acosh(di/pi))
    points = []
    for radius in radii:
        v = p*math.cosh(radius)-d
        points.append(np.zeros(2) if radius == 0 else
                      -radius*v/np.linalg.norm(v) if np.linalg.norm(v) else np.array([radius, 0.]))
    return points


class BlockGlobalTests(unittest.TestCase):
    def assert_enclosed(self, result, expected):
        self.assertTrue(result["all_exterior_intervals_excluded"])
        self.assertFalse(result["full_domain_positive_curvature_asserted"])
        self.assertLessEqual(result["lower_bound"], expected+2e-12)
        self.assertGreaterEqual(result["feasible_upper"], expected-2e-12)
        self.assertLessEqual(result["normalized_gap"], GAP_RTOL)
        self.assertGreater(result["coefficient_growth_lower_bound"], CURVATURE_MARGIN)

    def test_diagonal_all_stationary_candidates_and_endpoints(self):
        rho, scale = math.log(1.5), 1/math.sqrt(2)
        # The first problem has THREE interior stationary points (two wells),
        # and a strict unique winner; the second has a constrained endpoint
        # winner despite potentially negative curvature elsewhere.
        for P, D, multiple in ((np.eye(2), np.diag([2.0101, 2.0099]), True),
                              (np.eye(2), np.diag([3., 1.]), False)):
            points = diagonal_candidates(P, D, rho)
            if multiple:
                self.assertGreaterEqual(len(points), 5)
            values = [matrix_value(P, D, [point, 0.]) for point in points]
            result = certify_block(P, D, rho, scale, "local_diagonal",
                                   check=lambda: None, seconds=30., max_cells=4096)
            self.assert_enclosed(result, min(values))
            x = result["coordinate_intervals"][0]
            actual = (x.lo+x.hi)/2
            self.assertAlmostEqual(matrix_value(P, D, [actual, 0.]), min(values), places=10)
            self.assertLess(BlockLoss(P, D, rho, scale, "local_diagonal").actual_value_gap([actual])[1], GAP_RTOL)

    def test_shear_radial_candidates_full_coordinates_and_scale(self):
        rho, scale = math.log(1.5), 1/math.sqrt(2)
        P = np.array([[1.3, .2], [.2, .7]])
        for D, interior in ((np.array([[.8, -.1], [-.1, 1.1]]), False),
                            (np.array([[1.15, .13], [.13, .85]]), True)):
            points = shear_candidates(P, D, rho)
            expected = min(matrix_value(P, D, point) for point in points)
            result = certify_block(P, D, rho, scale, "local_shear",
                                   check=lambda: None, seconds=30., max_cells=4096)
            self.assert_enclosed(result, expected)
            x = np.array([(item.lo+item.hi)/2 for item in result["coordinate_intervals"]])
            self.assertLessEqual(np.linalg.norm(x), rho+1e-12)
            if interior:
                self.assertEqual(result["local_method"], "strict_interior_radial_growth")
                self.assertLess(np.linalg.norm(x), rho-1e-3)
            self.assertAlmostEqual(matrix_value(P, D, x), expected, places=10)
            loss = BlockLoss(P, D, rho, scale, "local_shear")
            self.assertLess(loss.actual_value_gap(x)[1], GAP_RTOL)
            # Fixed off-optimum points use independent matrix exponentials.
            # Growth is for theta=x/scale, including angular change.
            for point in (np.zeros(2), np.array([rho, 0.]), np.array([0., -rho])):
                lower = .5*result["coefficient_growth_lower_bound"]*np.sum(((point-x)/scale)**2)
                self.assertGreaterEqual(matrix_value(P, D, point)-expected+1e-11, lower)

    def test_origin_is_exact_and_positive_radius_angular_tie_refuses(self):
        rho, scale = math.log(1.25), 1/math.sqrt(2)
        result = certify_block(np.eye(2), np.eye(2), rho, scale, "local_shear",
                               check=lambda: None, seconds=15., max_cells=128)
        self.assert_enclosed(result, -2.)
        self.assertEqual(result["local_method"], "strict_origin_growth")
        self.assertTrue(all(item.lo == item.hi == 0 for item in result["coordinate_intervals"]))
        # Large equal D makes the optimum a positive-radius circle. It is
        # never admissible as a unique fit, even though the radial minimum is
        # strict at the disk boundary.
        with self.assertRaisesRegex(M4CertificateUnverified, "local radial/angular"):
            certify_block(np.eye(2), 3*np.eye(2), rho, scale, "local_shear",
                          check=lambda: None, seconds=15., max_cells=128)
        # Rounded subtractions collide, but exact binary moment differences do
        # not. That case must not obtain the special zero-gradient origin proof.
        P, D = np.diag([1., 2.**-55]), np.diag([1., 2.**-54])
        self.assertEqual(P[0, 0]-P[1, 1], D[0, 0]-D[1, 1])
        self.assertFalse(BlockLoss(P, D, rho, scale, "local_shear").origin_stationary)


if __name__ == "__main__":
    unittest.main()
