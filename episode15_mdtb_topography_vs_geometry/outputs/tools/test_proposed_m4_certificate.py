"""Focused mathematical checks for an INACTIVE proposal; no empirical inputs.

Run only in a coordinator-owned Slurm allocation. These checks do not adopt
the proposal, fit any participant, or qualify an empirical scientific claim.
"""
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm_frechet

import proposed_m4_certificate as candidate
from paired_stretch_variants import M4CertificateUnverified, stretch_specifications


def _decimal_hessian(P, D, theta, s):
    """Independent matrix-power derivative recurrence, not radial formulas.

    At 70 decimal digits and 64 terms, the fixture arguments (norm below 1.2)
    have truncation/rounding far below double precision. Every input float is
    converted exactly, including the actual stored mode coefficient.
    """
    with localcontext() as context:
        context.prec = 70
        zero = lambda: [[Decimal(0), Decimal(0)], [Decimal(0), Decimal(0)]]

        def multiply(a, b):
            return [[sum((a[i][k]*b[k][j] for k in range(2)), Decimal(0))
                     for j in range(2)] for i in range(2)]

        def add(*matrices):
            return [[sum((a[i][j] for a in matrices), Decimal(0))
                     for j in range(2)] for i in range(2)]

        def scale(a, factor):
            return [[v*factor for v in row] for row in a]

        ds = Decimal.from_float(float(s))
        x, y = (Decimal.from_float(float(v)) for v in theta)
        C = [[ds*x, ds*y], [ds*y, -ds*x]]
        modes = [[[ds, Decimal(0)], [Decimal(0), -ds]],
                 [[Decimal(0), ds], [ds, Decimal(0)]]]
        answer = zero()
        for matrix, k, weight in ((P, 2, 1), (D, 1, -2)):
            Z = scale(C, Decimal(k))
            E = [scale(m, Decimal(k)) for m in modes]
            numeric = [[Decimal.from_float(float(v)) for v in row] for row in matrix]
            T = [[Decimal(1), Decimal(0)], [Decimal(0), Decimal(1)]]
            V, W = [zero(), zero()], [[zero(), zero()], [zero(), zero()]]
            total = [[zero(), zero()], [zero(), zero()]]
            for n in range(1, 65):
                factor = Decimal(1)/Decimal(n)
                next_W = [[scale(add(multiply(W[i][j], Z), multiply(V[i], E[j]),
                                     multiply(V[j], E[i])), factor)
                           for j in range(2)] for i in range(2)]
                next_V = [scale(add(multiply(V[i], Z), multiply(T, E[i])), factor)
                          for i in range(2)]
                T = scale(multiply(T, Z), factor)
                V, W = next_V, next_W
                total = [[add(total[i][j], W[i][j]) for j in range(2)] for i in range(2)]
            for i in range(2):
                for j in range(2):
                    trace = sum((numeric[a][b]*total[i][j][b][a]
                                 for a in range(2) for b in range(2)), Decimal(0))
                    answer[i][j] += Decimal(weight)*trace
        return answer


def _objective(specification, last_block_anisotropic=False):
    # Full-rank fixed 16-dimensional source; these are arithmetic fixtures.
    diagonal = np.ones(16)
    if last_block_anisotropic:
        diagonal[-2] = 0.1
    return SimpleNamespace(weights=np.full(16, 1/16),
        rotated_source=4*np.diag(diagonal), active_target=np.zeros((16, 16)),
        gain=1., modes=specification.modes)


class ProspectiveM4CertificateTests(unittest.TestCase):
    def test_shear_enclosures_against_independent_decimal_matrix_derivatives(self):
        specification = stretch_specifications()[3]
        s = float(specification.modes[0, 0, 0])
        radius = specification.spectral_norm_bound/s
        points = ((0., 0.), (1e-10, -2e-10), (radius, 0.),
                  (-radius, 0.), (radius/math.sqrt(2), -radius/math.sqrt(2)),
                  (radius, radius))  # Includes an outside-disk square center.
        base_p = np.array([[1.3, .2], [.2, .7]])
        base_d = np.array([[.8, -.1], [-.1, 1.1]])
        for magnitude in (1e-12, 1., 1e12):
            P, D = magnitude*base_p, magnitude*base_d
            for theta in points:
                expected = _decimal_hessian(P, D, theta, s)
                enclosed = candidate._shear_hessian(P, D, theta, s)
                for i in range(2):
                    for j in range(2):
                        self.assertLessEqual(Decimal.from_float(enclosed[i][j].lo), expected[i][j])
                        self.assertGreaterEqual(Decimal.from_float(enclosed[i][j].hi), expected[i][j])

    def test_shear_hessian_agrees_with_existing_frechet_gradient_arithmetic(self):
        specification = stretch_specifications()[2]
        modes = specification.modes[:2, :2, :2]
        s = float(modes[0, 0, 0])
        P = np.array([[1.3, .2], [.2, .7]])
        D = np.array([[.8, -.1], [-.1, 1.1]])
        theta = np.array([.11, -.17])

        def gradient(value):
            C = np.einsum("i,ijk->jk", value, modes)
            return np.array([np.trace(P@expm_frechet(2*C, 2*mode, compute_expm=False))
                             -2*np.trace(D@expm_frechet(C, mode, compute_expm=False))
                             for mode in modes])

        step = 1e-5
        finite_difference = np.column_stack([
            (gradient(theta+step*direction)-gradient(theta-step*direction))/(2*step)
            for direction in np.eye(2)])
        intervals = candidate._shear_hessian(P, D, theta, s)
        midpoints = np.array([[(x.lo+x.hi)/2 for x in row] for row in intervals])
        np.testing.assert_allclose(midpoints, finite_difference, rtol=2e-8, atol=2e-9)

    def test_complete_domain_bounds_include_endpoints_and_disk_boundary(self):
        specification = stretch_specifications()[3]
        rho = specification.spectral_norm_bound
        s = float(specification.modes[0, 0, 0])
        P, D = np.eye(2), np.zeros((2, 2))
        for kind in ("local_diagonal", "local_shear"):
            budget = candidate._Budget(candidate.CoverLimits(), lambda: None)
            result = candidate._cover_bound(P, D, rho, s, 1e-10*2, kind, budget)
            self.assertTrue(result["complete_domain_cover"])
            self.assertGreater(result["hessian_lower_bound"], 1e-10*2)
            if kind == "local_diagonal":
                for t in (-rho, 0., rho):
                    exact = 4*s*s*(math.exp(2*t)+math.exp(-2*t))
                    self.assertLessEqual(result["hessian_lower_bound"], exact)
            else:
                self.assertGreater(result["visited_cells"], 1)
                self.assertGreater(result["disjoint_cells"], 0)
                for angle in (0., math.pi/4, math.pi/2, math.pi, 3*math.pi/2):
                    theta = rho/s*np.array([math.cos(angle), math.sin(angle)])
                    reference = _decimal_hessian(P, D, theta, s)
                    eigenvalue = np.linalg.eigvalsh(np.array(reference, dtype=float))[0]
                    self.assertLessEqual(result["hessian_lower_bound"], eigenvalue)

    def test_unresolved_nonpositive_and_resource_caps_refuse(self):
        rho = stretch_specifications()[3].spectral_norm_bound
        s = float(stretch_specifications()[3].modes[0, 0, 0])
        P = np.eye(2)
        for D, limits in ((np.zeros((2, 2)), candidate.CoverLimits(max_cells_per_block=1)),
                          (10*np.eye(2), candidate.CoverLimits(max_depth=2))):
            budget = candidate._Budget(limits, lambda: None)
            with self.assertRaises(M4CertificateUnverified):
                candidate._cover_bound(P, D, rho, s, 1e-10, "local_shear", budget)
        with patch.object(candidate.time, "monotonic", side_effect=(0., 31.)):
            budget = candidate._Budget(candidate.CoverLimits(), lambda: None)
            with self.assertRaisesRegex(M4CertificateUnverified, "time cap"):
                budget.tick()
        specification = stretch_specifications()[0]
        objective = _objective(specification)
        objective.active_target = 10*objective.rotated_source
        with self.assertRaisesRegex(M4CertificateUnverified, "diagonal curvature is nonpositive"):
            candidate.proposed_curvature_certificate(objective, specification, check=lambda: None)

    def test_local_diagonal_interval_handles_signed_cross_moments_and_endpoints(self):
        specification = stretch_specifications()[1]
        rho = specification.spectral_norm_bound
        s = float(specification.modes[0, 0, 0])
        P = np.diag([.3, 2.])
        D = np.diag([-.7, 1.4])
        full = candidate._local_diagonal_interval(P, D, -rho, rho, s)
        for t in (-rho, -.07, 0., .13, rho):
            expected = _decimal_hessian(P, D, (t/s, 0.), s)[0][0]
            self.assertLessEqual(Decimal.from_float(full.lo), expected)
            self.assertGreaterEqual(Decimal.from_float(full.hi), expected)

    def test_original_pass_short_circuits_and_modes_cannot_be_changed(self):
        specification = stretch_specifications()[2]
        objective = _objective(specification)
        with patch.object(candidate, "_cover_bound", side_effect=AssertionError("must not subdivide")):
            result = candidate.proposed_curvature_certificate(objective, specification,
                check=lambda: None, limits=candidate.CoverLimits(max_cells_per_block=1))
        self.assertEqual(len(result["block_curvature"]), 8)
        self.assertTrue(all(not b["alternative_evaluated"] for b in result["block_curvature"]))
        self.assertFalse(result["production_adopted"])
        changed = specification.modes.copy()
        changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.inf)
        other = SimpleNamespace(name=specification.name, modes=changed,
                                spectral_norm_bound=specification.spectral_norm_bound)
        objective.modes = changed
        with self.assertRaisesRegex(ValueError, "canonical"):
            candidate.proposed_curvature_certificate(objective, other, check=lambda: None)

    def test_anisotropic_block_improvement_keeps_all_eight_margins(self):
        specification = stretch_specifications()[0]
        objective = _objective(specification, last_block_anisotropic=True)
        result = candidate.proposed_curvature_certificate(objective, specification, check=lambda: None)
        self.assertEqual(len(result["block_curvature"]), 8)
        for block in result["block_curvature"]:
            self.assertGreater(block["hessian_lower_bound"], block["required_positive_margin"])
            self.assertEqual(block["required_positive_margin"], 1e-10*block["objective_scale"])
        last = result["block_curvature"][-1]
        self.assertTrue(last["alternative_evaluated"])
        self.assertLessEqual(last["original_hessian_lower_bound"], last["required_positive_margin"])
        self.assertLessEqual(last["coefficient_gram_lower_bound"], 1.)
        self.assertFalse(result["interval_arithmetic_proof"])


if __name__ == "__main__":
    unittest.main()
