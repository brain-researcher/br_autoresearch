"""Numerical identities required before the DDM primitives enter a fit."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import deadline_ddm as ddm
import collapsing_ddm
from qualify_assets import SourceError, normalize_trial


class FirstPassageTests(unittest.TestCase):
    def test_response_plus_survival_is_one(self):
        for a, z, drift, deadline in [(1., .5, 0., 4.), (2., .3, -1., 4.),
                                       (2., .7, 1.3, .1), (.7, .2, 3., .4)]:
            total = sum(quad(lambda t: float(ddm.response_density(t, drift, a, z, c)),
                             0., deadline, epsabs=1e-10)[0] for c in [0, 1])
            self.assertAlmostEqual(total + float(ddm.survival(deadline, drift, a, z)), 1., places=8)

    def test_reflection_and_unbiased_symmetry(self):
        times = np.array([.001, .01, .1, .1499, .1501, .5, 2.])
        np.testing.assert_allclose(ddm.response_density(times, 1., 1., .3, 1),
                                   ddm.response_density(times, -1., 1., .7, 0))
        np.testing.assert_allclose(ddm.response_density(times, 0., 1., .5, 1),
                                   ddm.response_density(times, 0., 1., .5, 0))

    def test_zero_drift_eventual_acceptance_is_start_fraction(self):
        for z in [.2, .5, .8]:
            mass = quad(lambda t: float(ddm.response_density(t, 0., 1., z, 1)),
                        0., 10., epsabs=1e-10)[0]
            self.assertAlmostEqual(mass, z, places=8)

    def test_survival_derivative_is_negative_total_flux(self):
        for t in [.01, .149, .151, 1.]:
            eps = 1e-6
            deriv = (ddm.survival(t+eps, .4, 1., .3)-ddm.survival(t-eps, .4, 1., .3))/(2*eps)
            flux = sum(ddm.response_density(t, .4, 1., .3, c) for c in [0, 1])
            self.assertAlmostEqual(float(deriv), -float(flux), places=6)

    def test_missing_is_survival_not_rejection(self):
        actual = ddm.trial_log_likelihood([np.nan, .1], [-1, 1], [.5, .5],
                                         1., .5, .2, .02, .1)
        expected = .98 * float(ddm.survival(3.8, .5, 1., .5)) + .02*.1
        self.assertAlmostEqual(actual[0], np.log(expected))
        self.assertAlmostEqual(actual[1], np.log(.02*.9/8))

    def test_dollar_anchor_and_nesting(self):
        self.assertEqual(float(ddm.drift_rate(10, 10, 0, gamma_gain=.5, gamma_loss=1.5)), 0.)
        self.assertEqual(float(ddm.drift_rate(20, 10, 0)), 1.)

    def test_cdf_conserves_deadline_mass_and_matches_density(self):
        for t in [.01, .149, .151, 1.]:
            lo = float(ddm.lower_cdf(t, .7, 1., .3))
            up = float(ddm.lower_cdf(t, -.7, 1., .7))
            self.assertAlmostEqual(lo+up+float(ddm.survival(t, .7, 1., .3)), 1., places=9)
            eps = 1e-6
            deriv = (ddm.lower_cdf(t+eps, .7, 1., .3)-ddm.lower_cdf(t-eps, .7, 1., .3))/(2*eps)
            self.assertAlmostEqual(float(deriv), float(ddm.lower_density(t, .7, 1., .3)), places=6)

    def test_simulation_has_the_declared_deadline_distribution(self):
        rng = np.random.default_rng(27003)
        choices, rt = ddm.simulate(rng, np.zeros(20000), 3., .5, .2, 0., 0.)
        p_miss = float(ddm.survival(3.8, 0., 3., .5))
        self.assertLess(abs(np.mean(choices == -1)-p_miss), .012)
        self.assertLess(abs(np.mean(choices == 1)-(1-p_miss)/2), .015)
        self.assertTrue(np.isnan(rt[choices == -1]).all())
        self.assertTrue(((rt[choices != -1] > .2) & (rt[choices != -1] <= 4.)).all())


class OriginalEventsTests(unittest.TestCase):
    mapping = {"fields": {"gain": "gain", "loss": "loss", "onset": "onset",
                           "response": "response", "response_time": "rt"},
               "loss_multiplier": -1, "response_categories": {"1": "strong_accept", "n/a": "no_response"},
               "no_response_rt_values": [0]}

    def test_miss_survives_and_loss_convention_is_explicit(self):
        event = normalize_trial({"gain": "10", "loss": "-5", "onset": "2",
                                 "response": "n/a", "rt": "0"}, self.mapping, "sub-001", "EI", 1, 1)
        self.assertEqual((event["choice"], event["no_response"], event["response_time"], event["loss"]), (-1, 1, "", 5))

    def test_unknown_response_is_not_guessed(self):
        with self.assertRaises(SourceError):
            normalize_trial({"gain": "10", "loss": "-5", "onset": "2",
                             "response": "2", "rt": ".5"}, self.mapping, "sub-001", "EI", 1, 1)


class CollapsingBoundsTests(unittest.TestCase):
    def test_conservation_symmetry_and_zero_urgency_limit(self):
        out = collapsing_ddm.solve(0., 1., .5, 0., .5, dt=.0005, intervals=200)
        self.assertAlmostEqual(out.reject_mass.sum()+out.accept_mass.sum()+out.survival_mass, 1., places=9)
        np.testing.assert_allclose(out.reject_mass, out.accept_mass, atol=1e-12)
        self.assertLess(abs(out.survival_mass-float(ddm.survival(.5, 0., 1., .5))), .001)

    def test_registered_symmetric_collapse_accelerates_absorption(self):
        fixed = collapsing_ddm.solve(0., 2., .5, 0., 1., dt=.002, intervals=100)
        collapsing = collapsing_ddm.solve(0., 2., .5, .5, 1., dt=.002, intervals=100)
        self.assertLess(collapsing.survival_mass, fixed.survival_mass)
        np.testing.assert_allclose(collapsing.reject_mass, collapsing.accept_mass, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
