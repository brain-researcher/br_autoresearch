"""Three source-only Radau fixtures; no production method is selected here."""
import math
import unittest

import numpy as np
from numpy.polynomial.polynomial import polyadd, polymul, polyval
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from analog_conditioning import ControlledVoltageDrives
from conditioned_array_solver import _Counter
from conditioned_radau_step import radau_iia_step
from sparse_array_rc import SolverConvergenceError
from test_conditioned_array_solver import channel, controls


def fixture(*, source_c=2., source_g=.7, gain=1., initial=(.1, -.1), selected_controls=None,
            check=lambda: None):
    size = 128
    selected = channel(gain=gain, offset_volts=0., buffer_conductance_siemens=2.)
    drives = ControlledVoltageDrives(channels=(selected,) * 64, source_nodes=tuple(range(64)),
        buffer_nodes=tuple(range(64, 128)), state_count=size)
    c = diags(np.repeat([source_c, 1.], 64), format="csc")
    diagonal = np.repeat([source_g, 2.], 64)
    factor = splu(c)
    selected_controls = selected_controls or controls(relative_tolerance=1e-10, voltage_atol=1e-12,
        charge_atol_coulomb=1e-13, linear_charge_atol_coulomb=1e-15)
    counter = _Counter(selected_controls, check)

    def product(value):
        counter.product()
        return diagonal * value

    args = dict(capacitance=c, factor_solve=factor.solve, passive_product=product,
        passive_diagonal=diagonal, controls=selected_controls, counter=counter, check=check,
        old=np.repeat(initial, 64), current=np.zeros(size), controlled_drives=drives)
    return args


def integral(coefficients, duration):
    return sum(float(value) * duration**(index + 1) / (index + 1)
               for index, value in enumerate(coefficients))


class ConditionedRadauStepTests(unittest.TestCase):
    def test_stiff_decay_matches_exponential_and_separates_numerical_dissipation(self):
        rate, amplitude = 2 * math.pi * 1e8, 1e-7
        duration = 800. / rate
        selected = controls(relative_tolerance=1e-5, voltage_atol=1e-8,
            charge_atol_coulomb=1e-20, linear_relative_tolerance=1e-9,
            linear_charge_atol_coulomb=1e-22, energy_atol_joule=1e-18)
        args = fixture(source_c=1e-12, source_g=rate * 1e-12, gain=0.,
                       initial=(amplitude, 0.), selected_controls=selected)
        before = args["old"].copy()
        result = radau_iia_step(**args, step_seconds=duration, stage_times=(duration / 3., duration))
        exact_end = amplitude * math.exp(-800.)
        exact_mean = amplitude * (-math.expm1(-800.)) / 800.
        np.testing.assert_allclose(result.endpoint[:64], exact_end, rtol=0., atol=selected.voltage_atol)
        np.testing.assert_allclose(result.mean[:64], exact_mean, rtol=0., atol=1e-12)
        np.testing.assert_array_equal(result.endpoint[64:], np.zeros(64))
        # Independent closed-form demonstration of the original midpoint
        # aliasing: two/four pieces agree within10nV but retain almost100nV.
        midpoint_two = amplitude * (199. / 201.)**2
        midpoint_four = amplitude * (99. / 101.)**4
        self.assertLess(abs(midpoint_two - midpoint_four), selected.voltage_atol)
        self.assertGreater(abs(midpoint_four - exact_end), 9 * selected.voltage_atol)
        c = args["capacitance"]
        energy_change = .5 * (result.endpoint @ (c @ result.endpoint) - before @ (c @ before))
        physical_heat = result.work_increments[3]
        self.assertGreater(result.numerical_dissipation_joule, 10 * physical_heat)
        self.assertAlmostEqual(energy_change + physical_heat + result.numerical_dissipation_joule,
                               0., delta=1e-30)
        self.assertLess(abs(energy_change + physical_heat), selected.energy_atol_joule)
        np.testing.assert_array_equal(args["old"], before)
        self.assertLessEqual(result.maximum_scaled_residual, 1.)

    def test_polynomial_state_means_and_all_physical_work_match_analytic_integrals(self):
        args = fixture()
        base = args["controlled_drives"]
        duration = .2
        # Prescribe exact quadratic source/buffer solutions, then derive their
        # physical inputs from C v'=I-Gv. Their integrals/powers below use only
        # polynomial algebra, never stage samples or returned solver states.
        source, buffer = np.array([.1, .2, .3]), np.array([-.1, .4, .2])
        target = np.array([.1, .6, .2])  # buffer + buffer'/2
        amplifier = target - source
        access = np.array([.03, .1])
        branch_g, field, remote = .2, .1, .4
        eta = np.array([.444, 1.32, .21])
        total = polyadd(eta, branch_g * access)
        args["current"][:64] = branch_g * field
        calls = []

        class Forcing:
            def values(self, point, left):
                if not 0 <= point <= duration or (not left and point == duration):
                    raise AssertionError("wrong side of physical endpoint")
                calls.append((left, point))
                extra = np.r_[np.full(64, polyval(point, total)), np.zeros(64)]
                return extra, base.with_input_noise_volts(np.full(64, polyval(point, amplifier)))

            def evaluate(self, point):
                return self.values(point, False)

            def evaluate_left(self, point):
                return self.values(point, True)

            def powers(self, point, state):
                current, voltage = polyval(point, eta), polyval(point, access)
                return (current * float(np.sum(state[:64] - remote)),
                    branch_g * voltage * float(np.sum(field + voltage - state[:64])),
                    64 * field * branch_g * voltage, 64 * remote * current)

            work_rates = powers
            work_rates_left = powers

        original = args["old"].copy()
        state, mean_integral, works, numerical = original.copy(), np.zeros(128), np.zeros(10), 0.
        for index in range(64):
            start, end = duration * index / 64., duration * (index + 1) / 64.
            step = end - start
            result = radau_iia_step(**{**args, "old": state}, step_seconds=step,
                stage_times=(start + step / 3., end), time_varying_forcing=Forcing())
            state = result.endpoint
            mean_integral += step * result.mean
            works += result.work_increments
            numerical += result.numerical_dissipation_joule
        expected_end = np.repeat([polyval(duration, source), polyval(duration, buffer)], 64)
        expected_mean = np.repeat([integral(source, duration) / duration,
                                   integral(buffer, duration) / duration], 64)
        np.testing.assert_allclose(state, expected_end, rtol=0., atol=2e-10)
        np.testing.assert_allclose(mean_integral / duration, expected_mean, rtol=0., atol=2e-10)
        difference = target - buffer
        current_drop = polyadd(source, [-remote])
        access_drop = polyadd(polyadd([field], access), -source)
        power_polynomials = [2 * polymul(target, difference), 2 * polymul(difference, difference),
            2 * polymul(target, buffer), polyadd(.7 * polymul(source, source), 2 * polymul(buffer, buffer)),
            2 * polymul(buffer, buffer), polymul(total, source), polymul(eta, current_drop),
            branch_g * polymul(access, access_drop), field * branch_g * access, remote * eta]
        expected_work = 64 * np.array([integral(item, duration) for item in power_polynomials])
        np.testing.assert_allclose(works, expected_work, rtol=2e-6, atol=2e-8)
        energy_change = .5 * (state @ (args["capacitance"] @ state)
                              - original @ (args["capacitance"] @ original))
        nodal_work = float(args["current"] @ mean_integral) + works[2] + works[5]
        self.assertAlmostEqual(energy_change - nodal_work + works[3] + numerical, 0., delta=2e-10)
        self.assertGreater(numerical, 0.)
        self.assertEqual(len(calls), 128)
        self.assertEqual(calls[-1], (True, duration))
        self.assertTrue(all(point < duration for left, point in calls if not left))

    def test_resource_and_late_forcing_refusals_return_no_step_and_preserve_input(self):
        for changes, message in ((dict(max_workspace_bytes=1), "2N numeric-workspace"),
                                  (dict(max_operator_products=1), "operator-product"),
                                  (dict(max_newton_iterations=1), "Newton limit")):
            args = fixture(selected_controls=controls(**changes))
            before = args["old"].copy()
            with self.subTest(changes=changes), self.assertRaisesRegex(SolverConvergenceError, message):
                radau_iia_step(**args, step_seconds=.1, stage_times=(.1 / 3., .1))
            np.testing.assert_array_equal(args["old"], before)
        calls = []
        stopped = RuntimeError("fixture deadline")

        def check():
            calls.append(None)
            if len(calls) == 8:
                raise stopped

        args = fixture(check=check)
        before = args["old"].copy()
        with self.assertRaises(RuntimeError) as raised:
            radau_iia_step(**args, step_seconds=.1, stage_times=(.1 / 3., .1))
        self.assertIs(raised.exception, stopped)
        self.assertGreater(args["counter"].newton, 0)
        np.testing.assert_array_equal(args["old"], before)

        args = fixture()

        class BadEndpointWork:
            def evaluate(self, point):
                return np.zeros(128), args["controlled_drives"]

            evaluate_left = evaluate

            def work_rates(self, point, state):
                return (0., 0., 0., 0.)

            def work_rates_left(self, point, state):
                return (0., 0., np.nan, 0.)

        before = args["old"].copy()
        with self.assertRaisesRegex(ValueError, "physical work rates"):
            radau_iia_step(**args, step_seconds=.1, stage_times=(.1 / 3., .1),
                          time_varying_forcing=BadEndpointWork())
        np.testing.assert_array_equal(args["old"], before)


if __name__ == "__main__":
    unittest.main()
