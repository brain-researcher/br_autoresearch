"""Two deterministic endpoint checks; no saved world or noise draw is read."""
import math
import unittest

import numpy as np

from analog_conditioning import ControlledVoltageDrives, SourceConditioning
from daq_mux_hold import ExternalDAQBranch, ExternalDAQBranchDrive, external_daq_branch_id
from physical_noise_ports import ExternalDAQCurrentPort, NoiseLimits, NodalCurrentPort, PhysicalNoiseLayout
from polynomial_noise import NoisePolynomial
from radau_polynomial_forcing import RadauPolynomialCircuitForcing


def fixture(*, duration=.5, check=lambda: None, external_conductance=.5):
    states = tuple(f"v{i}" for i in range(129))
    branch_id = external_daq_branch_id("adc", "external")
    layout = PhysicalNoiseLayout(state_ids=states, voltage_reference_id="common",
        nodal_current_ports=(NodalCurrentPort("edge", "v0", "v1", "conditional", "edge_01"),),
        contact_stars=(((0, 1), (.25, .5), 2.),) + (((0,), (0.,), 0.),) * 4095,
        contact_ownership="all_eliminated_star_branch_and_access_noise_owned_here",
        contact_noise_role="conditional", amplifier_input_noise_role="conditional",
        excluded_retained_contact_components=(), limits=NoiseLimits(20000, 2**20),
        external_daq_branches=(ExternalDAQBranch(branch_id, "adc", "external", "v128", .5, 0.),),
        external_daq_current_ports=(ExternalDAQCurrentPort("external_eta", branch_id,
            "adc", "external", "v128", "conditional"),))
    coefficients = np.zeros((4, layout.numeric_value_count))
    for name, values in {
        "edge": (.125, .25, .5, 1.),
        "contact:0000:star_branch:0": (.0625, .125, 0., .25),
        "contact:0000:star_branch:1": (-.03125, .0625, -.125, .5),
        "contact:0000:star_access_voltage": (.25, .5, 0., 0.),
        "amplifier_input_voltage:00": (.125, 0., 0., 0.),
        "external_eta": (.125, -.25, .5, -1.),
    }.items():
        coefficients[:, layout.port_ids.index(name)] = values
    polynomial = NoisePolynomial(layout, coefficients, 1.25, 1., 1.5)
    channel = SourceConditioning(gain=2., offset_volts=0., drive_lower_volts=-4., drive_upper_volts=4.,
        input_capacitance_farad=0., input_conductance_siemens=0., buffer_capacitance_farad=1.,
        buffer_conductance_siemens=.5, output_conductance_siemens=1., filter_capacitance_farad=1.,
        filter_leak_conductance_siemens=0., initial_buffer_volts=0., initial_filter_volts=0.)
    base = ControlledVoltageDrives(channels=(channel,) * 64,
        source_nodes=tuple(range(64)), buffer_nodes=tuple(range(64, 128)), state_count=129)
    field = np.zeros(4096)
    field[0] = .5
    context = (ExternalDAQBranchDrive("adc", "external", 128, external_conductance, .625, "common"),)
    forcing = RadauPolynomialCircuitForcing(polynomial, interval_start_seconds=1., duration_seconds=duration,
        base_controlled_drives=base, field_port_volts=field, external_branch_drives=context,
        max_stored_numeric_bytes=2**20, check=check)
    state = np.zeros(129)
    state[0], state[1], state[128] = .25, -.125, .375
    return forcing, polynomial, base, state


class RadauPolynomialForcingTests(unittest.TestCase):
    def test_exact_right_endpoint_preserves_original_currents_and_port_work(self):
        calls = []
        forcing, polynomial, base, state = fixture(check=lambda: calls.append(1))
        coefficients = polynomial.numeric_array.copy()
        calls.clear()
        current, controlled = forcing.evaluate_left(.5)
        rates = forcing.work_rates_left(.5, state)
        self.assertEqual(len(calls), 2)
        # Independent original-source equations at t-origin = 1/4.
        edge, eta0, eta1, access, external = 15/64, 25/256, -1/64, 3/8, 5/64
        h = np.array([.25, .5])
        eta = np.array([eta0, eta1])
        denominator = 1 + 2 * h.sum()
        expected_current = np.zeros(129)
        expected_current[:2] = eta - 2 * h * eta.sum() / denominator + h * access / denominator
        expected_current[0] += edge
        expected_current[1] -= edge
        expected_current[128] = external
        np.testing.assert_allclose(current, expected_current, rtol=2e-15, atol=2e-16)
        junction = (.5 + access + 2 * (h @ state[:2] - eta.sum())) / denominator
        access_current = h @ (junction - state[:2]) + eta.sum()
        expected_work = (edge * (state[0] - state[1]) + eta @ (state[:2] - junction)
                         + external * (state[128] - .625),
                         access * access_current,
                         .5 * (h.sum() * access + eta.sum()) / denominator,
                         .625 * external)
        np.testing.assert_allclose(rates, expected_work, rtol=2e-15, atol=2e-16)
        self.assertEqual(controlled.evaluate(state).current_amp[0], .375)
        self.assertEqual(base.evaluate(state).current_amp[0], .25)
        self.assertEqual(controlled.source_nodes, base.source_nodes)
        self.assertEqual(controlled.buffer_nodes, base.buffer_nodes)
        np.testing.assert_array_equal(polynomial.numeric_array, coefficients)
        self.assertIs(forcing._polynomial, polynomial)
        for time in (0., .125, .25):
            np.testing.assert_array_equal(forcing.evaluate_left(time)[0], forcing.evaluate(time)[0])
            self.assertEqual(forcing.work_rates_left(time, state), forcing.work_rates(time, state))

    def test_endpoint_permission_is_explicit_and_bounded_by_current_interval(self):
        forcing, polynomial, _, state = fixture()
        for call in (lambda: forcing.evaluate(.5), lambda: forcing.work_rates(.5, state),
                     lambda: polynomial.sample_at(1.5)):
            with self.assertRaises(ValueError):
                call()
        for invalid in (-.01, math.nextafter(.5, math.inf), 1., math.inf, math.nan, True, np.bool_(True)):
            with self.subTest(time=invalid):
                with self.assertRaises(ValueError):
                    forcing.evaluate_left(invalid)
                with self.assertRaises(ValueError):
                    forcing.work_rates_left(invalid, state)
        short, _, _, _ = fixture(duration=.25)
        np.testing.assert_array_equal(short.evaluate_left(.25)[0], forcing.evaluate(.25)[0])
        self.assertEqual(short.work_rates_left(.25, state), forcing.work_rates(.25, state))
        for call in (lambda: short.evaluate_left(math.nextafter(.25, math.inf)),
                     lambda: short.work_rates_left(math.nextafter(.25, math.inf), state),
                     lambda: fixture(external_conductance=0.),
                     lambda: forcing.work_rates_left(.5, np.full_like(state, math.nan))):
            with self.assertRaises(ValueError):
                call()
        failure = RuntimeError("selected budget stop")
        def stop():
            raise failure
        forcing._check = stop
        for call in (lambda: forcing.evaluate_left(.5), lambda: forcing.work_rates_left(.5, state)):
            with self.assertRaises(RuntimeError) as caught:
                call()
            self.assertIs(caught.exception, failure)


if __name__ == "__main__":
    unittest.main()
