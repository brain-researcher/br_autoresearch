"""Analytic small-network identities, not NeuroCam fitted/simulated candidates."""

import math
import unittest

import numpy as np

from coupled_rc import PassiveRCNetwork
from switched_line import SourceLinePole


class CoupledRCTests(unittest.TestCase):
    def network(self, initial=(1.0, 0.0), capacitance=((1.0, 0.0), (0.0, 1.0))):
        return PassiveRCNetwork(state_ids=("read", "off_gate"), initial_volts=initial,
                                capacitance_farad=capacitance)

    def advance(self, network, **changes):
        arguments = dict(
            duration_seconds=0.7, conductance_siemens=[[2.0, -1.0], [-1.0, 2.0]],
            input_conductance_siemens=[[0.0], [0.0]], input_volts=[0.0],
            observation_matrix=[[1.0, 0.0]], feedthrough_matrix=[[0.0]],
        )
        arguments.update(changes)
        return network.advance(**arguments)

    def test_scalar_limit_matches_existing_r1_and_input_work(self):
        scalar = SourceLinePole(initial_volts=0.3, capacitance_farad=2.0)
        expected = scalar.advance(duration_seconds=0.8, effective_input_volts=1.7,
                                  gain=1.4, tau_seconds=0.5, switch_charge_coulomb=0.0)
        network = PassiveRCNetwork(state_ids=("source",), initial_volts=[0.3], capacitance_farad=[[2.0]])
        result = network.advance(duration_seconds=0.8, conductance_siemens=[[4.0]],
                                 input_conductance_siemens=[[5.6]], input_volts=[1.7],
                                 observation_matrix=[[2.0]], feedthrough_matrix=[[-0.25]])
        self.assertAlmostEqual(result.end_volts[0], expected.end_volts)
        self.assertAlmostEqual(result.mean_volts[0], expected.mean_volts)
        self.assertAlmostEqual(result.output_mean_volts[0], 2 * expected.mean_volts - 0.25 * 1.7)
        self.assertAlmostEqual(result.output_end_volts[0], 2 * expected.end_volts - 0.25 * 1.7)
        self.assertAlmostEqual(result.input_work_joule, 0.8 * 5.6 * 1.7 * expected.mean_volts)
        self.assertGreater(result.input_work_joule, result.end_energy_joule - result.start_energy_joule)
        # Scaling C/G/B together preserves voltage trajectories but scales
        # charge, stored energy and drive work by the same physical factor.
        scale = 1e-12
        scaled_network = PassiveRCNetwork(state_ids=("source",), initial_volts=[0.3],
                                          capacitance_farad=[[2.0 * scale]])
        scaled = scaled_network.advance(duration_seconds=0.8, conductance_siemens=[[4.0 * scale]],
                                        input_conductance_siemens=[[5.6 * scale]], input_volts=[1.7],
                                        observation_matrix=[[2.0]], feedthrough_matrix=[[-0.25]])
        np.testing.assert_allclose(scaled.end_volts, result.end_volts, rtol=1e-13)
        np.testing.assert_allclose(scaled.mean_volts, result.mean_volts, rtol=1e-13)
        np.testing.assert_allclose(np.array(scaled.end_charge_coulomb) / scale,
                                   result.end_charge_coulomb, rtol=1e-13)
        self.assertAlmostEqual(scaled.end_energy_joule / scale, result.end_energy_joule)
        self.assertAlmostEqual(scaled.input_work_joule / scale, result.input_work_joule)

    def test_known_coupled_eigenmodes_and_passive_energy_decay(self):
        # [1, 0] = .5*[1, 1] + .5*[1, -1], with decay rates 1 and 3.
        result = self.advance(self.network())
        slow, fast = math.exp(-0.7), math.exp(-2.1)
        expected = [0.5 * (slow + fast), 0.5 * (slow - fast)]
        mean_slow, mean_fast = -math.expm1(-0.7) / 0.7, -math.expm1(-2.1) / 2.1
        np.testing.assert_allclose(result.end_volts, expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(result.mean_volts, [0.5 * (mean_slow + mean_fast),
                                                      0.5 * (mean_slow - mean_fast)], rtol=1e-13)
        self.assertAlmostEqual(result.end_energy_joule, 0.25 * (slow**2 + fast**2))
        self.assertLess(result.end_energy_joule, result.start_energy_joule)
        self.assertEqual(result.input_work_joule, 0.0)

    def test_singular_conductance_preserves_charge_and_equilibrates(self):
        # A resistor joining two equal capacitors has an undamped common mode.
        network = self.network(initial=(2.0, 0.0))
        result = self.advance(network, duration_seconds=0.9,
                              conductance_siemens=[[1.0, -1.0], [-1.0, 1.0]])
        decay = math.exp(-1.8)
        np.testing.assert_allclose(result.end_volts, [1 + decay, 1 - decay], rtol=1e-13)
        self.assertAlmostEqual(sum(result.start_charge_coulomb), sum(result.end_charge_coulomb))
        self.assertAlmostEqual(result.end_energy_joule, 1 + decay**2)

    def test_partitioned_interval_preserves_state_and_weighted_mean(self):
        capacitance = [[2.0, -0.5], [-0.5, 1.0]]
        whole = self.network(initial=(0.4, -0.1), capacitance=capacitance)
        split = self.network(initial=(0.4, -0.1), capacitance=capacitance)
        drive = dict(input_conductance_siemens=[[0.8], [0.3]], input_volts=[1.4])
        full = self.advance(whole, duration_seconds=0.9, **drive)
        first = self.advance(split, duration_seconds=0.2, **drive)
        second = self.advance(split, duration_seconds=0.7, **drive)
        np.testing.assert_allclose(second.end_volts, full.end_volts, rtol=1e-13)
        combined = (0.2 * np.array(first.mean_volts) + 0.7 * np.array(second.mean_volts)) / 0.9
        np.testing.assert_allclose(combined, full.mean_volts, rtol=1e-13)
        self.assertAlmostEqual(first.input_work_joule + second.input_work_joule, full.input_work_joule)

    def test_changing_capacitance_conserves_nodal_charge_and_records_energy(self):
        network = self.network(initial=(1.0, 2.0), capacitance=[[2.0, 0.0], [0.0, 3.0]])
        result = network.switch_capacitance(capacitance_farad=[[2.0, -1.0], [-1.0, 2.0]],
                                            injected_charge_coulomb=[1.0, -2.0])
        np.testing.assert_allclose(result.before_charge_coulomb, [2.0, 6.0])
        np.testing.assert_allclose(result.after_charge_coulomb, [3.0, 4.0])
        np.testing.assert_allclose(result.after_volts, [10 / 3, 11 / 3])
        self.assertEqual(result.before_energy_joule, 7.0)
        self.assertAlmostEqual(result.after_energy_joule, 37 / 3)
        self.assertAlmostEqual(result.stored_energy_change_joule, 16 / 3)
        unchanged_charge = network.switch_capacitance(capacitance_farad=[[4.0, -2.0], [-2.0, 4.0]],
                                                      injected_charge_coulomb=[0.0, 0.0])
        np.testing.assert_allclose(unchanged_charge.after_charge_coulomb, [3.0, 4.0])
        np.testing.assert_allclose(unchanged_charge.after_volts, [5 / 3, 11 / 6])
        self.assertAlmostEqual(unchanged_charge.after_energy_joule, 37 / 6)
        self.assertAlmostEqual(unchanged_charge.stored_energy_change_joule, -37 / 6)

    def test_zero_conductance_storage_has_exact_linear_ramp_and_work(self):
        network = self.network(initial=(1.0, 2.0), capacitance=[[2.0, 0.0], [0.0, 4.0]])
        result = self.advance(network, duration_seconds=0.6,
                              conductance_siemens=[[0.0, 0.0], [0.0, 0.0]],
                              input_conductance_siemens=[[2.0], [-4.0]], input_volts=[0.5])
        np.testing.assert_allclose(result.end_volts, [1.3, 1.7], rtol=1e-13)
        np.testing.assert_allclose(result.mean_volts, [1.15, 1.85], rtol=1e-13)
        self.assertAlmostEqual(result.input_work_joule, result.end_energy_joule - result.start_energy_joule)

    def test_observation_switch_does_not_reset_unread_state(self):
        network = self.network()
        first = self.advance(network, duration_seconds=0.2)
        second = self.advance(network, duration_seconds=0.5, observation_matrix=[[0.0, 1.0]])
        full = self.advance(self.network(), duration_seconds=0.7)
        self.assertEqual(second.start_volts, first.end_volts)
        self.assertGreater(second.start_volts[1], 0.0)
        np.testing.assert_allclose(second.end_volts, full.end_volts, rtol=1e-13)
        self.assertEqual(second.output_end_volts[0], second.end_volts[1])
        self.assertEqual(network.state_ids, ("read", "off_gate"))

    def test_invalid_parameters_leave_state_and_capacitance_unchanged(self):
        network = self.network()
        for changes in (
            {"duration_seconds": 0.0}, {"duration_seconds": math.inf},
            {"conductance_siemens": [[1, 1], [0, 1]]},
            {"conductance_siemens": [[1, 2], [2, 1]]},
            {"input_conductance_siemens": [[1.0]]}, {"input_volts": [math.nan]},
            {"observation_matrix": [[1.0]]}, {"feedthrough_matrix": [[1j]]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.advance(network, **changes)
            self.assertEqual(network.volts, (1.0, 0.0))
        for capacitance, charge in (
            ([[1, 0], [0, 0]], [0, 0]), ([[1, 2], [2, 1]], [0, 0]),
            ([[1, 1], [0, 1]], [0, 0]), ([[2, 0], [0, 2]], [0]),
            ([[2, 0], [0, 2]], [math.inf, 0]),
        ):
            with self.subTest(capacitance=capacitance, charge=charge), self.assertRaises(ValueError):
                network.switch_capacitance(capacitance_farad=capacitance, injected_charge_coulomb=charge)
            self.assertEqual(network.volts, (1.0, 0.0))
        # A failed switch must not have installed its otherwise valid C=2I.
        result = self.advance(network)
        self.assertAlmostEqual(result.start_energy_joule, 0.5)
        # A G-scale tolerance would admit this exact negative mode. Tiny C
        # would amplify it to exp(1) voltage growth in .01s with no drive.
        tiny_capacitance = self.network(capacitance=[[1e-18, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            self.advance(tiny_capacitance, duration_seconds=0.01,
                         conductance_siemens=[[-1e-16, 0.0], [0.0, 1.0]])
        self.assertEqual(tiny_capacitance.volts, (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
