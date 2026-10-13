"""Bounded analytic identities; fixture numbers are not NeuroCam estimates."""

import math
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from causal_cpe import realize_contact_cpe
from coupled_rc import PassiveRCNetwork
from passive_contact import ContactParameters, contact_impedance_parts
from sparse_array_rc import SparsePassiveRCNetwork, passive_matrix
from test_sparse_array_rc import controls


def realize(**changes):
    arguments = dict(parameters=ContactParameters(7.0, 0.2, 1.5, 0.5, 0.7),
                     pad_area_um2=4725.0, pole_low_rad_per_second=1e-8,
                     pole_high_rad_per_second=1e8, branch_count=128,
                     forward_band_hz=(0.1, 10.0), check_frequencies_hz=np.geomspace(0.1, 10, 9),
                     grid_relative_tolerance=0.002, truncation_relative_tolerance=0.002)
    arguments.update(changes)
    return realize_contact_cpe(**arguments)


def small_contact():
    # Modest poles avoid making a numerical stiffness benchmark from an
    # identity test. Its permissive approximation tolerances are explicit.
    return realize(pole_low_rad_per_second=1.0, pole_high_rad_per_second=10.0,
                   branch_count=2, forward_band_hz=(0.5, 1.0),
                   check_frequencies_hz=(0.5, 0.75, 1.0),
                   grid_relative_tolerance=1.0, truncation_relative_tolerance=1.0)


class CausalCPETests(unittest.TestCase):
    def test_positive_branches_match_fractional_admittance_and_area_scaling(self):
        contact = realize()
        self.assertEqual(contact.internal_state_count, 128)
        self.assertTrue(all(value > 0 for value in contact.branch_conductance_siemens))
        self.assertTrue(all(value > 0 for value in contact.branch_capacitance_farad))
        np.testing.assert_allclose(np.array(contact.branch_conductance_siemens)
                                   / np.array(contact.branch_capacitance_farad),
                                   contact.poles_rad_per_second, rtol=2e-15)
        frequencies = np.array(contact.checked_frequencies_hz)
        actual = contact.admittance(frequencies_hz=frequencies)
        parameters = ContactParameters(7.0, 0.2, 1.5, 0.5, 0.7)
        expected = np.array([1 / contact_impedance_parts(frequency_hz=float(f),
                             pad_area_um2=4725.0, parameters=parameters).interface_ohm
                             for f in frequencies])
        normalized_error = np.max(np.abs(actual - expected)
                                  / np.abs(expected - contact.charge_transfer_siemens))
        self.assertLess(normalized_error, 0.002)
        self.assertAlmostEqual(normalized_error, contact.maximum_grid_relative_error, places=13)
        larger = realize(pad_area_um2=9450.0)
        np.testing.assert_allclose(larger.admittance(frequencies_hz=frequencies),
                                   actual * 2 ** 0.7, rtol=2e-15)

    def test_finite_positive_time_step_current_has_the_fractional_response(self):
        contact = realize()
        times = np.array([0.01, 0.1, 1.0])
        voltage = 0.3
        expected = voltage * (contact.charge_transfer_siemens
                              + contact.scaled_q_siemens_seconds_alpha
                              * times ** -contact.alpha / math.gamma(1 - contact.alpha))
        np.testing.assert_allclose(contact.step_current(times_seconds=times, step_volts=voltage),
                                   expected, rtol=0.002, atol=0)
        with self.assertRaises(ValueError):
            contact.step_current(times_seconds=[0.0], step_volts=1.0)

    def test_stamped_series_branches_have_the_declared_two_terminal_admittance(self):
        contact = small_contact()
        stamp = contact.stamp(terminal_a=0, terminal_b=1, first_internal_index=2, state_prefix="contact")
        size = 2 + contact.internal_state_count
        c = passive_matrix(shunts=np.zeros(size), edges=stamp.mutual_capacitance_edges,
                           positive_shunts=False).toarray()
        g = passive_matrix(shunts=np.zeros(size), edges=stamp.resistor_edges,
                           positive_shunts=False).toarray()
        for frequency in (0.5, 0.75, 1.0):
            y = g + 2j * math.pi * frequency * c
            reduced = y[:2, :2] - y[:2, 2:] @ np.linalg.solve(y[2:, 2:], y[2:, :2])
            expected = contact.admittance(frequencies_hz=[frequency])[0] * np.array([[1, -1], [-1, 1]])
            np.testing.assert_allclose(reduced, expected, rtol=2e-14, atol=2e-14)

    def test_contact_states_persist_across_explicit_switches(self):
        contact = small_contact()
        stamp = contact.stamp(terminal_a=0, terminal_b=1, first_internal_index=2, state_prefix="contact")
        names = ("pad", "return") + stamp.internal_state_ids
        shunts = (2.0, 3.0) + stamp.internal_capacitance_to_reference_farad
        initial = [1.0, -0.2, 0.3, 0.6]
        sparse = SparsePassiveRCNetwork(state_ids=names, initial_volts=initial,
                                        capacitance_to_reference_farad=shunts,
                                        mutual_capacitance_edges=stamp.mutual_capacitance_edges)
        c = passive_matrix(shunts=shunts, edges=stamp.mutual_capacitance_edges,
                           positive_shunts=False).toarray()
        dense = PassiveRCNetwork(state_ids=names, initial_volts=initial, capacitance_farad=c)
        previous = np.array(initial)
        loads = np.array([0.2, 0.3, 0.0, 0.0])
        for switched_conductance in (0.1, 0.7):
            edges = stamp.resistor_edges + ((0, 1, switched_conductance),)
            result = sparse.advance(duration_seconds=0.2, conductance_to_reference_siemens=loads,
                                    resistor_edges=edges, drive_current_amp=np.zeros(4),
                                    observation_matrix=csr_matrix([[1, -1, 0, 0]]),
                                    feedthrough_volts=[0], injected_charge_coulomb=np.zeros(4), controls=controls())
            np.testing.assert_array_equal(result.boundary.transition.before_volts, previous)
            conductance = passive_matrix(shunts=loads, edges=edges, positive_shunts=False).toarray()
            expected = dense.advance(duration_seconds=0.2, conductance_siemens=conductance,
                                     input_conductance_siemens=np.zeros((4, 1)), input_volts=[0],
                                     observation_matrix=[[1, -1, 0, 0]], feedthrough_matrix=[[0]])
            np.testing.assert_allclose(result.observation.end_volts, expected.end_volts, rtol=2e-11, atol=1e-13)
            np.testing.assert_allclose(result.observation.output_mean_volts, expected.output_mean_volts,
                                       rtol=2e-11, atol=1e-13)
            self.assertLess(result.observation.end_energy_joule, result.observation.start_energy_joule)
            previous = np.array(sparse.volts)
        self.assertGreater(np.linalg.norm(previous[2:]), 0.1)

    def test_alpha_one_is_exact_mutual_capacitor_with_no_internal_states(self):
        contact = realize(parameters=ContactParameters(7, 0.2, 1.5, 1.0, 1.0),
                           pole_low_rad_per_second=None, pole_high_rad_per_second=None, branch_count=0)
        self.assertEqual(contact.internal_state_count, 0)
        self.assertEqual(contact.ideal_capacitance_farad, 0.75)
        self.assertEqual(contact.maximum_grid_relative_error, 0)
        self.assertEqual(contact.uniform_truncation_relative_bound, 0)
        np.testing.assert_allclose(contact.admittance(frequencies_hz=[1, 2]),
                                   0.1 + 2j * math.pi * np.array([1, 2]) * 0.75, rtol=0, atol=0)
        stamp = contact.stamp(terminal_a=0, terminal_b=1, first_internal_index=2, state_prefix="ideal")
        self.assertEqual(stamp.internal_state_ids, ())
        self.assertEqual(stamp.mutual_capacitance_edges, ((0, 1, 0.75),))
        self.assertEqual(stamp.resistor_edges, ((0, 1, 0.1),))
        np.testing.assert_allclose(contact.step_current(times_seconds=[1e-5, 1], step_volts=2), [0.2, 0.2])

    def test_truncation_and_grid_failures_are_separate_and_cannot_be_hidden_by_leakage(self):
        with self.assertRaisesRegex(ValueError, "truncation"):
            realize(truncation_relative_tolerance=1e-6)
        with self.assertRaisesRegex(ValueError, "finite-grid"):
            realize(parameters=ContactParameters(7, 1e12, 1.5, 0.5, 0.7),
                    grid_relative_tolerance=1e-6)
        for changes in ({"check_frequencies_hz": (0.2, 10)}, {"branch_count": 0},
                        {"pad_area_um2": 1 + 1j}, {"pole_low_rad_per_second": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                realize(**changes)
        contact = small_contact()
        for a, b, first in ((0, 0, 2), (0, 2, 2), (0, 1, 1)):
            with self.subTest(terminals=(a, b, first)), self.assertRaises(ValueError):
                contact.stamp(terminal_a=a, terminal_b=b, first_internal_index=first, state_prefix="bad")


if __name__ == "__main__":
    unittest.main()
