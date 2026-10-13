"""Focused physical-boundary/assembly identities, not device parameter sweeps."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from causal_cpe import realize_contact_cpe
from cpe_array import AssemblyLimits, NeurocamCPEArray
from passive_contact import ContactParameters
from sparse_array_rc import PassiveStarConductance
from test_sparse_array_rc import controls


def ideal_contact():
    return realize_contact_cpe(parameters=ContactParameters(0, 0, 2, 1, 1),
                               pad_area_um2=9450, pole_low_rad_per_second=None,
                               pole_high_rad_per_second=None, branch_count=0,
                               forward_band_hz=(0.5, 1), check_frequencies_hz=(0.5, 1),
                               grid_relative_tolerance=1e-12, truncation_relative_tolerance=1e-12)


def fractional_contact():
    return realize_contact_cpe(parameters=ContactParameters(0, 0.2, 1.5, 0.5, 0.7),
                               pad_area_um2=4725, pole_low_rad_per_second=1,
                               pole_high_rad_per_second=10, branch_count=2,
                               forward_band_hz=(0.5, 1), check_frequencies_hz=(0.5, 0.75, 1),
                               grid_relative_tolerance=1, truncation_relative_tolerance=1)


def make_array(**changes):
    ideal = ideal_contact()
    contacts = changes.pop("contacts", (ideal,) * 4096)
    access = changes.pop("access_resistance_ohm", np.zeros((64, 64)))
    extra = sum(c.internal_state_count if c.alpha < 1 else int(r > 0)
                for c, r in zip(contacts, np.asarray(access).ravel()))
    values = dict(contacts=contacts, access_resistance_ohm=access, access_resistance_owner="assembler",
                  voltage_reference_id="explicit_fixture_reference", initial_field_volts=np.zeros((64, 64)),
                  initial_pad_volts=np.zeros((64, 64)), initial_source_volts=np.zeros(64),
                  initial_contact_state_volts=np.zeros(extra), pad_capacitance_farad=np.ones((64, 64)),
                  source_capacitance_farad=np.ones(64), mutual_capacitance_edges=[],
                  tft_on_conductance_siemens=np.full((64, 64), 0.2),
                  tft_off_conductance_siemens=np.full((64, 64), 0.1),
                  pad_leak_conductance_siemens=np.zeros((64, 64)),
                  source_load_conductance_siemens=np.zeros(64), other_resistor_edges=[],
                  limits=AssemblyLimits(100000, 100000, 100000, 64 * 1024 * 1024))
    values.update(changes)
    return NeurocamCPEArray(**values)


def advance(array, **changes):
    field = np.zeros((64, 64))
    field[0, 0] = 1
    values = dict(gate_row=0, duration_seconds=0.02, field_volts=field,
                  voltage_reference_id="explicit_fixture_reference", observed_sources=(0,),
                  injected_charge_coulomb=np.zeros(len(array.state_ids)), controls=controls())
    values.update(changes)
    return array.advance_row(**values)


class CPEArrayTests(unittest.TestCase):
    def test_structured_stars_match_dense_schur_and_physical_port_work(self):
        h = np.array([0.2, 0.7, 1.1])
        resistance = 3.0
        star = PassiveStarConductance(state_count=4,
                                     groups=(((0, 1, 2), tuple(h), resistance), ((3,), (0.4,), 0.0)))
        expected = np.zeros((4, 4))
        expected[:3, :3] = np.diag(h) - resistance * np.outer(h, h) / (1 + resistance * sum(h))
        expected[3, 3] = 0.4
        vectors = np.array([[1, 0.3], [1, -0.1], [1, 0.6], [1, 0.4]])
        np.testing.assert_allclose(star @ vectors, expected @ vectors, rtol=2e-14, atol=2e-15)
        np.testing.assert_allclose(star.T @ vectors, expected.T @ vectors, rtol=2e-14, atol=2e-15)
        np.testing.assert_allclose(star.diagonal(), np.diag(expected), rtol=2e-15)
        ports = np.array([1.2, -0.4])
        b = h / (1 + resistance * sum(h))
        current = np.r_[b * ports[0], 0.4 * ports[1]]
        np.testing.assert_allclose(star.drive_current(port_volts=ports), current, rtol=2e-15)
        mean = vectors[:, 1]
        junction = (ports[0] / resistance + h @ mean[:3]) / (1 / resistance + sum(h))
        expected_work = 0.2 * (ports[0] * (ports[0] - junction) / resistance
                               + ports[1] * 0.4 * (ports[1] - mean[3]))
        actual = star.source_work(port_volts=ports, mean_state_volts=mean, duration_seconds=0.2)
        self.assertAlmostEqual(actual, expected_work, places=14)
        self.assertNotAlmostEqual(actual, float(0.2 * (current @ mean)), places=6)

    def test_direct_ideal_capacitor_jump_and_constant_voltage_energy_balance(self):
        array = make_array()
        result = advance(array)
        kernel = result.kernel_interval
        self.assertAlmostEqual(kernel.observation.start_volts[0], 2 / 3, places=13)
        self.assertAlmostEqual(result.physical_energy_before_boundary_joule, 0.0, places=14)
        self.assertAlmostEqual(result.physical_energy_after_boundary_joule, 1 / 3, places=13)
        self.assertAlmostEqual(result.field_source_boundary_work_joule, 1 / 3, places=13)
        self.assertEqual(result.supplied_charge_boundary_work_joule, 0.0)
        # Exact symmetry reduction: driven pad, the other 63 pads on its source,
        # and the shared source. No other column acquires nonzero voltage.
        a = np.array([[-0.2 / 3, 0, 0.2 / 3], [0, -0.1 / 3, 0.1 / 3], [0.2, 6.3, -6.5]])
        start = np.array([2 / 3, 0, 0])
        expected_end = expm(0.02 * a) @ start
        np.testing.assert_allclose(np.array(kernel.observation.end_volts)[[0, 64, 4096]],
                                   expected_end, rtol=2e-10, atol=1e-13)
        # Integrate the analytic quadratic resistor loss using an independent
        # 10-dimensional dense exponential on x tensor x, including its zero mode.
        dissipation = np.array([[0.2, 0, -0.2], [0, 6.3, -6.3], [-0.2, -6.3, 6.5]])
        augmented = np.zeros((10, 10))
        augmented[:9, :9] = np.kron(a, np.eye(3)) + np.kron(np.eye(3), a)
        augmented[9, :9] = dissipation.ravel()
        loss = (expm(0.02 * augmented) @ np.r_[np.kron(start, start), 0])[9]
        physical_change = result.physical_energy_end_joule - result.physical_energy_after_boundary_joule
        self.assertAlmostEqual(result.field_source_interval_work_joule - physical_change, loss, places=11)
        self.assertEqual(kernel.observation.input_work_joule, 0.0)  # No Norton resistor drive in this fixture.
        self.assertGreater(result.field_source_interval_work_joule, 0.0)

    def test_simultaneous_explicit_charge_and_voltage_jump_conserves_physical_energy(self):
        array = make_array()
        charge = np.zeros(len(array.state_ids))
        charge[0] = 0.3
        result = advance(array, injected_charge_coulomb=charge)
        self.assertAlmostEqual(result.kernel_interval.observation.start_volts[0], 2.3 / 3, places=13)
        change = result.physical_energy_after_boundary_joule - result.physical_energy_before_boundary_joule
        self.assertAlmostEqual(change, result.field_source_boundary_work_joule
                               + result.supplied_charge_boundary_work_joule, places=13)

    def test_ideal_contact_with_access_resistance_matches_independent_dense_response(self):
        access = np.zeros((64, 64))
        access[0, 0] = 2.0
        array = make_array(access_resistance_ohm=access)
        result = advance(array)
        self.assertEqual(len(array.state_ids), 4161)
        self.assertEqual(result.field_source_boundary_work_joule, 0.0)
        # Four physical states: pad, 63 symmetric other pads, source, interface.
        c = np.array([[3, 0, 0, -2], [0, 189, 0, 0], [0, 0, 1, 0], [-2, 0, 0, 2]], dtype=float)
        g = np.array([[0.2, 0, -0.2, 0], [0, 6.3, -6.3, 0], [-0.2, -6.3, 6.5, 0], [0, 0, 0, 0.5]])
        a = -np.linalg.solve(c, g)
        drive = np.linalg.solve(c, [0, 0, 0, 0.5])
        augmented = np.zeros((9, 9))
        augmented[:4, :4] = a
        augmented[:4, 4] = drive
        augmented[5:, :4] = np.eye(4)
        expected = expm(0.02 * augmented) @ np.r_[np.zeros(4), 1, np.zeros(4)]
        kernel = result.kernel_interval
        selected = [0, 64, 4096, 4160]
        np.testing.assert_allclose(np.array(kernel.observation.end_volts)[selected], expected[:4],
                                   rtol=2e-10, atol=1e-13)
        np.testing.assert_allclose(np.array(kernel.observation.mean_volts)[selected], expected[5:] / 0.02,
                                   rtol=2e-10, atol=1e-13)
        self.assertAlmostEqual(result.field_source_interval_work_joule, 0.02 * 0.5 * (1 - expected[8] / 0.02), places=12)
        self.assertNotAlmostEqual(result.field_source_interval_work_joule, kernel.observation.input_work_joule, places=6)

    def test_full_mixed_array_retains_every_contact_and_off_gate_state(self):
        contacts = [fractional_contact()] * 4096
        contacts[0] = contacts[1] = ideal_contact()
        access = np.ones((64, 64))
        access[0, 0] = 0.0
        extra = 4094 * 2 + 1
        initial_contact = np.zeros(extra)
        initial_contact[-1] = 0.1  # Retained state on an off-gate, unread contact.
        array = make_array(contacts=contacts, access_resistance_ohm=access,
                           initial_contact_state_volts=initial_contact)
        self.assertEqual(array.resources.total_states, 4160 + extra)
        self.assertEqual(array.resources.fractional_contact_states, 4094 * 2)
        self.assertEqual(array.resources.ideal_series_interface_states, 1)
        self.assertEqual(array.resources.star_terms, 4096 + 4094 * 2)
        first = advance(array, duration_seconds=0.001)
        saved = array.volts
        second = advance(array, gate_row=63, observed_sources=(7, 19), duration_seconds=0.001)
        self.assertEqual(second.kernel_interval.boundary.transition.before_volts, saved)
        self.assertGreater(abs(second.kernel_interval.observation.end_volts[-1]), 0.0)
        self.assertEqual(len(second.kernel_interval.observation.output_mean_volts), 2)
        self.assertEqual(second.field_source_boundary_work_joule, 0.0)
        self.assertEqual(len(first.kernel_interval.observation.end_volts), 4160 + extra)

    def test_accounting_refusal_preserves_state_and_previous_field(self):
        array = make_array()
        before, previous = array.volts, array.previous_field_volts
        with patch.object(array, "_physical_energy", side_effect=ValueError("explicit accounting refusal")):
            with self.assertRaisesRegex(ValueError, "accounting refusal"):
                advance(array)
        self.assertEqual(array.volts, before)
        self.assertEqual(array.previous_field_volts, previous)
        result = advance(array)
        self.assertAlmostEqual(result.kernel_interval.observation.start_volts[0], 2 / 3, places=13)

    def test_reference_ownership_and_explicit_resource_limits_refuse(self):
        with self.assertRaisesRegex(ValueError, "assembly limit"):
            make_array(limits=AssemblyLimits(10, 100000, 100000, 64 * 1024 * 1024))
        with self.assertRaisesRegex(ValueError, "numeric-storage"):
            make_array(limits=AssemblyLimits(100000, 100000, 100000, 1))
        with self.assertRaisesRegex(ValueError, "upstream-owned"):
            make_array(access_resistance_owner="upstream", access_resistance_ohm=np.ones((64, 64)))
        array = make_array()
        before = array.volts
        with self.assertRaisesRegex(ValueError, "reference convention"):
            advance(array, voltage_reference_id="different_reference")
        self.assertEqual(array.volts, before)


if __name__ == "__main__":
    unittest.main()
