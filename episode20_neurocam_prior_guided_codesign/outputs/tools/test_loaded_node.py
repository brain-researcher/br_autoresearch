"""Analytic loading/noise checks; all constants are non-device fixtures."""

import math
import unittest

from loaded_node import independent_loaded_node_spectrum
from passive_contact import BOLTZMANN_J_PER_K, ContactParameters, contact_impedance_parts


class LoadedNodeTests(unittest.TestCase):
    def spectrum(self, **changes):
        arguments = dict(
            source_impedance_ohm=4.0,
            input_impedance_ohm=12.0,
            owned_contact_impedance_ohm=2.0,
            temperature_kelvin=300.0,
            voltage_noise_psd_v2_per_hz=3e-20,
            current_noise_psd_a2_per_hz=5e-20,
            other_node_psds_v2_per_hz={"TFT": 7e-20, "reference": 11e-20},
        )
        arguments.update(changes)
        return independent_loaded_node_spectrum(**arguments)

    def test_resistive_divider_and_separate_noise_contributions(self):
        result = self.spectrum()
        expected_contact = 0.75**2 * 4 * BOLTZMANN_J_PER_K * 300 * 2
        self.assertEqual(result.signal_transfer, 0.75)
        self.assertEqual(result.parallel_impedance_ohm, 3.0)
        self.assertAlmostEqual(result.contact_node_psd_v2_per_hz / expected_contact, 1.0)
        self.assertAlmostEqual(result.current_node_psd_v2_per_hz / (9 * 5e-20), 1.0)
        self.assertEqual(dict(result.other_node_psds_v2_per_hz), {"TFT": 7e-20, "reference": 11e-20})
        expected_total = expected_contact + (3 + 45 + 7 + 11) * 1e-20
        self.assertAlmostEqual(result.total_node_psd_v2_per_hz / expected_total, 1.0)

    def test_contact_interface_noise_is_not_counted_twice(self):
        result = self.spectrum(owned_contact_impedance_ohm=0.5)
        # The source path remains 4 ohms; selecting the owned noise portion
        # must not add it to the source or change the voltage transfer.
        self.assertEqual(result.signal_transfer, 0.75)
        expected = 0.75**2 * 4 * BOLTZMANN_J_PER_K * 300 * 0.5
        self.assertAlmostEqual(result.contact_node_psd_v2_per_hz / expected, 1.0)

    def test_capacitive_contact_has_no_fictitious_equilibrium_noise(self):
        contact = contact_impedance_parts(
            frequency_hz=1 / (2 * math.pi), pad_area_um2=9450.0,
            parameters=ContactParameters(0.0, 0.0, 0.5, 1.0, 1.0),
        )
        result = self.spectrum(
            source_impedance_ohm=contact.total_ohm, input_impedance_ohm=2.0,
            owned_contact_impedance_ohm=contact.interface_ohm,
        )
        self.assertAlmostEqual(result.signal_transfer, complex(0.5, 0.5))
        self.assertAlmostEqual(result.parallel_impedance_ohm, complex(1.0, -1.0))
        self.assertEqual(result.contact_node_psd_v2_per_hz, 0.0)
        self.assertAlmostEqual(result.current_node_psd_v2_per_hz / 1e-19, 1.0)

    def test_complex_transfer_magnitude_is_squared_without_clipping(self):
        result = self.spectrum(source_impedance_ohm=complex(1, 3),
                               input_impedance_ohm=complex(1, -3),
                               owned_contact_impedance_ohm=1.0)
        self.assertEqual(result.signal_transfer, complex(0.5, -1.5))
        self.assertEqual(result.parallel_impedance_ohm, 5.0)
        expected = 2.5 * 4 * BOLTZMANN_J_PER_K * 300
        self.assertAlmostEqual(result.contact_node_psd_v2_per_hz / expected, 1.0)

    def test_short_input_load_suppresses_source_and_current_terms(self):
        result = self.spectrum(input_impedance_ohm=0.0)
        self.assertEqual(result.signal_transfer, 0.0)
        self.assertEqual(result.parallel_impedance_ohm, 0.0)
        self.assertEqual(result.contact_node_psd_v2_per_hz, 0.0)
        self.assertEqual(result.current_node_psd_v2_per_hz, 0.0)
        self.assertAlmostEqual(result.total_node_psd_v2_per_hz / 21e-20, 1.0)

    def test_invalid_or_singular_inputs_are_not_repaired(self):
        for changes in (
            {"source_impedance_ohm": -12.0},
            {"input_impedance_ohm": complex(1, math.inf)},
            {"owned_contact_impedance_ohm": -1.0},
            {"temperature_kelvin": -1.0},
            {"voltage_noise_psd_v2_per_hz": math.nan},
            {"current_noise_psd_a2_per_hz": -1.0},
            {"other_node_psds_v2_per_hz": {"reference": -1.0}},
            {"other_node_psds_v2_per_hz": {"": 0.0}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.spectrum(**changes)


if __name__ == "__main__":
    unittest.main()
