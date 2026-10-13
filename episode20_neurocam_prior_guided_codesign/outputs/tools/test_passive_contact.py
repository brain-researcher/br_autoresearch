"""Analytic checks of EP20 equations, not NeuroCam calibration or simulation.

Every numeric circuit value below is an explicit arithmetic test fixture.
No value is a NeuroCam estimate, scientific uncertainty bound or model default.
Run on a compute node: python3 -B outputs/tools/test_passive_contact.py
"""

from dataclasses import replace
import math
import unittest

from passive_contact import (
    BOLTZMANN_J_PER_K,
    ContactParameters,
    REFERENCE_PAD_AREA_UM2,
    contact_impedance_parts,
    equilibrium_noise_psd,
)


class PassiveContactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = ContactParameters(
            r_access_l_ohm=8.0,
            g_ct_l_siemens=0.25,
            q_l_siemens_seconds_alpha=0.125,
            alpha=1.0,
            gamma=1.0,
        )

    def parts(self, *, frequency_hz=4 / math.pi, parameters=None, area_um2=9450.0):
        return contact_impedance_parts(
            frequency_hz=frequency_hz,
            pad_area_um2=area_um2,
            parameters=self.parameters if parameters is None else parameters,
        )

    def assertComplexClose(self, actual, expected):
        self.assertTrue(math.isclose(actual.real, expected.real, rel_tol=1e-12, abs_tol=1e-14))
        self.assertTrue(math.isclose(actual.imag, expected.imag, rel_tol=1e-12, abs_tol=1e-14))

    def test_finite_charge_transfer_dc_is_series_resistance(self):
        parts = self.parts(frequency_hz=0)
        self.assertEqual(parts.access_ohm, 8.0)
        self.assertEqual(parts.interface_ohm, 4.0)
        self.assertEqual(parts.total_ohm, 12.0)

    def test_parallel_rc_analytic_value(self):
        # omega=8 rad/s gives Y_interface=1/4+j, so Z=(4-16j)/17.
        parts = self.parts()
        self.assertComplexClose(parts.interface_ohm, complex(4 / 17, -16 / 17))
        self.assertComplexClose(parts.total_ohm, complex(8 + 4 / 17, -16 / 17))

    def test_ideal_blocking_capacitor_has_no_equilibrium_noise(self):
        parameters = replace(self.parameters, r_access_l_ohm=0.0, g_ct_l_siemens=0.0)
        parts = self.parts(parameters=parameters)
        self.assertEqual(parts.interface_ohm, -1j)
        self.assertEqual(parts.total_ohm.real, 0.0)
        self.assertEqual(
            equilibrium_noise_psd(impedance_ohm=parts.total_ohm, temperature_kelvin=300.0),
            0.0,
        )
        self.assertGreater(abs(parts.total_ohm), 0.0)

    def test_fractional_blocking_contact_has_expected_phase_and_dissipation(self):
        parameters = replace(
            self.parameters, r_access_l_ohm=0.0, g_ct_l_siemens=0.0, alpha=0.5
        )
        parts = self.parts(parameters=parameters)
        # Q=1/8, omega=8 and alpha=1/2 give Z=2-2j.
        self.assertComplexClose(parts.interface_ohm, 2 - 2j)
        self.assertTrue(math.isclose(math.atan2(parts.interface_ohm.imag, parts.interface_ohm.real), -math.pi / 4))

    def test_pad_scaling_keeps_access_and_interface_distinct(self):
        for gamma in (1.0, 0.7):
            parameters = replace(self.parameters, gamma=gamma)
            large = self.parts(parameters=parameters)
            small = self.parts(parameters=parameters, area_um2=REFERENCE_PAD_AREA_UM2 / 4)
            with self.subTest(gamma=gamma):
                self.assertAlmostEqual(small.access_ohm / large.access_ohm, 2.0)
                self.assertComplexClose(small.interface_ohm / large.interface_ohm, 4**gamma)
                self.assertFalse(math.isclose(abs(small.total_ohm / large.total_ohm), 4**gamma))

    def test_passivity_and_capacitive_sign_across_both_laws(self):
        # Includes out-of-primary-band frequencies: this component is not an
        # oracle forward band-pass filter.
        for gamma in (1.0, 0.7):
            for alpha in (0.25, 0.7, 1.0):
                for conductance in (0.0, 0.25):
                    parameters = replace(
                        self.parameters, gamma=gamma, alpha=alpha, g_ct_l_siemens=conductance
                    )
                    for frequency_hz in (0.1, 1.0, 100.0, 100000.0):
                        with self.subTest(gamma=gamma, alpha=alpha, conductance=conductance, frequency_hz=frequency_hz):
                            parts = self.parts(frequency_hz=frequency_hz, parameters=parameters, area_um2=2450.0)
                            self.assertGreaterEqual(parts.interface_ohm.real, 0)
                            self.assertLess(parts.interface_ohm.imag, 0)
                            self.assertGreater(parts.total_ohm.real, 0)

    def test_thermal_noise_composes_once_and_uses_real_part(self):
        parts = self.parts()
        temperature = 300.0
        interface_noise = equilibrium_noise_psd(
            impedance_ohm=parts.interface_ohm, temperature_kelvin=temperature
        )
        access_noise = equilibrium_noise_psd(
            impedance_ohm=parts.access_ohm, temperature_kelvin=temperature
        )
        total_noise = equilibrium_noise_psd(
            impedance_ohm=parts.total_ohm, temperature_kelvin=temperature
        )
        self.assertTrue(math.isclose(total_noise, interface_noise + access_noise, rel_tol=1e-12))
        expected = 4 * BOLTZMANN_J_PER_K * temperature * (4 / 17)
        self.assertTrue(math.isclose(interface_noise, expected, rel_tol=1e-12))
        self.assertTrue(math.isclose(total_noise - interface_noise, access_noise, rel_tol=1e-12))

    def test_temperature_is_explicit_and_scales_noise(self):
        cold = equilibrium_noise_psd(impedance_ohm=2 - 2j, temperature_kelvin=100.0)
        warm = equilibrium_noise_psd(impedance_ohm=2 - 2j, temperature_kelvin=200.0)
        self.assertEqual(warm, 2 * cold)
        self.assertEqual(equilibrium_noise_psd(impedance_ohm=2 - 2j, temperature_kelvin=0.0), 0.0)
        with self.assertRaises(TypeError):
            equilibrium_noise_psd(impedance_ohm=2 - 2j)

    def test_unbounded_blocking_dc_is_not_regularized(self):
        for alpha in (0.5, 1.0):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "unbounded"):
                self.parts(frequency_hz=0.0, parameters=replace(self.parameters, g_ct_l_siemens=0.0, alpha=alpha))

    def test_parameter_domains_and_missing_values(self):
        invalid = (
            {"r_access_l_ohm": -1},
            {"g_ct_l_siemens": -1},
            {"q_l_siemens_seconds_alpha": 0},
            {"alpha": 0},
            {"alpha": 1.1},
            {"gamma": 0.8},
            {"r_access_l_ohm": math.nan},
            {"q_l_siemens_seconds_alpha": math.inf},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.parameters, **change)
        with self.assertRaises(TypeError):
            ContactParameters()

    def test_invalid_frequency_area_temperature_and_active_impedance(self):
        for frequency_hz in (-1.0, math.nan, math.inf):
            with self.subTest(frequency_hz=frequency_hz), self.assertRaises(ValueError):
                self.parts(frequency_hz=frequency_hz)
        for area_um2 in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(area_um2=area_um2), self.assertRaises(ValueError):
                self.parts(area_um2=area_um2)
        for temperature in (-1.0, math.nan, math.inf):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                equilibrium_noise_psd(impedance_ohm=1.0, temperature_kelvin=temperature)
        for impedance in (-1 + 1j, complex(math.nan, 0), complex(0, math.inf)):
            with self.subTest(impedance=impedance), self.assertRaises(ValueError):
                equilibrium_noise_psd(impedance_ohm=impedance, temperature_kelvin=300.0)


if __name__ == "__main__":
    unittest.main()
