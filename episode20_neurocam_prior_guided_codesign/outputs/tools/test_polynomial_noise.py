"""Original-port cubic ownership, analytic remainder and fixed-cell refusal.

Three focused fixtures. No saved trajectory, hardware parameters or new draw
is used; physical port constraints reuse the existing deterministic layout.
"""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from circuit_noise_provider import SpectralCircuitNoise, SpectralCubicCircuitNoise
from polynomial_noise import NoisePolynomial
from raw_input_hold import PhysicalAction, compile_midpoint_hold
from raw_world_time import RawWorldTime
from spectral_noise import LatentQuadrature, SpectralNoise, SpectralNoiseLimits
from streaming_midpoint_hold import compile_streaming_midpoint_hold
from test_physical_noise_ports import layout


def fixture(*, cell_for_key=None, evaluation_cap=20):
    spec = layout()
    access = spec.port_ids.index("contact:0000:star_access_voltage")
    spectrum = SpectralNoise(frequencies_hz=(1., 7.), quadrature_widths_hz=(.5, .25),
        port_ids=spec.port_ids, port_units=spec.port_units, latent_ids=("z0", "z1"),
        sparse_terms=((0, 0, 0, 1+2j), (0, access, 0, .7-.2j),
                      (1, 0, 1, .3j), (1, access, 1, -.4)),
        quadratures=(LatentQuadrature(0, "z0", 1., -.25), LatentQuadrature(1, "z1", -.5, .75)),
        time_window_seconds=(0., .02),
        limits=SpectralNoiseLimits(4, 2, 2, 10000, 2**20, evaluation_cap, 1000), check=lambda: None)
    tolerance = tuple(1e-3 if unit == "A" else 2e-3 for unit in spec.port_units)
    if cell_for_key is None:
        cell_for_key = lambda key: (0., .01, .005) if key < .01 else (.01, .02, .015)
    provider = SpectralCubicCircuitNoise(layout=spec, spectrum=spectrum,
        port_absolute_tolerances=tolerance, cell_for_key=cell_for_key, check=lambda: None)
    return spec, spectrum, provider


class PolynomialNoiseTests(unittest.TestCase):
    def test_original_coefficient_ownership_half_open_domain_and_zero_port_admission(self):
        spec = layout()
        coefficients = np.zeros((4, spec.numeric_value_count))
        coefficients[:, 0] = (.7, -.2, .05, .01)
        coefficients[:, 1] = (-.1, .3, -.02, .04)
        saved = coefficients.copy()
        polynomial = NoisePolynomial(spec, coefficients, .5, 0., 1.)
        coefficients[:] = 9.
        np.testing.assert_array_equal(polynomial.numeric_array, saved)
        self.assertEqual(polynomial.sample_numeric_bytes, 32 * spec.numeric_value_count + 24)
        self.assertEqual(polynomial.state_ids, spec.state_ids)
        self.assertEqual(polynomial.voltage_reference_id, spec.voltage_reference_id)
        with self.assertRaises(ValueError):
            polynomial.numeric_array.flags.writeable = True
        with self.assertRaises(FrozenInstanceError):
            polynomial.origin_seconds = .4
        for time in (0., .4, .5, .9, math.nextafter(1., 0.)):
            delta = time - .5
            expected = sum(saved[order] * delta**order for order in range(4))
            point = polynomial.sample_at(time)
            self.assertIs(point.layout, spec)
            np.testing.assert_allclose(point.numeric_array, expected, rtol=2e-15, atol=2e-16)
        for invalid in (-.1, 1., math.inf, math.nan, True):
            with self.subTest(time=invalid), self.assertRaises(ValueError):
                polynomial.sample_at(invalid)
        for invalid in (np.zeros_like(saved, dtype=bool), np.zeros_like(saved, dtype=complex), saved[:3],
                        np.full_like(saved, math.nan)):
            with self.assertRaises(ValueError):
                NoisePolynomial(spec, invalid, .5, 0., 1.)
        # A forbidden higher-degree source cannot hide behind zero c0.
        for port in ("contact:0001:star_access_voltage", "contact:0002:star_branch:0"):
            invalid = np.zeros_like(saved)
            invalid[3, spec.port_ids.index(port)] = 1.
            with self.assertRaises(ValueError):
                NoisePolynomial(spec, invalid, .5, 0., 1.)
        overflow = np.zeros_like(saved)
        overflow[:2, 0] = np.finfo(np.float64).max
        with self.assertRaises(ValueError):
            NoisePolynomial(spec, overflow, .5, 0., 1.).sample_at(.9)

    def test_same_spectral_trajectory_outward_remainder_and_unchanged_cell_across_splits(self):
        spec, spectrum, provider = fixture()
        with patch.object(np.random, "Generator", side_effect=AssertionError("no redraw")):
            polynomial = provider(.002)
            self.assertIs(provider(.008), polynomial)
            self.assertEqual(spectrum.accounting()["cubic_coefficient_evaluations"], 1)
            bound = provider.hold_bound(0., .004, .002)
            self.assertEqual(provider.hold_bound(.004, .01, .007), bound)
            self.assertLessEqual(bound, 1.)
            _, derivative = spectrum.bounds(0., .02)
            normalized = max(Fraction(value) / Fraction(tolerance)
                             for value, tolerance in zip(derivative, provider.port_absolute_tolerances))
            omega = Fraction(max(spectrum.angular_frequencies_rad_per_second))
            radius = max(Fraction(.005), Fraction(.01) - Fraction(.005))
            self.assertGreaterEqual(Fraction(bound), normalized * omega**3 * radius**4 / 24)
            for time in (0., .002, .005, .009, math.nextafter(.01, 0.)):
                exact, cubic = spectrum.evaluate(time), polynomial.sample_at(time).numeric_array
                allowance = np.asarray(provider.port_absolute_tolerances) * bound + 2e-14
                self.assertTrue(np.all(np.abs(exact - cubic) <= allowance))
            next_polynomial = provider(.012)
            self.assertIsNot(next_polynomial, polynomial)
            self.assertEqual(spectrum.accounting()["cubic_coefficient_evaluations"], 2)
        raw = Mock(spec=RawWorldTime)
        raw.metadata = {"world_id": "existing_port_cubic_fixture"}
        raw.breakpoints.return_value = np.array([0., .02])
        raw.bind_contact_factors.return_value.derivative_bound_volts_per_second.return_value = 0.
        actions = (PhysicalAction(0., .01, 0, "common", None, ((0, 1e-12),)),
                   PhysicalAction(.01, .02, 1, "common", None, ((1, -1e-12),)))
        controls = dict(epsilon_input_volts=1e-8, max_intervals=2, max_work=20,
                        check=lambda: None, noise_hold_bound=provider.hold_bound)
        eager = compile_midpoint_hold(raw, object(), actions, **controls)
        streamed = compile_streaming_midpoint_hold(raw, object(), lambda: iter(actions),
            support=(0., .02), **controls)
        self.assertEqual(eager.intervals, tuple(streamed.interval_factory()))
        self.assertEqual(len(eager.intervals), 2)
        self.assertEqual(sum(bool(interval.injected_charge_coulomb) for interval in eager.intervals), 2)
        self.assertIsInstance(provider, SpectralCircuitNoise)
        self.assertEqual(provider.sample_numeric_bytes, polynomial.sample_numeric_bytes)

    def test_unfit_original_cell_refuses_before_coefficients_and_replay_or_work_drift_refuses(self):
        spec, spectrum, provider = fixture(cell_for_key=lambda key: (0., .02, .01))
        # Same frozen path/physical support, deliberately tighter input errors.
        impossible = SpectralCubicCircuitNoise(layout=spec, spectrum=spectrum,
            port_absolute_tolerances=(1e-12,) * spec.numeric_value_count,
            cell_for_key=lambda key: (0., .02, .01), check=lambda: None)
        with self.assertRaisesRegex(ValueError, "original cubic noise cell"):
            impossible.hold_bound(0., .0001, .00005)
        with self.assertRaises(ValueError):
            impossible(.005)
        self.assertEqual(spectrum.accounting()["cubic_coefficient_evaluations"], 0)
        cell = [(0., .01, .005)]
        _, _, changing = fixture(cell_for_key=lambda key: cell[0])
        changing(.002)
        cell[0] = (0., .01, .004)
        with self.assertRaisesRegex(ValueError, "changed"):
            changing(.003)
        _, _, bounded = fixture(evaluation_cap=1)
        self.assertIs(bounded(.002), bounded(.008))
        with self.assertRaisesRegex(ValueError, "evaluation/work"):
            bounded(.012)
        for mapping in (lambda key: (0., .03, .005), lambda key: (0., .01, .01),
                        lambda key: [0., .01, .005]):
            _, _, invalid = fixture(cell_for_key=mapping)
            with self.assertRaises(ValueError):
                invalid(.002)
        _, _, valid = fixture()
        with self.assertRaisesRegex(ValueError, "cross"):
            valid.hold_bound(.009, .012, .0105)
        for key in (0., .02, math.nan):
            with self.assertRaises(ValueError):
                valid(key)


if __name__ == "__main__":
    unittest.main()
