"""Three analytic source-free checks; no noise calibration or production draw."""
from dataclasses import replace
from fractions import Fraction
import math
import unittest
from unittest import mock

import numpy as np

from spectral_noise import (LatentQuadrature, SpectralNoise, SpectralNoiseLimits,
                            draw_standard_normal_quadratures)


def limits(**changes):
    values = dict(max_terms=20, max_draws=20, max_frequencies=4, max_ports=4,
                  max_numeric_bytes=8192, max_evaluations=200, max_term_evaluations=2000)
    values.update(changes)
    return SpectralNoiseLimits(**values)


def configuration():
    return dict(frequencies_hz=(1., 3.), quadrature_widths_hz=(.25, .5),
                port_ids=("current", "voltage"), port_units=("A", "V"),
                latent_ids=("independent_0", "independent_1"),
                sparse_terms=((0, 0, 0, 1 + 2j), (0, 1, 0, 3 - 1j),
                              (1, 0, 1, .5j), (1, 1, 1, 2.)),
                quadratures=(LatentQuadrature(0, "independent_0", 0., 0.),
                             LatentQuadrature(1, "independent_1", 0., 0.)),
                time_window_seconds=(0., 2.), limits=limits(), check=lambda: None)


class SpectralNoiseTests(unittest.TestCase):
    def test_one_sided_quadrature_normalization_and_complex_cross_covariance_sign(self):
        selected = configuration()
        time, lag = .17, .09
        actual = np.zeros((2, 2))
        # Identity quadrature basis integrates Gaussian second moments exactly:
        # E[a_i a_j]=E[b_i b_j]=delta_ij; all mixed moments vanish.
        for draw_index in range(2):
            for component in ("a", "b"):
                draws = list(selected["quadratures"])
                draws[draw_index] = replace(draws[draw_index], **{component: 1.})
                noise = SpectralNoise(**{**selected, "quadratures": tuple(draws)})
                actual += np.outer(noise.evaluate(time + lag), noise.evaluate(time))
        factors = (np.array([1 + 2j, 3 - 1j]), np.array([.5j, 2.]))
        expected = sum(width * np.real(np.outer(factor, factor.conj()) * np.exp(2j * math.pi * frequency * lag))
                       for frequency, width, factor in zip((1., 3.), (.25, .5), factors))
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        self.assertGreater(abs(expected[0, 1] - expected[1, 0]), .1)  # Detects conjugation/sign reversal.
        variance = 0.
        for a, b in ((1., 0.), (0., 1.)):
            one = SpectralNoise(frequencies_hz=(1.,), quadrature_widths_hz=(2.,), port_ids=("x",),
                port_units=("V",), latent_ids=("z",), sparse_terms=((0, 0, 0, 3.),),
                quadratures=(LatentQuadrature(0, "z", a, b),), time_window_seconds=(0., 2.),
                limits=limits(), check=lambda: None)
            variance += one.evaluate(.123)[0] ** 2
        self.assertAlmostEqual(variance, 2 * 3**2, places=12)  # No extra factor two.

    def test_replay_owned_draws_derivative_bounds_and_no_redraw(self):
        selected = configuration()
        basis = ((1, "independent_1"), (0, "independent_0"))
        draws = draw_standard_normal_quadratures(frequency_latent_ids=basis, seed=17,
                                                max_draws=2, check=lambda: None)
        self.assertEqual(draws, draw_standard_normal_quadratures(frequency_latent_ids=tuple(reversed(basis)),
                                                               seed=17, max_draws=2, check=lambda: None))
        supplied = list(draws)
        noise = SpectralNoise(**{**selected, "quadratures": supplied})
        supplied[0] = replace(supplied[0], a=100.)
        with mock.patch.object(np.random, "Generator", side_effect=AssertionError("evaluation may not draw")):
            first = noise.evaluate(.11)
            first[:] = 1000.
            np.testing.assert_array_equal(noise.evaluate(.11), noise.evaluate(.11))
            absolute, derivative = noise.bounds(0., 2.)
            expected_bounds = [Fraction(0), Fraction(0)]
            expected_derivative = [Fraction(0), Fraction(0)]
            for k, p, l, value in noise.sparse_terms:
                draw = next(d for d in noise.quadratures if d.frequency_index == k and d.latent_id == noise.latent_ids[l])
                amplitude = math.sqrt(noise.quadrature_widths_hz[k]) * value * complex(draw.a, draw.b)
                envelope = Fraction(abs(amplitude.real)) + Fraction(abs(amplitude.imag))
                expected_bounds[p] += envelope
                expected_derivative[p] += Fraction(2 * math.pi * noise.frequencies_hz[k]) * envelope
            for p in range(2):
                self.assertGreaterEqual(Fraction(absolute[p]), expected_bounds[p])
                self.assertGreaterEqual(Fraction(derivative[p]), expected_derivative[p])
            for time in (0., .11, .9, 1.99):
                values = noise.evaluate(time)
                self.assertTrue(np.all(np.abs(values) <= np.array(absolute)))
                analytic_derivative = np.zeros(2)
                for k, p, amplitude in noise._amplitudes:
                    omega = 2 * math.pi * noise.frequencies_hz[k]
                    analytic_derivative[p] += -omega * (amplitude.real * math.sin(omega * time)
                                                        + amplitude.imag * math.cos(omega * time))
                self.assertTrue(np.all(np.abs(analytic_derivative) <= np.array(derivative)))
            with self.assertRaises(AttributeError):
                noise.port_ids = ("mutated",)
        self.assertEqual(noise.quadratures, draws)
        self.assertFalse(noise.accounting()["continuous_white_noise"])
        # Keep the old scalar fsum path as an independent evaluation oracle,
        # including multiple latent factors coalesced at one port/frequency.
        coalesced = SpectralNoise(**{**selected,
            "sparse_terms": selected["sparse_terms"] + ((0, 0, 1, -.5 + .75j),),
            "quadratures": draws + (LatentQuadrature(0, "independent_1", .75, -.5),)})
        for model in (noise, coalesced):
            packed = (model._omega_array, model._frequency_indices, model._port_indices,
                      model._amplitude_real, model._amplitude_imag)
            self.assertEqual(sum(array.nbytes for array in packed),
                             model.accounting()["packed_evaluation_numeric_bytes"])
            self.assertTrue(all(not array.flags.writeable for array in packed))
            for time in (0., .11, .9, 1.99):
                actual = model.evaluate(time)
                contributions, magnitudes = [[], []], [[], []]
                for k, p, amplitude in model._amplitudes:
                    phase = 2 * math.pi * model.frequencies_hz[k] * time
                    real = amplitude.real * math.cos(phase)
                    imaginary = amplitude.imag * math.sin(phase)
                    contributions[p].append(real - imaginary)
                    magnitudes[p].extend((abs(real), abs(imaginary)))
                for p in range(2):
                    # Explicit binary64 evaluation/reduction allowance, not a
                    # change to the analytic trajectory or its hold tolerance.
                    tolerance = 16 * np.finfo(np.float64).eps * math.fsum(magnitudes[p])
                    self.assertLessEqual(abs(actual[p] - math.fsum(contributions[p])), tolerance)
            before = model.accounting()
            origin = .11
            cubic = model.cubic_coefficients(origin)
            self.assertEqual(cubic.shape, (4, 2))
            for order in range(4):
                for port in range(2):
                    terms = []
                    for k, p, amplitude in model._amplitudes:
                        if p == port:
                            omega = model.angular_frequencies_rad_per_second[k]
                            rotation = complex(math.cos(omega * origin), math.sin(omega * origin))
                            terms.append(((1j * omega)**order * amplitude * rotation).real / math.factorial(order))
                    allowance = 32 * np.finfo(np.float64).eps * math.fsum(abs(value) for value in terms)
                    self.assertLessEqual(abs(cubic[order, port] - math.fsum(terms)), allowance)
            after = model.accounting()
            self.assertEqual(after["evaluations"] - before["evaluations"], 1)
            self.assertEqual(after["cubic_coefficient_evaluations"] - before["cubic_coefficient_evaluations"], 1)
            self.assertEqual(after["term_evaluations"] - before["term_evaluations"],
                             len(model.frequencies_hz) + 4 * len(model._amplitudes))
            np.testing.assert_array_equal(cubic[0], model.evaluate(origin))

    def test_finite_caps_invalid_basis_units_and_domain_refuse_without_repair(self):
        selected = configuration()
        variants = (dict(frequencies_hz=(1., 1.)), dict(quadrature_widths_hz=(.25, -1.)),
                    dict(port_units=("A", "uV")), dict(quadratures=selected["quadratures"][:1]),
                    dict(sparse_terms=selected["sparse_terms"] + selected["sparse_terms"][:1]),
                    dict(sparse_terms=((0, 0, 0, complex(math.inf, 0.)),)),
                    dict(limits=limits(max_terms=1)), dict(limits=limits(max_draws=1)),
                    dict(limits=limits(max_frequencies=1)), dict(limits=limits(max_ports=1)),
                    dict(limits=limits(max_numeric_bytes=8)))
        for delta in variants:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                SpectralNoise(**{**selected, **delta})
        noise = SpectralNoise(**{**selected, "limits": limits(max_evaluations=1)})
        noise.evaluate(0.)
        with self.assertRaises(ValueError):
            noise.evaluate(.1)
        self.assertEqual(noise.accounting()["evaluations"], 1)
        noise = SpectralNoise(**{**selected, "limits": limits(max_term_evaluations=1)})
        with self.assertRaises(ValueError):
            noise.evaluate(0.)
        self.assertEqual(noise.accounting()["evaluations"], 0)
        with self.assertRaises(ValueError):
            noise.cubic_coefficients(.1)
        self.assertEqual(noise.accounting()["cubic_coefficient_evaluations"], 0)
        noise = SpectralNoise(**selected)
        # The retained numeric cap includes all packed vector-evaluation arrays.
        exact_bytes = noise.numeric_storage_bytes
        self.assertEqual(SpectralNoise(**{**selected, "limits": limits(max_numeric_bytes=exact_bytes)})
                         .numeric_storage_bytes, exact_bytes)
        with self.assertRaises(ValueError):
            SpectralNoise(**{**selected, "limits": limits(max_numeric_bytes=exact_bytes - 1)})
        for time in (-.1, 2., math.nan):
            with self.assertRaises(ValueError):
                noise.evaluate(time)
        with self.assertRaises(ValueError):
            noise.bounds(.8, .2)
        with self.assertRaises(TimeoutError):
            SpectralNoise(**{**selected, "check": lambda: (_ for _ in ()).throw(TimeoutError("fixture"))})


if __name__ == "__main__":
    unittest.main()
