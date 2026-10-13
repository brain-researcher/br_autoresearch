"""Noise-forcing bounds and unchanged keys through both real hold compilers.

Port packing is a stub here; real contact/ADC paths have separate integration
checks. These constants describe arithmetic fixtures, not device parameters.
"""
from fractions import Fraction
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from circuit_noise_provider import SpectralCircuitNoise
from physical_noise_ports import PhysicalNoiseLayout
from raw_input_hold import PhysicalAction, HoldResolutionError, compile_midpoint_hold
from raw_world_time import RawWorldTime
from spectral_noise import LatentQuadrature, SpectralNoise, SpectralNoiseLimits
from streaming_midpoint_hold import compile_streaming_midpoint_hold


def provider():
    layout = Mock(spec=PhysicalNoiseLayout)
    layout.port_ids, layout.port_units = ("branch_current", "series_voltage"), ("A", "V")
    layout.sample_from_values.side_effect = lambda values: tuple(values)
    spectrum = SpectralNoise(frequencies_hz=(1.,), quadrature_widths_hz=(1.,),
        port_ids=layout.port_ids, port_units=layout.port_units, latent_ids=("shared",),
        sparse_terms=((0, 0, 0, 1.), (0, 1, 0, 2j)),
        quadratures=(LatentQuadrature(0, "shared", 1., 0.),),
        time_window_seconds=(0., 1.),
        limits=SpectralNoiseLimits(2, 1, 1, 2, 4096, 100, 1000), check=lambda: None)
    return SpectralCircuitNoise(layout=layout, spectrum=spectrum,
        port_absolute_tolerances=(.2, .3), check=lambda: None)


def field_fixture():
    raw = Mock(spec=RawWorldTime)
    raw.metadata = {"world_id": "physical_noise_bound_fixture"}
    raw.breakpoints.return_value = np.array([0., 1.])
    raw.bind_contact_factors.return_value.derivative_bound_volts_per_second.return_value = 0.
    actions = (PhysicalAction(0., 1., 0, "fixture_reference", None, ((0, 1e-12),)),)
    return raw, actions


class CircuitNoiseProviderTests(unittest.TestCase):
    def test_original_port_signs_replay_and_unit_specific_analytic_hold_bound(self):
        noise = provider()
        np.testing.assert_allclose(noise(.25), (0., -2.), atol=2e-15)
        self.assertEqual(noise(.125), noise(.125))
        before = noise.spectrum.accounting()["evaluations"]
        bound = noise.hold_bound(.1, .2, .15)
        self.assertGreaterEqual(bound, .05 * 4 * math.pi / .3)
        self.assertLess(bound, .05 * 4 * math.pi / .3 + 1e-12)
        self.assertEqual(noise.spectrum.accounting()["evaluations"], before)
        # Cache only the global normalized bound: each query still validates
        # support, but cannot revisit all port/tolerance divisions.
        with patch("circuit_noise_provider._upper_div", side_effect=AssertionError("must be cached")), \
                patch.object(noise.spectrum, "bounds", wraps=noise.spectrum.bounds) as support:
            cached = noise.hold_bound(.1, .2, .15)
            support.assert_called_once_with(.1, .2)
            with self.assertRaises(ValueError):
                noise.hold_bound(-.1, .2, .05)
        _, derivative = noise.spectrum.bounds(0., 1.)
        radius = max(Fraction(.15) - Fraction(.1), Fraction(.2) - Fraction(.15))
        exact_bound = max(radius * Fraction(value) / Fraction(tolerance)
                          for value, tolerance in zip(derivative, (.2, .3)))
        self.assertGreaterEqual(Fraction(cached), exact_bound)
        self.assertEqual(noise.metadata["normalized_derivative_cache_numeric_bytes"], 8)
        with self.assertRaises(ValueError):
            noise.hold_bound(0., .5, .5)
        wrong = Mock(spec=PhysicalNoiseLayout)
        wrong.port_ids, wrong.port_units = noise.layout.port_ids, ("V", "A")
        with self.assertRaises(ValueError):
            SpectralCircuitNoise(layout=wrong, spectrum=noise.spectrum,
                port_absolute_tolerances=(.2, .3), check=lambda: None)

    def test_both_hold_compilers_refine_noise_and_preserve_keys_and_once_only_charge(self):
        raw, actions = field_fixture()
        noise = provider()
        common = dict(epsilon_input_volts=1., max_intervals=256, max_work=1000,
                      check=lambda: None, noise_hold_bound=noise.hold_bound)
        eager = compile_midpoint_hold(raw, object(), actions, **common)
        streamed = compile_streaming_midpoint_hold(raw, object(), lambda: iter(actions),
                                                   support=(0., 1.), **common)
        replay = tuple(streamed.interval_factory())
        self.assertEqual(eager.intervals, replay)
        self.assertEqual(replay, tuple(streamed.interval_factory()))
        self.assertGreater(len(replay), 1)  # The field alone is exactly constant.
        self.assertEqual(sum(bool(value.injected_charge_coulomb) for value in replay), 1)
        for interval in replay:
            self.assertEqual(interval.field_key, interval.start + (interval.end - interval.start) / 2)
            self.assertLessEqual(noise.hold_bound(interval.start, interval.end, interval.field_key), 1.)
            at_midpoint = np.array(noise(interval.field_key))
            # Explicit continuous forcing checks are independent of planner bounds.
            for time in (interval.start, interval.end):
                exact = np.array([math.cos(2 * math.pi * time), -2 * math.sin(2 * math.pi * time)])
                self.assertTrue(np.all(np.abs(exact - at_midpoint) <= (.2, .3)))
        self.assertEqual(eager.report["maximum_analytic_hold_bound_volts"], 0.)
        self.assertEqual(noise.spectrum.accounting()["evaluations"], len(replay))
        raw.bind_contact_factors.return_value.volts.assert_not_called()

    def test_noise_bounds_obey_work_caps_and_refuse_changed_replay_or_invalid_errors(self):
        raw, actions = field_fixture()
        common = dict(epsilon_input_volts=1., max_intervals=256, max_work=1000, check=lambda: None)
        with self.assertRaises(HoldResolutionError):
            compile_midpoint_hold(raw, object(), actions,
                **{**common, "max_work": 2}, noise_hold_bound=provider().hold_bound)
        for value in (-1., math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compile_midpoint_hold(raw, object(), actions, **common,
                                      noise_hold_bound=lambda a, b, m: value)
        scale = [100.]
        plan = compile_streaming_midpoint_hold(raw, object(), lambda: iter(actions),
            support=(0., 1.), **common, noise_hold_bound=lambda a, b, m: scale[0] * (b - a))
        scale[0] = 0.
        with self.assertRaises(ValueError):
            tuple(plan.interval_factory())


if __name__ == "__main__":
    unittest.main()
