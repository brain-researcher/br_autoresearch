"""Three focused ownership/port/replay checks; no saved draw or solver sweep."""
from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np

from array_acquisition import ArrayAcquisition, _joint_input_digest, _provider_noise
from circuit_noise_provider import SpectralCubicCircuitNoise
from conditional_acquisition_recipe import conditional_noise_cell
from daq_mux_hold import DAQGroupDrive
from frozen_cubic_noise_binding import (FrozenCubicNoiseBinding, FrozenOriginalCellClock,
                                        bind_frozen_cubic_noise)
from physical_noise_ports import PhysicalNoiseLayout
from spectral_noise import LatentQuadrature, SpectralNoise, SpectralNoiseLimits
from streaming_array_acquisition import StreamingArrayAcquisition
from test_physical_noise_ports import layout as small_layout
from test_polynomial_acquisition import (fixture as array_fixture, eager_arguments, stream_arguments,
                                          WORK_FIELDS)
from test_conditioned_cpe_array import ADC, REF, selections


def provider(layout, clock, *, port=0, frequency=.001, factor=.0001):
    spectrum = SpectralNoise(frequencies_hz=(frequency,), quadrature_widths_hz=(1.,),
        port_ids=layout.port_ids, port_units=layout.port_units, latent_ids=("fixed",),
        sparse_terms=((0, port, 0, factor),),
        quadratures=(LatentQuadrature(0, "fixed", .7, -.25),),
        time_window_seconds=(0., 16.),
        limits=SpectralNoiseLimits(1, 1, 1, 100000, 8 * 1024**2, 100, 1000), check=lambda: None)
    return SpectralCubicCircuitNoise(layout=layout, spectrum=spectrum,
        port_absolute_tolerances=(.001,) * layout.numeric_value_count,
        cell_for_key=clock, check=lambda: None)


def bind(original, clock, **changes):
    arguments = dict(clock=clock, max_numeric_bytes=32 * 1024**2, check=lambda: None)
    arguments.update(changes)
    return bind_frozen_cubic_noise(original, **arguments)


class FrozenCubicNoiseBindingTests(unittest.TestCase):
    def test_owned_arrays_and_original_clock_survive_caller_mutation_without_recentering(self):
        clock = FrozenOriginalCellClock(4000, 8000, 5, 2000000)
        keys = (.00000125, .00399875, .00400125, 8.00000125, 15.99999875)
        for key in keys:
            self.assertEqual(clock(key), conditional_noise_cell(key))
        for boundary in (0., .004, 16.):
            with self.assertRaises(ValueError):
                clock(boundary)
        original = provider(small_layout(), clock)
        expected = original.spectrum.cubic_coefficients(clock(keys[0])[2]).copy()
        owned = bind(original, clock)
        self.assertIs(type(owned), FrozenCubicNoiseBinding)
        before = owned.spectrum.accounting()["cubic_coefficient_evaluations"]
        # Read-only flags alone on the old source could be toggled. The new
        # bytes-owned numerical snapshot is independent of that entire source.
        original.spectrum._amplitude_real.flags.writeable = True
        original.spectrum._amplitude_real[:] += 9.
        original.spectrum._omega_array.flags.writeable = True
        original.spectrum._omega_array[:] *= 3.
        first = owned(keys[0])
        np.testing.assert_array_equal(first.numeric_array, expected)
        self.assertEqual(owned.spectrum.accounting()["cubic_coefficient_evaluations"], before + 1)
        self.assertIs(owned(keys[0] + .0000001), first)
        self.assertEqual((first.support_start_seconds, first.support_end_seconds, first.origin_seconds), clock(keys[0]))
        for name in ("_omega_array", "_frequency_indices", "_port_indices", "_amplitude_real", "_amplitude_imag"):
            with self.assertRaises(ValueError):
                getattr(owned.spectrum, name).flags.writeable = True
        with self.assertRaises(AttributeError):
            owned.spectrum._omega = (9.,)
        with self.assertRaises(AttributeError):
            del owned.spectrum._amplitude_real
        with self.assertRaises(FrozenInstanceError):
            clock.slot_ticks = 6
        # Binding a source already changed behind its tuple metadata refuses.
        with self.assertRaisesRegex(ValueError, "evaluation arrays"):
            bind(original, clock)

    def test_descriptor_preserves_actual_layout_reference_support_and_zero_branch_obligations(self):
        array, layout, _ = array_fixture()
        clock = FrozenOriginalCellClock(1, 20000, 20000, 10000)
        original = provider(layout, clock, port=layout.port_ids.index("external_i"))
        owned = bind(original, clock)
        self.assertIs(type(owned), FrozenCubicNoiseBinding)
        phase = selections("G1", .4)
        descriptor = owned.interval_descriptor(array, 1.004, 1., 1.005, phase)
        actual = owned(1.004)
        array.validate_noise_interval(actual, daq_selected_sources=phase)
        field = np.zeros((64, 64))
        self.assertEqual(owned.joint_digest(field, descriptor), owned.joint_digest(field, actual))
        self.assertEqual(descriptor.sample_numeric_bytes, 32 * layout.numeric_value_count + 24)
        bad_phase = {**phase, ADC: DAQGroupDrive("G1", (("G1", .4),), REF + "_wrong")}
        for operation in (lambda: owned.interval_descriptor(array, 1.004, 1., 1.005, bad_phase),
                          lambda: array.validate_noise_interval(actual, daq_selected_sources=bad_phase)):
            with self.assertRaises(ValueError):
                operation()
        with self.assertRaises(ValueError):
            owned.interval_descriptor(array, 1.004, 1., 2.01, phase)
        other, _, _ = array_fixture()
        with self.assertRaises(ValueError):
            owned.interval_descriptor(other, 1.004, 1., 1.005, phase)
        with patch.object(array, "_analog_conditioning", None):
            with self.assertRaises(ValueError):
                owned.interval_descriptor(array, 1.004, 1., 1.005, phase)
        # Actual zero-off conductance requires an identically zero original
        # waveform for this optimization. Nonzero paths retain exhaustive
        # phase checking rather than acquiring a certificate for an open path.
        branches = tuple(replace(v, off_conductance_siemens=0.) for v in layout.external_daq_branches)
        zero_off = PhysicalNoiseLayout(state_ids=layout.state_ids, voltage_reference_id=layout.voltage_reference_id,
            nodal_current_ports=layout.nodal_current_ports, contact_stars=layout.contact_stars,
            contact_ownership=layout.contact_ownership, contact_noise_role=layout.contact_noise_role,
            amplifier_input_noise_role=layout.amplifier_input_noise_role,
            excluded_retained_contact_components=layout.excluded_retained_contact_components,
            external_daq_branches=branches, external_daq_current_ports=layout.external_daq_current_ports,
            limits=layout.limits)
        nonzero = provider(zero_off, clock, port=zero_off.port_ids.index("external_i"))
        self.assertIs(bind(nonzero, clock), nonzero)
        absent = provider(zero_off, clock)
        self.assertIs(type(bind(absent, clock)), FrozenCubicNoiseBinding)
        forbidden = provider(layout, clock, port=layout.port_ids.index("contact:0000:star_access_voltage"))
        self.assertIs(bind(forbidden, clock), forbidden)
        huge = provider(layout, clock, frequency=1., factor=1e301)
        self.assertIs(bind(huge, clock), huge)
        self.assertIs(bind(original, clock, max_numeric_bytes=1), original)

    def test_preflight_and_replay_skip_coefficients_but_actual_use_is_unchanged_and_fallback_exhaustive(self):
        array, layout, windows = array_fixture()
        clock = FrozenOriginalCellClock(1, 20000, 20000, 10000)
        original = provider(layout, clock)
        owned = bind(original, clock)
        eager_runner = ArrayAcquisition(**eager_arguments(array, windows, owned))
        streamed, stream_layout, stream_windows = array_fixture()
        stream_owned = bind(provider(stream_layout, clock), clock)
        raw, observations = [], []
        stream_runner = StreamingArrayAcquisition(**stream_arguments(streamed, stream_windows, stream_owned,
            raw=raw, observations=observations))
        self.assertEqual(owned.spectrum.accounting()["cubic_coefficient_evaluations"], 0)
        self.assertEqual(stream_owned.spectrum.accounting()["cubic_coefficient_evaluations"], 0)
        field = np.zeros((64, 64))
        event = windows[0].physical_intervals[0].interval
        arguments = dict(array=array, provider=owned, layout=layout, key=event.field_key,
            daq_selected_sources=event.daq_selected_sources, interval_start=event.start,
            interval_end=event.end, admitted_numeric_bytes=owned.sample_numeric_bytes)
        descriptor = _provider_noise(**arguments, verify_only=True)
        verified = _joint_input_digest(field, descriptor, noise_provider=owned)
        actual = _provider_noise(**arguments)
        self.assertEqual(owned.spectrum.accounting()["cubic_coefficient_evaluations"], 1)
        np.testing.assert_array_equal(actual.numeric_array, original(event.field_key).numeric_array)
        self.assertEqual(verified, _joint_input_digest(field, actual, noise_provider=owned))
        # Two prior passes plus window verification consumed no coefficients.
        _provider_noise(**arguments, verify_only=True)
        self.assertEqual(owned.spectrum.accounting()["cubic_coefficient_evaluations"], 1)
        with self.assertRaises(ValueError):
            _provider_noise(**{**arguments, "admitted_numeric_bytes": owned.sample_numeric_bytes - 1}, verify_only=True)
        with self.assertRaises(ValueError):
            owned.joint_digest(field, original(event.field_key))
        ordinary_array, ordinary_layout, ordinary_windows = array_fixture()
        ordinary = provider(ordinary_layout, lambda key: clock(key))
        self.assertIs(bind(ordinary, clock), ordinary)
        ordinary_runner = ArrayAcquisition(**eager_arguments(ordinary_array, ordinary_windows, ordinary))
        self.assertGreater(ordinary.spectrum.accounting()["cubic_coefficient_evaluations"], 0)
        # Same existing five electrical/aperture/storage segments, including
        # one boundary charge and delayed publication: execute both optimized
        # paths and the unchanged ordinary callback to the same final state.
        physical = []
        advance = array.advance_row
        def capture(**kwargs):
            result = advance(**kwargs)
            physical.append(result)
            return result
        with patch.object(array, "advance_row", side_effect=capture):
            eager_report = eager_runner.run()
        stream_report = stream_runner.run()
        ordinary_report = ordinary_runner.run()
        np.testing.assert_array_equal(array.volts, streamed.volts)
        np.testing.assert_array_equal(array.volts, ordinary_array.volts)
        self.assertEqual(tuple(raw), eager_report["raw_values"])
        self.assertEqual(tuple(observations), eager_report["observations"])
        self.assertEqual(eager_report["raw_values"], ordinary_report["raw_values"])
        self.assertEqual(eager_report["observations"], ordinary_report["observations"])
        self.assertEqual(eager_report["split_intervals_executed"], 5)
        for name in WORK_FIELDS:
            self.assertAlmostEqual(stream_report["physical_work"][name],
                sum(getattr(result, name) for result in physical), places=14)
        self.assertNotEqual(stream_report["physical_work"]["noise_current_source_interval_work_joule"], 0.)
        self.assertEqual(stream_owned.spectrum.accounting()["cubic_coefficient_evaluations"], 1)
        with self.assertRaises(ValueError):
            bind_frozen_cubic_noise(original, clock=clock, max_numeric_bytes=0, check=lambda: None)


if __name__ == "__main__":
    unittest.main()
