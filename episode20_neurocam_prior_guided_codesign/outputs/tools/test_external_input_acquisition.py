"""Three acquisition seams for explicit external ports; no device recipe.

The numerical fixture retains the full array and 24 physical ADC hold states.
Its partition, component values and two conversions are test choices only.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import (AcquisitionExecutionError, AcquisitionLimits, ArrayAcquisition,
                               ArrayInputBinding, ProviderInterval)
from daq_mux_hold import DAQGroupDrive, ExternalVoltageInput
from development_simulation import _AccountingArray
from raw_input_hold import PhysicalAction
from streaming_acquisition_readout import ReadoutWindow, StreamingReadoutLimits
from streaming_array_acquisition import (ArrayWindow, StreamPhysicalInterval, StreamingArrayLimits,
                                         StreamingArrayAcquisition)
import test_cpe_array
from test_daq_mux_hold import frontend, group
from test_sparse_array_rc import controls


REF = "explicit_fixture_reference"
ADC = "fixture_adc_0"
TRANSFER = QuantizerTransfer(-1., 1., (-.5, 0., .5), (-.75, -.25, .25, .75))
LIMITS = StreamingArrayLimits(4, 1, 4, 8, 4, 32, 16, 1024 * 1024, 128, 200000)
READOUT_LIMITS = StreamingReadoutLimits(4, 4, 4, 32, 4, 4, 4, 4, 1024 * 1024, 4)
EAGER_LIMITS = AcquisitionLimits(4, 32, 4, 4, 4, 1024 * 1024)
WORK = "daq_external_source_interval_work_joule"


def make_fixture(*, accounted=False):
    groups = []
    for index in range(24):
        sources = tuple(range(index, 64, 24))
        groups.append(group(adc_id=f"fixture_adc_{index}", source_indices=sources,
            initial_hold_volts=0., track_conductance_siemens=(.8,) * len(sources),
            off_conductance_siemens=(.03,) * len(sources),
            external_inputs=(ExternalVoltageInput("marker", 1.2, .04),) if index == 0 else ()))
    kwargs = dict(daq_frontend=frontend(*groups))
    if accounted:
        def constructor(**parameters):
            return _AccountingArray(total_operator_limit=200000, checkpoint=lambda: None, **parameters)
        with patch.object(test_cpe_array, "NeurocamCPEArray", constructor):
            circuit = test_cpe_array.make_array(**kwargs)
    else:
        circuit = test_cpe_array.make_array(**kwargs)
    base = {f"fixture_adc_{index}": None for index in range(24)}
    marker = {**base, ADC: DAQGroupDrive("marker", (("marker", 1.),), REF)}
    signal = {**base, ADC: DAQGroupDrive(0, (("marker", 1.),), REF)}
    raw = (RawConversion("r:0", ADC, "hold", "marker", .1, .6, .6, .1, TRANSFER, True),
           RawConversion("r:1", ADC, "hold", "signal", .8, .9, .9, .8, TRANSFER, True))
    samples = (ReturnedSample("s:0", ("r:0",), (1.,), 1., .1, .7, None),
               ReturnedSample("s:1", ("r:1",), (1.,), 1., .8, 1.1, None))
    observations = (Observation("o:0", "marker_port", "marker", ("s:0",), (1.,), 1., .1, 1.2, 16, 0),
                    Observation("o:1", "pixel0", "pixel", ("s:1",), (1.,), 1., .8, 1.2, 16, 0))
    readouts = (ReadoutWindow(0, 0., .5, raw[:1], samples[:1], observations[:1],
                              (("r:0", 1),), (("s:0", 1),)),
                ReadoutWindow(1, .5, 1., raw[1:], samples[1:], observations[1:],
                              (("r:1", 1),), (("s:1", 1),)))
    first = ProviderInterval(0., .5, 0, 0, REF, marker, ())
    continued = ProviderInterval(.5, .75, 0, 0, REF, marker, ())
    changed = ProviderInterval(.75, 1., 1, 0, REF, signal, ())
    windows = (ArrayWindow(0, 0., .5, (StreamPhysicalInterval(0, first),), readouts[0]),
               ArrayWindow(1, .5, 1., (StreamPhysicalInterval(0, continued),
                                      StreamPhysicalInterval(1, changed)), readouts[1]))
    bindings = (ArrayInputBinding("hold", ADC, "hold", None, ADC),)
    return circuit, windows, bindings


def streaming_arguments(circuit, windows, bindings, *, raw_sink=None, observation_sink=None):
    return dict(array=circuit, window_factory=lambda: iter(windows),
        field_provider=lambda key: np.zeros((64, 64)), input_bindings=bindings,
        acquisition_window=(0., 1.), final_publication_horizon=1.2, controls=controls(),
        limits=LIMITS, readout_limits=READOUT_LIMITS, check=lambda: None,
        on_raw=(lambda _: None) if raw_sink is None else raw_sink,
        on_observation=(lambda _: None) if observation_sink is None else observation_sink,
        on_interval=lambda _: None)


def eager_arguments(circuit, windows, bindings):
    return dict(array=circuit,
        physical_intervals=tuple(fragment.interval for window in windows for fragment in window.physical_intervals),
        field_provider=lambda key: np.zeros((64, 64)), input_bindings=bindings,
        conversions=tuple(item for window in windows for item in window.readout_window.conversions),
        returned_samples=tuple(item for window in windows for item in window.readout_window.returned_samples),
        observations=tuple(item for window in windows for item in window.readout_window.observations),
        acquisition_window=(0., 1.), final_publication_horizon=1.2, controls=controls(), limits=EAGER_LIMITS)


def replace_late_drive(windows, drive):
    final = windows[1].physical_intervals[1]
    selections = {**final.interval.daq_selected_sources, ADC: drive}
    changed = replace(final, interval=replace(final.interval, daq_selected_sources=selections))
    return (windows[0], replace(windows[1], physical_intervals=(windows[1].physical_intervals[0], changed)))


class ExternalInputAcquisitionTests(unittest.TestCase):
    def test_same_adc_marker_and_signal_keep_history_counts_and_physical_work(self):
        circuit, windows, bindings = make_fixture()
        raw, published = [], []
        runner = StreamingArrayAcquisition(**streaming_arguments(circuit, windows, bindings,
            raw_sink=raw.append, observation_sink=published.append))
        self.assertEqual((circuit.SOURCES, circuit.resources.daq_hold_states,
                          circuit.resources.total_states), (64, 24, 4184))
        self.assertEqual(circuit.resources.daq_switch_branches, 65)  # 64 neural paths + one port, not 65 sources.
        runner.advance_window()
        self.assertFalse(raw)  # Marker aperture continues across the storage seam.
        self.assertGreater(circuit.volts[4160], 0.)
        report = runner.run()
        self.assertEqual(report["actual_internal_conversions"], 2)
        self.assertEqual(report["conversions_by_role"], {"marker": 1, "signal": 1})
        self.assertEqual(report["conversions_by_adc"], {ADC: 2})
        self.assertEqual(report["hardware_returned_samples"], 2)
        self.assertEqual(report["retained_pixel_observations"], 1)
        self.assertEqual(report["physical_input_events"], 2)
        self.assertEqual(report["physical_fragments"], 3)
        self.assertGreater(raw[1].aperture_mean_volts, 0.)  # Signal uses the charged, leaking retained hold.
        self.assertGreater(report["physical_work"][WORK], 0.)
        self.assertEqual(report["physical_work"]["field_source_interval_work_joule"], 0.)
        eager, _, _ = make_fixture(accounted=True)
        expected = ArrayAcquisition(**eager_arguments(eager, windows, bindings)).run()
        np.testing.assert_allclose(circuit.volts, eager.volts, rtol=1e-13, atol=1e-15)
        self.assertEqual(tuple(raw), expected["raw_values"])
        self.assertEqual(tuple(published), expected["observations"])
        self.assertAlmostEqual(report["physical_work"][WORK], eager.accounting()[WORK], places=14)
        self.assertEqual(report["propagation_operator_products"], expected["propagation_operator_products"])
        self.assertEqual(expected["stored_plan_numeric_bytes"], 8 * 4184 + 3 * 32 + 3 * 8)
        self.assertEqual(report["maximum_window_numeric_bytes"], 2 * 32 + 8 * 4096 + 2 * 8 * 4184 + 2 * 8)

    def test_late_reference_missing_off_voltage_and_numeric_caps_refuse_before_advance(self):
        circuit, windows, bindings = make_fixture()
        before = circuit.volts, circuit.previous_field_volts
        for drive in (DAQGroupDrive(0, (("marker", 1.),), "different_reference"),
                      DAQGroupDrive(0, (), REF)):
            changed = replace_late_drive(windows, drive)
            with self.subTest(drive=drive), patch.object(circuit, "advance_row", wraps=circuit.advance_row) as advance:
                with self.assertRaises(ValueError):
                    StreamingArrayAcquisition(**streaming_arguments(circuit, changed, bindings))
                with self.assertRaises(ValueError):
                    ArrayAcquisition(**eager_arguments(circuit, changed, bindings))
                advance.assert_not_called()
            self.assertEqual((circuit.volts, circuit.previous_field_volts), before)
        # Caps that covered all previous numeric payloads now refuse the added
        # external voltages; the other arrays/digests fit these exact bounds.
        with patch.object(circuit, "advance_row", side_effect=AssertionError("preflight advanced")):
            with self.assertRaisesRegex(ValueError, "numeric-byte"):
                StreamingArrayAcquisition(**{**streaming_arguments(circuit, windows, bindings),
                    "limits": replace(LIMITS, max_stored_window_numeric_bytes=2 * 32 + 8 * 4096 + 2 * 8 * 4184)})
            with self.assertRaisesRegex(ValueError, "numeric-byte"):
                ArrayAcquisition(**{**eager_arguments(circuit, windows, bindings),
                    "limits": replace(EAGER_LIMITS, max_stored_plan_numeric_bytes=8 * 4184 + 3 * 32)})
        supplied = [["marker", 1.]]
        mapping = {ADC: DAQGroupDrive("marker", supplied, REF)}
        action = PhysicalAction(0., 1., 0, REF, mapping)
        supplied[0][1] = 3.
        mapping.clear()
        self.assertEqual(action.daq_selected_sources[ADC].external_volts, (("marker", 1.),))
        for drive in (DAQGroupDrive("marker", (("marker", 1.),), "different_reference"),
                      DAQGroupDrive(64, (("marker", 1.),), REF)):
            with self.assertRaises(ValueError):
                PhysicalAction(0., 1., 0, REF, {ADC: drive})

    def test_changed_external_value_is_bound_in_replay_before_affected_window(self):
        circuit, windows, bindings = make_fixture()
        calls = []
        def factory():
            calls.append(None)
            selected = windows if len(calls) == 1 else replace_late_drive(
                windows, DAQGroupDrive(0, (("marker", .9),), REF))
            return iter(selected)
        runner = StreamingArrayAcquisition(**{**streaming_arguments(circuit, windows, bindings),
                                               "window_factory": factory})
        runner.advance_window()
        saved = circuit.volts, circuit.previous_field_volts
        with patch.object(circuit, "advance_row", wraps=circuit.advance_row) as advance:
            with self.assertRaises(AcquisitionExecutionError):
                runner.advance_window()
            advance.assert_not_called()
        self.assertEqual((circuit.volts, circuit.previous_field_volts), saved)
        self.assertEqual(runner.physical_time, .5)
        self.assertEqual(runner.failure["phase"], "window replay verification")
        self.assertGreater(runner.failure["partial_accounting"]["physical_work"][WORK], 0.)
        with self.assertRaises(AcquisitionExecutionError):
            runner.run()


if __name__ == "__main__":
    unittest.main()
