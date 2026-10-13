"""Physical-noise replay seams; no PSD, hardware recipe or noise fit."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import (AcquisitionExecutionError, AcquisitionLimits, ArrayAcquisition,
                               ArrayInputBinding, PhysicalInterval, ProviderInterval)
from daq_mux_hold import (DAQGroupDrive, ExternalVoltageInput, PassiveDAQFrontend,
                          PassiveMuxHoldGroup, external_daq_branch_id)
from physical_noise_ports import ExternalDAQCurrentPort, NoiseLimits, NodalCurrentPort
from streaming_acquisition_readout import ReadoutWindow, StreamingReadoutLimits
from streaming_array_acquisition import (ArrayWindow, StreamPhysicalInterval,
                                         StreamingArrayAcquisition, StreamingArrayLimits)
from test_cpe_array import fractional_contact, ideal_contact, make_array
from test_sparse_array_rc import controls


ADC = "noise_fixture_0"
KEYS = (.004, .010)
EAGER_LIMITS = AcquisitionLimits(4, 16, 2, 2, 2, 1024 * 1024)
STREAM_LIMITS = StreamingArrayLimits(4, 1, 4, 8, 4, 16, 8, 1024 * 1024, 128, 200000)
READOUT_LIMITS = StreamingReadoutLimits(2, 2, 2, 16, 2, 2, 2, 2, 1024 * 1024, 2)
WORK_FIELDS = ("field_source_interval_work_joule", "noise_current_source_interval_work_joule",
               "noise_access_voltage_source_interval_work_joule", "daq_external_source_interval_work_joule")


def fixture(*, noise=True, role="conditional_fixture_current", external=False, external_off=.02):
    contacts = (fractional_contact(),) + (ideal_contact(),) * 4095
    access = np.zeros((64, 64))
    access[0, 0] = .3
    groups = []
    for index in range(24):
        sources = tuple(range(index, 64, 24))
        groups.append(PassiveMuxHoldGroup(f"noise_fixture_{index}", sources, .7, 0.,
            (.8,) * len(sources), (.03,) * len(sources), .05,
            (ExternalVoltageInput("G1", .6, external_off),) if index == 0 and external else ()))
    array = make_array(contacts=contacts, access_resistance_ohm=access,
        daq_frontend=PassiveDAQFrontend(tuple(groups), "additional_fixed_shunt_excludes_frontend"))
    if noise:
        array.make_noise_layout(nodal_current_ports=(
            NodalCurrentPort("source_i", "source_00", None, role, "input_current:first"),
            NodalCurrentPort("source_j", "source_00", None, role, "input_current:second")),
            contact_noise_role="conditional_finite_band_fixture", amplifier_input_noise_role=None,
            external_daq_current_ports=(ExternalDAQCurrentPort("external_i",
                external_daq_branch_id(ADC, "G1"), ADC, "G1", f"daq_hold_{ADC}",
                "conditional_fixture_switch_noise"),) if external else (),
            excluded_retained_contact_components=(), limits=NoiseLimits(100000, 8 * 1024**2))
    selected = {f"noise_fixture_{index}": None for index in range(24)}
    selected[ADC] = 0
    ref = array.voltage_reference_id
    if external:
        selected[ADC] = DAQGroupDrive("G1", (("G1", .4),), ref)
    first = ProviderInterval(0., .005, 0, KEYS[0], ref, selected, ((0, .03),))
    continued = replace(first, start=.005, end=.008, injected_charge_coulomb=())
    following = DAQGroupDrive(24, (("G1", -.2),), ref) if external else 24
    changed = ProviderInterval(.008, .012, 1, KEYS[1], ref, {**selected, ADC: following}, ())
    transfer = QuantizerTransfer(-1., 1., (-.5, 0., .5), (-.75, -.25, .25, .75))
    raw = RawConversion("r:0", ADC, "hold", "signal", .002, .010, .010, .002, transfer, True)
    returned = ReturnedSample("s:0", ("r:0",), (1.,), 1., .002, .012, None)
    observation = Observation("o:0", "fixture_pixel", "pixel", ("s:0",), (1.,), 1., .002, .015, 16, 0)
    readouts = (ReadoutWindow(0, 0., .005, (raw,), (returned,), (observation,),
                              (("r:0", 1),), (("s:0", 1),)),
                ReadoutWindow(1, .005, .012, (), (), (), (), ()))
    windows = (ArrayWindow(0, 0., .005, (StreamPhysicalInterval(0, first),), readouts[0]),
               ArrayWindow(1, .005, .012, (StreamPhysicalInterval(0, continued),
                                          StreamPhysicalInterval(1, changed)), readouts[1]))
    return array, windows


def field(key):
    values = np.zeros((64, 64))
    values[0, 0] = .2 if key == KEYS[0] else .15
    return values


def values(layout, key, *, canceling_drift=False):
    amplitudes = np.zeros(layout.numeric_value_count)
    named = {"source_i": .0625 if key == KEYS[0] else .03125,
             "contact:0000:star_branch:0": .01,
             "contact:0000:star_branch:1": -.005,
             "contact:0000:star_branch:2": .007,
             "contact:0000:star_access_voltage": .002}
    if "external_i" in layout.port_ids:
        named["external_i"] = .0125 if key == KEYS[0] else -.025
    for name, amplitude in named.items():
        amplitudes[layout.port_ids.index(name)] = amplitude
    if canceling_drift:
        amplitudes[layout.port_ids.index("source_i")] += .125
        amplitudes[layout.port_ids.index("source_j")] -= .125
    return amplitudes


def provider(array, *, calls=None, references=None):
    layout = array.physical_noise_layout
    def sample(key):
        if calls is not None:
            calls.append(key)
        result = layout.sample_from_values(values(layout, key))
        if references is not None:
            references.append(weakref.ref(result))
        return result
    return sample


def eager_arguments(array, windows, noise_provider):
    return dict(array=array,
        physical_intervals=tuple(fragment.interval for window in windows for fragment in window.physical_intervals),
        field_provider=field, noise_provider=noise_provider,
        input_bindings=(ArrayInputBinding("hold", ADC, "hold", None, ADC),),
        conversions=tuple(v for w in windows for v in w.readout_window.conversions),
        returned_samples=tuple(v for w in windows for v in w.readout_window.returned_samples),
        observations=tuple(v for w in windows for v in w.readout_window.observations),
        acquisition_window=(0., .012), final_publication_horizon=.015,
        controls=controls(), limits=EAGER_LIMITS)


def stream_arguments(array, windows, noise_provider, *, raw=None, observations=None):
    return dict(array=array, window_factory=lambda: iter(windows), field_provider=field,
        noise_provider=noise_provider, input_bindings=(ArrayInputBinding("hold", ADC, "hold", None, ADC),),
        acquisition_window=(0., .012), final_publication_horizon=.015, controls=controls(),
        limits=STREAM_LIMITS, readout_limits=READOUT_LIMITS, check=lambda: None,
        on_raw=(lambda _: None) if raw is None else raw.append,
        on_observation=(lambda _: None) if observations is None else observations.append,
        on_interval=lambda _: None)


class PhysicalNoiseAcquisitionTests(unittest.TestCase):
    def test_external_branch_work_survives_mux_and_storage_seams_and_late_zero_path_refuses(self):
        array, windows = fixture(external=True)
        raw, observations, results = [], [], []
        original = array.advance_row
        def capture(**kwargs):
            result = original(**kwargs)
            selection = kwargs["daq_selected_sources"][ADC]
            voltage = dict(selection.external_volts)["G1"]
            conductance = .6 if selection.selected_input == "G1" else .02
            sample = kwargs["noise_sample"]
            current = sample.numeric_values[sample.layout.port_ids.index("external_i")]
            hold = next(value for value in result.daq_hold_observations if value.adc_id == ADC)
            expected = kwargs["duration_seconds"] * voltage * (conductance * (voltage-hold.mean_volts) + current)
            self.assertAlmostEqual(result.daq_external_source_interval_work_joule, expected, places=14)
            results.append(result)
            return result
        runner = StreamingArrayAcquisition(**stream_arguments(array, windows, provider(array),
            raw=raw, observations=observations))
        with patch.object(array, "advance_row", side_effect=capture):
            report = runner.run()
        self.assertEqual(len(results), 5)
        for name in WORK_FIELDS:
            self.assertAlmostEqual(report["physical_work"][name],
                                   sum(getattr(result, name) for result in results), places=14)
        self.assertNotEqual(report["physical_work"]["daq_external_source_interval_work_joule"], 0.)
        eager, eager_windows = fixture(external=True)
        expected = ArrayAcquisition(**eager_arguments(eager, eager_windows, provider(eager))).run()
        np.testing.assert_array_equal(array.volts, eager.volts)
        self.assertEqual(tuple(raw), expected["raw_values"])
        self.assertEqual(tuple(observations), expected["observations"])

        # Only the later event is impossible: its G1 branch is disconnected,
        # while the frozen sample still supplies nonzero switch-path noise.
        invalid, invalid_windows = fixture(external=True, external_off=0.)
        before = invalid.volts, invalid.previous_field_volts
        with patch.object(invalid, "advance_row", wraps=invalid.advance_row) as advance:
            with self.assertRaisesRegex(ValueError, "zero.*conductance"):
                StreamingArrayAcquisition(**stream_arguments(invalid, invalid_windows, provider(invalid)))
            with self.assertRaisesRegex(ValueError, "zero.*conductance"):
                ArrayAcquisition(**eager_arguments(invalid, invalid_windows, provider(invalid)))
            advance.assert_not_called()
        self.assertEqual((invalid.volts, invalid.previous_field_volts), before)

    def test_same_physical_sample_survives_storage_aperture_and_solver_seams_with_work(self):
        array, windows = fixture()
        raw, observations, calls = [], [], []
        physical = []
        original = array.advance_row
        def capture(**kwargs):
            sample = kwargs["noise_sample"]
            result = original(**kwargs)
            physical.append((sample.numeric_values, result))
            return result
        runner = StreamingArrayAcquisition(**stream_arguments(array, windows,
            provider(array, calls=calls), raw=raw, observations=observations))
        self.assertEqual(calls, [KEYS[0], KEYS[0], KEYS[1]])
        self.assertEqual((array.resources.base_states, array.resources.daq_hold_states), (4160, 24))
        with patch.object(array, "advance_row", side_effect=capture):
            runner.advance_window()
            self.assertFalse(raw)
            self.assertFalse(observations)
            self.assertEqual(calls, [KEYS[0], KEYS[0], KEYS[1], KEYS[0], KEYS[0]])
            report = runner.run()
        self.assertEqual(calls, [KEYS[0], KEYS[0], KEYS[1], KEYS[0], KEYS[0],
                                  KEYS[0], KEYS[1], KEYS[0], KEYS[1]])
        self.assertEqual(report["physical_input_events"], 2)
        self.assertEqual(report["physical_fragments"], 3)
        self.assertEqual(report["actual_internal_conversions"], 1)
        self.assertEqual(report["hardware_returned_samples"], 1)
        self.assertEqual(report["retained_pixel_observations"], 1)
        self.assertEqual(report["live_physical_noise_numeric_bytes"], array.physical_noise_layout.sample_numeric_bytes)
        self.assertEqual(len(physical), 5)  # 0,.002,.005,.008,.010,.012
        self.assertTrue(all(sample == physical[0][0] for sample, _ in physical[:3]))
        self.assertTrue(all(sample == physical[3][0] for sample, _ in physical[3:]))
        for key in WORK_FIELDS:
            self.assertAlmostEqual(report["physical_work"][key], sum(getattr(result, key) for _, result in physical), places=14)
        self.assertNotEqual(report["physical_work"]["noise_current_source_interval_work_joule"], 0.)
        self.assertNotEqual(report["physical_work"]["noise_access_voltage_source_interval_work_joule"], 0.)
        eager, eager_windows = fixture()
        expected = ArrayAcquisition(**eager_arguments(eager, eager_windows, provider(eager))).run()
        np.testing.assert_array_equal(array.volts, eager.volts)
        self.assertEqual(tuple(raw), expected["raw_values"])
        self.assertEqual(tuple(observations), expected["observations"])
        self.assertEqual(report["propagation_operator_products"], expected["propagation_operator_products"])

    def test_late_bad_layout_or_sample_and_canceling_replay_drift_refuse_before_mutation(self):
        array, windows = fixture()
        foreign, _ = fixture(role="different_owned_role")
        saved = array.volts, array.previous_field_volts
        for problem in ("foreign_layout", "late_nonfinite"):
            def bad(key):
                layout = array.physical_noise_layout if key == KEYS[0] else foreign.physical_noise_layout
                supplied = values(layout, key)
                if problem == "late_nonfinite" and key == KEYS[1]:
                    supplied[0] = np.nan
                return layout.sample_from_values(supplied)
            with self.subTest(problem=problem), patch.object(array, "advance_row", wraps=array.advance_row) as advance:
                with self.assertRaises(ValueError):
                    StreamingArrayAcquisition(**stream_arguments(array, windows, bad))
                with self.assertRaises(ValueError):
                    ArrayAcquisition(**eager_arguments(array, windows, bad))
                advance.assert_not_called()
            self.assertEqual((array.volts, array.previous_field_volts), saved)
        # These two original current sources cancel exactly after projection.
        # A digest of only the nodal vector would lose the changed physical ports.
        layout = array.physical_noise_layout
        normal = layout.sample_from_values(values(layout, KEYS[1]))
        changed = layout.sample_from_values(values(layout, KEYS[1], canceling_drift=True))
        np.testing.assert_array_equal(layout.project_current(normal), layout.project_current(changed))
        drifting = [False]
        def replay(key):
            return layout.sample_from_values(values(layout, key, canceling_drift=drifting[0] and key == KEYS[1]))
        runner = StreamingArrayAcquisition(**stream_arguments(array, windows, replay))
        runner.advance_window()
        drifting[0] = True
        saved = array.volts, array.previous_field_volts
        with patch.object(array, "advance_row", wraps=array.advance_row) as advance:
            with self.assertRaises(AcquisitionExecutionError):
                runner.advance_window()
            advance.assert_not_called()
        self.assertEqual((array.volts, array.previous_field_volts), saved)
        self.assertEqual(runner.failure["phase"], "window replay verification")
        with self.assertRaises(AcquisitionExecutionError):
            runner.run()
        # A provider may also change after the window itself was verified.
        fresh, fresh_windows = fixture()
        counts = {KEYS[0]: 0, KEYS[1]: 0}
        def at_use(key):
            counts[key] += 1
            return fresh.physical_noise_layout.sample_from_values(values(fresh.physical_noise_layout, key,
                canceling_drift=key == KEYS[0] and counts[key] == 4))
        fresh_runner = StreamingArrayAcquisition(**stream_arguments(fresh, fresh_windows, at_use))
        before = fresh.volts
        with patch.object(fresh, "advance_row", wraps=fresh.advance_row) as advance:
            with self.assertRaises(AcquisitionExecutionError):
                fresh_runner.advance_window()
            advance.assert_not_called()
        self.assertEqual(fresh.volts, before)

    def test_single_live_sample_numeric_cap_and_legacy_none_path(self):
        array, windows = fixture()
        references = []
        args = eager_arguments(array, windows, provider(array, references=references))
        base_bytes = 8 * len(array.state_ids) + 3 * 32 + 16
        noise_bytes = array.physical_noise_layout.sample_numeric_bytes
        with self.assertRaisesRegex(ValueError, "numeric-byte"):
            ArrayAcquisition(**{**args, "limits": replace(EAGER_LIMITS, max_stored_plan_numeric_bytes=base_bytes)})
        self.assertFalse(references)  # The amplitude count is known before any provider call.
        runner = ArrayAcquisition(**args)
        self.assertTrue(all(ref() is None for ref in references))
        runner.advance_next()
        self.assertEqual(sum(ref() is not None for ref in references), 1)
        self.assertIsNotNone(runner._materialized_noise)
        report = runner.run()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertIsNone(runner._materialized_noise)
        self.assertEqual(report["stored_plan_numeric_bytes"], base_bytes + noise_bytes)
        another, other_windows = fixture()
        streaming_base = 2 * 32 + 8 * 4096 + 2 * 8 * len(another.state_ids)
        calls = []
        with self.assertRaisesRegex(ValueError, "numeric-byte"):
            StreamingArrayAcquisition(**{**stream_arguments(another, other_windows, provider(another, calls=calls)),
                "limits": replace(STREAM_LIMITS, max_stored_window_numeric_bytes=streaming_base)})
        self.assertFalse(calls)
        valid = StreamingArrayAcquisition(**stream_arguments(another, other_windows, provider(another)))
        self.assertEqual(valid.preflight_summary["maximum_window_numeric_bytes"], streaming_base + noise_bytes)
        # Noise needs an explicit midpoint key; an eager field alone cannot
        # silently choose when the stochastic physical input was sampled.
        intervals = tuple(PhysicalInterval(v.interval.start, v.interval.end, v.interval.gate_row,
            field(v.interval.field_key), v.interval.voltage_reference_id, v.interval.daq_selected_sources,
            np.zeros(len(another.state_ids))) for w in other_windows for v in w.physical_intervals)
        with self.assertRaisesRegex(ValueError, "ProviderInterval"):
            ArrayAcquisition(**{**eager_arguments(another, other_windows, provider(another)),
                                 "physical_intervals": intervals, "field_provider": None})
        plain, plain_windows = fixture(noise=False)
        plain_args = stream_arguments(plain, plain_windows, None)
        implicit = dict(plain_args)
        del implicit["noise_provider"]
        before = plain.volts
        default = StreamingArrayAcquisition(**implicit)
        explicit = StreamingArrayAcquisition(**plain_args)
        self.assertEqual(default._expected_digests, explicit._expected_digests)
        self.assertEqual(default.preflight_summary, explicit.preflight_summary)
        self.assertEqual(plain.volts, before)


if __name__ == "__main__":
    unittest.main()
