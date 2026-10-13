"""Four bounded seam/ownership checks; no production recipes or payloads."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import (AcquisitionExecutionError, AcquisitionLimits, ArrayAcquisition,
                               ArrayInputBinding, PhysicalInterval, ProviderInterval)
from streaming_acquisition_readout import ReadoutWindow, StreamingReadoutLimits
from streaming_array_acquisition import (ArrayWindow, StreamPhysicalInterval, StreamingArrayLimits,
                                         StreamingArrayAcquisition, _window_digest)
from test_cpe_array import make_array
from test_daq_mux_hold import frontend, group
from test_sparse_array_rc import controls
from observation_nuisance import ObservationNuisanceBinding
from test_observation_nuisance import nuisance_fixture


TRANSFER = QuantizerTransfer(-1., 1., (-.5, 0., .5), (-.75, -.25, .25, .75))
LIMITS = StreamingArrayLimits(8, 2, 8, 16, 4, 32, 16, 1024 * 1024, 256, 200000)
READOUT_LIMITS = StreamingReadoutLimits(8, 8, 8, 32, 8, 4, 4, 4, 1024 * 1024, 4)


def make_fixture():
    circuit = make_array(daq_frontend=frontend(group(initial_hold_volts=.2)),
                         initial_source_volts=np.linspace(-.1, .2, 64))
    ref = circuit.voltage_reference_id
    fields = (np.full((64, 64), .03), np.full((64, 64), -.02))
    raw = (RawConversion("r:0", "fixture_adc_a", "hold", "signal", .2, .7, .7, .2, TRANSFER, True),
           RawConversion("r:1", "fixture_adc_b", "direct", "signal", .82, .9, .9, .82, TRANSFER, True))
    samples = (ReturnedSample("s:0", ("r:0",), (1.,), 1., .2, .75, None),
               ReturnedSample("s:1", ("r:0", "r:1"), (1., -.25), 1., .2, .95, None))
    observations = (Observation("o:0", "site", "pixel", ("s:0",), (1.,), 1., .2, 1.2, 16, None),
                    Observation("o:1", "site", "pixel", ("s:1",), (1.,), 1., .2, 1.2, 16, None))
    readouts = (ReadoutWindow(0, 0., .5, raw[:1], samples[:1], observations[:1], (("r:0", 1),), (("s:0", 1),)),
                ReadoutWindow(1, .5, 1., raw[1:], samples[1:], observations[1:], (("r:1", 1),), (("s:1", 1),)))
    first = ProviderInterval(0., .5, 0, 0, ref, {"fixture_adc_a": 0}, ((4160, .14),))
    continued = ProviderInterval(.5, .8, 0, 0, ref, {"fixture_adc_a": 0}, ())
    changed = ProviderInterval(.8, 1., 1, 1, ref, {"fixture_adc_a": 1}, ((4096, .03),))
    windows = (ArrayWindow(0, 0., .5, (StreamPhysicalInterval(0, first),), readouts[0]),
               ArrayWindow(1, .5, 1., (StreamPhysicalInterval(0, continued), StreamPhysicalInterval(1, changed)), readouts[1]))
    bindings = (ArrayInputBinding("hold", "fixture_adc_a", "hold", None, "fixture_adc_a"),
                ArrayInputBinding("direct", "fixture_adc_b", "source", 2, None))
    return circuit, fields, windows, bindings


def runner_arguments(circuit, fields, windows, bindings, *, raw_sink=None, observation_sink=None, interval_sink=None):
    return dict(array=circuit, window_factory=lambda: iter(windows), field_provider=lambda key: fields[key],
        input_bindings=bindings, acquisition_window=(0., 1.), final_publication_horizon=1.2,
        controls=controls(), limits=LIMITS, readout_limits=READOUT_LIMITS,
        on_raw=(lambda _: None) if raw_sink is None else raw_sink,
        on_observation=(lambda _: None) if observation_sink is None else observation_sink,
        on_interval=(lambda _: None) if interval_sink is None else interval_sink, check=lambda: None)


class StreamingArrayAcquisitionTests(unittest.TestCase):
    def test_cross_seam_aperture_dependencies_charge_and_final_flush_match_eager(self):
        circuit, fields, windows, bindings = make_fixture()
        raw_values, observations, evidence = [], [], []
        runner = StreamingArrayAcquisition(**runner_arguments(circuit, fields, windows, bindings,
            raw_sink=raw_values.append, observation_sink=observations.append, interval_sink=evidence.append))
        self.assertEqual(runner.preflight_summary["physical_input_events"], 2)
        self.assertEqual(runner.preflight_summary["physical_fragments"], 3)
        self.assertEqual(runner.preflight_summary["window_digest_bytes"], 64)
        self.assertEqual(runner.status, "ready")
        runner.advance_window()
        self.assertEqual(runner.physical_time, .5)
        self.assertFalse(raw_values)  # First aperture remains unfinished across seam.
        self.assertFalse(observations)
        runner.advance_window()
        self.assertEqual(runner.status, "physical_complete")
        self.assertEqual(len(raw_values), 2)
        self.assertFalse(observations)  # Both are delayed past physical end.
        end_state = circuit.volts
        report = runner.finalize()
        self.assertEqual(circuit.volts, end_state)
        self.assertEqual(runner.physical_time, 1.)
        self.assertEqual(len(observations), 2)
        self.assertEqual([item.start for item in evidence if item.event_boundary_charge_applied], [0., .8])
        self.assertEqual(report["physical_input_events"], 2)
        self.assertEqual(report["physical_fragments"], 3)
        self.assertEqual(report["split_intervals_executed"], 7)
        self.assertEqual(report["actual_internal_conversions"], 2)
        self.assertIsNone(report["modeled_complete_device_power_watt"])
        self.assertLessEqual(report["peak_live_counts"]["conversions"], 2)
        self.assertFalse(report["full_field_or_conversion_history_retained"])
        eager, _, _, _ = make_fixture()
        intervals = []
        for window in windows:
            for fragment in window.physical_intervals:
                item = fragment.interval
                charge = np.zeros(len(eager.state_ids))
                for index, value in item.injected_charge_coulomb:
                    charge[index] = value
                intervals.append(PhysicalInterval(item.start, item.end, item.gate_row, fields[item.field_key],
                    item.voltage_reference_id, item.daq_selected_sources, charge))
        expected = ArrayAcquisition(array=eager, physical_intervals=intervals, input_bindings=bindings,
            conversions=tuple(r for w in windows for r in w.readout_window.conversions),
            returned_samples=tuple(r for w in windows for r in w.readout_window.returned_samples),
            observations=tuple(r for w in windows for r in w.readout_window.observations),
            acquisition_window=(0., 1.), final_publication_horizon=1.2, controls=controls(),
            limits=AcquisitionLimits(8, 32, 8, 8, 8, 1024 * 1024)).run()
        np.testing.assert_allclose(circuit.volts, eager.volts, rtol=1e-13, atol=1e-15)
        self.assertEqual(tuple(raw_values), expected["raw_values"])
        self.assertEqual(tuple(observations), expected["observations"])
        self.assertEqual(report["propagation_operator_products"], expected["propagation_operator_products"])
        self.assertEqual(report["actual_internal_conversions"], runner.preflight_summary["readout"]["actual_internal_conversions"])
        self.assertEqual(runner.finalize(), report)

    def test_whole_plan_preflight_refuses_late_field_dependency_or_repeated_seam_charge(self):
        for problem in ("late_field", "short_dependency", "seam_charge"):
            circuit, fields, windows, bindings = make_fixture()
            args = runner_arguments(circuit, fields, windows, bindings)
            if problem == "late_field":
                args["field_provider"] = lambda key: fields[key] if key == 0 else np.full((64, 64), np.nan)
            elif problem == "short_dependency":
                first = replace(windows[0].readout_window, raw_last_use_windows=(("r:0", 0),))
                args["window_factory"] = lambda: iter((replace(windows[0], readout_window=first), windows[1]))
            else:
                continued = windows[1].physical_intervals[0]
                bad = replace(continued, interval=replace(continued.interval, injected_charge_coulomb=((4160, .14),)))
                args["window_factory"] = lambda: iter((windows[0], replace(windows[1], physical_intervals=(bad, windows[1].physical_intervals[1]))))
            before = circuit.volts, circuit.previous_field_volts
            with self.subTest(problem=problem), patch.object(circuit, "advance_row", wraps=circuit.advance_row) as advance:
                with self.assertRaises(ValueError):
                    StreamingArrayAcquisition(**args)
                advance.assert_not_called()
            self.assertEqual((circuit.volts, circuit.previous_field_volts), before)

    def test_replay_or_provider_drift_stops_before_affected_window_or_interval(self):
        for problem in ("window_metadata", "provider_at_use", "nuisance_binding"):
            circuit, fields, windows, bindings = make_fixture()
            args = runner_arguments(circuit, fields, windows, bindings)
            if problem in ("window_metadata", "nuisance_binding"):
                if problem == "nuisance_binding":
                    _, _, adapter = nuisance_fixture("N4")
                    windows = tuple(replace(window, readout_window=replace(window.readout_window,
                        nuisance_bindings=(ObservationNuisanceBinding(f"o:{index}", index,
                                                .3 if index == 0 else .83),)))
                                    for index, window in enumerate(windows))
                    args["observation_nuisance"] = adapter
                factory_calls = []
                def factory():
                    factory_calls.append(None)
                    second = windows[1]
                    if len(factory_calls) > 1:
                        if problem == "nuisance_binding":
                            changed = replace(second.readout_window.nuisance_bindings[0], acquisition_time_s=.85)
                            second = replace(second, readout_window=replace(second.readout_window,
                                                                          nuisance_bindings=(changed,)))
                        else:
                            last = second.physical_intervals[1]
                            changed = replace(last, interval=replace(last.interval, injected_charge_coulomb=((4096, .04),)))
                            second = replace(second, physical_intervals=(second.physical_intervals[0], changed))
                    return iter((windows[0], second))
                args["window_factory"] = factory
            else:
                calls = []
                def provide(key):
                    calls.append(key)
                    # Preflight3; window0 verification1/use1; window1
                    # verification2; then first actual window1 use is call8.
                    return fields[key] + (.001 if len(calls) == 8 else 0.)
                args["field_provider"] = provide
            runner = StreamingArrayAcquisition(**args)
            runner.advance_window()
            saved = circuit.volts, circuit.previous_field_volts
            with self.subTest(problem=problem), patch.object(circuit, "advance_row", wraps=circuit.advance_row) as advance:
                with self.assertRaises(AcquisitionExecutionError):
                    runner.advance_window()
                advance.assert_not_called()
            self.assertEqual((circuit.volts, circuit.previous_field_volts), saved)
            self.assertEqual(runner.physical_time, .5)
            self.assertEqual(runner.accounting_summary["physical_input_events"], 1)
            self.assertFalse(runner.accounting_summary["attempt_complete"])
            self.assertIn("physical_work", runner.failure["partial_accounting"])
            with self.assertRaises(AcquisitionExecutionError):
                runner.run()

    def test_cumulative_limits_failed_sink_and_shared_quantizer_binding(self):
        circuit, fields, windows, bindings = make_fixture()
        for changed in (replace(LIMITS, max_total_fragments=2), replace(LIMITS, max_window_digest_bytes=32),
                        replace(LIMITS, max_split_intervals=6), replace(LIMITS, max_stored_window_numeric_bytes=1)):
            with self.assertRaises(ValueError), patch.object(circuit, "advance_row", side_effect=AssertionError("preflight advanced")):
                StreamingArrayAcquisition(**{**runner_arguments(circuit, fields, windows, bindings), "limits": changed})
        limited = StreamingArrayAcquisition(**{**runner_arguments(circuit, fields, windows, bindings),
                                               "limits": replace(LIMITS, max_total_operator_products=1)})
        before = circuit.volts
        with self.assertRaises(AcquisitionExecutionError):
            limited.advance_window()
        self.assertEqual(circuit.volts, before)
        self.assertEqual(limited.accounting_summary["physical_fragments"], 0)
        self.assertIsNone(limited.accounting_summary["failed_call_operator_products"])
        self.assertEqual(limited.accounting_summary["failed_call_operator_upper_bound"], 1)
        self.assertGreater(limited.accounting_summary["measured_failed_solver_seconds"], 0.)
        circuit, fields, windows, bindings = make_fixture()
        calls = []
        def fail_sink(value):
            calls.append(value)
            raise OSError("fixture output failure")
        runner = StreamingArrayAcquisition(**runner_arguments(circuit, fields, windows, bindings, interval_sink=fail_sink))
        with self.assertRaises(AcquisitionExecutionError):
            runner.advance_window()
        fresh_circuit, _, _, _ = make_fixture()
        expired = [False]
        def check():
            if expired[0]:
                raise TimeoutError("explicit caller deadline")
        timed = StreamingArrayAcquisition(**{**runner_arguments(fresh_circuit, fields, windows, bindings), "check": check})
        timed.advance_window()
        before = fresh_circuit.volts
        expired[0] = True
        with self.assertRaises(AcquisitionExecutionError):
            timed.advance_window()
        self.assertEqual(fresh_circuit.volts, before)
        self.assertEqual(timed.physical_time, .5)
        self.assertEqual(timed.failure["exception_type"], "TimeoutError")
        self.assertEqual(len(calls), 1)
        self.assertEqual(runner.physical_time, .2)
        self.assertEqual(runner.accounting_summary["completed_electrical_intervals"], 1)
        self.assertEqual(runner.accounting_summary["physical_fragments"], 1)
        self.assertGreater(runner.accounting_summary["measured_solver_seconds"], 0.)
        self.assertIn("readout_accounting", runner.failure["partial_accounting"])
        with self.assertRaises(AcquisitionExecutionError):
            runner.advance_window()
        # Shared transfer tables do not expand per raw conversion; value-equal
        # independent quantizers bind identically, while one changed code does not.
        first = windows[0].readout_window.conversions[0]
        many = tuple(replace(first, conversion_id=f"r:{i}") for i in range(8))
        base = replace(windows[0], readout_window=replace(windows[0].readout_window,
            conversions=many, raw_last_use_windows=tuple((f"r:{i}", 1) for i in range(8))))
        copies = tuple(replace(item, quantizer=replace(item.quantizer)) for item in many)
        equivalent = replace(base, readout_window=replace(base.readout_window, conversions=copies))
        self.assertEqual(_window_digest(base, b"fixture"), _window_digest(equivalent, b"fixture"))
        different = replace(copies[-1], quantizer=replace(TRANSFER, code_volts=(-.7, -.25, .25, .75)))
        changed = replace(equivalent, readout_window=replace(equivalent.readout_window, conversions=(*copies[:-1], different)))
        self.assertNotEqual(_window_digest(base, b"fixture"), _window_digest(changed, b"fixture"))


if __name__ == "__main__":
    unittest.main()
