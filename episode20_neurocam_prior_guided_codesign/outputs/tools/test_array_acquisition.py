"""Finite open-loop integration fixtures; no calibration or device defaults."""

from dataclasses import FrozenInstanceError, replace
import math
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import (AcquisitionExecutionError, AcquisitionLimits, ArrayAcquisition,
                               ArrayInputBinding, PhysicalInterval, ProviderInterval)
from test_cpe_array import make_array
from test_daq_mux_hold import frontend, group
from test_sparse_array_rc import controls


TRANSFER = QuantizerTransfer(-2, 2, (0,), (-1, 1))
LIMITS = AcquisitionLimits(4, 12, 4, 4, 4, 1024 * 1024)


def raw(key, start, end, adc="fixture_adc_a"):
    return RawConversion(key, adc, "input", "signal", start, end, end, start, TRANSFER, True)


def outputs(conversions, publication_times):
    samples = [ReturnedSample("word_" + value.conversion_id, (value.conversion_id,), (1,), 1,
                              value.assigned_timestamp, value.completed_at, None) for value in conversions]
    observations = [Observation("obs_" + value.conversion_id, "fixture_site", "pixel",
                                 (sample.sample_id,), (1,), 1, value.assigned_timestamp, time, 8, None)
                    for value, sample, time in zip(conversions, samples, publication_times)]
    return samples, observations


def event(array, start, end, *, charge=None, selections=None, field=None):
    return PhysicalInterval(start, end, 0, np.zeros((64, 64)) if field is None else field,
                            array.voltage_reference_id, selections,
                            np.zeros(len(array.state_ids)) if charge is None else charge)


def arguments(array, *, intervals=None, conversions=None, publications=None, binding=None,
              window=(0, 0.25), horizon=0.5):
    conversions = [raw("a", 0, 0.25)] if conversions is None else conversions
    samples, observations = outputs(conversions, [horizon] * len(conversions) if publications is None else publications)
    return dict(array=array, physical_intervals=[event(array, 0, 0.25)] if intervals is None else intervals,
                input_bindings=[ArrayInputBinding("input", "fixture_adc_a", "source", 0, None)
                                if binding is None else binding],
                conversions=conversions, returned_samples=samples, observations=observations,
                acquisition_window=window, final_publication_horizon=horizon, controls=controls(), limits=LIMITS)


class ArrayAcquisitionTests(unittest.TestCase):
    def test_provider_matches_eager_fields_across_apertures_and_once_only_sparse_charge(self):
        specification = frontend(group(source_indices=(0,), track_conductance_siemens=(0.8,),
                                       off_conductance_siemens=(0,), hold_leak_conductance_siemens=0.2,
                                       initial_hold_volts=0.2))
        eager_array, generated_array = (make_array(daq_frontend=specification) for _ in range(2))
        charge = np.zeros(len(eager_array.state_ids))
        charge[-1] = 0.14
        fields = (np.full((64, 64), 0.003), np.full((64, 64), -0.002))
        selections = {"fixture_adc_a": None}
        sparse = [[len(charge) - 1, 0.14]]
        generated = [ProviderInterval(0, 0.5, 0, 0, generated_array.voltage_reference_id, selections, sparse),
                     ProviderInterval(0.5, 1, 0, 1, generated_array.voltage_reference_id, selections, ())]
        large_key = np.int64(2**53 + 1)
        self.assertEqual(replace(generated[0], field_key=large_key).field_key, 2**53 + 1)
        eager = [event(eager_array, 0, 0.5, charge=charge, selections=selections, field=fields[0]),
                 event(eager_array, 0.5, 1, selections=selections, field=fields[1])]
        conversions = [raw("a", 0.125, 0.25), raw("b", 0.625, 0.875)]
        common = dict(conversions=conversions, publications=[0.5, 1.25], window=(0, 1), horizon=1.25,
                      binding=ArrayInputBinding("input", "fixture_adc_a", "hold", None, "fixture_adc_a"))
        eager_runner = ArrayAcquisition(**arguments(eager_array, intervals=eager, **common))
        # The caller's reused field buffer is copied once per physical event,
        # not reused as mutable storage during subsequent aperture splits.
        buffer, calls = np.empty((64, 64), order="F"), []
        def provide(key):
            calls.append(key)
            buffer[:] = fields[key]
            return buffer
        generated_runner = ArrayAcquisition(**arguments(generated_array, intervals=generated, **common),
                                            field_provider=provide)
        self.assertEqual(calls, [0, 1])  # Entire field plan prevalidated first.
        sparse[0][1] = 700
        selections["fixture_adc_a"] = 0
        with self.assertRaises(FrozenInstanceError):
            generated[0].field_key = 9
        with self.assertRaises(TypeError):
            generated[0].daq_selected_sources["fixture_adc_a"] = 0
        self.assertEqual(generated_runner.advance_next(), ())
        buffer[:] = 800  # Must not affect the copied field of the active event.
        generated_report, eager_report = generated_runner.run(), eager_runner.run()
        self.assertEqual(calls, [0, 1, 0, 1])  # No regeneration at aperture splits.
        self.assertEqual(generated_array.volts, eager_array.volts)
        self.assertEqual(generated_array.previous_field_volts, eager_array.previous_field_volts)
        for key in ("raw_values", "interval_evidence", "actual_internal_conversions", "physical_end_time",
                    "final_publication_horizon", "propagation_operator_products"):
            self.assertEqual(generated_report[key], eager_report[key])
        self.assertEqual(generated_report["provider_field_digest_bytes"], 64)
        self.assertEqual(generated_report["stored_plan_numeric_bytes"], 8 * len(charge) + 64 + 16)
        self.assertIsNone(generated_runner._materialized_field)

    def test_provider_prevalidates_late_fields_and_all_schedule_before_electrical_mutation(self):
        array = make_array()
        intervals = [ProviderInterval(0, 0.125, 0, 0, array.voltage_reference_id, None, ()),
                     ProviderInterval(0.125, 0.25, 0, 1, array.voltage_reference_id, None, ())]
        base, saved = arguments(array, intervals=intervals), (array.volts, array.previous_field_volts)
        for invalid in (np.full((64, 64), np.nan), np.zeros((63, 64))):
            calls = []
            def provide(key):
                calls.append(key)
                return np.zeros((64, 64)) if key == 0 else invalid
            with patch.object(array, "advance_row", wraps=array.advance_row) as advance:
                with self.assertRaises(ValueError):
                    ArrayAcquisition(**base, field_provider=provide)
                advance.assert_not_called()
            self.assertEqual(calls, [0, 1])
            self.assertEqual((array.volts, array.previous_field_volts), saved)
        malformed = (
            {"physical_intervals": [intervals[0], replace(intervals[1], start=0.2)]},
            {"physical_intervals": [intervals[0], replace(intervals[1], gate_row=64)]},
            {"physical_intervals": [intervals[0], replace(intervals[1],
                injected_charge_coulomb=((len(array.state_ids), 0.1),))]},
            {"conversions": [replace(base["conversions"][0], completed_at=0.3)]},
            {"limits": replace(LIMITS, max_split_intervals=1)},
        )
        for changes in malformed:
            with self.subTest(changes=tuple(changes)), patch("array_acquisition._field_digest") as digest:
                with self.assertRaises(ValueError):
                    ArrayAcquisition(**{**base, **changes}, field_provider=lambda _: np.zeros((64, 64)))
                digest.assert_not_called()
            self.assertEqual((array.volts, array.previous_field_volts), saved)

    def test_provider_finite_drift_or_exception_is_terminal_before_current_interval_mutation(self):
        for failure in ("finite drift", "exception"):
            with self.subTest(failure=failure):
                array = make_array(initial_source_volts=np.ones(64))
                intervals = [ProviderInterval(0, 0.125, 0, 0, array.voltage_reference_id, None, ()),
                             ProviderInterval(0.125, 0.25, 0, 1, array.voltage_reference_id, None, ())]
                calls = []
                def provide(key):
                    calls.append(key)
                    if len(calls) == 4:
                        if failure == "exception":
                            raise ValueError("forced provider failure")
                        return np.ones((64, 64))
                    return np.zeros((64, 64))
                runner = ArrayAcquisition(**arguments(array, intervals=intervals), field_provider=provide)
                runner.advance_next()
                saved = array.volts, array.previous_field_volts
                with patch.object(array, "advance_row", wraps=array.advance_row) as advance:
                    with self.assertRaises(AcquisitionExecutionError):
                        runner.advance_next()
                    advance.assert_not_called()
                self.assertEqual((array.volts, array.previous_field_volts), saved)
                self.assertEqual((runner.status, runner.physical_time), ("failed", 0.125))
                self.assertEqual(runner.failure.phase, "field materialization")
                self.assertFalse(runner.failure.array_call_returned)
                self.assertEqual(runner.failure.completed_split_intervals, 1)
                self.assertIsNone(runner._materialized_field)
                with self.assertRaises(AcquisitionExecutionError):
                    runner.run()
                self.assertEqual(calls, [0, 1, 0, 1])

    def test_provider_retains_packed_digests_not_generated_fields_or_dense_zero_charges(self):
        array, count, produced = make_array(), 256, []
        intervals = [ProviderInterval(index / count, (index + 1) / count, index % 64, index,
                                       array.voltage_reference_id, None, ()) for index in range(count)]
        expected = 8 * len(array.state_ids) + 32 * count
        bounded = replace(LIMITS, max_physical_intervals=count, max_split_intervals=count,
                          max_stored_plan_numeric_bytes=expected)
        def provide(_):
            value = np.zeros((64, 64))
            produced.append(weakref.ref(value))
            return value
        base = arguments(array, intervals=intervals, window=(0, 1), horizon=1)
        runner = ArrayAcquisition(**{**base, "limits": bounded}, field_provider=provide)
        self.assertEqual(len(produced), count)
        self.assertTrue(all(ref() is None for ref in produced))
        self.assertIsInstance(runner._field_digests, bytes)
        self.assertEqual(len(runner._field_digests), 32 * count)
        self.assertEqual(runner._numeric_bytes, expected)
        self.assertIsNone(runner._materialized_field)
        self.assertTrue(all(isinstance(item, ProviderInterval) and item.injected_charge_coulomb == ()
                            for item in runner._events))
        with self.assertRaises(ValueError):
            ArrayAcquisition(**{**base, "limits": replace(bounded, max_stored_plan_numeric_bytes=expected - 1)},
                             field_provider=provide)
        self.assertEqual(len(produced), count)  # Rejected before another traversal.

    def test_exact_splitting_once_only_charge_unsampled_decay_and_publication_horizon(self):
        specification = frontend(group(source_indices=(0,), track_conductance_siemens=(0.8,),
                                       off_conductance_siemens=(0,), hold_leak_conductance_siemens=0.2,
                                       initial_hold_volts=0.2))
        array = make_array(daq_frontend=specification)
        charge = np.zeros(len(array.state_ids))
        charge[-1] = 0.14
        field = np.zeros((64, 64))
        selections = {"fixture_adc_a": None}
        intervals = [event(array, 0, 0.5, charge=charge, selections=selections, field=field),
                     event(array, 0.5, 1, selections=selections, field=field)]
        conversions = [raw("a", 0.125, 0.25), raw("b", 0.625, 0.875)]
        runner = ArrayAcquisition(**arguments(
            array, intervals=intervals, conversions=conversions, publications=[0.5, 1.25],
            binding=ArrayInputBinding("input", "fixture_adc_a", "hold", None, "fixture_adc_a"),
            window=(0, 1), horizon=1.25))
        # Prevalidated events belong to the runner; external mutation cannot
        # alter field, switching or charge between its subintervals.
        field[:] = 100
        charge[-1] = 700
        selections["fixture_adc_a"] = 0
        self.assertEqual(runner.advance_next(), ())  # Unsampled [0, .125].
        self.assertEqual(runner.advance_next(), ())
        with self.assertRaises(ValueError):
            runner.observations_available_by(0.5)
        # The interval loop must never reconstruct the whole published prefix;
        # the existing cumulative API remains available for explicit queries.
        with patch.object(runner._readout, "observations_available_by",
                          side_effect=AssertionError("interval loop used cumulative publication")):
            first = runner.advance_next()
            self.assertEqual(tuple(value.observation_id for value in first), ("obs_a",))
            for _ in range(3):
                self.assertEqual(runner.advance_next(), ())
        self.assertEqual((runner.status, runner.physical_time), ("physical_complete", 1))
        with self.assertRaises(ValueError):
            runner.observations_for_history(target_time=1, history_seconds=1, publication_deadline=1.25)
        decay = 0.2 / 0.7
        self.assertAlmostEqual(array.volts[-1], 0.4 * math.exp(-decay), places=11)
        report = runner.finalize()
        self.assertEqual((runner.status, runner.physical_time), ("complete", 1))
        self.assertEqual((report["physical_end_time"], report["final_publication_horizon"]), (1, 1.25))
        for value, conversion in zip(report["raw_values"], conversions):
            expected = (0.4 * (math.exp(-decay * conversion.aperture_start)
                                - math.exp(-decay * conversion.aperture_end))
                        / (decay * (conversion.aperture_end - conversion.aperture_start)))
            self.assertAlmostEqual(value.aperture_mean_volts, expected, places=11)
        self.assertEqual(report["split_intervals_executed"], 6)
        self.assertEqual([step.event_boundary_charge_applied for step in runner.evidence],
                         [True, False, False, True, False, False])
        self.assertGreater(report["propagation_operator_products"], 0)
        self.assertEqual(report["actual_internal_conversions"], 2)
        history = runner.observations_for_history(target_time=0.5, history_seconds=0.5,
                                                   publication_deadline=1.25)
        self.assertEqual(tuple(value.observation_id for value in history), ("obs_a",))

    def test_direct_source_mean_and_whole_plan_prevalidation_before_mutation(self):
        initial_sources = np.zeros(64)
        initial_sources[0] = 1
        array = make_array(initial_source_volts=initial_sources,
                           tft_on_conductance_siemens=np.full((64, 64), 0.2),
                           tft_off_conductance_siemens=np.full((64, 64), 0.2))
        saved = array.volts, array.previous_field_volts
        base = arguments(array)
        malformed = [
            {"physical_intervals": [event(array, 0, 0.125), event(array, 0.2, 0.25)]},
            {"physical_intervals": [event(array, 0, 0.125), replace(event(array, 0.125, 0.25), gate_row=64)]},
            {"input_bindings": []},
            {"input_bindings": [ArrayInputBinding("input", "wrong_adc", "source", 0, None)]},
            {"input_bindings": [ArrayInputBinding("input", "fixture_adc_a", "hold", None, "fixture_adc_a")]},
            {"conversions": [replace(base["conversions"][0], completed_at=0.3)]},
            {"limits": replace(LIMITS, max_stored_plan_numeric_bytes=1)},
            {"physical_intervals": [event(array, 0, 0.125), event(array, 0.125, 0.25)],
             "limits": replace(LIMITS, max_split_intervals=1)},
        ]
        for changes in malformed:
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                ArrayAcquisition(**{**base, **changes})
            self.assertEqual((array.volts, array.previous_field_volts), saved)
        visits = []
        def too_many_events():
            for index in range(100):
                visits.append(index)
                yield event(array, index * 0.25, (index + 1) * 0.25)
        with self.assertRaises(ValueError):
            ArrayAcquisition(**{**base, "physical_intervals": too_many_events()})
        self.assertEqual(len(visits), LIMITS.max_physical_intervals + 1)
        with_hold = make_array(daq_frontend=frontend(group()))
        with self.assertRaises(ValueError):
            ArrayAcquisition(**arguments(with_hold, intervals=[event(with_hold, 0, 0.25,
                                                                     selections={"fixture_adc_a": None})]))
        report = ArrayAcquisition(**base).run()
        # Independent two-node reduction: 64 identical pad capacitors aggregate
        # to 192 F, source C=1 F, joining G=12.8 S in this arithmetic fixture.
        rate = 12.8 * (1 + 1 / 192)
        expected = 1 / 193 + (192 / 193) * (-math.expm1(-rate * 0.25)) / (rate * 0.25)
        self.assertAlmostEqual(report["raw_values"][0].aperture_mean_volts, expected, places=10)
        self.assertEqual(report["actual_internal_conversions"], 1)

    def test_failure_after_physical_commit_is_terminal_and_cannot_replay(self):
        array = make_array(initial_source_volts=np.ones(64))
        runner = ArrayAcquisition(**arguments(array))
        with patch.object(runner._readout, "add_interval", side_effect=ValueError("forced downstream refusal")):
            with self.assertRaises(AcquisitionExecutionError):
                runner.advance_next()
        self.assertEqual((runner.status, runner.physical_time), ("failed", 0.25))
        self.assertEqual(runner.failure.phase, "readout accumulation")
        self.assertTrue(runner.failure.array_call_returned)
        self.assertEqual(runner.failure.completed_split_intervals, 0)
        self.assertGreater(runner.failure.propagation_operator_products, 0)
        saved = array.volts
        for operation in (runner.advance_next, runner.run, runner.finalize,
                          lambda: runner.observations_available_by(0.25)):
            with self.assertRaises(AcquisitionExecutionError):
                operation()
            self.assertEqual(array.volts, saved)
        # A propagation refusal also closes the attempt, even if this supplied
        # array happened to reject before committing any physical state.
        untouched = make_array()
        failed = ArrayAcquisition(**arguments(untouched))
        with patch.object(untouched, "advance_row", side_effect=ValueError("forced propagation refusal")):
            with self.assertRaises(AcquisitionExecutionError):
                failed.advance_next()
        self.assertEqual((failed.status, failed.physical_time), ("failed", 0))
        self.assertFalse(failed.failure.array_call_returned)
        interrupted_array = make_array(initial_source_volts=np.ones(64))
        interrupted = ArrayAcquisition(**arguments(interrupted_array))
        with patch.object(interrupted._readout, "add_interval", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                interrupted.advance_next()
        self.assertEqual((interrupted.status, interrupted.physical_time), ("failed", 0.25))
        self.assertEqual(interrupted.failure.exception_type, "KeyboardInterrupt")
        after_interrupt = interrupted_array.volts
        with self.assertRaises(AcquisitionExecutionError):
            interrupted.advance_next()
        self.assertEqual(interrupted_array.volts, after_interrupt)


if __name__ == "__main__":
    unittest.main()
