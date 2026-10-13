"""Focused streaming seams; explicit arithmetic fixtures, never a DAQ recipe."""

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from acquisition_readout import (AcquisitionReadout, Observation, QuantizerTransfer,
                                 RawConversion, ReturnedSample)
from streaming_acquisition_readout import (ReadoutWindow, StreamingAcquisitionReadout,
                                           StreamingReadoutLimits)
from observation_nuisance import ObservationNuisanceBinding
from raw_world_time import RawWorldTime
from test_observation_nuisance import nuisance_fixture


Q = QuantizerTransfer(-2., 2., (-1., 0., 1.), (-1.5, -.5, .5, 1.5))


def limits(**changes):
    result = StreamingReadoutLimits(max_conversions=1000, max_returned_samples=1000,
        max_observations=1000, max_interval_updates=2000, max_windows=1000,
        max_live_conversions=4, max_live_returned_samples=4, max_live_observations=4,
        max_stored_plan_numeric_bytes=4096, max_accounting_categories=4)
    return replace(result, **changes)


def raw(i, start, end, *, completion=None, valid=True, role="signal"):
    return RawConversion(f"r:{i}", "adc", "input", role, start, end,
                         end if completion is None else completion, end, Q, valid)


def sample(i, ids, available, *, weights=None, divisor=1., output_quantizer=None):
    return ReturnedSample(f"s:{i}", tuple(ids), tuple(1. for _ in ids) if weights is None else weights,
                          divisor, available, available, output_quantizer)


def observation(i, ids, available, *, weights=None, overhead=2):
    return Observation(f"o:{i}", "site", "pixel", tuple(ids),
                       tuple(1. for _ in ids) if weights is None else weights,
                       1., available, available, 8, overhead)


def stream(end, horizon=None, *, validate_only=False, on_raw=lambda value: None,
           on_observation=lambda value: None, selected_limits=None, observation_nuisance=None):
    return StreamingAcquisitionReadout(input_ids=("input",), acquisition_start=0.,
        acquisition_end=end, final_publication_horizon=end if horizon is None else horizon,
        limits=limits() if selected_limits is None else selected_limits,
        on_raw=on_raw, on_observation=on_observation, validate_only=validate_only,
        observation_nuisance=observation_nuisance)


def step(reader, start, end, mean):
    active = reader.active_input_ids(start, end)
    if reader.validate_only:
        reader.validate_interval(start=start, end=end, input_ids=active)
    else:
        reader.add_interval(start=start, end=end, mean_volts_by_input={key: mean for key in active})
    reader.advance_publication(end)


class StreamingAcquisitionReadoutTests(unittest.TestCase):
    def test_nuisance_preflight_is_metadata_only_and_delayed_overlay_preserves_raw_and_marker(self):
        for condition in ("N1", "N4"):
            world, model, adapter = nuisance_fixture(condition)
            site = (world.metadata["details"]["nuisance"]["dead_physical_row_zero_based"] * 64
                    if condition == "N4" else 0)
            raws = (raw(0, 0., .4), raw(1, .6, .9, valid=False))
            samples = (sample(0, ("r:0",), .4), sample(1, ("r:1",), .9))
            observations = (observation(0, ("s:0",), 1.2), observation(1, ("s:1",), 1.2),
                            replace(observation(2, ("s:0",), 1.2), role="marker"))
            bindings = (ObservationNuisanceBinding("o:0", site, .2),
                        ObservationNuisanceBinding("o:1", (site + 64) % 4096, .7))
            window = ReadoutWindow(0, 0., 1., raws, samples, observations,
                (("r:0", 0), ("r:1", 0)), (("s:0", 0), ("s:1", 0)), bindings)
            raw_values, published = [], []
            actual = stream(1., 1.2, observation_nuisance=adapter,
                            on_raw=raw_values.append, on_observation=published.append)
            validator = stream(1., 1.2, validate_only=True, observation_nuisance=adapter)
            with patch.object(RawWorldTime, "nuisance_at_pairs", side_effect=AssertionError("preflight read nuisance values")):
                validator.accept_window(window)
                for start, end in ((0., .4), (.4, .6), (.6, .9), (.9, 1.)):
                    step(validator, start, end, None)
                validator.close_window(0)
                validation = validator.finalize()
            self.assertIsNone(validation["nuisance_observations_processed"])
            actual.accept_window(window)
            for start, end in ((0., .4), (.4, .6), (.6, .9), (.9, 1.)):
                step(actual, start, end, 4.)
            actual.close_window(0)
            self.assertEqual(published, [])  # The acquisition point never grants early visibility.
            report = actual.finalize()
            self.assertEqual(report["nuisance_observations_processed"], 2)
            self.assertEqual(report["actual_internal_conversions"], 2)
            self.assertEqual(report["hardware_returned_samples"], 2)
            self.assertEqual(report["output_bits"], 30)
            self.assertEqual([value.decoded_volts for value in raw_values], [1.5, 1.5])
            self.assertTrue(all(value.clipped for value in raw_values))
            self.assertEqual([value.available_at for value in published], [1.2] * 3)
            self.assertEqual([value.assigned_timestamp for value in published], [1.2] * 3)
            self.assertTrue(all(value.clipped for value in published))
            self.assertIsNone(published[1].value_volts)  # Prior invalidity cannot be repaired.
            self.assertFalse(published[1].valid)
            self.assertEqual(published[2].value_volts, 1.5)  # Marker is untouched.
            if condition == "N4":
                self.assertIsNone(published[0].value_volts)
                self.assertEqual(report["valid_retained_pixel_observations"], 0)
            else:
                expected = model.nuisance_at_pairs(np.array([.2]),
                    site_indices=np.array([site], dtype=np.int64))["spatially_common_line_uv"][0]
                self.assertEqual(published[0].value_volts, 1.5 + expected / 1e6)
                self.assertEqual(report["valid_retained_pixel_observations"], 1)
            # Exact coverage is required, and a gap inside the bounding support
            # interval is not an eligible nuisance query point.
            missing = replace(window, nuisance_bindings=bindings[:1])
            gap_observation = replace(observations[0], sample_ids=("s:0", "s:1"), weights=(1., 0.))
            gap = replace(window, observations=(gap_observation, *observations[1:]),
                          nuisance_bindings=(replace(bindings[0], acquisition_time_s=.5), bindings[1]))
            for invalid in (missing, gap):
                reader = stream(1., 1.2, validate_only=True, observation_nuisance=adapter)
                with self.assertRaises(ValueError):
                    reader.accept_window(invalid)
                self.assertEqual(reader.physical_time, 0.)
            with self.assertRaisesRegex(ValueError, "enabled saved-world adapter"):
                stream(1., 1.2).accept_window(window)

    def test_cross_window_aperture_zero_weight_validity_delays_and_eager_equality(self):
        r0 = raw(0, 0., 1.5, completion=1.75, valid=False)
        r1 = raw(1, 1., 2.)
        r2 = raw(2, 2., 3., role="discarded_marker")
        s0 = sample(0, ("r:0",), 1.8)
        # A zero-weight invalid contributor still governs validity and timing.
        s1 = sample(1, ("r:0", "r:1"), 2.25, weights=(0., 1.))
        s2 = sample(2, ("r:1",), 2.5, weights=(-2.,), divisor=2., output_quantizer=Q)
        o0 = observation(0, ("s:1", "s:0"), 3.5, weights=(1., 0.), overhead=None)
        o1 = observation(1, ("s:2",), 2.7)
        windows = (
            ReadoutWindow(0, 0., 1., (r0,), (s0,), (), (("r:0", 2),), (("s:0", 2),)),
            ReadoutWindow(1, 1., 2., (r1,), (s1, s2), (o0,), (("r:1", 2),),
                          (("s:1", 2), ("s:2", 2))),
            ReadoutWindow(2, 2., 3., (r2,), (), (o1,), (("r:2", 2),), ()))
        segments = (((0., 1., .25),), ((1., 1.5, -.75), (1.5, 2., .75)),
                    ((2., 2.5, 4.), (2.5, 3., 4.)))
        raw_values, published = [], []
        actual = stream(3., 4., on_raw=raw_values.append, on_observation=published.append)
        def no_values(_):
            self.fail("Metadata-only preflight must not manufacture or emit values")
        validator = stream(3., 4., validate_only=True, on_raw=no_values, on_observation=no_values)
        for reader in (validator, actual):
            for window, intervals in zip(windows, segments):
                reader.accept_window(window)
                for start, end, mean in intervals:
                    step(reader, start, end, mean)
                    if reader is actual and end == 1.5:
                        self.assertEqual(raw_values, [])  # aperture ended, conversion not completed
                reader.close_window(window.index)
            if reader is actual:
                self.assertEqual([value.observation_id for value in published], ["o:1"])
        expected = AcquisitionReadout(conversions=(r0, r1, r2), returned_samples=(s0, s1, s2),
            observations=(o0, o1), max_conversions=3, max_returned_samples=3,
            max_observations=2, max_interval_updates=5)
        for intervals in segments:
            for start, end, mean in intervals:
                expected.add_interval(start=start, end=end, mean_volts_by_input={"input": mean})
        eager = expected.finalize(finished_at=4., acquisition_start=0., acquisition_end=3.)
        report, validation = actual.finalize(), validator.finalize()
        self.assertEqual(tuple(raw_values), eager["raw_values"])
        self.assertEqual(tuple(published), eager["observations"])
        self.assertEqual(published[-1].contributing_apertures,
                         (("r:0", 0., 1.5), ("r:1", 1., 2.)))
        self.assertIsNone(published[-1].value_volts)
        self.assertTrue(raw_values[-1].clipped)
        for key, value in eager.items():
            if key not in ("raw_values", "observations"):
                self.assertEqual(report[key], value, key)
                if key != "valid_retained_pixel_observations":
                    self.assertEqual(validation[key], value, key)
        self.assertEqual(report["conversions_without_returned_sample"], 1)
        self.assertIsNone(report["output_bits"])
        self.assertEqual(report["known_output_bit_subtotal"], 18)
        self.assertTrue(validation["metadata_only"])
        self.assertFalse(report["metadata_only"])
        self.assertEqual(actual.live_counts["conversions"], 0)

    def test_closure_late_dependency_identity_and_missing_coverage_refuse(self):
        # An aperture itself may cross a seam only with a sufficient lifetime.
        reader = stream(2., validate_only=True)
        reader.accept_window(ReadoutWindow(0, 0., 1., (raw(0, 0., 1.5),), (), (), (("r:0", 0),), ()))
        step(reader, 0., 1., None)
        with self.assertRaisesRegex(ValueError, "incomplete or still-needed"):
            reader.close_window(0)
        self.assertTrue(reader.failed)
        # Once closure evicts a record, neither a late consumer nor a reused ID
        # may reintroduce it. No unbounded historical identity set is needed.
        for kind in ("reference", "identity", "late_publication"):
            with self.subTest(kind=kind):
                reader = stream(2., validate_only=True)
                reader.accept_window(ReadoutWindow(0, 0., 1., (raw(0, 0., 1.),), (), (), (("r:0", 0),), ()))
                step(reader, 0., 1., None)
                reader.close_window(0)
                if kind == "reference":
                    window = ReadoutWindow(1, 1., 2., (), (sample(0, ("r:0",), 2.),), (), (), (("s:0", 1),))
                elif kind == "identity":
                    window = ReadoutWindow(1, 1., 2., (raw(0, 1., 2.),), (), (), (("r:0", 1),), ())
                else:
                    window = ReadoutWindow(1, 1., 2., (raw(1, 1., 2.),), (sample(0, ("r:1",), 1.),),
                                           (), (("r:1", 1),), (("s:0", 1),))
                with self.assertRaises(ValueError):
                    reader.accept_window(window)
                self.assertTrue(reader.failed)
        # Exact input coverage cannot be weakened to support an omitted channel.
        reader = stream(1.)
        reader.accept_window(ReadoutWindow(0, 0., 1., (raw(0, 0., 1.),), (), (), (("r:0", 0),), ()))
        with self.assertRaisesRegex(ValueError, "every active"):
            reader.add_interval(start=0., end=1., mean_volts_by_input={})
        self.assertEqual(reader.physical_time, 0.)

    def test_live_retention_stays_bounded_and_cumulative_counts_survive_eviction(self):
        calls = Counter()
        reader = stream(100., selected_limits=limits(max_live_conversions=1,
            max_live_returned_samples=1, max_live_observations=1),
            on_raw=lambda _: calls.update(("raw",)), on_observation=lambda _: calls.update(("observation",)))
        for index in range(100):
            start, end = float(index), float(index + 1)
            window = ReadoutWindow(index, start, end, (raw(index, start, end),),
                (sample(index, (f"r:{index}",), end),),
                (observation(index, (f"s:{index}",), end),),
                ((f"r:{index}", index),), ((f"s:{index}", index),))
            reader.accept_window(window)
            step(reader, start, end, .25)
            reader.close_window(index)
            if index < 99:
                self.assertEqual(reader.live_counts, dict(conversions=0, returned_samples=0,
                                                         observations=0, stored_plan_numeric_bytes=0))
        report = reader.finalize()
        self.assertEqual(calls, Counter(raw=100, observation=100))
        self.assertEqual(report["actual_internal_conversions"], 100)
        self.assertEqual(report["hardware_returned_samples"], 100)
        self.assertEqual(report["retained_pixel_observations"], 100)
        self.assertEqual(report["valid_retained_pixel_observations"], 100)
        self.assertEqual(report["conversions_by_role"], {"signal": 100})
        self.assertEqual(report["conversions_by_adc"], {"adc": 100})
        self.assertEqual(report["conversions_without_returned_sample"], 0)
        self.assertEqual(report["output_bits"], 1000)
        self.assertEqual(report["interval_updates"], 100)
        for key in ("conversions", "returned_samples", "observations"):
            self.assertEqual(report["peak_live_counts"][key], 1)
        self.assertNotIn("raw_values", report)
        self.assertNotIn("observations", report)

    def test_final_delayed_dependency_and_callback_failure_are_terminal(self):
        reader = stream(2., 3., validate_only=True)
        reader.accept_window(ReadoutWindow(0, 0., 1., (raw(0, 0., 1.),),
            (sample(0, ("r:0",), 1.),), (observation(0, ("s:0",), 2.5),),
            (("r:0", 0),), (("s:0", 0),)))
        step(reader, 0., 1., None)
        with self.assertRaisesRegex(ValueError, "still-needed"):
            reader.close_window(0)
        emitted = []
        def failing_sink(value):
            emitted.append(value)
            raise OSError("selected sink write failed")
        reader = stream(1., on_raw=failing_sink)
        reader.accept_window(ReadoutWindow(0, 0., 1., (raw(0, 0., 1.),), (), (), (("r:0", 0),), ()))
        reader.add_interval(start=0., end=1., mean_volts_by_input={"input": .25})
        with self.assertRaises(OSError):
            reader.advance_publication(1.)
        self.assertTrue(reader.failed)
        partial = reader.accounting()
        self.assertEqual(partial["completed_conversions"], 1)
        self.assertEqual(partial["raw_callbacks_returned"], 0)
        self.assertEqual(partial["callback_with_uncertain_effect"], ("raw", "r:0"))
        self.assertFalse(partial["callback_return_confirms_persistence"])
        self.assertFalse(partial["complete"])
        with self.assertRaisesRegex(ValueError, "terminal"):
            reader.advance_publication(1.)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(reader.accounting(), partial)


if __name__ == "__main__":
    unittest.main()
