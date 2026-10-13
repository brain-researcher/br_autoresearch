"""Finite arithmetic/timing fixtures, not inferred DAQ settings or calibration."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from acquisition_readout import (
    AcquisitionReadout, Observation, QuantizerTransfer, RawConversion, ReturnedSample,
)


# Deliberately unit-scale binary transfer for transparent arithmetic identities.
TRANSFER = QuantizerTransfer(-2, 2, (0,), (-1, 1))


def raw(key, start, end, *, input_id=None, role="signal", valid=True, adc_id="test_adc"):
    return RawConversion(key, adc_id, input_id or key, role, start, end, end,
                         start, TRANSFER, valid)


def sample(key, conversions, *, available, divisor=1):
    return ReturnedSample(key, tuple(conversions), (1,) * len(conversions), divisor,
                          0, available, None)


def observation(key, samples, *, available, role="pixel", overhead=None):
    return Observation(key, "fixture_site", role, tuple(samples), (1,) * len(samples),
                       1, 0, available, 8, overhead)


def readout(conversions, returned, observations):
    return AcquisitionReadout(conversions=conversions, returned_samples=returned,
                              observations=observations, max_conversions=8,
                              max_returned_samples=8, max_observations=8, max_interval_updates=12)


class AcquisitionReadoutTests(unittest.TestCase):
    def test_incremental_delivery_visits_only_new_plans_and_cumulative_queries_do_not_consume_it(self):
        conversions = [raw("a", 0, 1), raw("b", 0, 1)]
        returned = [sample("word_a", ("a",), available=1), sample("word_b", ("b",), available=1)]
        # Deliberately nonchronological insertion, with two publication ties.
        observations = [observation("late", ("word_a",), available=4),
                        observation("tied_z", ("word_b",), available=2),
                        observation("early", ("word_a",), available=1),
                        observation("tied_a", ("word_a",), available=2)]
        plan = readout(conversions, returned, observations)
        plan.add_interval(start=0, end=1, mean_volts_by_input={"a": .2, "b": -.2})
        expected = plan.observations_available_by(4)
        self.assertEqual(plan._publication_cursor, 0)
        visited, delivered = [], []
        publish = plan._publish
        def record(plans):
            values = tuple(plans)
            visited.extend(value.observation_id for value in values)
            return publish(values)
        with patch.object(plan, "_publish", side_effect=record):
            for deadline in (.1, .5, .9, 1., 1., 1.25, 1.5, 2., 2., 2.5, 3., 4., 4., 5.):
                delivered.extend(plan.new_observations_available_by(deadline))
        self.assertEqual(tuple(delivered), expected)
        self.assertEqual(visited, ["early", "tied_a", "tied_z", "late"])
        self.assertEqual(plan.observations_available_by(2), expected[:3])
        self.assertEqual(plan._publication_cursor, len(observations))
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            plan.new_observations_available_by(4)
        report = plan.finalize(finished_at=5, acquisition_start=0, acquisition_end=1)
        self.assertEqual(report["observations"], expected)
        self.assertEqual((report["actual_internal_conversions"], report["hardware_returned_samples"],
                          report["retained_pixel_observations"]), (2, 2, 4))

    def test_incremental_failed_batch_keeps_cursor_and_prior_deadline_for_later_completion(self):
        plan = readout([raw("a", 0, 1), raw("b", 0, 1, valid=False)],
                       [sample("word_a", ("a",), available=1), sample("word_b", ("b",), available=1)],
                       [observation("first", ("word_a",), available=1),
                        observation("second", ("word_b",), available=1)])
        self.assertEqual(plan.new_observations_available_by(.5), ())
        plan.add_interval(start=0, end=1, mean_volts_by_input={"a": 3.})
        with self.assertRaisesRegex(ValueError, "aperture is incomplete"):
            plan.new_observations_available_by(1)
        self.assertEqual((plan._publication_cursor, plan._incremental_deadline), (0, .5))
        # The first value may be memoized internally, but neither tied output
        # was delivered or skipped by the failed batch. Timing/invalidity stay.
        self.assertEqual(plan.new_observations_available_by(.75), ())
        plan.add_interval(start=0, end=1, mean_volts_by_input={"b": -.2})
        published = plan.new_observations_available_by(1)
        self.assertEqual(tuple(value.observation_id for value in published), ("first", "second"))
        self.assertTrue(published[0].clipped)
        self.assertFalse(published[1].valid)
        self.assertIsNone(published[1].value_volts)
        self.assertEqual(plan.new_observations_available_by(1), ())
        self.assertEqual(plan.observations_available_by(1), published)

    def test_raw_quantization_precedes_average_and_all_work_counts(self):
        plan = readout(
            [raw("a", 0, 1, input_id="signal"),
             raw("b", 1, 2, input_id="signal"),
             raw("marker", 0, 1, role="marker", adc_id="marker_adc"),
             raw("ref", 0, 1, role="reference", adc_id="reference_adc", valid=False),
             raw("discard", 0, 1, role="startup", adc_id="startup_adc")],
            [sample("averaged", ("a", "b"), available=2.5, divisor=2),
             sample("marker_word", ("marker",), available=1.5),
             sample("reference_word", ("ref",), available=1.5)],
            [observation("pixel", ("averaged",), available=3),
             observation("invalid_pixel", ("averaged", "reference_word"), available=3),
             observation("marker_output", ("marker_word",), available=4, role="marker", overhead=2)])
        plan.add_interval(start=0, end=1,
                          mean_volts_by_input={"signal": -0.1, "marker": 3, "ref": -0.3, "discard": 0.6})
        plan.add_interval(start=1, end=2, mean_volts_by_input={"signal": 0.2})
        result = plan.finalize(finished_at=4, acquisition_start=0, acquisition_end=2)
        values = {v.observation_id: v for v in result["observations"]}
        self.assertEqual(values["pixel"].value_volts, 0)  # (-1 + 1) / 2
        self.assertEqual(TRANSFER.convert((-0.1 + 0.2) / 2)[1], 1)  # Different order.
        self.assertTrue(values["pixel"].valid)
        self.assertFalse(values["pixel"].clipped)
        self.assertFalse(values["invalid_pixel"].valid)
        self.assertIsNone(values["invalid_pixel"].value_volts)  # No fill or reweighting.
        self.assertFalse(values["invalid_pixel"].clipped)
        self.assertTrue(values["marker_output"].valid)
        self.assertTrue(values["marker_output"].clipped)  # Clipping is a separate flag.
        self.assertEqual((result["actual_internal_conversions"], result["hardware_returned_samples"],
                          result["retained_pixel_observations"]), (5, 3, 2))
        self.assertEqual(result["valid_retained_pixel_observations"], 1)
        self.assertEqual(result["conversions_without_returned_sample"], 1)
        self.assertEqual(result["conversions_by_role"], {"signal": 2, "marker": 1, "reference": 1, "startup": 1})
        self.assertEqual(result["conversions_by_adc"]["test_adc"], 2)
        self.assertEqual((result["raw_conversion_rate_hz"], result["hardware_returned_sample_rate_hz"],
                          result["retained_pixel_rate_hz"]), (2.5, 1.5, 1))
        self.assertIsNone(result["output_bits"])
        self.assertEqual(result["known_output_bit_subtotal"], 26)
        self.assertIsNone(result["output_bitrate_bps"])
        self.assertIsNone(result["hardware_transport_bits"])
        self.assertIsNone(result["modeled_power_watt"])
        self.assertIsNone(result["modeled_conversion_energy_joule"])
        self.assertFalse(result["actual_DAQ_configuration_qualified"])

    def test_rectangular_coverage_refusal_is_atomic_and_no_gap_is_prorated(self):
        plan = readout([raw("a", 0, 1), raw("b", 0.5, 1)], [], [])
        self.assertEqual(plan.required_aperture_boundaries, (0, 0.5, 1))
        # Input a validates first; b's aperture only partially intersects this
        # interval. Nothing from a may commit when b refuses the shared update.
        with self.assertRaises(ValueError):
            plan.add_interval(start=0, end=1, mean_volts_by_input={"a": 100, "b": 100})
        plan.add_interval(start=0, end=0.5, mean_volts_by_input={"a": -1})
        for start, end in ((0.75, 1), (0.25, 0.5), (0.5, 1.5)):
            with self.assertRaises(ValueError):
                plan.add_interval(start=start, end=end, mean_volts_by_input={"a": 100})
        with self.assertRaises(ValueError):
            plan.finalize(finished_at=1, acquisition_start=0, acquisition_end=1)
        plan.add_interval(start=0.5, end=1, mean_volts_by_input={"a": 1, "b": 0.5})
        result = plan.finalize(finished_at=1, acquisition_start=0, acquisition_end=1)
        means = {v.conversion_id: v.aperture_mean_volts for v in result["raw_values"]}
        self.assertEqual(means, {"a": 0, "b": 0.5})

    def test_history_filters_metadata_before_requiring_unseen_values(self):
        conversions = [raw("past", 0, 1), raw("future", 2, 3),
                       raw("crosses_lower", -0.5, 0.5), raw("late", 0, 1)]
        returned = [sample("past_word", ("past",), available=2),
                    sample("future_word", ("future",), available=3),
                    sample("cross_word", ("crosses_lower",), available=1),
                    sample("late_word", ("late",), available=2)]
        observations = [observation("past_pixel", ("past_word",), available=3),
                        observation("future_pixel", ("future_word",), available=4),
                        observation("cross_pixel", ("cross_word",), available=2),
                        observation("late_pixel", ("late_word",), available=5)]
        plan = readout(conversions, returned, observations)
        plan.add_interval(start=0, end=1, mean_volts_by_input={"past": 0.25})
        self.assertEqual(plan.observations_available_by(1.5), ())
        self.assertEqual(plan.observations_for_history(target_time=1, history_seconds=1,
                                                      publication_deadline=2.5), ())
        # Future/cross-window/late conversions remain wholly unintegrated.
        # Their nominal timestamps cannot grant history access or force reads.
        eligible = plan.observations_for_history(target_time=1, history_seconds=1, publication_deadline=4)
        self.assertEqual(tuple(v.observation_id for v in eligible), ("past_pixel",))
        self.assertEqual(eligible[0].contributing_apertures, (("past", 0, 1),))
        self.assertEqual(eligible[0].value_volts, 1)
        self.assertEqual(plan.observations_for_history(target_time=1, history_seconds=0.5,
                                                      publication_deadline=4), ())
        with self.assertRaises(ValueError):
            readout(conversions, returned, [replace(observations[0], available_at=1.5)])


if __name__ == "__main__":
    unittest.main()
