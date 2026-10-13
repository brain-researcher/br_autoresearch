"""Small target-grid fixtures; no biological or hardware evidence."""
import unittest

import numpy as np

from development_focal_worlds import build_focal_world


def band_projection(drivers):
    frequencies = np.fft.rfftfreq(drivers.shape[1], d=1 / 500)
    coefficients = np.fft.rfft(drivers, axis=1, norm="ortho")
    coefficients[:, (frequencies < 1) | (frequencies > 100)] = 0
    return np.fft.irfft(coefficients, n=drivers.shape[1], axis=1, norm="ortho")


class DevelopmentFocalWorldTests(unittest.TestCase):
    def setUp(self):
        self.axis = (np.arange(32) + 0.5) * 300.
        self.times = np.arange(8000) / 500
        self.arguments = dict(x_um=self.axis, y_um=self.axis, times_s=self.times,
            replicate_seed=2001, primary_band_transform=band_projection,
            check=lambda: None, max_elements=1_000_000)

    def build(self, condition):
        return build_focal_world(condition, **self.arguments)

    def test_all_four_focal_templates_have_complete_support_and_requested_event_amplitude(self):
        for condition in ("F1", "F2", "F3", "F4"):
            with self.subTest(condition=condition):
                spatial, raw, report = self.build(condition)
                transformed = band_projection(raw)
                self.assertGreater(report["complete_event_count"], 0)
                self.assertEqual(len(report["events"]), len(spatial))
                for event in report["events"]:
                    index = event["mode_index"]
                    self.assertLessEqual(event["support_end_s"], self.times[-1])
                    np.testing.assert_array_equal(raw[index, self.times < event["support_start_s"]], 0)
                    np.testing.assert_array_equal(raw[index, self.times > event["support_end_s"]], 0)
                    measured = spatial[index].max() * np.ptp(transformed[index])
                    self.assertAlmostEqual(measured, report["target_event_peak_to_peak_uv"], places=9)
                    self.assertGreater(event["raw_half_height_span_s"], 0)
                self.assertFalse(report["sum_renormalized"])

    def test_uniform_and_edge_surprises_preserve_raw_preludes_and_filter_ringing(self):
        for condition, prelude in (("S1", .5), ("S2", 2.), ("S3", .5)):
            with self.subTest(condition=condition):
                _, raw, report = self.build(condition)
                np.testing.assert_array_equal(raw[:, self.times < prelude], 0)
                self.assertGreater(np.max(np.abs(band_projection(raw)[:, self.times < prelude])), 0)
                for event in report["events"]:
                    self.assertGreaterEqual(event["support_start_s"], prelude)
                    if condition == "S2":
                        x, y = event["centre_um_xy"]
                        self.assertTrue(x <= 600 or x >= 9000 or y <= 600 or y >= 9000)

    def test_four_location_surprise_merges_only_identical_spatial_modes_and_keeps_overlap(self):
        spatial, raw, report = self.build("S4")
        self.assertEqual(spatial.shape[0], 4)
        self.assertEqual(raw.shape, (4, len(self.times)))
        self.assertTrue(report["S4_identical_spatial_modes_merged_by_driver_sum"])
        np.testing.assert_array_equal(raw[:, self.times < 2.], 0)
        self.assertGreater(report["complete_event_count"], 4)
        self.assertEqual(len({tuple(event["centre_um_xy"]) for event in report["events"]}), 4)
        for mode in range(4):
            selected = [event for event in report["events"] if event["mode_index"] == mode]
            self.assertGreater(len(selected), 1)
            self.assertEqual(len({tuple(event["centre_um_xy"]) for event in selected}), 1)

    def test_paired_draws_are_reproducible_and_global_rng_is_untouched(self):
        before = np.random.get_state()
        spatial, raw, first = self.build("F2")
        _, _, other = self.build("F4")
        again_spatial, again_raw, again = self.build("F2")
        after = np.random.get_state()
        np.testing.assert_array_equal(spatial, again_spatial)
        np.testing.assert_array_equal(raw, again_raw)
        self.assertEqual(first, again)
        for left, right in zip(first["events"], other["events"]):
            self.assertEqual(left["centre_um_xy"], right["centre_um_xy"])
            self.assertEqual(left["support_start_s"], right["support_start_s"])
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(first["shared_innovation_group"], other["shared_innovation_group"])

    def test_resource_filter_and_finite_time_failures_stop_without_replacement_draw(self):
        with self.assertRaisesRegex(ValueError, "max_elements"):
            build_focal_world("F2", **{**self.arguments, "max_elements": 10000})
        with self.assertRaisesRegex(ValueError, "no complete"):
            build_focal_world("S2", **{**self.arguments, "times_s": self.times[:500]})
        with self.assertRaisesRegex(ValueError, "both polarities"):
            build_focal_world("F3", **{**self.arguments,
                "primary_band_transform": lambda raw: np.zeros_like(raw)})
        def expired():
            raise TimeoutError("fixture deadline")
        with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
            build_focal_world("F2", **{**self.arguments, "check": expired})


if __name__ == "__main__":
    unittest.main()
