"""Small dense-equivalence fixtures; run in coordinator-owned Slurm only."""
import math
import unittest

import numpy as np

from field_power_scales import field_power_scales
from factorized_field_power_scales import factorized_field_power_scales


class FactorizedFieldPowerScaleTests(unittest.TestCase):
    def compare(self, spatial, temporal, *, dx=25., dy=25., dt=1 / 512):
        mask = np.ones(spatial.shape[1:], dtype=bool)
        arguments = dict(dx_micrometres=dx, dy_micrometres=dy,
                         dt_seconds=dt, common_mask=mask)
        dense = np.einsum("ryx,rt->tyx", spatial, temporal)
        expected = field_power_scales(dense, **arguments)
        actual = factorized_field_power_scales(spatial, temporal,
                                               max_elements=1_000_000, **arguments)
        self.assertEqual(actual["estimator"], expected["estimator"])
        self.assertEqual(actual["grid"], expected["grid"])
        self.assertEqual(actual["conventions"], expected["conventions"])
        for metric in ("lambda", "tau"):
            self.assertEqual(set(actual[metric]), set(expected[metric]))
            for key, wanted in expected[metric].items():
                got = actual[metric][key]
                with self.subTest(metric=metric, key=key):
                    if isinstance(wanted, float) and math.isfinite(wanted):
                        np.testing.assert_allclose(got, wanted, rtol=2e-11, atol=2e-12)
                    else:
                        self.assertEqual(got, wanted)
        return actual

    def test_nonorthogonal_modes_correlated_drivers_and_temporal_means_match_dense(self):
        rng = np.random.default_rng(201)
        spatial = rng.normal(size=(5, 7, 9))
        spatial[2] += 0.8 * spatial[0]
        temporal = rng.normal(size=(5, 257))
        temporal[1] += temporal[0] + 4.0
        self.compare(spatial, temporal, dx=25., dy=40., dt=1 / 500)

    def test_cross_terms_are_retained_for_constructive_and_destructive_interference(self):
        x = np.arange(16)
        shared = np.broadcast_to(np.cos(2 * np.pi * x / 16), (8, 16)).copy()
        orthogonal = np.broadcast_to(np.cos(2 * np.pi * 3 * x / 16), (8, 16)).copy()
        spatial = np.stack((shared, shared, orthogonal))
        time = np.arange(512) / 512
        low, high = np.sin(2 * np.pi * 20 * time), np.sin(2 * np.pi * 150 * time)
        for coefficient in (1.0, -0.75):
            with self.subTest(coefficient=coefficient):
                self.compare(spatial, np.stack((low, coefficient * low, high)))

    def test_exact_zero_cancellation_and_constant_truth_remain_undefined_without_floor(self):
        spatial = np.stack((np.ones((4, 4)), np.ones((4, 4))))
        driver = np.tile(np.array([0., 1., 0., -1.]), 32)
        cancelled = self.compare(spatial, np.stack((driver, -driver)))
        for metric in ("lambda", "tau"):
            self.assertEqual(cancelled[metric]["status"], "undefined_zero_power")
        self.compare(spatial, np.full((2, 128), 7.0))
        tiny = self.compare(spatial[:1], driver[None, :] * 1e-30)
        self.assertGreater(tiny["tau"]["total_power"], 0)

    def test_raw_out_of_band_support_nyquist_weights_and_band_edges_match_dense(self):
        x = np.arange(16)
        spatial = np.stack([np.broadcast_to(np.cos(2 * np.pi * k * x / 16), (4, 16))
                            for k in (1, 2, 3, 4, 5)])
        time = np.arange(512) / 512
        temporal = np.stack([np.sin(2 * np.pi * frequency * time)
                             for frequency in (1, 20, 100, 150)]
                            + [(-1.) ** np.arange(512)])
        temporal[3] *= 4
        actual = self.compare(spatial, temporal)
        self.assertEqual(actual["tau"]["f90_hz"], 150)
        self.assertEqual(actual["lambda"]["selected_temporal_bin_count"], 100)

    def test_finite_aperture_radial_ties_common_mode_and_missing_band_are_preserved(self):
        n = 16
        plane = (np.cos(2 * np.pi * np.arange(n)[None, :] / n)
                 + np.cos(2 * np.pi * np.arange(n)[:, None] / n))
        driver = np.sin(2 * np.pi * 20 * np.arange(512) / 512)
        self.compare(plane[None, :, :], driver[None, :])
        self.compare(plane[None, :12, :], driver[None, :])
        common = self.compare(np.ones((1, 4, 4)), driver[None, :])
        self.assertTrue(math.isinf(common["lambda"]["micrometres"]))
        empty = self.compare(np.ones((1, 4, 4)), np.array([[0., 1., 0., -1.]]), dt=1 / 4096)
        self.assertEqual(empty["lambda"]["status"], "undefined_no_primary_band_bins")

    def test_resource_shape_mask_finite_input_and_callback_refusals(self):
        spatial, temporal = np.ones((2, 4, 4)), np.ones((2, 32))
        args = dict(dx_micrometres=25., dy_micrometres=25., dt_seconds=1 / 500,
                    common_mask=np.ones((4, 4), dtype=bool), max_elements=1000)
        for key, value in (("max_elements", True), ("max_elements", 63),
                           ("common_mask", np.eye(4, dtype=bool)), ("dt_seconds", 0),
                           ("check", 7)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                factorized_field_power_scales(spatial, temporal, **{**args, key: value})
        for a, b in ((spatial.astype(complex), temporal), (spatial, temporal[:1]),
                     (spatial * np.nan, temporal), (spatial, temporal[:, :1])):
            with self.assertRaises(ValueError):
                factorized_field_power_scales(a, b, **args)
        checks = []
        def budget():
            checks.append(True)
            if len(checks) == 3:
                raise TimeoutError("fixture allocation ended")
        with self.assertRaisesRegex(TimeoutError, "allocation ended"):
            factorized_field_power_scales(spatial, temporal, **{**args, "check": budget})


if __name__ == "__main__":
    unittest.main()
