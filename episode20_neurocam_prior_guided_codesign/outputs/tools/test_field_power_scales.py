"""Analytic estimator checks; run only inside coordinator-owned Slurm compute."""
import math
import unittest

import numpy as np

from field_power_scales import ESTIMATOR, field_power_scales


class FieldPowerScaleTests(unittest.TestCase):
    def estimate(self, truth, dx=10.0, dy=10.0, dt=1 / 512):
        return field_power_scales(truth, dx_micrometres=dx, dy_micrometres=dy,
            dt_seconds=dt, common_mask=np.ones(truth.shape[1:], dtype=bool))

    def test_on_bin_sinusoid_has_analytic_spatial_and_temporal_scales(self):
        nt, ny, nx = 512, 12, 16
        time = np.arange(nt) / 512
        phase = 2 * np.pi * (2 * np.arange(nx)[None, :] / nx + np.arange(ny)[:, None] / ny)
        truth = np.sin(2 * np.pi * 20 * time)[:, None, None] * np.cos(phase)[None, :, :]
        for dx, dy in ((10.0, 10.0), (20.0, 30.0)):
            with self.subTest(dx=dx, dy=dy):
                result = self.estimate(truth, dx, dy)
                k = math.hypot(2 / (nx * dx), 1 / (ny * dy))
                self.assertAlmostEqual(result["lambda"]["k90_cycles_per_micrometre"], k)
                self.assertAlmostEqual(result["lambda"]["micrometres"], 1 / (2 * k))
                self.assertAlmostEqual(result["tau"]["seconds"], 1 / 40)
                if dx == dy:
                    self.assertAlmostEqual(result["lambda"]["pixels"], 1 / (2 * k * dx))
                else:
                    self.assertIsNone(result["lambda"]["pixels"])
                self.assertAlmostEqual(result["lambda"]["total_power"], float(np.sum(truth**2)), places=7)
                self.assertEqual(result["estimator"], ESTIMATOR)

    def test_spatial_common_mode_is_infinite_but_zero_truth_is_undefined(self):
        temporal = np.sin(2 * np.pi * 16 * np.arange(512) / 512)
        truth = np.broadcast_to(temporal[:, None, None], (512, 8, 8))
        result = self.estimate(truth)
        self.assertEqual(result["lambda"]["status"], "infinite_zero_cutoff")
        self.assertEqual(result["lambda"]["k90_cycles_per_micrometre"], 0)
        self.assertTrue(math.isinf(result["lambda"]["micrometres"]))
        self.assertAlmostEqual(result["tau"]["f90_hz"], 16)
        for constant in (0.0, 7.0):
            result = self.estimate(np.full((64, 8, 8), constant))
            for key in ("lambda", "tau"):
                self.assertEqual(result[key]["status"], "undefined_zero_power")
            self.assertIsNone(result["lambda"]["micrometres"])
            self.assertIsNone(result["tau"]["seconds"])

    def test_tau_includes_out_of_band_truth_while_lambda_uses_primary_band(self):
        time = np.arange(512) / 512
        x = np.arange(16)
        low = np.cos(2 * np.pi * 2 * x / 16)
        high = np.cos(2 * np.pi * 5 * x / 16)
        truth = (np.sin(2 * np.pi * 20 * time)[:, None, None] * low[None, None, :]
                 + 4 * np.sin(2 * np.pi * 150 * time)[:, None, None] * high[None, None, :])
        truth = np.broadcast_to(truth, (512, 8, 16))
        result = self.estimate(truth)
        self.assertAlmostEqual(result["lambda"]["micrometres"], 40)
        self.assertEqual(result["tau"]["f90_hz"], 150)
        self.assertAlmostEqual(result["tau"]["seconds"], 1 / 300)

    def test_real_fft_doubling_does_not_double_nyquist(self):
        time = np.arange(512) / 512
        temporal = np.sqrt(20) * np.sin(2 * np.pi * 30 * time) + (-1.0) ** np.arange(512)
        truth = np.broadcast_to(temporal[:, None, None], (512, 4, 4))
        result = self.estimate(truth)
        # Ordinary-bin energy is10 and Nyquist energy1 per temporal sample.
        # Failing to double ordinary bins, or doubling Nyquist, moves f90 up.
        self.assertEqual(result["tau"]["f90_hz"], 30)
        self.assertAlmostEqual(result["tau"]["total_power"], 512 * 16 * 11, places=7)

    def test_primary_band_edges_are_inclusive_and_empty_band_is_explicit(self):
        x = np.cos(2 * np.pi * np.arange(16) / 16)
        for frequency in (1, 100):
            with self.subTest(frequency=frequency):
                time = np.sin(2 * np.pi * frequency * np.arange(512) / 512)
                truth = np.broadcast_to(time[:, None, None] * x[None, None, :], (512, 4, 16))
                result = self.estimate(truth)
                self.assertEqual(result["lambda"]["status"], "finite")
                self.assertAlmostEqual(result["lambda"]["micrometres"], 80)
        truth = np.broadcast_to(np.array([0., 1., 0., -1.])[:, None, None], (4, 4, 4))
        result = self.estimate(truth, dt=1 / 4096)
        self.assertEqual(result["lambda"]["status"], "undefined_no_primary_band_bins")
        self.assertIsNone(result["lambda"]["micrometres"])
        self.assertEqual(result["tau"]["f90_hz"], 1024)

    def test_full_grid_is_required_and_malformed_input_is_rejected(self):
        truth = np.zeros((32, 4, 4))
        kwargs = {"dx_micrometres": 10., "dy_micrometres": 10., "dt_seconds": 1 / 512,
                  "common_mask": np.ones((4, 4), dtype=bool)}
        for key, value in (("dt_seconds", 0), ("dt_seconds", [1, 2]), ("dx_micrometres", True),
                           ("dy_micrometres", math.inf), ("common_mask", np.ones((4, 4), dtype=int)),
                           ("common_mask", np.ones((3, 4), dtype=bool))):
            with self.subTest(key=key, value=str(value)), self.assertRaises(ValueError):
                field_power_scales(truth, **{**kwargs, key: value})
        for mask in (np.eye(4, dtype=bool), np.pad(np.ones((2, 2), dtype=bool), 1)):
            with self.assertRaisesRegex(ValueError, "full rectangular"):
                field_power_scales(truth, **{**kwargs, "common_mask": mask})
        for bad in (truth[0], truth[:1], truth[:, :0], truth.astype(complex), truth + np.nan):
            with self.assertRaises(ValueError):
                field_power_scales(bad, **kwargs)

    def test_equal_radial_modes_are_one_complete_power_shell(self):
        n = 16
        field = np.cos(2 * np.pi * np.arange(n)[None, :] / n) + np.cos(2 * np.pi * np.arange(n)[:, None] / n)
        temporal = np.sin(2 * np.pi * 20 * np.arange(512) / 512)
        result = self.estimate(temporal[:, None, None] * field[None, :, :])
        self.assertAlmostEqual(result["lambda"]["k90_cycles_per_micrometre"], 1 / 160)
        self.assertLess(result["lambda"]["power_fraction_below_cutoff"], 1e-12)
        self.assertAlmostEqual(result["lambda"]["power_fraction_through_cutoff"], 1)

    def test_finite_aperture_changes_an_off_bin_scale_without_padding_or_taper(self):
        time = np.sin(2 * np.pi * 20 * np.arange(512) / 512)
        field = np.cos(2 * np.pi * 2 * np.arange(32) / 32)
        truth = np.broadcast_to(time[:, None, None] * field[None, None, :], (512, 8, 32))
        full = self.estimate(truth)
        cropped = self.estimate(truth[:, :, :24])
        self.assertAlmostEqual(full["lambda"]["k90_cycles_per_micrometre"], 2 / 320)
        self.assertNotAlmostEqual(cropped["lambda"]["k90_cycles_per_micrometre"], 2 / 320)
        grid_units = cropped["lambda"]["k90_cycles_per_micrometre"] * 240
        self.assertAlmostEqual(grid_units, round(grid_units))
        self.assertEqual(cropped["grid"]["spatial_DFT_period_micrometres_yx"], [80., 240.])
        self.assertEqual(cropped["grid"]["shape"], [512, 8, 24])
        self.assertEqual(cropped["conventions"]["padding"], "none")
        self.assertEqual(cropped["conventions"]["taper"], "none")
        self.assertFalse(cropped["grid"]["crop_applied"])


if __name__ == "__main__":
    unittest.main()
