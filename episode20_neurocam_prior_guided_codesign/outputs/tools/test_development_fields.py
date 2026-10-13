"""Small analytic fixtures for selected primitives, not new EP20 conditions.

Run only in the coordinator's Slurm allocation. Test numbers and the explicit
periodic FFT fixture are not adopted world parameters or an evaluation filter.
"""

import math
import unittest

import numpy as np

from development_fields import (
    add_nuisance_and_missingness, combine_spatial_modes, expanding_phase,
    focal_event, gaussian_covariance_block, gaussian_spatial_envelope,
    mix_components, normalize_primary_band, planar_phase,
)


def identity_fixture(field, times):
    """Only for arrays analytically treated as already in the band in fixtures."""
    return field.copy()


def periodic_band_fixture(field, times):
    """Explicit periodic uniform-grid filter for unit-test identities only."""
    frequencies = np.fft.rfftfreq(len(times), d=times[1] - times[0])
    coefficients = np.fft.rfft(field, axis=1)
    coefficients[:, (frequencies < 1) | (frequencies > 100)] = 0
    return np.fft.irfft(coefficients, n=len(times), axis=1)


class DevelopmentFieldsTests(unittest.TestCase):
    def test_covariance_axes_blocks_and_rotation_preserve_positive_semidefinite_kernel(self):
        points = np.array([[0., 0.], [2., 0.], [0., 1.]])
        kernel = gaussian_covariance_block(
            points, points, ell_um=1, axis_ratio=2, orientation_rad=0, max_elements=32)
        np.testing.assert_allclose(np.diag(kernel), 1)
        np.testing.assert_allclose(kernel, kernel.T)
        self.assertAlmostEqual(kernel[0, 1], math.exp(-0.5))
        self.assertAlmostEqual(kernel[0, 2], math.exp(-0.5))
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-14)
        block = gaussian_covariance_block(
            points[:1], points[1:], ell_um=1, axis_ratio=2,
            orientation_rad=0, max_elements=8)
        np.testing.assert_allclose(block, kernel[:1, 1:])
        rotated = points[:, ::-1].copy()
        np.testing.assert_allclose(gaussian_covariance_block(
            rotated, rotated, ell_um=1, axis_ratio=2,
            orientation_rad=math.pi / 2, max_elements=32), kernel)

    def test_spatial_fwhm_is_minor_width_and_has_no_amplitude_calibration(self):
        points = np.array([[0., 0.], [4., 0.], [0., 2.]])
        envelope = gaussian_spatial_envelope(
            points, center_um=np.zeros(2), fwhm_um=4, axis_ratio=2,
            orientation_rad=0, max_elements=16)
        np.testing.assert_allclose(envelope, [1, 0.5, 0.5])

    def test_supplied_modes_retain_common_mode_and_linear_cancellation(self):
        modes = np.ones((3, 2))
        drivers = np.array([[1., -1., 1., -1.], [-1., 1., -1., 1.]])
        np.testing.assert_array_equal(
            combine_spatial_modes(modes, drivers, max_elements=16), np.zeros((3, 4)))
        np.testing.assert_array_equal(
            combine_spatial_modes(modes[:, :1], drivers[:1], max_elements=16),
            np.tile(drivers[0], (3, 1)))

    def test_focal_raw_prelude_and_overlap_preserve_supplied_biphasic_driver(self):
        times = np.array([9., 10., 10.25, 10.5, 11., 12., 13., 14., 15., 15.5, 15.75, 16., 17.])
        driver = times - 13
        taper = np.array([0., 0., 0.5, 1., 1., 1., 1., 1., 1., 1., 0.5, 0., 0.])
        arguments = dict(
            center_um=np.zeros(2), onset_s=10, spatial_fwhm_um=2,
            axis_ratio=1, orientation_rad=0,
            temporal_fwhm_s=2 * math.sqrt(2 * math.log(2)),
            biphasic_driver_uv=driver, outer_taper=taper, max_elements=64)
        raw = focal_event(np.array([[0., 0.], [1., 0.]]), times, **arguments)
        np.testing.assert_array_equal(raw[:, times <= 10], 0)
        np.testing.assert_array_equal(raw[:, times >= 16], 0)
        np.testing.assert_allclose(raw[1], raw[0] / 2)
        self.assertAlmostEqual(raw[0, 5], -math.exp(-0.5))
        self.assertAlmostEqual(raw[0, 7], math.exp(-0.5))
        # Summing coincident events retains overlap; it is not rejected/redrawn.
        overlap = raw + focal_event(np.array([[0., 0.], [1., 0.]]), times, **arguments)
        self.assertAlmostEqual(overlap[0, 7], 2 * math.exp(-0.5))
        bad_taper = taper.copy()
        bad_taper[0] = 0.1
        with self.assertRaisesRegex(ValueError, "finite support"):
            focal_event(np.array([[0., 0.]]), times,
                        **{**arguments, "outer_taper": bad_taper})
        with self.assertRaisesRegex(ValueError, "both biphasic"):
            focal_event(np.array([[0., 0.]]), times,
                        **{**arguments, "biphasic_driver_uv": np.ones_like(times)})

    def test_planar_translation_and_radial_expansion_obey_cyclic_wavenumber(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [0., -1.]])
        times = np.array([0., 0.25, 0.5])
        phase = planar_phase(points, times, wavelength_um=4, frequency_hz=1,
                             direction_rad=0, phase_rad=0.3, origin_um=np.zeros(2),
                             reference_time_s=0, max_elements=32)
        self.assertAlmostEqual(phase[1, 1], phase[0, 0])  # speed 4 um/s
        self.assertAlmostEqual(phase[1, 0] - phase[0, 0], math.pi / 2)
        np.testing.assert_allclose(phase[2], phase[0])
        radial = expanding_phase(points, times, center_um=np.zeros(2),
                                 wavelength_um=4, frequency_hz=1,
                                 reference_radius_um=1, reference_time_s=0,
                                 phase_rad=0.3, max_elements=32)
        np.testing.assert_allclose(radial[1], radial[2])
        np.testing.assert_allclose(radial[1], radial[3])
        self.assertAlmostEqual(radial[1, 0], 0.3)
        self.assertAlmostEqual(radial[1, 1], radial[0, 0])

    def test_band_normalization_uses_explicit_filter_and_retains_raw_support_and_ringing(self):
        times = np.arange(8, dtype=float) / 256
        raw = np.array([[0., 0., 0., 1., -1., 0., 0., 0.]])
        original = raw.copy()
        mask = np.ones(raw.shape, dtype=bool)
        result = normalize_primary_band(
            raw, times, common_mask=mask, primary_band_transform=periodic_band_fixture,
            target_uv=5, measure="rms", max_elements=32)
        self.assertAlmostEqual(np.mean(result.primary_band_uv ** 2), 25)
        np.testing.assert_array_equal(result.raw_uv[:, :3], 0)
        self.assertGreater(np.max(np.abs(result.primary_band_uv[:, :3])), 0)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_allclose(periodic_band_fixture(result.raw_uv, times), result.primary_band_uv)
        # No spatial demeaning: a spatially constant temporal field survives.
        common = normalize_primary_band(
            np.repeat(raw, 2, axis=0), times,
            common_mask=np.ones((2, 8), dtype=bool),
            primary_band_transform=periodic_band_fixture, target_uv=5,
            measure="rms", max_elements=32)
        np.testing.assert_allclose(common.primary_band_uv[0], common.primary_band_uv[1])
        self.assertAlmostEqual(np.mean(common.primary_band_uv ** 2), 25)

    def test_component_power_fraction_reports_cross_terms_without_erasing_cancellation(self):
        times = np.arange(4, dtype=float)
        broad = np.array([[1., -1., 1., -1.]])
        mask = np.ones(broad.shape, dtype=bool)
        arguments = dict(common_mask=mask, primary_band_transform=identity_fixture,
                         local_fraction=0.5, component_rms_uv=2, max_elements=16)
        opposed = mix_components(broad, -broad, times, **arguments)
        self.assertAlmostEqual(opposed.report["component_power_fraction"], 0.5)
        self.assertAlmostEqual(opposed.report["broad_power_uv2"], 2)
        self.assertAlmostEqual(opposed.report["local_power_uv2"], 2)
        self.assertAlmostEqual(opposed.report["covariance_uv2"], -2)
        self.assertAlmostEqual(opposed.report["summed_power_uv2"], 0)
        np.testing.assert_array_equal(opposed.raw_uv, 0)
        aligned = mix_components(broad, broad, times, **arguments)
        self.assertAlmostEqual(aligned.report["summed_power_uv2"], 8)
        self.assertAlmostEqual(aligned.report["summed_power_uv2"],
                               aligned.report["broad_power_uv2"]
                               + aligned.report["local_power_uv2"]
                               + aligned.report["cross_term_uv2"])
        # Nonzero means distinguish covariance from the power cross moment.
        shifted = mix_components(broad + 3, broad + 3, times, **arguments)
        self.assertLess(shifted.report["covariance_uv2"], shifted.report["cross_moment_uv2"])

    def test_zero_eligible_power_and_mask_are_not_repaired_with_defaults(self):
        times = np.arange(4, dtype=float)
        raw = np.array([[2., -2., 9., -9.]])
        mask = np.array([[True, True, False, False]])
        result = normalize_primary_band(
            raw, times, common_mask=mask, primary_band_transform=identity_fixture,
            target_uv=8, measure="peak_to_peak", max_elements=16)
        self.assertEqual(result.report["scale"], 2)
        np.testing.assert_array_equal(result.raw_uv, raw * 2)
        zero = np.zeros(raw.shape)
        with self.assertRaisesRegex(ValueError, "zero eligible"):
            normalize_primary_band(zero, times, common_mask=mask,
                                   primary_band_transform=identity_fixture,
                                   target_uv=8, measure="rms", max_elements=16)
        with self.assertRaisesRegex(ValueError, "active component"):
            mix_components(raw, zero, times, common_mask=mask,
                           primary_band_transform=identity_fixture,
                           local_fraction=0.1, component_rms_uv=2, max_elements=16)
        endpoint = mix_components(raw, zero, times, common_mask=mask,
                                  primary_band_transform=identity_fixture,
                                  local_fraction=0, component_rms_uv=2, max_elements=16)
        self.assertEqual(endpoint.report["local_power_uv2"], 0)
        self.assertEqual(endpoint.report["component_power_fraction"], 0)

    def test_dropout_and_nuisance_do_not_rewrite_truth_or_evaluation_support(self):
        truth = np.array([[1., -1.], [2., -2.]])
        nuisance = np.ones((2, 2)) * 3
        observed = np.array([[True, True], [False, False]])
        evaluated = np.ones((2, 2), dtype=bool)
        result = add_nuisance_and_missingness(
            truth, nuisance, observation_mask=observed,
            evaluation_mask=evaluated, max_elements=16)
        np.testing.assert_array_equal(result.truth_uv, truth)
        np.testing.assert_array_equal(result.evaluation_mask, evaluated)
        np.testing.assert_array_equal(result.observed_uv[0], truth[0] + 3)
        self.assertTrue(np.all(np.isnan(result.observed_uv[1])))
        truth[:] = 7
        evaluated[:] = False
        observed[:] = False
        self.assertEqual(result.truth_uv[1, 0], 2)
        self.assertTrue(result.evaluation_mask.all())
        self.assertTrue(result.observation_mask[0].all())

    def test_product_limits_and_ambiguous_inputs_fail_before_expanding_arrays(self):
        points = np.zeros((5, 2))
        with self.assertRaisesRegex(ValueError, "max_elements"):
            gaussian_covariance_block(points, points, ell_um=1, axis_ratio=1,
                                      orientation_rad=0, max_elements=10)
        with self.assertRaisesRegex(ValueError, "max_elements"):
            combine_spatial_modes(np.ones((5, 1)), np.ones((1, 5)), max_elements=10)
        with self.assertRaisesRegex(ValueError, "exact field shape"):
            normalize_primary_band(np.ones((2, 4)), np.arange(4, dtype=float),
                                   common_mask=np.ones((1, 4), dtype=bool),
                                   primary_band_transform=identity_fixture,
                                   target_uv=1, measure="rms", max_elements=16)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            normalize_primary_band(np.ones((1, 2)), np.array([1., 0.]),
                                   common_mask=np.ones((1, 2), dtype=bool),
                                   primary_band_transform=identity_fixture,
                                   target_uv=1, measure="rms", max_elements=16)
        with self.assertRaisesRegex(ValueError, "at least one"):
            gaussian_covariance_block(points, points, ell_um=1, axis_ratio=0.5,
                                      orientation_rad=0, max_elements=32)


if __name__ == "__main__":
    unittest.main()
