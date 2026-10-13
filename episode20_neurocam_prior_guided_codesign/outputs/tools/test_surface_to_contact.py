"""Six focused analytic/spatial bridge checks; root-owned compute only."""
import math
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from development_worlds import DevelopmentWorld, RECIPE
from surface_to_contact import (PadGeometry, build_contact_factors,
    integrate_spatial_modes, registered_pad_geometry)

LIMIT = 20_000_000


def axis(count):
    return (np.arange(count) + .5) * (9600. / count)


def pads(centres, dimensions):
    return PadGeometry(np.array(centres, dtype=float), np.array(dimensions, dtype=float), {})


def clamped_mean(low, high, first, last):
    area = first * max(0., min(high, first) - low)
    a, b = max(low, first), min(high, last)
    if b > a:
        area += (b * b - a * a) / 2
    area += last * max(0., high - max(low, last))
    return area / (high - low)


class SurfaceToContactTests(unittest.TestCase):
    def test_registered_geometry_orientation_counts_and_virtual_registration(self):
        for geometry_id, small in (("G0", None), ("G-I2-M", "M"), ("G-T8-M", "M"),
                                   ("G-I2-S", "S"), ("G-T8-S", "S")):
            x = registered_pad_geometry(geometry_id, short_axis="x", max_elements=LIMIT)
            y = registered_pad_geometry(geometry_id, short_axis="y", max_elements=LIMIT)
            self.assertEqual(x.centres_um_xy.shape, (4096, 2))
            np.testing.assert_array_equal(x.centres_um_xy[[0, 63, 64, -1]],
                [[75., 75.], [9525., 75.], [75., 225.], [9525., 9525.]])
            np.testing.assert_array_equal(y.dimensions_um_xy, x.dimensions_um_xy[:, ::-1])
            self.assertEqual(x.metadata["class_counts"]["L"], 4096 if small is None else 2048)
            if small:
                self.assertEqual(x.metadata["class_counts"][small], 2048)
            self.assertFalse(x.metadata["published_orientation_alignment_verified"])
            self.assertFalse(x.metadata["physical_eligibility_established"])
        with self.assertRaises(TypeError):
            registered_pad_geometry("G0", max_elements=LIMIT)
        with self.assertRaises(ValueError):
            registered_pad_geometry("G0", short_axis="radial", max_elements=LIMIT)

    def test_bilinear_integral_preserves_common_mode_and_outer_half_cell_convention(self):
        x, y = axis(8), axis(6)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        spatial = np.stack((np.ones_like(xx), xx + 2 * yy + xx * yy / 1000.))
        geometry = pads([[1., 1.], [900., 600.], [4800., 4800.], [9500., 9300.]],
                        [[2., 2.], [1200., 800.], [1700., 1300.], [200., 600.]])
        observed = integrate_spatial_modes(spatial, x_um=x, y_um=y,
            descriptors=[{"kind": "raster_bilinear"}] * 2, geometry=geometry, max_elements=LIMIT)
        expected = []
        for centre, width in zip(geometry.centres_um_xy, geometry.dimensions_um_xy):
            low, high = centre - width / 2, centre + width / 2
            mx = clamped_mean(low[0], high[0], x[0], x[-1])
            my = clamped_mean(low[1], high[1], y[0], y[-1])
            expected.append(mx + 2 * my + mx * my / 1000.)
        np.testing.assert_allclose(observed[:, 0], 1., rtol=0, atol=2e-14)
        np.testing.assert_allclose(observed[:, 1], expected, rtol=2e-14, atol=2e-10)

    def test_Gaussian_analytic_average_against_independent_tensor_quadrature_and_tail(self):
        x, y = axis(64), axis(64)
        centre, fwhm = np.array([4750., 4725.]), 150.
        sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
        raster = np.exp(-.5 * (((x[None, :] - centre[0]) / sigma)**2 + ((y[:, None] - centre[1]) / sigma)**2))
        geometry = pads([[4775., 4750.], [4700., 4770.], [5450., 4725.]],
                        [[70., 135.], [35., 70.], [35., 70.]])
        actual = integrate_spatial_modes(raster[None], x_um=x, y_um=y,
            descriptors=[{"kind": "gaussian", "centre_um_xy": centre.tolist(), "fwhm_um": fwhm}],
            geometry=geometry, max_elements=LIMIT)[:, 0]
        nodes, weights = leggauss(40)
        expected = []
        for position, dimensions in zip(geometry.centres_um_xy, geometry.dimensions_um_xy):
            qx = position[0] + dimensions[0] * nodes / 2
            qy = position[1] + dimensions[1] * nodes / 2
            values = np.exp(-.5 * (((qx[None, :] - centre[0]) / sigma)**2 + ((qy[:, None] - centre[1]) / sigma)**2))
            expected.append(float(weights @ values @ weights / 4))
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.)
        self.assertGreater(actual[-1], 0.)
        wrong = {"kind": "gaussian", "centre_um_xy": centre.tolist(), "fwhm_um": fwhm * 2}
        with self.assertRaises(ValueError):
            integrate_spatial_modes(raster[None], x_um=x, y_um=y, descriptors=[wrong],
                                    geometry=geometry, max_elements=LIMIT)

    def test_directional_wave_sinc_and_mixed_mode_no_radial_shortcut(self):
        x, y = axis(64), axis(64)
        kx, ky, phase = 1 / 300., 1 / 450., .7
        angle = 2 * np.pi * (kx * (x[None, :] - 4800.) + ky * (y[:, None] - 4800.)) + phase
        spatial = np.stack((np.cos(angle), np.sin(angle), np.ones_like(angle)))
        base = {"kx_cycles_per_um": kx, "ky_cycles_per_um": ky,
                "origin_um_xy": [4800., 4800.], "phase_rad": phase}
        descriptors = [dict(base, kind="wave_cos"), dict(base, kind="wave_sin"), {"kind": "raster_bilinear"}]
        geometry = pads([[4725., 4875.], [4725., 4875.]], [[70., 135.], [135., 70.]])
        actual = integrate_spatial_modes(spatial, x_um=x, y_um=y, descriptors=descriptors,
                                         geometry=geometry, max_elements=LIMIT)
        nodes, weights = leggauss(40)
        for i, dimensions in enumerate(geometry.dimensions_um_xy):
            qx = geometry.centres_um_xy[i, 0] + dimensions[0] * nodes / 2
            qy = geometry.centres_um_xy[i, 1] + dimensions[1] * nodes / 2
            phases = 2 * np.pi * (kx * (qx[None, :] - 4800.) + ky * (qy[:, None] - 4800.)) + phase
            expected = [weights @ np.cos(phases) @ weights / 4, weights @ np.sin(phases) @ weights / 4, 1.]
            np.testing.assert_allclose(actual[i], expected, rtol=1e-12, atol=2e-14)
        self.assertGreater(float(np.max(np.abs(actual[0, :2] - actual[1, :2]))), .001)

    def test_production_bridge_preserves_raw_amplitudes_units_roles_and_atomic_input(self):
        # Finite explicit world fixture isolates the interface; it is not a new
        # generated scientific world or a claim about production amplitudes.
        nodes = axis(384)
        centres = [[4800., 4800.], [4950., 4725.]]
        width = 600.
        sigma = width / (2 * math.sqrt(2 * math.log(2)))
        spatial = np.stack([np.exp(-.5 * (((nodes[None, :] - c[0]) / sigma)**2
                                         + ((nodes[:, None] - c[1]) / sigma)**2)) for c in centres])
        raw = np.zeros((2, 8000)); raw[:, :3] = [[100., -30., 0.], [-40., 90., 0.]]
        metadata = {"recipe": RECIPE, "production_grid": True, "grid_shape_time_y_x": [8000, 384, 384],
            "common_mask": "full_rectangle", "condition_id": "I2", "world_id": "I2:2001",
            "independence_group": "development_replicate_2001", "role": "development_preparation_unassigned",
            "details": {"centres_um_xy": centres, "spatial_fwhm_um": width}}
        world = DevelopmentWorld(spatial, raw, np.full_like(raw, 999.), metadata, {"unapplied": np.ones(1)})
        geometry = registered_pad_geometry("G-I2-M", short_axis="x", max_elements=LIMIT)
        result = build_contact_factors(world, geometry, max_elements=LIMIT)
        np.testing.assert_array_equal(result.raw_drivers, raw)
        expected = result.contact_modes @ raw[:, :3] / 1e6
        np.testing.assert_allclose(result.raw_voltage_block(0, 3, max_elements=LIMIT), expected, rtol=2e-15, atol=0.)
        raw[:] = 0.
        self.assertTrue(np.any(result.raw_drivers != 0))
        self.assertFalse(result.raw_drivers.flags.writeable)
        self.assertEqual(result.metadata["independence_group"], metadata["independence_group"])
        self.assertFalse(result.metadata["reference_subtracted"])
        self.assertFalse(result.metadata["nuisance_applied"])
        with self.assertRaises(ValueError):
            result.raw_voltage_block(0, 8000, max_elements=LIMIT)
        with self.assertRaises(ValueError):
            result.raw_voltage_block(-1, 2, max_elements=LIMIT)

    def test_missing_analytic_metadata_mismatch_geometry_and_array_bounds_refuse(self):
        x, y = axis(4), axis(4)
        geometry = pads([[75., 75.]], [[70., 135.]])
        raster = np.ones((1, 4, 4))
        with self.assertRaises(ValueError):
            integrate_spatial_modes(raster, x_um=x, y_um=y, descriptors=[{}], geometry=geometry, max_elements=LIMIT)
        with self.assertRaises(ValueError):
            integrate_spatial_modes(raster, x_um=x, y_um=y, descriptors=[{"kind": "gaussian"}],
                                    geometry=geometry, max_elements=LIMIT)
        with self.assertRaises(ValueError):
            integrate_spatial_modes(raster, x_um=x, y_um=y, descriptors=[{"kind": "raster_bilinear"}],
                                    geometry=pads([[1., 1.]], [[70., 135.]]), max_elements=LIMIT)
        with self.assertRaises(ValueError):
            integrate_spatial_modes(raster, x_um=x, y_um=y, descriptors=[{"kind": "raster_bilinear"}],
                                    geometry=geometry, max_elements=4)
        with self.assertRaises(ValueError):
            registered_pad_geometry("G0", short_axis="x", max_elements=4096)


if __name__ == "__main__":
    unittest.main()
