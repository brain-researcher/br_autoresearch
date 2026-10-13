"""Focused selected-world method checks; run only in root's compute allocation.

Smaller private fixture grids test algebra/roles, never authorize a lower-fidelity
production world. The public builder has no grid/seed-law override.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from development_worlds import (
    CONDITIONS, DevelopmentWorld, GenerationDeadline, MAX_ELEMENTS, _Grid,
    _axis_root, _broad, _build, _interaction, _mixture, _nuisance, _project,
    _wave, generate_panel,
)


LIMITS = {"check": lambda: None, "max_elements": MAX_ELEMENTS}


def direct(spatial, drivers):
    return spatial.reshape(len(spatial), -1).T @ drivers


class SelectedDevelopmentWorldTests(unittest.TestCase):
    def test_covariance_psd_approximation_pairing_and_registered_amplitude(self):
        coordinate = (np.arange(24) + .5) * 400.
        root, report = _axis_root(coordinate, 4800., **LIMITS)
        expected = np.exp(-.5 * ((coordinate[:, None] - coordinate[None, :]) / 4800.)**2)
        np.testing.assert_allclose(root @ root.T, expected, rtol=2e-13, atol=2e-13)
        self.assertLess(report["max_absolute_covariance_reconstruction_error"], 2e-13)
        grid = _Grid(24, 24, 1000)
        first = _broad("B4", 2001, grid, **LIMITS)
        repeated = _broad("B4", 2001, grid, **LIMITS)
        np.testing.assert_array_equal(first[0], repeated[0])
        np.testing.assert_array_equal(first[1], repeated[1])
        self.assertEqual(first[2]["y_covariance"]["ell_um"], 2400.)
        self.assertEqual(first[2]["x_covariance"]["ell_um"], 600.)
        band = _project(first[1], grid.dt, **LIMITS)
        self.assertAlmostEqual(float(np.sqrt(np.mean(direct(first[0], band)**2))), 10., places=10)
        self.assertFalse(np.array_equal(first[0], _broad("B4", 2002, grid, **LIMITS)[0]))

    def test_wave_factorization_matches_continuous_phase_and_paired_rotation(self):
        grid = _Grid(12, 10, 1000)
        for condition in ("W1", "W2", "W3", "W4"):
            spatial, raw, report = _wave(condition, 2001, grid, **LIMITS)
            xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
            direction = math.radians(report["direction_degrees"])
            projected = (xx - 4800.) * math.cos(direction) + (yy - 4800.) * math.sin(direction)
            phase = 2 * np.pi * (projected.ravel()[:, None] / report["wavelength_um"]
                                - report["carrier_hz"] * grid.times) + report["phase_rad"]
            expected = report["applied_scale"] * np.cos(phase)
            np.testing.assert_allclose(direct(spatial, raw), expected, rtol=2e-12, atol=2e-11)
        left = _wave("W3", 2001, grid, **LIMITS)[2]
        right = _wave("W4", 2001, grid, **LIMITS)[2]
        self.assertEqual(left["phase_rad"], right["phase_rad"])
        self.assertEqual(left["wavelength_um"], 100000. / 37.)

    def test_mixture_power_fractions_cross_terms_and_shared_draws(self):
        grid = _Grid(16, 16, 1000)
        previous_modes = None
        for condition, q in (("M1", .1), ("M2", .5), ("M3", .9)):
            modes, raw, report = _mixture(condition, 2001, grid, **LIMITS)
            if previous_modes is not None:
                np.testing.assert_array_equal(modes, previous_modes)
            previous_modes = modes.copy()
            band = _project(raw, grid.dt, **LIMITS)
            broad, local = direct(modes[:2], band[:2]), direct(modes[2:], band[2:])
            self.assertAlmostEqual(float(np.mean(broad**2)), 2500 * (1 - q), places=8)
            self.assertAlmostEqual(float(np.mean(local**2)), 2500 * q, places=8)
            self.assertAlmostEqual(report["whole_record_mixture"]["uncentred_cross_moment_uv2"],
                                   float(np.mean(broad * local)), places=9)
            self.assertAlmostEqual(report["whole_record_mixture"]["sum_power_uv2"],
                                   float(np.mean((broad + local)**2)), places=8)

    def test_M4_achieved_post_filter_interval_powers_and_raw_switch_spectrum(self):
        grid = _Grid(12, 12)
        modes, raw, report = _mixture("M4", 2001, grid, **LIMITS)
        band = _project(raw, grid.dt, **LIMITS)
        broad, local = direct(modes[:2], band[:2]), direct(modes[2:], band[2:])
        for interval in range(16):
            keep = (grid.times >= interval) & (grid.times < interval + 1)
            q = .1 if interval % 2 == 0 else .9
            np.testing.assert_allclose([np.mean(broad[:, keep]**2), np.mean(local[:, keep]**2)],
                                       [2500 * (1 - q), 2500 * q], rtol=1e-6, atol=0.)
            self.assertAlmostEqual(report["interval_mixtures"][interval]["local_component_power_fraction"], q, places=6)
        self.assertTrue(report["broad_switch_normalization"]["solver_success"])
        self.assertTrue(report["local_switch_normalization"]["solver_success"])
        frequencies = np.fft.rfftfreq(grid.nt, grid.dt)
        self.assertGreater(float(np.sum(np.abs(np.fft.rfft(raw)[:, frequencies > 100])**2)), 0.)
        self.assertGreater(float(np.max(np.abs(raw - band))), 0.)

    def test_interaction_geometry_phase_and_full_sum_normalization(self):
        grid = _Grid(24, 24, 1000)
        for condition, distance in (("I1", 4800.), ("I2", 300.), ("I3", 1200.), ("I4", 300.)):
            spatial, raw, report = _interaction(condition, 2001, grid, **LIMITS)
            centres = np.array(report["centres_um_xy"])
            distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
            self.assertAlmostEqual(float(distances[distances > 0].min()), distance, places=8)
            self.assertTrue(np.all((centres >= 0) & (centres <= 9600)))
            band = _project(raw, grid.dt, **LIMITS)
            self.assertAlmostEqual(float(np.sqrt(np.mean(direct(spatial, band)**2))), 50., places=9)
            phases = report["phases_rad"]
            if condition == "I2":
                self.assertEqual(phases[0], phases[1])
            if condition == "I3":
                self.assertAlmostEqual(phases[1] - phases[0], np.pi)

    def test_observation_nuisances_preserve_exact_B2_truth_and_clock(self):
        grid = _Grid(12, 12, 1000)
        truth = _build("B2", 2001, grid, include_scales=False, **LIMITS)
        for condition in ("N1", "N2", "N3", "N4"):
            world = _build(condition, 2001, grid, include_scales=False, **LIMITS)
            np.testing.assert_array_equal(world.spatial_modes, truth.spatial_modes)
            np.testing.assert_array_equal(world.raw_drivers, truth.raw_drivers)
            np.testing.assert_array_equal(world.primary_band_drivers, truth.primary_band_drivers)
            self.assertEqual(world.metadata["independence_group"], "development_replicate_2001")
            nuisance = world.metadata["details"]["nuisance"]
            self.assertTrue(nuisance["observation_side_only"])
            if condition in ("N1", "N2"):
                line = world.nuisance_arrays["spatially_common_line_uv"]
                self.assertAlmostEqual(float(np.sqrt(np.mean(line**2))), 5. if condition == "N1" else 50.)
            if condition == "N3":
                packed = world.nuisance_arrays["missing_sites_packed_little"]
                actual = np.unpackbits(packed, axis=1, bitorder="little")
                self.assertEqual(actual.shape, (1000, 4096))
                self.assertEqual(int(actual.sum()), nuisance["realized_missing_count"])
                self.assertEqual(nuisance["nuisance_clock_hz"], 500.)
                repeat = _nuisance("N3", 2001, grid, **LIMITS)[0]
                np.testing.assert_array_equal(packed, repeat["missing_sites_packed_little"])
            if condition == "N4":
                self.assertIn(nuisance["dead_physical_row_zero_based"], range(64))

    def test_complete_catalogue_dispatch_keeps_unassigned_paired_roles(self):
        # Agent's separate F/S checks establish finite support and isolated-event
        # amplitude; this checks that the retained panel actually integrates them.
        self.assertEqual(len(CONDITIONS), 28)
        grid = _Grid(48, 48)
        calls = []
        for condition in CONDITIONS:
            world = _build(condition, 2001, grid, check=lambda: calls.append(1),
                           max_elements=MAX_ELEMENTS, include_scales=False)
            self.assertEqual(world.metadata["world_id"], f"{condition}:2001")
            self.assertEqual(world.metadata["role"], "development_preparation_unassigned")
            self.assertTrue(world.metadata["all_same_seed_conditions_must_share_role"])
            self.assertFalse(world.metadata["production_grid"])
            self.assertFalse(world.metadata["physical_DAQ_calibration_claim"])
            self.assertGreater(world.metadata["achieved_primary_band_rms_uv"], 0.)
            self.assertEqual(world.raw_drivers.shape, world.primary_band_drivers.shape)
        self.assertGreater(len(calls), 28)
        with self.assertRaises(ValueError):
            _build("FINAL1", 2001, grid, **LIMITS)
        with self.assertRaises(ValueError):
            _build("B1", 9000, grid, **LIMITS)

    def test_container_completion_deadline_and_failure_preserve_exact_remaining_ids(self):
        def make(condition, seed, **kwargs):
            if condition == "B2":
                raise GenerationDeadline("fixture deadline")
            return DevelopmentWorld(np.ones((1, 2, 2)), np.ones((1, 4)), np.ones((1, 4)),
                {"world_id": f"{condition}:{seed}", "power_scales": {"lambda": float("inf")}}, {})

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "panel"
            summary = generate_panel(output, [("B1", 2001), ("B2", 2001), ("B3", 2001)], builder=make)
            self.assertEqual(summary["status"], "deadline_partial")
            self.assertEqual(summary["completed_world_ids"], ["B1:2001"])
            self.assertEqual(summary["remaining_world_ids"], ["B2:2001", "B3:2001"])
            self.assertEqual(json.loads((output / "summary.json").read_text())["interrupted_world_id"], "B2:2001")
            with h5py.File(output / "development_worlds.h5", "r") as container:
                self.assertTrue(container["B1_seed_2001"].attrs["complete"])
                report = json.loads(container["B1_seed_2001/metadata_json"][()])
                self.assertEqual(report["power_scales"]["lambda"], "+Infinity")
                self.assertNotIn("B2_seed_2001", container)
            with self.assertRaises(FileExistsError):
                generate_panel(output, [("B1", 2001)], builder=make)

            def failed(condition, seed, **kwargs):
                raise ValueError("fixture unresolved draw")

            summary = generate_panel(Path(temporary) / "failed", [("M4", 2001)], builder=failed)
            self.assertEqual(summary["status"], "unresolved")
            self.assertEqual(summary["remaining_world_ids"], ["M4:2001"])
            self.assertEqual(summary["failed_world_id"], "M4:2001")


if __name__ == "__main__":
    unittest.main()
