"""Six bounded raw-time checks; execute only in root's coordinated allocation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np

from development_worlds import CONDITIONS, DevelopmentWorld, _Grid, _build
from raw_world_time import RawWorldTime, REPLAY_TOLERANCE
from surface_to_contact import ContactFactors

LIMIT = 20_000_000


class RawWorldTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Private spatial fixtures retain the full selected temporal record.
        # No stored production panel, scientific payload or output file is read.
        cls.worlds = {condition: _build(condition, 2001, _Grid(48, 48),
            check=lambda: None, max_elements=LIMIT, include_scales=False) for condition in CONDITIONS}
        cls.models = {condition: RawWorldTime(world, max_elements=LIMIT)
                      for condition, world in cls.worlds.items()}

    def test_all28_native_replays_preserve_saved_scales_phase_and_roles(self):
        times = np.arange(8000) * .002
        self.assertEqual(len(self.models), 28)
        for condition, world in self.worlds.items():
            model = self.models[condition]
            actual = model.drivers_uv(times)
            peak = np.maximum(1., np.max(np.abs(world.raw_drivers), axis=1))[:, None]
            self.assertTrue(np.all(np.abs(actual - world.raw_drivers)
                                  <= REPLAY_TOLERANCE * (peak + np.abs(world.raw_drivers))), condition)
            self.assertEqual(model.metadata["independence_group"], "development_replicate_2001")
            self.assertFalse(model.metadata["raw_amplitudes_refitted"])
            self.assertFalse(model.metadata["primary_band_drivers_used"])
            self.assertFalse(model.metadata["hold_or_interpolation_selected"])
            self.assertFalse(model.metadata["production_grid"])

    def test_continuous_carriers_are_analytic_between_samples_and_wave_phase_stays_spatial(self):
        times = np.array([.00017, .1234567, .80003, 15.999999])
        for condition in ("B4", "W3", "I3", "M2"):
            details = self.worlds[condition].metadata["details"]
            if condition == "I3":
                expected = np.cos(2 * np.pi * 11. * times[None, :] + np.array(details["phases_rad"])[:, None]) * details["applied_scale"]
            elif condition == "M2":
                expected = np.stack([details[name]["applied_scale"] * trig(2 * np.pi * frequency * times)
                    for name, frequency in (("broad_normalization", 3.), ("local_normalization", 37.))
                    for trig in (np.cos, np.sin)])
            else:
                angle = 2 * np.pi * details["carrier_hz"] * times
                expected = details["applied_scale"] * np.stack((np.cos(angle), np.sin(angle)))
            np.testing.assert_allclose(self.models[condition].drivers_uv(times), expected, rtol=1e-14, atol=1e-13)
        # Duplicate and nonchronological queries must not introduce replay state.
        repeated = np.array([times[-1], times[0], times[-1]])
        result = self.models["B4"].drivers_uv(repeated)
        np.testing.assert_array_equal(result[:, 0], result[:, 2])

    def test_finite_event_support_quiet_preludes_overlaps_and_support_breakpoints(self):
        for condition in ("S1", "S2", "S3", "S4"):
            model = self.models[condition]
            events = self.worlds[condition].metadata["details"]["events"]
            first = min(event["support_start_s"] for event in events)
            query = np.array([0., first / 2, first, np.nextafter(first, 0.)])
            np.testing.assert_array_equal(model.drivers_uv(query), 0.)
            report = model.breakpoint_metadata()
            for event in events:
                self.assertIn(event["support_start_s"], report["support_boundaries_s"])
                self.assertIn(event["support_end_s"], report["support_boundaries_s"])
            self.assertEqual(model.breakpoints()[0], 0.)
            self.assertEqual(model.breakpoints()[-1], 16.)
        world, model = self.worlds["F2"], self.models["F2"]
        events = world.metadata["details"]["events"]
        first, second = events[:2]
        self.assertLess(second["support_start_s"], first["support_end_s"])
        time = (second["support_start_s"] + first["support_end_s"]) / 2
        expected = np.zeros(len(world.raw_drivers))
        for event in events:
            if event["support_start_s"] < time < event["support_end_s"]:
                sigma = event["nominal_envelope_fwhm_s"] / (2 * np.sqrt(2 * np.log(2)))
                z = (time - event["support_start_s"]) / sigma - 3
                u = 2 * (abs(z) - 2.5)
                taper = 1. if abs(z) <= 2.5 else 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
                expected[event["mode_index"]] += -z * np.exp(-.5 * z*z) * taper * event["applied_scale"]
        actual = model.drivers_uv(np.array([time]))[:, 0]
        self.assertGreater(np.count_nonzero(actual), 1)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
        quiet = self.worlds["S2"].metadata["details"]["minimum_local_prelude_s"]
        native_quiet = np.arange(8000) * .002 < quiet
        self.assertGreater(float(np.max(np.abs(self.worlds["S2"].primary_band_drivers[:, native_quiet]))), 0.)
        np.testing.assert_array_equal(self.models["S2"].drivers_uv(np.array([.001, quiet / 2])), 0.)

    def test_M4_saved_step_coefficients_apply_right_continuously_without_filter_ringing(self):
        details = self.worlds["M4"].metadata["details"]
        model = self.models["M4"]
        times = np.array([1 - 1e-9, 1., 1 + 1e-9, 2., 15.999999])
        phase, envelope_phase = details["common_carrier_phase_rad"], details["common_envelope_phase_rad"]
        expected = []
        for name, frequency in (("broad_switch_normalization", 3.), ("local_switch_normalization", 37.)):
            amplitudes = np.array(details[name]["interval_amplitudes"])[np.floor(times).astype(int)]
            envelope = 1 + .25 * np.cos(2 * np.pi * .5 * times + envelope_phase)
            expected.extend([amplitudes * envelope * trig(2 * np.pi * frequency * times + phase)
                             for trig in (np.cos, np.sin)])
        actual = model.drivers_uv(times)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)
        self.assertGreater(float(np.linalg.norm(actual[:, 1] - actual[:, 0])), .1)
        self.assertEqual(model.breakpoint_metadata()["raw_amplitude_discontinuities_s"], list(map(float, range(1, 16))))
        self.assertEqual(len(model.breakpoints()), 17)

    def test_nuisance_clock_line_replay_and_dead_row_stay_separate(self):
        times = np.array([0., np.nextafter(.002, 0.), .002, np.nextafter(.002, 1.), 15.999999])
        sites = np.array([0, 1, 7, 8, 4095])
        for condition in ("N1", "N2"):
            actual = self.models[condition].nuisance_at(np.arange(8000) * .002)
            np.testing.assert_allclose(actual["spatially_common_line_uv"],
                self.worlds[condition].nuisance_arrays["spatially_common_line_uv"], rtol=1e-14, atol=1e-13)
            self.assertFalse(actual["applied_to_truth"])
            np.testing.assert_allclose(self.models[condition].drivers_uv(times), self.models["B2"].drivers_uv(times))
        packed = self.worlds["N3"].nuisance_arrays["missing_sites_packed_little"]
        rows = np.array([0, 0, 1, 1, 7999])
        expected = np.unpackbits(packed[rows], axis=1, bitorder="little")[:, sites].astype(bool)
        np.testing.assert_array_equal(self.models["N3"].nuisance_at(times, site_indices=sites)["missing_sites"], expected)
        self.assertEqual(len(self.models["N3"].breakpoints()), 2)
        self.assertEqual(len(self.models["N3"].breakpoints(include_nuisance=True)), 8001)
        dead = self.worlds["N4"].metadata["details"]["nuisance"]["dead_physical_row_zero_based"]
        selected = np.array([dead * 64, dead * 64 + 63, ((dead + 1) % 64) * 64])
        actual = self.models["N4"].nuisance_at(times, site_indices=selected)["missing_sites"]
        np.testing.assert_array_equal(actual, np.tile([True, True, False], (len(times), 1)))

    def test_contact_units_immutability_bounds_and_inconsistent_replay_refusal(self):
        world, model = self.worlds["B2"], self.models["B2"]
        matrices = np.tile([1., -.5], (4096, 1))
        metadata = {"world_id": "B2:2001", "independence_group": "development_replicate_2001",
                    "role": "development_preparation_unassigned"}
        contact = ContactFactors(matrices, world.raw_drivers.copy(), metadata)
        times = np.array([.12345, 2.001])
        np.testing.assert_allclose(model.contact_volts(contact, times), matrices @ model.drivers_uv(times) / 1e6,
                                   rtol=2e-15, atol=0.)
        changed = deepcopy(world)
        independent = RawWorldTime(changed, max_elements=LIMIT)
        expected = independent.drivers_uv(times)
        changed.metadata["details"]["applied_scale"] *= 2
        changed.raw_drivers[:] = 0
        np.testing.assert_array_equal(independent.drivers_uv(times), expected)
        with self.assertRaises(ValueError):
            RawWorldTime(changed, max_elements=LIMIT)
        for invalid in (np.array([-1.]), np.array([16.]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                model.drivers_uv(invalid)
        with self.assertRaises(ValueError):
            model.contact_volts(contact, np.arange(8000) * .002)
        with self.assertRaises(ValueError):
            self.models["N3"].nuisance_at(times)
        mismatch = ContactFactors(matrices, world.primary_band_drivers + 1., metadata)
        with self.assertRaises(ValueError):
            model.contact_volts(mismatch, times)
        missing = deepcopy(world)
        del missing.metadata["details"]["applied_scale"]
        with self.assertRaises((ValueError, KeyError)):
            RawWorldTime(missing, max_elements=LIMIT)
        bound = model.bind_contact_factors(contact)
        before_mutation = bound.volts(times)
        matrices[:] = 0.
        contact.raw_drivers[:] = 0.
        metadata["world_id"] = "different_world"
        # Binding owns modes and the temporal recipe: mutation of original
        # arrays/metadata cannot alter the already validated interval provider.
        np.testing.assert_array_equal(bound.volts(times), before_mutation)
        with self.assertRaises(ValueError):
            model.bind_contact_factors(contact)


if __name__ == "__main__":
    unittest.main()
