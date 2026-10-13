"""Three finite nuisance-adapter fixtures, not generated production worlds."""

from dataclasses import asdict, replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from acquisition_readout import Observation, PublishedValue
from development_worlds import DevelopmentWorld, RECIPE
from observation_nuisance import ObservationNuisanceBinding, SavedObservationNuisance
from raw_world_time import RawWorldTime


def nuisance_fixture(condition):
    """Small private spatial fixture with exact native time and known nuisances.

    No RNG, covariance draw, production artifact or condition panel is used.
    The N3 mask is deliberately constructed to expose clock/bit-order errors;
    its realized probability is not asserted to be a production 1% draw.
    """
    times = np.arange(8000) * .002
    raw = np.stack((np.cos(2 * np.pi * 11. * times), np.sin(2 * np.pi * 11. * times)))
    arrays = {}
    nuisance = dict(observation_side_only=True)
    if condition in ("N1", "N2"):
        frequency, rms, phase = (60., 5., .2) if condition == "N1" else (50., 50., .7)
        sine = np.sin(2 * np.pi * frequency * times + phase)
        arrays["spatially_common_line_uv"] = sine * (rms / np.sqrt(np.mean(sine**2)))
        nuisance.update(line_hz=frequency, line_rms_uv=rms, line_phase_rad=phase)
    elif condition == "N3":
        packed = np.zeros((8000, 512), dtype=np.uint8)
        packed[1, 1] = 1  # physical site8 becomes missing at exactly .002s
        packed[1, 511] = 128  # physical site4095, little-endian bit7
        packed[7999, 0] = 2  # physical site1 at the final native clock cell
        arrays["missing_sites_packed_little"] = packed
        nuisance.update(nuisance_clock_hz=500., bitorder="little", one_means="missing",
                        physical_site_shape=[64, 64])
    elif condition == "N4":
        nuisance["dead_physical_row_zero_based"] = 7
    else:
        raise ValueError("private fixture requires N1–N4")
    metadata = dict(recipe=RECIPE, condition_id=condition, replicate_seed=2001,
        world_id=f"{condition}:2001", independence_group="development_replicate_2001",
        role="development_preparation_unassigned", final_world_access=False,
        grid_shape_time_y_x=[8000, 2, 2], factor_modes=2, production_grid=False,
        details=dict(carrier_hz=11., applied_scale=1., nuisance=nuisance))
    world = DevelopmentWorld(np.ones((2, 2, 2)), raw, raw.copy(), metadata, arrays)
    model = RawWorldTime(world, max_elements=5_000_000)
    return world, model, SavedObservationNuisance(model)


def published(*, role="pixel", valid=True, value=1e-4, available=1., support=(("r:0", 0., .2),)):
    return PublishedValue("o:0", "explicit_site", role, value, valid, True,
                          .05, available, support)


class ObservationNuisanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {condition: nuisance_fixture(condition) for condition in ("N1", "N2", "N3", "N4")}

    def test_saved_line_is_added_after_decoding_without_changing_metadata_or_truth(self):
        point = .12345
        value = published()
        for condition, frequency, rms, phase in (("N1", 60., 5., .2), ("N2", 50., 50., .7)):
            world, model, adapter = self.fixtures[condition]
            truth_before = model.drivers_uv(np.array([point]))
            original = world.raw_drivers.copy()
            binding = ObservationNuisanceBinding("o:0", 8, point)
            adjusted = adapter.apply(value, binding)
            line = math.sqrt(2.) * rms * math.sin(2 * math.pi * frequency * point + phase) / 1e6
            self.assertAlmostEqual(adjusted.value_volts - value.value_volts, line, delta=1e-17)
            expected = asdict(value)
            expected["value_volts"] = adjusted.value_volts
            self.assertEqual(asdict(adjusted), expected)
            np.testing.assert_array_equal(model.drivers_uv(np.array([point])), truth_before)
            np.testing.assert_array_equal(world.raw_drivers, original)
            self.assertTrue(adapter.metadata["truth_and_evaluation_mask_unchanged"])
            self.assertFalse(adapter.metadata["physical_line_pickup_modeled"])
            # Assigned/publication time are intentionally different from the
            # explicitly bound acquisition point; neither is substituted.
            assigned_line = model.nuisance_at_pairs(np.array([value.assigned_timestamp]),
                site_indices=np.array([8]))["spatially_common_line_uv"][0] / 1e6
            self.assertNotAlmostEqual(line, assigned_line, places=10)
            marker = replace(value, role="marker")
            invalid = replace(value, valid=False, value_volts=None)
            with patch.object(model, "nuisance_at_pairs", side_effect=AssertionError("no lookup permitted")):
                self.assertIs(adapter.apply(marker, None), marker)
                self.assertIs(adapter.apply(invalid, binding), invalid)

    def test_paired_clock_lookup_and_dead_row_only_remove_observation_availability(self):
        _, model, adapter = self.fixtures["N3"]
        before = np.nextafter(.002, 0.)
        times = np.array([before, .002, .002, np.nextafter(16., 0.)])
        sites = np.array([8, 8, 4095, 1])
        # Paired calls must neither rebuild the 8000-edge clock nor allocate a
        # Cartesian result. Compare with known fixture bits and old API diagonal.
        with patch("raw_world_time.np.arange", side_effect=AssertionError("clock rebuilt")):
            actual = model.nuisance_at_pairs(times, site_indices=sites)["missing_sites"]
        self.assertEqual(actual.shape, (4,))
        np.testing.assert_array_equal(actual, [False, True, True, True])
        np.testing.assert_array_equal(actual, np.diag(model.nuisance_at(times, site_indices=sites)["missing_sites"]))
        value = published(support=(("r:0", 0., .004),))
        self.assertIs(adapter.apply(value, ObservationNuisanceBinding("o:0", 8, before)), value)
        missing = adapter.apply(value, ObservationNuisanceBinding("o:0", 8, .002))
        self.assertEqual(missing, replace(value, value_volts=None, valid=False))
        _, _, dead_row = self.fixtures["N4"]
        for site in (7 * 64, 7 * 64 + 63):
            self.assertEqual(dead_row.apply(value, ObservationNuisanceBinding("o:0", site, .002)),
                             replace(value, value_volts=None, valid=False))
        self.assertIs(dead_row.apply(value, ObservationNuisanceBinding("o:0", 8 * 64, .002)), value)
        for bad_times, bad_sites in ((np.array([16.]), np.array([8])),
                                    (np.array([.002, .003]), np.array([8])),
                                    (np.array([.002]), np.array([4096]))):
            with self.assertRaises(ValueError):
                model.nuisance_at_pairs(bad_times, site_indices=bad_sites)

    def test_binding_preflight_is_metadata_only_and_rejects_support_gaps_and_guessed_points(self):
        _, model, adapter = self.fixtures["N3"]
        observation = Observation("o:0", "explicit_site", "pixel", ("s:0",), (1.,), 1., .02, .8, 8, 2)
        support = (("r:0", 0., .1), ("r:1", .3, .4))
        with patch.object(model, "nuisance_at_pairs", side_effect=AssertionError("preflight read values")):
            adapter.validate_binding(ObservationNuisanceBinding("o:0", 8, .1), observation, support)
            for binding in (ObservationNuisanceBinding("o:0", 8, .2),
                            ObservationNuisanceBinding("o:0", 8, .8),
                            ObservationNuisanceBinding("other", 8, .02)):
                with self.assertRaises(ValueError):
                    adapter.validate_binding(binding, observation, support)
            with self.assertRaises(ValueError):
                adapter.validate_binding(ObservationNuisanceBinding("o:0", 8, .9), observation,
                                         (("r:0", 0., 1.),))
        for site, point in ((-1, .02), (4096, .02), (True, .02), (8, 16.), (8, float("nan"))):
            with self.assertRaises(ValueError):
                ObservationNuisanceBinding("o:0", site, point)
        with self.assertRaises(ValueError):
            adapter.apply(replace(published(), role="reference"), ObservationNuisanceBinding("o:0", 8, .02))


if __name__ == "__main__":
    unittest.main()
