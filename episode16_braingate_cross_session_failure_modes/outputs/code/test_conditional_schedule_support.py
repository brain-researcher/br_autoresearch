"""Focused adopted-rule checks; run only in coordinator-owned compute."""
import unittest

import numpy as np

from apply_conditional_schedule_support import (
    compatibility_components, contrast_information, geometry_with_spacing,
    match_geometry, support_gates,
)


GEOMETRY = {"median_spacing_fraction": 0.001, "minimum_spacing_fraction": 0.01}
SUPPORT = {"condition_number_maximum": 100.0, "residual_L2_minimum": 0.1,
           "retained_energy_minimum": 0.1, "maximum_node_share": 0.25}


class AdoptedRules(unittest.TestCase):
    def test_allowed_transform_and_forbidden_rotation_reflection(self):
        points = np.array([[0., 0.], [1., 0.], [0., 2.], [3., 1.]])
        left = geometry_with_spacing(points)
        moved = geometry_with_spacing(7.0 * points + np.array([12., -19.]))
        self.assertEqual(match_geometry(left, moved, GEOMETRY)["state"], "compatible")
        rotated = geometry_with_spacing(points @ np.array([[0., -1.], [1., 0.]]))
        reflected = geometry_with_spacing(points * np.array([-1., 1.]))
        self.assertNotEqual(match_geometry(left, rotated, GEOMETRY)["state"], "compatible")
        self.assertNotEqual(match_geometry(left, reflected, GEOMETRY)["state"], "compatible")

    def test_near_duplicates_are_preserved_and_unresolved(self):
        points = [[0., 0.], [1., 0.], [1. + 2e-15, 0.], [0., 1.]]
        layout = geometry_with_spacing(points)
        self.assertEqual(len(layout["points"]), 4)
        self.assertTrue(layout["near_duplicate_distinctness_unresolved"])
        self.assertEqual(match_geometry(layout, layout, GEOMETRY)["state"],
                         "unresolved_near_duplicate_distinctness")

    def test_nontransitive_matches_do_not_create_greedy_subgroups(self):
        result = compatibility_components([0, 1, 2, 3], [(0, 1), (1, 2)])
        self.assertEqual(result, [{"nodes": [0, 1, 2], "complete_clique": False},
                                  {"nodes": [3], "complete_clique": True}])

    def test_exact_geometric_boundary_is_unresolved(self):
        # Exact binary fractions isolate the comparison from floating jitter.
        left = {"points": np.array([[0., 0.], [1., 0.]]), "minimum_spacing": 1.,
                "median_spacing": 1., "near_duplicate_distinctness_unresolved": False}
        right = dict(left, points=left["points"] + np.array([0.125, 0.]))
        rule = {"median_spacing_fraction": 0.125, "minimum_spacing_fraction": 0.5}
        self.assertEqual(match_geometry(left, right, rule)["state"], "unresolved_exact_tolerance_boundary")

    def test_weighted_residual_information_known_answer(self):
        weights = np.full(4, 0.25)
        nuisance = np.sqrt(weights)[:, None]
        indicator = np.array([0., 0., 1., 1.])
        edges = [(0, 1, "near"), (0, 2, "near"), (3, 4, "long"), (3, 5, "long")]
        result = contrast_information(nuisance, indicator, weights, edges)
        self.assertAlmostEqual(result["residual_L2"], 0.5)
        self.assertAlmostEqual(result["retained_energy_fraction"], 1.)
        self.assertAlmostEqual(result["node_share_sum"], 2.)
        self.assertAlmostEqual(result["node_shares"][0], 0.5)
        self.assertAlmostEqual(result["node_shares"][3], 0.5)
        gates = support_gates(True, 2., result, True, SUPPORT)
        self.assertFalse(gates["node_share_strictly_below_maximum"])
        self.assertTrue(gates["retained_energy_strictly_above_minimum"])

    def test_exact_threshold_equality_is_unresolved(self):
        result = {"residual_L2": 0.1, "retained_energy_fraction": 0.1, "maximum_node_share": 0.25}
        gates = support_gates(True, 100., result, True, SUPPORT)
        self.assertTrue(gates["full_constrained_rank"])
        self.assertTrue(gates["both_lag_classes"])
        self.assertFalse(any(value for key, value in gates.items()
                             if key not in {"full_constrained_rank", "both_lag_classes"}))


if __name__ == "__main__":
    unittest.main()
