"""Focused checks for choices that would change EP03's empirical comparison.

No empirical metadata, image reader, model weights or score file is loaded.
Run unittest in a coordinator-owned compute allocation, not the login shell.
"""

import copy
import unittest

import numpy as np

from grouped_evaluator import family_prototypes, grouped_relative_gain, normalize_targets, predictions
from prepare_grouped_methods import lineage_groups, metadata_exclusions, prepare, split_for_groups


class GroupedMethodsTests(unittest.TestCase):
    def test_normalization_preserves_sign_and_fixed_support(self):
        raw = np.array([[2., 4., 8., np.nan], [-2., -4., -8., np.nan]])
        z = normalize_targets(raw, np.array([True, True, True, False]))
        np.testing.assert_allclose(z.sum(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1)
        np.testing.assert_allclose(z[1], -z[0])
        with self.assertRaises(ValueError):
            normalize_targets(raw, np.ones(4, dtype=bool))
        with self.assertRaises(ValueError):
            normalize_targets(np.ones((1, 4)), np.ones(4, dtype=bool))

    def test_prototype_equal_groups_and_no_heldout_fit(self):
        a = np.array([-1., 0., 1.]) / np.sqrt(2)
        b = np.array([-1., 1., 0.]) / np.sqrt(2)
        z = np.stack([a, a, a, b, -a])
        group = ["a", "a", "a", "b", "test"]
        proto = family_prototypes(z, group, ["f"] * 5, [1] * 5, ["a", "b"])
        np.testing.assert_allclose(proto["f"], (a + b) / 2)
        z[-1] = np.nan
        np.testing.assert_allclose(family_prototypes(z, group, ["f"] * 5, [1] * 5, ["a", "b"])["f"], proto["f"])

    def test_complete_prediction_reverses_once(self):
        p = {"f": np.array([.2, -.2])}
        r = np.array([[.3, -.1], [.3, -.1]])
        base, model = predictions(p, ["f", "f"], [1, -1], r)
        np.testing.assert_allclose(base[0], -base[1])
        np.testing.assert_allclose(model[0], -model[1])
        np.testing.assert_allclose(model[0], [.5, -.3])
        with self.assertRaises(ValueError):
            predictions(p, ["missing"], [1])

    def test_ratio_after_group_loss_and_equal_group_average(self):
        z = np.zeros((3, 1))
        baseline = np.array([[1.], [3.], [2.]])
        model = np.array([[1.], [1.], [3.]])
        result = grouped_relative_gain(z, baseline, model, ["a", "a", "b"])
        # Group a: 1 - mean(1,1)/mean(1,9)=.8; b: 1-9/4=-1.25.
        self.assertAlmostEqual(result["groups"][0]["relative_gain"], .8)
        self.assertAlmostEqual(result["mean_relative_gain"], -.225)
        self.assertEqual(result["scientific_acceptance"], "not_evaluated")
        with self.assertRaises(ValueError):
            grouped_relative_gain(z, z, model, ["a", "a", "b"])

    def test_overlap_merges_transitively_without_identity_invention(self):
        edges = [
            {"accessions": ["a", "b"], "relation": "shared_participants", "evidence": "methods record"},
            {"accessions": ["b", "c"], "relation": "shared_first_level_inputs", "evidence": "input lineage"}]
        groups = lineage_groups(["a", "b", "c", "d"], edges)
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["a"], groups["d"])
        self.assertEqual(len(set(lineage_groups(["a", "b"], []).values())), 2)
        bad = copy.deepcopy(edges)
        bad[0]["relation"] = "both_have_sub01"
        with self.assertRaises(ValueError):
            lineage_groups(["a", "b", "c", "d"], bad)

    def test_fold_keeps_all_session_and_contrast_copies_together(self):
        rows = [{"map_id": "a1", "group": "a", "family": "f", "metadata_exclusions": []},
                {"map_id": "a2", "group": "a", "family": "f", "metadata_exclusions": []},
                {"map_id": "b1", "group": "b", "family": "f", "metadata_exclusions": []},
                {"map_id": "a3", "group": "a", "family": "singleton", "metadata_exclusions": []}]
        fold = split_for_groups(rows, ["a"])
        self.assertEqual(fold["train_map_ids"], ["b1"])
        self.assertEqual(fold["matched_score_map_ids"], ["a1", "a2"])
        self.assertEqual(fold["unrepresented_family_map_ids"], ["a3"])

    def test_frozen_status_and_screens_without_legacy_role_change(self):
        row = {"s": "1", "status": "proposed_rule", "flags": "", "n_inputs": "16",
               "dataset_id": "a", "partition_status": "excluded"}
        self.assertEqual(metadata_exclusions(row, "f"), [])
        row["dataset_id"] = "ds003545"
        self.assertIn("adopted_onset_timing_hold", metadata_exclusions(row, "f"))
        row["n_inputs"] = "2"
        row["flags"] = "regressor_not_convolved:x;duplicate_definition"
        reasons = metadata_exclusions(row, "f")
        self.assertIn("never_convolved_regressor", reasons)
        self.assertIn("duplicate_definition", reasons)
        self.assertIn("fitted_n_at_most_2", reasons)

    def test_preparation_retains_all_maps_and_does_not_release_scoring(self):
        spec = {"taxonomy": {"families": {"f": {"routes": ["a/task-x", "b/task-y"],
                "rationale": "methods", "assumption": "coarse family"}}, "unresolved_routes": {}},
                "lineage": {"merge_edges": [], "status": "assumed"},
                "normalization": {}, "decisions_not_adopted": {"negative_control_pass_threshold": None}}
        route = [{"route": a + "/task-" + t, "dataset_id": a, "dataset_z_maps": "1"}
                 for a, t in [("a", "x"), ("b", "y")]]
        maps = [{**r, "session": "", "contrast": "c", "task": t, "s": "1", "status": "proposed_rule",
                 "flags": "", "n_inputs": "16", "partition_status": "excluded"}
                for r, t in zip(route, ["x", "y"])]
        result = prepare(spec, route, maps, {"routes": {}})
        self.assertEqual(result["source_maps"], 2)
        self.assertEqual(result["groups_with_any_outer_comparison"], 2)
        self.assertFalse(result["baseline_target_access"])
        self.assertFalse(result["scientific_scoring"])
        self.assertTrue(all(m["lineage_status"].endswith("assumed") for m in result["maps"]))


if __name__ == "__main__":
    unittest.main()
