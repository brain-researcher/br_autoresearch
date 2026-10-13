"""Focused finite-panel and grouped-fold coverage checks; compute only."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from model_recipes import reference_jobs, reference_recipes, self_paced_folds, way_inner_folds


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((Path(__file__).resolve().parents[1] /
            "model_recipe_registry.json").read_text())

    def test_complete_five_family_twelve_configuration_three_seed_panel(self):
        recipes = list(reference_recipes(self.registry))
        self.assertEqual(len(recipes), 60)
        self.assertEqual(len({r.recipe_id for r in recipes}), 60)
        for family in self.registry["reference_grid"]["family_variants"]:
            self.assertEqual(sum(r.family == family for r in recipes), 12)
        self.assertTrue(all(r.seeds == (1901, 1902, 1903) for r in recipes))
        self.assertEqual(len(list(reference_jobs(self.registry))), 60 * 3 * 4)
        selected = [r.recipe_id for r in recipes if r.recipe_id.endswith(".c00")]
        self.assertEqual(len(list(reference_jobs(self.registry, final_surrogates=True,
            selected_recipe_ids=selected))), 5 * 3 * 10)
        self.assertEqual(len(list(reference_jobs(self.registry, factorial=True,
            selected_recipe_ids=selected))), 5 * 3 * 4)
        with self.assertRaises(ValueError):
            list(reference_jobs(self.registry, factorial=True))
        self.assertEqual(self.registry["matched_conditions"]["factorial"]["Q00"], [0, 0])

    def test_all_fifteen_people_stay_grouped_and_no_held_out_enters(self):
        expected = {"sub-02", "sub-03", "sub-04", "sub-06", "sub-08",
                    "sub-09", "sub-11", "sub-12", "sub-14", "sub-15",
                    "sub-17", "sub-18", "sub-20", "sub-21", "sub-22"}
        folds = self_paced_folds(self.registry)
        self.assertEqual([len(f.fit) for f in folds], [12] * 5)
        self.assertEqual({p for f in folds for p in f.validation}, expected)
        self.assertEqual(sum(len(f.validation) for f in folds), 15)
        for fold in folds:
            self.assertFalse(set(fold.fit) & set(fold.validation))
            self.assertEqual(set(fold.fit) | set(fold.validation), expected)

    def test_way_feedback_and_held_out_never_enter_inner_folds(self):
        for participant in range(1, 13):
            folds = way_inner_folds(self.registry, participant)
            self.assertEqual(len(folds), 6)
            self.assertEqual({f.validation[0] for f in folds}, set(range(1, 7)))
            for fold in folds:
                self.assertFalse(set(fold.fit) & set(fold.validation))
                self.assertEqual(set(fold.fit) | set(fold.validation), set(range(1, 7)))
        with self.assertRaises(ValueError):
            way_inner_folds(self.registry, 13)


if __name__ == "__main__":
    unittest.main()
