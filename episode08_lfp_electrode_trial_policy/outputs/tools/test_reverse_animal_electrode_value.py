"""Scientific fixtures for the fixed other-animal learned value transfer."""
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np

from prepare_development_trial_roles import DEVELOPMENT
from reverse_animal_electrode_value import (
    ANIMALS, MC_SEEDS, VIEWS, build_ranker_bank, fit_animal_ranker,
)


def records(view="outer", animals=None):
    animals = animals or {name: "Mihili" if i < 4 else "Chewie"
                          for i, name in enumerate(DEVELOPMENT)}
    signs = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (24, 1))
    x = np.column_stack((signs, np.zeros(96)))
    rows = []
    for i, name in enumerate(DEVELOPMENT):
        offset = (-3., -1., 1., 3.)[i % 4]
        y = 3.+2*x[:, 0]-4*x[:, 1]+offset
        rows.append({"filename": name, "animal": animals[name], "view": str(view),
            "channel_ids": list(range(2, 194, 2)), "predictors": x.tolist(),
            "value_labels": y.tolist(), "fitting_rows": 32 if view == "outer" else 24,
            "label_denominator": 64., "label_baseline_SSE": 64., "target_unit_count": 2})
    return rows, animals


class ReverseAnimalValueTest(unittest.TestCase):
    def test_all_same_animal_and_wrong_views_are_excluded_before_values(self):
        # Risk: target or another same-animal session's future labels can leak
        # into donor scales even when removed later from regression rows.
        # Deliberately reversed fixture roles also prohibit filename guessing.
        animals = {name: "Chewie" if i < 4 else "Mihili"
                   for i, name in enumerate(DEVELOPMENT)}
        rows, _ = records("0", animals)
        expected = fit_animal_ranker(rows, "Chewie", "0", animals)
        poisoned = copy.deepcopy(rows)
        for row in poisoned:
            if animals[row["filename"]] == "Chewie":
                row["predictors"], row["value_labels"] = object(), object()
        wrong, _ = records("outer", animals)
        for row in wrong:
            row["predictors"], row["value_labels"] = object(), object()
        actual = fit_animal_ranker(poisoned+wrong, "Chewie", "0", animals)
        self.assertEqual(set(actual["donor_sessions"]), set(DEVELOPMENT[4:]))
        for key in ("predictor_mean", "predictor_sd", "coefficients", "intercept"):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_four_session_weights_sum_one_and_preserve_mean_loss_alpha_one(self):
        # Risk: retaining the old half-animal mass doubles effective ridge
        # regularization after restricting to a single donor animal.
        rows, animals = records()
        for recipient in ANIMALS:
            state = fit_animal_ranker(rows, recipient, "outer", animals)
            np.testing.assert_allclose(state["record_weights"], np.full(384, 1/384), atol=1e-18)
            self.assertAlmostEqual(sum(state["record_weights"]), 1., places=13)
            self.assertEqual([r["session_weight"] for r in state["donor_weights"]], [.25]*4)
            self.assertEqual([r["electrode_weight"] for r in state["donor_weights"]], [1/384]*4)
            np.testing.assert_allclose(state["predictor_mean"], [0., 0., 0.], atol=2e-13)
            np.testing.assert_allclose(state["predictor_sd"], [1., 1., 1.], atol=2e-13)
            # Orthogonal unit-variance predictors shrink exactly by 1/2.
            np.testing.assert_allclose(state["coefficients"], [1., -2., 0.], atol=2e-13)
            self.assertAlmostEqual(state["intercept"], 3., places=12)

    def test_matching_24_and_32_row_views_cannot_substitute_for_each_other(self):
        # Risk: an outer32 donor label relation could select electrodes for a
        # target inner24 fold, invalidating its prospectively matched view.
        inner, animals = records("1")
        outer, _ = records("outer", animals)
        for row in outer:
            row["value_labels"] = (-np.asarray(row["value_labels"])).tolist()
        joined = inner+outer
        first = fit_animal_ranker(joined, "Mihili", "1", animals)
        second = fit_animal_ranker(joined, "Mihili", "outer", animals)
        np.testing.assert_allclose(first["coefficients"], [1., -2., 0.], atol=2e-13)
        np.testing.assert_allclose(second["coefficients"], [-1., 2., 0.], atol=2e-13)
        malformed = copy.deepcopy(inner)
        for row in malformed:
            row["fitting_rows"] = 32
        with self.assertRaises(ValueError):
            fit_animal_ranker(malformed, "Mihili", "1", animals)

    def test_missing_or_duplicate_other_animal_donors_do_not_trigger_fallback(self):
        # Risk: filling a missing other-animal donor with same-animal training
        # would silently turn the transfer experiment back into LOSO fitting.
        rows, animals = records()
        incomplete = rows[:-1]
        with self.assertRaises(ValueError):
            fit_animal_ranker(incomplete, "Mihili", "outer", animals)
        duplicate = rows[:-1]+[copy.deepcopy(rows[-2])]
        with self.assertRaises(ValueError):
            fit_animal_ranker(duplicate, "Mihili", "outer", animals)

    def test_original_ordinary_label_maps_are_reused_without_redrawing_identity(self):
        # Risk: new transfer-specific random labels or forced derangements
        # confound the donor-composition contrast with a different control.
        rows, animals = records("2")
        state = fit_animal_ranker(rows, "Mihili", "2", animals, replicate=1)
        reverse = fit_animal_ranker(list(reversed(rows)), "Mihili", "2", animals, replicate=1)
        self.assertEqual(state, reverse)
        for item in state["label_permutations"]:
            literal = f"ep08-learned-electrode-label-v1|20261003|{item['donor_filename']}|2|1"
            order = list(range(96))
            rng = random.Random()
            rng.seed(literal, version=2)
            rng.shuffle(order)
            self.assertEqual(item["seed_literal"], literal)
            self.assertEqual(item["donor_channel_ids"], [item["recipient_channel_ids"][i] for i in order])
        calls = []
        class IdentityRandom:
            def seed(self, value, version):
                calls.append((value, version))
            def shuffle(self, order):
                self.assert_ascending = order == list(range(96))
                if not self.assert_ascending:
                    raise AssertionError("Fresh ascending physical-channel positions required")
        with patch("reverse_animal_electrode_value.random.Random", IdentityRandom):
            identity = fit_animal_ranker(rows, "Mihili", "2", animals, replicate=1)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(version == 2 for _, version in calls))
        for item in identity["label_permutations"]:
            self.assertEqual(item["recipient_channel_ids"], item["donor_channel_ids"])
        clean = fit_animal_ranker(rows, "Mihili", "2", animals)
        np.testing.assert_array_equal(identity["coefficients"], clean["coefficients"])

    def test_exact_40_animal_view_seed_rankers_are_shared_across_targets(self):
        # Risk: fitting 160 target-specific rankers can accidentally reinsert
        # target identity/exclusion or conceal the declared inherited-fit cost.
        rows = []
        for view in VIEWS:
            part, animals = records(view)
            rows.extend(part)
        progress = {"reverse_ranker_fits": 0}
        checks = []
        with patch("reverse_animal_electrode_value.fit_animal_ranker", wraps=fit_animal_ranker) as fitter:
            bank = build_ranker_bank(rows, animals, progress, check=lambda: checks.append(True))
        expected = {(animal, view, seed) for animal in ANIMALS for view in VIEWS
                    for seed in (None, *MC_SEEDS)}
        self.assertEqual(set(bank), expected)
        self.assertEqual((len(bank), fitter.call_count, len(checks), progress["reverse_ranker_fits"]), (40, 40, 40, 40))
        self.assertEqual(len({id(state) for state in bank.values()}), 40)
        for (animal, view, seed), state in bank.items():
            self.assertEqual((state["recipient_animal"], state["view"], state["replicate"]), (animal, view, seed))
            self.assertEqual(set(state["donor_sessions"]), {n for n, a in animals.items() if a != animal})


if __name__ == "__main__":
    unittest.main()
