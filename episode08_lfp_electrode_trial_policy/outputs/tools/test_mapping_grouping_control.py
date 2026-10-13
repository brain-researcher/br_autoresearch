"""Scientific checks of count-preserving acquired-row utility grouping."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from arrival_order_reader import OrderTrajectory
from development_design import ELECTRODE_COUNTS
from direction_training import residual_statistics
from mapping_grouping_control import GroupingDesign
from trial_allocation import CHECKPOINTS


def fixture(order=None, zeros=False):
    rng = np.random.default_rng(83101)
    labels = np.repeat(np.arange(8), 4)
    x = np.zeros((32, 6)) if zeros else rng.normal(size=(32, 6))
    y = rng.normal(size=(32, 3))
    ids = np.array([f"trial{i}" for i in range(32)])
    stats = residual_statistics(x, y, labels)
    indices = np.arange(32) if order is None else np.asarray(order)
    state = GroupingDesign(x[indices], labels[indices], stats, .03, ids[indices].tolist(),
                           "Mihili_fixture.mat", "original", 8, 2, 1)
    return state, x, y, labels, ids, stats


class MappingGroupingControlTest(unittest.TestCase):
    def test_count_matrix_and_weighted_utility_invariants(self):
        # Risk: pseudo labels must change utility grouping, not the acquired
        # design, actual labels, legal counts or fixed initializer coordinates.
        state, _, _, _, _, _ = fixture()
        for j in range(6):
            direction = j % 3
            state.observe(np.arange(6, dtype=float)+j, direction, f"later{j}")
        before = {key: np.array(getattr(state, key), copy=True)
                  for key in ("inverse", "z", "z0", "labels", "direction_mean", "mean", "scale")}
        counts = np.bincount(state.labels, minlength=8)
        chosen, record = state.choose(counts.copy())
        canonical_ids = list(map(str, record["canonical_acquired_trial_ids"]))
        pseudo = np.asarray(record["pseudo_direction_labels"])
        self.assertEqual(canonical_ids, sorted(map(str, state.trial_ids)))
        np.testing.assert_array_equal(np.bincount(pseudo, minlength=8), counts)
        inverse = np.linalg.inv(state.eta*np.eye(state.z.shape[1])+state.z.T@state.z)
        gains = np.array([np.sum((state.z0@inverse@z)**2)/(32*(1+z@inverse@z)) for z in state.z])
        by_id = {str(t): gain for t, gain in zip(state.trial_ids, gains)}
        canonical_gains = np.array([by_id[t] for t in canonical_ids])
        expected = np.array([canonical_gains[pseudo == d].mean() for d in range(8)])
        np.testing.assert_allclose(record["utility"], expected, rtol=2e-11, atol=1e-13)
        self.assertAlmostEqual(float(counts@np.asarray(record["utility"])), float(gains.sum()), places=11)
        self.assertEqual(chosen, min(np.flatnonzero(counts < 12), key=lambda d: (-expected[d], counts[d], d)))
        for key, original in before.items():
            np.testing.assert_array_equal(getattr(state, key), original)

    def test_assignment_is_independent_of_acquisition_array_order_and_prior_rng_calls(self):
        # Risk: incidental acquired-row ordering or earlier draws must not
        # alter the registered original-ID based randomized utility grouping.
        first, _, _, _, _, _ = fixture()
        order = np.random.default_rng(83102).permutation(32)
        other, _, _, _, _, _ = fixture(order=order)
        expected_choice, expected = first.choose(np.full(8, 4, dtype=int))
        unrelated = random.Random(991)
        for _ in range(137):
            unrelated.random()
            random.random()
        actual_choice, actual = other.choose(np.full(8, 4, dtype=int))
        self.assertEqual(list(map(str, expected["canonical_acquired_trial_ids"])),
                         list(map(str, actual["canonical_acquired_trial_ids"])))
        self.assertEqual(expected["pseudo_direction_labels"], actual["pseudo_direction_labels"])
        self.assertEqual(expected["seed_literal"],
                         "ep08-mapping-grouping-control-v1|20261003|Mihili_fixture.mat|original|8|2|1|32")
        self.assertEqual(expected["seed_literal"], actual["seed_literal"])
        np.testing.assert_allclose(expected["utility"], actual["utility"], rtol=2e-12, atol=1e-13)
        self.assertEqual(expected_choice, actual_choice)
        repeated_choice, repeated = first.choose(np.full(8, 4, dtype=int))
        self.assertEqual(expected_choice, repeated_choice)
        self.assertEqual(expected["pseudo_direction_labels"], repeated["pseudo_direction_labels"])

    def test_exhausted_directions_stay_in_grouping_but_cannot_be_chosen(self):
        # Risk: deleting exhausted groups changes the randomization, while
        # using pseudo availability can request a future row past its queue.
        state, _, _, _, _, _ = fixture(zeros=True)
        for j in range(8):
            state.observe(np.zeros(6), 0, f"exhaust0_{j}")
        counts = np.bincount(state.labels, minlength=8)
        chosen, record = state.choose(counts)
        np.testing.assert_array_equal(np.bincount(record["pseudo_direction_labels"], minlength=8), counts)
        self.assertEqual(counts[0], 12)
        self.assertEqual(chosen, 1)
        self.assertEqual(record["utility"], [0.]*8)

    def test_chance_unchanged_grouping_is_accepted_after_one_shuffle(self):
        # Risk: rejection of an unchanged draw silently turns ordinary
        # count-preserving random assignment into a different null distribution.
        state, _, _, _, _, _ = fixture()
        label_by_id = {str(t): int(d) for t, d in zip(state.trial_ids, state.labels)}
        true_canonical_labels = [label_by_id[t] for t in sorted(label_by_id)]
        class UnchangedDraw:
            calls = 0
            def seed(self, literal, version):
                self.literal, self.version = literal, version
            def shuffle(self, values):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("A chance unchanged assignment must not be redrawn")
                values[:] = true_canonical_labels
        draw = UnchangedDraw()
        with patch("mapping_grouping_control.random.Random", return_value=draw):
            _, record = state.choose(np.full(8, 4, dtype=int))
        self.assertEqual(draw.calls, 1)
        self.assertEqual(draw.version, 2)
        self.assertEqual(record["pseudo_direction_labels"], true_canonical_labels)

    def test_only_chosen_next_row_is_exposed_and_true_targets_labels_are_unchanged(self):
        # Risk: a grouping control must not read another history's future row
        # or apply pseudo labels to the actual decoder features/targets.
        pools = [list(range(d*12, (d+1)*12)) for d in range(8)]
        role = {"fitting_pool_by_direction": pools}
        calls = []
        def read(ids, channels):
            calls.append((tuple(ids), tuple(channels)))
            return np.array([[t, t*t, -t] for t in ids], dtype=float), np.array([[t, 2*t] for t in ids], dtype=float)
        initial_ids = [t for pool in pools for t in pool[:4]]
        x, y = read(initial_ids, [3])
        calls.clear()
        labels = np.repeat(np.arange(8), 4)
        history = OrderTrajectory(SimpleNamespace(role=role, read=read), [3], x, y, tuple(tuple(p) for p in pools))
        other_history = OrderTrajectory(SimpleNamespace(role=role, read=read), [3], x, y, tuple(tuple(p) for p in pools))
        proxy = GroupingDesign(x, labels, residual_statistics(x, y, labels), .1, initial_ids,
                               "Mihili_fixture.mat", "original", 8, 0, 0)
        original_y = {t: row.copy() for t, row in history.y.items()}
        chosen, record = proxy.choose(history.counts.copy())
        self.assertEqual(calls, [])
        trial = history.acquire(chosen)
        self.assertEqual(calls, [((pools[chosen][4],), (3,))])
        self.assertEqual(len(proxy.labels), 32)
        self.assertNotIn(trial, other_history.x)
        proxy.observe(history.x[trial].copy(), chosen, trial)
        self.assertEqual(len(proxy.labels), 33)
        self.assertEqual(proxy.labels[-1], chosen)
        self.assertEqual(str(proxy.trial_ids[-1]), str(trial))
        for t, target in original_y.items():
            np.testing.assert_array_equal(history.y[t], target)
        _, actual_targets, actual_labels, actual_ids = history.observed()
        for t, target, label in zip(actual_ids, actual_targets, actual_labels):
            np.testing.assert_array_equal(target, [t, 2*t])
            self.assertEqual(label, t//12)
        self.assertEqual(record["counts_before"], [4]*8)

    def test_three_seed_SSE_mean_preserves_normalization_and_histories(self):
        # Risk: scoring averaged predictions creates an ensemble; requiring
        # identical acquired counts instead would erase legitimate policy divergence.
        from development_mapping_grouping import average_seed_scores
        standardized = (2., 8., 18.)
        wrong = (12., 18., 24.)
        native = (1., 9., 4.)
        panels = []
        for mc in range(3):
            rows, cells = [], []
            for n in CHECKPOINTS:
                for e in ELECTRODE_COUNTS:
                    scores = {"Q_common": 1-standardized[mc]/40,
                              "Q_wrong_common": 1-wrong[mc]/40,
                              "pairing_increment": (wrong[mc]-standardized[mc])/40,
                              "native_target_SSE": native[mc],
                              "initial_SD_standardized_SSE": standardized[mc],
                              "initial_mean_null_SSE": 40.}
                    costs = {"LFP_electrode_trials": n*e, "LFP_numeric_elements": n*e*3*10,
                             "target_numeric_elements": n*3*10}
                    cells.append({"training_trials": n, "electrodes": e, **scores, **costs})
                    for seed in ([0] if e == 96 else range(5)):
                        counts = [n//8]*8
                        if n not in (32, 96):
                            counts[0] -= mc
                            counts[1] += mc
                        rows.append({"training_trials": n, "electrodes": e, "seed": seed,
                                     "rank": 1, "alpha": .1, "direction_counts": counts,
                                     **scores, **costs})
            panels.append({"fitted_models": rows, "cells": cells})
        averaged = average_seed_scores(panels)
        self.assertEqual(len(averaged["fitted_models"]), 80)
        self.assertEqual(len(averaged["cells"]), 20)
        for row in averaged["fitted_models"]+averaged["cells"]:
            self.assertAlmostEqual(row["initial_SD_standardized_SSE"], 28/3)
            self.assertAlmostEqual(row["initial_SD_standardized_wrong_SSE"], 18.)
            self.assertAlmostEqual(row["native_target_SSE"], 14/3)
            self.assertEqual(row["initial_mean_null_SSE"], 40.)
            self.assertAlmostEqual(row["Q_common"], 23/30)
            self.assertAlmostEqual(row["Q_wrong_common"], .55)
            self.assertAlmostEqual(row["pairing_increment"], 13/60)
        intermediate = next(row for row in averaged["fitted_models"] if row["training_trials"] == 48)
        self.assertNotIn("direction_counts", intermediate)
        self.assertEqual(len(intermediate["direction_counts_by_mc_seed"]), 3)
        broken = copy.deepcopy(panels)
        broken[1]["fitted_models"][0]["initial_mean_null_SSE"] += 1
        with self.assertRaises(ArithmeticError):
            average_seed_scores(broken)


if __name__ == "__main__":
    unittest.main()
