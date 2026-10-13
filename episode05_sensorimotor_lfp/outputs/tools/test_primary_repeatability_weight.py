"""Independent repeatability-energy, ridge-metric, leakage and accounting checks."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np

import primary_repeatability_weight as weighted
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def example(seed=95001):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 2)
    waveform = np.sin(np.linspace(-np.pi, np.pi, 15))
    x = rng.normal(size=(16, 15, 4))+waveform[None, :, None]*np.array([5., 2., .6, 0.])[None, None, :]
    y = rng.normal(size=(16, 15, 2))+.5*x[:, :, :2]
    ids = np.array([f"example_trial{i+1:06d}" for i in range(16)])
    return x, y, labels, ids, [2, 7, 11, 19]


class RepeatabilityWeightTest(unittest.TestCase):
    def test_T_P_equal_explicit_ordered_distinct_trial_pairs_with_unequal_counts(self):
        rng = np.random.default_rng(95002)
        labels = np.concatenate([np.full(d+2, d) for d in range(7)])
        x = rng.normal(size=(len(labels), 15, 4))
        x += np.sin(np.arange(15))[None, :, None]*np.array([.2, 1., 2., 3.])[None, None, :]
        totals, pairs = [], []
        for direction in range(7):
            block = x[labels == direction]
            totals.append(np.mean(block**2, axis=(0, 1)))
            products = [np.mean(block[i]*block[j], axis=0)
                        for i in range(len(block)) for j in range(len(block)) if i != j]
            pairs.append(np.mean(products, axis=0))
        result = weighted.repeatability_statistics(x, labels)
        np.testing.assert_allclose(result["T"], np.mean(totals, axis=0), rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["P"], np.mean(pairs, axis=0), rtol=2e-14, atol=2e-14)
        self.assertEqual(result["fitting_direction_counts"], list(range(2, 9)))
        expected_r = np.clip(np.mean(pairs, axis=0)/np.mean(totals, axis=0), 0, 1)
        np.testing.assert_allclose(result["r"], expected_r, atol=2e-14)
        np.testing.assert_allclose(result["a"], .1+expected_r, atol=2e-14)
        self.assertFalse({"y", "targets", "target"} & set(inspect.signature(weighted.repeatability_statistics).parameters))

    def test_negative_agreement_zero_energy_and_perfect_repeatability_have_fixed_floor(self):
        labels = np.repeat(np.arange(7), 2)
        waveform = np.arange(15, dtype=float)-7
        x = np.zeros((14, 15, 3))
        x[:, :, 0] = waveform
        x[::2, :, 1], x[1::2, :, 1] = waveform, -waveform
        result = weighted.repeatability_statistics(x, labels)
        np.testing.assert_allclose(result["P"][:2], [np.mean(waveform**2), -np.mean(waveform**2)])
        np.testing.assert_allclose(result["r"], [1., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(result["a"], [1.1, .1, .1], atol=1e-14)
        self.assertLess(result["raw_ratio"][1], 0)
        self.assertIsNone(result["raw_ratio"][2])
        self.assertEqual(result["clip_category"][1:], ["lower_clipped", "zero_energy"])

    def test_trace_preservation_matches_generalized_ridge_and_zero_column_identity_edge(self):
        gram = np.array([[1., .4, 0.], [.4, 2., 0.], [0., 0., 0.]])
        base = np.array([.7, .7, .1])
        identity, identity_info = weighted.trace_weights(base, gram)
        np.testing.assert_array_equal(identity, np.ones(3))
        self.assertTrue(identity_info["structural_identity"])
        changed, info = weighted.trace_weights(base[[2, 1, 0]], gram)
        self.assertFalse(info["structural_identity"])
        self.assertTrue(np.all(changed > 0))
        diagonal = np.diag(changed)
        self.assertAlmostEqual(np.trace(diagonal@gram@diagonal), np.trace(gram), places=13)
        cross = np.array([[.3, -.2], [.4, .7], [0., 0.]])
        for penalty in LAMBDAS:
            transformed = diagonal@np.linalg.solve(diagonal@gram@diagonal+penalty*np.eye(3), diagonal@cross)
            generalized = np.linalg.solve(gram+penalty*np.diag(1/changed**2), cross)
            np.testing.assert_allclose(transformed, generalized, rtol=2e-13, atol=2e-14)
        zero, zero_info = weighted.trace_weights(np.array([.1, .3, 1.1]), np.zeros((3, 3)))
        np.testing.assert_array_equal(zero, np.ones(3))
        self.assertTrue(zero_info["structural_identity"])

    def test_original_physical_ID_maps_accept_identity_and_use_exact_seed_literal(self):
        ids = [2, 7, 11, 19]
        mapping, record = weighted.weight_map(ids, "example", 3)
        self.assertEqual(sorted(mapping.tolist()), list(range(4)))
        self.assertEqual(record["recipient_electrode_ids"], ids)
        self.assertEqual(record["donor_electrode_ids"], np.asarray(ids)[mapping].tolist())
        self.assertEqual(record["seed_literal"], "ep05-lmp-repeatability-weight-label-v1|20261004|example|3")
        draws = []
        class Identity:
            def seed(self, literal, version): draws.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(weighted.random, "Random", Identity):
            unchanged, _ = weighted.weight_map(ids, "example", 3)
        np.testing.assert_array_equal(unchanged, np.arange(4))
        self.assertEqual(draws, [(record["seed_literal"], 2)])
        with self.assertRaises(ValueError):
            weighted.weight_map([7, 2, 11, 19], "example", 3)

    def test_controls_change_only_weights_have_own_Grams_and_true_targets_mean_SSE(self):
        x, y, labels, ids, channels = example()
        observed = []
        original_fit = weighted.ridge_predictions
        def fit(training, evaluation, residual, check):
            observed.append((training.copy(), evaluation.copy(), residual.copy()))
            return original_fit(training, evaluation, residual, check)
        with patch.object(weighted, "ridge_predictions", side_effect=fit), \
                patch.object(weighted, "weight_map", wraps=weighted.weight_map) as maps:
            banks, _, predictions = weighted.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        self.assertEqual(len(observed), 72)
        self.assertEqual(maps.call_count, 8)
        metadata = banks[weighted.RECIPES[0]]["direction_fit_metadata"]
        saw_distinct_gram = False
        for held in range(8):
            train, test = labels != held, labels == held
            template = weighted.DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
            u = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
            u -= u.mean(axis=0)
            z, _ = weighted.prepare_split(x, train)
            arms = [metadata[held]["paired_weights"], *metadata[held]["control_weights_by_replicate"]]
            true_gram = None
            for arm, info in enumerate(arms):
                training, evaluation, actual_u = observed[9*held+arm]
                np.testing.assert_array_equal(actual_u, u)
                np.testing.assert_allclose(training, z[train]*np.asarray(info["weights"]), rtol=0, atol=0)
                np.testing.assert_allclose(evaluation, z[test]*np.asarray(info["weights"]), rtol=0, atol=0)
                flat = training.reshape(-1, 4)
                gram = flat.T@flat/len(flat)
                self.assertAlmostEqual(np.trace(gram), info["original_trace"], places=12)
                if arm == 0:
                    true_gram = gram
                elif not np.allclose(gram, true_gram, rtol=1e-9, atol=1e-9):
                    saw_distinct_gram = True
        self.assertTrue(saw_distinct_gram)
        control = predictions[weighted.RECIPES[1]]
        for index in range(len(LAMBDAS)):
            row = banks[weighted.RECIPES[1]]["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_SSE"], float(np.mean(np.sum((control[:, index]-y)**2, axis=(1, 2, 3)))), places=9)
            self.assertAlmostEqual(row["wrong_pair_SSE"], float(np.mean([wrong_pair_sse(y, control[r, index], labels) for r in range(8)])), places=9)

    def test_weights_exclude_all_velocity_values_and_held_direction_features(self):
        x, y, labels, ids, channels = example(95003)
        original, _, _ = weighted.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        changed_y = y+1e4*np.random.default_rng(95004).normal(size=y.shape)
        other, _, _ = weighted.fit_banks(x, changed_y, labels, ids, "example", electrode_ids=channels)
        first = original[weighted.RECIPES[0]]["direction_fit_metadata"]
        second = other[weighted.RECIPES[0]]["direction_fit_metadata"]
        for before, after in zip(first, second):
            self.assertEqual(before, after)
        changed_x = x.copy()
        changed_x[labels == 2] += 1e4
        poisoned, _, _ = weighted.fit_banks(changed_x, y, labels, ids, "example", electrode_ids=channels)
        self.assertEqual(first[2], poisoned[weighted.RECIPES[0]]["direction_fit_metadata"][2])

    def test_identity_maps_keep_nonidentity_candidate_and_identical_controls(self):
        x, y, labels, ids, channels = example(95005)
        class Identity:
            def seed(self, literal, version): pass
            def shuffle(self, values): pass
        with patch.object(weighted.random, "Random", Identity):
            banks, _, predictions = weighted.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        self.assertFalse(banks[weighted.RECIPES[0]]["structural_identity"])
        self.assertFalse(banks[weighted.RECIPES[1]]["structural_identity"])
        self.assertTrue(banks[weighted.RECIPES[0]]["all_controls_equal_candidate"])
        for control in predictions[weighted.RECIPES[1]]:
            np.testing.assert_array_equal(control, predictions[weighted.RECIPES[0]])

    def test_outer_lambda_exclusion_and_independent_zero_one_two_recipe_charges(self):
        def bank(recipe, paired):
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "baseline_SSE": 5., "denominator": 10., "direction_counts": [2]*8,
                "structural_identity": False, "all_controls_equal_candidate": False,
                "lambda_rows": [{"lambda": penalty, "delta_R2": .01*i if paired else .04-.01*i,
                    "identity_increment": .01, "candidate_SSE": 5., "wrong_pair_SSE": 5.1,
                    "candidate_R2": .5} for i, penalty in enumerate(LAMBDAS)]}
        sessions = {s: {weighted.RECIPES[0]: bank(weighted.RECIPES[0], True),
                    weighted.RECIPES[1]: weighted.average_control_banks([bank(weighted.RECIPES[1], False) for _ in range(8)])}
                    for s in SESSIONS}
        anchors = {s: bank(weighted.ANCHOR, False) for s in SESSIONS}
        anchor_panel = {"recipe": weighted.ANCHOR, "sessions": [{"session_id": s, "animal": ANIMALS[s],
                        **anchors[s]["lambda_rows"][0]} for s in SESSIONS]}
        held = SESSIONS[0]
        for i, row in enumerate(sessions[held][weighted.RECIPES[0]]["lambda_rows"]):
            row["delta_R2"] = 1e7*(5-i)
        for candidate_identity, control_identity, same, candidate_charge, control_charge in (
                (False, False, False, 1, 1), (True, False, False, 0, 1),
                (False, True, False, 1, 0), (True, True, True, 0, 0), (False, False, True, 1, 0)):
            for banks in sessions.values():
                banks[weighted.RECIPES[0]]["structural_identity"] = candidate_identity
                banks[weighted.RECIPES[1]]["structural_identity"] = control_identity
                banks[weighted.RECIPES[0]]["all_controls_equal_candidate"] = same
            result = weighted.summarize_banks(sessions, anchors, anchor_panel)
            self.assertEqual(result["scientific_trial_count_increment"], candidate_charge+control_charge)
            self.assertEqual(result["control_trial_count_increment"], control_charge)
            self.assertEqual(result["scientific_trial_count_by_recipe"],
                             dict(zip(weighted.RECIPES, (candidate_charge, control_charge))))
            for panel in result["panels"]:
                row = next(row for row in panel["sessions"] if row["session_id"] == held)
                self.assertEqual(row["lambda_index"], 4)
                self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})


if __name__ == "__main__":
    unittest.main()
