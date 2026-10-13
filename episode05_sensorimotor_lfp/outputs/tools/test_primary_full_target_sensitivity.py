"""Independent direct-target ridge, trajectory-control and accounting checks."""
import unittest
from unittest.mock import patch

import numpy as np

import primary_full_target_sensitivity as full
import primary_multichannel as residual
from method_components import DirectionTime
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def example(seed=125001):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 3)
    # Match the qualified reader's per-trial transpose then stack layout.
    x = np.stack([rng.normal(size=(4, 15)).T.astype(np.float64) for _ in labels])
    theta, time = labels*np.pi/4, np.linspace(-1, 1, 15)
    y = rng.normal(size=(len(labels), 15, 2))
    y[:, :, 0] += 13+20*np.cos(theta)[:, None]*(1+time[None, :])+.7*x[:, :, 0]
    y[:, :, 1] += -7+30*np.sin(theta)[:, None]*(1-time[None, :])-.4*x[:, :, 1]
    ids = np.array([f"example_trial{i+1:06d}" for i in range(len(labels))])
    return x, y, labels, ids, [2, 7, 11, 19]


class FullTargetSensitivityTest(unittest.TestCase):
    def test_original_flat_X_normalization_native_target_mean_and_separate_B(self):
        x, y, labels, _, _ = example()
        held, train = 3, labels != 3
        split = full.prepare_split(x, y, labels, held)
        flat, target = x[train].reshape(-1, 4), y[train].reshape(-1, 2)
        mean, sd, mu = flat.mean(axis=0), flat.std(axis=0), target.mean(axis=0)
        z = (flat-mean)/np.where(sd == 0, 1., sd)
        np.testing.assert_array_equal(split["physical_mean"], mean)
        np.testing.assert_array_equal(split["physical_sd"], sd)
        np.testing.assert_array_equal(split["fitting"], z)
        np.testing.assert_array_equal(split["evaluation"], (x[~train].reshape(-1, 4)-mean)/np.where(sd == 0, 1., sd))
        np.testing.assert_array_equal(split["mu"], mu)
        np.testing.assert_array_equal(split["centered_targets"], target-mu)
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        c = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2).mean(axis=0)
        np.testing.assert_array_equal(split["baseline_training"], template.predict(labels[train]*np.pi/4)+c)
        np.testing.assert_array_equal(split["baseline_evaluation"], template.predict(labels[~train]*np.pi/4)+c)
        self.assertFalse(split["baseline_equals_mu"])
        self.assertGreater(np.linalg.norm(split["baseline_evaluation"]-mu), 1.)

    def test_full_prediction_matches_augmented_least_squares_without_adding_B(self):
        x, y, labels, ids, channels = example(125002)
        _, baseline, predictions = full.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        for held in range(8):
            train, test = labels != held, labels == held
            flat, target = x[train].reshape(-1, 4), y[train].reshape(-1, 2)
            mean, sd = flat.mean(axis=0), flat.std(axis=0)
            z, ev = (flat-mean)/np.where(sd == 0, 1., sd), (x[test].reshape(-1, 4)-mean)/np.where(sd == 0, 1., sd)
            mu = target.mean(axis=0)
            for index, penalty in enumerate(LAMBDAS):
                # Separate reference algorithm: rectangular augmented OLS.
                design = np.vstack([z, np.sqrt(len(z)*penalty)*np.eye(4)])
                response = np.vstack([target-mu, np.zeros((4, 2))])
                beta = np.linalg.lstsq(design, response, rcond=None)[0]
                expected = (mu+ev@beta).reshape(y[test].shape)
                np.testing.assert_allclose(predictions[full.RECIPES[0]][index, test], expected,
                                           rtol=2e-12, atol=2e-12)
                self.assertGreater(np.linalg.norm(expected+baseline[test]-predictions[full.RECIPES[0]][index, test]), 1.)

    def test_canonical_maps_preserve_whole_XY_trajectories_and_direction_time_means(self):
        held = 3
        labels = np.concatenate([np.full(d+2, d) for d in range(8) if d != held])
        rng = np.random.default_rng(125003)
        numbers = rng.permutation(np.arange(len(labels))*11+1)
        ids = np.array([f"example_trial{i:06d}" for i in numbers])
        target = np.arange(len(labels)*30, dtype=float).reshape(-1, 15, 2)
        mapping, records = full.trial_map(ids, labels, "example", held, 4)
        np.testing.assert_array_equal(labels[mapping], labels)
        self.assertEqual(sorted(mapping.tolist()), list(range(len(labels))))
        for direction in sorted(set(labels.tolist())):
            np.testing.assert_array_equal(target[mapping][labels == direction].mean(axis=0),
                                           target[labels == direction].mean(axis=0))
        reverse = np.arange(len(labels))[::-1]
        reversed_map, reversed_records = full.trial_map(ids[reverse], labels[reverse], "example", held, 4)
        self.assertEqual(records, reversed_records)
        np.testing.assert_array_equal(ids[reverse][reversed_map], ids[mapping][reverse])
        literals = []
        class Identity:
            def seed(self, literal, version): literals.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(full.random, "Random", Identity):
            identity, identity_records = full.trial_map(ids, labels, "example", held, 4)
        np.testing.assert_array_equal(identity, np.arange(len(labels)))
        self.assertEqual(literals, [(f"ep05-lmp-full-target-training-label-v1|20261004|example|3|4|{d}", 2)
                                    for d in sorted(set(labels.tolist()))])
        self.assertEqual(len(identity_records), 7)
        malformed = ids.copy()
        malformed[0] = "wrong_trial000001"
        with self.assertRaises(ValueError):
            full.trial_map(malformed, labels, "example", held, 4)

    def test_controls_share_true_mu_and_Gram_but_use_own_permuted_target_cross_products(self):
        x, y, labels, ids, channels = example(125004)
        observed = []
        original = full.solve_path
        def solve(gram, cross, nfit, check):
            coefficients = original(gram, cross, nfit, check)
            observed.append((gram.copy(), cross.copy(), nfit, coefficients.copy()))
            return coefficients
        with patch.object(full, "solve_path", side_effect=solve):
            banks, _, predictions = full.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        self.assertEqual(len(observed), 72)
        distinct_cross = False
        for held in range(8):
            split = full.prepare_split(x, y, labels, held)
            train, test = labels != held, labels == held
            z, target = split["fitting"], split["centered_targets"].reshape(-1, 15, 2)
            gram, mu = z.T@z, split["mu"]
            paired_cross = z.T@target.reshape(-1, 2)
            for arm in range(9):
                g, cross, nfit, coefficients = observed[9*held+arm]
                np.testing.assert_array_equal(g, gram)
                self.assertEqual(nfit, len(z))
                if arm:
                    mapping, _ = full.trial_map(ids[train], labels[train], "example", held, arm-1)
                    expected_cross = z.T@target[mapping].reshape(-1, 2)
                    distinct_cross |= not np.array_equal(expected_cross, paired_cross)
                    # NumPy combined advanced indexing may move the test axis;
                    # slice the replicate first to preserve lambda-first order.
                    actual = predictions[full.RECIPES[1]][arm-1][:, test]
                else:
                    expected_cross = paired_cross
                    actual = predictions[full.RECIPES[0]][:, test]
                np.testing.assert_array_equal(cross, expected_cross)
                expected = np.asarray([(mu+split["evaluation"]@beta).reshape(y[test].shape)
                                       for beta in coefficients])
                np.testing.assert_array_equal(actual, expected)
        self.assertTrue(distinct_cross)
        control = predictions[full.RECIPES[1]]
        for index in range(len(LAMBDAS)):
            row = banks[full.RECIPES[1]]["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_SSE"], float(np.mean(np.sum((control[:, index]-y)**2, axis=(1, 2, 3)))), places=8)
            self.assertAlmostEqual(row["wrong_pair_SSE"], float(np.mean([wrong_pair_sse(y, control[r, index], labels) for r in range(8)])), places=8)

    def test_held_values_do_not_fit_scaling_mu_Gram_or_B(self):
        x, y, labels, _, _ = example(125005)
        split = full.prepare_split(x, y, labels, 2)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[labels == 2] += 1e6
        changed_y[labels == 2] -= 1e6
        other = full.prepare_split(changed_x, changed_y, labels, 2)
        for key in ("physical_mean", "physical_sd", "mu", "fitting", "centered_targets",
                    "baseline_training", "baseline_evaluation"):
            np.testing.assert_array_equal(split[key], other[key])

    def test_zero_features_predict_mu_not_B_and_constant_B_gives_residual_recipe(self):
        x, y, labels, ids, channels = example(125006)
        zero = np.zeros_like(x)
        banks, baseline, predictions = full.fit_banks(zero, y, labels, ids, "example", electrode_ids=channels)
        self.assertFalse(banks[full.RECIPES[0]]["structural_residual_anchor_duplicate"])
        self.assertTrue(banks[full.RECIPES[0]]["all_controls_equal_candidate"])
        self.assertTrue(banks[full.RECIPES[1]]["structural_paired_duplicate"])
        for held in range(8):
            mu = y[labels != held].reshape(-1, 2).mean(axis=0)
            for prediction in predictions[full.RECIPES[0]]:
                np.testing.assert_array_equal(prediction[labels == held], np.broadcast_to(mu, y[labels == held].shape))
            self.assertGreater(np.linalg.norm(baseline[labels == held]-mu), 1.)
        # Identically zero targets exercise literal B==mu without introducing
        # an undefined all-zero scoring denominator into fit_banks.
        exact = full.prepare_split(x, np.zeros_like(y), labels, 2)
        self.assertTrue(exact["baseline_equals_mu"])
        self.assertTrue(np.all(exact["baseline_training"] == 0.))
        self.assertTrue(np.all(exact["baseline_evaluation"] == 0.))
        # Symmetric nonzero targets provide a defined score and an algebraic
        # constant-B reference. Weighted least squares can retain roundoff in
        # its template, so this is numerical equivalence, not an exact flag.
        constant_B_y = np.zeros_like(y)
        wave = np.arange(30, dtype=float).reshape(15, 2)-15
        constant_B_y[::3], constant_B_y[1::3] = wave, -wave
        same, base, direct = full.fit_banks(x, constant_B_y, labels, ids, "example", electrode_ids=channels)
        _, old_base, old_predictions = residual.fit_bank(x, constant_B_y, labels, full.ANCHOR)
        self.assertEqual(same[full.RECIPES[0]]["structural_residual_anchor_duplicate"],
                         all(row["baseline_equals_mu"] for row in
                             same[full.RECIPES[0]]["direction_fit_metadata"]))
        np.testing.assert_array_equal(base, old_base)
        np.testing.assert_allclose(direct[full.RECIPES[0]], old_predictions, rtol=2e-13, atol=2e-13)

    def test_identity_training_maps_reproduce_the_direct_candidate_and_exact_cross_flags(self):
        x, y, labels, ids, channels = example(125007)
        class Identity:
            def seed(self, literal, version): pass
            def shuffle(self, values): pass
        with patch.object(full.random, "Random", Identity):
            banks, _, predictions = full.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        self.assertFalse(banks[full.RECIPES[0]]["structural_residual_anchor_duplicate"])
        self.assertTrue(banks[full.RECIPES[0]]["all_controls_equal_candidate"])
        self.assertTrue(banks[full.RECIPES[1]]["structural_paired_duplicate"])
        for row in banks[full.RECIPES[0]]["direction_fit_metadata"]:
            self.assertEqual(row["controls_equal_candidate_by_replicate"], [True]*8)
            for cross in row["control_cross_by_replicate"]:
                self.assertEqual(cross, row["paired_cross"])
        for prediction in predictions[full.RECIPES[1]]:
            np.testing.assert_array_equal(prediction, predictions[full.RECIPES[0]])

    def test_outer_session_exclusion_stronger_ties_shared_lambda_and_zero_one_two_counts(self):
        def bank(recipe, paired):
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "baseline_SSE": 5., "denominator": 10., "direction_counts": [3]*8,
                "structural_residual_anchor_duplicate": False, "all_controls_equal_candidate": False,
                "structural_paired_duplicate": False,
                "lambda_rows": [{"lambda": penalty, "delta_R2": .01 if paired else .04-.01*i,
                    "identity_increment": .01, "candidate_SSE": 5., "wrong_pair_SSE": 5.1,
                    "candidate_R2": .5} for i, penalty in enumerate(LAMBDAS)]}
        sessions = {s: {full.RECIPES[0]: bank(full.RECIPES[0], True),
                    full.RECIPES[1]: full.average_control_banks([bank(full.RECIPES[1], False) for _ in range(8)])}
                    for s in SESSIONS}
        anchors = {s: bank(full.ANCHOR, False) for s in SESSIONS}
        anchor_panel = {"recipe": full.ANCHOR, "sessions": [
            {"session_id": s, "animal": ANIMALS[s], **anchors[s]["lambda_rows"][0]} for s in SESSIONS]}
        held = SESSIONS[0]
        for i, row in enumerate(sessions[held][full.RECIPES[0]]["lambda_rows"]):
            row["delta_R2"] = 1e7*(5-i)
        for anchor_duplicate, same, candidate_charge, control_charge in (
                (False, False, 1, 1), (True, False, 0, 1), (False, True, 1, 0), (True, True, 0, 0)):
            for banks in sessions.values():
                banks[full.RECIPES[0]]["structural_residual_anchor_duplicate"] = anchor_duplicate
                banks[full.RECIPES[0]]["all_controls_equal_candidate"] = same
                banks[full.RECIPES[1]]["structural_paired_duplicate"] = same
            result = full.summarize_banks(sessions, anchors, anchor_panel)
            self.assertEqual(result["scientific_trial_count_increment"], candidate_charge+control_charge)
            self.assertEqual(result["control_trial_count_increment"], control_charge)
            self.assertEqual(result["scientific_trial_count_by_recipe"],
                             dict(zip(full.RECIPES, (candidate_charge, control_charge))))
            self.assertTrue(result["sensitivity_only"])
            self.assertFalse(result["promotion_authorized"])
            self.assertIsNone(result["anchor_noninferiority_requirement"])
            self.assertEqual(set(result["anchor_contrasts"]), {"independently_selected", "full_target_lambda_matched"})
            for panel in result["panels"]:
                self.assertTrue(panel["sensitivity_only"])
                self.assertFalse(panel["promotion_authorized"])
                row = next(row for row in panel["sessions"] if row["session_id"] == held)
                self.assertEqual(row["lambda_index"], 4)
                self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
            matched = next(row for row in result["matched_lambda_anchor_sessions"] if row["session_id"] == held)
            self.assertEqual(matched["lambda"], LAMBDAS[4])
            original = next(row for row in anchor_panel["sessions"] if row["session_id"] == held)
            self.assertEqual(original["lambda"], LAMBDAS[0])


if __name__ == "__main__":
    unittest.main()
