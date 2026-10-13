"""Scientific fixtures for aligned LMP/high features and high-block derangement."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from primary_lmp_high import (
    RECIPES, REPLICATES, average_control_banks, fit_banks, joint_arm, prepare_split,
    ridge_predictions, summarize_banks, trial_maps,
)
from primary_multichannel import guide_columns
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


class JointHighTest(unittest.TestCase):
    def test_exact_physical_electrode_band_order_survives_guide_reordering(self):
        # Risk: a joint concatenation could silently pair the wrong electrode
        # or reverse the two power bands while retaining the expected width.
        guide = np.array([[2, 200, 400], [1, 0, 0], [2, 0, 0],
                          [1, 100, 200], [2, 100, 200], [1, 200, 400]])
        lmp_columns = guide_columns(guide, [1, 2], ((0, 0),))
        high_columns = guide_columns(guide, [1, 2], ((100, 200), (200, 400)))
        self.assertEqual(lmp_columns, [1, 2])
        self.assertEqual(high_columns, [3, 5, 4, 0])
        values = np.random.default_rng(50800).normal(size=(3, 15, 6))
        train = np.array([True, True, False])
        assembled = joint_arm(values[:, :, lmp_columns], values[:, :, high_columns], train)
        expected = values[:, :, [1, 2, 3, 5, 4, 0]]
        expected -= expected[train].mean(axis=(0, 1))
        np.testing.assert_allclose(assembled, expected, atol=1e-14, rtol=1e-14)

    def test_physical_training_metric_and_same_lambda_lmp_submodel(self):
        # Risk: evaluation-driven scales or hidden block weighting would change
        # the LMP anchor while claiming a nested same-penalty comparison.
        rng = np.random.default_rng(50801)
        labels = np.repeat(np.arange(8), 3)
        train = labels != 3
        lmp, high = rng.normal(size=(24, 15, 2)), rng.normal(size=(24, 15, 4))
        zl, zh = prepare_split(lmp, high, train)
        changed_lmp, changed_high = lmp.copy(), high.copy()
        changed_lmp[~train], changed_high[~train] = 1e9, -1e9
        other_lmp, other_high = prepare_split(changed_lmp, changed_high, train)
        np.testing.assert_array_equal(zl[train], other_lmp[train])
        np.testing.assert_array_equal(zh[train], other_high[train])
        expected_lmp = (lmp-lmp[train].mean(axis=(0, 1)))/lmp[train].std(axis=(0, 1))
        expected_high = (high-high[train].mean(axis=(0, 1)))/high[train].std(axis=(0, 1))
        np.testing.assert_allclose(zl, expected_lmp, atol=5e-15)
        np.testing.assert_allclose(zh, expected_high, atol=5e-15)
        joint = joint_arm(zl, zh, train)
        beta = rng.normal(size=(2, 2))
        embedded = np.vstack((beta, np.zeros((4, 2))))
        expected = (expected_lmp-expected_lmp[train].mean(axis=(0, 1)))@beta
        np.testing.assert_allclose(joint@embedded, expected, atol=4e-15, rtol=1e-12)
        np.testing.assert_allclose(joint[train].std(axis=(0, 1)), np.ones(6), atol=5e-15)

    def test_maps_are_canonical_derangements_without_partition_crossings(self):
        # Risk: incidental array order or a held fold could change the control
        # draw; a donor from another direction would leak across the fold.
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i}" for i in range(24)])
        reordered = np.random.default_rng(50802).permutation(24)
        for replicate in REPLICATES:
            donors, records = trial_maps(ids, labels, "synthetic", replicate)
            again, _ = trial_maps(ids, labels, "synthetic", replicate)
            other, _ = trial_maps(ids[reordered], labels[reordered], "synthetic", replicate)
            np.testing.assert_array_equal(donors, again)
            self.assertFalse(np.any(donors == np.arange(24)))
            np.testing.assert_array_equal(labels[donors], labels)
            self.assertEqual(dict(zip(ids, ids[donors])),
                             dict(zip(ids[reordered], ids[reordered][other])))
            self.assertEqual(len(records), 8)
            for held in range(8):
                np.testing.assert_array_equal(labels[donors] == held, labels == held)

    def test_only_complete_high_trajectories_move_and_inputs_remain_intact(self):
        # Risk: shuffling individual bands/times or the recipient LMP/target
        # would test a different scientific alternative.
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i}" for i in range(24)])
        train = labels != 2
        rng = np.random.default_rng(50803)
        lmp, high = rng.normal(size=(24, 15, 2)), rng.normal(size=(24, 15, 4))
        zl, zh = prepare_split(lmp, high, train)
        previous_lmp, previous_high = zl.copy(), zh.copy()
        donors, _ = trial_maps(ids, labels, "synthetic", 1)
        actual = joint_arm(zl, zh, train, donors)
        expected = np.concatenate((zl, zh[donors]), axis=2)
        expected -= expected[train].mean(axis=(0, 1))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(zl, previous_lmp)
        np.testing.assert_array_equal(zh, previous_high)
        paired = joint_arm(zl, zh, train)
        np.testing.assert_array_equal(actual[:, :, :2], paired[:, :, :2])

    def test_unchanged_high_gram_changed_cross_gram_and_independent_joint_solve(self):
        # Risk: reusing a shared joint Gram would be wrong even though both
        # within-block Grams survive a whole-trial high permutation.
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i}" for i in range(24)])
        train = labels != 6
        rng = np.random.default_rng(50804)
        lmp = rng.normal(size=(24, 15, 2))
        high = np.concatenate((lmp+.1*rng.normal(size=lmp.shape),
                               rng.normal(size=lmp.shape)), axis=2)
        zl, zh = prepare_split(lmp, high, train)
        donors, _ = trial_maps(ids, labels, "synthetic", 0)
        paired, control = joint_arm(zl, zh, train), joint_arm(zl, zh, train, donors)
        first, changed = paired[train].reshape(-1, 6), control[train].reshape(-1, 6)
        np.testing.assert_allclose(first[:, :2].T@first[:, :2], changed[:, :2].T@changed[:, :2], atol=2e-12)
        np.testing.assert_allclose(first[:, 2:].T@first[:, 2:], changed[:, 2:].T@changed[:, 2:], atol=2e-12)
        self.assertGreater(np.linalg.norm(first[:, :2].T@first[:, 2:]-changed[:, :2].T@changed[:, 2:]), 1.)
        residual = rng.normal(size=(len(changed), 2))
        predicted = ridge_predictions(control[train], control[~train], residual)
        for index, penalty in enumerate(LAMBDAS):
            weights = np.linalg.solve(changed.T@changed+len(changed)*penalty*np.eye(6), changed.T@residual)
            expected = control[~train].reshape(-1, 6)@weights
            np.testing.assert_allclose(predicted[index], expected, atol=2e-14, rtol=2e-11)

    def test_constant_features_preserve_baseline_and_recipient_targets(self):
        # Risk: a new centering/intercept path could manufacture gain when
        # neither physical feature block contains any varying information.
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i}" for i in range(24)])
        lmp, high = np.ones((24, 15, 2)), np.ones((24, 15, 4))
        target = np.random.default_rng(50805).normal(size=(24, 15, 2))
        original = (lmp.copy(), high.copy(), target.copy(), labels.copy(), ids.copy())
        banks, baseline, predictions = fit_banks(lmp, high, target, labels, ids, "constant")
        for prediction in predictions[RECIPES[0]]:
            np.testing.assert_array_equal(prediction, baseline)
        for replicate in predictions[RECIPES[1]]:
            for prediction in replicate:
                np.testing.assert_array_equal(prediction, baseline)
        for actual, before in zip((lmp, high, target, labels, ids), original):
            np.testing.assert_array_equal(actual, before)
        for recipe in RECIPES:
            self.assertTrue(all(row["delta_R2"] == 0 for row in banks[recipe]["lambda_rows"]))

    def test_control_bank_averages_actual_seed_errors_not_predictions(self):
        # Risk: averaging predictions would turn eight randomized controls
        # into an unregistered ensemble with a different error.
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i}" for i in range(24)])
        rng = np.random.default_rng(50806)
        lmp, high = rng.normal(size=(24, 15, 2)), rng.normal(size=(24, 15, 4))
        target = lmp+.3*high[:, :, :2]+rng.normal(size=lmp.shape)
        banks, _, predictions = fit_banks(lmp, high, target, labels, ids, "mean-score")
        control = predictions[RECIPES[1]]
        for index in range(len(LAMBDAS)):
            errors = np.square(target[None]-control[:, index]).sum(axis=(1, 2, 3))
            actual = banks[RECIPES[1]]["lambda_rows"][index]["candidate_SSE"]
            self.assertAlmostEqual(actual, float(errors.mean()), places=9)
            wrong = np.mean([wrong_pair_sse(target, control[r, index], labels) for r in REPLICATES])
            self.assertAlmostEqual(banks[RECIPES[1]]["lambda_rows"][index]["wrong_pair_SSE"], float(wrong), places=9)
        mean_error = float(np.square(target-control[:, 0].mean(axis=0)).sum())
        self.assertGreater(banks[RECIPES[1]]["lambda_rows"][0]["candidate_SSE"]-mean_error, 1e-6)

    def test_rejection_draw_uses_exact_seed_and_first_complete_derangement(self):
        # Risk: accepting fixed points or adaptively reseeding changes the
        # declared uniform derangement distribution.
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i}" for i in range(24)])
        class TwoDraws:
            def __init__(self):
                self.calls = 0
            def seed(self, literal, version):
                self.literal, self.version = literal, version
            def shuffle(self, values):
                self.calls += 1
                if self.calls == 2:
                    values[:] = values[1:]+values[:1]
        draws = [TwoDraws() for _ in range(8)]
        with patch("primary_lmp_high.random.Random", side_effect=draws):
            donors, records = trial_maps(ids, labels, "synthetic", 2)
        for direction, (draw, record) in enumerate(zip(draws, records)):
            canonical = sorted(ids[labels == direction].tolist())
            literal = f"ep05-lmp-high-trial-label-v1|20261003|synthetic|{direction}|2"
            self.assertEqual(draw.version, 2)
            self.assertEqual(draw.literal, literal)
            self.assertEqual(draw.calls, 2)
            self.assertEqual(record["draw_count"], 2)
            self.assertEqual(record["seed_literal"], literal)
            self.assertEqual(record["recipient_trial_ids"], canonical)
            self.assertEqual(record["donor_trial_ids"], canonical[1:]+canonical[:1])
        self.assertFalse(np.any(donors == np.arange(24)))

    def test_outer_joint_lambda_excludes_held_scores_and_is_shared_by_control_and_anchor(self):
        # Risk: the held session, a favorable randomized control or the old
        # independently chosen anchor could silently choose the joint penalty.
        def bank(recipe, kind):
            rows = []
            for index, penalty in enumerate(LAMBDAS):
                delta = .01*index if kind == "joint" else (.04-.01*index if kind == "control" else .06-.01*index)
                identity = .01+.001*index if kind == "joint" else (.004 if kind == "control" else .002+.001*index)
                sse = 5-10*delta
                rows.append({"lambda": penalty, "delta_R2": delta, "identity_increment": identity,
                             "candidate_SSE": sse, "wrong_pair_SSE": sse+10*identity,
                             "candidate_R2": 1-sse/10})
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                    "baseline_SSE": 5., "denominator": 10., "direction_counts": [3]*8,
                    "lambda_rows": rows}
        sessions = {session: {RECIPES[0]: bank(RECIPES[0], "joint"),
                              RECIPES[1]: average_control_banks([bank(RECIPES[1], "control") for _ in REPLICATES])}
                    for session in SESSIONS}
        anchors = {session: bank("lmp_full_channel_ridge_v1", "anchor") for session in SESSIONS}
        independent = {"recipe": "lmp_full_channel_ridge_v1", "sessions": [
            {"session_id": session, "animal": ANIMALS[session], **anchors[session]["lambda_rows"][0]}
            for session in SESSIONS]}
        before = summarize_banks(sessions, anchors, independent)
        self.assertAlmostEqual(before["anchor_contrasts"]["independently_selected"]["primary_increment"], -.02)
        self.assertAlmostEqual(before["anchor_contrasts"]["joint_lambda_matched"]["primary_increment"], .02)
        self.assertFalse(before["all_complementarity_screens_pass"])
        for held in SESSIONS:
            changed = copy.deepcopy(sessions)
            for index, row in enumerate(changed[held][RECIPES[0]]["lambda_rows"]):
                row["delta_R2"] = 1e8*(5-index)
            after = summarize_banks(changed, anchors, independent)
            for original_panel, changed_panel in zip(before["panels"], after["panels"]):
                original_row = next(r for r in original_panel["sessions"] if r["session_id"] == held)
                changed_row = next(r for r in changed_panel["sessions"] if r["session_id"] == held)
                self.assertEqual(original_row["lambda_index"], 4)
                self.assertEqual(changed_row["lambda_index"], 4)
                self.assertEqual(original_row["joint_inner_animal_balanced_scores"], changed_row["joint_inner_animal_balanced_scores"])
                self.assertEqual(set(changed_row["outer_training_sessions"]), set(SESSIONS)-{held})
            matched = next(r for r in after["matched_lambda_anchor_sessions"] if r["session_id"] == held)
            self.assertEqual(matched["lambda"], LAMBDAS[4])


if __name__ == "__main__":
    unittest.main()
