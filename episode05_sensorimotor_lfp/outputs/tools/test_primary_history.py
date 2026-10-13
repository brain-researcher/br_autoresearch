"""Scientific checks of fixed native history and physical-channel-label control."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from history_lmp_reader import source_slices
from primary_capacity import score_bank
from primary_history import (
    LAGS, RECIPES, REPLICATES, average_control_banks, center_arm, channel_maps,
    fit_banks, history_blocks, permute_channels, prepare_split, ridge_predictions,
    standardize_history, summarize_banks,
)
from primary_reference import LAMBDAS, SESSIONS
from run_primary_history import check_history_reference_after_fit


class HistoryTest(unittest.TestCase):
    def test_native_source_slices_and_all_four_lags_preserve_full_target_window(self):
        # Risk: a pre-onset history can silently shift or truncate the primary Y endpoint.
        feature_slice, target_slice = source_slices(7, 30)
        np.testing.assert_array_equal(np.arange(30)[feature_slice], np.arange(4, 22))
        np.testing.assert_array_equal(np.arange(30)[target_slice], np.arange(7, 22))
        source = (100*np.arange(-3, 15)[:, None]+np.arange(2)[None, :])[None]
        blocks = history_blocks(source)
        self.assertEqual(blocks.shape, (1, 15, 4, 2))
        for t in range(15):
            for lag_index, lag in enumerate(LAGS):
                np.testing.assert_array_equal(blocks[0, t, lag_index], [100*(t+lag), 100*(t+lag)+1])
        np.testing.assert_array_equal(blocks[:, :, 0], source[:, 3:18])
        for onset, total in ((2, 30), (7, 21)):
            with self.assertRaises(ValueError):
                source_slices(onset, total)

    def test_physical_scaling_and_arm_centering_exclude_held_direction(self):
        # Risk: held-direction values must not determine physical normalization
        # or the post-permutation training-column means.
        rng = np.random.default_rng(50702)
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"t{i}" for i in range(24)])
        x, y = rng.normal(size=(24, 18, 3)), rng.normal(size=(24, 15, 2))
        blocks = history_blocks(x)
        original = prepare_split(blocks, y, labels, 3)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[labels == 3], changed_y[labels == 3] = 1e8, -1e8
        changed = prepare_split(history_blocks(changed_x), changed_y, labels, 3)
        for field in ("feature_mean", "feature_sd"):
            np.testing.assert_array_equal(getattr(original["history_state"], field),
                                          getattr(changed["history_state"], field))
        for key in ("residual", "intercept", "fitting_standardized"):
            np.testing.assert_array_equal(original[key], changed[key])
        maps = channel_maps(ids, 3, "synthetic", 0)
        train, test = original["train"], original["test"]
        first = center_arm(permute_channels(original["fitting_standardized"], maps[train]),
                           permute_channels(standardize_history(blocks[test], original["history_state"]), maps[test]))
        other = center_arm(permute_channels(changed["fitting_standardized"], maps[train]),
                           permute_channels(standardize_history(history_blocks(changed_x)[test], changed["history_state"]), maps[test]))
        np.testing.assert_array_equal(first[0], other[0])
        np.testing.assert_array_equal(first[2], other[2])
        np.testing.assert_allclose(first[0].mean(axis=(0, 1)), 0, atol=5e-16)

    def test_current_bin_stationary_map_is_nested_without_block_rescaling(self):
        # Risk: hidden sqrt(lag-count) scaling changes the stationary anchor's ridge metric.
        rng = np.random.default_rng(50703)
        labels = np.repeat(np.arange(8), 2)
        x, y = rng.normal(size=(16, 18, 4)), rng.normal(size=(16, 15, 2))
        blocks = history_blocks(x)
        split = prepare_split(blocks, y, labels, 0)
        train, test = split["train"], split["test"]
        fitting, evaluation, _ = center_arm(split["fitting_standardized"],
                                             standardize_history(blocks[test], split["history_state"]))
        flat_current = x[train, 3:18].reshape(-1, 4)
        expected = (x[test, 3:18]-flat_current.mean(axis=0))/flat_current.std(axis=0)
        beta = rng.normal(size=(4, 2))
        embedded = np.vstack((beta, np.zeros((12, 2))))
        np.testing.assert_allclose(evaluation@embedded, expected@beta, atol=3e-15, rtol=2e-13)
        np.testing.assert_allclose(fitting[:, :, :4].mean(axis=(0, 1)), 0, atol=3e-16)

    def test_trial_map_moves_whole_electrode_trajectories_and_accepts_identity(self):
        # Risk: independently shuffling lags/times changes temporal content;
        # rejecting identity permutations changes the registered control distribution.
        blocks = np.arange(2*15*4*3).reshape(2, 15, 4, 3).astype(float)
        maps = np.array([[2, 0, 1], [1, 2, 0]])
        actual = permute_channels(blocks, maps)
        for trial in range(2):
            for destination, source in enumerate(maps[trial]):
                np.testing.assert_array_equal(actual[trial, :, :, destination], blocks[trial, :, :, source])
        ids = np.array(["trial10", "trial2"])
        for replicate in REPLICATES:
            first = channel_maps(ids, 3, "synthetic", replicate)
            np.testing.assert_array_equal(first, channel_maps(ids, 3, "synthetic", replicate))
            np.testing.assert_array_equal(first[::-1], channel_maps(ids[::-1], 3, "synthetic", replicate))
        class IdentityDraw:
            def __init__(self):
                self.calls = 0
            def seed(self, literal, version):
                self.literal, self.version = literal, version
            def shuffle(self, values):
                self.calls += 1
        draw = IdentityDraw()
        with patch("primary_history.random.Random", return_value=draw):
            identity = channel_maps(np.array(["trial10"]), 3, "synthetic", 2)
        np.testing.assert_array_equal(identity, [[0, 1, 2]])
        self.assertEqual(draw.calls, 1)
        self.assertEqual(draw.version, 2)
        self.assertEqual(draw.literal, "ep05-lmp-history-channel-label-v1|20261003|synthetic|trial10|2")

    def test_label_arm_uses_its_changed_gram_and_matches_independent_solve(self):
        # Risk: copying the previous row-shuffle shared-Gram shortcut would fit
        # the wrong channel-label model after trial-specific permutations.
        rng = np.random.default_rng(50705)
        train = rng.normal(size=(4, 15, 4, 3))
        test = rng.normal(size=(2, 15, 4, 3))
        maps_train = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]])
        maps_test = np.array([[2, 0, 1], [1, 2, 0]])
        paired, _, _ = center_arm(train, test)
        fitting, evaluation, _ = center_arm(permute_channels(train, maps_train), permute_channels(test, maps_test))
        flat = fitting.reshape(60, 12)
        self.assertGreater(np.linalg.norm(flat.T@flat-paired.reshape(60, 12).T@paired.reshape(60, 12)), 1.)
        residual = rng.normal(size=(60, 2))
        actual = ridge_predictions(fitting, evaluation, residual)
        for index, penalty in enumerate(LAMBDAS):
            coefficient = np.linalg.solve(flat.T@flat+60*penalty*np.eye(12), flat.T@residual)
            np.testing.assert_allclose(actual[index], evaluation.reshape(30, 12)@coefficient,
                                       atol=4e-15, rtol=2e-12)

    def test_constant_history_keeps_exact_baseline_and_targets_unchanged(self):
        # Risk: preprocessing/intercept differences must not create an artificial neural gain.
        rng = np.random.default_rng(50706)
        labels = np.repeat(np.arange(8), 2)
        ids = np.array([f"trial{i}" for i in range(16)])
        x, y = np.ones((16, 18, 3)), rng.normal(size=(16, 15, 2))
        original_x, original_y = x.copy(), y.copy()
        banks, baseline, predictions = fit_banks(x, y, labels, ids, "constant")
        for prediction in predictions[RECIPES[0]]:
            np.testing.assert_array_equal(prediction, baseline)
        for replicate in predictions[RECIPES[1]]:
            for prediction in replicate:
                np.testing.assert_array_equal(prediction, baseline)
        np.testing.assert_array_equal(x, original_x)
        np.testing.assert_array_equal(y, original_y)
        maps = banks[RECIPES[0]]["channel_label_trial_maps"]
        self.assertEqual(maps["trial_ids"], ids.tolist())
        self.assertEqual(maps["replicate_ids"], list(REPLICATES))

    def test_mean_seed_score_is_not_score_of_mean_prediction(self):
        # Risk: eight controls cannot become an unintended prediction ensemble.
        rng = np.random.default_rng(50707)
        labels = np.repeat(np.arange(8), 2)
        y, noise = rng.normal(size=(16, 15, 2)), rng.normal(size=(16, 15, 2))
        banks, predictions = [], []
        for replicate in REPLICATES:
            prediction = y+(-1 if replicate % 2 else 1)*noise
            predictions.append(prediction)
            banks.append(score_bank(y, labels, np.zeros_like(y),
                                    np.repeat(prediction[None], len(LAMBDAS), axis=0), RECIPES[1]))
        averaged = average_control_banks(banks)
        self.assertAlmostEqual(averaged["lambda_rows"][0]["candidate_SSE"], float(np.square(noise).sum()))
        self.assertLess(float(np.square(y-np.mean(predictions, axis=0)).sum()), 1e-25)

    def test_outer_history_lambda_exclusion_and_same_control_penalty(self):
        # Risk: a held session or favorable control score must not choose its penalty.
        def bank(recipe, reverse=False):
            rows = []
            for i, p in enumerate(LAMBDAS):
                delta = .01*(4-i if reverse else i)
                rows.append({"lambda": p, "delta_R2": delta, "identity_increment": .01,
                             "candidate_SSE": 5-10*delta, "wrong_pair_SSE": 5-10*delta+.1,
                             "candidate_R2": .5+delta})
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "lag_offsets": list(LAGS),
                    "baseline_R2": .5, "baseline_SSE": 5., "denominator": 10.,
                    "direction_counts": [2]*8, "lambda_rows": rows}
        sessions = {s: {RECIPES[0]: bank(RECIPES[0]),
                        RECIPES[1]: average_control_banks([bank(RECIPES[1], True) for _ in REPLICATES])}
                    for s in SESSIONS}
        before, _ = summarize_banks(sessions)
        for held in SESSIONS:
            changed = copy.deepcopy(sessions)
            for i, row in enumerate(changed[held][RECIPES[0]]["lambda_rows"]):
                row["delta_R2"] = 1e8*(5-i)
            after, _ = summarize_banks(changed)
            for first_panel, second_panel in zip(before, after):
                first = next(r for r in first_panel["sessions"] if r["session_id"] == held)
                second = next(r for r in second_panel["sessions"] if r["session_id"] == held)
                self.assertEqual(first["lambda_index"], 4)
                self.assertEqual(first["lambda_index"], second["lambda_index"])
                self.assertEqual(first["paired_inner_animal_balanced_scores"], second["paired_inner_animal_balanced_scores"])

    def test_reference_check_preserves_distinct_declared_windows(self):
        # Risk: relabeling the wider feature window as the target window would
        # conceal a changed source scope while asserting a replay.
        source = {"native_LMP_offsets": [-3, 14], "native_target_offsets": [0, 14],
                  "lag_offsets": list(LAGS), "native_bin_seconds": .03}
        original = copy.deepcopy(source)
        with patch("run_primary_history.check_reference_after_fit") as prior:
            check_history_reference_after_fit(None, "session", source, None, None, None, None)
            self.assertIs(prior.call_args.args[2], source)
            self.assertEqual(source, original)
            self.assertNotIn("native_offsets", source)
            broken = {**source, "native_LMP_offsets": [0, 14]}
            with self.assertRaises(ValueError):
                check_history_reference_after_fit(None, "session", broken, None, None, None, None)
            self.assertEqual(prior.call_count, 1)


if __name__ == "__main__":
    unittest.main()
