"""Independent scientific fixtures for fixed HW105 and its pairing control."""
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import primary_history_rbf as model
import run_primary_history_rbf as runner
import run_primary_history_rbf_panel as dispatch
from primary_capacity import fit_nystrom, landmark_rows, raw_nystrom, score_bank
from primary_capacity_matched_surrogate import average_banks, trial_map
from primary_history import history_blocks
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from run_primary_history_blend import restore_target_layout
from test_primary_elapsed_only import example


def synthetic_panel():
    def scores(index, penalty, delta):
        sse = 5.-10*delta
        return {"lambda": penalty, "candidate_R2": 1-sse/10., "candidate_SSE": sse,
            "wrong_pair_SSE": sse+.2, "delta_R2": delta, "identity_increment": .02}
    sessions, anchors = {}, {n: {} for n in ("HW", "K")}
    original_rows = {n: {} for n in model.ANCHORS}
    for s in SESSIONS:
        base = {"lambdas": list(LAMBDAS), "baseline_R2": .5, "baseline_SSE": 5.,
                "denominator": 10., "direction_counts": [3]*8}
        candidate = {**base, "recipe": model.RECIPES[0],
            "lambda_rows": [scores(i, p, .02-i*.001) for i, p in enumerate(LAMBDAS)]}
        seeds = [{**base, "recipe": model.RECIPES[1],
            "lambda_rows": [scores(i, p, .001+i*.001-r*.00001) for i, p in enumerate(LAMBDAS)]}
            for r in range(8)]
        control = average_banks(seeds)
        sessions[s] = {model.RECIPES[0]: candidate, model.RECIPES[1]: control}
        for n in ("HW", "K"):
            # The historical policies have different fixed independently
            # selected penalties from the candidate, and from each other.
            best = 2 if n == "HW" else 4
            bank = {**base, "recipe": model.ANCHORS[n],
                "lambda_rows": [scores(i, p, .01-.0001*abs(i-best)) for i, p in enumerate(LAMBDAS)]}
            anchors[n][s] = bank
            original_rows[n][s] = {"session_id": s, "animal": ANIMALS[s], "lambda_index": best,
                "outer_training_sessions": [other for other in SESSIONS if other != s],
                "baseline_R2": .5, **bank["lambda_rows"][best]}
        original_rows["blend"][s] = {"session_id": s, "animal": ANIMALS[s], "lambda_index": 3,
            "outer_training_sessions": [other for other in SESSIONS if other != s],
            "baseline_R2": .5, "alpha": .5, **scores(3, 1., .011)}
    return sessions, anchors, original_rows


class HistoryRBFTest(unittest.TestCase):
    def test_exact_HW_state_and_inherited_kernel_operator(self):
        x, y, labels, ids, _ = example(185001)
        held = 3
        blocks = history_blocks(x)
        result = model.prepare_kernel_fold(blocks, x[:, 3:18], y, labels, ids, "example", held)
        old = prepare_fold(blocks, x[:, 3:18], y, labels, held)
        fitting = old["fitting_history"]
        flat = fitting.reshape(-1, fitting.shape[-1])
        weights, info = history_trace_weights(old["repeatability"]["a"], flat.T@flat/len(flat))
        self.assertEqual(result["paired_weights"], info)
        np.testing.assert_array_equal(result["weighted_fit"], fitting*np.tile(weights, 4))
        np.testing.assert_array_equal(result["weighted_test"], old["evaluation_history"]*np.tile(weights, 4))
        rows, chosen = landmark_rows(ids[labels != held], labels[labels != held], "example", held)
        np.testing.assert_array_equal(result["landmark_rows"], rows)
        self.assertEqual(result["landmark_trial_ids"], chosen)
        expected, state = fit_nystrom(result["kernel_fit"].reshape(-1, 12), rows, 12)
        np.testing.assert_array_equal(result["phi"].reshape(-1, 105), expected)
        np.testing.assert_array_equal(result["phi_test"].reshape(-1, 105),
            raw_nystrom(result["kernel_test"].reshape(-1, 12), state)-state.feature_mean)
        # Independent metric calculation checks that history's weights enter
        # the distance exactly once, with the inherited dimension bandwidth.
        z = result["weighted_fit"].reshape(-1, 12)
        kernel = np.exp(-np.square(z[:4, None]-z[rows][None]).sum(axis=2)/(2*12))
        raw = np.linalg.solve(state.cholesky, kernel.T).T
        np.testing.assert_allclose(expected[:4], raw-state.feature_mean, rtol=3e-9, atol=3e-9)

    def test_held_direction_cannot_change_kernel_training_state_or_native_residual(self):
        x, y, labels, ids, _ = example(185002)
        held = 2
        changed_x, changed_y = x.copy(order="K"), y.copy()
        changed_x[labels == held] += 1e5
        changed_y[labels == held] -= 1e5
        before = model.prepare_kernel_fold(history_blocks(x), x[:, 3:18], y, labels, ids, "example", held)
        after = model.prepare_kernel_fold(history_blocks(changed_x), changed_x[:, 3:18], changed_y,
            labels, ids, "example", held)
        for key in ("phi", "weighted_fit", "residual", "landmark_rows", "active_kernel_columns"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(before["paired_weights"], after["paired_weights"])
        self.assertEqual(before["landmark_trial_ids"], after["landmark_trial_ids"])
        for attribute in ("landmarks", "cholesky", "feature_mean"):
            np.testing.assert_array_equal(getattr(before["kernel_state"], attribute),
                                          getattr(after["kernel_state"], attribute))

    def test_constant_columns_are_lag_specific_and_zero_feature_branch_is_baseline(self):
        x, y, labels, ids, _ = example(185003)
        held, train = 3, labels != 3
        x[train, 3:, 0] = 2.
        x[~train, 3:, 0] += 900.
        state = model.prepare_kernel_fold(history_blocks(x), x[:, 3:18], y, labels, ids, "example", held)
        self.assertFalse(state["active_kernel_columns"][0])
        self.assertTrue(state["active_kernel_columns"][9])
        self.assertEqual(state["repeatability"]["a"][0], .1)
        np.testing.assert_array_equal(state["kernel_fit"][..., 0], 0.)
        np.testing.assert_array_equal(state["kernel_test"][..., 0], 0.)
        self.assertTrue(np.any(state["weighted_test"][..., 0] != 0))
        x[train] = 0.
        zero = model.prepare_kernel_fold(history_blocks(x), x[:, 3:18], y, labels, ids, "example", held)
        self.assertEqual(zero["kernel_state"].sigma_squared, 0.)
        np.testing.assert_array_equal(zero["phi"], 0.)
        np.testing.assert_array_equal(zero["phi_test"], 0.)
        maps = [trial_map(ids[train], labels[train], "example", held, r)[0] for r in range(8)]
        candidate, controls, _ = model.shared_heads(zero["phi"], zero["phi_test"], zero["residual"], maps)
        np.testing.assert_array_equal(candidate, 0.)
        np.testing.assert_array_equal(controls, 0.)

    def test_shared_kernel_Gram_heads_match_direct_permuted_ridge_and_identity(self):
        rng = np.random.default_rng(185004)
        fitting, evaluation = rng.normal(size=(5, 15, 7)), rng.normal(size=(2, 15, 7))
        residual = rng.normal(size=(75, 2))
        maps = [np.arange(5)]+[rng.permutation(5) for _ in range(7)]
        candidate, controls, metadata = model.shared_heads(fitting, evaluation, residual, maps)
        flat, test = fitting.reshape(-1, 7), evaluation.reshape(-1, 7)
        for arm, mapping in enumerate([np.arange(5)]+maps):
            shifted = fitting[mapping].reshape(-1, 7)
            for index, penalty in enumerate(LAMBDAS):
                coefficient = np.linalg.solve(shifted.T@shifted+len(flat)*penalty*np.eye(7), shifted.T@residual)
                actual = candidate[index] if arm == 0 else controls[arm-1, index]
                np.testing.assert_allclose(actual, test@coefficient, rtol=3e-12, atol=3e-12)
        np.testing.assert_array_equal(candidate, controls[0])
        self.assertEqual(metadata["shared_head_Gram_eigendecompositions"], 1)
        self.assertEqual(metadata["surrogate_specific_Gram_eigendecompositions"], 0)
        self.assertTrue(metadata["surrogate_cross_product_equals_candidate_by_replicate"][0])
        bad = maps.copy()
        bad[2] = np.array([0, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            model.shared_heads(fitting, evaluation, residual, bad)

    def test_control_mean_losses_and_exact_whole_trial_evaluation_pairing(self):
        rng = np.random.default_rng(185005)
        labels = np.repeat(np.arange(8), 2)
        y, baseline = rng.normal(size=(16, 15, 2)), np.zeros((16, 15, 2))
        values = np.stack([np.stack([rng.normal(size=y.shape)+(r-3.5) for _ in LAMBDAS]) for r in range(8)])
        banks = [score_bank(y, labels, baseline, values[r], model.RECIPES[1]) for r in range(8)]
        mean = average_banks(banks)
        averaged_predictions = score_bank(y, labels, baseline, values.mean(axis=0), model.RECIPES[1])
        for index in range(5):
            wrong = [sum(float(np.square(y[labels == d]-values[r, index, labels == d][::-1]).sum())
                         for d in range(8)) for r in range(8)]
            self.assertAlmostEqual(mean["lambda_rows"][index]["wrong_pair_SSE"], float(np.mean(wrong)), places=9)
            self.assertEqual(mean["lambda_rows"][index]["candidate_SSE"],
                float(np.mean([bank["lambda_rows"][index]["candidate_SSE"] for bank in banks])))
            self.assertGreater(mean["lambda_rows"][index]["candidate_SSE"],
                               averaged_predictions["lambda_rows"][index]["candidate_SSE"])

    def test_outer_selection_excludes_held_session_controls_and_original_anchor_choices(self):
        sessions, anchors, original = synthetic_panel()
        result = model.summarize_banks(sessions, original, anchors)
        self.assertTrue(all(r["lambda_index"] == 0 for r in result["candidate"]["sessions"]))
        self.assertTrue(all(r["lambda_index"] == 0 for r in result["control"]["sessions"]))
        self.assertTrue(all(r["lambda_index"] == 2 for r in result["original_anchors"]["HW"]["sessions"]))
        self.assertTrue(all(r["lambda_index"] == 4 for r in result["original_anchors"]["K"]["sessions"]))
        self.assertTrue(all(r["reference_lambda"] == LAMBDAS[0] for r in
            result["reference_contrasts"]["candidate_minus_K_lambda_matched"]["sessions"]))
        held = SESSIONS[0]
        changed = copy.deepcopy(sessions)
        changed[held][model.RECIPES[0]]["lambda_rows"][4]["delta_R2"] = 1e6
        new = model.summarize_banks(changed, original, anchors)
        self.assertEqual(new["candidate"]["sessions"][0]["lambda_index"], 0)
        self.assertNotIn(held, new["candidate"]["sessions"][0]["outer_training_sessions"])
        for s in SESSIONS:
            for row in sessions[s][model.RECIPES[0]]["lambda_rows"]:
                row["delta_R2"] = .02
        tied = model.summarize_banks(sessions, original, anchors)
        self.assertTrue(all(r["lambda_index"] == 4 for r in tied["candidate"]["sessions"]))

    def test_global_duplicates_require_one_same_panel_and_every_control_draw(self):
        names = (*model.ANCHORS, "baseline")
        candidate = {s: {n: False for n in names} for s in SESSIONS}
        controls = {s: {n: [False]*8 for n in (*names, "candidate")} for s in SESSIONS}
        self.assertEqual(model.duplicate_accounting(candidate, controls)["scientific_trial_count_increment"], 2)
        for i, s in enumerate(SESSIONS):
            candidate[s]["HW" if i % 2 else "K"] = True
            controls[s]["candidate"][0] = True
        self.assertEqual(model.duplicate_accounting(candidate, controls)["scientific_trial_count_increment"], 2)
        for s in SESSIONS:
            candidate[s]["HW"] = True
            controls[s]["candidate"] = [True]*8
        account = model.duplicate_accounting(candidate, controls)
        self.assertEqual(account["candidate_exact_duplicate_of"], ["HW"])
        self.assertEqual(account["control_exact_duplicate_of"], ["candidate"])
        self.assertEqual(account["scientific_trial_count_increment"], 0)
        controls[SESSIONS[-1]]["candidate"][-1] = False
        account = model.duplicate_accounting(candidate, controls)
        self.assertEqual(account["scientific_trial_count_increment"], 1)
        self.assertEqual(account["alternative_explanation_count_increment"], 1)
        with self.assertRaises(ValueError):
            model.duplicate_accounting({s: candidate[s] for s in SESSIONS[:-1]}, controls)

    def test_full_fit_addback_native_layout_and_actual_runner_metadata(self):
        x, y, labels, ids, electrodes = example(185006)
        y = restore_target_layout(np.asarray(y, dtype=np.float64))
        seen = []
        original_heads = model.shared_heads
        def record(fitting, evaluation, residual, maps, check):
            result = original_heads(fitting, evaluation, residual, maps, check)
            seen.append(result)
            return result
        with patch.object(model, "shared_heads", side_effect=record):
            banks, baseline, candidate, controls = model.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        for held, (fitted, shuffled, _) in enumerate(seen):
            split = prepare_fold(history_blocks(x), x[:, 3:18], y, labels, held)
            test = labels == held
            base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
            np.testing.assert_array_equal(baseline[test], base)
            for index in range(5):
                np.testing.assert_array_equal(candidate[index, test], base+fitted[index].reshape(base.shape))
                for r in range(8):
                    np.testing.assert_array_equal(controls[r, index, test], base+shuffled[r, index].reshape(base.shape))
        restored_y = restore_target_layout(np.array(y, order="C"))
        restored_b = restore_target_layout(np.array(baseline, order="C"))
        replay = score_bank(restored_y, labels, restored_b, candidate, model.RECIPES[0])
        self.assertEqual(replay["lambda_rows"], banks[model.RECIPES[0]]["lambda_rows"])
        self.assertEqual(replay["denominator"], banks[model.RECIPES[0]]["denominator"])
        source = {"session_id": "example", "native_bin_seconds": .03, "native_LMP_offsets": [-3, 14],
            "native_target_offsets": [0, 14], "lag_offsets": [0, -1, -2, -3],
            "common_LMP_electrodes": electrodes, "complete_trials": len(y)}
        item = {"status": runner.SESSION_STATUS, "session_id": "example", "banks": banks, "source": source,
            "exact_original_HW_neural_state_replay": True, "exact_original_HW_saved_arrays_and_scoring_replay": True,
            "exact_original_HW_weighted_covariance_replay": True, "original_HW_refitted": False,
            "promotion_authorized": False, "scientific_trial_count_increment": 0}
        runner.validate_session(item, "example")
        altered = copy.deepcopy(item)
        altered["banks"][model.RECIPES[0]]["direction_fit_metadata"][0]["landmark_trial_ids"][0] = "example_trial999999"
        with self.assertRaises(ValueError):
            runner.validate_session(altered, "example")
        altered = copy.deepcopy(item)
        altered["banks"][model.RECIPES[1]]["lambda_rows"][0]["candidate_SSE"] += 1.
        with self.assertRaises(ValueError):
            runner.validate_session(altered, "example")
        with self.assertRaises(ValueError):
            restore_target_layout(np.array(y, dtype=np.float32))

    def test_pilot_command_preserves_fixed_source_paths_and_internal_bound(self):
        args = SimpleNamespace(source=Path("/source"), trial_table=Path("/trials"), scalar_bank_root=Path("/scalar"),
                               hw_bank_root=Path("/hw"), bank_root=Path("/new"))
        command = dispatch.session_command(args, dispatch.PILOT)
        self.assertEqual(command[command.index("--session")+1], "Chewie_20150313")
        self.assertEqual(command[command.index("--max-seconds")+1], "600")
        self.assertEqual(command[command.index("--out")+1], "/new/Chewie_20150313")
        self.assertEqual(dispatch.CHILD_EXTERNAL_SECONDS, 900)


if __name__ == "__main__":
    unittest.main()
