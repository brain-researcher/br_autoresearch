"""Independent source-free checks for successive PLS2 and full-label controls."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

import primary_history_pls as model
from primary_capacity import score_bank
from primary_covariance_svd import trial_map
from primary_history import history_blocks
from primary_history_repeatability_weight import history_trace_weights, prepare_fold
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from run_primary_history_blend import restore_target_layout
from test_primary_elapsed_only import direct_wrong_pair_sse, example


def analytic_case():
    """Known C span is e0/e2; PLS's second score is X1 minus .5*X0."""
    gram = np.array([[2., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    root = np.linalg.cholesky(gram)
    fitting = np.vstack((root.T, -root.T))
    cross = np.array([[1., 0.], [0., 0.], [0., .25]])
    residual = fitting @ np.linalg.solve(gram, cross)
    return fitting, residual


def direct_heads(scores, evaluation, residual):
    n, k = scores.shape
    if k == 0:
        return np.zeros((len(LAMBDAS), len(evaluation), 2))
    return np.stack([evaluation @ np.linalg.solve(scores.T @ scores + n * penalty * np.eye(k),
                                                  scores.T @ residual) for penalty in LAMBDAS])


class HistoryPLSTest(unittest.TestCase):
    def test_deflated_second_direction_leaves_original_cross_covariance_span(self):
        fitting, residual = analytic_case()
        scores, state = model.fit_pls(fitting, residual)
        self.assertEqual(state["retained_components"], 2)
        self.assertEqual(state["component_cap"], 2)
        expected = np.column_stack((fitting[:, 0], fitting[:, 1] - .5 * fitting[:, 0]))
        np.testing.assert_allclose(scores, expected, atol=3e-13, rtol=3e-12)
        self.assertAlmostEqual(float(scores[:, 0] @ scores[:, 1]), 0., places=12)
        # No score standardization: the first component has variance 2/3.
        self.assertAlmostEqual(float(np.mean(scores[:, 0] ** 2)), 2 / 3, places=12)
        weights = np.column_stack([item["weight"] for item in state["components"]])
        loadings = np.column_stack([item["input_loading"] for item in state["components"]])
        rotation = weights @ np.linalg.inv(loadings.T @ weights)
        initial, _, _ = np.linalg.svd(fitting.T @ residual, full_matrices=False)
        outside = rotation[:, 1] - initial @ (initial.T @ rotation[:, 1])
        self.assertGreater(float(np.linalg.norm(outside)), .9)
        json.dumps(state, allow_nan=False)

    def test_sequential_inference_matches_effective_rotation_not_raw_weights(self):
        fitting, residual = analytic_case()
        _, state = model.fit_pls(fitting, residual)
        evaluation = np.array([[3., 1., 2.], [-1., 4., 2.], [2., -2., .5]])
        predicted = model.apply_pls(evaluation, state)
        expected = np.column_stack((evaluation[:, 0], evaluation[:, 1] - .5 * evaluation[:, 0]))
        np.testing.assert_allclose(predicted, expected, atol=3e-13, rtol=3e-12)
        weights = np.column_stack([item["weight"] for item in state["components"]])
        loadings = np.column_stack([item["input_loading"] for item in state["components"]])
        rotation = weights @ np.linalg.inv(loadings.T @ weights)
        np.testing.assert_allclose(predicted, evaluation @ rotation - state["score_mean"], atol=3e-13, rtol=3e-12)
        self.assertGreater(float(np.max(np.abs(predicted - (evaluation @ weights - state["score_mean"])))), 1.)
        # Applying to a different evaluation matrix cannot mutate fitted state.
        before = copy.deepcopy(state)
        model.apply_pls(evaluation * 1000, state)
        self.assertEqual(state, before)

    def test_exact_tie_canonical_direction_is_invariant_to_supplied_basis_rotation(self):
        fitting = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        scores, state = model.fit_pls(fitting, fitting)
        first = state["components"][0]
        self.assertEqual(first["tie_multiplicity"], 2)
        self.assertEqual(first["canonical_coordinate"], 0)
        np.testing.assert_array_equal(first["weight"], [1., 0.])
        original_svd = np.linalg.svd
        rotation = np.array([[.6, -.8], [.8, .6]])
        rotated_calls = []
        def rotated_svd(value, *args, **kwargs):
            left, singular, right = original_svd(value, *args, **kwargs)
            if len(singular) == 2 and singular[0] == singular[1] and singular[0] > 0:
                rotated_calls.append(True)
                return left @ rotation, singular, rotation.T @ right
            return left, singular, right
        with patch.object(model.np.linalg, "svd", side_effect=rotated_svd):
            other_scores, other = model.fit_pls(fitting, fitting)
        self.assertTrue(rotated_calls)
        np.testing.assert_allclose(other_scores, scores, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(other["components"][0]["weight"], first["weight"], atol=2e-14)

    def test_rank_zero_rank_one_and_invalid_accepted_energy_are_distinct(self):
        fitting = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        for residual in (np.zeros_like(fitting), fitting * 1e-16):
            scores, state = model.fit_pls(fitting, residual)
            self.assertEqual(scores.shape, (4, 0))
            self.assertEqual(state["retained_components"], 0)
            self.assertEqual(model.apply_pls(fitting, state).shape, (4, 0))
        rank_one = np.array([[1., 0.], [-1., 0.], [2., 0.], [-2., 0.]])
        _, state = model.fit_pls(rank_one, rank_one)
        self.assertEqual(state["retained_components"], 1)
        for where in ("fitting", "residual"):
            x, u = fitting.copy(), fitting.copy()
            (x if where == "fitting" else u)[0, 0] = np.nan
            with self.subTest(where=where), self.assertRaises((ValueError, ArithmeticError)):
                model.fit_pls(x, u)
        with self.assertRaises((ValueError, ArithmeticError)):
            model.apply_pls(np.full((2, 2), np.inf), state)
        # Cross-covariance is finite/nonzero, but accepted t't underflows to0.
        # This must fail, not silently become a valid zero-component fit.
        with self.assertRaises((ValueError, ArithmeticError)):
            model.fit_pls(rank_one * 1e-200, rank_one * 1e200)

    def test_every_control_relearns_projection_and_head_on_its_own_whole_trial_targets(self):
        rng = np.random.default_rng(195001)
        fitting = rng.normal(size=(5, 15, 6))
        fitting -= fitting.mean(axis=(0, 1))
        evaluation = rng.normal(size=(2, 15, 6))
        residual = fitting.reshape(-1, 6)[:, :2] + rng.normal(size=(75, 2)) * .3
        residual -= residual.mean(axis=0)
        maps = [np.roll(np.arange(5), r % 5) for r in range(8)]
        candidate, controls, metadata = model.fit_fold_arms(fitting, evaluation, residual, maps)
        flat, test = fitting.reshape(-1, 6), evaluation.reshape(-1, 6)
        for arm, mapping in enumerate([np.arange(5)] + maps):
            target = residual.reshape(5, 15, 2)[mapping].reshape(75, 2)
            scores, state = model.fit_pls(flat, target)
            projected = model.apply_pls(test, state)
            expected = direct_heads(scores, projected, target)
            actual = candidate if arm == 0 else controls[arm - 1]
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
            if arm == 2:
                wrong_true_target_head = direct_heads(scores, projected, residual)
                self.assertGreater(float(np.max(np.abs(actual - wrong_true_target_head))), 1e-3)
                original_scores, original_state = model.fit_pls(flat, residual)
                frozen_projection = direct_heads(original_scores, model.apply_pls(test, original_state), target)
                self.assertGreater(float(np.max(np.abs(actual - frozen_projection))), 1e-3)
        np.testing.assert_array_equal(controls[0], candidate)
        self.assertEqual(metadata["supervised_projection_fits"], 9)
        self.assertFalse(metadata["exact_capacity_match"])
        self.assertEqual(metadata["target_shuffle_scope"], "projection_and_head")
        bad = maps.copy()
        bad[2] = np.array([0, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            model.fit_fold_arms(fitting, evaluation, residual, bad)

    def test_zero_component_heads_are_literal_zero_for_all_original_penalties(self):
        fitting, evaluation = np.zeros((3, 15, 4)), np.ones((2, 15, 4))
        residual = np.arange(90, dtype=float).reshape(45, 2)
        residual -= residual.mean(axis=0)
        maps = [np.roll(np.arange(3), r % 3) for r in range(8)]
        candidate, controls, _ = model.fit_fold_arms(fitting, evaluation, residual, maps)
        self.assertEqual(candidate.shape, (5, 30, 2))
        self.assertEqual(controls.shape, (8, 5, 30, 2))
        np.testing.assert_array_equal(candidate, 0.)
        np.testing.assert_array_equal(controls, 0.)

    def test_full_fit_preserves_HW_inputs_native_layout_and_one_baseline_addback(self):
        x, y, labels, ids, electrodes = example(195002)
        y = restore_target_layout(np.asarray(y, dtype=np.float64))
        calls = []
        original = model.fit_fold_arms
        def record(fitting, evaluation, residual, maps, check):
            result = original(fitting, evaluation, residual, maps, check)
            calls.append((fitting.copy(), evaluation.copy(), residual.copy(), maps, result))
            return result
        with patch.object(model, "fit_fold_arms", side_effect=record):
            banks, baseline, candidate, controls = model.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        self.assertEqual(len(calls), 8)
        for held, (fitting, evaluation, residual, maps, fitted) in enumerate(calls):
            split = prepare_fold(history_blocks(x), x[:, 3:18], y, labels, held)
            flat = split["fitting_history"].reshape(-1, 12)
            weights, _ = history_trace_weights(split["repeatability"]["a"], flat.T @ flat / len(flat))
            np.testing.assert_array_equal(fitting, split["fitting_history"] * np.tile(weights, 4))
            np.testing.assert_array_equal(evaluation, split["evaluation_history"] * np.tile(weights, 4))
            np.testing.assert_array_equal(residual, split["residual"])
            train, test = labels != held, labels == held
            for replicate, mapping in enumerate(maps):
                expected, _ = trial_map(ids[train], labels[train], "example", held, replicate)
                np.testing.assert_array_equal(mapping, expected)
                np.testing.assert_array_equal(labels[train][mapping], labels[train])
            base = split["template"].predict(labels[test] * np.pi / 4) + split["intercept"]
            np.testing.assert_array_equal(baseline[test], base)
            for index in range(5):
                np.testing.assert_array_equal(candidate[index, test], base + fitted[0][index].reshape(base.shape))
                for replicate in range(8):
                    np.testing.assert_array_equal(controls[replicate, index, test], base + fitted[1][replicate, index].reshape(base.shape))
        replay = score_bank(restore_target_layout(np.array(y, order="C")), labels,
                            restore_target_layout(np.array(baseline, order="C")), candidate, model.RECIPES[0])
        self.assertEqual(replay["lambda_rows"], banks[model.RECIPES[0]]["lambda_rows"])
        self.assertEqual(replay["denominator"], banks[model.RECIPES[0]]["denominator"])

    def test_held_direction_values_cannot_change_own_fits_and_controls_average_losses(self):
        x, y, labels, ids, electrodes = example(195003)
        y = restore_target_layout(np.asarray(y, dtype=np.float64))
        original = model.fit_fold_arms
        calls = []
        def record(fitting, evaluation, residual, maps, check):
            result = original(fitting, evaluation, residual, maps, check)
            calls.append((fitting.copy(), residual.copy(), result[2]))
            return result
        with patch.object(model, "fit_fold_arms", side_effect=record):
            banks, baseline, candidate, controls = model.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        held = 3
        changed_x, changed_y = x.copy(order="K"), y.copy(order="K")
        changed_x[labels == held] += 1e4
        changed_y[labels == held] -= 1e5
        first_calls = calls[:]
        calls.clear()
        with patch.object(model, "fit_fold_arms", side_effect=record):
            _, other_baseline, _, _ = model.fit_banks(changed_x, changed_y, labels, ids, "example", electrode_ids=electrodes)
        np.testing.assert_array_equal(calls[held][0], first_calls[held][0])
        np.testing.assert_array_equal(calls[held][1], first_calls[held][1])
        fitted_state = lambda info: {key: value for key, value in info.items() if not key.endswith("_seconds")}
        self.assertEqual(fitted_state(calls[held][2]), fitted_state(first_calls[held][2]))
        np.testing.assert_array_equal(other_baseline[labels == held], baseline[labels == held])
        for index in range(5):
            expected_sse = float(np.mean([np.square(y - controls[r, index]).sum() for r in range(8)]))
            expected_wrong = float(np.mean([direct_wrong_pair_sse(y, controls[r, index], labels) for r in range(8)]))
            row = banks[model.RECIPES[1]]["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_SSE"], expected_sse, places=8)
            self.assertAlmostEqual(row["wrong_pair_SSE"], expected_wrong, places=8)
        ensemble_sse = float(np.square(y - controls[:, 0].mean(axis=0)).sum())
        self.assertGreater(banks[model.RECIPES[1]]["lambda_rows"][0]["candidate_SSE"] - ensemble_sse, 1e-5)

    def test_outer_selector_uses_only_other_candidate_sessions_and_controls_share_it(self):
        # Only scalar score banks are synthesized; these are not scientific
        # outcomes. Actual tiny fitted states populate diagnostic dependencies.
        rng = np.random.default_rng(195004)
        x = rng.normal(size=(2, 15, 4))
        x -= x.mean(axis=(0, 1))
        u = rng.normal(size=(30, 2))
        u -= u.mean(axis=0)
        _, _, transform = model.fit_fold_arms(x, x[:1], u, [np.arange(2)] * 8)
        def row(index, gain):
            return {"lambda": LAMBDAS[index], "delta_R2": gain, "identity_increment": .02,
                    "candidate_SSE": 5. - 10. * gain, "wrong_pair_SSE": 5.2 - 10. * gain,
                    "candidate_R2": .5 + gain}
        base = {"lambdas": list(LAMBDAS), "baseline_R2": .5, "baseline_SSE": 5.,
                "denominator": 10., "direction_counts": [2] * 8}
        sessions, hw, anchors = {}, {}, {"HW": {}, "blend": {}}
        for session in SESSIONS:
            candidate = {**base, "recipe": model.RECIPES[0],
                         "lambda_rows": [row(i, .02 - .001 * i) for i in range(5)],
                         "direction_fit_metadata": [{"held_direction": held, "pls": transform} for held in range(8)]}
            seeds = [{**base, "recipe": model.RECIPES[1],
                      "lambda_rows": [row(i, .001 + .001 * i - replicate * 1e-5) for i in range(5)]}
                     for replicate in range(8)]
            sessions[session] = {model.RECIPES[0]: candidate, model.RECIPES[1]: model.average_banks(seeds)}
            hw[session] = {**base, "recipe": model.ANCHORS["HW"],
                           "lambda_rows": [row(i, .009 - .001 * abs(i - 2)) for i in range(5)]}
            for name, index in (("HW", 2), ("blend", 3)):
                anchors[name][session] = {"session_id": session, "animal": ANIMALS[session],
                    "lambda_index": index, "baseline_R2": .5, **row(index, .009)}
        result = model.summarize_banks(sessions, anchors, hw)
        self.assertTrue(all(item["lambda_index"] == 0 for item in result["candidate"]["sessions"]))
        self.assertTrue(all(item["lambda_index"] == 0 for item in result["control"]["sessions"]))
        self.assertTrue(all(item["lambda_index"] == 2 for item in result["original_anchors"]["HW"]["sessions"]))
        self.assertTrue(all(item["lambda_index"] == 3 for item in result["original_anchors"]["blend"]["sessions"]))
        held = SESSIONS[0]
        changed = copy.deepcopy(sessions)
        changed[held][model.RECIPES[0]]["lambda_rows"][4]["delta_R2"] = 1e6
        other = model.summarize_banks(changed, anchors, hw)
        self.assertEqual(other["candidate"]["sessions"][0]["lambda_index"], 0)
        self.assertNotIn(held, other["candidate"]["sessions"][0]["outer_training_sessions"])
        for session in SESSIONS:
            for item in sessions[session][model.RECIPES[0]]["lambda_rows"]:
                item["delta_R2"] = .02
        tied = model.summarize_banks(sessions, anchors, hw)
        self.assertTrue(all(item["lambda_index"] == 4 for item in tied["candidate"]["sessions"]))
        self.assertFalse(tied["promotion_authorized"])

    def test_duplicates_require_one_complete_prior_panel_and_every_control_draw(self):
        names = model.DUPLICATE_ANCHORS
        candidate = {session: {name: False for name in names} for session in SESSIONS}
        controls = {session: {name: [False] * 8 for name in (*names, "candidate")} for session in SESSIONS}
        for index, session in enumerate(SESSIONS):
            candidate[session]["HW" if index % 2 else "HW105"] = True
            controls[session]["candidate"][0] = True
        self.assertEqual(model.duplicate_accounting(candidate, controls)["scientific_trial_count_increment"], 2)
        for session in SESSIONS:
            candidate[session]["HW"] = True
            controls[session]["candidate"] = [True] * 8
        result = model.duplicate_accounting(candidate, controls)
        self.assertEqual(result["scientific_trial_count_increment"], 0)
        self.assertEqual(result["candidate_exact_duplicate_of"], ["HW"])
        self.assertEqual(result["control_exact_duplicate_of"], ["candidate"])
        controls[SESSIONS[-1]]["candidate"][-1] = False
        result = model.duplicate_accounting(candidate, controls)
        self.assertEqual(result["scientific_trial_count_increment"], 1)
        self.assertEqual(result["literal_null_control_count_increment"], 1)
        self.assertFalse(result["control_updates_primary_incumbent"])
        self.assertEqual(result["count_order"], list(model.RECIPES))
        with self.assertRaises(ValueError):
            model.duplicate_accounting({session: candidate[session] for session in SESSIONS[:-1]}, controls)


if __name__ == "__main__":
    unittest.main()
