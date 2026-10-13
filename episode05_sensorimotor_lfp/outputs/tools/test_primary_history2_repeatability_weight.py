"""Independent source-free two-lag, shared-capacity and selection fixtures."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import primary_capacity_matched_surrogate as old_control
import primary_history as history
import primary_history_repeatability_weight as original
import primary_history2_repeatability_weight as two
import run_primary_history2_repeatability_weight as runner
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from test_primary_elapsed_only import direct_wrong_pair_sse, example


def data(seed=205001):
    x, y, labels, ids, electrodes = example(seed, equal_counts=True)
    # Both provider inputs are time-fast within each trial.
    y = np.stack([row.T.copy(order="C").T for row in y])
    return x, y, labels, ids, electrodes


def complete_panel():
    x, y, labels, ids, electrodes = data(205006)
    banks, baseline, predictions = two.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
    candidate = banks[two.RECIPES[0]]
    candidate["exact_prediction_equals_components"] = {name: [[False]*5 for _ in range(5)] for name in two.ANCHORS}
    candidate["exact_surrogate_equals_original_HW_surrogate_by_lambda"] = [[False]*5 for _ in range(5)]
    sessions = {session: copy.deepcopy(banks) for session in SESSIONS}
    anchors = {name: {session: score_bank(y, labels, baseline, predictions[two.RECIPES[0]]+.1*(index+1), recipe)
                      for session in SESSIONS} for index, (name, recipe) in enumerate(two.ANCHORS.items())}
    panels = {}
    for name in two.ANCHORS:
        rows = []
        for held in SESSIONS:
            training = [session for session in SESSIONS if session != held]
            scores = [sum(float(np.mean([anchors[name][session]["lambda_rows"][index]["delta_R2"]
                                         for session in training if ANIMALS[session] == animal]))
                          for animal in ("Mihili", "Chewie"))/2 for index in range(5)]
            chosen = max(range(5), key=lambda index: (scores[index], LAMBDAS[index]))
            rows.append({"session_id": held, "animal": ANIMALS[held], "lambda_index": chosen,
                         "outer_training_sessions": training, **anchors[name][held]["lambda_rows"][chosen]})
        panels[name] = {"recipe": two.ANCHORS[name], "sessions": rows}
    return sessions, anchors, panels


def set_scores(bank, increments):
    for row, increment in zip(bank["lambda_rows"], increments):
        correct = bank["baseline_SSE"]-increment*bank["denominator"]
        row.update(delta_R2=increment, candidate_SSE=correct,
                   candidate_R2=1-correct/bank["denominator"], wrong_pair_SSE=correct+.001*bank["denominator"],
                   identity_increment=.001)


class History2RepeatabilityTest(unittest.TestCase):
    def test_lag_major_current_then_previous_exact_original_retained_state(self):
        x, y, labels, _, _ = data()
        blocks = history.history_blocks(x)
        self.assertEqual(tuple(two.LAGS), (0, -1))
        for time in (0, 14):
            np.testing.assert_array_equal(blocks[:, time, 0], x[:, time+3])
            np.testing.assert_array_equal(blocks[:, time, 1], x[:, time+2])
        split = two.prepare_two_lag_fold(blocks, x[:, 3:18], y, labels, 2)
        old = original.prepare_fold(blocks, x[:, 3:18], y, labels, 2)
        np.testing.assert_array_equal(split["fitting_two"], old["fitting_history"][:, :, :6])
        np.testing.assert_array_equal(split["evaluation_two"], old["evaluation_history"][:, :, :6])
        np.testing.assert_array_equal(split["residual"], old["residual"])
        np.testing.assert_array_equal(split["intercept"], old["intercept"])
        for key in ("physical_mean", "physical_sd"):
            np.testing.assert_array_equal(split["weight_state"][key], old["weight_state"][key])
        for key in ("T", "P", "r", "a"):
            np.testing.assert_array_equal(split["repeatability"][key], old["repeatability"][key])

    def test_omitted_expanded_lag_blocks_cannot_change_retained_model_state(self):
        x, y, labels, _, _ = data(205002)
        blocks = history.history_blocks(x)
        altered = blocks.copy(order="K")
        # Raw bins overlap across lags. Change ONLY omitted expanded columns;
        # editing a raw bin would also alter legitimate current/previous bins.
        altered[:, :, 2:, :] = 1e6+31*np.square(altered[:, :, 2:, :])
        first = two.prepare_two_lag_fold(blocks, x[:, 3:18], y, labels, 4)
        second = two.prepare_two_lag_fold(altered, x[:, 3:18], y, labels, 4)
        self.assertFalse(np.array_equal(first["fitting_history"][:, :, 6:],
                                       second["fitting_history"][:, :, 6:]))
        for key in ("fitting_two", "evaluation_two", "residual", "intercept"):
            np.testing.assert_array_equal(first[key], second[key])
        for key in ("physical_mean", "physical_sd"):
            np.testing.assert_array_equal(first["weight_state"][key], second["weight_state"][key])
        for key in ("T", "P", "r", "a"):
            np.testing.assert_array_equal(first["repeatability"][key], second["repeatability"][key])
        a = first["fitting_two"].reshape(-1, 6)
        b = second["fitting_two"].reshape(-1, 6)
        np.testing.assert_array_equal(a.T@a, b.T@b)

    def test_two_lag_trace_metric_recomputed_zero_uniform_and_exact_flags(self):
        a = np.array([.1, .4, 1.1])
        gram = np.diag([1., 2., 3., 4., 5., 6.])
        weights, info = two.history2_trace_weights(a, gram)
        q = np.array([5., 7., 9.])
        expected = np.sqrt(a*21./(a@q))
        np.testing.assert_array_equal(weights, expected)
        self.assertFalse(info["structural_identity"])
        self.assertAlmostEqual(float(np.square(weights)@q), 21., places=13)
        expanded = np.diag([1., 2., 3., 4., 5., 6., 100., 2., 1., 20., 4., 2.])
        old, _ = original.history_trace_weights(a, expanded)
        self.assertGreater(np.max(abs(weights-old)), .05)
        zero, zero_info = two.history2_trace_weights(a, np.zeros((6, 6)))
        np.testing.assert_array_equal(zero, np.ones(3))
        self.assertTrue(zero_info["structural_identity"])
        uniform, uniform_info = two.history2_trace_weights(np.full(3, .4), gram)
        np.testing.assert_array_equal(uniform, np.ones(3))
        self.assertTrue(uniform_info["structural_identity"])
        almost = np.array([.4, .4, np.nextafter(.4, 1.)])
        self.assertFalse(two.history2_trace_weights(almost, gram)[1]["structural_identity"])
        for bad in (np.eye(12), np.diag([-1., 2., 3., 4., 5., 6.])):
            with self.assertRaises(ValueError):
                two.history2_trace_weights(a, bad)

    def test_one_eigensystem_true_and_each_own_surrogate_rhs_match_direct_ridge(self):
        rng = np.random.default_rng(205003)
        fitting = rng.normal(size=(4, 15, 6))
        evaluation = rng.normal(size=(2, 15, 6))
        residual = rng.normal(size=(60, 2))
        maps = [np.arange(4)]+[np.array([1, 0, 3, 2])]*7
        eigh = np.linalg.eigh
        with patch.object(two.np.linalg, "eigh", wraps=eigh) as eig:
            candidate, controls, metadata = two.shared_predictions(fitting, evaluation, residual, maps)
        self.assertEqual(eig.call_count, 1)
        self.assertEqual(metadata["shared_Gram_eigendecompositions"], 1)
        self.assertEqual(metadata["surrogate_specific_Gram_eigendecompositions"], 0)
        phi, test = fitting.reshape(-1, 6), evaluation.reshape(-1, 6)
        gram = phi.T@phi
        for index, penalty in enumerate(LAMBDAS):
            head = np.linalg.solve(gram+len(phi)*penalty*np.eye(6), phi.T@residual)
            np.testing.assert_allclose(candidate[index], test@head, rtol=3e-12, atol=3e-12)
            for replicate, mapping in enumerate(maps):
                permuted = fitting[mapping].reshape(-1, 6)
                head = np.linalg.solve(gram+len(phi)*penalty*np.eye(6), permuted.T@residual)
                np.testing.assert_allclose(controls[replicate, index], test@head, rtol=3e-12, atol=3e-12)
        np.testing.assert_array_equal(candidate, controls[0])
        self.assertGreater(np.max(abs(candidate-controls[1])), .01)
        self.assertEqual(metadata["surrogate_cross_product_equals_original_by_replicate"], [True]+[False]*7)
        self.assertEqual(metadata["whole_fitting_map_identity_by_replicate"], [True]+[False]*7)
        with self.assertRaises(ValueError):
            two.shared_predictions(fitting, evaluation, residual, [np.zeros(4, dtype=int)]+maps[1:])

    def test_actual_fit_uses_original_prior_B_residual_native_XY_and_eight_original_maps(self):
        x, y, labels, ids, electrodes = data(205004)
        seen = []
        solve = two.shared_predictions
        def record(fitting, evaluation, residual, maps, check):
            result = solve(fitting, evaluation, residual, maps, check)
            seen.append((fitting.copy(), evaluation.copy(), residual.copy(),
                         [mapping.copy() for mapping in maps], result[0].copy(), result[1].copy()))
            return result
        with patch.object(two, "shared_predictions", side_effect=record):
            banks, baseline, predictions = two.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        self.assertIs(two.trial_map, old_control.trial_map)
        self.assertEqual(len(seen), 8)
        for held in range(8):
            old = original.prepare_fold(history.history_blocks(x), x[:, 3:18], y, labels, held)
            fitting, evaluation = old["fitting_history"][:, :, :6], old["evaluation_history"][:, :, :6]
            phi = fitting.reshape(-1, 6)
            weights, _ = two.history2_trace_weights(old["repeatability"]["a"], phi.T@phi/len(phi))
            np.testing.assert_array_equal(seen[held][0], fitting*np.tile(weights, 2))
            np.testing.assert_array_equal(seen[held][1], evaluation*np.tile(weights, 2))
            np.testing.assert_array_equal(seen[held][2], old["residual"])
            train, test = labels != held, labels == held
            base = old["template"].predict(labels[test]*np.pi/4)+old["intercept"]
            np.testing.assert_array_equal(baseline[test], base)
            metadata = banks[two.RECIPES[0]]["direction_fit_metadata"][held]
            for key in ("physical_mean", "physical_sd"):
                np.testing.assert_array_equal(metadata[key], old["weight_state"][key])
            for replicate in range(8):
                mapping, records = old_control.trial_map(ids[train], labels[train], "example", held, replicate)
                np.testing.assert_array_equal(seen[held][3][replicate], mapping)
                self.assertEqual(metadata["surrogate_trial_maps"][replicate], records)
                for index in range(5):
                    np.testing.assert_array_equal(predictions[two.RECIPES[1]][replicate, index, test],
                        base+seen[held][5][replicate, index].reshape(base.shape))
            for index in range(5):
                np.testing.assert_array_equal(predictions[two.RECIPES[0]][index, test],
                    base+seen[held][4][index].reshape(base.shape))
        candidate, control = (banks[recipe] for recipe in two.RECIPES)
        self.assertEqual(candidate["history_column_count"], 6)
        self.assertEqual(candidate["lag_offsets"], [0, -1])
        direct = score_bank(y, labels, baseline, predictions[two.RECIPES[0]], two.RECIPES[0])
        self.assertEqual(candidate["lambda_rows"], direct["lambda_rows"])
        for index, row in enumerate(candidate["lambda_rows"]):
            self.assertAlmostEqual(row["wrong_pair_SSE"],
                direct_wrong_pair_sse(y, predictions[two.RECIPES[0]][index], labels), places=7)
        seed_banks = [score_bank(y, labels, baseline, values, two.RECIPES[1])
                      for values in predictions[two.RECIPES[1]]]
        averaged = old_control.average_banks(seed_banks)
        self.assertEqual(control["lambda_rows"], averaged["lambda_rows"])
        ensemble = score_bank(y, labels, baseline, predictions[two.RECIPES[1]].mean(axis=0), two.RECIPES[1])
        self.assertGreater(control["lambda_rows"][0]["candidate_SSE"]-ensemble["lambda_rows"][0]["candidate_SSE"], .01)

    def test_held_direction_cannot_fit_two_lag_design_prior_residual_or_control_maps(self):
        x, y, labels, ids, electrodes = data(205005)
        other_x, other_y = x.copy(order="K"), y.copy(order="K")
        held = 3
        other_x[labels == held] += 1e5
        other_y[labels == held] -= 1e5
        seen = []
        solve = two.shared_predictions
        def record(fitting, evaluation, residual, maps, check):
            seen.append((fitting.copy(), residual.copy(), [mapping.copy() for mapping in maps]))
            return solve(fitting, evaluation, residual, maps, check)
        with patch.object(two, "shared_predictions", side_effect=record):
            first = two.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)[0]
            second = two.fit_banks(other_x, other_y, labels, ids, "example", electrode_ids=electrodes)[0]
        for left, right in zip(seen[held][:2], seen[8+held][:2]):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(seen[held][2], seen[8+held][2]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first[two.RECIPES[0]]["direction_fit_metadata"][held],
                         second[two.RECIPES[0]]["direction_fit_metadata"][held])

    def test_outer_selection_excludes_held_and_never_optimizes_null_penalty(self):
        sessions, anchors, panels = complete_panel()
        for banks in sessions.values():
            set_scores(banks[two.RECIPES[0]], [.02, .01, 0., -.01, -.02])
            set_scores(banks[two.RECIPES[1]], [-.02, -.01, 0., .01, .02])
            for seed in banks[two.RECIPES[1]]["surrogate_seed_rows"]:
                seed["lambda_rows"] = copy.deepcopy(banks[two.RECIPES[1]]["lambda_rows"])
        held = SESSIONS[0]
        # This extreme held-session outcome must not influence its own choice.
        set_scores(sessions[held][two.RECIPES[0]], [-10., -10., -10., -10., .05])
        result = two.summarize_banks(sessions, anchors, panels)
        selected = {row["session_id"]: row for row in result["candidate"]["sessions"]}
        null = {row["session_id"]: row for row in result["mean_surrogate"]["sessions"]}
        self.assertEqual(selected[held]["lambda_index"], 0)
        self.assertEqual(null[held]["lambda_index"], 0)
        for session, row in selected.items():
            training = [other for other in SESSIONS if other != session]
            scores = [sum(float(np.mean([sessions[s][two.RECIPES[0]]["lambda_rows"][index]["delta_R2"]
                                         for s in training if ANIMALS[s] == animal]))
                          for animal in ("Mihili", "Chewie"))/2 for index in range(5)]
            expected = max(range(5), key=lambda index: (scores[index], LAMBDAS[index]))
            self.assertEqual(row["outer_training_sessions"], training)
            self.assertEqual(row["lambda_index"], expected)
            self.assertEqual(null[session]["lambda_index"], expected)
        for banks in sessions.values():
            set_scores(banks[two.RECIPES[0]], [.01]*5)
        tied = two.summarize_banks(sessions, anchors, panels)
        self.assertTrue(all(row["lambda"] == 10. for row in tied["candidate"]["sessions"]))
        # With unequal session counts, animal-balanced choice differs from a
        # pooled session average: (+.02 in Mihili,-.015 in Chewie) beats zero.
        for session, banks in sessions.items():
            value = .02 if ANIMALS[session] == "Mihili" else -.015
            set_scores(banks[two.RECIPES[0]], [value, -.1, -.1, -.1, 0.])
        balanced = two.summarize_banks(sessions, anchors, panels)
        self.assertTrue(all(row["lambda_index"] == 0 for row in balanced["candidate"]["sessions"]))

    def test_separate_exact_duplicate_rules_energy_uniformity_and_all_sessions(self):
        sessions, anchors, panels = complete_panel()
        for banks in sessions.values():
            candidate = banks[two.RECIPES[0]]
            candidate["structural_control_equals_candidate"] = False
            for row in candidate["direction_fit_metadata"]:
                row["retained_past_fitting_energy_exactly_zero"] = True
                row["discarded_past_fitting_energy_exactly_zero"] = True
                row["h2_weights"]["structural_identity"] = True
        result = two.summarize_banks(sessions, anchors, panels)
        self.assertEqual(result["scientific_trial_count_by_recipe"], dict.fromkeys(two.RECIPES, 1))
        self.assertEqual(result["scientific_trial_count_increment"], 2)
        self.assertEqual(result["literal_null_control_count_increment"], 1)
        self.assertEqual(result["alternative_explanation_count_increment"], 1)
        self.assertTrue(result["eligible_for_comparable_primary_incumbent"])
        for banks in sessions.values():
            banks[two.RECIPES[0]]["exact_prediction_equals_components"]["W"] = [[True]*5 for _ in range(5)]
        duplicated = two.summarize_banks(sessions, anchors, panels)
        self.assertEqual(duplicated["scientific_trial_count_by_recipe"], {two.RECIPES[0]: 0, two.RECIPES[1]: 1})
        self.assertFalse(duplicated["eligible_for_comparable_primary_incumbent"])
        self.assertEqual(duplicated["structural_duplicate_of"][two.RECIPES[0]], ["W"])
        # One nonidentical selected session defeats global literal duplication.
        sessions[SESSIONS[-1]][two.RECIPES[0]]["exact_prediction_equals_components"]["W"] = [[False]*5 for _ in range(5)]
        self.assertEqual(two.summarize_banks(sessions, anchors, panels)["scientific_trial_count_increment"], 2)
        for banks in sessions.values():
            banks[two.RECIPES[0]]["structural_control_equals_candidate"] = True
        shared_rhs = two.summarize_banks(sessions, anchors, panels)
        self.assertEqual(shared_rhs["scientific_trial_count_by_recipe"], {two.RECIPES[0]: 1, two.RECIPES[1]: 0})
        self.assertEqual(shared_rhs["literal_null_control_count_increment"], 0)
        self.assertEqual(shared_rhs["alternative_explanation_count_increment"], 0)
        for banks in sessions.values():
            banks[two.RECIPES[0]]["structural_control_equals_candidate"] = False
            banks[two.RECIPES[0]]["exact_surrogate_equals_original_HW_surrogate_by_lambda"] = [[True]*5 for _ in range(5)]
        old_null = two.summarize_banks(sessions, anchors, panels)
        self.assertEqual(old_null["structural_duplicate_of"][two.RECIPES[1]], ["original_HW_capacity_surrogate"])
        self.assertEqual(old_null["scientific_trial_count_by_recipe"][two.RECIPES[1]], 0)

    def test_optional_frozen_blend_anchor_and_common_denominator_contrasts(self):
        sessions, anchors, panels = complete_panel()
        fixed_blend = copy.deepcopy(panels["W"])
        for row in fixed_blend["sessions"]:
            row["alpha"] = .5
            bank = sessions[row["session_id"]][two.RECIPES[0]]
            bank["blend_reference_row"] = copy.deepcopy(row)
            bank["exact_prediction_equals_blend"] = [True]*5
        result = two.summarize_banks(sessions, anchors, panels, fixed_blend)
        self.assertTrue(result["blend_reference_available"])
        self.assertEqual(result["structural_duplicate_of"][two.RECIPES[0]], ["blend"])
        self.assertEqual(len(result["reference_contrasts"]), 5)
        selected = {row["session_id"]: row for row in result["candidate"]["sessions"]}
        original = {row["session_id"]: row for row in fixed_blend["sessions"]}
        for row in result["reference_contrasts"]["blend_originally_selected"]["sessions"]:
            session = row["session_id"]
            denominator = sessions[session][two.RECIPES[0]]["denominator"]
            expected = (original[session]["candidate_SSE"]-selected[session]["candidate_SSE"])/denominator
            self.assertEqual(row["delta_R2"], expected)
        changed = copy.deepcopy(fixed_blend)
        changed["sessions"][0]["candidate_SSE"] += 1.
        with self.assertRaises(ValueError):
            two.summarize_banks(sessions, anchors, panels, changed)
        bad_anchors = copy.deepcopy(anchors)
        bad_anchors["W"][SESSIONS[0]]["denominator"] += 1.
        with self.assertRaises(ValueError):
            two.summarize_banks(sessions, bad_anchors, panels)

    def test_runner_validates_actual_fit_and_rejects_wrong_interval_maps_seed_losses(self):
        x, y, labels, ids, electrodes = data(205007)
        banks, _, _ = two.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        candidate = banks[two.RECIPES[0]]
        candidate.update(original_component_score_rows={name: copy.deepcopy(candidate["lambda_rows"]) for name in two.ANCHORS},
            exact_prediction_equals_components={name: [[False]*5 for _ in range(5)] for name in two.ANCHORS},
            exact_surrogate_equals_original_HW_surrogate_by_lambda=[[False]*5 for _ in range(5)])
        packet = {"status": "complete_history2_repeatability_session_banks", "session_id": "example",
            "source": {"session_id": "example", "native_bin_seconds": .03, "native_LMP_offsets": [-3, 14],
                "native_target_offsets": [0, 14], "lag_offsets": [0, -1, -2, -3],
                "common_LMP_electrodes": electrodes, "complete_trials": len(y)},
            "banks": banks, "original_HW_refitted": False, "original_W_refitted": False,
            "promotion_authorized": False, "scientific_trial_count_increment": 0,
            **dict.fromkeys(runner.REPLAY_FLAGS, True)}
        runner.validate_session(packet, "example")
        changed = copy.deepcopy(packet)
        changed["banks"][two.RECIPES[0]]["effective_LMP_offsets"] = [-3, 14]
        with self.assertRaises(ValueError):
            runner.validate_session(changed, "example")
        changed = copy.deepcopy(packet)
        changed["banks"][two.RECIPES[0]]["direction_fit_metadata"][0]["surrogate_trial_maps"][0][0]["donor_trial_ids"][0] = "wrong_trial000001"
        with self.assertRaises(ValueError):
            runner.validate_session(changed, "example")
        changed = copy.deepcopy(packet)
        changed["banks"][two.RECIPES[1]]["surrogate_seed_rows"][0]["lambda_rows"][0]["candidate_SSE"] += 1.
        with self.assertRaises(ValueError):
            runner.validate_session(changed, "example")
        changed = copy.deepcopy(packet)
        changed["banks"][two.RECIPES[0]]["structural_control_equals_candidate"] = not candidate["structural_control_equals_candidate"]
        with self.assertRaises(ValueError):
            runner.validate_session(changed, "example")


if __name__ == "__main__":
    unittest.main()
