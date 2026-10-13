"""Independent mapping, shared-ridge, original-HW and null-accounting fixtures."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import primary_capacity_matched_surrogate as surrogate
import primary_history as history
import primary_history_repeatability_weight as original
import run_primary_capacity_matched_surrogate as runner
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from test_primary_elapsed_only import example


class CapacityMatchedSurrogateTest(unittest.TestCase):
    def test_canonical_maps_move_complete_trajectories_within_fitting_directions(self):
        held = 3
        labels = np.concatenate([np.full(2+direction % 3, direction)
                                 for direction in range(8) if direction != held])
        rng = np.random.default_rng(165001)
        numbers = rng.permutation(np.arange(len(labels))*13+1)
        ids = np.array([f"example_trial{number:06d}" for number in numbers])
        fitting = np.arange(len(labels)*15*12).reshape(len(labels), 15, 12)
        mapping, records = surrogate.trial_map(ids, labels, "example", held, 5)
        np.testing.assert_array_equal(labels[mapping], labels)
        np.testing.assert_array_equal(np.sort(mapping), np.arange(len(labels)))
        for record in records:
            direction = record["direction"]
            ordered = sorted(np.flatnonzero(labels == direction), key=lambda index: numbers[index])
            self.assertEqual(record["recipient_trial_ids"], ids[ordered].tolist())
            self.assertEqual(record["donor_trial_ids"], ids[mapping[ordered]].tolist())
            self.assertEqual(record["seed_literal"],
                f"ep05-lmp-hw-capacity-training-feature-surrogate-v1|20261004|example|3|5|{direction}")
            for recipient in ordered:
                np.testing.assert_array_equal(fitting[mapping][recipient], fitting[mapping[recipient]])
        reverse = np.arange(len(labels))[::-1]
        reversed_map, reversed_records = surrogate.trial_map(ids[reverse], labels[reverse], "example", held, 5)
        self.assertEqual(records, reversed_records)
        np.testing.assert_array_equal(ids[reverse][reversed_map], ids[mapping][reverse])
        bad = ids.copy()
        bad[0] = "wrong_trial000001"
        with self.assertRaises(ValueError):
            surrogate.trial_map(bad, labels, "example", held, 5)
        with self.assertRaises(ValueError):
            surrogate.trial_map(ids, np.full_like(labels, held), "example", held, 5)

    def test_identity_maps_are_accepted_once_without_redrawing(self):
        _, _, labels, ids, _ = example()
        train = labels != 2
        literals, shuffled = [], []
        class IdentityRandom:
            def seed(self, literal, version): literals.append((literal, version))
            def shuffle(self, values): shuffled.append(list(values))
        with patch.object(surrogate.random, "Random", IdentityRandom):
            mapping, records = surrogate.trial_map(ids[train], labels[train], "example", 2, 7)
        np.testing.assert_array_equal(mapping, np.arange(np.count_nonzero(train)))
        self.assertEqual(len(shuffled), 7)
        self.assertEqual(literals, [(f"ep05-lmp-hw-capacity-training-feature-surrogate-v1|20261004|example|2|7|{d}", 2)
                                   for d in range(8) if d != 2])
        self.assertTrue(all(record["identity_map"] for record in records))
        self.assertEqual(sum(record["fixed_point_count"] for record in records), np.count_nonzero(train))

    def test_one_shared_Gram_matches_direct_permuted_solve_and_smoother_conjugacy(self):
        rng = np.random.default_rng(165002)
        fitting, evaluation = rng.normal(size=(4, 15, 4)), rng.normal(size=(2, 15, 4))
        residual = rng.normal(size=(60, 2))
        maps = [np.array([0, 1, 2, 3]) if r == 0 else np.array([1, 0, 3, 2]) for r in range(8)]
        eigh = np.linalg.eigh
        with patch.object(surrogate.np.linalg, "eigh", wraps=eigh) as decomposition:
            predictions, metadata = surrogate.shared_surrogate_predictions(fitting, evaluation, residual, maps)
        self.assertEqual(decomposition.call_count, 1)
        self.assertEqual(metadata["shared_Gram_eigendecompositions"], 1)
        self.assertEqual(metadata["surrogate_specific_Gram_eigendecompositions"], 0)
        phi, ev = fitting.reshape(-1, 4), evaluation.reshape(-1, 4)
        gram = phi.T@phi
        for replicate, mapping in enumerate(maps):
            permuted = fitting[mapping].reshape(-1, 4)
            np.testing.assert_allclose(permuted.T@permuted, gram, rtol=2e-14, atol=2e-14)
            for index, penalty in enumerate(LAMBDAS):
                coefficient = np.linalg.solve(permuted.T@permuted+len(phi)*penalty*np.eye(4),
                                              permuted.T@residual)
                np.testing.assert_allclose(predictions[replicate, index], ev@coefficient,
                                           rtol=3e-12, atol=3e-12)
        row_map = (15*maps[1][:, None]+np.arange(15)[None, :]).reshape(-1)
        permuted = fitting[maps[1]].reshape(-1, 4)
        for penalty in LAMBDAS:
            smoother = phi@np.linalg.solve(gram+len(phi)*penalty*np.eye(4), phi.T)
            other = permuted@np.linalg.solve(permuted.T@permuted+len(phi)*penalty*np.eye(4), permuted.T)
            np.testing.assert_allclose(other, smoother[np.ix_(row_map, row_map)], rtol=3e-12, atol=3e-12)
            self.assertAlmostEqual(np.trace(other), np.trace(smoother), places=12)
            self.assertAlmostEqual(np.trace(other@other), np.trace(smoother@smoother), places=12)

    def test_malformed_permutations_stop_and_exact_identity_is_not_whole_recipe_duplication(self):
        rng = np.random.default_rng(165003)
        fitting, evaluation = rng.normal(size=(4, 15, 3)), rng.normal(size=(2, 15, 3))
        residual = rng.normal(size=(60, 2))
        identity, reverse = np.arange(4), np.arange(4)[::-1]
        maps = [identity]+[reverse]*7
        _, metadata = surrogate.shared_surrogate_predictions(fitting, evaluation, residual, maps)
        self.assertEqual(metadata["whole_fitting_map_identity_by_replicate"], [True]+[False]*7)
        self.assertEqual(metadata["surrogate_cross_product_equals_original_by_replicate"], [True]+[False]*7)
        for invalid in (np.array([0, 0, 2, 3]), np.array([0., 1., 2., 3.]), np.array([0, 1, 2])):
            with self.assertRaises(ValueError):
                surrogate.shared_surrogate_predictions(fitting, evaluation, residual, [invalid]+[identity]*7)
        predictions, zero = surrogate.shared_surrogate_predictions(np.zeros_like(fitting), evaluation, residual, maps)
        self.assertTrue(all(zero["surrogate_cross_product_equals_original_by_replicate"]))
        self.assertTrue(all(zero["surrogate_design_equals_original_by_replicate"]))
        np.testing.assert_array_equal(predictions, np.zeros_like(predictions))

    def test_full_fit_preserves_HW_state_true_residual_addback_and_evaluation_pairing(self):
        x, y, labels, ids, electrodes = example(165004)
        seen = []
        solve = surrogate.shared_surrogate_predictions
        def record(fitting, evaluation, residual, maps, check):
            values = solve(fitting, evaluation, residual, maps, check)
            seen.append((fitting.copy(), evaluation.copy(), residual.copy(), [m.copy() for m in maps], values[0].copy()))
            return values
        with patch.object(surrogate, "shared_surrogate_predictions", side_effect=record):
            bank, baseline, predictions = surrogate.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        self.assertEqual(len(seen), 8)
        for held in range(8):
            split = original.prepare_fold(history.history_blocks(x), x[:, 3:18], y, labels, held)
            fitting = split["fitting_history"]
            flat = fitting.reshape(-1, fitting.shape[-1])
            weights, info = original.history_trace_weights(split["repeatability"]["a"], flat.T@flat/len(flat))
            repeated = np.tile(weights, 4)
            np.testing.assert_array_equal(seen[held][0], fitting*repeated)
            np.testing.assert_array_equal(seen[held][1], split["evaluation_history"]*repeated)
            np.testing.assert_array_equal(seen[held][2], split["residual"])
            metadata = bank["direction_fit_metadata"][held]
            self.assertEqual(metadata["paired_weights"], info)
            for key in ("physical_mean", "physical_sd"):
                np.testing.assert_array_equal(metadata[key], split["weight_state"][key])
            train, test = labels != held, labels == held
            base = split["template"].predict(labels[test]*np.pi/4)+split["intercept"]
            np.testing.assert_array_equal(baseline[test], base)
            for replicate in range(8):
                mapping, records = surrogate.trial_map(ids[train], labels[train], "example", held, replicate)
                np.testing.assert_array_equal(seen[held][3][replicate], mapping)
                self.assertEqual(metadata["surrogate_trial_maps"][replicate], records)
                for index in range(5):
                    np.testing.assert_array_equal(predictions[replicate, index, test],
                        base+seen[held][4][replicate, index].reshape(base.shape))

    def test_held_data_cannot_fit_shared_design_response_or_maps(self):
        x, y, labels, ids, electrodes = example(165005)
        other_x, other_y = x.copy(order="K"), y.copy()
        held = 2
        other_x[labels == held] += 1e5
        other_y[labels == held] -= 1e5
        seen = []
        solve = surrogate.shared_surrogate_predictions
        def record(fitting, evaluation, residual, maps, check):
            seen.append((fitting.copy(), residual.copy(), [m.copy() for m in maps]))
            return solve(fitting, evaluation, residual, maps, check)
        with patch.object(surrogate, "shared_surrogate_predictions", side_effect=record):
            first = surrogate.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)[0]
            second = surrogate.fit_banks(other_x, other_y, labels, ids, "example", electrode_ids=electrodes)[0]
        for left, right in zip(seen[held][:2], seen[8+held][:2]):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(seen[held][2], seen[8+held][2]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first["direction_fit_metadata"][held], second["direction_fit_metadata"][held])

    def test_scoring_averages_seed_losses_and_retains_identity_draws_not_prediction_ensemble(self):
        _, y, labels, _, _ = example(165006, equal_counts=True)
        baseline = np.zeros_like(y)
        predictions = np.stack([np.stack([baseline+(r-3.5)*(index+1) for index in range(5)])
                                for r in range(8)])
        banks = [score_bank(y, labels, baseline, prediction, surrogate.RECIPE) for prediction in predictions]
        result = surrogate.average_banks(banks)
        ensemble = score_bank(y, labels, baseline, predictions.mean(axis=0), surrogate.RECIPE)
        for index, row in enumerate(result["lambda_rows"]):
            correct = np.mean([bank["lambda_rows"][index]["candidate_SSE"] for bank in banks])
            wrong = np.mean([bank["lambda_rows"][index]["wrong_pair_SSE"] for bank in banks])
            self.assertEqual(row["candidate_SSE"], correct)
            self.assertEqual(row["wrong_pair_SSE"], wrong)
            self.assertAlmostEqual(row["identity_increment"], (wrong-correct)/result["denominator"], places=13)
            self.assertGreater(correct-ensemble["lambda_rows"][index]["candidate_SSE"], 1.)
        self.assertEqual([row["replicate"] for row in result["surrogate_seed_rows"]], list(range(8)))
        with self.assertRaises(ValueError):
            surrogate.average_banks(banks[:7])

    def test_identity_feature_maps_replay_original_ridge_and_all_seed_duplicate_flags(self):
        rng = np.random.default_rng(165007)
        fitting, evaluation = rng.normal(size=(4, 15, 3)), rng.normal(size=(2, 15, 3))
        residual = rng.normal(size=(60, 2))
        predictions, metadata = surrogate.shared_surrogate_predictions(
            fitting, evaluation, residual, [np.arange(4)]*8)
        original_predictions = history.ridge_predictions(fitting, evaluation, residual)
        self.assertTrue(all(metadata["surrogate_cross_product_equals_original_by_replicate"]))
        self.assertTrue(all(metadata["surrogate_design_equals_original_by_replicate"]))
        for prediction in predictions:
            np.testing.assert_array_equal(prediction, original_predictions)

    def test_original_saved_lambda_is_used_without_surrogate_retuning_and_null_counts_once(self):
        def score(penalty, sse):
            return {"lambda": penalty, "candidate_SSE": sse, "candidate_R2": 1-sse/10.,
                    "wrong_pair_SSE": sse+.1, "delta_R2": (5.-sse)/10., "identity_increment": .01}
        anchors, sessions = {}, {}
        for session in SESSIONS:
            anchors[session] = {"recipe": surrogate.ANCHOR, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "baseline_SSE": 5., "denominator": 10., "direction_counts": [3]*8,
                "lambda_rows": [score(penalty, 4.+index*.01) for index, penalty in enumerate(LAMBDAS)]}
            seeds = [{"recipe": surrogate.RECIPE, **{key: anchors[session][key] for key in
                ("lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts")},
                "lambda_rows": [score(penalty, 3.-index*.1+replicate*.01)
                                for index, penalty in enumerate(LAMBDAS)]} for replicate in range(8)]
            bank = surrogate.average_banks(seeds)
            bank.update(original_HW_saved_prediction_rows=copy.deepcopy(anchors[session]["lambda_rows"]),
                physical_electrode_count=3, history_column_count=12, structural_original_HW_duplicate=False,
                direction_fit_metadata=[{"held_direction": held, "fitting_bin_rows": 315,
                    "paired_weighted_covariance": {"trace": 12., "numerical_rank": 12,
                        "ridge_df_by_lambda": [{"lambda": p, "effective_df": 12./(1+p)} for p in LAMBDAS]},
                    "capacity_identity": {"surrogate_cross_product_equals_original_by_replicate": [False]*8}}
                    for held in range(8)])
            sessions[session] = bank
        # Frozen anchor selects lambda=1 even though surrogate's best score is
        # at10. Own surrogate outcomes never choose a replacement penalty.
        anchor_panel = {"recipe": surrogate.ANCHOR, "sessions": [{"session_id": session,
            "animal": ANIMALS[session], "lambda_index": 3,
            "outer_training_sessions": [other for other in SESSIONS if other != session],
            **anchors[session]["lambda_rows"][3]} for session in SESSIONS]}
        for duplicate, charge in ((False, 1), (True, 0)):
            for bank in sessions.values():
                bank["structural_original_HW_duplicate"] = duplicate
            result = surrogate.summarize_banks(sessions, anchors, anchor_panel)
            self.assertEqual(result["scientific_trial_count_increment"], charge)
            self.assertEqual(result["literal_null_control_count_increment"], charge)
            self.assertEqual(result["alternative_explanation_count_increment"], charge)
            self.assertFalse(result["updates_primary_incumbent"])
            self.assertFalse(result["new_promotion_gate"])
            self.assertFalse(result["original_HW_refitted"])
            self.assertEqual([seed["replicate"] for seed in result["original_HW_minus_each_surrogate"]], list(range(8)))
            for row in result["surrogate"]["sessions"]:
                self.assertEqual(row["lambda_index"], 3)
                self.assertEqual(row["lambda"], 1.)
            contrast = result["original_HW_minus_mean_surrogate"]["sessions"][0]
            expected = (sessions[SESSIONS[0]]["lambda_rows"][3]["candidate_SSE"]
                        -anchors[SESSIONS[0]]["lambda_rows"][3]["candidate_SSE"])/10.
            self.assertEqual(contrast["delta_R2"], expected)
        altered = copy.deepcopy(anchor_panel)
        altered["sessions"][0]["outer_training_sessions"].append(SESSIONS[0])
        with self.assertRaises(ValueError):
            surrogate.summarize_banks(sessions, anchors, altered)

    def test_runner_validates_actual_fit_output_native_ids_directions_and_maps(self):
        x, y, labels, ids, electrodes = example(165008)
        bank, _, _ = surrogate.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        np.testing.assert_array_equal(bank["original_trial_ids"], ids)
        np.testing.assert_array_equal(bank["original_direction_labels"], labels)
        # Anchor replay has an independently tested reader boundary. Supply
        # only its already-replayed row field and flags for this validator test.
        bank["original_HW_saved_prediction_rows"] = copy.deepcopy(bank["lambda_rows"])
        source = {"session_id": "example", "native_bin_seconds": .03,
                  "native_LMP_offsets": [-3, 14], "native_target_offsets": [0, 14],
                  "lag_offsets": [0, -1, -2, -3], "common_LMP_electrodes": electrodes,
                  "complete_trials": len(y)}
        packet = {"status": "complete_capacity_matched_surrogate_session_bank", "session_id": "example",
                  "original_HW_refitted": False, "exact_original_HW_neural_state_replay": True,
                  "exact_original_HW_saved_arrays_and_scoring_replay": True,
                  "exact_original_HW_weighted_covariance_replay": True, "promotion_authorized": False,
                  "source": source, "bank": bank}
        runner.validate_session(packet, "example")
        altered = copy.deepcopy(packet)
        altered["bank"]["original_trial_ids"][0] = "wrong_trial000001"
        with self.assertRaises(ValueError):
            runner.validate_session(altered, "example")
        altered = copy.deepcopy(packet)
        altered["bank"]["original_direction_labels"][0] = 7
        with self.assertRaises(ValueError):
            runner.validate_session(altered, "example")
        altered = copy.deepcopy(packet)
        altered["bank"]["direction_fit_metadata"][0]["surrogate_trial_maps"][0][0]["seed_literal"] += "changed"
        with self.assertRaises(ValueError):
            runner.validate_session(altered, "example")
        altered = copy.deepcopy(packet)
        altered["bank"]["lambda_rows"][0]["candidate_SSE"] += 1.
        with self.assertRaises(ValueError):
            runner.validate_session(altered, "example")


if __name__ == "__main__":
    unittest.main()
