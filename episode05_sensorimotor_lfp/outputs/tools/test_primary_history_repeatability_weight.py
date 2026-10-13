"""Independent fixed-history normalization, penalty and duplicate checks."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import primary_history as history
import primary_history_repeatability_weight as weighted
import primary_repeatability_weight as current
import run_primary_history_repeatability_weight as runner
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def example(seed=105001):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 2)
    wave = np.sin(np.linspace(-np.pi, np.pi, 18))
    x = rng.normal(size=(16, 18, 4))+wave[None, :, None]*np.array([5., 2., .6, 0.])[None, None, :]
    y = rng.normal(size=(16, 15, 2))+.5*x[:, 3:, :2]
    ids = np.array([f"example_trial{i+1:06d}" for i in range(16)])
    return x, y, labels, ids, [2, 7, 11, 19]


def reader_layout(dtype):
    """Source-free reproduction of both original transpose/astype/stack readers."""
    rng = np.random.default_rng(105008)
    labels = np.repeat(np.arange(8), 3)
    source_trials = [(rng.normal(size=(7, 18))*np.geomspace(1e-9, 1e12, 18)).astype(dtype)
                     for _ in range(len(labels))]
    # Both readers reorder physical rows before transpose. The selected
    # channel identities are fixed, and only the requested time window differs.
    physical_rows = [4, 1, 6, 0, 3, 5, 2]
    old_current = np.stack([raw[physical_rows, 3:18].T.astype(np.float64) for raw in source_trials])
    full_history = np.stack([raw[physical_rows, :18].T.astype(np.float64) for raw in source_trials])
    y = rng.normal(size=(len(labels), 15, 2))
    ids = np.array([f"example_trial{i+1:06d}" for i in range(len(labels))])
    return old_current, full_history, y, labels, ids, [2, 7, 11, 19, 23, 31, 43]


class HistoryRepeatabilityWeightTest(unittest.TestCase):
    def test_reader_layout_slice_exactly_replays_W_state_while_C_coercion_changes_arithmetic(self):
        for dtype in (np.float32, np.float64):
            old, x, y, labels, _, _ = reader_layout(dtype)
            sliced, coerced = x[:, 3:18, :], np.ascontiguousarray(x[:, 3:18, :])
            np.testing.assert_array_equal(old, sliced)
            np.testing.assert_array_equal(old, coerced)
            self.assertFalse(old.flags.c_contiguous)
            self.assertTrue(coerced.flags.c_contiguous)
            changed_mean = False
            for held in range(8):
                train = labels != held
                z, state = current.prepare_split(old, train)
                expected = current.repeatability_statistics(z[train], labels[train])
                result = weighted.prepare_fold(history.history_blocks(x), sliced, y, labels, held)
                for key in ("physical_mean", "physical_sd"):
                    np.testing.assert_array_equal(result["weight_state"][key], state[key])
                for key, value in expected.items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(result["repeatability"][key], value)
                    else:
                        self.assertEqual(result["repeatability"][key], value)
                _, altered = current.prepare_split(coerced, train)
                changed_mean |= not np.array_equal(altered["physical_mean"], state["physical_mean"])
            self.assertTrue(changed_mean, f"Adversarial {dtype} fixture must detect a C-layout reduction change")

    def test_fit_banks_preserves_original_reader_layout_for_exact_current_W_replay(self):
        for dtype in (np.float32, np.float64):
            old, x, y, labels, ids, channels = reader_layout(dtype)
            def no_head(fitting, evaluation, residual, check):
                return np.zeros((len(LAMBDAS), len(evaluation)*15, 2))
            with patch.object(weighted, "ridge_predictions", side_effect=no_head), \
                    patch.object(weighted, "projected_covariance", return_value={}):
                banks, _, _ = weighted.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
            for row in banks[weighted.RECIPES[0]]["direction_fit_metadata"]:
                train = labels != row["held_direction"]
                z, state = current.prepare_split(old, train)
                expected = current.repeatability_statistics(z[train], labels[train])
                for key in ("physical_mean", "physical_sd"):
                    np.testing.assert_array_equal(row[key], state[key])
                for key, value in expected.items():
                    self.assertEqual(row["repeatability"][key], value.tolist() if isinstance(value, np.ndarray) else value)

    def test_exact_lag_target_layout_and_separate_preserved_preprocessing(self):
        x, y, labels, _, _ = example()
        blocks = history.history_blocks(x)
        for lag_index, offset in enumerate((0, -1, -2, -3)):
            np.testing.assert_array_equal(blocks[:, :, lag_index], x[:, 3+offset:18+offset])
        held = 3
        original = history.prepare_split(blocks, y, labels, held)
        expected_fit, expected_eval, expected_center = history.center_arm(
            original["fitting_standardized"],
            history.standardize_history(blocks[original["test"]], original["history_state"]))
        result = weighted.prepare_fold(blocks, x[:, 3:], y, labels, held)
        np.testing.assert_array_equal(result["fitting_history"], expected_fit)
        np.testing.assert_array_equal(result["evaluation_history"], expected_eval)
        np.testing.assert_array_equal(result["history_center"], expected_center)
        z, state = current.prepare_split(x[:, 3:], labels != held)
        expected_statistics = current.repeatability_statistics(z[labels != held], labels[labels != held])
        for key in ("physical_mean", "physical_sd"):
            np.testing.assert_array_equal(result["weight_state"][key], state[key])
        for key in ("T", "P", "r", "a"):
            np.testing.assert_array_equal(result["repeatability"][key], expected_statistics[key])

    def test_trace_uses_all_lag_energy_and_matches_diagonal_generalized_ridge(self):
        # Electrode 0 has no current energy but does carry past information.
        diagonal = np.array([0., 1., 1., 3., 0., 0., 2., 0., 0., 1., 0., 0.])
        gram = np.diag(diagonal)
        base = np.array([.1, .7, .7])
        weights, info = weighted.history_trace_weights(base, gram)
        self.assertFalse(info["structural_identity"])
        self.assertEqual(info["positive_energy_columns"], [0, 1, 2])
        np.testing.assert_array_equal(info["history_energy_by_electrode"], [6., 1., 1.])
        np.testing.assert_allclose(weights**2, base*8/(base@np.array([6., 1., 1.])), rtol=1e-14)
        expanded = np.tile(weights, 4)
        self.assertAlmostEqual(float(expanded**2@diagonal), float(diagonal.sum()), places=13)
        rng = np.random.default_rng(105002)
        design = rng.normal(size=(25, 12))
        design[:, 6] = 0.
        gram = design.T@design/len(design)
        weights, info = weighted.history_trace_weights(np.array([.2, .6, 1.]), gram)
        expanded = np.tile(weights, 4)
        d = np.diag(expanded)
        cross = design.T@rng.normal(size=(25, 2))/len(design)
        for penalty in LAMBDAS:
            transformed = d@np.linalg.solve(d@gram@d+penalty*np.eye(12), d@cross)
            generalized = np.linalg.solve(gram+penalty*np.diag(1/expanded**2), cross)
            np.testing.assert_allclose(transformed, generalized, rtol=2e-11, atol=2e-12)
        self.assertAlmostEqual(np.trace(d@gram@d), np.trace(gram), places=12)
        same, same_info = weighted.history_trace_weights(np.full(3, .7), gram)
        np.testing.assert_array_equal(same, np.ones(3))
        self.assertTrue(same_info["structural_identity"])
        zero, zero_info = weighted.history_trace_weights(base, np.zeros((12, 12)))
        np.testing.assert_array_equal(zero, np.ones(3))
        self.assertTrue(zero_info["structural_identity"])

    def test_current_zero_past_active_and_held_values_cannot_change_fitting_prior(self):
        x, y, labels, _, _ = example(105003)
        x[:, 3:, 0] = 0.
        blocks = history.history_blocks(x)
        result = weighted.prepare_fold(blocks, x[:, 3:], y, labels, 2)
        self.assertEqual(result["repeatability"]["T"][0], 0.)
        self.assertEqual(result["repeatability"]["a"][0], .1)
        flat = result["fitting_history"].reshape(-1, 16)
        _, info = weighted.history_trace_weights(result["repeatability"]["a"], flat.T@flat/len(flat))
        self.assertGreater(info["history_energy_by_electrode"][0], 0.)
        changed_x = x.copy()
        changed_x[labels == 2] += 1e5
        changed_y = y+1e4*np.random.default_rng(105004).normal(size=y.shape)
        other = weighted.prepare_fold(history.history_blocks(changed_x), changed_x[:, 3:], changed_y, labels, 2)
        for key in ("T", "P", "r", "a"):
            np.testing.assert_array_equal(result["repeatability"][key], other["repeatability"][key])
        for key in ("physical_mean", "physical_sd"):
            np.testing.assert_array_equal(result["weight_state"][key], other["weight_state"][key])
        np.testing.assert_array_equal(result["fitting_history"], other["fitting_history"])

    def test_controls_repeat_physical_weights_over_lags_have_own_Grams_and_true_U(self):
        x, y, labels, ids, channels = example(105005)
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
        distinct = False
        for held in range(8):
            split = weighted.prepare_fold(history.history_blocks(x), x[:, 3:], y, labels, held)
            arms = [metadata[held]["paired_weights"], *metadata[held]["control_weights_by_replicate"]]
            paired_gram = None
            for arm, info in enumerate(arms):
                training, evaluation, residual = observed[9*held+arm]
                w = np.tile(info["weights"], 4)
                np.testing.assert_array_equal(residual, split["residual"])
                np.testing.assert_array_equal(training, split["fitting_history"]*w)
                np.testing.assert_array_equal(evaluation, split["evaluation_history"]*w)
                flat = training.reshape(-1, 16)
                gram = flat.T@flat/len(flat)
                self.assertAlmostEqual(np.trace(gram), info["original_trace"], places=11)
                if arm == 0:
                    paired_gram = gram
                elif not np.allclose(gram, paired_gram, rtol=1e-9, atol=1e-9):
                    distinct = True
        self.assertTrue(distinct)
        self.assertEqual(banks[weighted.RECIPES[0]]["weight_label_maps"],
                         [current.weight_map(channels, "example", r)[1] for r in range(8)])
        control = predictions[weighted.RECIPES[1]]
        for index in range(len(LAMBDAS)):
            row = banks[weighted.RECIPES[1]]["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_SSE"], float(np.mean(np.sum((control[:, index]-y)**2, axis=(1, 2, 3)))), places=9)
            self.assertAlmostEqual(row["wrong_pair_SSE"], float(np.mean([wrong_pair_sse(y, control[r, index], labels) for r in range(8)])), places=9)

    def test_identity_physical_maps_match_candidate_without_changing_history_signal(self):
        x, y, labels, ids, channels = example(105006)
        class Identity:
            def seed(self, literal, version): pass
            def shuffle(self, values): pass
        with patch.object(current.random, "Random", Identity):
            banks, _, predictions = weighted.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        self.assertFalse(banks[weighted.RECIPES[0]]["structural_identity"])
        self.assertFalse(banks[weighted.RECIPES[1]]["structural_identity"])
        self.assertTrue(banks[weighted.RECIPES[0]]["all_controls_equal_candidate"])
        for control in predictions[weighted.RECIPES[1]]:
            np.testing.assert_array_equal(control, predictions[weighted.RECIPES[0]])

    def test_no_past_energy_is_algebraic_current_W_duplicate_with_roundoff_tolerance(self):
        x, y, labels, ids, channels = example(105007)
        x[:, :17] = 0.
        banks, baseline, predictions = weighted.fit_banks(x, y, labels, ids, "example", electrode_ids=channels)
        old, old_baseline, old_predictions = current.fit_banks(x[:, 3:], y, labels, ids,
                                                               "example", electrode_ids=channels)
        self.assertTrue(banks[weighted.RECIPES[0]]["all_past_lags_zero"])
        np.testing.assert_array_equal(baseline, old_baseline)
        for new_recipe, old_recipe in zip(weighted.RECIPES, current.RECIPES):
            np.testing.assert_allclose(predictions[new_recipe], old_predictions[old_recipe], rtol=2e-10, atol=2e-10)
            for row, old_row in zip(banks[new_recipe]["lambda_rows"], old[old_recipe]["lambda_rows"]):
                self.assertAlmostEqual(row["candidate_SSE"], old_row["candidate_SSE"], places=8)

    def test_outer_lambda_both_anchor_families_and_independent_duplicate_charges(self):
        def bank(recipe, paired):
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "baseline_SSE": 5., "denominator": 10., "direction_counts": [2]*8,
                "lag_offsets": [0, -1, -2, -3], "all_past_lags_zero": False,
                "structural_identity": False, "all_controls_equal_candidate": False,
                "lambda_rows": [{"lambda": penalty, "delta_R2": .01*i if paired else .04-.01*i,
                    "identity_increment": .01, "candidate_SSE": 5., "wrong_pair_SSE": 5.1,
                    "candidate_R2": .5} for i, penalty in enumerate(LAMBDAS)]}
        sessions = {s: {weighted.RECIPES[0]: bank(weighted.RECIPES[0], True),
                    weighted.RECIPES[1]: weighted.average_control_banks([bank(weighted.RECIPES[1], False) for _ in range(8)])}
                    for s in SESSIONS}
        anchors = {name: {s: bank(recipe, False) for s in SESSIONS}
                   for name, recipe in weighted.ANCHORS.items()}
        anchor_panels = {name: {"recipe": recipe,
            "sessions": [{"session_id": s, "animal": ANIMALS[s], **anchors[name][s]["lambda_rows"][0]}
                         for s in SESSIONS]} for name, recipe in weighted.ANCHORS.items()}
        held = SESSIONS[0]
        for i, row in enumerate(sessions[held][weighted.RECIPES[0]]["lambda_rows"]):
            row["delta_R2"] = 1e7*(5-i)
        # Candidate and controls are separate recipes; no-past duplicates W
        # even when weighting is nonidentity in its current feature span.
        for candidate_identity, control_identity, same, no_past, candidate_charge, control_charge in (
                (False, False, False, False, 1, 1), (True, False, False, False, 0, 1),
                (False, True, False, False, 1, 0), (True, True, True, False, 0, 0),
                (False, False, True, False, 1, 0), (False, False, False, True, 0, 0)):
            for banks in sessions.values():
                banks[weighted.RECIPES[0]]["structural_identity"] = candidate_identity
                banks[weighted.RECIPES[1]]["structural_identity"] = control_identity
                banks[weighted.RECIPES[0]]["all_controls_equal_candidate"] = same
                for recipe in weighted.RECIPES:
                    banks[recipe]["all_past_lags_zero"] = no_past
            result = weighted.summarize_banks(sessions, anchors, anchor_panels)
            self.assertEqual(result["scientific_trial_count_increment"], candidate_charge+control_charge)
            self.assertEqual(result["control_trial_count_increment"], control_charge)
            self.assertEqual(result["structural_current_only_duplicate"], no_past)
            self.assertEqual(result["scientific_trial_count_by_recipe"],
                             dict(zip(weighted.RECIPES, (candidate_charge, control_charge))))
            self.assertEqual(set(result["anchor_contrasts"]), {
                "history_independently_selected", "history_HW_lambda_matched",
                "repeatability_independently_selected", "repeatability_HW_lambda_matched"})
            for panel in result["panels"]:
                row = next(row for row in panel["sessions"] if row["session_id"] == held)
                self.assertEqual(row["lambda_index"], 4)
                self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
            for name in weighted.ANCHORS:
                matched = next(row for row in result["matched_lambda_anchor_sessions"][name]
                               if row["session_id"] == held)
                self.assertEqual(matched["lambda"], LAMBDAS[4])
                self.assertEqual(matched["delta_R2"], anchors[name][held]["lambda_rows"][4]["delta_R2"])
                independent = next(row for row in anchor_panels[name]["sessions"] if row["session_id"] == held)
                self.assertEqual(independent["lambda"], LAMBDAS[0])

    def test_postfit_W_replay_requires_original_current_state_not_history_normalized_weights(self):
        # This is the post-fit metadata comparator; the preserved W validator
        # has its own tests and is mocked to isolate the new replay contract.
        previous = {"physical_electrode_ids": [2, 7], "weight_label_maps": [{"seed": 0}],
            "direction_fit_metadata": [{"held_direction": held, "fitting_bin_rows": 210,
                "physical_mean": [.25, -.3], "physical_sd": [2., 3.], "physical_zero_SD_columns": [],
                "repeatability": {"T": [1., 1.], "P": [.2, .3], "raw_ratio": [.2, .3],
                    "r": [.2, .3], "a": [.3, .4], "clip_category": ["unclipped", "unclipped"],
                    "fitting_directions": [d for d in range(8) if d != held], "fitting_direction_counts": [2]*7},
                "paired_weights": {"weights": [.8, 1.2]}} for held in range(8)]}
        candidate = copy.deepcopy(previous)
        for row in candidate["direction_fit_metadata"]:
            row["paired_weights"]["weights"] = [.9, 1.1]
        packet = {"banks": {weighted.ANCHORS["repeatability"]: previous}}
        with patch.object(Path, "read_text", return_value=json.dumps(packet)), \
                patch.object(runner, "validate_weight_session") as validate:
            runner.check_weight_reference_after_fit(Path("unused"), "example", {weighted.RECIPES[0]: candidate})
            validate.assert_called_once_with(packet, "example")
            for field in ("physical_mean", "physical_sd", "repeatability", "weight_label_maps"):
                changed = copy.deepcopy(candidate)
                if field == "weight_label_maps":
                    changed[field] = [{"seed": 1}]
                elif field == "repeatability":
                    changed["direction_fit_metadata"][3][field]["T"][0] += .01
                else:
                    changed["direction_fit_metadata"][3][field][0] += .01
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.check_weight_reference_after_fit(Path("unused"), "example", {weighted.RECIPES[0]: changed})


if __name__ == "__main__":
    unittest.main()
