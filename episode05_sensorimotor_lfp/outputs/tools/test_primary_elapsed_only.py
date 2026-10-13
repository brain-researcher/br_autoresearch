"""Independent elapsed-only baseline, preserved-HW and saved-anchor fixtures."""
import copy
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import primary_elapsed_only as elapsed
import primary_history as history
import primary_history_repeatability_weight as original
import run_primary_elapsed_only as runner
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS


def example(seed=155001, equal_counts=False):
    rng = np.random.default_rng(seed)
    counts = [3] * 8 if equal_counts else [2, 3, 4, 5, 2, 3, 4, 5]
    labels = np.repeat(np.arange(8), counts)
    # Preserve the provider reader's time-fast transpose/stack layout.
    x = np.stack([rng.normal(size=(3, 18)).T.astype(np.float64) for _ in labels])
    theta = labels * np.pi / 4
    wave = np.linspace(-1., 1., 15)
    y = rng.normal(size=(len(labels), 15, 2))
    y[:, :, 0] += 7. + 12. * np.cos(theta)[:, None] * (2. + wave) + .8 * x[:, 3:, 0]
    y[:, :, 1] += -3. + 8. * np.sin(theta)[:, None] * (1. - wave) - .6 * x[:, 3:, 1]
    ids = np.array([f"example_trial{index + 1:06d}" for index in range(len(labels))])
    return x, y, labels, ids, [2, 7, 11]


def elapsed_reference(y, labels, held):
    """Direct equal-direction mean; does not call the fitted model helper."""
    train, test = labels != held, labels == held
    template = np.mean([y[labels == direction].mean(axis=0)
                        for direction in range(8) if direction != held], axis=0)
    residual = (y[train] - template).reshape(-1, 2)
    intercept = residual.mean(axis=0)
    return template, intercept, np.broadcast_to(template + intercept, y[test].shape)


def direct_wrong_pair_sse(y, prediction, labels):
    """Enumerate complete-trajectory derangements for tiny per-direction groups."""
    result = 0.
    for direction in range(8):
        actual, fitted = y[labels == direction], prediction[labels == direction]
        values = [float(np.square(actual - fitted[list(order)]).sum())
                  for order in itertools.permutations(range(len(actual)))
                  if all(index != donor for index, donor in enumerate(order))]
        result += float(np.mean(values))
    return result


class ElapsedOnlyTest(unittest.TestCase):
    def test_elapsed_template_weights_directions_equally_and_keeps_common_intercept(self):
        _, y, labels, _, _ = example()
        held = 3
        result = elapsed.elapsed_baseline(y, labels, held)
        template, intercept, predicted = elapsed_reference(y, labels, held)
        np.testing.assert_allclose(result["template"].coefficients[0], template, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["intercept"], intercept, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(result["evaluation_baseline"], predicted, rtol=2e-14, atol=2e-14)
        self.assertEqual(result["template"].harmonics, 0)
        self.assertGreater(np.linalg.norm(intercept), .1)
        # A pooled trial mean and the original harmonic intercept are both
        # plausible but incorrect substitutes under unequal direction counts.
        self.assertGreater(np.linalg.norm(template-y[labels != held].mean(axis=0)), 1.)
        harmonic = elapsed.DirectionTime.fit(labels[labels != held]*np.pi/4,
                                             y[labels != held], 1, labels[labels != held])
        self.assertGreater(np.linalg.norm(template-harmonic.coefficients[0]), 1.)
        expected = (y[result["train"]]-result["template"].predict(
            labels[result["train"]]*np.pi/4)).reshape(-1, 2)-result["intercept"]
        np.testing.assert_array_equal(result["residual"], expected)
        np.testing.assert_allclose(result["residual"].mean(axis=0), 0., atol=2e-14)

    def test_held_direction_values_cannot_fit_time_template_or_HW_state(self):
        x, y, labels, ids, electrodes = example(155002)
        held = 2
        other_x, other_y = x.copy(order="K"), y.copy()
        other_x[labels == held] += 1e6
        other_y[labels == held] -= 1e6
        original_time = elapsed.elapsed_baseline(y, labels, held)
        altered_time = elapsed.elapsed_baseline(other_y, labels, held)
        for key in ("intercept", "fitting_baseline", "evaluation_baseline", "residual"):
            np.testing.assert_array_equal(original_time[key], altered_time[key])
        seen = []
        def no_head(fitting, evaluation, residual, check):
            seen.append((fitting.copy(), residual.copy()))
            return np.zeros((len(LAMBDAS), len(evaluation)*15, 2))
        with patch.object(elapsed, "ridge_predictions", side_effect=no_head):
            before = elapsed.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)[0]
            after = elapsed.fit_banks(other_x, other_y, labels, ids, "example", electrode_ids=electrodes)[0]
        for left, right in zip(seen[held], seen[8+held]):
            np.testing.assert_array_equal(left, right)
        a, b = before["direction_fit_metadata"][held], after["direction_fit_metadata"][held]
        for key in ("physical_mean", "physical_sd", "history_physical_mean", "history_physical_sd",
                    "history_fitting_column_mean", "repeatability", "paired_weights",
                    "elapsed_template_coefficients", "elapsed_common_residual_intercept"):
            self.assertEqual(a[key], b[key])

    def test_preserved_HW_design_with_new_response_and_addback_matches_augmented_OLS(self):
        x, y, labels, ids, electrodes = example(155003)
        seen = []
        solve = elapsed.ridge_predictions
        def record(fitting, evaluation, residual, check):
            seen.append((fitting.copy(), evaluation.copy(), residual.copy()))
            return solve(fitting, evaluation, residual, check)
        with patch.object(elapsed, "ridge_predictions", side_effect=record):
            bank, time, old_baseline, predictions = elapsed.fit_banks(
                x, y, labels, ids, "example", electrode_ids=electrodes)
        self.assertEqual(len(seen), 8)
        for held in range(8):
            split = original.prepare_fold(history.history_blocks(x), x[:, 3:18, :], y, labels, held)
            fitting = split["fitting_history"]
            flat = fitting.reshape(-1, fitting.shape[-1])
            weights, _ = original.history_trace_weights(split["repeatability"]["a"], flat.T@flat/len(flat))
            repeated = np.tile(weights, 4)
            train, test = labels != held, labels == held
            np.testing.assert_array_equal(seen[held][0], fitting*repeated)
            np.testing.assert_array_equal(seen[held][1], split["evaluation_history"]*repeated)
            state = bank["direction_fit_metadata"][held]
            for key in ("physical_mean", "physical_sd"):
                np.testing.assert_array_equal(state[key], split["weight_state"][key])
            for key in ("T", "P", "r", "a"):
                np.testing.assert_array_equal(state["repeatability"][key], split["repeatability"][key])
            expected_time = elapsed.elapsed_baseline(y, labels, held)
            np.testing.assert_array_equal(seen[held][2], expected_time["residual"])
            self.assertGreater(np.linalg.norm(seen[held][2]-split["residual"]), 1.)
            np.testing.assert_array_equal(old_baseline[test],
                split["template"].predict(labels[test]*np.pi/4)+split["intercept"])
            np.testing.assert_array_equal(time[test], expected_time["evaluation_baseline"])
            z = seen[held][0].reshape(-1, 12)
            ev = seen[held][1].reshape(-1, 12)
            template, intercept, expected_base = elapsed_reference(y, labels, held)
            response = (y[train]-template).reshape(-1, 2)-intercept
            for index, penalty in enumerate(LAMBDAS):
                # Independent rectangular least squares, not the eigensolver.
                design = np.vstack((z, np.sqrt(len(z)*penalty)*np.eye(12)))
                target = np.vstack((response, np.zeros((12, 2))))
                coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
                expected = expected_base+(ev@coefficient).reshape(y[test].shape)
                np.testing.assert_allclose(predictions[index, test], expected, rtol=2e-11, atol=2e-11)

    def test_native_joint_scores_and_exact_whole_trajectory_wrong_pair_mean(self):
        x, y, labels, ids, electrodes = example(155004, equal_counts=True)
        bank, time, _, predictions = elapsed.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
        self.assertEqual(bank["denominator"], denominator)
        self.assertEqual(bank["baseline_SSE"], float(np.square(y-time).sum()))
        for index, row in enumerate(bank["lambda_rows"]):
            sse = float(np.square(y-predictions[index]).sum())
            wrong = direct_wrong_pair_sse(y, predictions[index], labels)
            self.assertEqual(row["candidate_SSE"], sse)
            self.assertAlmostEqual(row["wrong_pair_SSE"], wrong, places=8)
            self.assertAlmostEqual(row["delta_R2"], (bank["baseline_SSE"]-sse)/denominator, places=13)
            self.assertAlmostEqual(row["identity_increment"], (wrong-sse)/denominator, places=13)
        self.assertAlmostEqual(direct_wrong_pair_sse(y, time, labels), bank["baseline_SSE"], places=8)

    def test_full_model_contrast_uses_common_loss_not_mismatched_baseline_increments(self):
        candidate = {"candidate_SSE": 60., "delta_R2": .2, "identity_increment": .04}
        anchor = {"candidate_SSE": 40., "delta_R2": .1, "identity_increment": .03}
        contrast = elapsed.full_model_contrast(candidate, anchor, 100.)
        self.assertAlmostEqual(contrast["delta_R2"], -.2)
        self.assertAlmostEqual(contrast["identity_increment"], .01)
        self.assertNotEqual(contrast["delta_R2"], candidate["delta_R2"]-anchor["delta_R2"])
        for denominator in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                elapsed.full_model_contrast(candidate, anchor, denominator)

    def test_exact_time_only_and_original_baseline_duplicate_flags_are_structural(self):
        x, y, labels, ids, electrodes = example(155005)
        bank, time, _, predictions = elapsed.fit_banks(
            np.zeros_like(x), y, labels, ids, "example", electrode_ids=electrodes)
        self.assertTrue(bank["structural_time_only_duplicate"])
        self.assertFalse(bank["structural_original_HW_duplicate"])
        for prediction in predictions:
            np.testing.assert_array_equal(prediction, time)
        self.assertTrue(all(row["exact_zero_weighted_design_response_cross_product"]
                            for row in bank["direction_fit_metadata"]))
        # Zero Y isolates literal T==B and zero cross-products. Scoring such a
        # target is undefined, so stub only scoring; all actual fit logic runs.
        with patch.object(elapsed, "score_bank", return_value={}):
            zero, _, _, _ = elapsed.fit_banks(x, np.zeros_like(y), labels, ids,
                                             "example", electrode_ids=electrodes)
        self.assertTrue(zero["structural_original_HW_duplicate"])
        self.assertTrue(zero["structural_time_only_duplicate"])

    def test_outer_exclusion_saved_penalties_direct_anchor_algebra_and_duplicate_accounting(self):
        def score(penalty, sse, baseline):
            return {"lambda": penalty, "candidate_R2": 1.-sse/10., "candidate_SSE": sse,
                    "wrong_pair_SSE": sse+.1, "delta_R2": (baseline-sse)/10., "identity_increment": .01}
        anchors, sessions = {}, {}
        for session in SESSIONS:
            anchors[session] = {"recipe": elapsed.ANCHOR, "lambdas": list(LAMBDAS),
                "denominator": 10., "baseline_SSE": 5., "baseline_R2": .5, "direction_counts": [3]*8,
                "lambda_rows": [score(penalty, 4.+.1*index, 5.) for index, penalty in enumerate(LAMBDAS)]}
            sessions[session] = {"recipe": elapsed.RECIPE, "lambdas": list(LAMBDAS),
                "denominator": 10., "baseline_SSE": 8., "baseline_R2": .2, "direction_counts": [3]*8,
                "original_baseline_SSE": 5., "original_HW_saved_prediction_rows": copy.deepcopy(anchors[session]["lambda_rows"]),
                "lambda_rows": [score(penalty, 6., 8.) for penalty in LAMBDAS],
                "structural_original_HW_duplicate": False, "structural_time_only_duplicate": False}
        anchor_panel = {"recipe": elapsed.ANCHOR, "sessions": [
            {"session_id": session, "animal": ANIMALS[session], "lambda_index": 0,
             **anchors[session]["lambda_rows"][0]} for session in SESSIONS]}
        held = SESSIONS[0]
        sessions[held]["lambda_rows"] = [score(penalty, 1e6*(index+1), 8.) for index, penalty in enumerate(LAMBDAS)]
        for duplicate_HW, duplicate_time, charge in ((False, False, 1), (True, False, 0),
                                                     (False, True, 0), (True, True, 0)):
            for bank in sessions.values():
                bank["structural_original_HW_duplicate"] = duplicate_HW
                bank["structural_time_only_duplicate"] = duplicate_time
            result = elapsed.summarize_banks(sessions, anchors, anchor_panel)
            self.assertEqual(result["scientific_trial_count_increment"], charge)
            self.assertEqual(result["alternative_explanation_stress_count_increment"], charge)
            self.assertEqual(result["null_control_trial_count_increment"], 0)
            self.assertTrue(result["sensitivity_only"])
            self.assertFalse(result["updates_primary_incumbent"])
            self.assertFalse(result["promotion_authorized"])
            selected = next(row for row in result["elapsed_sensitivity"]["sessions"] if row["session_id"] == held)
            self.assertEqual(selected["lambda_index"], 4)  # ties use stronger λ; own outcomes excluded
            self.assertEqual(set(selected["outer_training_sessions"]), set(SESSIONS)-{held})
            for name, anchor_index in (("HW_independently_selected", 0), ("HW_ET_HW_lambda_matched", 4)):
                contrast = next(row for row in result["full_model_anchor_contrasts"][name]["sessions"]
                                if row["session_id"] == held)
                self.assertEqual(contrast["HW_lambda"], LAMBDAS[anchor_index])
                self.assertEqual(contrast["ET_HW_lambda"], LAMBDAS[4])
                self.assertEqual(contrast["delta_R2"],
                    (anchors[held]["lambda_rows"][anchor_index]["candidate_SSE"]-selected["candidate_SSE"])/10.)
            old = result["original_HW_versus_original_B_replayed"]["primary_increment"]
            b_t = result["original_B_versus_elapsed_baseline"]["primary_increment"]
            hw_t = result["original_HW_versus_elapsed_baseline"]["primary_increment"]
            self.assertAlmostEqual(old+b_t, hw_t, places=13)

    def test_saved_anchor_denominator_and_selected_rows_cannot_silently_change(self):
        # This test keeps scoring consistent and changes only one frozen-anchor
        # field at a time, targeting the new cross-baseline comparison boundary.
        def bank(recipe, baseline):
            rows = [{"lambda": penalty, "candidate_R2": .6, "candidate_SSE": 4., "wrong_pair_SSE": 4.1,
                     "delta_R2": (baseline-4.)/10., "identity_increment": .01} for penalty in LAMBDAS]
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "denominator": 10.,
                    "baseline_SSE": baseline, "baseline_R2": 1.-baseline/10.,
                    "direction_counts": [3]*8, "lambda_rows": rows}
        anchors = {session: bank(elapsed.ANCHOR, 5.) for session in SESSIONS}
        sessions = {session: {**bank(elapsed.RECIPE, 8.), "original_baseline_SSE": 5.,
            "original_HW_saved_prediction_rows": copy.deepcopy(anchors[session]["lambda_rows"]),
            "structural_original_HW_duplicate": False, "structural_time_only_duplicate": False}
            for session in SESSIONS}
        panel = {"recipe": elapsed.ANCHOR, "sessions": [{"session_id": session,
            "animal": ANIMALS[session], "lambda_index": 0, **anchors[session]["lambda_rows"][0]}
            for session in SESSIONS]}
        for field, value in (("denominator", 11.), ("original_baseline_SSE", 6.),
                             ("original_HW_saved_prediction_rows", [])):
            altered = copy.deepcopy(sessions)
            altered[SESSIONS[0]][field] = value
            with self.assertRaises(ValueError):
                elapsed.summarize_banks(altered, anchors, panel)
        altered_panel = copy.deepcopy(panel)
        altered_panel["sessions"][0]["candidate_SSE"] = 3.
        with self.assertRaises(ValueError):
            elapsed.summarize_banks(sessions, anchors, altered_panel)

    def test_saved_array_replay_uses_original_B_and_rejects_changed_neural_state(self):
        x, y, labels, ids, electrodes = example(155006)
        bank, time, baseline, _ = elapsed.fit_banks(x, y, labels, ids, "example", electrode_ids=electrodes)
        # A fixed synthetic anchor isolates replay semantics; it is never an
        # empirical score. Existing full-HW validation is tested separately.
        predictions = np.ascontiguousarray(np.stack([baseline+(index+1)*.01 for index in range(5)]))
        anchor = score_bank(y, labels, baseline, predictions, elapsed.ANCHOR)
        anchor.update(physical_electrode_ids=electrodes,
                      direction_fit_metadata=copy.deepcopy(bank["direction_fit_metadata"]))
        source = {"common_LMP_electrodes": electrodes, "complete_trials": len(y)}
        packet = {"source": source, "banks": {elapsed.ANCHOR: anchor}}
        saved = {"actual_xy": y.copy(), "baseline_xy": baseline.copy(), "directions": labels.copy(),
                 "trial_ids": ids.copy(), "candidate_xy_by_lambda": predictions.copy()}
        loaded = MagicMock()
        loaded.__enter__.return_value = saved
        with patch.object(Path, "read_text", return_value=json.dumps(packet)), \
                patch.object(runner, "validate_hw_session"), patch.object(runner.np, "load", return_value=loaded):
            replayed = copy.deepcopy(bank)
            runner.check_anchor_after_fit(Path("unused"), "example", source, y, labels, ids, baseline, replayed)
            self.assertEqual(replayed["original_HW_saved_prediction_rows"], anchor["lambda_rows"])
            self.assertEqual(replayed["original_baseline_SSE"], anchor["baseline_SSE"])
            with self.assertRaisesRegex(ValueError, "baseline_xy"):
                runner.check_anchor_after_fit(Path("unused"), "example", source, y, labels, ids, time, copy.deepcopy(bank))
            changed = copy.deepcopy(bank)
            changed["direction_fit_metadata"][0]["physical_mean"][0] += 1.
            with self.assertRaisesRegex(ValueError, "physical_mean"):
                runner.check_anchor_after_fit(Path("unused"), "example", source, y, labels, ids, baseline, changed)
            saved["trial_ids"][0] = "example_trial999999"
            with self.assertRaisesRegex(ValueError, "trial_ids"):
                runner.check_anchor_after_fit(Path("unused"), "example", source, y, labels, ids, baseline, copy.deepcopy(bank))


if __name__ == "__main__":
    unittest.main()
