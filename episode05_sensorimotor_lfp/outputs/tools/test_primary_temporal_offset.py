"""Scientific fixtures for a frozen neural-only within-trial timing control."""
import copy
import unittest

import numpy as np

from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal_offset import (
    ANCHOR, OFFSETS, RECIPE, frozen_selection, offset_prediction, prediction_scores,
    replay_selected_anchor, score_session, summarize_sessions,
)


def fixture(session=SESSIONS[0]):
    rng = np.random.default_rng(20261004)
    labels = np.repeat(np.arange(8), 2)
    y = rng.normal(size=(16, 15, 2))
    baseline = np.broadcast_to(np.arange(30).reshape(1, 15, 2)/40, y.shape).copy()
    prediction = baseline+.45*y+.15*rng.normal(size=y.shape)
    denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
    baseline_sse = float(np.square(y-baseline).sum())
    score = prediction_scores(y, prediction, labels, baseline_sse, denominator)
    bank = {"recipe": ANCHOR, "lambdas": list(LAMBDAS), "direction_counts": [2]*8,
        "baseline_SSE": baseline_sse, "baseline_R2": 1-baseline_sse/denominator,
        "denominator": denominator, "lambda_rows": [{"lambda": p, **score} for p in LAMBDAS]}
    selected = {"session_id": session, "animal": ANIMALS[session], "lambda_index": 3,
        "lambda": LAMBDAS[3], "outer_training_sessions": [s for s in SESSIONS if s != session],
        "inner_animal_balanced_scores": [0., .01, .02, .03, .01], **score}
    return y, baseline, prediction, labels, bank, selected


class TemporalOffsetTest(unittest.TestCase):
    def test_exact_time_map_preserves_trials_xy_vectors_and_zero_offset(self):
        # Risk: moving trials or one coordinate separately would become an
        # identity/coordinate control instead of a within-trial timing test.
        baseline = np.arange(90).reshape(3, 15, 2)*10.
        residual = np.arange(90).reshape(3, 15, 2).astype(float)
        prediction = baseline+residual
        before = prediction.copy()
        for k in OFFSETS:
            actual = offset_prediction(prediction, baseline, k)
            expected = np.empty_like(actual)
            for i in range(3):
                for t in range(15):
                    expected[i, t] = baseline[i, t]+residual[i, (t+k) % 15]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(sorted((np.arange(15)+k) % 15), list(range(15)))
        np.testing.assert_array_equal(offset_prediction(prediction, baseline, 0), prediction)
        np.testing.assert_array_equal(prediction, before)

    def test_baseline_and_trial_constant_neural_correction_survive_every_offset(self):
        # Risk: rotating the direction/time template or shared intercept could
        # create a false timing effect when the neural branch has none.
        baseline = np.arange(120).reshape(4, 15, 2).astype(float)
        correction = np.array([[3., 9.], [-4., 7.], [12., -8.], [0., 0.]])[:, None, :]
        for k in OFFSETS:
            np.testing.assert_array_equal(offset_prediction(baseline, baseline, k), baseline)
            np.testing.assert_array_equal(offset_prediction(baseline+correction, baseline, k), baseline+correction)

    def test_neural_offset_equals_frozen_stationary_feature_map(self):
        # Risk: manipulating saved predictions might fail to represent the
        # claimed fixed-model evaluation-feature offset counterfactual.
        rng = np.random.default_rng(610041)
        x = rng.normal(size=(4, 15, 3))
        mean, sd = np.array([1., -2., 3.]), np.array([2., 4., 5.])
        beta = np.array([[.2, -.3], [1.2, .7], [-.8, .1]])
        baseline = rng.normal(size=(4, 15, 2))+np.array([7., -4.])
        prediction = baseline+((x-mean)/sd)@beta
        for k in OFFSETS:
            expected = baseline+((x[:, (np.arange(15)+k) % 15]-mean)/sd)@beta
            np.testing.assert_allclose(offset_prediction(prediction, baseline, k), expected, atol=3e-15)
        self.assertGreater(np.max(np.abs(offset_prediction(prediction, baseline, 7)
                                        -prediction[:, (np.arange(15)+7) % 15])), 1.)

    def test_original_lambda_stays_fixed_and_changed_reference_is_rejected(self):
        # Risk: control outcomes or evaluated-session performance could choose
        # a different lambda while the report claims a frozen decoder.
        _, _, _, _, bank, selected = fixture()
        self.assertEqual(frozen_selection(SESSIONS[0], selected, bank), 3)
        altered = copy.deepcopy(bank)
        for index, row in enumerate(altered["lambda_rows"]):
            if index != 3:
                row["delta_R2"] = 1e6*(index+1)
        self.assertEqual(frozen_selection(SESSIONS[0], selected, altered), 3)
        changed = copy.deepcopy(selected)
        changed["outer_training_sessions"].append(SESSIONS[0])
        with self.assertRaises(ValueError):
            frozen_selection(SESSIONS[0], changed, bank)
        changed = copy.deepcopy(selected)
        changed["candidate_SSE"] += 1
        with self.assertRaises(ValueError):
            frozen_selection(SESSIONS[0], changed, bank)

    def test_selected_unshifted_sse_replay_precedes_offset_scoring(self):
        # Risk: stale/mismatched predictions, targets or score support could
        # appear as a timing loss without reproducing the original decoder.
        y, baseline, prediction, labels, bank, selected = fixture()
        replay = replay_selected_anchor(y, baseline, prediction, labels, bank, selected)
        self.assertEqual(replay["candidate_SSE"], selected["candidate_SSE"])
        altered_target = y.copy()
        altered_target[0, 0, 0] += 2
        with self.assertRaises(ArithmeticError):
            score_session(altered_target, baseline, prediction, labels, bank, selected)
        altered = copy.deepcopy(selected)
        altered["wrong_pair_SSE"] += 1
        with self.assertRaises(ArithmeticError):
            score_session(y, baseline, prediction, labels, bank, altered)

    def test_mean_offset_correct_and_wrong_sse_not_ensemble(self):
        # Risk: an ensemble of cyclic predictions suppresses variation; it is
        # not the declared exact mean risk across misaligned trajectories.
        y, baseline, prediction, labels, bank, selected = fixture()
        old_y, old_baseline = y.copy(), baseline.copy()
        result = score_session(y, baseline, prediction, labels, bank, selected)
        correct, wrong, predictions = [], [], []
        for k in OFFSETS:
            shifted = baseline+(prediction-baseline)[:, (np.arange(15)+k) % 15]
            predictions.append(shifted)
            correct.append(float(np.square(y-shifted).sum()))
            # Exactly two original trials per direction: its only whole-trial
            # derangement swaps these trajectories, an independent oracle.
            total = 0.
            for direction in range(8):
                indices = np.flatnonzero(labels == direction)
                total += float(np.square(y[indices]-shifted[indices[::-1]]).sum())
            wrong.append(total)
        self.assertAlmostEqual(result["candidate_SSE"], float(np.mean(correct)), places=10)
        self.assertAlmostEqual(result["wrong_pair_SSE"], float(np.mean(wrong)), places=10)
        self.assertEqual(result["denominator"], bank["denominator"])
        self.assertEqual(result["baseline_SSE"], bank["baseline_SSE"])
        ensemble_sse = float(np.square(y-np.mean(predictions, axis=0)).sum())
        self.assertGreater(result["candidate_SSE"]-ensemble_sse, 1.)
        np.testing.assert_array_equal(y, old_y)
        np.testing.assert_array_equal(baseline, old_baseline)

    def test_complete_panel_uses_equal_animal_weights_and_one_control_charge(self):
        # Risk: pooling 10 Chewie with6 Mihili or treating14 offsets as animals
        # would distort the scientific unit and falsely expand search depth.
        y, baseline, prediction, labels, bank, selected = fixture()
        prototype = score_session(y, baseline, prediction, labels, bank, selected)
        sessions = {}
        for session in SESSIONS:
            row = copy.deepcopy(prototype)
            animal = ANIMALS[session]
            row.update(session_id=session, animal=animal, delta_R2=1. if animal == "Mihili" else 3.,
                       identity_increment=2. if animal == "Mihili" else 4.)
            row["original_selected_anchor"].update(session_id=session, animal=animal,
                delta_R2=row["delta_R2"]+.2, identity_increment=row["identity_increment"]+.1)
            sessions[session] = row
        result = summarize_sessions(sessions)
        self.assertEqual(result["control"]["primary_increment"], 2.)
        self.assertEqual(result["control"]["identity_increment"], 3.)
        self.assertEqual(result["scientific_trial_count_increment"], 1)
        self.assertAlmostEqual(result["aligned_minus_offset"]["primary_increment"], .2)
        self.assertTrue(result["aligned_minus_offset"]["all_directional_timing_screens_pass"])
        self.assertEqual(result["new_fits"], 0)
        incomplete = dict(sessions)
        incomplete.pop(SESSIONS[-1])
        with self.assertRaises(ValueError):
            summarize_sessions(incomplete)
        incomplete = copy.deepcopy(sessions)
        incomplete[SESSIONS[0]]["offset_rows"].pop()
        with self.assertRaises(ValueError):
            summarize_sessions(incomplete)


if __name__ == "__main__":
    unittest.main()
