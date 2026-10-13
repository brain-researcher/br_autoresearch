"""Independent scientific checks for supervised input subspaces and basis nulls."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import primary_covariance_svd as cs
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def example(seed=59001, electrodes=6):
    rng = np.random.default_rng(seed)
    labels, ids = np.repeat(np.arange(8), 2), np.arange(16)*17+100
    x = rng.normal(size=(16, 15, electrodes))
    y = rng.normal(size=(16, 15, 2))+.4*x[:, :, :2]
    return x, y, labels, ids


class CovarianceSVDTest(unittest.TestCase):
    def test_native_source_tokens_preserve_identity_and_numeric_permutations(self):
        x, y, labels, ids = example(59010)
        session = "Chewie_20150313"
        tokens = np.array([f"{session}_trial{value:06d}" for value in ids])
        np.testing.assert_array_equal(cs.numeric_trial_ids(tokens, session), ids)
        train = labels != 7
        native_map, records = cs.trial_map(tokens[train], labels[train], session, 7, 3)
        integer_map, _ = cs.trial_map(ids[train], labels[train], session, 7, 3)
        np.testing.assert_array_equal(native_map, integer_map)
        for record in records:
            self.assertTrue(all(value.startswith(session + "_trial")
                                for value in record["recipient_trial_ids"] + record["donor_trial_ids"]))
        _, baseline, predictions = cs.fit_banks(x, y, labels, tokens, session)
        _, reference, integer_predictions = cs.fit_banks(x, y, labels, ids, session)
        np.testing.assert_array_equal(baseline, reference)
        for recipe in cs.RECIPES:
            np.testing.assert_array_equal(predictions[recipe], integer_predictions[recipe])
        with self.assertRaises(ValueError):
            cs.numeric_trial_ids(tokens, "Mihili_20140217")
        with self.assertRaises(ValueError):
            cs.numeric_trial_ids([session + "_trial1"], session)

    def test_direct_native_cross_covariance_and_svd_projector(self):
        x = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 2., 0.], [0., -2., 0.]])
        u = np.array([[3., 0.], [-3., 0.], [0., 5.], [0., -5.]])
        basis, meta = cs.covariance_basis(x, u)
        np.testing.assert_allclose(meta["cross_covariance"], [[1.5, 0.], [0., 5.], [0., 0.]], atol=1e-14)
        np.testing.assert_allclose(meta["singular_values"], [5., 1.5], atol=1e-14)
        np.testing.assert_allclose(basis@basis.T, np.diag([1., 1., 0.]), atol=1e-14)
        self.assertEqual(meta["singular_value_tolerance"], 64*np.finfo(float).eps*3*5)
        self.assertEqual(meta["retained_components"], 2)
        self.assertFalse(meta["native_XY_scaled"])
        self.assertFalse(meta["whitened"])

    def test_zero_and_rank_one_have_no_arbitrary_null_space_or_padding(self):
        rng = np.random.default_rng(59002)
        x = rng.normal(size=(4, 15, 5))
        basis, meta = cs.covariance_basis(x, np.zeros((60, 2)))
        self.assertEqual(basis.shape, (5, 0))
        self.assertEqual(meta["retained_components"], 0)
        train = np.array([True, True, True, False])
        h = cs.project_arm(x, basis, train)
        self.assertEqual(h.shape, (4, 15, 0))
        covariance = cs.projected_covariance(h[train])
        self.assertEqual(covariance["eigenvalues"], [])
        self.assertEqual(covariance["trace"], 0)
        self.assertTrue(all(row["effective_df"] == 0 for row in covariance["ridge_df_by_lambda"]))
        pred = cs.ridge_predictions(h[train], h[~train], rng.normal(size=(45, 2)))
        np.testing.assert_array_equal(pred, np.zeros((len(LAMBDAS), 15, 2)))
        signal = x[:, :, 0].reshape(-1)
        basis, meta = cs.covariance_basis(x, np.column_stack((signal, 2*signal)))
        self.assertEqual(meta["retained_components"], 1)
        self.assertEqual(basis.shape, (5, 1))

    def test_unwhitened_ridge_is_invariant_to_basis_rotation(self):
        rng = np.random.default_rng(59003)
        x = rng.normal(size=(8, 15, 5))*np.arange(1, 6)
        train = np.arange(8) < 6
        z, state = cs.prepare_split(x, train)
        u = rng.normal(size=(90, 2))
        u -= u.mean(axis=0)
        basis, _ = cs.covariance_basis(z[train], u)
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        h, rotated = cs.project_arm(z, basis, train), cs.project_arm(z, basis@rotation, train)
        pred = cs.ridge_predictions(h[train], h[~train], u)
        other = cs.ridge_predictions(rotated[train], rotated[~train], u)
        np.testing.assert_allclose(pred, other, atol=2e-12, rtol=2e-11)
        changed = x.copy()
        changed[~train] += 1e7
        _, other_state = cs.prepare_split(changed, train)
        np.testing.assert_array_equal(state["physical_mean"], other_state["physical_mean"])
        np.testing.assert_array_equal(state["physical_sd"], other_state["physical_sd"])

    def test_original_id_maps_are_order_independent_and_accept_identity(self):
        labels = np.repeat(np.arange(7), 3)
        ids = np.random.default_rng(59004).permutation(np.arange(21)*19+100)
        mapping, records = cs.trial_map(ids, labels, "example", 7, 2)
        np.testing.assert_array_equal(labels[mapping], labels)
        reordered = np.arange(21)[::-1]
        other, other_records = cs.trial_map(ids[reordered], labels[reordered], "example", 7, 2)
        self.assertEqual(dict(zip(ids, ids[mapping])), dict(zip(ids[reordered], ids[reordered][other])))
        self.assertEqual(records, other_records)
        draws = []
        class Identity:
            def seed(self, literal, version): draws.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(cs.random, "Random", side_effect=Identity):
            identity, _ = cs.trial_map(ids, labels, "example", 7, 2)
        np.testing.assert_array_equal(identity, np.arange(21))
        self.assertEqual(draws, [(f"ep05-lmp-covariance-svd-basis-label-v1|20261004|example|7|2|{d}", 2)
                                 for d in range(7)])
        with self.assertRaises(ValueError):
            cs.trial_map(np.append(ids, 9999), np.append(labels, 7), "example", 7, 2)

    def test_only_basis_targets_are_permuted_and_every_head_uses_true_U(self):
        x, y, labels, ids = example(59005)
        basis_residuals, head_residuals = [], []
        original_basis, original_ridge = cs.covariance_basis, cs.ridge_predictions
        def basis(fitting, residual):
            basis_residuals.append(np.array(residual).reshape(-1, 2).copy())
            return original_basis(fitting, residual)
        def ridge(fitting, evaluation, residual, check):
            head_residuals.append(np.array(residual).copy())
            return original_ridge(fitting, evaluation, residual, check)
        with patch.object(cs, "covariance_basis", side_effect=basis), patch.object(cs, "ridge_predictions", side_effect=ridge):
            banks, _, _ = cs.fit_banks(x, y, labels, ids, "example")
        self.assertEqual(len(basis_residuals), 72)
        self.assertEqual(len(head_residuals), 72)
        for held in range(8):
            true = basis_residuals[9*held]
            for arm in range(9):
                np.testing.assert_array_equal(head_residuals[9*held+arm], true)
            for replicate in range(8):
                train = labels != held
                mapping, _ = cs.trial_map(ids[train], labels[train], "example", held, replicate)
                expected = true.reshape(-1, 15, 2)[mapping].reshape(-1, 2)
                np.testing.assert_array_equal(basis_residuals[9*held+1+replicate], expected)
        metadata = banks[cs.RECIPES[0]]["direction_fit_metadata"]
        self.assertTrue(any(abs(control["trace"]-fold["paired_projected_covariance"]["trace"]) > 1e-8
            for fold in metadata for control in fold["control_projected_covariance_by_replicate"]))

    def test_control_averages_eight_correct_and_wrong_SSE_not_predictions(self):
        x, y, labels, ids = example(59006)
        banks, _, predictions = cs.fit_banks(x, y, labels, ids, "example")
        controls = predictions[cs.RECIPES[1]]
        for index in range(len(LAMBDAS)):
            errors = ((y[None]-controls[:, index])**2).sum(axis=(1, 2, 3))
            row = banks[cs.RECIPES[1]]["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_SSE"], float(errors.mean()), places=9)
            self.assertAlmostEqual(row["wrong_pair_SSE"], float(np.mean([
                wrong_pair_sse(y, controls[seed, index], labels) for seed in range(8)])), places=9)
        ensemble_sse = float(((y-controls[:, 0].mean(axis=0))**2).sum())
        self.assertGreater(banks[cs.RECIPES[1]]["lambda_rows"][0]["candidate_SSE"]-ensemble_sse, 1e-6)

    def test_held_features_and_targets_cannot_change_own_fits_or_scaling(self):
        x, y, labels, ids = example(59007)
        banks, baseline, predictions = cs.fit_banks(x, y, labels, ids, "example")
        changed_x, changed_y = x.copy(), y.copy()
        held = 2
        changed_x[labels == held] += 1e5
        changed_y[labels == held] -= 1e6
        changed, other_baseline, _ = cs.fit_banks(changed_x, changed_y, labels, ids, "example")
        self.assertEqual(banks[cs.RECIPES[0]]["direction_fit_metadata"][held],
                         changed[cs.RECIPES[0]]["direction_fit_metadata"][held])
        np.testing.assert_array_equal(baseline[labels == held], other_baseline[labels == held])
        _, same_x_baseline, same_x = cs.fit_banks(x, changed_y, labels, ids, "example")
        np.testing.assert_array_equal(same_x_baseline[labels == held], baseline[labels == held])
        np.testing.assert_array_equal(same_x[cs.RECIPES[0]][:, labels == held], predictions[cs.RECIPES[0]][:, labels == held])
        np.testing.assert_array_equal(same_x[cs.RECIPES[1]][:, :, labels == held], predictions[cs.RECIPES[1]][:, :, labels == held])

    def test_full_input_span_replays_paired_and_all_true_target_control_heads(self):
        x, y, labels, ids = example(59008, electrodes=2)
        banks, _, predictions = cs.fit_banks(x, y, labels, ids, "full")
        self.assertTrue(banks[cs.RECIPES[0]]["all_arms_all_folds_full_span"])
        for replicate in range(8):
            np.testing.assert_allclose(predictions[cs.RECIPES[1]][replicate], predictions[cs.RECIPES[0]],
                                       atol=3e-12, rtol=3e-11)

    def test_other_sessions_select_shared_lambda_and_duplicates_add_no_depth(self):
        def bank(recipe, candidate):
            rows = []
            for index, penalty in enumerate(LAMBDAS):
                gain = .01*index if candidate else .04-.01*index
                rows.append({"lambda": penalty, "delta_R2": gain, "identity_increment": .01,
                    "candidate_SSE": 5-10*gain, "wrong_pair_SSE": 5.1-10*gain, "candidate_R2": .5+gain})
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                    "baseline_SSE": 5., "denominator": 10., "direction_counts": [2]*8,
                    "lambda_rows": rows, "all_arms_all_folds_full_span": False,
                    "all_arms_all_folds_zero_rank": False}
        sessions = {session: {cs.RECIPES[0]: bank(cs.RECIPES[0], True),
                    cs.RECIPES[1]: cs.average_control_banks([bank(cs.RECIPES[1], False) for _ in range(8)])}
                    for session in SESSIONS}
        anchors = {session: bank(cs.ANCHOR, False) for session in SESSIONS}
        panel = {"recipe": cs.ANCHOR, "sessions": [{"session_id": session, "animal": ANIMALS[session],
                 **anchors[session]["lambda_rows"][0]} for session in SESSIONS]}
        result = cs.summarize_banks(sessions, anchors, panel)
        self.assertEqual(result["scientific_trial_count_increment"], 2)
        self.assertAlmostEqual(result["anchor_contrasts"]["covariance_svd_lambda_matched"]["primary_increment"], .04)
        for held in SESSIONS:
            changed = copy.deepcopy(sessions)
            for index, row in enumerate(changed[held][cs.RECIPES[0]]["lambda_rows"]):
                row["delta_R2"] = 1e7*(5-index)
            result = cs.summarize_banks(changed, anchors, panel)
            for scored in result["panels"]:
                row = next(row for row in scored["sessions"] if row["session_id"] == held)
                self.assertEqual(row["lambda_index"], 4)
                self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
        for flag, summary_flag in (("all_arms_all_folds_full_span", "structural_full_span_duplicate"),
                                   ("all_arms_all_folds_zero_rank", "structural_baseline_duplicate")):
            changed = copy.deepcopy(sessions)
            for banks in changed.values(): banks[cs.RECIPES[0]][flag] = True
            result = cs.summarize_banks(changed, anchors, panel)
            self.assertTrue(result[summary_flag])
            self.assertEqual(result["scientific_trial_count_increment"], 0)


if __name__ == "__main__":
    unittest.main()
