"""Scientific fixtures for training PCA and frozen loading-label controls."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from primary_pca import (
    ANCHOR, RECIPES, REPLICATES, average_control_banks, fit_banks, loading_map,
    pca_basis, prepare_split, project_arm, projected_covariance,
    ridge_predictions, summarize_banks,
)
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


class PCATest(unittest.TestCase):
    def test_scaling_basis_and_projection_depend_only_on_fitting_rows(self):
        # Evaluation features must not determine physical scales, PCA rank,
        # loading subspace or the arm's fitting centering.
        rng = np.random.default_rng(50901)
        x = rng.normal(size=(8, 15, 4))
        x[:, :, 3] = 7.
        train = np.arange(8) < 6
        z, state = prepare_split(x, train)
        basis, meta = pca_basis(z[train])
        altered = x.copy()
        altered[~train] = 1e8
        other, other_state = prepare_split(altered, train)
        other_basis, other_meta = pca_basis(other[train])
        for key in state:
            np.testing.assert_array_equal(state[key], other_state[key])
        np.testing.assert_array_equal(z[train], other[train])
        np.testing.assert_array_equal(basis@basis.T, other_basis@other_basis.T)
        self.assertEqual(meta, other_meta)
        np.testing.assert_array_equal(z[:, :, 3], np.zeros((8, 15)))
        projected, changed = project_arm(z, basis, train), project_arm(other, basis, train)
        np.testing.assert_array_equal(projected[train], changed[train])
        expected = z@basis
        expected -= expected[train].mean(axis=(0, 1))
        np.testing.assert_array_equal(projected, expected)

    def test_exact_ninety_percent_cutoff_includes_whole_exact_tied_space(self):
        # Controlled eigenspectra isolate the exact threshold and exact-tie
        # convention from roundoff in constructing a synthetic covariance.
        fitting = np.ones((2, 15, 3))
        with patch("primary_pca.np.linalg.eigh", return_value=(np.array([0., 1., 9.]), np.eye(3))):
            basis, meta = pca_basis(fitting)
        self.assertEqual(basis.shape, (3, 1))
        self.assertEqual(meta["retained_components"], 1)
        self.assertEqual(meta["retained_variance_fraction"], .9)
        with patch("primary_pca.np.linalg.eigh", return_value=(np.array([1., 1., 8.]), np.eye(3))):
            basis, meta = pca_basis(fitting)
        self.assertEqual(basis.shape, (3, 3))
        self.assertEqual(meta["retained_variance_fraction"], 1.)
        self.assertTrue(meta["full_span"])
        self.assertFalse(meta["whitened"])

    def test_psd_roundoff_is_clipped_but_material_negative_and_zero_total_stop(self):
        # A numerical PSD allowance must not hide invalid covariance or create
        # a fallback representation from an all-constant candidate.
        tolerance = 64*np.finfo(np.float64).eps*2
        fitting = np.ones((2, 15, 2))
        with patch("primary_pca.np.linalg.eigh", return_value=(np.array([-tolerance/2, 1.]), np.eye(2))):
            _, meta = pca_basis(fitting)
        self.assertEqual(meta["eigenvalues"], [1., 0.])
        self.assertEqual(meta["covariance_tolerance"], tolerance)
        with patch("primary_pca.np.linalg.eigh", return_value=(np.array([-2*tolerance, 1.]), np.eye(2))):
            with self.assertRaises(ArithmeticError):
                pca_basis(fitting)
        with self.assertRaises(ValueError):
            pca_basis(np.zeros((2, 15, 2)))

    def test_full_span_rotation_and_loading_permutation_equal_physical_ridge(self):
        # A full unwhitened orthogonal basis cannot create a new model or a
        # meaningful label control at the same isotropic ridge penalty.
        rng = np.random.default_rng(50902)
        x = rng.normal(size=(7, 15, 3))
        train = np.arange(7) < 5
        z, _ = prepare_split(x, train)
        basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        residual = rng.normal(size=(75, 2))
        physical = project_arm(z, np.eye(3), train)
        phi, evaluation = physical[train].reshape(-1, 3), physical[~train].reshape(-1, 3)
        expected = np.stack([evaluation@np.linalg.solve(phi.T@phi+len(phi)*p*np.eye(3), phi.T@residual)
                             for p in LAMBDAS])
        for mapping in (None, np.array([2, 0, 1])):
            arm = project_arm(z, basis, train, mapping)
            actual = ridge_predictions(arm[train], arm[~train], residual)
            np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=1e-11)

    def test_truncated_permutation_changes_gram_and_zero_energy_control_is_valid(self):
        # Nominal K=1 can hide an effective rank/energy change from one to zero;
        # the declared control must remain an intercept-only ridge fit.
        rng = np.random.default_rng(50903)
        x = np.stack((rng.normal(size=(6, 15)), np.zeros((6, 15))), axis=2)
        train = np.arange(6) < 4
        z, _ = prepare_split(x, train)
        basis, meta = pca_basis(z[train])
        self.assertEqual(meta["retained_components"], 1)
        paired = project_arm(z, basis, train)
        control = project_arm(z, basis, train, np.array([1, 0]))
        np.testing.assert_array_equal(control, np.zeros((6, 15, 1)))
        diagnostic = projected_covariance(control[train])
        self.assertEqual(diagnostic["numerical_rank"], 0)
        self.assertEqual(diagnostic["trace"], 0.)
        self.assertTrue(all(row["effective_df"] == 0 for row in diagnostic["ridge_df_by_lambda"]))
        good = projected_covariance(paired[train])
        self.assertEqual(good["numerical_rank"], 1)
        self.assertAlmostEqual(good["trace"], 1., places=14)
        for row in good["ridge_df_by_lambda"]:
            self.assertAlmostEqual(row["effective_df"], 1/(1+row["lambda"]), places=14)
        residual = rng.normal(size=(60, 2))
        predicted = ridge_predictions(control[train], control[~train], residual)
        np.testing.assert_array_equal(predicted, np.zeros((5, 30, 2)))
        phi, evaluation = paired[train].reshape(-1, 1), paired[~train].reshape(-1, 1)
        actual = ridge_predictions(paired[train], paired[~train], residual)
        for index, penalty in enumerate(LAMBDAS):
            expected = evaluation@np.linalg.solve(phi.T@phi+len(phi)*penalty*np.eye(1), phi.T@residual)
            np.testing.assert_allclose(actual[index], expected, atol=2e-14, rtol=1e-12)

    def test_loading_map_uses_one_frozen_ordinary_shuffle_even_if_identity(self):
        # Redrawing fixed points or keying on a held fold would change the
        # prospective control distribution and its cross-fold consistency.
        class IdentityDraw:
            def __init__(self):
                self.calls = 0
            def seed(self, literal, version):
                self.literal, self.version = literal, version
            def shuffle(self, values):
                self.calls += 1
        draw = IdentityDraw()
        with patch("primary_pca.random.Random", return_value=draw):
            actual = loading_map(4, "synthetic", 2)
        np.testing.assert_array_equal(actual, np.arange(4))
        self.assertEqual(draw.calls, 1)
        self.assertEqual(draw.version, 2)
        self.assertEqual(draw.literal, "ep05-lmp-pca90-loading-label-v1|20261004|synthetic|2")
        expected = loading_map(9, "synthetic", 3)
        loading_map(9, "unrelated", 7)
        np.testing.assert_array_equal(expected, loading_map(9, "synthetic", 3))

    def test_zero_energy_controls_keep_native_target_and_shared_baseline(self):
        # Integration must preserve original XY and the exact shared baseline
        # when a loading permutation sends the projection onto a constant axis.
        rng = np.random.default_rng(50904)
        labels = np.repeat(np.arange(8), 3)
        ids = np.arange(24)
        x = np.stack((rng.normal(size=(24, 15)), np.zeros((24, 15))), axis=2)
        y = rng.normal(size=(24, 15, 2))
        originals = [v.copy() for v in (x, y, labels, ids)]
        with patch("primary_pca.loading_map", return_value=np.array([1, 0])):
            banks, baseline, predictions = fit_banks(x, y, labels, ids, "zero-control")
        for replicate in REPLICATES:
            for prediction in predictions[RECIPES[1]][replicate]:
                np.testing.assert_array_equal(prediction, baseline)
        self.assertTrue(all(row["delta_R2"] == 0 for row in banks[RECIPES[1]]["lambda_rows"]))
        for actual, before in zip((x, y, labels, ids), originals):
            np.testing.assert_array_equal(actual, before)

    def test_control_scores_are_mean_seed_errors_not_prediction_ensemble(self):
        # A control ensemble would receive an unregistered variance benefit;
        # a held direction's target must not affect its own predictions either.
        rng = np.random.default_rng(50905)
        labels, ids = np.repeat(np.arange(8), 3), np.arange(24)
        signal = rng.normal(size=(24, 15))
        x = np.stack((signal, signal+.1*rng.normal(size=signal.shape),
                      np.zeros_like(signal)), axis=2)
        y = np.stack((signal, -signal), axis=2)+.2*rng.normal(size=(24, 15, 2))
        banks, baseline, predictions = fit_banks(x, y, labels, ids, "means")
        controls = predictions[RECIPES[1]]
        for index in range(len(LAMBDAS)):
            errors = ((y[None]-controls[:, index])**2).sum(axis=(1, 2, 3))
            wrong = np.mean([wrong_pair_sse(y, controls[r, index], labels) for r in REPLICATES])
            row = banks[RECIPES[1]]["lambda_rows"][index]
            self.assertAlmostEqual(row["candidate_SSE"], float(errors.mean()), places=9)
            self.assertAlmostEqual(row["wrong_pair_SSE"], float(wrong), places=9)
        ensemble = float(((y-controls[:, 0].mean(axis=0))**2).sum())
        self.assertGreater(banks[RECIPES[1]]["lambda_rows"][0]["candidate_SSE"]-ensemble, 1e-6)
        changed = y.copy()
        changed[labels == 2] += 1e6
        _, other_baseline, other = fit_banks(x, changed, labels, ids, "means")
        np.testing.assert_array_equal(baseline[labels == 2], other_baseline[labels == 2])
        np.testing.assert_array_equal(predictions[RECIPES[0]][:, labels == 2], other[RECIPES[0]][:, labels == 2])
        np.testing.assert_array_equal(controls[:, :, labels == 2], other[RECIPES[1]][:, :, labels == 2])

    def test_paired_outer_lambda_and_full_span_duplicate_accounting(self):
        # Held scores/control scores must not choose P's lambda; matched anchor
        # and all controls share it. A fully spanning panel adds no trial depth.
        def bank(recipe, kind):
            rows = []
            for index, penalty in enumerate(LAMBDAS):
                delta = .01*index if kind == "paired" else (.04-.01*index if kind == "control" else .06-.01*index)
                identity = .01+.001*index if kind == "paired" else .004
                sse = 5-10*delta
                rows.append({"lambda": penalty, "delta_R2": delta, "identity_increment": identity,
                             "candidate_SSE": sse, "wrong_pair_SSE": sse+10*identity,
                             "candidate_R2": 1-sse/10})
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                    "baseline_SSE": 5., "denominator": 10., "direction_counts": [3]*8,
                    "lambda_rows": rows, "all_folds_full_span": False}
        sessions = {s: {RECIPES[0]: bank(RECIPES[0], "paired"),
                        RECIPES[1]: average_control_banks([bank(RECIPES[1], "control") for _ in REPLICATES])}
                    for s in SESSIONS}
        anchors = {s: bank(ANCHOR, "anchor") for s in SESSIONS}
        independent = {"recipe": ANCHOR, "sessions": [
            {"session_id": s, "animal": ANIMALS[s], **anchors[s]["lambda_rows"][0]} for s in SESSIONS]}
        before = summarize_banks(sessions, anchors, independent)
        self.assertAlmostEqual(before["anchor_contrasts"]["independently_selected"]["primary_increment"], -.02)
        self.assertAlmostEqual(before["anchor_contrasts"]["pca_lambda_matched"]["primary_increment"], .02)
        for held in SESSIONS:
            changed = copy.deepcopy(sessions)
            for index, row in enumerate(changed[held][RECIPES[0]]["lambda_rows"]):
                row["delta_R2"] = 1e8*(5-index)
            after = summarize_banks(changed, anchors, independent)
            for panel in after["panels"]:
                row = next(r for r in panel["sessions"] if r["session_id"] == held)
                self.assertEqual(row["lambda_index"], 4)
                self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
            row = next(r for r in after["matched_lambda_anchor_sessions"] if r["session_id"] == held)
            self.assertEqual(row["lambda"], LAMBDAS[4])
        self.assertEqual(before["scientific_trial_count_increment"], 2)
        for banks in sessions.values():
            banks[RECIPES[0]]["all_folds_full_span"] = True
        duplicate = summarize_banks(sessions, anchors, independent)
        self.assertTrue(duplicate["structural_full_span_duplicate"])
        self.assertEqual(duplicate["scientific_trial_count_increment"], 0)
        self.assertTrue(all(p["scientific_trial_count_increment"] == 0 for p in duplicate["panels"]))


if __name__ == "__main__":
    unittest.main()
