"""Independent native-objective, convergence and leakage checks for elastic net."""
import copy
import itertools
import unittest
from unittest.mock import patch

import numpy as np

import primary_elastic_net as en
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


def enumerated_solution(gram, cross, penalty, scale):
    """Enumerate active signs for a small problem, solving each smooth subproblem."""
    output = np.zeros_like(cross)
    threshold = penalty*scale
    for target in range(2):
        feasible = []
        for signs in itertools.product((-1, 0, 1), repeat=len(gram)):
            signs = np.asarray(signs)
            active = np.flatnonzero(signs)
            w = np.zeros(len(gram))
            if len(active):
                w[active] = np.linalg.solve(gram[np.ix_(active, active)]+penalty*np.eye(len(active)),
                                           cross[active, target]-threshold*signs[active])
                if np.any(w[active]*signs[active] <= 0):
                    continue
            gradient = gram@w-cross[:, target]+penalty*w
            if np.any(np.abs(gradient[signs == 0]) > threshold+1e-12):
                continue
            objective = .5*w@gram@w-w@cross[:, target]+penalty*.5*(w@w)+threshold*np.abs(w).sum()
            feasible.append((objective, w))
        if not feasible:
            raise AssertionError("Independent sign enumeration found no solution")
        output[:, target] = min(feasible, key=lambda item: item[0])[1]
    return output


def example(seed=85001):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(8), 2)
    ids = np.array([f"example_trial{i+1:06d}" for i in range(16)])
    x = rng.normal(size=(16, 15, 3))
    y = rng.normal(size=(16, 15, 2))+.5*x[:, :, :2]
    return x, y, labels, ids


class ElasticNetTest(unittest.TestCase):
    def test_correlated_native_objective_matches_independent_active_sign_solution(self):
        gram = np.array([[1., .6, -.25], [.6, 1.4, .1], [-.25, .1, .8]])
        cross = np.array([[1.2, -.2], [.8, 1.6], [-.9, .45]])
        scale = .7
        fitted, diagnostics = en.solve_path(gram, cross, scale)
        for i, penalty in enumerate(LAMBDAS):
            exact = enumerated_solution(gram, cross, penalty, scale)
            np.testing.assert_allclose(fitted[i], exact, rtol=2e-6, atol=2e-6)
            gradient = gram@fitted[i]-cross+penalty*fitted[i]
            residual = np.where(fitted[i] != 0, np.abs(gradient+penalty*scale*np.sign(fitted[i])),
                                np.maximum(np.abs(gradient)-penalty*scale, 0.))
            self.assertLessEqual(float(residual.max()), 1e-8*max(1, abs(cross).max()))
            info = diagnostics[i]
            self.assertTrue(info["converged"])
            self.assertAlmostEqual(info["objective_error_upper_bound"], float(np.sum(residual**2)/(2*penalty)))
            self.assertEqual(info["lambda"], penalty)
            def objective(w):
                return .5*np.sum(w*(gram@w))-np.sum(w*cross)+.5*penalty*np.sum(w*w)+penalty*scale*np.abs(w).sum()
            self.assertLessEqual(objective(fitted[i])-objective(exact), info["objective_error_upper_bound"]+2e-12)

    def test_orthogonal_soft_threshold_ridge_limit_and_zero_features(self):
        gram = np.diag([1., 2., 0.])
        cross = np.array([[.8, -.4], [-1.7, .05], [0., 0.]])
        fitted, _ = en.solve_path(gram, cross, .6)
        ridge, _ = en.solve_path(gram, cross, 0.)
        for i, penalty in enumerate(LAMBDAS):
            expected = np.sign(cross)*np.maximum(np.abs(cross)-penalty*.6, 0)/(np.diag(gram)[:, None]+penalty)
            np.testing.assert_allclose(fitted[i], expected, rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(ridge[i], np.linalg.solve(gram+penalty*np.eye(3), cross), atol=1e-14)
        zero, diagnostics = en.solve_path(gram, np.zeros((3, 2)), 0.)
        np.testing.assert_array_equal(zero, np.zeros((5, 3, 2)))
        self.assertTrue(all(d["zero_coefficients"] == 6 and d["active_channel_union"] == 0 for d in diagnostics))
        np.testing.assert_array_equal(fitted[-1], np.zeros((3, 2)))
        self.assertEqual(en.penalty_scale(np.zeros((4, 15, 2))), 0.)

    def test_common_velocity_scale_and_coefficients_change_consistently(self):
        rng = np.random.default_rng(85002)
        x, u = rng.normal(size=(60, 3)), 3*rng.normal(size=(60, 2))
        gram, cross = x.T@x/60, x.T@u/60
        scale = en.penalty_scale(u)
        self.assertAlmostEqual(scale, float(np.sqrt(np.sum(u*u)/(2*60))))
        first, _ = en.solve_path(gram, cross, scale)
        second, _ = en.solve_path(gram, 7*cross, en.penalty_scale(7*u))
        np.testing.assert_allclose(second, 7*first, rtol=2e-6, atol=2e-6)

    def test_nonconvergence_is_an_error_even_with_finite_coefficients(self):
        gram, cross = np.array([[1., .9], [.9, 1.]]), np.array([[2., 1.2], [1., 2.]])
        with self.assertRaisesRegex(RuntimeError, "KKT"):
            en.solve_path(gram, cross, .1, max_sweeps=1)
        with self.assertRaises(ValueError): en.solve_path(gram, cross, .1, max_sweeps=10001)
        cert = en.kkt_certificate(gram, cross, np.zeros((2, 2)), .1, .1)
        self.assertFalse(cert["converged"])

    def test_native_original_ID_maps_keep_whole_trials_and_accept_identity(self):
        labels = np.repeat(np.arange(7), 3)
        numbers = np.random.default_rng(85003).permutation(np.arange(21)+1)
        ids = np.array([f"example_trial{i:06d}" for i in numbers])
        mapping, records = en.trial_map(ids, labels, "example", 7, 3)
        np.testing.assert_array_equal(labels[mapping], labels)
        reversed_rows = np.arange(21)[::-1]
        reverse, reverse_records = en.trial_map(ids[reversed_rows], labels[reversed_rows], "example", 7, 3)
        self.assertEqual(records, reverse_records)
        self.assertEqual(dict(zip(ids, ids[mapping])), dict(zip(ids[reversed_rows], ids[reversed_rows][reverse])))
        draws = []
        class Identity:
            def seed(self, literal, version): draws.append((literal, version))
            def shuffle(self, values): pass
        with patch.object(en.random, "Random", Identity):
            identity, _ = en.trial_map(ids, labels, "example", 7, 3)
        np.testing.assert_array_equal(identity, np.arange(21))
        self.assertEqual(draws, [(f"ep05-lmp-elastic-net-training-label-v1|20261004|example|7|3|{d}", 2) for d in range(7)])

    def test_controls_share_exact_Gram_and_scale_but_use_permuted_LMP_cross_covariance(self):
        x, y, labels, ids = example(85004)
        seen = []
        original = en.solve_path
        def solve(gram, cross, scale, check):
            seen.append((gram, cross.copy(), scale))
            return original(gram, cross, scale, check)
        with patch.object(en, "solve_path", side_effect=solve):
            en.fit_banks(x, y, labels, ids, "example")
        self.assertEqual(len(seen), 72)
        for held in range(8):
            train = labels != held
            baseline = en.DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
            u = (y[train]-baseline.predict(labels[train]*np.pi/4)).reshape(-1, 2)
            u -= u.mean(axis=0)
            z, _ = en.prepare_split(x, train)
            gram = z[train].reshape(-1, 3).T@z[train].reshape(-1, 3)/len(u)
            np.testing.assert_array_equal(seen[9*held][0], gram)
            for arm in range(9):
                self.assertIs(seen[9*held+arm][0], seen[9*held][0])
                self.assertEqual(seen[9*held+arm][2], en.penalty_scale(u))
                if arm:
                    mapping, _ = en.trial_map(ids[train], labels[train], "example", held, arm-1)
                    expected = z[train][mapping].reshape(-1, 3).T@u/len(u)
                    np.testing.assert_array_equal(seen[9*held+arm][1], expected)
        self.assertTrue(any(not np.array_equal(seen[0][1], entry[1]) for entry in seen[1:9]))

    def test_held_targets_cannot_change_fits_and_controls_average_SSE(self):
        x, y, labels, ids = example(85005)
        banks, baseline, predictions = en.fit_banks(x, y, labels, ids, "example")
        other_y = y.copy()
        other_y[labels == 2] += 1e6
        other, other_baseline, other_predictions = en.fit_banks(x, other_y, labels, ids, "example")
        self.assertEqual(banks[en.RECIPES[0]]["direction_fit_metadata"][2], other[en.RECIPES[0]]["direction_fit_metadata"][2])
        np.testing.assert_array_equal(baseline[labels == 2], other_baseline[labels == 2])
        np.testing.assert_array_equal(predictions[en.RECIPES[0]][:, labels == 2], other_predictions[en.RECIPES[0]][:, labels == 2])
        control = predictions[en.RECIPES[1]]
        for i in range(len(LAMBDAS)):
            row = banks[en.RECIPES[1]]["lambda_rows"][i]
            self.assertAlmostEqual(row["candidate_SSE"], float(np.mean(np.sum((control[:, i]-y)**2, axis=(1, 2, 3)))), places=9)
            self.assertAlmostEqual(row["wrong_pair_SSE"], float(np.mean([wrong_pair_sse(y, control[r, i], labels) for r in range(8)])), places=9)

    def test_outer_session_lambda_exclusion_and_shared_control_selection(self):
        def bank(recipe, paired):
            rows = [{"lambda": p, "delta_R2": .01*i if paired else .04-.01*i,
                     "identity_increment": .01, "candidate_SSE": 5., "wrong_pair_SSE": 5.1,
                     "candidate_R2": .5} for i, p in enumerate(LAMBDAS)]
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                    "baseline_SSE": 5., "denominator": 10., "direction_counts": [2]*8,
                    "lambda_rows": rows, "all_folds_zero_residual_scale": False}
        sessions = {s: {en.RECIPES[0]: bank(en.RECIPES[0], True),
                    en.RECIPES[1]: en.average_control_banks([bank(en.RECIPES[1], False) for _ in range(8)])} for s in SESSIONS}
        anchors = {s: bank(en.ANCHOR, False) for s in SESSIONS}
        anchor_panel = {"recipe": en.ANCHOR, "sessions": [{"session_id": s, "animal": ANIMALS[s],
                        **anchors[s]["lambda_rows"][0]} for s in SESSIONS]}
        held = SESSIONS[0]
        for i, row in enumerate(sessions[held][en.RECIPES[0]]["lambda_rows"]): row["delta_R2"] = 1e7*(5-i)
        result = en.summarize_banks(sessions, anchors, anchor_panel)
        for panel in result["panels"]:
            row = next(row for row in panel["sessions"] if row["session_id"] == held)
            self.assertEqual(row["lambda_index"], 4)
            self.assertEqual(set(row["outer_training_sessions"]), set(SESSIONS)-{held})
        self.assertEqual(result["scientific_trial_count_increment"], 2)
        for banks in sessions.values(): banks[en.RECIPES[0]]["all_folds_zero_residual_scale"] = True
        self.assertEqual(en.summarize_banks(sessions, anchors, anchor_panel)["scientific_trial_count_increment"], 0)


if __name__ == "__main__":
    unittest.main()
