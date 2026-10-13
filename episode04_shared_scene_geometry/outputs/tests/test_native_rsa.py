"""Mathematical invariants and counterexamples for frozen EP04 equations."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import native_rsa as rsa


def symmetric(edges):
    matrix = np.zeros((4, 4))
    matrix[np.triu_indices(4, 1)] = edges
    return matrix + matrix.T


class NativeRSATests(unittest.TestCase):
    def test_cosine_scaling_and_neural_correlation_offset_invariance(self):
        rows = np.array([[1., 2., 4.], [3., -1., 2.], [2., 2., -1.], [-2., 3., 1.]])
        np.testing.assert_allclose(rsa.rdm(rows), rsa.rdm(rows * np.array([2., 3., 4., 5.])[:, None]), atol=1e-14)
        np.testing.assert_allclose(rsa.rdm(rows, "correlation"),
                                   rsa.rdm(rows + np.arange(4)[:, None], "correlation"), atol=1e-14)
        with self.assertRaises(rsa.NotEvaluable):
            rsa.rdm(np.ones((4, 3)), "correlation")

    def test_tied_ranks_and_spearman_against_hand_rank_vectors(self):
        np.testing.assert_array_equal(rsa.average_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
        left = symmetric([1, 1, 2, 3, 4, 5])
        right = symmetric([2, 4, 3, 1, 6, 5])
        expected = np.corrcoef([1.5, 1.5, 3, 4, 5, 6], [2, 4, 3, 1, 6, 5])[0, 1]
        self.assertAlmostEqual(rsa.spearman_rsa(left, right), expected, places=14)

    def test_partial_rank_regression_agrees_with_three_correlation_identity(self):
        x = np.array([4., 1., 6., 2., 5., 3.])
        y = np.array([2., 6., 1., 4., 3., 5.])
        low = np.arange(1., 7.)
        corr = np.corrcoef([x, y, low])
        expected = (corr[0, 1] - corr[0, 2] * corr[1, 2]) / np.sqrt((1 - corr[0, 2]**2) * (1 - corr[1, 2]**2))
        self.assertAlmostEqual(rsa.partial_spearman_rsa(symmetric(x), symmetric(y), symmetric(low)), expected, places=13)
        with self.assertRaises(rsa.NotEvaluable):
            rsa.partial_spearman_rsa(symmetric(low), symmetric(y), symmetric(low))

    def test_whole_image_null_preserves_strata_and_pairs(self):
        strata = np.array(["laion", "things", "laion", "thingsplus", "things", "laion"])
        permutation = rsa.whole_image_permutation(strata, 17)
        np.testing.assert_array_equal(strata[permutation], strata)
        matrix = np.arange(36).reshape(6, 6)
        moved = rsa.relabel_rdm(matrix, permutation)
        for left in range(6):
            for right in range(6):
                self.assertEqual(moved[left, right], matrix[permutation[left], permutation[right]])
        np.testing.assert_array_equal(permutation, rsa.whole_image_permutation(strata, 17))

    def test_random_geometry_preserves_pre_l2_covariance_and_mean(self):
        rng = np.random.Generator(np.random.PCG64(34))
        for shape in ((13, 4), (7, 11)):
            raw = rng.normal(size=shape) + np.arange(shape[1]) / 10
            changed, normalized = rsa.covariance_matched_geometry(raw, 1, 7)
            np.testing.assert_allclose(changed.mean(axis=0), raw.mean(axis=0), atol=1e-12)
            np.testing.assert_allclose(np.cov(changed, rowvar=False), np.cov(raw, rowvar=False), atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1., atol=1e-12)
            np.testing.assert_array_equal(changed, rsa.covariance_matched_geometry(raw, 1, 7)[0])
            self.assertFalse(np.allclose(changed, rsa.covariance_matched_geometry(raw, 1, 8)[0]))

    def test_repeat_rule_averages_every_valid_presentation(self):
        identifiers, patterns, counts = rsa.repeat_average(
            [[1, 2], [3, 4], [20, 30], [np.nan, np.nan], [5, 6], [7, 8]],
            ["a", "a", "b", "b", "a", "c"], [True, True, True, False, True, True])
        self.assertEqual(identifiers, ["a"])
        np.testing.assert_array_equal(patterns, [[3, 4]])
        np.testing.assert_array_equal(counts, [3])

    def test_participant_fold_sector_equal_weights(self):
        dino = np.zeros((5, 2, 3))
        vlm = np.zeros((3, 5, 2, 3))
        for participant in range(5):
            vlm[0, participant] = participant + np.arange(2)[:, None] * 10 + np.arange(3)[None, :] * 100
        effects = rsa.panel_effects(vlm, dino)
        np.testing.assert_allclose(effects["participant"][0], [105, 106, 107, 108, 109])
        self.assertEqual(effects["gain"][0], 107)
        np.testing.assert_allclose(effects["participant_loo"][0], [107.5, 107.25, 107., 106.75, 106.5])
        with self.assertRaises(rsa.NotEvaluable):
            rsa.panel_effects(vlm[:, :4], dino[:4])

    def test_null_argmax_is_unconditional_and_keeps_signed_unsubtracted_statistic(self):
        identities = [rsa.ModelIdentity("clip", 1024), rsa.ModelIdentity("pe", 1024), rsa.ModelIdentity("sig", 1152)]
        vlm = np.broadcast_to(np.array([-.2, -.1, -.3])[:, None, None, None], (3, 5, 2, 3))
        selected, statistic = rsa.familywise_null_statistic(vlm, np.full((5, 2, 3), .2), identities)
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(statistic, -.1)

    def test_candidate_screen_precedes_selection_and_missing_falsifier_blocks(self):
        identities = [rsa.ModelIdentity("clip", 1024), rsa.ModelIdentity("pe", 1024), rsa.ModelIdentity("sig", 1152)]
        vlm = np.broadcast_to(np.array([.3, .25, .21])[:, None, None, None], (3, 5, 2, 3))
        dino = np.full((5, 2, 3), .2)
        controls = {name: ["pass"] * 3 for name in rsa.MANDATORY_SENSITIVITIES}
        controls["low_level_partial_rsa"][0] = "fail"
        diagnostics = {name: True for name in rsa.REQUIRED_DIAGNOSTICS}
        result = rsa.screen_complete_panel(vlm, dino, identities, .1, .1, controls, diagnostics)
        self.assertEqual(result["selected_index"], 1)
        self.assertFalse(result["configuration_lock_authorized"])
        controls["complete_repeats"][2] = "not_evaluable"
        with self.assertRaises(rsa.NotEvaluable):
            rsa.screen_complete_panel(vlm, dino, identities, .1, .1, controls, diagnostics)

    def test_tie_and_undefined_fisher_rules(self):
        identities = [rsa.ModelIdentity("b", 8), rsa.ModelIdentity("a", 4), rsa.ModelIdentity("c", 4)]
        self.assertEqual(rsa.frozen_argmax([.1, .1 - 5e-7, .1 - 7e-7], identities), 1)
        self.assertAlmostEqual(rsa.fisher_z(.5), .5 * np.log(3), places=14)
        for value in (1., -1., float("nan")):
            with self.assertRaises(rsa.NotEvaluable):
                rsa.fisher_z(value)


if __name__ == "__main__":
    unittest.main()
