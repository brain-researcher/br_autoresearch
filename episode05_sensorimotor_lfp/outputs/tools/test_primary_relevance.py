"""Independent scientific checks for nested channel relevance and its control."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import primary_relevance as pr
from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse


class RelevanceTest(unittest.TestCase):
    def test_scalar_ridge_uses_mean_loss_and_two_native_targets(self):
        fit = np.array([[[1., 2.], [-1., -2.]]])
        test = np.array([[[3., 4.]]])
        residual = np.array([[2., -4.], [-2., 4.]])
        actual = pr.single_channel_predictions(fit, test, residual)
        # z1'z1=2, z2'z2=8, n=2. No target normalization or 1/n omission.
        expected = np.array([[[3., -6.], [3.2, -6.4]]])
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_outer_rows_removed_before_inner_baseline_and_scaling(self):
        rng = np.random.default_rng(51001)
        labels = np.repeat(np.arange(8), np.arange(2, 10))
        x, y = rng.normal(size=(len(labels), 15, 3)), rng.normal(size=(len(labels), 15, 2))
        held = 3
        calls, scales = [], []
        original_fit, original_scale = pr.DirectionTime.fit, pr.prepare_split
        def fit(theta, targets, harmonic, directions):
            calls.append(set(directions.tolist()))
            return original_fit(theta, targets, harmonic, directions)
        def scale(values, train):
            scales.append((len(values), int(np.count_nonzero(train))))
            return original_scale(values, train)
        with patch.object(pr.DirectionTime, 'fit', side_effect=fit), patch.object(pr, 'prepare_split', side_effect=scale):
            order, meta = pr.rank_channels(x, y, labels, [1, 2, 3], held)
        self.assertEqual(len(calls), 7)
        for inner, directions in zip([d for d in range(8) if d != held], calls):
            self.assertEqual(directions, set(range(8))-{held, inner})
        self.assertTrue(all(n == np.count_nonzero(labels != held) for n, _ in scales))
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[labels == held] = np.nan
        changed_y[labels == held] = np.nan
        other, other_meta = pr.rank_channels(changed_x, changed_y, labels, [1, 2, 3], held)
        np.testing.assert_array_equal(order, other)
        self.assertEqual(meta, other_meta)
        contributions = np.array([f['gain_SSE_by_physical_channel'] for f in meta['inner_fits']])
        denominator = 15*np.count_nonzero(labels != held)
        self.assertEqual(meta['common_evaluation_bin_rows'], denominator)
        np.testing.assert_allclose(meta['signed_mean_gain_by_physical_channel'], contributions.sum(axis=0)/denominator)

    def test_no_skill_keeps_zero_scores_and_physical_id_tie_order(self):
        labels = np.repeat(np.arange(8), 2)
        x = np.zeros((16, 15, 4))
        y = np.random.default_rng(51002).normal(size=(16, 15, 2))
        order, meta = pr.rank_channels(x, y, labels, [40, 10, 30, 20], 7)
        np.testing.assert_array_equal(order, [1, 3, 2, 0])
        self.assertEqual(meta['positive_gain_channels'], 0)
        self.assertTrue(meta['all_raw_gains_tied'])
        self.assertEqual(meta['cutoff_tied_channels'], 4)
        # Mathematically zero SSE gains can retain signed summation roundoff;
        # the scientific rule must keep those values, not clip them to zero.
        np.testing.assert_allclose(meta['raw_gain_range'], [0., 0.], rtol=0, atol=1e-13)

    def test_rank_labels_map_strict_ranks_and_accept_one_identity_draw(self):
        class Draw:
            calls = 0
            def seed(self, literal, version): self.literal, self.version = literal, version
            def shuffle(self, values): self.calls += 1
        draw = Draw()
        with patch.object(pr.random, 'Random', return_value=draw):
            mapping = pr.rank_map(20, 'example', 2)
        np.testing.assert_array_equal(mapping, np.arange(20))
        self.assertEqual(draw.calls, 1)
        self.assertEqual(draw.version, 2)
        self.assertEqual(draw.literal, 'ep05-lmp-inner-lodo-top16-rank-label-v1|20261004|example|2')
        np.testing.assert_array_equal(pr.retained_order([3, 0, 2, 1], [1, 3, 0, 2]), [1, 2, 3, 0])
        order = np.arange(20)[::-1]
        self.assertEqual(len(pr.retained_order(order)), 16)
        self.assertEqual(len(pr.retained_order(order, np.roll(np.arange(20), 5))), 16)

    def test_control_set_changes_gram_without_post_retention_scaling(self):
        rng = np.random.default_rng(51003)
        z = rng.normal(size=(7, 15, 20))*np.arange(1, 21)
        train = np.arange(7) < 5
        paired, ids = pr.retain_arm(z, np.arange(20), train)
        control, selected = pr.retain_arm(z, np.arange(20), train, np.arange(20)[::-1])
        np.testing.assert_array_equal(ids, np.arange(16))
        np.testing.assert_array_equal(selected, np.arange(19, 3, -1))
        expected = z[:, :, selected]-z[train][:, :, selected].mean(axis=(0, 1))
        np.testing.assert_allclose(control, expected, rtol=0, atol=1e-13)
        self.assertGreater(abs(float(np.sum(paired[train]**2)-np.sum(control[train]**2))), 1.)
        altered = z.copy()
        altered[~train] += 1e8
        again, _ = pr.retain_arm(altered, np.arange(20), train)
        np.testing.assert_array_equal(again[train], paired[train])

    def test_full_retention_permutations_equal_original_ridge_predictions(self):
        rng = np.random.default_rng(51004)
        labels, ids = np.repeat(np.arange(8), 2), np.arange(16)
        x, y = rng.normal(size=(16, 15, 4)), rng.normal(size=(16, 15, 2))
        banks, baseline, predictions = pr.fit_banks(x, y, labels, ids, [1, 2, 3, 4], 'full')
        self.assertTrue(banks[pr.RECIPES[0]]['all_folds_full_retention'])
        for seed in range(8):
            np.testing.assert_allclose(predictions[pr.RECIPES[1]][seed], predictions[pr.RECIPES[0]], atol=2e-13, rtol=1e-11)

    def test_mean_seed_sse_and_held_targets_do_not_change_own_predictions(self):
        rng = np.random.default_rng(51005)
        labels, ids = np.repeat(np.arange(8), 2), np.arange(16)
        x = rng.normal(size=(16, 15, 20))
        y = np.stack((x[:, :, 0], -x[:, :, 1]), axis=2)+rng.normal(size=(16, 15, 2))*.1
        channels = list(range(1, 21))
        banks, baseline, predictions = pr.fit_banks(x, y, labels, ids, channels, 'scores')
        controls = predictions[pr.RECIPES[1]]
        for i in range(len(LAMBDAS)):
            errors = ((y[None]-controls[:, i])**2).sum(axis=(1, 2, 3))
            row = banks[pr.RECIPES[1]]['lambda_rows'][i]
            self.assertAlmostEqual(row['candidate_SSE'], float(errors.mean()), places=9)
            self.assertAlmostEqual(row['wrong_pair_SSE'], float(np.mean([wrong_pair_sse(y, controls[s, i], labels) for s in range(8)])), places=9)
        ensemble_sse = float(((y-controls[:, 0].mean(axis=0))**2).sum())
        self.assertGreater(banks[pr.RECIPES[1]]['lambda_rows'][0]['candidate_SSE']-ensemble_sse, 1e-6)
        changed = y.copy()
        changed[labels == 2] += 1e6
        _, other_baseline, other = pr.fit_banks(x, changed, labels, ids, channels, 'scores')
        np.testing.assert_array_equal(other_baseline[labels == 2], baseline[labels == 2])
        np.testing.assert_array_equal(other[pr.RECIPES[0]][:, labels == 2], predictions[pr.RECIPES[0]][:, labels == 2])
        np.testing.assert_array_equal(other[pr.RECIPES[1]][:, :, labels == 2], controls[:, :, labels == 2])

    def test_outer_lambda_controls_anchor_and_duplicate_accounting(self):
        def bank(recipe, candidate):
            rows = []
            for i, penalty in enumerate(LAMBDAS):
                gain = .01*i if candidate else .04-.01*i
                rows.append({'lambda': penalty, 'delta_R2': gain, 'identity_increment': .01,
                             'candidate_SSE': 5-10*gain, 'wrong_pair_SSE': 5.1-10*gain, 'candidate_R2': .5+gain})
            return {'recipe': recipe, 'lambdas': list(LAMBDAS), 'baseline_R2': .5, 'baseline_SSE': 5.,
                    'denominator': 10., 'direction_counts': [2]*8, 'lambda_rows': rows, 'all_folds_full_retention': False}
        sessions = {s: {pr.RECIPES[0]: bank(pr.RECIPES[0], True), pr.RECIPES[1]: pr.average_control_banks([bank(pr.RECIPES[1], False) for _ in range(8)])} for s in SESSIONS}
        anchors = {s: bank(pr.ANCHOR, False) for s in SESSIONS}
        panel = {'recipe': pr.ANCHOR, 'sessions': [{'session_id': s, 'animal': ANIMALS[s], **anchors[s]['lambda_rows'][0]} for s in SESSIONS]}
        initial = pr.summarize_banks(sessions, anchors, panel)
        self.assertEqual(initial['scientific_trial_count_increment'], 2)
        self.assertAlmostEqual(initial['anchor_contrasts']['relevance_lambda_matched']['primary_increment'], .04)
        for held in SESSIONS:
            changed = copy.deepcopy(sessions)
            for i, row in enumerate(changed[held][pr.RECIPES[0]]['lambda_rows']): row['delta_R2'] = 1e7*(5-i)
            summary = pr.summarize_banks(changed, anchors, panel)
            for result in summary['panels']:
                target = next(r for r in result['sessions'] if r['session_id'] == held)
                self.assertEqual(target['lambda_index'], 4)
                self.assertEqual(set(target['outer_training_sessions']), set(SESSIONS)-{held})
        for banks in sessions.values(): banks[pr.RECIPES[0]]['all_folds_full_retention'] = True
        duplicate = pr.summarize_banks(sessions, anchors, panel)
        self.assertTrue(duplicate['structural_full_retention_duplicate'])
        self.assertEqual(duplicate['scientific_trial_count_increment'], 0)


if __name__ == '__main__':
    unittest.main()
