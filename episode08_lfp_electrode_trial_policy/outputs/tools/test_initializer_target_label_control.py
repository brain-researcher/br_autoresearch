"""Scientific controls for fitting-view label maps and unchanged true targets."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import initializer_target_label_control as control
import initializer_target_label_experiment as experiment
from residual_selection_training import selector_order


class InitializerLabelTest(unittest.TestCase):
    def test_maps_use_original_ids_and_keep_whole_vectors_within_direction(self):
        ids = np.arange(101, 133)
        directions = np.repeat(np.arange(8), 4)
        mapping, records = control.within_direction_map(ids, directions, 'session', 'outer', 3)
        np.testing.assert_array_equal(directions[mapping], directions)
        self.assertEqual(sorted(mapping.tolist()), list(range(32)))
        order = np.random.default_rng(80851).permutation(32)
        changed, other_records = control.within_direction_map(ids[order], directions[order], 'session', 'outer', 3)
        self.assertEqual(records, other_records)
        self.assertEqual(dict(zip(ids, ids[mapping])), dict(zip(ids[order], ids[order][changed])))
        targets = np.column_stack((ids, -2*ids, ids+700))
        moved = targets[mapping]
        np.testing.assert_array_equal(moved[:, 1], -2*moved[:, 0])
        np.testing.assert_array_equal(moved[:, 2], moved[:, 0]+700)
        for record in records:
            self.assertEqual(set(record['recipient_trial_ids']), set(record['donor_trial_ids']))

    def test_identity_draw_is_accepted_once_with_exact_seed_literal(self):
        draws = []
        class Identity:
            def __init__(self): self.calls = 0; draws.append(self)
            def seed(self, literal, version): self.literal, self.version = literal, version
            def shuffle(self, values): self.calls += 1
        ids, directions = np.arange(24), np.repeat(np.arange(8), 3)
        with patch.object(control.random, 'Random', Identity):
            mapping, _ = control.within_direction_map(ids, directions, 'example', '2', 4)
        np.testing.assert_array_equal(mapping, np.arange(24))
        self.assertEqual(len(draws), 8)
        for direction, draw in enumerate(draws):
            self.assertEqual(draw.calls, 1)
            self.assertEqual(draw.version, 2)
            self.assertEqual(draw.literal, f'ep08-acquired-marginal-target-label-v1|20261004|example|2|4|{direction}')
        with self.assertRaises(ValueError):
            control.within_direction_map(ids, directions, 'example', 'outer', 4)
        with self.assertRaises(ValueError):
            control.within_direction_map(np.zeros(24, dtype=int), directions, 'example', '2', 4)

    def test_rank_uses_permuted_normalized_targets_and_original_gram(self):
        rng = np.random.default_rng(80852)
        x, y = rng.normal(size=(32, 6)), rng.normal(size=(32, 3))
        directions, ids = np.repeat(np.arange(8), 4), np.arange(32)
        originals = [a.copy() for a in (x, y)]
        prepared = control.prepare_view(x, y, directions, [2, 9])
        blocks, normalized_y, gram_plus_identity = prepared
        mapping, _ = control.within_direction_map(ids, directions, 'example', 'outer', 1)
        order, scores, _ = control.rank_view(x, y, directions, ids, [2, 9], 'example', 'outer', 1)
        expected = []
        for electrode in range(2):
            xx = blocks[:, electrode, :]
            matrix = np.eye(3)+xx.T@xx/32
            cross = xx.T@normalized_y[mapping]/32
            expected.append(np.trace(cross.T@np.linalg.solve(matrix, cross))/3)
            np.testing.assert_allclose(gram_plus_identity[electrode], matrix, atol=1e-14)
        np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(order, np.lexsort(([2, 9], -np.asarray(expected))))
        for before, after in zip(originals, (x, y)):
            np.testing.assert_array_equal(before, after)

    def test_identity_map_replays_true_marginal_recipe_and_ties(self):
        rng = np.random.default_rng(80853)
        x, y = rng.normal(size=(32, 12)), rng.normal(size=(32, 3))
        labels, ids, channels = np.repeat(np.arange(8), 4), np.arange(32), [1, 3, 8, 12]
        with patch.object(control, 'within_direction_map', return_value=(np.arange(32), [])):
            order, _, _ = control.rank_view(x, y, labels, ids, channels, 'example', 'outer', 0)
        expected = selector_order(x, y, labels, 'residual_marginal', 4, 'unused')
        np.testing.assert_array_equal(order, expected)
        tied, scores, _ = control.rank_view(np.zeros_like(x), y, labels, ids, channels, 'example', 'outer', 0)
        np.testing.assert_array_equal(tied, np.arange(4))
        np.testing.assert_array_equal(scores, np.zeros(4))

    def test_inner_views_exclude_their_validation_rows_and_freeze_all25_orders(self):
        rng = np.random.default_rng(80854)
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 3))
        labels, ids, channels = np.repeat(np.arange(8), 4), np.arange(32), list(range(1, 97))
        folds = [(np.flatnonzero(np.arange(32)%4 != i), np.flatnonzero(np.arange(32)%4 == i)) for i in range(4)]
        def progress(): return {'selector_orderings': 0, 'direction_permutation_maps': 0, 'selection_seconds': 0.}
        first_progress = progress()
        banks, records = control.order_bank(x, y, labels, ids, channels, folds, 'example', first_progress)
        self.assertEqual(len(records), 25)
        self.assertEqual(first_progress['selector_orderings'], 25)
        self.assertEqual(first_progress['direction_permutation_maps'], 200)
        altered_x, altered_y = x.copy(), y.copy()
        altered_x[folds[0][1]] += 1e6
        altered_y[folds[0][1]] -= 1e6
        other, other_records = control.order_bank(altered_x, altered_y, labels, ids, channels, folds, 'example', progress())
        for seed in control.MC_SEEDS:
            np.testing.assert_array_equal(banks[seed]['0'], other[seed]['0'])
            old = next(r for r in records if r['mc_seed'] == seed and r['view'] == '0')
            new = next(r for r in other_records if r['mc_seed'] == seed and r['view'] == '0')
            self.assertEqual(old, new)
            self.assertEqual(set(old['fitting_trial_ids']), set(folds[0][0]))
        self.assertTrue(all(len(order) == 32 for seed in banks.values() for order in seed.values()))

    def test_five_seed_sse_not_ensemble_and_all_twenty_cells_required(self):
        panels = []
        for amplitude in range(1, 6):
            error = float(amplitude**2)
            cells = [{'training_trials': n, 'electrodes': e, 'null_SSE': 100.,
                      'standardized_SSE': error, 'standardized_wrong_SSE': error+5.,
                      'native_target_SSE': 2*error, 'Q': 1-error/100,
                      'Q_wrong': 1-(error+5)/100, 'pairing_increment': .05,
                      'direction_only_Q': .1}
                     for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)]
            panels.append({'cells': cells})
        result = control.mean_control_cells(panels)
        self.assertAlmostEqual(result['mean_Q'], .89)
        self.assertNotAlmostEqual(result['mean_Q'], .91)  # Mean prediction3 would give SSE9.
        self.assertAlmostEqual(result['mean_Q_wrong'], .84)
        self.assertAlmostEqual(result['mean_pairing_increment'], .05)
        self.assertTrue(all(c['native_target_SSE'] == 22 for c in result['cells']))
        with self.assertRaises(ValueError): control.mean_control_cells(panels[:4])
        broken = copy.deepcopy(panels)
        broken[0]['cells'].pop()
        with self.assertRaises(ValueError): control.mean_control_cells(broken)
        broken = copy.deepcopy(panels)
        broken[0]['cells'][0]['null_SSE'] = 101.
        with self.assertRaises(ArithmeticError): control.mean_control_cells(broken)

    def test_every_order_precedes_future_reads_and_decoders_keep_true_y(self):
        rng = np.random.default_rng(80855)
        x, y = rng.normal(size=(32, 288)), rng.normal(size=(32, 3))
        original_y = y.copy()
        channels, ids = list(range(1, 97)), list(range(32))
        pools = [list(range(4*d, 4*d+4))+list(range(100+8*d, 108+8*d)) for d in range(8)]
        role = {'filename': 'synthetic', 'fitting_pool_by_direction': pools,
                'nested_training_sets': {'32': {'training_trials': ids}}}
        source = {'common_physical_channels': channels}
        records = {i: {'direction_index': i//4} for i in ids}
        folds = [(np.flatnonzero(np.arange(32)%4 != i), np.flatnonzero(np.arange(32)%4 == i)) for i in range(4)]
        state = {'orders_ready': False, 'decoder_calls': 0, 'future_calls': 0}
        banks = {seed: {view: np.roll(np.arange(96), seed)[:32] for view in control.VIEWS} for seed in control.MC_SEEDS}
        def orders(*args):
            self.assertEqual(state['future_calls'], 0)
            np.testing.assert_array_equal(args[1], original_y)
            state['orders_ready'] = True
            return banks, [{} for _ in range(25)]
        def selected(xx, yy, labels, actual_folds, inner, outer, count, progress):
            self.assertTrue(state['orders_ready'])
            np.testing.assert_array_equal(yy, original_y)
            state['decoder_calls'] += 1
            return object(), np.array([3*e+b for e in outer[:count] for b in range(3)])
        def capacity(xx, yy, *_):
            np.testing.assert_array_equal(yy, original_y)
            return object()
        outer_self = self
        class Trajectory:
            def __init__(self, *args): outer_self.assertTrue(state['orders_ready'])
            def initialize_selected(self, xx, yy, *args): np.testing.assert_array_equal(yy, original_y)
        def later(*args):
            self.assertTrue(state['orders_ready'])
            state['future_calls'] += 1
            return object(), object()
        with patch.object(experiment, 'read_trials', return_value=(x, y, 'guide')), \
             patch.object(experiment, 'relative_initial_folds', return_value=folds), \
             patch.object(experiment, 'order_bank', side_effect=orders), \
             patch.object(experiment, 'fit_direction_null', return_value=object()), \
             patch.object(experiment, 'Trajectory', Trajectory), \
             patch.object(experiment, 'fit_capacity', side_effect=capacity), \
             patch.object(experiment, 'fit_selected_rank', side_effect=selected), \
             patch.object(experiment, 'later_fit', side_effect=later):
            models, _, _, selections = experiment.fit_target('unused', source, role, records,
                {'direction_only_models': 0}, lambda: None, None)
        self.assertEqual(len(selections), 25)
        self.assertEqual(len(models), 100)
        self.assertEqual(len({id(model[-1]) for model in models}), 80)
        self.assertEqual(state['decoder_calls'], 15)
        self.assertEqual(state['future_calls'], 64)
        np.testing.assert_array_equal(y, original_y)


if __name__ == '__main__':
    unittest.main()
