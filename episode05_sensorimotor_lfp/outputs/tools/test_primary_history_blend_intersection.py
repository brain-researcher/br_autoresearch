"""Independent reduced-current fitting, saved-HW reuse and mixture-support fixtures."""
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import primary_history_blend as constant
import primary_history_blend_intersection as blend
import primary_intersection as intersection
import primary_repeatability_weight as original_w
from method_components import DirectionTime
from primary_capacity import score_bank
from primary_matched_budget import fit_view as original_hw_view
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from test_primary_intersection import description, reference_packet
from test_primary_history_blend import direct_wrong, saved_components


def reference_current_fit(selected, y, labels):
    """Independent augmented least-squares oracle, using explicit pair energy."""
    current = selected[:, 3:18, :]
    baseline = np.empty_like(y)
    predictions = np.empty((5, *y.shape))
    for held in range(8):
        train, test = labels != held, labels == held
        theta = labels*np.pi/4
        template = DirectionTime.fit(theta[train], y[train], 1, labels[train])
        residual = (y[train]-template.predict(theta[train])).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        u = residual-intercept
        base = template.predict(theta[test])+intercept
        baseline[test] = base
        mean, sd = current[train].mean(axis=(0, 1)), current[train].std(axis=(0, 1))
        z = (current-mean)/np.where(sd == 0, 1., sd)
        totals, paired = [], []
        for direction in range(8):
            if direction == held:
                continue
            block = z[labels == direction]
            totals.append(np.square(block).mean(axis=(0, 1)))
            pairs = [np.mean(block[i]*block[j], axis=0) for i in range(len(block))
                     for j in range(len(block)) if i != j]
            paired.append(np.mean(pairs, axis=0))
        total, pair = np.mean(totals, axis=0), np.mean(paired, axis=0)
        a = .1+np.clip(np.divide(pair, total, out=np.zeros_like(pair), where=total > 0), 0, 1)
        fitting = z[train].reshape(-1, current.shape[-1])
        gram = fitting.T@fitting/len(fitting)
        active, trace = np.diag(gram) > 0, float(np.trace(gram))
        weights = np.ones(len(a)) if trace == 0 or np.all(a[active] == a[active][0]) else np.sqrt(a*trace/(a@np.diag(gram)))
        phi, evaluation = fitting*weights, z[test].reshape(-1, len(a))*weights
        for index, penalty in enumerate(LAMBDAS):
            augmented = np.concatenate((phi, np.sqrt(len(phi)*penalty)*np.eye(len(a))), axis=0)
            target = np.concatenate((u, np.zeros((len(a), 2))), axis=0)
            coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
            predictions[index, test] = base+(evaluation@coefficients).reshape(base.shape)
    return baseline, predictions


def fitted_example():
    desc, records, (x, y, labels, ids) = description()
    support = desc['support_sets'][0]
    selected = x[:, :, support['electrode_positions']]
    baseline, hw, metadata = original_hw_view(selected, y, labels)
    return desc, records, (x, y, labels, ids), selected, baseline, hw, metadata


def mixture_inputs(style='ordinary'):
    desc, _, (_, y, labels, _), _, baseline, _, _ = fitted_example()
    rng = np.random.default_rng(510023)
    cache = {}
    for key in itertools.combinations(desc['original_electrode_ids'], len(desc['intersection_electrode_ids'])):
        perturbation = rng.normal(scale=.25, size=y.shape)
        if style == 'new_W':
            w = np.stack([y+perturbation*.1]*5)
            hw = np.stack([baseline]*5)
        elif style == 'old_HW':
            hw = np.stack([y+perturbation*.1]*5)
            w = np.stack([baseline]*5)
        elif style == 'one_prediction':
            w = hw = np.stack([y+.25]*5)
        else:
            w = np.stack([baseline+(i+1)*.12*(y-baseline)+perturbation for i in range(5)])
            hw = np.stack([baseline+(i+1)*.09*(y-baseline)-.7*perturbation for i in range(5)])
        cache[key] = (np.ascontiguousarray(w), np.ascontiguousarray(hw))
    full_blend = np.ascontiguousarray(baseline+.35*(y-baseline)+.75)
    return y, baseline, labels, cache, full_blend


def complete_mixture_family(style='ordinary'):
    y, baseline, labels, cache, full_blend = mixture_inputs(style)
    sessions, descriptions = {}, {}
    for session in SESSIONS:
        desc, _, _ = description(session)
        descriptions[session] = desc
        wp = [cache[tuple(s['electrode_ids'])][0] for s in desc['support_sets']]
        hp = [cache[tuple(s['electrode_ids'])][1] for s in desc['support_sets']]
        old_bank = score_bank(y, labels, baseline, hp[0], blend.OLD_RECIPES[0])
        old_index = max(range(5), key=lambda i: (old_bank['lambda_rows'][i]['delta_R2'], LAMBDAS[i]))
        sessions[session] = blend.score_views(y, baseline, labels, desc, wp, hp, old_index, full_blend)
    old_panels = {'I': {'recipe': blend.OLD_RECIPES[0], 'sessions': []},
                  'R': {'recipe': blend.OLD_RECIPES[1], 'sessions': []}}
    full_panel = {'recipe': constant.RECIPE, 'sessions': []}
    for held in SESSIONS:
        training = [s for s in SESSIONS if s != held]
        scores = [float(sum(np.mean([sessions[s]['views'][0]['HW_bank']['lambda_rows'][i]['delta_R2']
                    for s in training if ANIMALS[s] == animal]) for animal in ('Mihili', 'Chewie'))/2)
                    for i in range(5)]
        index = max(range(5), key=lambda i: (scores[i], LAMBDAS[i]))
        bank = sessions[held]
        self_support = {'session_id': held, 'animal': ANIMALS[held], 'lambda_index': index,
                        'outer_training_sessions': training}
        old_panels['I']['sessions'].append({**self_support, **bank['views'][0]['HW_bank']['lambda_rows'][index]})
        old_random = intersection.average_banks([v['HW_bank'] for v in bank['views'][1:]])
        old_panels['R']['sessions'].append({**self_support, **old_random['lambda_rows'][index],
            'selected_supports': [{'support': v['support'], **v['HW_bank']['lambda_rows'][index]}
                                  for v in bank['views'][1:]]})
        full_panel['sessions'].append({'session_id': held, 'animal': ANIMALS[held], 'lambda': 1., 'alpha': .5,
                                      **bank['full_blend_scores']})
    return sessions, old_panels, full_panel, (y, baseline, labels, cache, full_blend)


def summarize_family(inputs, valid=38, patience=5):
    return blend.summarize_banks(*inputs[:3], prior_valid=valid, prior_patience=patience,
                                prior_alternatives=19, prior_nulls=13)


def alter_interior_score(bank, index, value):
    assert index % 5 in (1, 2, 3)
    cell = bank['cells'][index]
    sse = bank['baseline_SSE']-value*bank['denominator']
    cell.update(delta_R2=value, candidate_SSE=sse, candidate_R2=1-sse/bank['denominator'],
                wrong_pair_SSE=sse+.002*bank['denominator'], identity_increment=.002)


class BlendIntersectionFitTest(unittest.TestCase):
    def test_reduced_W_exact_original_arm_and_independent_rectangular_oracle(self):
        desc, _, (_, y, labels, ids), selected, old_b, _, hw_metadata = fitted_example()
        baseline, predictions, metadata = blend.fit_current_view(selected, y, labels, hw_metadata)
        support = desc['support_sets'][0]
        banks, original_b, old_predictions = original_w.fit_banks(selected[:, 3:18, :], y, labels, ids,
            desc['session_id'], electrode_ids=support['electrode_ids'])
        np.testing.assert_array_equal(baseline, original_b)
        np.testing.assert_array_equal(baseline, old_b)
        self.assertEqual(baseline.strides, y.strides)
        np.testing.assert_array_equal(predictions, old_predictions[original_w.RECIPES[0]])
        reference_b, reference_p = reference_current_fit(selected, y, labels)
        np.testing.assert_array_equal(baseline, reference_b)
        np.testing.assert_allclose(predictions, reference_p, rtol=3e-13, atol=3e-12)
        for actual, prior, old in zip(metadata, hw_metadata, banks[original_w.RECIPES[0]]['direction_fit_metadata']):
            for key in ('held_direction', 'fitting_bin_rows', 'physical_mean', 'physical_sd', 'repeatability'):
                self.assertEqual(actual[key], prior[key])
                self.assertEqual(actual[key], old[key])
            self.assertEqual(actual['paired_weights'], old['paired_weights'])
            self.assertEqual(actual['paired_weighted_covariance'], old['paired_weighted_covariance'])

    def test_exact_saved_prior_state_failure_and_held_target_past_bin_exclusion(self):
        _, _, (_, y, labels, _), selected, _, _, hw_metadata = fitted_example()
        original = blend.fit_current_view(selected, y, labels, hw_metadata)
        held = 2
        changed_y = y.copy(order='K')
        changed_y[labels == held] += 1e4
        poisoned = blend.fit_current_view(selected, changed_y, labels, hw_metadata)
        np.testing.assert_array_equal(original[0][labels == held], poisoned[0][labels == held])
        np.testing.assert_array_equal(original[1][:, labels == held], poisoned[1][:, labels == held])
        self.assertEqual(original[2][held], poisoned[2][held])
        changed_x = selected.copy(order='K')
        changed_x[:, :3, :] += 1e5
        no_history = blend.fit_current_view(changed_x, y, labels, hw_metadata)
        np.testing.assert_array_equal(original[0], no_history[0])
        np.testing.assert_array_equal(original[1], no_history[1])
        self.assertEqual(original[2], no_history[2])
        for key in ('physical_mean', 'physical_sd', 'repeatability'):
            changed = copy.deepcopy(hw_metadata)
            if key == 'repeatability':
                changed[0][key]['T'][0] += 1e-12
            else:
                changed[0][key][0] += 1e-12
            with self.subTest(key=key), self.assertRaises(ValueError):
                blend.fit_current_view(selected, y, labels, changed)


class BlendIntersectionScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = complete_mixture_family()

    def test_all_nine_native_mixture_grids_literal_endpoints_and_actual_cross_terms(self):
        sessions, _, _, arrays = self.inputs
        y, baseline, labels, cache, _ = arrays
        bank = sessions[SESSIONS[0]]
        denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
        self.assertEqual(bank['denominator'], denominator)
        for view in bank['views']:
            w, hw = cache[tuple(view['support']['electrode_ids'])]
            for index, cell in enumerate(view['blend_bank']['cells']):
                li, ai = divmod(index, 5)
                alpha = constant.ALPHAS[ai]
                prediction = w[li] if ai == 0 else hw[li] if ai == 4 else (1-alpha)*w[li]+alpha*hw[li]
                sse = float(np.square(y-prediction).sum())
                self.assertEqual(cell['candidate_SSE'], sse)
                self.assertAlmostEqual(cell['wrong_pair_SSE'], direct_wrong(y, prediction, labels), places=10)
                cross = float(np.sum((y-w[li])*(y-hw[li])))
                self.assertEqual(view['error_cross_products_by_lambda'][li], cross)
                reconstructed = ((1-alpha)**2*view['W_bank']['lambda_rows'][li]['candidate_SSE']
                    +alpha**2*view['HW_bank']['lambda_rows'][li]['candidate_SSE']+2*alpha*(1-alpha)*cross)
                self.assertAlmostEqual(sse, reconstructed, places=8)
                if ai in (0, 4):
                    original = view['W_bank' if ai == 0 else 'HW_bank']['lambda_rows'][li]
                    self.assertEqual({key: cell[key] for key in constant.SCORE_KEYS},
                                     {key: original[key] for key in constant.SCORE_KEYS})
        # Explicit opposite errors distinguish full mixtures from endpoint-loss averages.
        self.assertLess(float(np.square(y-constant.blend_prediction(y+2, y-2, .5)).sum()), 1e-25)

    def test_random_views_average_eight_losses_and_keep_same_parameter_cell(self):
        sessions, _, _, arrays = self.inputs
        y, baseline, labels, cache, _ = arrays
        bank = sessions[SESSIONS[0]]
        for index, row in enumerate(bank['mean_random_cells']):
            li, ai = divmod(index, 5)
            predictions = []
            for view in bank['views'][1:]:
                w, hw = cache[tuple(view['support']['electrode_ids'])]
                predictions.append(constant.blend_prediction(w[li], hw[li], constant.ALPHAS[ai]))
            self.assertEqual(row['candidate_SSE'], float(np.mean([float(np.square(y-p).sum()) for p in predictions])))
            self.assertEqual(row['wrong_pair_SSE'], float(np.mean([
                v['blend_bank']['cells'][index]['wrong_pair_SSE'] for v in bank['views'][1:]])))
            self.assertEqual(row['delta_R2'], (bank['baseline_SSE']-row['candidate_SSE'])/bank['denominator'])
        mixed = np.mean([cache[tuple(v['support']['electrode_ids'])][0][0] for v in bank['views'][1:]], axis=0)
        ensemble_sse = float(np.square(y-mixed).sum())
        self.assertGreater(bank['mean_random_cells'][0]['candidate_SSE']-ensemble_sse, .01)
        with self.assertRaises(ValueError):
            blend.mean_cells(bank['views'][1:8])
        changed = copy.deepcopy(bank)
        changed['mean_random_cells'][0]['candidate_SSE'] = ensemble_sse
        with self.assertRaises(ValueError):
            blend.validate_bank(changed)

    def test_I_only_outer_exclusion_and_shared_R_lambda_alpha_despite_R_preference(self):
        changed = copy.deepcopy(self.inputs)
        held = SESSIONS[0]
        for session, bank in changed[0].items():
            ib = bank['views'][0]['blend_bank']
            ceiling = ib['baseline_SSE']/ib['denominator']
            alter_interior_score(ib, 8, .98*ceiling)
            # The held outcome prefers a different cell and may not tune itself.
            if session == held:
                alter_interior_score(ib, 23, .999*ceiling)
            for view in bank['views'][1:]:
                alter_interior_score(view['blend_bank'], 21, .999*ceiling)
            bank['mean_random_cells'] = blend.mean_cells(bank['views'][1:])
        result = summarize_family(changed)
        selected = {row['session_id']: row for row in result['panels'][0]['sessions']}
        self.assertEqual(selected[held]['cell_index'], 8)
        for row in result['panels'][1]['sessions']:
            candidate = selected[row['session_id']]
            self.assertEqual((row['lambda'], row['alpha']), (candidate['lambda'], candidate['alpha']))
            self.assertEqual(row['outer_training_sessions'], [s for s in SESSIONS if s != row['session_id']])
            for draw in row['selected_supports']:
                self.assertEqual((draw['lambda'], draw['alpha']), (candidate['lambda'], candidate['alpha']))
        choices = {s: changed[0][s]['views'][0]['blend_bank'] for s in SESSIONS}
        self.assertIs(blend.select_cell, constant.select_cell)
        # Exercise both historical exact-tie preferences and equal-animal weighting.
        for s, bank in choices.items():
            for cell in bank['cells']:
                cell['delta_R2'] = 0.
        self.assertEqual(blend.select_cell(choices, held)[0], 20)
        for s, bank in choices.items():
            bank['cells'][0]['delta_R2'] = 2. if ANIMALS[s] == 'Mihili' else 0.
            bank['cells'][1]['delta_R2'] = 0. if ANIMALS[s] == 'Mihili' else 1.5
        index, _, scores = blend.select_cell(choices, held)
        self.assertEqual(index, 0)
        self.assertEqual(scores[:2], [1., .75])

    def test_previous_I_R_penalty_selectors_every_draw_and_full_anchor_replay(self):
        for damage in ('old_I_score', 'old_I_training', 'old_R_lambda', 'old_R_draw', 'old_R_support', 'full_score'):
            changed = copy.deepcopy(self.inputs)
            if damage == 'old_I_score':
                changed[1]['I']['sessions'][0]['candidate_SSE'] += 1e-8
            elif damage == 'old_I_training':
                changed[1]['I']['sessions'][0]['outer_training_sessions'][0] = SESSIONS[0]
            elif damage == 'old_R_lambda':
                changed[1]['R']['sessions'][0]['lambda_index'] = (changed[1]['R']['sessions'][0]['lambda_index']+1) % 5
            elif damage == 'old_R_draw':
                changed[1]['R']['sessions'][0]['selected_supports'][7]['wrong_pair_SSE'] += 1e-8
            elif damage == 'old_R_support':
                # Replace the entire record to avoid mutating the intentionally shared frozen support object.
                changed[1]['R']['sessions'][0]['selected_supports'][7]['support'] = {'changed': True}
            else:
                changed[2]['sessions'][0]['candidate_SSE'] += 1e-8
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                summarize_family(changed)

    def test_new_reduced_W_endpoint_and_old_HW_endpoint_have_different_charges(self):
        new = complete_mixture_family('new_W')
        new_result = summarize_family(new)
        self.assertTrue(all(row['alpha'] == 0 for row in new_result['panels'][0]['sessions']))
        self.assertEqual(new_result['scientific_trial_count_by_recipe'], dict.fromkeys(blend.RECIPES, 1))
        self.assertEqual(new_result['literal_null_control_count_increment'], 0)
        self.assertEqual(new_result['alternative_explanation_count_increment'], 2)
        self.assertFalse(new_result['incumbent_reset'])
        self.assertFalse(new_result['eligible_for_comparable_primary_incumbent'])
        self.assertEqual([row['completed_count'] for row in new_result['count_transitions']], [39, 40])
        self.assertEqual([row['patience'] for row in new_result['count_transitions']], [6, 7])
        old = summarize_family(complete_mixture_family('old_HW'))
        self.assertTrue(all(row['alpha'] == 1 for row in old['panels'][0]['sessions']))
        self.assertEqual(old['scientific_trial_count_increment'], 0)
        self.assertIn('original_I_HW', old['structural_duplicate_of'][blend.RECIPES[0]])
        self.assertIn('original_R_HW', old['structural_duplicate_of'][blend.RECIPES[1]])
        self.assertEqual(old['patience_after'], 5)
        with self.assertRaises(ValueError):
            summarize_family(new, valid=63)

    def test_independent_global_zero_one_two_counts_and_literal_changed_lambda_flags(self):
        base = copy.deepcopy(self.inputs)
        # Test the global accounting contract independently of prediction arithmetic.
        def flags(i_duplicate, r_duplicate, r_equals_i=False):
            changed = copy.deepcopy(base)
            for bank in changed[0].values():
                for vi, view in enumerate(bank['views']):
                    for cell in view['blend_bank']['cells']:
                        cell['exact_reuse'] = {'prior_I_HW': bool(i_duplicate and vi == 0),
                            'full_blend': False, 'prior_R_HW_draws': [bool(r_duplicate and vi == r+1) for r in range(8)],
                            'current_I_blend': bool(vi == 0 or r_equals_i)}
            return changed
        for i_dup, r_dup, expected in ((False, False, [1, 1]), (True, False, [0, 1]),
                                      (False, True, [1, 0]), (True, True, [0, 0])):
            result = summarize_family(flags(i_dup, r_dup))
            self.assertEqual([result['scientific_trial_count_by_recipe'][r] for r in blend.RECIPES], expected)
        result = summarize_family(flags(False, False, True))
        self.assertEqual(result['scientific_trial_count_by_recipe'], {blend.RECIPES[0]: 1, blend.RECIPES[1]: 0})
        # Exact array checks, not an alpha token, detect equality to an old-lambda HW endpoint.
        y, baseline, labels, cache, full = base[3]
        desc, _, _ = description()
        wp = [cache[tuple(s['electrode_ids'])][0] for s in desc['support_sets']]
        hp = [cache[tuple(s['electrode_ids'])][1] for s in desc['support_sets']]
        bank = blend.score_views(y, baseline, labels, desc, wp, hp, 0, full)
        self.assertTrue(bank['views'][0]['blend_bank']['cells'][4]['exact_reuse']['prior_I_HW'])
        self.assertFalse(bank['views'][0]['blend_bank']['cells'][24]['exact_reuse']['prior_I_HW'])
        for r, view in enumerate(bank['views'][1:]):
            self.assertTrue(view['blend_bank']['cells'][4]['exact_reuse']['prior_R_HW_draws'][r])
            self.assertFalse(view['blend_bank']['cells'][24]['exact_reuse']['prior_R_HW_draws'][r])
        mixed = flags(False, False)
        for session_index, bank in enumerate(mixed[0].values()):
            for cell in bank['views'][0]['blend_bank']['cells']:
                cell['exact_reuse']['prior_I_HW' if session_index % 2 else 'full_blend'] = True
        self.assertEqual(summarize_family(mixed)['scientific_trial_count_by_recipe'][blend.RECIPES[0]], 1)


def saved_reduced_packet(directory):
    """Real original producer shapes and fit-key archives, all from tiny arrays."""
    import run_primary_history_blend_intersection as runner
    desc, records, (x, y, labels, ids) = description()
    session = desc['session_id']
    reference = reference_packet(x, y, labels, ids)
    old_fits, old_routes = intersection.fit_supports(x, y, labels, ids, desc)
    banks, payload = intersection.finalize_banks(y, labels, ids, desc, old_fits, old_routes, reference)
    full_w_banks, baseline, full_w_predictions = original_w.fit_banks(x[:, 3:18, :], y, labels, ids,
        session, electrode_ids=desc['original_electrode_ids'])
    full_w = full_w_predictions[original_w.RECIPES[0]]
    full_hw = reference['candidate_xy_by_lambda']
    roots, _ = saved_components(directory, (y, baseline, labels, full_w, full_hw))
    for name, metadata in (('W', full_w_banks[original_w.RECIPES[0]]['direction_fit_metadata']),
                           ('HW', reference['fold_metadata'])):
        path = roots[name]/session/'score_banks.json'
        packet = json.loads(path.read_text())
        packet['source']['common_LMP_electrodes'] = desc['original_electrode_ids']
        packet['banks'][constant.COMPONENTS[name]].update(
            physical_electrode_ids=desc['original_electrode_ids'], direction_fit_metadata=metadata)
        path.write_text(json.dumps(packet))
    args = SimpleNamespace(w_bank_root=roots['W'], hw_bank_root=roots['HW'],
        intersection_bank_root=Path(directory)/'intersection', bank_root=Path(directory)/'new_blend')
    folder = args.intersection_bank_root/session
    folder.mkdir(parents=True)
    source = {'session_id': session, 'common_LMP_electrodes': desc['original_electrode_ids'],
        'complete_trials': len(y), 'native_LMP_offsets': [-3, 14], 'native_target_offsets': [0, 14],
        'lag_offsets': [0, -1, -2, -3], 'native_bin_seconds': .03, 'spike_values_read': False}
    old_packet = {'status': 'complete_intersection_sensitivity_session_banks', 'session_id': session,
        'global_metadata_qualified_before_source': True, 'exact_original_HW_target_baseline_trial_score_replay': True,
        'reference_refitted': False, 'sensitivity_only': True, 'promotion_authorized': False,
        'qualified_support': desc, 'source': source, 'banks': banks,
        'new_fit_metadata': {key: {field: value[field] for field in ('metadata', 'electrode_ids')}
                             for key, value in old_fits.items()}, 'new_unique_fit_count': len(old_fits)}
    (folder/'score_banks.json').write_text(json.dumps(old_packet))
    np.savez(folder/'new_fitted_predictions.npz', actual_xy=np.ascontiguousarray(y),
             directions=labels, trial_ids=ids, **payload)
    old_blend = constant.score_session(y, baseline, labels, full_w, full_hw)
    index = max(range(25), key=lambda i: (old_blend['cells'][i]['delta_R2'], LAMBDAS[i//5], -constant.ALPHAS[i%5]))
    row = {'session_id': session, 'animal': ANIMALS[session], **old_blend['cells'][index]}
    blend_path = Path(directory)/'selected_full_blend.npz'
    np.savez(blend_path, actual_xy=np.ascontiguousarray(y), baseline_xy=np.ascontiguousarray(baseline),
        directions=labels, trial_ids=ids,
        candidate_xy=constant.blend_prediction(full_w[row['lambda_index']], full_hw[row['lambda_index']], row['alpha']),
        selected_lambda=row['lambda'], selected_alpha=row['alpha'])
    old_index = max(range(5), key=lambda i: (banks[blend.OLD_RECIPES[0]]['lambda_rows'][i]['delta_R2'], LAMBDAS[i]))
    context = {'descriptions': {session: desc},
        'old_views': {session: banks[blend.OLD_RECIPES[0]]['views']+banks[blend.OLD_RECIPES[1]]['views']},
        'blend_rows': {session: row}, 'blend_report': {'selected_prediction_paths': {session: str(blend_path)}},
        'old_hw_rows': {'I': {session: {'lambda_index': old_index}}}}
    return args, context, desc, (x, y, labels, ids), old_packet


class BlendIntersectionRunnerTest(unittest.TestCase):
    def test_actual_saved_nine_HW_views_and_channel_only_cached_W_fit(self):
        import run_primary_history_blend_intersection as runner
        with tempfile.TemporaryDirectory() as directory:
            args, context, desc, (x, y, labels, ids), old_packet = saved_reduced_packet(directory)
            session = desc['session_id']
            with patch.object(runner, 'load_lmp_history_session', side_effect=AssertionError('Saved replay cannot read source')):
                saved = runner.load_saved_views(args, session, context)
            np.testing.assert_array_equal(saved['y'], y)
            np.testing.assert_array_equal(saved['ids'], ids)
            self.assertEqual(saved['y'].strides, y.strides)
            self.assertEqual(saved['baseline'].strides, y.strides)
            self.assertEqual(len(saved['HW_predictions']), 9)
            seen = []
            def inspect_fit(selected, actual, directions, metadata, check):
                self.assertIs(actual, y)
                self.assertIs(directions, labels)
                matches = [s for s in desc['support_sets'] if np.array_equal(selected, x[:, :, s['electrode_positions']])]
                self.assertTrue(matches)
                self.assertEqual(selected.strides, x[:, :, matches[0]['electrode_positions']].strides)
                seen.append(tuple(matches[0]['electrode_ids']))
                return blend.fit_current_view(selected, actual, directions, metadata, check)
            with patch.object(runner, 'fit_current_view', side_effect=inspect_fit):
                fits, routes, predictions, metadata = runner.fit_reduced_W(x, y, labels, desc, saved)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(len(fits), desc['unique_new_fit_supports'])
            for i, left in enumerate(desc['support_sets']):
                for j, right in enumerate(desc['support_sets']):
                    if left['electrode_ids'] == right['electrode_ids']:
                        self.assertEqual(routes[i], routes[j])
                        self.assertIs(predictions[i], predictions[j])
                        self.assertIs(saved['HW_predictions'][i], saved['HW_predictions'][j])
            # Full physical views reuse the two immutable original components without a fit.
            full_desc, _, _ = description(full=True)
            full_saved = {**saved, 'HW_metadata': [saved['full_banks']['HW']['direction_fit_metadata']]*9}
            with patch.object(runner, 'fit_current_view', side_effect=AssertionError('Full W must not be refit')):
                full = runner.fit_reduced_W(x, y, labels, full_desc, full_saved)
            self.assertEqual(full[0], {})
            self.assertEqual(full[1], ['full_W']*9)
            self.assertTrue(all(p is saved['full_components']['W'] for p in full[2]))
            # Damage a prediction in every distinct reused HW support; all-five replay is compulsory.
            path = args.intersection_bank_root/session/'new_fitted_predictions.npz'
            with np.load(path, allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            for key in old_packet['new_fit_metadata']:
                changed = {name: value.copy() for name, value in original.items()}
                changed[key+'_candidate_xy_by_lambda'][4, 0, 0, 0] += 1e-4
                np.savez(path, **changed)
                with self.subTest(fit_key=key), self.assertRaises(ValueError):
                    runner.load_saved_views(args, session, context)
            np.savez(path, **original)

    def test_complete_new_packet_saved_mixture_replay_and_precise_failure_boundaries(self):
        import run_primary_history_blend_intersection as runner
        with tempfile.TemporaryDirectory() as directory:
            args, context, desc, (x, y, labels, ids), old_packet = saved_reduced_packet(directory)
            session = desc['session_id']
            saved = runner.load_saved_views(args, session, context)
            fits, routes, predictions, metadata = runner.fit_reduced_W(x, y, labels, desc, saved)
            bank = blend.score_views(y, saved['baseline'], labels, desc, predictions, saved['HW_predictions'],
                context['old_hw_rows']['I'][session]['lambda_index'], saved['full_blend'])
            runner.attach_metadata(bank, routes, metadata, saved['HW_metadata'])
            item = {'status': runner.SESSION_STATUS, 'session_id': session, 'recipes': list(blend.RECIPES),
                'bank': bank, 'qualified_support': desc, 'source': old_packet['source'],
                'original_HW_refitted': False, 'full_W_refitted': False, 'sensitivity_only': True,
                'promotion_authorized': False, 'scientific_trial_count_increment': 0,
                'literal_null_control_count_increment': 0, 'alternative_explanation_count_increment': 0,
                'effective_W_offsets': [0, 14], 'W_reader_offsets': [-3, 14],
                'new_unique_W_fit_count': len(fits), 'new_W_ridge_heads': 40*len(fits),
                'new_W_fit_metadata': {key: {name: fit[name] for name in ('metadata', 'electrode_ids')}
                                      for key, fit in fits.items()}, **dict.fromkeys(runner.REPLAY_FLAGS, True)}
            runner.validate_session(item, session)
            folder = args.bank_root/session
            folder.mkdir(parents=True)
            payload = {'actual_xy': np.ascontiguousarray(y), 'baseline_xy': np.ascontiguousarray(saved['baseline']),
                'directions': labels, 'trial_ids': ids,
                **{key+'_candidate_xy_by_lambda': fit['predictions'] for key, fit in fits.items()}}
            np.savez(folder/'new_W_predictions.npz', **payload)
            restored, wp = runner.replay_new_session(args, session, context, item)
            self.assertEqual(len(wp), 9)
            np.testing.assert_array_equal(restored['y'], y)
            for mutate in (
                lambda obj: obj.update(new_W_ridge_heads=obj['new_W_ridge_heads']+1),
                lambda obj: obj['source'].update(spike_values_read=True),
                lambda obj: obj['bank']['mean_random_cells'][2].update(candidate_SSE=1e50),
                lambda obj: obj['bank']['views'][0]['W_fit_metadata'][0]['physical_mean'].__setitem__(0, 1e50),
            ):
                changed = copy.deepcopy(item)
                mutate(changed)
                with self.assertRaises(ValueError):
                    runner.validate_session(changed, session)
            for field in ('error_cross_products_by_lambda', 'exact_reuse'):
                changed = copy.deepcopy(item)
                if field == 'exact_reuse':
                    flags = changed['bank']['views'][8]['blend_bank']['cells'][3][field]
                    flags['full_blend'] = not flags['full_blend']
                else:
                    changed['bank']['views'][8][field][2] += 1e-6
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.replay_new_session(args, session, context, changed)
            damaged = {key: value.copy() for key, value in payload.items()}
            damaged['trial_ids'][0] = damaged['trial_ids'][1]
            np.savez(folder/'new_W_predictions.npz', **damaged)
            with self.assertRaises(ValueError):
                runner.replay_new_session(args, session, context, item)


if __name__ == '__main__':
    unittest.main()
