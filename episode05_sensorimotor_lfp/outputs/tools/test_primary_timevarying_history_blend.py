"""Independent finite-schedule arithmetic, nesting, replay and identity fixtures."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import primary_history_blend as constant
import primary_timevarying_history_blend as varying
from primary_capacity import score_bank
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from test_primary_history_blend import example, time_fast, direct_wrong, complete_family, saved_components


def independent_selected(banks, recipe):
    rows = []
    for held in SESSIONS:
        fitting = [s for s in SESSIONS if s != held]
        scores = [float(sum(np.mean([banks[s]['lambda_rows'][i]['delta_R2']
                    for s in fitting if ANIMALS[s] == animal])
                    for animal in ('Mihili', 'Chewie'))/2) for i in range(5)]
        index = max(range(5), key=lambda i: (scores[i], LAMBDAS[i]))
        rows.append({'session_id': held, 'animal': ANIMALS[held], 'lambda_index': index,
            'outer_training_sessions': fitting, **banks[held]['lambda_rows'][index]})
    return {'recipe': recipe, 'sessions': rows}


def family(values=None, h2=None):
    y, baseline, labels, w, hw = example(509390) if values is None else values
    old, anchors, panels = complete_family(y, baseline, labels, w, hw)
    panels['blend'] = {'recipe': constant.RECIPE,
        **constant.summarize_banks(old, anchors, panels)['blend']}
    h2 = np.ascontiguousarray(.7*w+.3*hw) if h2 is None else h2
    anchors['H2'] = {s: score_bank(y, labels, baseline, h2, varying.ANCHORS['H2']) for s in SESSIONS}
    panels['H2'] = independent_selected(anchors['H2'], varying.ANCHORS['H2'])
    originals = {name: {row['session_id']: row for row in panel['sessions']} for name, panel in panels.items()}
    sessions = {}
    for s in SESSIONS:
        old_row = originals['blend'][s]
        original_predictions = {name: arrays[originals[name][s]['lambda_index']]
            for name, arrays in (('W', w), ('HW', hw), ('H2', h2))}
        original_predictions['blend'] = constant.blend_prediction(
            w[old_row['lambda_index']], hw[old_row['lambda_index']], old_row['alpha'])
        sessions[s] = varying.score_session(y, baseline, labels, w, hw, original_predictions)
    return sessions, anchors, panels, old


class TimevaryingBlendTest(unittest.TestCase):
    def test_literal_endpoints_zero_slope_and_unrecentered_schedule(self):
        y, _, _, w, hw = example(509391)
        left, right = w[0].copy(), hw[0].copy()
        left[0, 0] = [-0., 0.]
        right[0, 14] = [-0., 0.]
        left[1, 0] = [1e150, -1e150]
        right[1, 14] = [-1e150, 1e150]
        for slope in varying.SLOPES:
            alpha = .5+slope*(2*np.arange(15, dtype=np.float64)/14-1)
            np.testing.assert_array_equal(varying.alpha_schedule(slope), alpha)
            self.assertTrue(np.all((alpha >= 0) & (alpha <= 1)))
            prediction = varying.timevarying_prediction(left, right, slope)
            expected = (1-alpha[None, :, None])*left+alpha[None, :, None]*right
            for t in range(15):
                if alpha[t] in (0., 1.):
                    endpoint = left if alpha[t] == 0 else right
                    np.testing.assert_array_equal(prediction[:, t].view(np.uint64), endpoint[:, t].view(np.uint64))
                else:
                    np.testing.assert_array_equal(prediction[:, t], expected[:, t])
        with patch.object(varying, 'blend_prediction', wraps=constant.blend_prediction) as old:
            zero = varying.timevarying_prediction(left, right, 0.)
            old.assert_called_once_with(left, right, .5)
        np.testing.assert_array_equal(zero.view(np.uint64), constant.blend_prediction(left, right, .5).view(np.uint64))
        for illegal in (.1, -.6, np.nan):
            with self.assertRaises(ValueError):
                varying.timevarying_prediction(left, right, illegal)
        with self.assertRaises(ValueError):
            varying.timevarying_prediction(left.astype('float32'), right, 0.)

    def test_all_cells_joint_native_XY_cross_terms_and_exact_anchor_flags(self):
        y, baseline, labels, w, hw = example(509392)
        originals = {'W': w[1], 'HW': hw[4], 'blend': constant.blend_prediction(w[2], hw[2], .5),
                     'H2': varying.timevarying_prediction(w[3], hw[3], -.25)}
        bank = varying.score_session(y, baseline, labels, w, hw, originals)
        denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
        baseline_sse = float(np.square(y-baseline).sum())
        for index, cell in enumerate(bank['cells']):
            li, si = divmod(index, 5)
            slope = varying.SLOPES[si]
            alpha = .5+slope*(2*np.arange(15, dtype=np.float64)/14-1)
            prediction = (1-alpha[None, :, None])*w[li]+alpha[None, :, None]*hw[li]
            sse = float(np.square(y-prediction).sum())
            wrong = direct_wrong(y, prediction, labels)
            self.assertEqual((cell['lambda'], cell['slope']), (LAMBDAS[li], slope))
            self.assertEqual(cell['candidate_SSE'], sse)
            self.assertAlmostEqual(cell['wrong_pair_SSE'], wrong, places=10)
            self.assertEqual(cell['delta_R2'], (baseline_sse-sse)/denominator)
            self.assertAlmostEqual(cell['identity_increment'], (wrong-sse)/denominator, places=13)
            self.assertEqual(cell['exact_prediction_equals_originals'],
                {name: bool(np.array_equal(prediction, original)) for name, original in originals.items()})
        # Opposite prediction errors cancel; averaging endpoint losses cannot reproduce this.
        left, right = y+2., y-2.
        prediction = varying.timevarying_prediction(left, right, 0.)
        self.assertLess(float(np.square(y-prediction).sum()), 1e-25)
        self.assertGreater(float((np.square(y-left)+np.square(y-right)).sum()/2), 100.)
        self.assertFalse(np.array_equal(prediction, baseline+prediction))
        # Near equality is never literal equality for any of the four anchors.
        near = {name: prediction+1e-9 for name in varying.ANCHORS}
        bank = varying.score_session(y, baseline, labels, np.stack([left]*5), np.stack([right]*5), near)
        self.assertTrue(np.allclose(prediction, near['H2']))
        self.assertFalse(any(bank['cells'][2]['exact_prediction_equals_originals'].values()))

    def test_outer_exclusion_all_tie_levels_and_no_near_tie_tolerance(self):
        sessions = family()[0]
        held = SESSIONS[0]
        for bank in sessions.values():
            for cell in bank['cells']:
                cell['delta_R2'] = 0.
        for index, cell in enumerate(sessions[held]['cells']):
            cell['delta_R2'] = 1e9*(25-index)
        selected, training, scores = varying.select_cell(sessions, held)
        self.assertEqual(selected, 22)  # lambda10, then |slope|=0
        self.assertEqual(training, [s for s in SESSIONS if s != held])
        self.assertEqual(scores, [0.]*25)
        for session in training:
            for index in (20, 21, 23, 24, 17):
                sessions[session]['cells'][index]['delta_R2'] = 1.
        self.assertEqual(varying.select_cell(sessions, held)[0], 21)  # -.25 beats +.25 and +/-.5
        for session in training:
            sessions[session]['cells'][4]['delta_R2'] = 1.+1e-10
        self.assertEqual(varying.select_cell(sessions, held)[0], 4)  # exact greater beats every tie preference

    def test_outer_equal_animal_weighting_is_not_pooled_session_weighting(self):
        sessions = family()[0]
        held = SESSIONS[0]
        self.assertEqual(ANIMALS[held], 'Mihili')
        for session, bank in sessions.items():
            for cell in bank['cells']:
                cell['delta_R2'] = -10.
            bank['cells'][0]['delta_R2'] = 2. if ANIMALS[session] == 'Mihili' else 0.
            bank['cells'][1]['delta_R2'] = 0. if ANIMALS[session] == 'Mihili' else 1.5
        selected, _, scores = varying.select_cell(sessions, held)
        self.assertEqual(selected, 0)
        self.assertEqual(scores[:2], [1., .75])

    def test_reversal_uses_selected_lambda_and_zero_slope_is_exact_duplicate(self):
        y, baseline, labels, _, _ = example(509393)
        alpha = np.arange(15, dtype=np.float64)/14
        w = np.stack([y+alpha[None, :, None]*2.]*5)
        hw = np.stack([y-(1-alpha[None, :, None])*2.]*5)
        inputs = family((y, baseline, labels, w, hw))
        result = varying.summarize_banks(*inputs)
        selected = result['candidate']['sessions']
        self.assertTrue(all(row['slope'] == .5 and row['lambda'] == 10. for row in selected))
        reverse = {row['session_id']: row for row in result['reversed_slope']['sessions']}
        contrasts = {row['session_id']: row for row in result['selected_minus_reversed']['sessions']}
        for row in selected:
            session = row['session_id']
            reference = inputs[0][session]['cells'][20]
            self.assertEqual(reverse[session]['lambda'], row['lambda'])
            self.assertEqual(reverse[session]['slope'], -.5)
            self.assertEqual(reverse[session]['candidate_SSE'], reference['candidate_SSE'])
            self.assertEqual(contrasts[session]['delta_R2'],
                (reference['candidate_SSE']-row['candidate_SSE'])/inputs[0][session]['denominator'])
        self.assertEqual(result['reversal_scientific_trial_count_increment'], 0)
        identical = np.stack([y]*5)
        zero_result = varying.summarize_banks(*family((y, baseline, labels, identical, identical), identical))
        for row in zero_result['selected_minus_reversed']['sessions']:
            self.assertEqual(row['slope'], 0.)
            self.assertEqual(row['delta_R2'], 0.)
            self.assertEqual(row['identity_increment'], 0.)

    def test_global_four_anchor_duplicates_independent_and_mixed_sessions_do_not_collapse(self):
        inputs = family()
        # Equality generation is checked above; isolate global accounting from score selection here.
        for name in varying.ANCHORS:
            changed = copy.deepcopy(inputs)
            for bank in changed[0].values():
                for cell in bank['cells']:
                    cell['exact_prediction_equals_originals'] = {key: key == name for key in varying.ANCHORS}
            result = varying.summarize_banks(*changed)
            self.assertEqual(result['structural_duplicate_of'], [name])
            self.assertEqual(result['scientific_trial_count_increment'], 0)
            self.assertFalse(result['eligible_for_comparable_primary_incumbent'])
        changed = copy.deepcopy(inputs)
        for index, bank in enumerate(changed[0].values()):
            for cell in bank['cells']:
                cell['exact_prediction_equals_originals'] = {key: key == tuple(varying.ANCHORS)[index % 4]
                                                            for key in varying.ANCHORS}
        result = varying.summarize_banks(*changed)
        self.assertEqual(result['structural_duplicate_of'], [])
        self.assertEqual(result['scientific_trial_count_increment'], 1)
        self.assertEqual(result['literal_null_control_count_increment'], 0)
        self.assertEqual(result['alternative_explanation_count_increment'], 0)
        self.assertEqual(result['neural_refits'], 0)
        self.assertFalse(result['promotion_authorized'])
        y, baseline, labels, _, _ = example(509394)
        same = np.stack([y]*5)
        result = varying.summarize_banks(*family((y, baseline, labels, same, same), same))
        self.assertEqual(result['structural_duplicate_of'], list(varying.ANCHORS))
        self.assertEqual(result['scientific_trial_count_increment'], 0)

    def test_all_seven_anchor_contrasts_and_original_half_blend_replay(self):
        inputs = family()
        sessions, anchors, panels, old = inputs
        result = varying.summarize_banks(*inputs)
        expected_keys = {name+'_independently_selected' for name in varying.ANCHORS}
        expected_keys |= {'W_schedule_lambda_matched', 'HW_schedule_lambda_matched', 'half_blend_schedule_lambda_matched'}
        self.assertEqual(set(result['reference_contrasts']), expected_keys)
        original = {name: {row['session_id']: row for row in p['sessions']} for name, p in panels.items()}
        for row in result['candidate']['sessions']:
            s, li = row['session_id'], row['lambda_index']
            references = {name+'_independently_selected': original[name][s] for name in varying.ANCHORS}
            references.update({name+'_schedule_lambda_matched': anchors[name][s]['lambda_rows'][li] for name in ('W', 'HW')})
            references['half_blend_schedule_lambda_matched'] = old[s]['cells'][5*li+2]
            for key, reference in references.items():
                contrast = next(r for r in result['reference_contrasts'][key]['sessions'] if r['session_id'] == s)
                self.assertEqual(contrast['delta_R2'], (reference['candidate_SSE']-row['candidate_SSE'])/sessions[s]['denominator'])
        for li in range(5):
            changed = copy.deepcopy(inputs)
            changed[0][SESSIONS[0]]['cells'][5*li+2]['candidate_SSE'] += 1e-8
            with self.assertRaisesRegex(ValueError, 'half-blend'):
                varying.summarize_banks(*changed)

    def test_original_four_selectors_scores_exclusions_and_support_fail_closed(self):
        inputs = family()
        for name in varying.ANCHORS:
            for field in ('candidate_SSE', 'outer_training_sessions'):
                changed = copy.deepcopy(inputs)
                row = changed[2][name]['sessions'][0]
                if field == 'candidate_SSE':
                    row[field] += 1e-8
                else:
                    row[field] = [row['session_id'], *row[field][1:]]
                with self.assertRaises(ValueError):
                    varying.summarize_banks(*changed)
        changed = copy.deepcopy(inputs)
        changed[2]['blend']['sessions'][0]['training_primary_scores'][0] += 1e-8
        with self.assertRaises(ValueError):
            varying.summarize_banks(*changed)
        changed = copy.deepcopy(inputs)
        changed[0][SESSIONS[0]]['denominator'] += 1e-8
        with self.assertRaises(ValueError):
            varying.summarize_banks(*changed)


def saved_anchor_family(directory):
    """Real small NPZ arrays plus 68 metadata packets; no provider source."""
    import run_primary_timevarying_history_blend as runner
    values = example(509395)
    y, baseline, labels, w, hw = values
    roots, ids = saved_components(directory, values)
    _, banks, panels, blends = family(values)
    args = SimpleNamespace(**{name+'_bank_root': roots[name.upper()] for name in ('w', 'hw')},
        blend_bank_root=Path(directory)/'blend', h2_bank_root=Path(directory)/'H2',
        **{name+'_summary': Path(directory)/(name+'_summary.json') for name in ('w', 'hw', 'blend', 'h2')})
    packets = {}
    h2 = np.ascontiguousarray(.7*w+.3*hw)
    selected_paths = {}
    for session in SESSIONS:
        session_ids = [f'{session}_trial{i+1:06d}' for i in range(len(y))]
        for name in ('W', 'HW'):
            original = json.loads((roots[name]/SESSIONS[0]/'score_banks.json').read_text())
            original['session_id'] = session
            original['source']['session_id'] = session
            banks[name][session].update(physical_electrode_ids=[1, 3, 7], physical_electrode_count=3,
                                        native_XY_scaled=False)
            original['banks'] = {varying.ANCHORS[name]: banks[name][session]}
            packets[getattr(args, name.lower()+'_bank_root')/session/'score_banks.json'] = original
        banks['H2'][session].update(physical_electrode_ids=[1, 3, 7], physical_electrode_count=3,
            native_XY_scaled=False, reader_LMP_offsets=[-3, 14], effective_LMP_offsets=[-1, 14],
            lag_offsets=[0, -1], original_trial_ids=session_ids, original_direction_labels=labels.tolist())
        source = {'session_id': session, 'native_bin_seconds': .03, 'native_LMP_offsets': [-3, 14],
            'native_target_offsets': [0, 14], 'lag_offsets': [0, -1, -2, -3], 'complete_trials': len(y),
            'common_LMP_electrodes': [1, 3, 7]}
        packets[args.h2_bank_root/session/'score_banks.json'] = {
            'status': 'complete_history2_repeatability_session_banks', 'session_id': session,
            'source': source, 'banks': {varying.ANCHORS['H2']: banks['H2'][session]},
            'original_HW_refitted': False, 'original_W_refitted': False,
            **dict.fromkeys(runner.H2_REPLAY_FLAGS, True)}
        blends[session]['component_score_rows'] = {name: banks[name][session]['lambda_rows'] for name in ('W', 'HW')}
        packets[args.blend_bank_root/session/'score_banks.json'] = {
            'status': 'complete_history_blend_session_bank', 'session_id': session, 'bank': blends[session],
            'source': {'session_id': session, 'native_target_offsets': [0, 14], 'native_bin_seconds': .03,
                'provider_source_values_read': False, 'component_input_offsets': {'W': [0, 14], 'HW': [-3, 14]},
                'complete_trials': len(y)},
            'exact_original_component_arrays_support_replay': True, 'exact_original_component_scores_replay': True,
            'provider_source_values_read': False, 'neural_refits': 0, 'scientific_trial_count_increment': 0}
        selected_paths[session] = str(Path(directory)/'selected_blend'/(session+'.npz'))
    for name in ('W', 'HW'):
        packets[getattr(args, name.lower()+'_summary')] = {
            'status': runner.SUMMARY_STATUSES[name], 'panels': [panels[name]]}
    packets[args.h2_summary] = {'status': 'complete_history2_repeatability_panel',
        'recipes': [varying.ANCHORS['H2']], 'candidate': panels['H2'],
        'all_original_replays': True, 'all_new_saved_scores_replayed': True,
        'original_selectors_independently_replayed': True, 'original_HW_refitted': False, 'original_W_refitted': False}
    packets[args.blend_summary] = {'status': 'complete_history_blend_panel', 'recipe': constant.RECIPE,
        'blend': panels['blend'], 'selected_prediction_paths': selected_paths,
        'original_selectors_independently_replayed': True, 'all_selected_prediction_scores_replayed': True,
        'all_original_component_arrays_support_replay': True, 'all_original_component_scores_replay': True}
    session = SESSIONS[0]
    h2_path = args.h2_bank_root/session/(varying.ANCHORS['H2']+'_predictions.npz')
    h2_path.parent.mkdir(parents=True)
    np.savez(h2_path, actual_xy=np.ascontiguousarray(y), baseline_xy=np.ascontiguousarray(baseline),
        directions=labels, trial_ids=ids, candidate_xy_by_lambda=h2)
    row = next(row for row in panels['blend']['sessions'] if row['session_id'] == session)
    prediction = constant.blend_prediction(w[row['lambda_index']], hw[row['lambda_index']], row['alpha'])
    blend_path = Path(selected_paths[session])
    blend_path.parent.mkdir(parents=True)
    np.savez(blend_path, actual_xy=np.ascontiguousarray(y), baseline_xy=np.ascontiguousarray(baseline),
        directions=labels, trial_ids=ids, candidate_xy=prediction,
        selected_lambda=row['lambda'], selected_alpha=row['alpha'])
    return args, packets, values, ids, h2_path, blend_path


class TimevaryingRunnerTest(unittest.TestCase):
    def test_real_saved_anchor_loader_restores_target_layout_and_exact_four_originals(self):
        import run_primary_timevarying_history_blend as runner
        with tempfile.TemporaryDirectory() as directory:
            args, packets, values, ids, _, _ = saved_anchor_family(directory)
            with patch.object(runner, 'read_json', side_effect=lambda path: copy.deepcopy(packets[Path(path)])) as read:
                context = runner.load_anchor_context(args)
            self.assertEqual(read.call_count, 68)
            actual = runner.load_session_anchors(args, SESSIONS[0], context)
            y, baseline, labels, actual_ids, components, originals, source = actual
            for new, old in zip((y, baseline, labels, actual_ids), (*values[:3], ids)):
                np.testing.assert_array_equal(new, old)
            self.assertEqual(y.strides, values[0].strides)
            self.assertEqual(baseline.strides, values[1].strides)
            self.assertFalse(source['provider_source_values_read'])
            self.assertEqual(source['anchor_effective_input_offsets']['H2'], [-1, 14])
            self.assertEqual(set(originals), set(varying.ANCHORS))
            for prediction in originals.values():
                self.assertTrue(prediction.flags.c_contiguous)
            bank = varying.score_session(y, baseline, labels, components['W'], components['HW'], originals)
            runner.attach_references(bank, SESSIONS[0], context)
            item = {'status': runner.SESSION_STATUS, 'session_id': SESSIONS[0], 'source': source, 'bank': bank,
                'provider_source_values_read': False, 'neural_refits': 0, 'scientific_trial_count_increment': 0,
                'literal_null_control_count_increment': 0, 'alternative_explanation_count_increment': 0,
                'promotion_authorized': False, **dict.fromkeys(runner.REPLAY_FLAGS, True)}
            runner.validate_session(item, SESSIONS[0])
            for mutate in (
                lambda obj: obj.update(provider_source_values_read=True),
                lambda obj: obj['source'].update(H2_reader_offsets=[-1, 14]),
                lambda obj: obj['bank']['cells'][2].update(candidate_SSE=obj['bank']['cells'][2]['candidate_SSE']+1e-8),
                lambda obj: obj['bank']['cells'][0]['same_lambda_reversal_contrast'].update(delta_R2=1e50),
            ):
                changed = copy.deepcopy(item)
                mutate(changed)
                with self.assertRaises(ValueError):
                    runner.validate_session(changed, SESSIONS[0])

    def test_saved_anchor_drift_and_stale_original_selection_are_exact_failures(self):
        import run_primary_timevarying_history_blend as runner
        with tempfile.TemporaryDirectory() as directory:
            args, packets, _, _, h2_path, blend_path = saved_anchor_family(directory)
            with patch.object(runner, 'read_json', side_effect=lambda path: copy.deepcopy(packets[Path(path)])):
                context = runner.load_anchor_context(args)
            for path, field in ((h2_path, 'actual_xy'), (h2_path, 'candidate_xy_by_lambda'),
                                (blend_path, 'candidate_xy'), (blend_path, 'selected_lambda')):
                with np.load(path, allow_pickle=False) as saved:
                    original = {key: saved[key] for key in saved.files}
                changed = {key: value.copy() for key, value in original.items()}
                changed[field].flat[0] += 1e-6
                np.savez(path, **changed)
                with self.assertRaises(ValueError):
                    runner.load_session_anchors(args, SESSIONS[0], context)
                np.savez(path, **original)
            # Historical selectors must be replayed before any new session array load.
            for name in ('W', 'HW', 'H2', 'blend'):
                changed = copy.deepcopy(packets)
                report = changed[getattr(args, name.lower()+'_summary')]
                panel = report['panels'][0] if name in ('W', 'HW') else report['candidate'] if name == 'H2' else report['blend']
                panel['sessions'][0]['outer_training_sessions'][0] = SESSIONS[0]
                with patch.object(runner, 'read_json', side_effect=lambda path: copy.deepcopy(changed[Path(path)])):
                    with self.assertRaises(ValueError):
                        runner.load_anchor_context(args)


if __name__ == '__main__':
    unittest.main()
