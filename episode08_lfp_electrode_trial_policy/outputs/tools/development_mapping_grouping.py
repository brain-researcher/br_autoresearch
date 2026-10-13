#!/usr/bin/env python3
"""Three-seed acquired-direction grouping ablation; unchanged decoder and sources."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from acquired_only_reader import Exposure, balanced_ids, read_trials
from allocation_reader import FittingSource
from arrival_order_reader import OrderTrajectory, order_queues
from capacity_training import fit_capacity
from development_acquired_only import relative_initial_folds
from development_design import ELECTRODE_COUNTS, candidate_designs
from development_mapping_variance import (
    POLICY as MAPPING, SCORE_KEYS, attach_references as replay_simple_references,
    model_key, score_mapping,
)
from development_reader import selected_metadata
from mapping_grouping_control import GroupingDesign, MC_SEEDS, ORDERS, POLICY
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress
from trial_allocation import CHECKPOINTS, POLICIES, freeze_initial_model, fit_frozen_rank

ALL_POLICIES = (*POLICIES, MAPPING, POLICY)
WRONG_SSE = 'initial_SD_standardized_wrong_SSE'


def fit_session(path, source, role, records, progress, check, exposure, group_stream):
    queues = {'original': tuple(tuple(pool[:12]) for pool in role['fitting_pool_by_direction']),
              **{str(seed): order_queues(role, seed) for seed in range(3)}}
    models = {order: {mc: [] for mc in MC_SEEDS} for order in ORDERS}
    histories = {order: [] for order in ORDERS}
    initial_ids = balanced_ids(role, 32)
    if initial_ids != role['nested_training_sets']['32']['training_trials']:
        raise ValueError('Unchanged initializer IDs/order required')
    labels = np.asarray([records[trial]['direction_index'] for trial in initial_ids])
    folds = relative_initial_folds(role)
    with FittingSource(path, source, role, records, check, exposure) as reader:
        for design in candidate_designs(role, source['common_physical_channels'], 5):
            if design['training_trials'] != 32:
                continue
            check()
            channels, count, seed = tuple(design['channels']), design['electrodes'], design['seed']
            initial_x, initial_y = reader.read(initial_ids, list(channels))
            selected = fit_capacity(initial_x, initial_y, labels, folds, 'reduced_rank', progress)
            initial, rank = freeze_initial_model(selected, initial_x.shape[1], initial_y.shape[1])
            fixed_mean, fixed_sd = initial.statistics.target_mean.copy(), initial.statistics.target_sd.copy()
            final_model, final_arrays = None, None
            for order in ORDERS:
                for mc in MC_SEEDS:
                    history = OrderTrajectory(reader, channels, initial_x, initial_y, queues[order])
                    proxy = GroupingDesign(initial_x.copy(), labels.copy(), initial.statistics, initial.alpha,
                                           initial_ids, role['filename'], order, count, seed, mc)
                    trace = []
                    models[order][mc].append((32, count, seed, channels, initial, fixed_mean, fixed_sd, tuple(history.counts)))
                    for total in range(33, 97):
                        check()
                        chosen, values = proxy.choose(history.counts.copy())
                        group_record = progress['grouping_prechoice_decisions']
                        group_stream.write(json.dumps({'record_id': group_record, 'filename': role['filename'],
                            'order': order, 'electrodes': count, 'subset_seed': seed, 'mc_seed': mc,
                            'acquired_before': total - 1, 'chosen_direction': chosen, **values}) + '\n')
                        trial = history.acquire(chosen)
                        proxy.observe(history.x[trial].copy(), chosen, trial)
                        progress['grouping_prechoice_decisions'] += 1
                        trace.append({'acquired_total': total, 'chosen_direction': chosen, 'chosen_trial': trial,
                            'counts_before': values['counts_before'], 'pre_choice_utilities': values['utility'],
                            'design_index_before': values['design_index_before'], 'group_record_id': group_record})
                        if total not in CHECKPOINTS:
                            continue
                        xx, yy, directions, ids = history.observed()
                        if total == 96 and final_model is not None:
                            if (ids != role['nested_training_sets']['96']['training_trials']
                                    or not all(np.array_equal(a, b) for a, b in zip((xx, yy, directions), final_arrays))):
                                raise ArithmeticError('Canonical final X/Y/labels differ across new histories')
                            fitted = final_model
                        else:
                            fitted = fit_frozen_rank(xx, yy, directions, initial.alpha, rank, progress)
                            if total == 96:
                                if ids != role['nested_training_sets']['96']['training_trials']:
                                    raise ArithmeticError('Complete canonical96trial pool required')
                                final_model, final_arrays = fitted, (xx.copy(), yy.copy(), directions.copy())
                        models[order][mc].append((total, count, seed, channels, fitted, fixed_mean, fixed_sd, tuple(history.counts)))
                    histories[order].append({'policy': POLICY, 'mc_seed': mc, 'electrodes': count,
                        'seed': seed, 'channels': list(channels), 'initial_selected_rank': selected.rank,
                        'frozen_numeric_rank': rank, 'frozen_alpha': initial.alpha, 'proxy_eta': proxy.eta,
                        'initial_candidate_count': selected.candidate_count, 'choices': trace})
        if any(len(models[order][mc]) != 80 for order in ORDERS for mc in MC_SEEDS):
            raise RuntimeError('All960 model references required before session evaluation')
        if any(len(histories[order]) != 48 for order in ORDERS):
            raise RuntimeError('All192 session trajectories required before evaluation')
        guide = reader.reference
    return models, histories, queues, guide


def average_seed_scores(panels):
    """Mean correct/wrong SSE within matched sets first; never ensemble predictions."""
    if len(panels) != len(MC_SEEDS):
        raise ValueError('All three fixed MC seed panels required')
    indexed = [{model_key(row): row for row in panel['fitted_models']} for panel in panels]
    if any(len(index) != 80 or set(index) != set(indexed[0]) for index in indexed):
        raise ValueError('Exactly80 matched per-seed fitted-model rows required')
    rows = []
    for key, first in indexed[0].items():
        seeds = [index[key] for index in indexed]
        denominator = first['initial_mean_null_SSE']
        if not np.isfinite(denominator) or denominator <= 0:
            raise ArithmeticError('Finite positive unchanged score denominator required')
        for row in seeds:
            if (row['rank'] != first['rank'] or row['alpha'] != first['alpha']
                    or row['initial_mean_null_SSE'] != denominator
                    or not all(np.isfinite(row[name]) for name in SCORE_KEYS)):
                raise ArithmeticError('Seed capacity, denominator or score support differs')
        mean_sse = float(np.mean([row['initial_SD_standardized_SSE'] for row in seeds]))
        mean_wrong = float(np.mean([(1. - row['Q_wrong_common']) * denominator for row in seeds]))
        averaged = {name: value for name, value in first.items() if name != 'direction_counts'}
        averaged.update(native_target_SSE=float(np.mean([row['native_target_SSE'] for row in seeds])),
                        initial_SD_standardized_SSE=mean_sse, Q_common=1. - mean_sse / denominator,
                        Q_wrong_common=1. - mean_wrong / denominator,
                        pairing_increment=(mean_wrong - mean_sse) / denominator)
        averaged[WRONG_SSE] = mean_wrong
        averaged['direction_counts_by_mc_seed'] = [row.get('direction_counts') for row in seeds]
        averaged['seed_Q_range'] = [min(row['Q_common'] for row in seeds), max(row['Q_common'] for row in seeds)]
        rows.append(averaged)
    cells = []
    for total in CHECKPOINTS:
        for count in ELECTRODE_COUNTS:
            selected = [row for row in rows if row['training_trials'] == total and row['electrodes'] == count]
            if len(selected) != (1 if count == 96 else 5):
                raise ValueError('Complete fixed subset support required')
            cell = {'training_trials': total, 'electrodes': count,
                    **{name: float(np.mean([row[name] for row in selected])) for name in (*SCORE_KEYS, WRONG_SSE)}}
            for name in ('LFP_electrode_trials', 'LFP_numeric_elements', 'target_numeric_elements'):
                if name in selected[0]:
                    if any(row[name] != selected[0][name] for row in selected):
                        raise ArithmeticError('Matched acquisition costs required')
                    cell[name] = selected[0][name]
            cells.append(cell)
    return {'fitted_models': rows, 'cells': cells, 'MC_seed_labels': list(MC_SEEDS),
            'aggregation': 'mean seed correct/wrong SSE, then subsets; never averaged predictions'}


def summary_keys():
    names = ('Q_common', 'Q_wrong_common', 'pairing_increment',
             *('delta_Q_common_from_' + other for other in ALL_POLICIES))
    return tuple(prefix + name for prefix in ('mean_', 'intermediate_mean_') for name in names)


def panel_summaries(panels):
    for policy in ALL_POLICIES:
        for index, cell in enumerate(panels[policy]['cells']):
            for other in ALL_POLICIES:
                reference = panels[other]['cells'][index]
                if (cell['training_trials'], cell['electrodes']) != (reference['training_trials'], reference['electrodes']):
                    raise ValueError('Matched resource cells required')
                cell['delta_Q_common_from_' + other] = cell['Q_common'] - reference['Q_common']
        for key in summary_keys():
            intermediate = key.startswith('intermediate_mean_')
            name = key.removeprefix('intermediate_mean_') if intermediate else key.removeprefix('mean_')
            selected = [cell for cell in panels[policy]['cells'] if not intermediate or cell['training_trials'] not in (32, 96)]
            panels[policy][key] = float(np.mean([cell[name] for cell in selected]))


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or any(sum(s['animal'] == a for s in sessions) != 4 for a in ('Chewie', 'Mihili'))):
        raise ValueError('All eight development sessions required')
    panels = []
    for policy in ALL_POLICIES:
        animals = [{'animal': animal, **{key: float(np.mean([s['panels'][policy][key]
            for s in sessions if s['animal'] == animal])) for key in summary_keys()}} for animal in ('Chewie', 'Mihili')]
        panels.append({'policy': policy, 'animals': animals,
                       **{key: float(np.mean([animal[key] for animal in animals])) for key in summary_keys()}})
    return panels


def influence(sessions):
    return [{'excluded_session': name, 'panels': aggregate([s for s in sessions if s['filename'] != name], False)}
            for name in DEVELOPMENT]


def attach_references(sessions, mapping, original, arrival):
    """All prior score packets enter trusted reporting only after all new fits."""
    if (mapping['status'] != 'completed_mapping_design_variance_comparison' or mapping['held_session_accessed']
            or {s['filename'] for s in mapping['sessions']} != set(DEVELOPMENT)):
        raise ValueError('Complete matching development-only mapping reference required')
    maximum = 0.
    # Reuse the qualified simple-reference identity/capacity/queue/score checks
    # independently for each seed, using a temporary reporting view only.
    for mc in MC_SEEDS:
        views = [{'filename': session['filename'], 'animal': session['animal'], 'orders': [
            {'order': order['order'], 'queues': order['queues'],
             'trajectories': [trace for trace in order['trajectories'] if trace['mc_seed'] == mc],
             'panels': {MAPPING: copy.deepcopy(order['seed_panels'][mc]['panel'])}}
            for order in session['orders']]} for session in sessions]
        maximum = max(maximum, replay_simple_references(views, original, arrival))
    prior_sessions = {session['filename']: session for session in mapping['sessions']}
    for session in sessions:
        prior = prior_sessions[session['filename']]
        if prior['animal'] != session['animal'] or tuple(order['order'] for order in prior['orders']) != ORDERS:
            raise ValueError('Mapping animal/order identities differ')
        for order, previous in zip(session['orders'], prior['orders']):
            if order['order'] != previous['order'] or order['queues'] != previous['queues']:
                raise ValueError('Mapping queue/order identities differ')
            history_index = {(trace['electrodes'], trace['seed']): trace for trace in previous['trajectories']}
            if len(history_index) != 16:
                raise ValueError('Every original mapping electrode set required')
            indexed = {model_key(row): row for row in previous['panels'][MAPPING]['fitted_models']}
            if len(indexed) != 80:
                raise ValueError('Every original mapping fitted row required')
            for trace in order['trajectories']:
                before = history_index[trace['electrodes'], trace['seed']]
                if any(trace[key] != before[key] for key in ('channels', 'frozen_numeric_rank', 'frozen_alpha', 'proxy_eta')):
                    raise ArithmeticError('Mapping initializer coordinates/capacity identity differs')
            for seed in order['seed_panels']:
                for row in seed['panel']['fitted_models']:
                    before = indexed[model_key(row)]
                    if row['rank'] != before['rank'] or row['alpha'] != before['alpha']:
                        raise ArithmeticError('Mapping numeric rank/alpha differs')
                    keys = SCORE_KEYS if row['training_trials'] in (32, 96) else ('initial_mean_null_SSE',)
                    for key in keys:
                        error = abs(row[key] - before[key])
                        limit = 1e-10 if key in SCORE_KEYS[:3] else 1e-10 * max(1., abs(before[key]))
                        if not np.isfinite(error) or error > limit:
                            raise ArithmeticError('Mapping endpoint score/normalizer replay failed')
                        if key in SCORE_KEYS[:3]:
                            maximum = max(maximum, error)
            order['panels'] = {**copy.deepcopy(previous['panels']),
                               POLICY: average_seed_scores([seed['panel'] for seed in order['seed_panels']])}
            panel_summaries(order['panels'])
        session['panels'] = {}
        for policy in ALL_POLICIES:
            orders = [order['panels'][policy] for order in session['orders']]
            cells = [{key: first[key] if key in ('training_trials', 'electrodes')
                      else float(np.mean([order['cells'][index][key] for order in orders])) for key in first}
                     for index, first in enumerate(orders[0]['cells'])]
            session['panels'][policy] = {'cells': cells,
                **{key: float(np.mean([order[key] for order in orders])) for key in summary_keys()}}
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'metadata-report', 'trial-roles', 'mapping-report', 'original-report', 'arrival-report', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Verified coordinator-owned Slurm compute allocation required')
    if not 0 < args.max_seconds <= 1140 or args.output_dir.exists():
        raise ValueError('Fresh output and finite bound<=1140seconds required')
    metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
    if len(roles['sessions']) != 8 or {session['filename'] for session in roles['sessions']} != set(DEVELOPMENT):
        raise ValueError('Exactly eight frozen development sessions required')
    sources = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    args.output_dir.mkdir(parents=True)
    started, block_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def deadline(*_):
        raise TimeoutError('Selected grouping-control comparison deadline reached')
    def check():
        if time.monotonic() - started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(capacity_inner_candidate_scores=0, grouping_prechoice_decisions=0)
    report = {'analysis_id': 'development_search', 'recipe': 'outputs/mapping_grouping_control_proposal.md',
        'status': 'running', 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
        'fit_progress': progress, 'sessions': [], 'completed_session_ids': [],
        'logical_neural_bytes': 0, 'unique_logical_neural_bytes': 0,
        'fitting_trial_read_calls': 0, 'evaluation_trials_read': 0, 'held_session_accessed': False,
        'valid_evaluations_charged': 0, 'held_openings_charged': 0, 'global_search_completed': False,
        'named_explanatory_amendment_activated': False, 'raw_arrays_or_coefficients_written': False,
        'orders': list(ORDERS), 'policy': POLICY, 'MC_seeds': list(MC_SEEDS),
        'true_features_labels_and_targets_preserved': True,
        'same_history_global_geometry_preserved': True, 'different_histories_can_have_different_A': True,
        'evaluation_normalization': 'fixed common32trial rawY_SD and global_mean_null',
        'boundary': 'own-history acquired rows only; no future or cross-history cache access',
        'seed_aggregation': 'mean correct/wrong SSE, not averaged predictions'}
    exposure, pending = Exposure(report), []
    try:
        for filename, source, role, records in sources:
            check()
            start = time.monotonic()
            groups_path = args.output_dir / (filename + '.group_assignments.jsonl')
            with groups_path.open('x') as group_stream:
                models, histories, queues, guide = fit_session(args.source_root / filename, source, role,
                    records, progress, check, exposure, group_stream)
            fit_end = time.monotonic()
            ex, ey, _ = read_trials(args.source_root / filename, source, role, records, role['evaluation_trials'],
                source['common_physical_channels'], check, exposure, guide, evaluation=True)
            labels = np.asarray([records[trial]['direction_index'] for trial in role['evaluation_trials']])
            orders = [{'order': order, 'queues': [list(queue) for queue in queues[order]],
                'seed_panels': [{'mc_seed': mc, 'panel': score_mapping(models[order][mc], ex, ey, labels,
                    source['common_physical_channels'])} for mc in MC_SEEDS],
                'trajectories': histories[order]} for order in ORDERS]
            pending.append({'filename': filename, 'animal': role['animal'], 'orders': orders,
                'all608_actual_models_and192_trajectories_fixed_before_evaluation': True,
                'initial_final_models_shared_across_orders_and_seeds': True,
                'group_assignments': str(groups_path), 'training_read_and_fit_seconds': fit_end - start,
                'elapsed_seconds': time.monotonic() - start})
            report['completed_session_ids'].append(filename)
            print(json.dumps({'session_completed': filename, 'elapsed_seconds': time.monotonic() - start}), flush=True)
            del models, histories, orders, ex, ey, guide
        # No policy choice or fitting is permitted after these score-packet reads.
        mapping, original, arrival = (json.loads(path.read_text()) for path in
            (args.mapping_report, args.original_report, args.arrival_report))
        report['endpoint_score_replay_max_absolute_difference'] = attach_references(pending, mapping, original, arrival)
        report['old_prediction_array_identity_claimed'] = False
        report['reference_reuse'] = [{'path': str(path), 'status': prior['status'], 'refitted': False,
            'logical_neural_bytes_previously_charged': prior['logical_neural_bytes'],
            'elapsed_seconds_previously_charged': prior['elapsed_seconds']} for path, prior in (
                (args.mapping_report, mapping), (args.original_report, original), (args.arrival_report, arrival))]
        report['sessions'], report['panels'] = pending, aggregate(pending)
        report['fixed_prediction_session_omission'] = influence(pending)
        report['by_order'] = []
        for order in ORDERS:
            selected = [{'filename': session['filename'], 'animal': session['animal'],
                         'panels': next(item['panels'] for item in session['orders'] if item['order'] == order)}
                        for session in pending]
            report['by_order'].append({'order': order, 'panels': aggregate(selected),
                                      'fixed_prediction_session_omission': influence(selected)})
        if (progress['inner_ridge_fits'] != 2560 or progress['outer_training_refits'] != 4864
                or progress['grouping_prechoice_decisions'] != 98304
                or report['fitting_trial_read_calls'] != 102400 or report['evaluation_trials_read'] != 320):
            raise ArithmeticError('Complete adopted model/read count differs')
        report['valid_evaluations_charged'], report['valid_evaluations_total'] = 1, 21
        report['status'] = 'completed_mapping_grouping_control_comparison'
    except Exception as exc:
        report['status'] = 'technical_failure_grouping_control_incomplete'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - started
        report['maximum_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['OS_block_input_operations_cache_dependent'] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock - block_start
        report['observed_utc'] = datetime.now(timezone.utc).isoformat()
        with (args.output_dir / 'mapping_grouping_control.json').open('w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': report['status'], 'valid_evaluations_charged': 1,
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
