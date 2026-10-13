"""One X-only selector with true-target heads and completed matched references."""
import copy
import numpy as np

from acquired_only_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from initializer_logdet import POLICY, order_bank
from initializer_repeatability import POLICY as REPEATABILITY, CONTROL as UNIFORM, MC_SEEDS
from prepare_development_trial_roles import DEVELOPMENT
from residual_selection_training import fit_selected_rank

ALL_POLICIES = (POLICY, UNIFORM, REPEATABILITY, 'initial_marginal')


def fit_target(path, source, role, records, progress, check, exposure):
    channels, ids = source['common_physical_channels'], balanced_ids(role, 32)
    if (len(channels) != 96 or channels != sorted(set(channels))
            or ids != role['nested_training_sets']['32']['training_trials']):
        raise ValueError('Original96-electrode/32-row initializer required')
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[t]['direction_index'] for t in ids])
    folds = relative_initial_folds(role)
    orders, selections = order_bank(x, labels, channels, folds, role['filename'], progress, check)
    if len(selections) != 5 or set(orders) != {'0', '1', '2', '3', 'outer'}:
        raise ArithmeticError('All five orders required before future reads')
    models, nulls = [], {32: fit_direction_null(y, labels)}
    progress['direction_only_models'] += 1
    full = Trajectory(path, source, role, records, channels, check, exposure)
    full.initialize_selected(x, y, channels, guide)
    fitted = fit_capacity(x.copy(), y.copy(), labels.copy(), folds, 'reduced_rank', progress)
    for total in CHECKPOINTS:
        check()
        if total > 32:
            fitted, nulls[total] = later_fit(full, total, records, progress, True)
        models.append((total, 96, tuple(channels), fitted))
    inner, outer = [orders[str(i)] for i in range(4)], orders['outer']
    for count in (8, 16, 32):
        fitted, columns = fit_selected_rank(x, y, labels, folds, inner, outer, count, progress)
        selected = [channels[int(column//3)] for column in columns[::3]]
        trajectory = Trajectory(path, source, role, records, selected, check, exposure)
        trajectory.initialize_selected(x, y, channels, guide)
        for total in CHECKPOINTS:
            check()
            if total > 32:
                fitted, _ = later_fit(trajectory, total, records, progress)
            models.append((total, count, tuple(selected), fitted))
    if len(models) != 20:
        raise ArithmeticError('All twenty distinct models required before evaluation')
    return models, nulls, guide, selections


def score_target(models, nulls, x, y, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = []
    for total, count, selected, fitted in models:
        columns = [3*positions[channel]+band for channel in selected for band in range(3)]
        prediction = predict_capacity(fitted, x[:, columns], labels)
        stats = fitted.statistics
        scores = prediction_scores(prediction, y, labels, stats.target_mean, stats.target_sd)
        denominator = float(np.sum(((y-stats.target_mean)/stats.target_sd)**2))
        correct = float(np.sum(((y-prediction)/stats.target_sd)**2))
        null = nulls[total]
        direction_q = prediction_scores(null.direction_target_mean[labels], y, labels,
                                        null.target_mean, null.target_sd)['Q']
        cost = 32*96+(total-32)*count
        rows.append({'training_trials': total, 'electrodes': count,
            'channels': list(selected), 'rank': fitted.rank, 'alpha': fitted.alpha,
            'candidate_count': fitted.candidate_count, 'null_SSE': denominator,
            'standardized_SSE': correct, 'standardized_wrong_SSE': (1-scores['Q_wrong'])*denominator,
            'native_target_SSE': float(np.sum((y-prediction)**2)), **scores,
            'direction_only_Q': direction_q, 'delta_Q_from_direction_only': scores['Q']-direction_q,
            'LFP_electrode_trials': cost, 'LFP_numeric_elements': cost*30,
            'target_numeric_elements': total*y.shape[1]*10, 'initial_trials': 32,
            'initial_LFP_electrodes': 96, 'input_coordinates_per_electrode': 3})
    rows.sort(key=lambda r: (r['training_trials'], r['electrodes']))
    cells = [{k: v for k, v in row.items() if k not in ('channels', 'rank', 'alpha', 'candidate_count')}
             for row in rows]
    return {'cells': cells, 'fitted_models': rows,
        **{'mean_'+key: float(np.mean([c[key] for c in cells])) for key in
           ('Q', 'Q_wrong', 'pairing_increment', 'delta_Q_from_direction_only')}}


def attach_references(sessions, reference):
    if (reference['status'] != 'completed_two_initializer_repeatability_comparisons'
            or reference['held_session_accessed'] or reference['policy'] != REPEATABILITY
            or reference['control'] != UNIFORM or reference['MC_seeds'] != list(MC_SEEDS)
            or reference['recipe'] != 'outputs/initializer_repeatability_proposal.md'
            or len(reference['sessions']) != 8
            or {s['filename'] for s in reference['sessions']} != set(DEVELOPMENT)):
        raise ValueError('Complete qualified repeatability/uniform/marginal reference required')
    old = {s['filename']: s for s in reference['sessions']}
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    maximum, target_max = 0., 0.
    for session in sessions:
        previous = old[session['filename']]
        if (previous['animal'] != session['animal'] or previous['source_roles'] != session['source_roles']
                or previous['target_unit_count'] != session['target_unit_count']
                or [p['mc_seed'] for p in previous['seed_panels']] != list(MC_SEEDS)):
            raise ValueError('Exact original role/target/seed support required')
        indices = {}
        for policy in (UNIFORM, REPEATABILITY, 'initial_marginal'):
            panel = previous['panels'][policy]
            index = {(c['training_trials'], c['electrodes']): c for c in panel['cells']}
            if len(panel['cells']) != 20 or set(index) != expected:
                raise ValueError('Every reference must retain all twenty cells')
            required = {'Q', 'Q_wrong', 'pairing_increment', 'direction_only_Q',
                'delta_Q_from_direction_only', 'LFP_electrode_trials', 'LFP_numeric_elements',
                'target_numeric_elements'}
            if policy != 'initial_marginal':
                required |= {'null_SSE', 'standardized_SSE', 'standardized_wrong_SSE'}
            if any(not required <= cell.keys() for cell in index.values()):
                raise ValueError('Required recorded reference fields are missing')
            indices[policy] = index
        full_references = []
        for panel in [previous['panels'][REPEATABILITY], *(s['panel'] for s in previous['seed_panels'])]:
            full = {r['training_trials']: r for r in panel['fitted_models'] if r['electrodes'] == 96}
            if set(full) != set(CHECKPOINTS):
                raise ValueError('All original full96 fitted-score records required')
            full_references.append(full)
        rows = session['panels'][POLICY]['fitted_models']
        if len(rows) != 20 or {(r['training_trials'], r['electrodes']) for r in rows} != expected:
            raise ValueError('All twenty new target fits required')
        for row in rows:
            key = row['training_trials'], row['electrodes']
            for policy, index in indices.items():
                before = index[key]
                if any(row[k] != before[k] for k in
                       ('LFP_electrode_trials', 'LFP_numeric_elements', 'target_numeric_elements')):
                    raise ValueError('Matched three-band physical/target exposure differs')
                if policy != 'initial_marginal':
                    if not np.isclose(row['null_SSE'], before['null_SSE'], rtol=1e-10, atol=1e-10):
                        raise ArithmeticError('Recorded target denominator does not replay')
                error = abs(row['direction_only_Q']-before['direction_only_Q'])
                if not np.isfinite(error) or error > 1e-10:
                    raise ArithmeticError('Direction-null Q does not replay')
                target_max = max(target_max, error)
            if row['electrodes'] == 96:
                for full in full_references:
                    before = full[row['training_trials']]
                    if any(row[k] != before[k] for k in ('channels', 'rank', 'alpha', 'candidate_count')):
                        raise ArithmeticError('Original full96 capacity/support does not replay')
                    for key in ('Q', 'Q_wrong', 'pairing_increment'):
                        error = abs(row[key]-before[key])
                        if not np.isfinite(error) or error > 1e-10:
                            raise ArithmeticError('Original full96 prediction scores do not replay')
                        maximum = max(maximum, error)
        for policy in (UNIFORM, REPEATABILITY, 'initial_marginal'):
            session['panels'][policy] = copy.deepcopy(previous['panels'][policy])
        session['uniform_seed_panels'] = copy.deepcopy(previous['seed_panels'])
        indices[POLICY] = {(c['training_trials'], c['electrodes']): c for c in session['panels'][POLICY]['cells']}
        for policy in ALL_POLICIES:
            panel = session['panels'][policy]
            for cell in panel['cells']:
                key = cell['training_trials'], cell['electrodes']
                for other in ALL_POLICIES:
                    cell['delta_Q_from_'+other] = cell['Q']-indices[other][key]['Q']
            for other in ALL_POLICIES:
                panel['mean_delta_Q_from_'+other] = float(np.mean([c['delta_Q_from_'+other] for c in panel['cells']]))
            panel['mean_delta_Q_from_direction_only'] = float(np.mean([c['delta_Q_from_direction_only'] for c in panel['cells']]))
    return maximum, target_max


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or {s['filename'] for s in sessions} != set(DEVELOPMENT)
                     or any(sum(s['animal'] == animal for s in sessions) != 4 for animal in ('Chewie', 'Mihili'))):
        raise ValueError('All eight original sessions required')
    keys = ('mean_Q', 'mean_Q_wrong', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
            *('mean_delta_Q_from_'+p for p in ALL_POLICIES))
    panels = []
    for policy in ALL_POLICIES:
        animals = [{'animal': animal, **{key: float(np.mean([s['panels'][policy][key]
                    for s in sessions if s['animal'] == animal])) for key in keys}}
                   for animal in ('Chewie', 'Mihili')]
        panels.append({'policy': policy, 'animals': animals,
                       **{key: float(np.mean([a[key] for a in animals])) for key in keys}})
    return panels
