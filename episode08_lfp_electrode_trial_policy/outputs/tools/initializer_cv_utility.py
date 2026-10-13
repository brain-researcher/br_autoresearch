"""Strictly internal fitting-only ridge utility and training-label controls."""
import random
import time
import numpy as np
from direction_training import residual_coordinates, residual_statistics
from initializer_target_label_control import mean_control_cells

POLICY = 'initial_crossvalidated_utility_v1'
CONTROL = 'initial_crossvalidated_utility_fit_labels5_v1'
MC_SEEDS = (0, 1, 2, 3, 4)
BRANCHES = (None, *MC_SEEDS)
VIEWS = ('0', '1', '2', '3', 'outer')


def internal_folds(trial_ids, directions):
    ids, labels = np.asarray(trial_ids), np.asarray(directions)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids.tolist())) != len(ids) or len(ids) not in (24, 32)
            or labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))):
        raise ValueError('Original distinct IDs in a balanced24/32 main view required')
    m = len(ids)//8
    if not np.array_equal(np.bincount(labels, minlength=8), np.full(8, m)):
        raise ValueError('Equal three/four rows per direction required')
    slots = np.empty(len(ids), dtype=int)
    for direction in range(8):
        positions = np.flatnonzero(labels == direction)
        positions = positions[np.argsort(ids[positions])]
        slots[positions] = (np.arange(m)+direction) % m
    return [(np.flatnonzero(slots != k), np.flatnonzero(slots == k)) for k in range(m)]


def fit_only_map(trial_ids, directions, filename, view, internal_fold, replicate):
    ids, labels = np.asarray(trial_ids), np.asarray(directions)
    count = 3 if str(view) == 'outer' else 2
    if (str(view) not in VIEWS or replicate not in MC_SEEDS
            or not isinstance(internal_fold, (int, np.integer)) or not 0 <= internal_fold < count+1
            or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) != 8*count or len(set(ids.tolist())) != len(ids)
            or labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))
            or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, count))):
        raise ValueError('Exact internal fitting-only balanced target map required')
    position = {int(trial): i for i, trial in enumerate(ids)}
    permutation, records = np.arange(len(ids)), []
    for direction in range(8):
        recipients = sorted(int(t) for t in ids[labels == direction])
        donors = recipients.copy()
        literal = (f'ep08-initial-cv-utility-fit-label-v1|20261004|{filename}|'
                   f'{view}|{internal_fold}|{replicate}|{direction}')
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donors)
        for recipient, donor in zip(recipients, donors):
            permutation[position[recipient]] = position[donor]
        records.append({'direction_index': direction, 'seed_literal': literal,
                        'recipient_trial_ids': recipients, 'donor_trial_ids': donors})
    return permutation, records


def prepare_internal_fold(x, y, directions, fit, valid):
    x, y, labels = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(directions)
    fit, valid = np.asarray(fit), np.asarray(valid)
    if (x.ndim != 2 or x.shape[1] % 3 or y.ndim != 2 or len(y) != len(x) or y.shape[1] < 1
            or labels.shape != (len(x),) or not np.issubdtype(labels.dtype, np.integer)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or len(x) not in (24, 32) or len(valid) != 8 or len(fit) != len(x)-8
            or not np.issubdtype(fit.dtype, np.integer) or not np.issubdtype(valid.dtype, np.integer)
            or len(set(fit.tolist())) != len(fit) or len(set(valid.tolist())) != len(valid)
            or set(fit) & set(valid) or set(fit) | set(valid) != set(range(len(x)))
            or not np.array_equal(np.bincount(labels[fit], minlength=8), np.full(8, len(fit)//8))
            or not np.array_equal(np.bincount(labels[valid], minlength=8), np.ones(8, dtype=int))):
        raise ValueError('Complete balanced internal fitting and validation partition required')
    stats = residual_statistics(x[fit], y[fit], labels[fit])
    xx, yy = residual_coordinates(stats, x[fit], y[fit], labels[fit])
    xv, yv = residual_coordinates(stats, x[valid], y[valid], labels[valid])
    b = stats.target_residual_mean.copy()
    r0v = yv+b
    blocks = xx.reshape(len(fit), x.shape[1]//3, 3)
    gram = np.einsum('nei,nej->eij', blocks, blocks)/len(fit)
    gram = (gram+gram.transpose(0, 2, 1))/2
    if not all(np.isfinite(a).all() for a in (xx, yy, xv, r0v, b, gram)):
        raise ArithmeticError('Nonfinite fitting-only proxy normalization')
    return {'xx': xx, 'yy': yy, 'xv': xv, 'r0v': r0v, 'b': b, 'gram': gram,
            'fit': fit.copy(), 'valid': valid.copy(), 'statistics': stats,
            'zero_SD_input_columns': list(stats.constant_input_columns),
            'zero_SD_target_columns': list(stats.constant_target_columns)}


def validation_utilities(prepared, permutation=None):
    xx, yy, xv = prepared['xx'], prepared['yy'], prepared['xv']
    if permutation is not None:
        permutation = np.asarray(permutation)
        if (permutation.shape != (len(yy),) or not np.issubdtype(permutation.dtype, np.integer)
                or sorted(permutation.tolist()) != list(range(len(yy)))):
            raise ValueError('Whole internal-fitting-row permutation required')
        yy = yy[permutation]
    electrodes = xx.shape[1]//3
    cross = (xx.T@yy/len(xx)).reshape(electrodes, 3, yy.shape[1])
    coefficients = np.linalg.solve(np.eye(3)[None]+prepared['gram'], cross)
    predicted = np.einsum('veb,ebu->veu', xv.reshape(len(xv), electrodes, 3), coefficients)+prepared['b'][None, None, :]
    residual = prepared['r0v'][:, None, :]
    utilities = np.sum(residual**2-(residual-predicted)**2, axis=(0, 2))
    if not np.isfinite(coefficients).all() or not np.isfinite(utilities).all():
        raise ArithmeticError('Nonfinite signed internal validation utility')
    return utilities


def prepare_view(x, y, directions, trial_ids, channels):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 3*len(channels) or list(channels) != sorted(set(channels)):
        raise ValueError('Three real coordinates per distinct ascending physical ID required')
    return [prepare_internal_fold(x, y, directions, fit, valid)
            for fit, valid in internal_folds(trial_ids, directions)]


def rank_prepared(prepared, trial_ids, directions, channels, filename, view, replicate, progress):
    ids, labels = np.asarray(trial_ids), np.asarray(directions)
    if str(view) not in VIEWS or len(ids) != (32 if str(view) == 'outer' else 24):
        raise ValueError('Exact original main initializer view required')
    gains, records = np.zeros(len(channels)), []
    for fold, state in enumerate(prepared):
        fit, valid = state['fit'], state['valid']
        permutation, maps = (None, []) if replicate is None else fit_only_map(
            ids[fit], labels[fit], filename, view, fold, replicate)
        gains += validation_utilities(state, permutation)
        progress['proxy_ridge_models'] += len(channels)
        progress['direction_permutation_maps'] += len(maps)
        records.append({'internal_fold': fold, 'fitting_trial_ids': ids[fit].tolist(),
            'validation_trial_ids': ids[valid].tolist(), 'target_label_permutations': maps,
            'zero_SD_input_columns': state['zero_SD_input_columns'],
            'zero_SD_target_columns': state['zero_SD_target_columns']})
    gains /= len(ids)*prepared[0]['yy'].shape[1]
    if not np.isfinite(gains).all():
        raise ArithmeticError('Nonfinite aggregate internal validation utility')
    return np.lexsort((np.asarray(channels), -gains)), gains, records


def order_bank(x, y, directions, trial_ids, channels, folds, filename, progress, check=lambda: None):
    started = time.monotonic()
    x, y, labels, ids = np.asarray(x), np.asarray(y), np.asarray(directions), np.asarray(trial_ids)
    if x.shape != (32, 288) or len(y) != 32 or len(ids) != 32 or len(folds) != 4:
        raise ValueError('Original32-row96-electrode initializer required')
    views = {str(i): np.asarray(fit) for i, (fit, _) in enumerate(folds)}
    views['outer'] = np.arange(32)
    prepared = {}
    for view, fit in views.items():
        check()
        prepared[view] = prepare_view(x[fit], y[fit], labels[fit], ids[fit], channels)
        progress['proxy_normalization_views'] += len(prepared[view])
    banks, records = {}, []
    for seed in BRANCHES:
        banks[seed] = {}
        for view in VIEWS:
            check()
            fit = views[view]
            order, utilities, internal = rank_prepared(prepared[view], ids[fit], labels[fit],
                                                       channels, filename, view, seed, progress)
            banks[seed][view] = order[:32]
            records.append({'policy': POLICY if seed is None else CONTROL, 'mc_seed': seed, 'view': view,
                'fitting_trial_ids': ids[fit].tolist(), 'physical_channels': list(channels),
                'input_coordinates_per_electrode': 3, 'internal_folds': internal,
                'signed_validation_utilities': utilities.tolist(),
                'selected_channel_order32': [int(channels[i]) for i in order[:32]]})
            progress['selector_orderings'] += 1
    progress['selection_seconds'] += time.monotonic()-started
    return banks, records
