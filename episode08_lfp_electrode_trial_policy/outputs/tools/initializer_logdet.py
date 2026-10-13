"""Target-free fitting-view residual-input log-determinant group selection."""
import time
import numpy as np

POLICY = 'initial_residual_logdet_v1'
VIEWS = ('0', '1', '2', '3', 'outer')


def prepare_view(x, directions, channels):
    x, labels = np.asarray(x, dtype=float), np.asarray(directions)
    if (x.ndim != 2 or x.shape[1] != 3*len(channels) or not np.isfinite(x).all()
            or list(channels) != sorted(set(channels)) or labels.shape != (len(x),)
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))
            or len(x) not in (24, 32)
            or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, len(x)//8))):
        raise ValueError('Balanced24/32 fitting-only X view with three real coordinates per physical ID required')
    means = np.stack([x[labels == direction].mean(axis=0) for direction in range(8)])
    residual = x-means[labels]
    center, sd = residual.mean(axis=0), residual.std(axis=0)
    constant = np.flatnonzero(sd == 0)
    sd[constant] = 1.
    xx = (residual-center)/sd
    if not np.isfinite(xx).all():
        raise ArithmeticError('Nonfinite normalized residual input')
    return xx, {'zero_SD_input_columns': constant.tolist(),
                'fitting_rows': len(x), 'input_coordinates_per_electrode': 3}


def logdet_gain(gram):
    gram = np.asarray(gram, dtype=float)
    if gram.shape != (3, 3) or not np.isfinite(gram).all():
        raise ArithmeticError('Finite3x3 group Gram required')
    gram = (gram+gram.T)/2
    sign, gain = np.linalg.slogdet(np.eye(3)+gram)
    tolerance = 128*np.finfo(float).eps*max(1., float(np.linalg.norm(gram, ord='fro')))
    if sign <= 0 or not np.isfinite(gain) or gain < -tolerance:
        raise ArithmeticError('Invalid determinant or materially negative logdet gain')
    return float(gain)


def greedy_order(xx, channels, maximum=32, check=lambda: None):
    xx = np.asarray(xx, dtype=float)
    if (xx.ndim != 2 or len(xx) < 1 or xx.shape[1] != 3*len(channels)
            or list(channels) != sorted(set(channels)) or not np.isfinite(xx).all()
            or not 1 <= maximum <= len(channels)):
        raise ValueError('Finite three-coordinate matrix and distinct ascending physical IDs required')
    ax = xx.copy()
    selected, gains, ties, negative = [], [], [], 0
    available = np.ones(len(channels), dtype=bool)
    for _ in range(maximum):
        check()
        scores = np.full(len(channels), -np.inf)
        grams = {}
        for index in np.flatnonzero(available):
            block = slice(3*index, 3*index+3)
            gram = xx[:, block].T@ax[:, block]/len(xx)
            gram = (gram+gram.T)/2
            scores[index] = logdet_gain(gram)
            negative += int(scores[index] < 0)
            grams[index] = gram
        choice = int(np.argmax(scores))
        ties.append(int(np.sum(scores[available] == scores[choice])))
        selected.append(choice)
        gains.append(float(scores[choice]))
        available[choice] = False
        block = slice(3*choice, 3*choice+3)
        h = ax[:, block].copy()
        rhs = xx[:, block].T@ax/len(xx)
        update = np.linalg.solve(np.eye(3)+grams[choice], rhs)
        if not np.isfinite(update).all():
            raise ArithmeticError('Nonfinite logdet rank-three solve')
        ax = ax-h@update
        if not np.isfinite(ax).all():
            raise ArithmeticError('Nonfinite updated logdet state')
    return np.asarray(selected), np.asarray(gains), ties, negative


def rank_view(x, directions, channels, maximum=32, check=lambda: None):
    xx, details = prepare_view(x, directions, channels)
    order, gains, ties, negative = greedy_order(xx, channels, maximum, check)
    details.update(selected_gains=gains.tolist(), available_tie_counts=ties,
                   numerical_negative_gain_occurrences=negative)
    return order, details


def order_bank(x, directions, channels, folds, filename, progress, check=lambda: None):
    started = time.monotonic()
    x, directions = np.asarray(x), np.asarray(directions)
    if x.shape != (32, 288) or len(folds) != 4:
        raise ValueError('Original32-row all96 three-band initializer required')
    views = {str(i): np.asarray(fit) for i, (fit, _) in enumerate(folds)}
    views['outer'] = np.arange(32)
    orders, records = {}, []
    for view in VIEWS:
        fit = views[view]
        order, details = rank_view(x[fit], directions[fit], channels, 32, check)
        orders[view] = order
        records.append({'policy': POLICY, 'filename': filename, 'view': view,
            'initializer_row_indices': fit.tolist(), 'physical_channels': list(channels),
            'selected_channel_order32': [int(channels[i]) for i in order], **details})
        progress['selector_orderings'] += 1
        progress['logdet_selection_steps'] += 32
        progress['numerical_negative_gain_occurrences'] += details['numerical_negative_gain_occurrences']
    progress['selection_seconds'] += time.monotonic()-started
    return orders, records
