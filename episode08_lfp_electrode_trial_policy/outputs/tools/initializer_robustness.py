"""Fitting-only residual marginal ranking under three separately paid initializers."""
import time
import random

import numpy as np

from initializer_target_label_control import MC_SEEDS, VIEWS, mean_control_cells
from direction_training import residual_statistics, residual_coordinates
from initializer_robustness_roles import ORDERS
from selector_training import validate_folds

POLICY = "initial_marginal_initializer_orders3_v1"
CONTROL = "initial_marginal_initializer_orders3_target_labels5_v1"
BRANCHES = (None, *MC_SEEDS)


def within_direction_map(trial_ids, directions, filename, view, replicate, order):
    """Canonical original-ID shuffles; never depend on supplied row ordering."""
    view = str(view)
    ids, labels = np.asarray(trial_ids), np.asarray(directions)
    count = 4 if view == "outer" else 3
    if (view not in VIEWS or replicate not in MC_SEEDS or order not in ORDERS or ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer) or len(set(ids.tolist())) != len(ids)
            or labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))
            or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, count))):
        raise ValueError("Exact unique original IDs in balanced matching 24/32-row view required")
    positions = {int(trial): index for index, trial in enumerate(ids)}
    permutation, records = np.arange(len(ids)), []
    for direction in range(8):
        recipient = sorted(int(trial) for trial in ids[labels == direction])
        donor = recipient.copy()
        literal = f"ep08-initializer-robustness-target-label-v1|20261004|{filename}|{order}|{view}|{replicate}|{direction}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donor)  # One ordinary draw, including identity and fixed points.
        for target, source in zip(recipient, donor):
            permutation[positions[target]] = positions[source]
        records.append({"direction": direction, "seed_literal": literal,
                        "recipient_trial_ids": recipient, "donor_trial_ids": donor})
    return permutation, records


def prepare_view(x, y, directions, channels):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    directions = np.asarray(directions)
    levels, counts = np.unique(directions, return_counts=True)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] < 1
            or x.shape != (len(directions), 3*len(channels)) or not len(channels)
            or list(channels) != sorted(set(channels))
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.array_equal(levels, np.arange(8)) or len(set(counts)) != 1
            or counts[0] not in (3, 4)):
        raise ValueError("Balanced24/32 true fitting arrays and canonical three-coordinate electrodes required")
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    if any(not np.isfinite(value).all() for value in vars(stats).values() if isinstance(value, np.ndarray)):
        raise ArithmeticError("Nonfinite residual normalization")
    blocks = xx.reshape(len(xx), len(channels), 3)
    gram = np.einsum("ngb,ngc->gbc", blocks, blocks)/len(xx)
    matrix = (gram+gram.transpose(0, 2, 1))/2+np.eye(3)[None]
    if not all(np.isfinite(value).all() for value in (xx, yy, matrix)):
        raise ArithmeticError("Nonfinite residual fitting coordinates")
    return blocks, yy, matrix


def rank_prepared(prepared, directions, trial_ids, channels, filename, view, replicate, order):
    ids, labels, view = np.asarray(trial_ids), np.asarray(directions), str(view)
    if (view not in VIEWS or replicate not in BRANCHES or order not in ORDERS
            or len(ids) != (32 if view == "outer" else 24)
            or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids.tolist())) != len(ids) or labels.shape != ids.shape
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))
            or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, len(ids)//8))):
        raise ValueError("Exact original-ID initializer view and fixed seed required")
    blocks, yy, matrix = prepared
    if blocks.shape != (len(ids), len(channels), 3) or len(yy) != len(ids):
        raise ValueError("Prepared fitting arrays differ from ranking support")
    permutation, records = np.arange(len(ids)), []
    if replicate is not None:
        permutation, records = within_direction_map(ids, labels, filename, view, replicate, order)
    cross = np.einsum("ngb,nu->gbu", blocks, yy[permutation])/len(yy)
    coefficients = np.linalg.solve(matrix, cross)
    scores = np.sum(cross*coefficients, axis=(1, 2))/yy.shape[1]
    if not np.isfinite(coefficients).all() or not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite residual marginal coefficients or gains")
    return np.lexsort((np.asarray(channels), -scores)), scores, records


def rank_view(x, y, directions, trial_ids, channels, filename, view, replicate, order):
    return rank_prepared(prepare_view(x, y, directions, channels), directions, trial_ids,
                         channels, filename, view, replicate, order)


def order_bank(x, y, directions, trial_ids, channels, folds, filename, order, progress, check=lambda: None):
    started = time.monotonic()
    x, y, directions, ids = np.asarray(x), np.asarray(y), np.asarray(directions), np.asarray(trial_ids)
    if x.shape != (32, 288) or len(y) != 32 or len(ids) != 32 or len(channels) != 96:
        raise ValueError("Original32-row/96-electrode initializer required")
    validate_folds(32, folds)
    views = {str(index): np.asarray(fit) for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    prepared = {}
    for view, fit in views.items():
        check()
        prepared[view] = prepare_view(x[fit], y[fit], directions[fit], channels)
        progress["residual_normalization_views"] += 1
    banks, records = {}, []
    for replicate in BRANCHES:
        banks[replicate] = {}
        for view in VIEWS:
            check()
            fit, ready = views[view], prepared[view]
            ranking, gains, maps = rank_prepared(ready, directions[fit], ids[fit], channels,
                                              filename, view, replicate, order)
            banks[replicate][view] = ranking[:32]
            records.append({"policy": POLICY if replicate is None else CONTROL,
                "mc_seed": replicate, "view": view, "initializer_order": order, "fitting_trial_ids": ids[fit].tolist(),
                "physical_channels": list(channels),
                "marginal_gains_by_physical_channel": gains.tolist(),
                "selected_channel_order32": [int(channels[index]) for index in ranking[:32]],
                "normalization": "true fitting direction residuals; residual-X SD and raw-Y SD",
                "target_label_permutations": maps})
            progress["selector_orderings"] += 1
            progress["direction_permutation_maps"] += len(maps)
            progress["marginal_score_solves"] += len(channels)
    progress["selection_seconds"] += time.monotonic()-started
    return banks, records
