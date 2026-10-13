"""Initializer-only conditional ridge proxy and matched target-label controls."""
import time

import numpy as np

from direction_training import residual_coordinates, residual_statistics
from initializer_target_label_control import MC_SEEDS, VIEWS, mean_control_cells, within_direction_map

POLICY = "initial_residual_conditional_v1"
CONTROL = "initial_residual_conditional_target_labels5_v1"
BRANCHES = (None, *MC_SEEDS)


def greedy_order(xx, yy, channels, maximum=32, check=lambda: None):
    """Exact qualified rank-three updates on already normalized fitting arrays."""
    x, y = np.asarray(xx, dtype=float), np.asarray(yy, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] < 1
            or x.shape[1] != 3*len(channels) or not 0 < maximum <= len(channels)
            or list(channels) != sorted(set(channels))
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Finite residual matrices and canonical three-coordinate electrode groups required")
    n, groups = len(x), len(channels)
    blocks, ax, ay = x.reshape(n, groups, 3), x.copy(), y.copy()
    available = np.ones(groups, dtype=bool)
    chosen, gains, ties = [], [], []
    for _ in range(maximum):
        check()
        gram = np.einsum("ngb,ngc->gbc", blocks, ax.reshape(n, groups, 3))/n
        matrix = (gram+gram.transpose(0, 2, 1))/2+np.eye(3)[None]
        cross = np.einsum("ngb,nu->gbu", blocks, ay)/n
        scores = np.sum(cross*np.linalg.solve(matrix, cross), axis=(1, 2))/y.shape[1]
        if not np.isfinite(scores).all():
            raise ArithmeticError("Nonfinite inherited conditional-gain score")
        electrode = int(np.argmax(np.where(available, scores, -np.inf)))
        chosen.append(electrode)
        gains.append(float(scores[electrode]))
        ties.append(int(np.count_nonzero(scores[available] == scores[electrode])))
        available[electrode] = False
        columns = slice(3*electrode, 3*electrode+3)
        update = ax[:, columns].copy()
        # Both right-hand sides use the same pre-update AX/AY state.
        rhs_x, rhs_y = x[:, columns].T@ax/n, x[:, columns].T@ay/n
        ax -= update@np.linalg.solve(matrix[electrode], rhs_x)
        ay -= update@np.linalg.solve(matrix[electrode], rhs_y)
    return np.asarray(chosen), np.asarray(gains), ties


def rank_view(x, y, directions, trial_ids, channels, filename, view, replicate=None,
              maximum=32, check=lambda: None):
    view = str(view)
    ids, labels = np.asarray(trial_ids), np.asarray(directions)
    m = 4 if view == "outer" else 3
    if (view not in VIEWS or len(x) != 8*m or len(y) != len(x) or ids.shape != (len(x),)
            or not np.issubdtype(ids.dtype, np.integer) or len(set(ids.tolist())) != len(ids)
            or labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))
            or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, m))):
        raise ValueError("Exact balanced fitting-only24/32 view with original trial IDs required")
    stats = residual_statistics(np.asarray(x), np.asarray(y), labels)
    xx, yy = residual_coordinates(stats, np.asarray(x), np.asarray(y), labels)
    maps = []
    if replicate is not None:
        permutation, maps = within_direction_map(ids, labels, filename, view, replicate)
        yy = yy[permutation]
    order, gains, ties = greedy_order(xx, yy, channels, maximum, check)
    return order, {"target_label_permutations": maps, "selected_gains": gains.tolist(),
                   "available_tie_counts": ties, "zero_SD_input_columns": list(stats.constant_input_columns),
                   "zero_SD_target_columns": list(stats.constant_target_columns)}


def order_bank(x, y, directions, trial_ids, channels, folds, filename, progress, check=lambda: None):
    started = time.monotonic()
    x, y, labels, ids = np.asarray(x), np.asarray(y), np.asarray(directions), np.asarray(trial_ids)
    if x.shape != (32, 288) or len(y) != 32 or len(ids) != 32 or len(channels) != 96 or len(folds) != 4:
        raise ValueError("Original32-row/96-electrode/four-inner-view initializer required")
    views = {str(index): np.asarray(fit) for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    banks, records = {}, []
    for replicate in BRANCHES:
        banks[replicate] = {}
        for view in VIEWS:
            check()
            fit = views[view]
            order, details = rank_view(x[fit], y[fit], labels[fit], ids[fit], channels,
                                       filename, view, replicate, 32, check)
            banks[replicate][view] = order
            records.append({"policy": POLICY if replicate is None else CONTROL,
                "mc_seed": replicate, "view": view, "fitting_trial_ids": ids[fit].tolist(),
                "physical_channels": list(channels),
                "selected_channel_order32": [int(channels[index]) for index in order], **details})
            progress["selector_orderings"] += 1
            progress["conditional_selection_steps"] += len(order)
            progress["direction_permutation_maps"] += len(details["target_label_permutations"])
    progress["selection_seconds"] += time.monotonic()-started
    return banks, records
