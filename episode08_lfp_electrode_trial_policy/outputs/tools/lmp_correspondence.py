"""Scalar native-LMP marginal selection and matched initializer label maps."""
import time

import numpy as np

from direction_training import residual_coordinates, residual_statistics
from initializer_target_label_control import MC_SEEDS, VIEWS, mean_control_cells, within_direction_map

POLICY = "initial_marginal_lmp_only_v1"
CONTROL = "initial_marginal_lmp_only_target_labels5_v1"
BRANCHES = (None, *MC_SEEDS)


def prepare_view(x, y, directions, channels):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] < 1
            or x.shape[1] != len(channels) or list(channels) != sorted(set(channels))
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("One finite real LMP coordinate per ascending physical electrode required")
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    gram = np.sum(xx*xx, axis=0)/len(xx)
    return xx, yy, gram


def rank_prepared(prepared, directions, trial_ids, channels, filename, view, replicate=None):
    xx, yy, gram = prepared
    if replicate is None:
        ids, labels = np.asarray(trial_ids), np.asarray(directions)
        count = 4 if str(view) == "outer" else 3
        if (str(view) not in VIEWS or ids.ndim != 1 or len(ids) != len(xx)
                or not np.issubdtype(ids.dtype, np.integer) or len(set(ids.tolist())) != len(ids)
                or labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer)
                or not np.array_equal(np.unique(labels), np.arange(8))
                or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, count))):
            raise ValueError("Exact true-selector balanced initializer view required")
        maps = []
    else:
        permutation, maps = within_direction_map(trial_ids, directions, filename, view, replicate)
        yy = yy[permutation]
    cross = xx.T@yy/len(xx)
    scores = np.sum(cross*cross, axis=1)/((1+gram)*yy.shape[1])
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite scalar marginal gain")
    return np.lexsort((np.asarray(channels), -scores)), scores, maps


def rank_view(x, y, directions, trial_ids, channels, filename, view, replicate=None):
    return rank_prepared(prepare_view(x, y, directions, channels), directions, trial_ids,
                         channels, filename, view, replicate)


def order_bank(x, y, directions, trial_ids, channels, folds, filename, progress, check=lambda: None):
    started = time.monotonic()
    x, y, directions, ids = np.asarray(x), np.asarray(y), np.asarray(directions), np.asarray(trial_ids)
    if x.shape != (32, 96) or len(y) != 32 or len(ids) != 32 or len(folds) != 4:
        raise ValueError("Original initializer with exactly96 scalar native-LMP columns required")
    views = {str(index): np.asarray(fit) for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    prepared = {view: prepare_view(x[fit], y[fit], directions[fit], channels) for view, fit in views.items()}
    banks, records = {}, []
    for replicate in BRANCHES:
        banks[replicate] = {}
        for view in VIEWS:
            check()
            fit = views[view]
            order, gains, maps = rank_prepared(prepared[view], directions[fit], ids[fit], channels,
                                               filename, view, replicate)
            banks[replicate][view] = order[:32]
            records.append({"policy": POLICY if replicate is None else CONTROL,
                "mc_seed": replicate, "view": view, "fitting_trial_ids": ids[fit].tolist(),
                "physical_channels": list(channels), "input_coordinates_per_electrode": 1,
                "marginal_gains_by_physical_channel": gains.tolist(),
                "selected_channel_order32": [int(channels[index]) for index in order[:32]],
                "target_label_permutations": maps})
            progress["selector_orderings"] += 1
            progress["direction_permutation_maps"] += len(maps)
    progress["selection_seconds"] += time.monotonic()-started
    return banks, records
