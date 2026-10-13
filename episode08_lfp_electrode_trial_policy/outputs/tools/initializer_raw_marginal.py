"""Fitting-only direction-inclusive marginal rankings with original label maps."""
import time

import numpy as np

from initializer_target_label_control import MC_SEEDS, VIEWS, mean_control_cells, within_direction_map
from ridge_training import scaling
from selector_training import validate_folds

POLICY = "initial_raw_marginal_v1"
CONTROL = "initial_raw_marginal_target_labels5_v1"
BRANCHES = (None, *MC_SEEDS)


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
    xm, xs, xc = scaling(x)
    ym, ys, yc = scaling(y)
    xx, yy = (x-xm)/xs, (y-ym)/ys
    blocks = xx.reshape(len(xx), len(channels), 3)
    gram = np.einsum("ngb,ngc->gbc", blocks, blocks)/len(xx)
    matrix = (gram+gram.transpose(0, 2, 1))/2+np.eye(3)[None]
    if not all(np.isfinite(value).all() for value in (xm, xs, ym, ys, xx, yy, matrix)):
        raise ArithmeticError("Nonfinite raw fitting normalization")
    return {"blocks": blocks, "yy": yy, "matrix": matrix,
            "x_mean": xm, "x_sd": xs, "y_mean": ym, "y_sd": ys,
            "zero_SD_input_columns": list(xc), "zero_SD_target_columns": list(yc)}


def rank_prepared(prepared, directions, trial_ids, channels, filename, view, replicate=None):
    ids, labels, view = np.asarray(trial_ids), np.asarray(directions), str(view)
    if (view not in VIEWS or replicate not in BRANCHES
            or len(ids) != (32 if view == "outer" else 24)
            or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids.tolist())) != len(ids) or labels.shape != ids.shape
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.array_equal(np.unique(labels), np.arange(8))
            or not np.array_equal(np.bincount(labels, minlength=8), np.full(8, len(ids)//8))):
        raise ValueError("Exact original-ID initializer view and fixed seed required")
    blocks, yy, matrix = prepared["blocks"], prepared["yy"], prepared["matrix"]
    if blocks.shape != (len(ids), len(channels), 3) or len(yy) != len(ids):
        raise ValueError("Prepared fitting arrays differ from ranking support")
    permutation, records = np.arange(len(ids)), []
    if replicate is not None:
        permutation, records = within_direction_map(ids, labels, filename, view, replicate)
    cross = np.einsum("ngb,nu->gbu", blocks, yy[permutation])/len(yy)
    coefficients = np.linalg.solve(matrix, cross)
    scores = np.sum(cross*coefficients, axis=(1, 2))/yy.shape[1]
    if not np.isfinite(coefficients).all() or not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite raw marginal coefficients or gains")
    return np.lexsort((np.asarray(channels), -scores)), scores, records


def rank_view(x, y, directions, trial_ids, channels, filename, view, replicate=None):
    return rank_prepared(prepare_view(x, y, directions, channels), directions, trial_ids,
                         channels, filename, view, replicate)


def order_bank(x, y, directions, trial_ids, channels, folds, filename, progress, check=lambda: None):
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
        progress["raw_normalization_views"] += 1
    banks, records = {}, []
    for replicate in BRANCHES:
        banks[replicate] = {}
        for view in VIEWS:
            check()
            fit, ready = views[view], prepared[view]
            order, gains, maps = rank_prepared(ready, directions[fit], ids[fit], channels,
                                              filename, view, replicate)
            banks[replicate][view] = order[:32]
            records.append({"policy": POLICY if replicate is None else CONTROL,
                "mc_seed": replicate, "view": view, "fitting_trial_ids": ids[fit].tolist(),
                "physical_channels": list(channels),
                "marginal_gains_by_physical_channel": gains.tolist(),
                "selected_channel_order32": [int(channels[index]) for index in order[:32]],
                "zero_SD_input_columns": ready["zero_SD_input_columns"],
                "zero_SD_target_columns": ready["zero_SD_target_columns"],
                "normalization": "true fitting global mean and population SD; no direction subtraction",
                "target_label_permutations": maps})
            progress["selector_orderings"] += 1
            progress["direction_permutation_maps"] += len(maps)
            progress["marginal_score_solves"] += len(channels)
    progress["selection_seconds"] += time.monotonic()-started
    return banks, records
