"""Fitting-only marginal electrode selection with ordinary target-label maps."""
import random
import time

import numpy as np

from direction_training import residual_coordinates, residual_statistics

CONTROL = "initial_marginal_target_labels5_v1"
MC_SEEDS = (0, 1, 2, 3, 4)
VIEWS = ("0", "1", "2", "3", "outer")


def within_direction_map(trial_ids, directions, filename, view, replicate):
    """Canonical original-ID shuffles; never depend on supplied row ordering."""
    view = str(view)
    ids, labels = np.asarray(trial_ids), np.asarray(directions)
    count = 4 if view == "outer" else 3
    if (view not in VIEWS or replicate not in MC_SEEDS or ids.ndim != 1
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
        literal = f"ep08-acquired-marginal-target-label-v1|20261004|{filename}|{view}|{replicate}|{direction}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(donor)  # One ordinary draw, including identity and fixed points.
        for target, source in zip(recipient, donor):
            permutation[positions[target]] = positions[source]
        records.append({"direction": direction, "seed_literal": literal,
                        "recipient_trial_ids": recipient, "donor_trial_ids": donor})
    return permutation, records


def prepare_view(x, y, directions, channels):
    x, y, directions = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(directions)
    channels = np.asarray(channels)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] < 1
            or x.shape[1] != 3*len(channels) or len(channels) < 1
            or channels.tolist() != sorted(set(channels.tolist()))
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Finite true fitting arrays and ascending distinct three-coordinate electrodes required")
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    blocks = xx.reshape(len(xx), len(channels), 3)
    gram = np.einsum("ngb,ngc->gbc", blocks, blocks)/len(xx)
    matrix = (gram+gram.transpose(0, 2, 1))/2+np.eye(3)[None]
    return blocks, yy, matrix


def rank_prepared(prepared, directions, trial_ids, channels, filename, view, replicate):
    permutation, records = within_direction_map(trial_ids, directions, filename, view, replicate)
    blocks, yy, matrix = prepared
    cross = np.einsum("ngb,nu->gbu", blocks, yy[permutation])/len(yy)
    scores = np.sum(cross*np.linalg.solve(matrix, cross), axis=(1, 2))/yy.shape[1]
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite target-label marginal gain")
    return np.lexsort((np.asarray(channels), -scores)), scores, records


def rank_view(x, y, directions, trial_ids, channels, filename, view, replicate):
    return rank_prepared(prepare_view(x, y, directions, channels), directions, trial_ids,
                         channels, filename, view, replicate)


def order_bank(x, y, directions, trial_ids, channels, folds, filename, progress, check=lambda: None):
    """All five fitting-only views and all five seeds before subsequent reads."""
    started = time.monotonic()
    x, y, directions, ids = np.asarray(x), np.asarray(y), np.asarray(directions), np.asarray(trial_ids)
    if len(x) != 32 or len(ids) != 32 or len(folds) != 4 or len(channels) != 96:
        raise ValueError("Original 32-row/96-electrode/four-inner-view initializer required")
    views = {str(index): np.asarray(fit) for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    prepared = {view: prepare_view(x[fit], y[fit], directions[fit], channels) for view, fit in views.items()}
    banks, records = {}, []
    for replicate in MC_SEEDS:
        banks[replicate] = {}
        for view in VIEWS:
            check()
            fit = views[view]
            order, gains, maps = rank_prepared(prepared[view], directions[fit], ids[fit], channels,
                                               filename, view, replicate)
            banks[replicate][view] = order[:32]
            records.append({"policy": CONTROL, "mc_seed": replicate, "view": view,
                "fitting_trial_ids": ids[fit].tolist(), "physical_channels": list(channels),
                "marginal_gains_by_physical_channel": gains.tolist(),
                "selected_channel_order32": [channels[index] for index in order[:32]],
                "target_label_permutations": maps})
            progress["selector_orderings"] += 1
            progress["direction_permutation_maps"] += len(maps)
    progress["selection_seconds"] += time.monotonic()-started
    return banks, records


def mean_control_cells(panels):
    if len(panels) != 5:
        raise ValueError("Exactly five predetermined complete target-label seed panels required")
    indices = [{(c["training_trials"], c["electrodes"]): c for c in panel["cells"]} for panel in panels]
    expected = {(n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}
    if any(len(panel["cells"]) != 20 or set(index) != expected for panel, index in zip(panels, indices)):
        raise ValueError("All twenty matching cells required for each seed")
    cells = []
    for key in sorted(expected):
        rows = [index[key] for index in indices]
        denominator = rows[0]["null_SSE"]
        if not np.isfinite(denominator) or denominator <= 0 or any(row["null_SSE"] != denominator for row in rows):
            raise ArithmeticError("Fixed common positive finite five-seed denominator required")
        cell = dict(rows[0])
        for name in ("standardized_SSE", "standardized_wrong_SSE", "native_target_SSE"):
            if not all(np.isfinite(row[name]) for row in rows):
                raise ArithmeticError("Every seed SSE must remain finite")
            cell[name] = float(np.mean([row[name] for row in rows]))
        cell.update(Q=1-cell["standardized_SSE"]/denominator,
                    Q_wrong=1-cell["standardized_wrong_SSE"]/denominator,
                    pairing_increment=(cell["standardized_wrong_SSE"]-cell["standardized_SSE"])/denominator,
                    seed_Q_range=[min(row["Q"] for row in rows), max(row["Q"] for row in rows)])
        if "direction_only_Q" in cell:
            if any(row["direction_only_Q"] != cell["direction_only_Q"] for row in rows):
                raise ArithmeticError("Same direction-only reference required")
            cell["delta_Q_from_direction_only"] = cell["Q"]-cell["direction_only_Q"]
        cells.append(cell)
    return {"cells": cells, "MC_seed_labels": list(MC_SEEDS),
            "aggregation": "mean five correct/wrong seed SSE; no prediction ensemble",
            **{"mean_"+key: float(np.mean([cell[key] for cell in cells]))
               for key in ("Q", "Q_wrong", "pairing_increment")}}
