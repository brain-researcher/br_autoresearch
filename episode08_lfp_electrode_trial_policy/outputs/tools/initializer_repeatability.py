"""Target-free direction-repeatability and uniform physical-electrode orders."""
import random
import time

import numpy as np

from initializer_target_label_control import MC_SEEDS, VIEWS, mean_control_cells
from ridge_training import scaling

POLICY = "initial_direction_repeatability_v1"
CONTROL = "initial_uniform_order5_v1"
BRANCHES = (None, *MC_SEEDS)


def repeatability_scores(x, directions, channels):
    x, directions = np.asarray(x, dtype=float), np.asarray(directions)
    levels, counts = np.unique(directions, return_counts=True)
    if (x.ndim != 2 or x.shape != (len(directions), 3*len(channels))
            or list(channels) != sorted(set(channels)) or not np.isfinite(x).all()
            or not np.array_equal(levels, np.arange(8)) or len(set(counts)) != 1
            or counts[0] not in (3, 4)):
        raise ValueError("Only balanced24/32 fitting rows and canonical three-band electrodes")
    mean, sd, constant = scaling(x)
    z = (x-mean)/sd
    means = np.stack([z[directions == d].mean(axis=0) for d in range(8)])
    between = counts[0]*np.sum((means-z.mean(axis=0))**2, axis=0)/7
    within = sum(np.sum((z[directions == d]-means[d])**2, axis=0) for d in range(8))/(len(z)-8)
    component = np.maximum((between-within)/counts[0], 0)
    reliability = np.divide(component, component+within, out=np.zeros_like(component), where=(component+within)>0)
    scores = reliability.reshape(len(channels), 3).mean(axis=1)
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite initializer repeatability")
    values, multiplicities = np.unique(scores, return_counts=True)
    return scores, {"feature_repeatability": reliability.reshape(len(channels), 3).tolist(),
        "MS_between": between.reshape(len(channels), 3).tolist(),
        "MS_within": within.reshape(len(channels), 3).tolist(),
        "tau_squared": component.reshape(len(channels), 3).tolist(),
        "zero_SD_columns": list(constant),
        "tied_score_groups": [{"score": float(value), "electrodes": int(count)}
                              for value, count in zip(values, multiplicities) if count > 1]}


def uniform_order(channels, filename, view, replicate):
    view = str(view)
    channels = list(channels)
    if view not in VIEWS or replicate not in MC_SEEDS or channels != sorted(set(channels)) or not channels:
        raise ValueError("Canonical physical IDs and fixed view/replicate required")
    literal = f"ep08-initial-uniform-order-v1|20261004|{filename}|{view}|{replicate}"
    shuffled = channels.copy()
    rng = random.Random()
    rng.seed(literal, version=2)
    rng.shuffle(shuffled)  # Direct order: no score sorting, redraw or tie bias.
    positions = {channel: index for index, channel in enumerate(channels)}
    return np.asarray([positions[channel] for channel in shuffled]), {
        "seed_literal": literal, "canonical_physical_ids": channels,
        "permuted_physical_ids": shuffled}


def rank_view(x, directions, channels, filename, view, replicate=None):
    view = str(view)
    if view not in VIEWS or len(x) != (32 if view == "outer" else 24):
        raise ValueError("Exact initializer information view required")
    if replicate is None:
        scores, details = repeatability_scores(x, directions, channels)
        order = np.lexsort((np.asarray(channels), -scores))
        record = {"repeatability_scores": scores.tolist(), **details}
    else:
        order, permutation = uniform_order(channels, filename, view, replicate)
        record = {"physical_id_permutation": permutation}
    return order, record


def order_bank(x, directions, channels, folds, filename, progress, check=lambda: None):
    started = time.monotonic()
    x, directions = np.asarray(x), np.asarray(directions)
    if x.shape != (32, 288) or len(channels) != 96 or len(folds) != 4:
        raise ValueError("Original96-electrode/three-band/32-row initializer required")
    views = {str(index): np.asarray(fit) for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    banks, records = {}, []
    for replicate in BRANCHES:
        banks[replicate] = {}
        for view in VIEWS:
            check()
            fit = views[view]
            order, details = rank_view(x[fit], directions[fit], channels, filename, view, replicate)
            banks[replicate][view] = order[:32]
            records.append({"policy": POLICY if replicate is None else CONTROL,
                "mc_seed": replicate, "view": view, "fitting_initializer_positions": fit.tolist(),
                "physical_channels": list(channels),
                "selected_channel_order32": [int(channels[index]) for index in order[:32]], **details})
            progress["selector_orderings"] += 1
            progress["repeatability_fitting_views" if replicate is None else "physical_id_permutations"] += 1
    progress["selection_seconds"] += time.monotonic()-started
    return banks, records
