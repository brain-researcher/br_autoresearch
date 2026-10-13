"""Pure donor-training arrays and target-initializer learned electrode ranking."""
import random

import numpy as np

from direction_training import residual_coordinates, residual_statistics
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import ridge_coefficients, scaling

POLICY = "learned_electrode_value_linear_v1"
CONTROL = "learned_electrode_value_labels3_v1"
MC_SEEDS = (0, 1, 2)
VIEWS = ("0", "1", "2", "3", "outer")
FEATURES = ("residual_marginal_gain", "direction_repeatability", "residual_redundancy")


def view_features(x, y, directions):
    x, y, directions = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(directions)
    levels, counts = np.unique(directions, return_counts=True)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or x.shape[1] % 3
            or x.shape[1] < 6 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.array_equal(levels, np.arange(8)) or len(set(counts)) != 1 or counts[0] not in (3, 4)):
        raise ValueError("Only balanced24/32 fitting rows and complete three-coordinate electrodes")
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    n, groups = len(xx), xx.shape[1]//3
    blocks = xx.reshape(n, groups, 3)
    gram = np.einsum("ngb,ngc->gbc", blocks, blocks)/n
    matrix = (gram+gram.transpose(0, 2, 1))/2+np.eye(3)[None]
    cross = np.einsum("ngb,nu->gbu", blocks, yy)/n
    marginal = np.sum(cross*np.linalg.solve(matrix, cross), axis=(1, 2))/y.shape[1]
    mean, sd, _ = scaling(x)
    raw = (x-mean)/sd
    means = np.stack([raw[directions == d].mean(axis=0) for d in range(8)])
    between = counts[0]*np.sum((means-raw.mean(axis=0))**2, axis=0)/7
    within = sum(np.sum((raw[directions == d]-means[d])**2, axis=0) for d in range(8))/(n-8)
    component = np.maximum((between-within)/counts[0], 0)
    reliability = np.divide(component, component+within, out=np.zeros_like(component), where=(component+within)>0)
    covariance = (xx.T@xx/n).reshape(groups, 3, groups, 3).transpose(0, 2, 1, 3)
    overlap = np.sum(covariance**2, axis=(2, 3))/9
    np.fill_diagonal(overlap, 0.)
    result = np.column_stack((marginal, reliability.reshape(groups, 3).mean(axis=1), overlap.sum(axis=1)/(groups-1)))
    if not np.isfinite(result).all():
        raise ArithmeticError("Nonfinite initializer-only predictors")
    return result


def donor_view(x, y, directions, later_x, later_y, later_directions, channels,
               filename, animal, view, progress=None, check=lambda: None):
    view = str(view)
    channels = list(channels)
    features = view_features(x, y, directions)
    later_x, later_y, later_directions = np.asarray(later_x), np.asarray(later_y), np.asarray(later_directions)
    if (view not in VIEWS or len(x) != (32 if view == "outer" else 24)
            or channels != sorted(set(channels)) or len(channels) != len(features)
            or later_x.shape != (64, 3*len(channels)) or later_y.shape != (64, y.shape[1])
            or not np.array_equal(np.bincount(later_directions, minlength=8), np.full(8, 8))
            or not np.isfinite(later_x).all() or not np.isfinite(later_y).all()):
        raise ValueError("Exact matching donor view and separate64-row label support required")
    stats = residual_statistics(x, y, directions)
    xx, yy = residual_coordinates(stats, x, y, directions)
    lx, ly = residual_coordinates(stats, later_x, later_y, later_directions)
    denominator = float(np.sum(((later_y-stats.target_mean)/stats.target_sd)**2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ArithmeticError("Positive finite view-scaled donor label denominator required")
    baseline_sse = float(np.sum(ly**2))
    labels = []
    for electrode in range(len(channels)):
        check()
        columns = slice(3*electrode, 3*electrode+3)
        # The qualified helper minimizes ||Y-XB||_F^2/n + alpha||B||_F^2.
        coefficient = ridge_coefficients(xx[:, columns], yy, 1.)
        sse = float(np.sum((ly-lx[:, columns]@coefficient)**2))
        labels.append((baseline_sse-sse)/denominator)
        if progress is not None:
            progress["donor_label_fits"] += 1
    return {"filename": filename, "animal": animal, "view": view,
            "channel_ids": channels, "predictors": features.tolist(), "value_labels": labels,
            "fitting_rows": len(x), "later_label_rows": 64, "label_denominator": denominator,
            "label_baseline_SSE": baseline_sse, "target_unit_count": y.shape[1],
            "donor_alpha": 1., "donor_objective": "squared_Frobenius_residual/n + alpha*squared_Frobenius_coefficients",
            "helper": "ridge_training.ridge_coefficients; direction_training.residual_statistics/residual_coordinates"}


def fit_ranker(records, target_filename, view, replicate=None):
    view = str(view)
    if target_filename not in DEVELOPMENT or view not in VIEWS or (replicate is not None and replicate not in MC_SEEDS):
        raise ValueError("Fixed target, matching information view and optional fixed control seed required")
    # Exclude the target before touching any predictors, values or fitting statistics.
    selected = [record for record in records if record["filename"] != target_filename and str(record["view"]) == view]
    expected = set(DEVELOPMENT)-{target_filename}
    if len(selected) != 7 or {record["filename"] for record in selected} != expected:
        raise ValueError("Exactly all seven non-target donor sessions required")
    selected.sort(key=lambda record: record["filename"])
    animal_counts = {animal: sum(record["animal"] == animal for record in selected) for animal in ("Chewie", "Mihili")}
    if sorted(animal_counts.values()) != [3, 4]:
        raise ValueError("Both donor animals with three/four sessions required")
    xx, yy, weights, weight_records, permutations = [], [], [], [], []
    for record in selected:
        channels = record["channel_ids"]
        features, labels = np.asarray(record["predictors"], dtype=float), np.asarray(record["value_labels"], dtype=float)
        if (channels != sorted(set(channels)) or len(channels) < 2 or features.shape != (len(channels), 3)
                or labels.shape != (len(channels),) or not np.isfinite(features).all() or not np.isfinite(labels).all()
                or record["fitting_rows"] != (32 if view == "outer" else 24)):
            raise ValueError("Finite complete donor-view electrode records required")
        if replicate is not None:
            literal = f"ep08-learned-electrode-label-v1|20261003|{record['filename']}|{view}|{replicate}"
            donor = list(range(len(channels)))
            rng = random.Random()
            rng.seed(literal, version=2)
            rng.shuffle(donor)  # Ordinary one-shot permutation; fixed points permitted.
            labels = labels[donor]
            permutations.append({"donor_filename": record["filename"], "view": view, "replicate": replicate,
                "seed_literal": literal, "recipient_channel_ids": channels,
                "donor_channel_ids": [channels[index] for index in donor]})
        per_electrode = 1./(2*animal_counts[record["animal"]]*len(channels))
        xx.append(features)
        yy.append(labels)
        weights.extend([per_electrode]*len(channels))
        weight_records.append({"filename": record["filename"], "animal": record["animal"],
                               "electrode_weight": per_electrode, "session_weight": per_electrode*len(channels)})
    xx, yy, weights = np.vstack(xx), np.concatenate(yy), np.asarray(weights)
    weights /= weights.sum()
    mean = np.sum(weights[:, None]*xx, axis=0)
    raw_sd = np.sqrt(np.sum(weights[:, None]*(xx-mean)**2, axis=0))
    sd = np.where(raw_sd == 0, 1., raw_sd)
    z = (xx-mean)/sd
    zm, ym = np.sum(weights[:, None]*z, axis=0), float(weights@yy)
    centered = z-zm
    coefficient = np.linalg.solve(centered.T@(weights[:, None]*centered)+np.eye(3),
                                   centered.T@(weights*(yy-ym)))
    intercept = ym-float(zm@coefficient)
    return {"target_filename": target_filename, "view": view, "replicate": replicate,
            "predictor_names": list(FEATURES), "predictor_mean": mean.tolist(), "predictor_sd": sd.tolist(),
            "zero_SD_predictors": np.flatnonzero(raw_sd == 0).tolist(),
            "coefficients": coefficient.tolist(), "intercept": intercept, "penalty": 1.,
            "objective": "weighted mean squared label error + squared coefficient norm; unpenalized intercept",
            "donor_sessions": [record["filename"] for record in selected],
            "record_weights": weights.tolist(), "donor_weights": weight_records,
            "label_permutations": permutations}


def rank_electrodes(features, channels, state):
    features, channels = np.asarray(features, dtype=float), np.asarray(channels)
    if features.shape != (len(channels), 3) or len(set(channels.tolist())) != len(channels):
        raise ValueError("One fixed three-predictor row per distinct physical electrode required")
    scores = (features-np.asarray(state["predictor_mean"]))/np.asarray(state["predictor_sd"])@np.asarray(state["coefficients"])+state["intercept"]
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite predicted electrode value")
    return np.lexsort((channels, -scores)), scores


def mean_control_cells(panels):
    if len(panels) != 3:
        raise ValueError("Exactly three fixed label-control panels required")
    indices = [{(c["training_trials"], c["electrodes"]): c for c in panel["cells"]} for panel in panels]
    expected = {(n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}
    if any(len(panel["cells"]) != 20 or set(index) != expected for panel, index in zip(panels, indices)):
        raise ValueError("All twenty identical control resource cells required")
    cells = []
    for key in sorted(expected):
        rows = [index[key] for index in indices]
        denominator = rows[0]["null_SSE"]
        if not np.isfinite(denominator) or denominator <= 0 or any(row["null_SSE"] != denominator for row in rows):
            raise ArithmeticError("Fixed identical control denominator required")
        cell = dict(rows[0])
        for name in ("standardized_SSE", "standardized_wrong_SSE", "native_target_SSE"):
            cell[name] = float(np.mean([row[name] for row in rows]))
        cell.update(Q=1-cell["standardized_SSE"]/denominator,
                    Q_wrong=1-cell["standardized_wrong_SSE"]/denominator,
                    pairing_increment=(cell["standardized_wrong_SSE"]-cell["standardized_SSE"])/denominator,
                    seed_Q_range=[min(row["Q"] for row in rows), max(row["Q"] for row in rows)])
        if "direction_only_Q" in cell:
            if any(row["direction_only_Q"] != cell["direction_only_Q"] for row in rows):
                raise ArithmeticError("Matched direction-only control denominator/prediction required")
            cell["delta_Q_from_direction_only"] = cell["Q"]-cell["direction_only_Q"]
        cells.append(cell)
    return {"cells": cells, "MC_seed_labels": list(MC_SEEDS),
            "aggregation": "mean correct/wrong seed SSE; no prediction ensemble",
            **{"mean_"+key: float(np.mean([cell[key] for cell in cells])) for key in ("Q", "Q_wrong", "pairing_increment")}}
