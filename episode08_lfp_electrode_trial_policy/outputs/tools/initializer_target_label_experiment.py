"""Target-label ranking control with qualified true-target decoders and scoring."""
import copy

import numpy as np

from acquired_only_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from initializer_target_label_control import CONTROL, MC_SEEDS, mean_control_cells, order_bank
from prepare_development_trial_roles import DEVELOPMENT
from residual_selection_training import fit_selected_rank

ALL_POLICIES = ("initial_marginal", CONTROL, "fixed_random")


def fit_target(path, source, role, records, progress, check, exposure):
    channels, ids = source["common_physical_channels"], balanced_ids(role, 32)
    if (len(channels) != 96 or channels != sorted(set(channels))
            or ids != role["nested_training_sets"]["32"]["training_trials"]):
        raise ValueError("Original 96-channel/32-row initializer required")
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[trial]["direction_index"] for trial in ids])
    folds = relative_initial_folds(role)
    # This call returns all25 orderings before constructing any future history.
    banks, selections = order_bank(x, y, labels, ids, channels, folds, role["filename"], progress, check)
    if len(selections) != 25 or set(banks) != set(MC_SEEDS):
        raise ArithmeticError("All five seeds and five views must precede future reads")
    models, nulls, shared = [], {32: fit_direction_null(y, labels)}, {}
    progress["direction_only_models"] += 1
    full = Trajectory(path, source, role, records, channels, check, exposure)
    full.initialize_selected(x, y, channels, guide)
    fitted = fit_capacity(x.copy(), y.copy(), labels.copy(), folds, "reduced_rank", progress)
    for total in CHECKPOINTS:
        check()
        if total > 32:
            fitted, null = later_fit(full, total, records, progress, True)
            nulls[total] = null
        shared[total] = fitted
    for seed in MC_SEEDS:
        orders = banks[seed]
        inner, outer = [orders[str(index)] for index in range(4)], orders["outer"]
        for count in (8, 16, 32):
            check()
            # Selector permutation never reaches the decoder: y is original.
            fitted, columns = fit_selected_rank(x, y, labels, folds, inner, outer, count, progress)
            selected = [channels[int(column//3)] for column in columns[::3]]
            trajectory = Trajectory(path, source, role, records, selected, check, exposure)
            trajectory.initialize_selected(x, y, channels, guide)
            for total in CHECKPOINTS:
                check()
                if total > 32:
                    fitted, _ = later_fit(trajectory, total, records, progress)
                models.append((seed, total, count, tuple(selected), fitted))
        for total in CHECKPOINTS:
            models.append((seed, total, 96, tuple(channels), shared[total]))
    if len(models) != 100:
        raise ArithmeticError("All100 references/80 distinct models required before evaluation")
    return models, nulls, guide, selections


def score_target(models, nulls, x, y, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = {seed: [] for seed in MC_SEEDS}
    for seed, total, count, selected, fitted in models:
        columns = [3*positions[channel]+band for channel in selected for band in range(3)]
        prediction = predict_capacity(fitted, x[:, columns], labels)
        stats = fitted.statistics
        score = prediction_scores(prediction, y, labels, stats.target_mean, stats.target_sd)
        denominator = float(np.sum(((y-stats.target_mean)/stats.target_sd)**2))
        correct = float(np.sum((((y-stats.target_mean)/stats.target_sd)-
                                ((prediction-stats.target_mean)/stats.target_sd))**2))
        wrong = (1-score["Q_wrong"])*denominator
        null = nulls[total]
        null_q = prediction_scores(null.direction_target_mean[labels], y, labels,
                                   null.target_mean, null.target_sd)["Q"]
        electrode_trials = 32*96+(total-32)*count
        rows[seed].append({"training_trials": total, "electrodes": count, "mc_seed": seed,
            "channels": list(selected), "rank": fitted.rank, "alpha": fitted.alpha,
            "candidate_count": fitted.candidate_count, "null_SSE": denominator,
            "standardized_SSE": correct, "standardized_wrong_SSE": wrong,
            "native_target_SSE": float(np.sum((y-prediction)**2)), **score,
            "direction_only_Q": null_q, "delta_Q_from_direction_only": score["Q"]-null_q,
            "LFP_electrode_trials": electrode_trials, "LFP_numeric_elements": electrode_trials*3*10,
            "target_numeric_elements": total*y.shape[1]*10, "initial_trials": 32, "initial_LFP_electrodes": 96})
    seed_panels = []
    for seed, fitted_rows in rows.items():
        fitted_rows.sort(key=lambda row: (row["training_trials"], row["electrodes"]))
        cells = [{key: value for key, value in row.items()
                  if key not in ("channels", "rank", "alpha", "candidate_count", "mc_seed")}
                 for row in fitted_rows]
        panel = {"cells": cells, "fitted_models": fitted_rows,
                 **{"mean_"+key: float(np.mean([cell[key] for cell in cells]))
                    for key in ("Q", "Q_wrong", "pairing_increment")}}
        seed_panels.append({"mc_seed": seed, "panel": panel})
    return {CONTROL: mean_control_cells([entry["panel"] for entry in seed_panels])}, seed_panels


def attach_reference(sessions, previous):
    if (previous["status"] != "completed_one_acquired_only_comparison_with_random_reference"
            or previous["recipe"] != "outputs/acquired_only_proposal.md" or previous["held_session_accessed"]
            or len(previous["sessions"]) != 8
            or {session["filename"] for session in previous["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete original acquired-only recipe/session support required")
    old = {session["filename"]: session for session in previous["sessions"]}
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    maximum = 0.
    for session in sessions:
        reference = old[session["filename"]]
        if reference["animal"] != session["animal"]:
            raise ValueError("Reference biological animal differs")
        original = reference["panels"]["initial_marginal"]
        original_full = {row["training_trials"]: row for row in original["fitted_models"] if row["electrodes"] == 96}
        if set(original_full) != set(CHECKPOINTS) or len(original["fitted_models"]) != 20:
            raise ValueError("All complete original marginal models required")
        for row in original["fitted_models"]:
            n, e = row["training_trials"], row["electrodes"]
            if (row["target_numeric_elements"] != n*session["target_unit_count"]*10
                    or row["initial_trials"] != 32 or row["initial_LFP_electrodes"] != 96
                    or row["LFP_electrode_trials"] != 32*96+(n-32)*e
                    or row["LFP_numeric_elements"] != row["LFP_electrode_trials"]*30):
                raise ValueError("Original target support/initializer exposure differs")
        if [entry["mc_seed"] for entry in session["seed_panels"]] != list(MC_SEEDS):
            raise ValueError("Every predetermined seed required before reference replay")
        for seed in session["seed_panels"]:
            full_rows = [row for row in seed["panel"]["fitted_models"] if row["electrodes"] == 96]
            if len(full_rows) != 5 or {row["training_trials"] for row in full_rows} != set(CHECKPOINTS):
                raise ValueError("Every seed needs all five common full-electrode cells")
            for row in full_rows:
                before = original_full[row["training_trials"]]
                if any(row[key] != before[key] for key in ("channels", "alpha", "rank", "candidate_count")):
                    raise ArithmeticError("Full-electrode channel/capacity replay differs")
                for key in ("Q", "Q_wrong", "pairing_increment"):
                    error = abs(row[key]-before[key])
                    if not np.isfinite(error) or error > 1e-10:
                        raise ArithmeticError("Full-electrode acquired-only score replay differs")
                    maximum = max(maximum, error)
        for policy in ("initial_marginal", "fixed_random"):
            session["panels"][policy] = copy.deepcopy(reference["panels"][policy])
        indices = {policy: {(cell["training_trials"], cell["electrodes"]): cell
                            for cell in session["panels"][policy]["cells"]} for policy in ALL_POLICIES}
        if any(set(index) != expected or len(session["panels"][policy]["cells"]) != 20
               for policy, index in indices.items()):
            raise ValueError("All twenty identical resource cells required")
        for policy in ALL_POLICIES:
            panel = session["panels"][policy]
            for cell in panel["cells"]:
                key = cell["training_trials"], cell["electrodes"]
                if cell["target_numeric_elements"] != key[0]*session["target_unit_count"]*10:
                    raise ValueError("Reference cell target support differs")
                for other in ALL_POLICIES:
                    cell["delta_Q_from_"+other] = cell["Q"]-indices[other][key]["Q"]
            for other in ALL_POLICIES:
                panel["mean_delta_Q_from_"+other] = float(np.mean([cell["delta_Q_from_"+other] for cell in panel["cells"]]))
    return maximum


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or {session["filename"] for session in sessions} != set(DEVELOPMENT)
                     or any(sum(session["animal"] == animal for session in sessions) != 4
                            for animal in ("Chewie", "Mihili"))):
        raise ValueError("All four original sessions per animal required")
    keys = ("mean_Q", "mean_Q_wrong", "mean_pairing_increment", *("mean_delta_Q_from_"+p for p in ALL_POLICIES))
    panels = []
    for policy in ALL_POLICIES:
        animals = [{"animal": animal, **{key: float(np.mean([session["panels"][policy][key]
                    for session in sessions if session["animal"] == animal])) for key in keys}}
                   for animal in ("Chewie", "Mihili")]
        panels.append({"policy": policy, "animals": animals,
                       **{key: float(np.mean([animal[key] for animal in animals])) for key in keys}})
    return panels
