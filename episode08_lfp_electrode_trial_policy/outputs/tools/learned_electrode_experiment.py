"""Trusted learned-selector source orchestration; unchanged qualified fitters."""
import copy
import time

import numpy as np

from acquired_only_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from learned_electrode_value import (CONTROL, MC_SEEDS, POLICY, VIEWS, donor_view,
                                    fit_ranker, mean_control_cells, rank_electrodes, view_features)
from prepare_development_trial_roles import DEVELOPMENT
from residual_selection_training import fit_selected_rank

BRANCHES = ((POLICY, None), *((CONTROL, seed) for seed in MC_SEEDS))
ALL_POLICIES = (POLICY, CONTROL, "initial_marginal", "fixed_random")


def donor_records(source_root, sources, filenames, progress, check, exposure):
    records_out = []
    for filename in filenames:
        check()
        source, role, records = sources[filename]
        channels = source["common_physical_channels"]
        if len(channels) != 96 or channels != sorted(set(channels)):
            raise ValueError("Original ascending96 physical electrodes required")
        ids = balanced_ids(role, 96)
        x, y, _ = read_trials(source_root/filename, source, role, records, ids, channels, check, exposure)
        positions = {trial: index for index, trial in enumerate(ids)}
        initial_ids = balanced_ids(role, 32)
        later_ids = [trial for pool in role["fitting_pool_by_direction"] for trial in pool[4:12]]
        if (set(initial_ids) & set(later_ids) or len(later_ids) != 64
                or set(initial_ids) | set(later_ids) != set(ids)):
            raise ValueError("Disjoint32 initializer and64 donor label rows required")
        initial = np.asarray([positions[trial] for trial in initial_ids])
        later = np.asarray([positions[trial] for trial in later_ids])
        labels = np.asarray([records[trial]["direction_index"] for trial in ids])
        folds = relative_initial_folds(role)
        views = {str(index): fit for index, (fit, _) in enumerate(folds)}
        views["outer"] = np.arange(32)
        for view in VIEWS:
            fitting = initial[views[view]]
            record = donor_view(x[fitting], y[fitting], labels[fitting], x[later], y[later], labels[later],
                                channels, filename, role["animal"], view, progress, check)
            record.update(fitting_trial_ids=[ids[index] for index in fitting], later_label_trial_ids=later_ids)
            records_out.append(record)
        del x, y
    return records_out


def fit_target(path, source, role, records, donors, progress, check, exposure):
    channels = source["common_physical_channels"]
    ids = balanced_ids(role, 32)
    if (len(channels) != 96 or channels != sorted(set(channels))
            or ids != role["nested_training_sets"]["32"]["training_trials"]):
        raise ValueError("Original all96/32-row initializer required")
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[trial]["direction_index"] for trial in ids])
    folds = relative_initial_folds(role)
    views = {str(index): fit for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    predictors = {view: view_features(x[fit], y[fit], labels[fit]) for view, fit in views.items()}
    banks, selection_records = {}, []
    # Freeze every target ordering before exposing ANY post-initial target row.
    for policy, seed in BRANCHES:
        orders = {}
        for view in VIEWS:
            check()
            state = fit_ranker(donors, role["filename"], view, seed)
            order, scores = rank_electrodes(predictors[view], channels, state)
            orders[view] = order[:32]
            selection_records.append({"policy": policy, "mc_seed": seed, "view": view,
                "fitting_trial_ids": [ids[index] for index in views[view]],
                "predictors": predictors[view].tolist(), "ranker": state,
                "predicted_values_by_physical_channel": scores.tolist(),
                "physical_channels": channels, "selected_channel_order32": [channels[index] for index in order[:32]]})
            progress["learned_ranker_fits"] += 1
        banks[policy, seed] = orders
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
    for policy, seed in BRANCHES:
        orders = banks[policy, seed]
        inner, outer = [orders[str(index)] for index in range(4)], orders["outer"]
        for count in (8, 16, 32):
            check()
            fitted, columns = fit_selected_rank(x, y, labels, folds, inner, outer, count, progress)
            selected = [channels[int(column//3)] for column in columns[::3]]
            trajectory = Trajectory(path, source, role, records, selected, check, exposure)
            trajectory.initialize_selected(x, y, channels, guide)
            for total in CHECKPOINTS:
                check()
                if total > 32:
                    fitted, _ = later_fit(trajectory, total, records, progress)
                models.append((policy, seed, total, count, tuple(selected), fitted))
        for total in CHECKPOINTS:
            models.append((policy, seed, total, 96, tuple(channels), shared[total]))
    if len(models) != 80 or len(selection_records) != 20:
        raise RuntimeError("All80model references/65actual models/20rankers required before evaluation")
    return models, nulls, guide, selection_records


def score_target(models, nulls, x, y, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = {(policy, seed): [] for policy, seed in BRANCHES}
    for policy, seed, total, count, selected, fitted in models:
        columns = [3*positions[channel]+band for channel in selected for band in range(3)]
        prediction = predict_capacity(fitted, x[:, columns], labels)
        stats = fitted.statistics
        score = prediction_scores(prediction, y, labels, stats.target_mean, stats.target_sd)
        denominator = float(np.sum(((y-stats.target_mean)/stats.target_sd)**2))
        standardized_sse = float(np.sum((((y-stats.target_mean)/stats.target_sd)-
                                          ((prediction-stats.target_mean)/stats.target_sd))**2))
        wrong_sse = (1-score["Q_wrong"])*denominator
        null = nulls[total]
        null_q = prediction_scores(null.direction_target_mean[labels], y, labels, null.target_mean, null.target_sd)["Q"]
        electrode_trials = 32*96+(total-32)*count
        rows[policy, seed].append({"training_trials": total, "electrodes": count, "mc_seed": seed,
            "channels": list(selected), "rank": fitted.rank, "alpha": fitted.alpha,
            "candidate_count": fitted.candidate_count, "null_SSE": denominator,
            "standardized_SSE": standardized_sse, "standardized_wrong_SSE": wrong_sse,
            "native_target_SSE": float(np.sum((y-prediction)**2)), **score,
            "direction_only_Q": null_q, "delta_Q_from_direction_only": score["Q"]-null_q,
            "LFP_electrode_trials": electrode_trials, "LFP_numeric_elements": electrode_trials*3*10,
            "target_numeric_elements": total*y.shape[1]*10, "initial_trials": 32,
            "initial_LFP_electrodes": 96})
    panel_by_branch = {}
    for branch, fitted_rows in rows.items():
        fitted_rows.sort(key=lambda row: (row["training_trials"], row["electrodes"]))
        cells = [{key: value for key, value in row.items() if key not in ("channels", "rank", "alpha", "candidate_count", "mc_seed")}
                 for row in fitted_rows]
        panel_by_branch[branch] = {"cells": cells, "fitted_models": fitted_rows,
            **{"mean_"+key: float(np.mean([cell[key] for cell in cells])) for key in ("Q", "Q_wrong", "pairing_increment")}}
    seed_panels = [{"mc_seed": seed, "panel": panel_by_branch[CONTROL, seed]} for seed in MC_SEEDS]
    panels = {POLICY: panel_by_branch[POLICY, None], CONTROL: mean_control_cells([item["panel"] for item in seed_panels])}
    return panels, seed_panels


def attach_reference(sessions, previous):
    if (previous["status"] != "completed_one_acquired_only_comparison_with_random_reference"
            or previous["held_session_accessed"] or {s["filename"] for s in previous["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete unchanged acquired-only reference required")
    old = {s["filename"]: s for s in previous["sessions"]}
    maximum = 0.
    for session in sessions:
        reference = old[session["filename"]]
        if reference["animal"] != session["animal"]:
            raise ValueError("Reference biological animal differs")
        full = {row["training_trials"]: row for row in reference["panels"]["initial_marginal"]["fitted_models"] if row["electrodes"] == 96}
        if set(full) != set(CHECKPOINTS):
            raise ValueError("All five complete full-electrode reference fits required")
        for panel in [session["panels"][POLICY], *(item["panel"] for item in session["seed_panels"])]:
            for row in panel["fitted_models"]:
                if row["electrodes"] != 96:
                    continue
                before = full[row["training_trials"]]
                if any(row[key] != before[key] for key in ("channels", "alpha", "rank", "candidate_count")):
                    raise ArithmeticError("Full-electrode decoder/physical support replay differs")
                for key in ("Q", "Q_wrong", "pairing_increment"):
                    error = abs(row[key]-before[key])
                    if not np.isfinite(error) or error > 1e-10:
                        raise ArithmeticError("Full-electrode score replay failed")
                    maximum = max(maximum, error)
        for policy in ("initial_marginal", "fixed_random"):
            session["panels"][policy] = copy.deepcopy(reference["panels"][policy])
        for policy, panel in session["panels"].items():
            for cell in panel["cells"]:
                for other in ALL_POLICIES:
                    matched = next(c for c in session["panels"][other]["cells"]
                                   if (c["training_trials"], c["electrodes"]) == (cell["training_trials"], cell["electrodes"]))
                    cell["delta_Q_from_"+other] = cell["Q"]-matched["Q"]
            for other in ALL_POLICIES:
                panel["mean_delta_Q_from_"+other] = float(np.mean([cell["delta_Q_from_"+other] for cell in panel["cells"]]))
    return maximum


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or {s["filename"] for s in sessions} != set(DEVELOPMENT)):
        raise ValueError("All eight sessions required for the two completed learned-selector panels")
    keys = ("mean_Q", "mean_Q_wrong", "mean_pairing_increment", *("mean_delta_Q_from_"+p for p in ALL_POLICIES))
    panels = []
    for policy in ALL_POLICIES:
        animals = [{"animal": animal, **{key: float(np.mean([s["panels"][policy][key]
                    for s in sessions if s["animal"] == animal])) for key in keys}} for animal in ("Chewie", "Mihili")]
        panels.append({"policy": policy, "animals": animals,
                       **{key: float(np.mean([animal[key] for animal in animals])) for key in keys}})
    return panels
