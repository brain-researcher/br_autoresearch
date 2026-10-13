"""Frozen initializer conditional-gain selection with qualified true-Y decoders."""
import copy

import numpy as np

from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from initializer_target_label_control import CONTROL as PREVIOUS_CONTROL
from acquired_only_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from initializer_conditional_gain import BRANCHES, CONTROL, MC_SEEDS, POLICY, mean_control_cells, order_bank
from initializer_repeatability import POLICY as REPEATABILITY, CONTROL as UNIFORM
from residual_selection_training import fit_selected_rank
from prepare_development_trial_roles import DEVELOPMENT

ALL_POLICIES = (POLICY, CONTROL, "initial_marginal", PREVIOUS_CONTROL, REPEATABILITY, UNIFORM, "fixed_random")


def fit_target(path, source, role, records, progress, check, exposure):
    channels, ids = source["common_physical_channels"], balanced_ids(role, 32)
    if (len(channels) != 96 or channels != sorted(set(channels))
            or ids != role["nested_training_sets"]["32"]["training_trials"]):
        raise ValueError("Original96-electrode/32-row initializer required")
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[trial]["direction_index"] for trial in ids])
    folds = relative_initial_folds(role)
    banks, selections = order_bank(x, y, labels, ids, channels, folds, role["filename"], progress, check)
    if len(selections) != 30 or set(banks) != set(BRANCHES):
        raise ArithmeticError("All30 orders required before future reads")
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
    for seed in BRANCHES:
        orders = banks[seed]
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
                models.append((seed, total, count, tuple(selected), fitted))
        for total in CHECKPOINTS:
            models.append((seed, total, 96, tuple(channels), shared[total]))
    if len(models) != 120:
        raise ArithmeticError("All120 model references/95 distinct models required before evaluation")
    return models, nulls, guide, selections


def score_target(models, nulls, x, y, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = {seed: [] for seed in BRANCHES}
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
        null_q = prediction_scores(null.direction_target_mean[labels], y, labels, null.target_mean, null.target_sd)["Q"]
        electrode_trials = 32*96+(total-32)*count
        rows[seed].append({"training_trials": total, "electrodes": count, "mc_seed": seed,
            "channels": list(selected), "rank": fitted.rank, "alpha": fitted.alpha,
            "candidate_count": fitted.candidate_count, "null_SSE": denominator,
            "standardized_SSE": correct, "standardized_wrong_SSE": wrong,
            "native_target_SSE": float(np.sum((y-prediction)**2)), **score,
            "direction_only_Q": null_q, "delta_Q_from_direction_only": score["Q"]-null_q,
            "LFP_electrode_trials": electrode_trials, "LFP_numeric_elements": electrode_trials*30,
            "target_numeric_elements": total*y.shape[1]*10, "initial_trials": 32, "initial_LFP_electrodes": 96,
            "input_coordinates_per_electrode": 3})
    branch_panels = {}
    for seed, fitted_rows in rows.items():
        fitted_rows.sort(key=lambda row: (row["training_trials"], row["electrodes"]))
        cells = [{key: value for key, value in row.items()
                  if key not in ("channels", "rank", "alpha", "candidate_count", "mc_seed")}
                 for row in fitted_rows]
        branch_panels[seed] = {"cells": cells, "fitted_models": fitted_rows,
            **{"mean_"+key: float(np.mean([cell[key] for cell in cells]))
               for key in ("Q", "Q_wrong", "pairing_increment", "delta_Q_from_direction_only")}}
    seeds = [{"mc_seed": seed, "panel": branch_panels[seed]} for seed in MC_SEEDS]
    control = mean_control_cells([entry["panel"] for entry in seeds])
    control["mean_delta_Q_from_direction_only"] = float(np.mean([cell["delta_Q_from_direction_only"] for cell in control["cells"]]))
    return {POLICY: branch_panels[None], CONTROL: control}, seeds


def attach_references(sessions, previous, uniform):
    if (previous["status"] != "completed_one_initializer_target_label_control" or previous["held_session_accessed"]
            or previous["control"] != PREVIOUS_CONTROL or previous["MC_seeds"] != list(MC_SEEDS)
            or len(previous["sessions"]) != 8
            or {session["filename"] for session in previous["sessions"]} != set(DEVELOPMENT)
            or uniform["status"] != "completed_two_initializer_repeatability_comparisons"
            or uniform["held_session_accessed"] or uniform["recipe"] != "outputs/initializer_repeatability_proposal.md"
            or uniform["policy"] != REPEATABILITY or uniform["control"] != UNIFORM
            or uniform["MC_seeds"] != list(MC_SEEDS) or len(uniform["sessions"]) != 8
            or {session["filename"] for session in uniform["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete original target-label and matched uniform/repeatability references required")
    old = {session["filename"]: session for session in previous["sessions"]}
    first = {session["filename"]: session for session in uniform["sessions"]}
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    maximum, target_max = 0., 0.
    for session in sessions:
        reference, original = old[session["filename"]], first[session["filename"]]
        if (reference["animal"] != session["animal"] or original["animal"] != session["animal"]
                or reference["source_roles"] != session["source_roles"]
                or reference["target_unit_count"] != session["target_unit_count"]
                or original["source_roles"] != session["source_roles"]
                or original["target_unit_count"] != session["target_unit_count"]):
            raise ValueError("Original animal/trial/electrode/target-count support differs")
        old_maps = {(record["mc_seed"], record["view"]): record["target_label_permutations"]
                    for record in reference["selection_records"]}
        new_maps = {(record["mc_seed"], record["view"]): record["target_label_permutations"]
                    for record in session["selection_records"] if record["mc_seed"] is not None}
        if len(new_maps) != 25 or new_maps != old_maps:
            raise ArithmeticError("Original initializer target-label view/direction maps differ")
        original_cells = {(cell["training_trials"], cell["electrodes"]): cell
                          for cell in reference["panels"][PREVIOUS_CONTROL]["cells"]}
        candidate_full = {row["training_trials"]: row for row in session["panels"][POLICY]["fitted_models"]
                          if row["electrodes"] == 96}
        historical_full = {row["training_trials"]: row for row in reference["seed_panels"][0]["panel"]["fitted_models"]
                           if row["electrodes"] == 96}
        if (set(candidate_full) != set(CHECKPOINTS) or set(historical_full) != set(CHECKPOINTS)
                or set(original_cells) != expected):
            raise ValueError("Complete new full-electrode and old reference cells required")
        for panel in [session["panels"][POLICY], *(seed["panel"] for seed in session["seed_panels"])]:
            if len(panel["fitted_models"]) != 20:
                raise ValueError("Every new branch needs twenty fitted models")
            for row in panel["fitted_models"]:
                n, e = row["training_trials"], row["electrodes"]
                before = original_cells[n, e]
                if (row["target_numeric_elements"] != before["target_numeric_elements"]
                        or row["LFP_electrode_trials"] != before["LFP_electrode_trials"]
                        or row["LFP_numeric_elements"] != before["LFP_numeric_elements"]):
                    raise ValueError("Target/physical-channel or three-band exposure differs")
                if not np.isclose(row["null_SSE"], before["null_SSE"], rtol=1e-10, atol=1e-10):
                    raise ArithmeticError("Representation-independent target denominator differs")
                difference = abs(row["direction_only_Q"]-before["direction_only_Q"])
                if not np.isfinite(difference) or difference > 1e-10:
                    raise ArithmeticError("Representation-independent direction-null score differs")
                target_max = max(target_max, difference)
                if e == 96:
                    for matched in (candidate_full[n], historical_full[n]):
                        if any(row[key] != matched[key] for key in ("channels", "rank", "alpha", "candidate_count")):
                            raise ArithmeticError("Full-electrode capacity/support replay differs")
                        for key in ("Q", "Q_wrong", "pairing_increment"):
                            error = abs(row[key]-matched[key])
                            if not np.isfinite(error) or error > 1e-10:
                                raise ArithmeticError("Full-electrode score replay differs")
                            maximum = max(maximum, error)
        for policy in ("initial_marginal", "fixed_random"):
            if reference["panels"][policy]["fitted_models"] != original["panels"][policy]["fitted_models"]:
                raise ValueError("Reused acquired-only fitted-score records disagree")
            session["panels"][policy] = copy.deepcopy(reference["panels"][policy])
        session["panels"][PREVIOUS_CONTROL] = copy.deepcopy(reference["panels"][PREVIOUS_CONTROL])
        for policy in (REPEATABILITY, UNIFORM):
            session["panels"][policy] = copy.deepcopy(original["panels"][policy])
        indices = {policy: {(cell["training_trials"], cell["electrodes"]): cell
                            for cell in session["panels"][policy]["cells"]} for policy in ALL_POLICIES}
        if any(set(index) != expected or len(session["panels"][policy]["cells"]) != 20
               for policy, index in indices.items()):
            raise ValueError("All twenty matching resource cells required")
        for policy in ALL_POLICIES:
            panel = session["panels"][policy]
            for cell in panel["cells"]:
                key = cell["training_trials"], cell["electrodes"]
                for other in ALL_POLICIES:
                    cell["delta_Q_from_"+other] = cell["Q"]-indices[other][key]["Q"]
            for other in ALL_POLICIES:
                panel["mean_delta_Q_from_"+other] = float(np.mean([cell["delta_Q_from_"+other] for cell in panel["cells"]]))
            panel["mean_delta_Q_from_direction_only"] = float(np.mean([cell["delta_Q_from_direction_only"] for cell in panel["cells"]]))
    return maximum, target_max


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or {session["filename"] for session in sessions} != set(DEVELOPMENT)
                     or any(sum(session["animal"] == animal for session in sessions) != 4
                            for animal in ("Chewie", "Mihili"))):
        raise ValueError("All four original development sessions per animal required")
    keys = ("mean_Q", "mean_Q_wrong", "mean_pairing_increment", "mean_delta_Q_from_direction_only",
            *("mean_delta_Q_from_"+policy for policy in ALL_POLICIES))
    panels = []
    for policy in ALL_POLICIES:
        animals = [{"animal": animal, **{key: float(np.mean([session["panels"][policy][key]
                    for session in sessions if session["animal"] == animal])) for key in keys}}
                   for animal in ("Chewie", "Mihili")]
        panels.append({"policy": policy, "animals": animals,
                       **{key: float(np.mean([animal[key] for animal in animals])) for key in keys}})
    return panels
