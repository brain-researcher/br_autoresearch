"""Trusted target fitting with shared other-animal rankers and original fitters."""
import copy

import numpy as np

from acquired_only_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_training import fit_direction_null
from learned_electrode_experiment import attach_reference as original_reference
from learned_electrode_experiment import score_target as original_score_target
from learned_electrode_value import CONTROL as PREVIOUS_CONTROL, POLICY as PREVIOUS_POLICY, VIEWS, view_features
from prepare_development_trial_roles import DEVELOPMENT
from residual_selection_training import fit_selected_rank
from reverse_animal_electrode_value import BRANCHES, CONTROL, MC_SEEDS, POLICY, rank_electrodes

ALL_POLICIES = (POLICY, CONTROL, "initial_marginal", PREVIOUS_POLICY, PREVIOUS_CONTROL, "fixed_random")


def fit_target(path, source, role, records, rankers, progress, check, exposure):
    channels, ids = source["common_physical_channels"], balanced_ids(role, 32)
    if (len(channels) != 96 or channels != sorted(set(channels))
            or ids != role["nested_training_sets"]["32"]["training_trials"]):
        raise ValueError("Original ninety-six-channel/thirty-two-trial initializer required")
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[trial]["direction_index"] for trial in ids])
    folds = relative_initial_folds(role)
    views = {str(index): fit for index, (fit, _) in enumerate(folds)}
    views["outer"] = np.arange(32)
    predictors = {view: view_features(x[fit], y[fit], labels[fit]) for view, fit in views.items()}
    order_bank, selections = {}, []
    # All20 orderings precede every post-initial target read. Donor states were
    # already fitted without any target animal's numerical training inputs.
    for policy, seed in BRANCHES:
        orders = {}
        for view in VIEWS:
            check()
            state = rankers[role["animal"], view, seed]
            if state["recipient_animal"] != role["animal"] or role["filename"] in state["donor_sessions"]:
                raise ValueError("Other-animal ranker identity differs from target role")
            order, scores = rank_electrodes(predictors[view], channels, state)
            orders[view] = order[:32]
            selections.append({"policy": policy, "mc_seed": seed, "view": view,
                "ranker_key": [role["animal"], view, seed],
                "fitting_trial_ids": [ids[index] for index in views[view]],
                "predictors": predictors[view].tolist(), "physical_channels": channels,
                "predicted_values_by_physical_channel": scores.tolist(),
                "selected_channel_order32": [channels[index] for index in order[:32]]})
            progress["target_orderings"] += 1
        order_bank[policy, seed] = orders
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
        orders = order_bank[policy, seed]
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
    if len(models) != 80 or len(selections) != 20:
        raise ArithmeticError("All eighty references/sixty-five models/twenty target orderings required")
    return models, nulls, guide, selections


def score_target(models, nulls, x, y, labels, channels):
    # Translate policy identifiers only. The qualified scorer receives the new
    # fitted objects/arrays and retains its exact score and mean-SSE equations.
    aliases = {POLICY: PREVIOUS_POLICY, CONTROL: PREVIOUS_CONTROL}
    translated = [(aliases[policy], seed, total, count, selected, fitted)
                  for policy, seed, total, count, selected, fitted in models]
    panels, seeds = original_score_target(translated, nulls, x, y, labels, channels)
    return {POLICY: panels[PREVIOUS_POLICY], CONTROL: panels[PREVIOUS_CONTROL]}, seeds


def attach_references(sessions, acquired, previous):
    # Reuse the original acquired-only validator on an explicit identifier view;
    # completed scientific packets are read by the runner after every new fit.
    views = [{"filename": s["filename"], "animal": s["animal"],
              "panels": {PREVIOUS_POLICY: copy.deepcopy(s["panels"][POLICY]),
                         PREVIOUS_CONTROL: copy.deepcopy(s["panels"][CONTROL])},
              "seed_panels": copy.deepcopy(s["seed_panels"])} for s in sessions]
    maximum = original_reference(views, acquired)
    if (previous["status"] != "completed_two_learned_selector_comparisons"
            or previous["held_session_accessed"] or previous["policy"] != PREVIOUS_POLICY
            or previous["control"] != PREVIOUS_CONTROL or previous["MC_seeds"] != list(MC_SEEDS)
            or len(previous["sessions"]) != 8
            or {s["filename"] for s in previous["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete unchanged seven-donor learned-selector reference required")
    old = {s["filename"]: s for s in previous["sessions"]}
    for session, validated in zip(sessions, views):
        before = old[session["filename"]]
        if before["animal"] != session["animal"] or before["source_roles"] != session["source_roles"]:
            raise ValueError("Completed learned reference source/support roles differ")
        full = {r["training_trials"]: r for r in before["panels"][PREVIOUS_POLICY]["fitted_models"] if r["electrodes"] == 96}
        if set(full) != set(CHECKPOINTS):
            raise ValueError("All five learned-reference full-electrode checkpoints required")
        for panel in [session["panels"][POLICY], *(seed["panel"] for seed in session["seed_panels"])]:
            for row in panel["fitted_models"]:
                if row["electrodes"] != 96:
                    continue
                original = full[row["training_trials"]]
                if any(row[key] != original[key] for key in ("channels", "rank", "alpha", "candidate_count")):
                    raise ArithmeticError("Full-electrode learned-reference capacity/channel replay differs")
                for key in ("Q", "Q_wrong", "pairing_increment"):
                    error = abs(row[key]-original[key])
                    if not np.isfinite(error) or error > 1e-10:
                        raise ArithmeticError("Learned-reference full-electrode score replay differs")
                    maximum = max(maximum, error)
                for key in ("null_SSE", "standardized_SSE", "standardized_wrong_SSE", "native_target_SSE"):
                    if not np.isclose(row[key], original[key], rtol=1e-10, atol=1e-10):
                        raise ArithmeticError("Learned-reference full-electrode error/normalizer replay differs")
        for policy in ("initial_marginal", "fixed_random"):
            session["panels"][policy] = validated["panels"][policy]
        for policy in (PREVIOUS_POLICY, PREVIOUS_CONTROL):
            session["panels"][policy] = copy.deepcopy(before["panels"][policy])
        indices = {policy: {(c["training_trials"], c["electrodes"]): c for c in session["panels"][policy]["cells"]}
                   for policy in ALL_POLICIES}
        expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
        if any(set(index) != expected or len(session["panels"][policy]["cells"]) != 20 for policy, index in indices.items()):
            raise ValueError("All twenty matching resource cells required for every reference")
        for policy in ALL_POLICIES:
            panel = session["panels"][policy]
            for cell in panel["cells"]:
                key = cell["training_trials"], cell["electrodes"]
                for other in ALL_POLICIES:
                    cell["delta_Q_from_"+other] = cell["Q"]-indices[other][key]["Q"]
            for other in ALL_POLICIES:
                panel["mean_delta_Q_from_"+other] = float(np.mean([c["delta_Q_from_"+other] for c in panel["cells"]]))
    return maximum


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or {s["filename"] for s in sessions} != set(DEVELOPMENT)):
        raise ValueError("Both full four-session transfer directions required")
    keys = ("mean_Q", "mean_Q_wrong", "mean_pairing_increment", *("mean_delta_Q_from_"+p for p in ALL_POLICIES))
    panels = []
    for policy in ALL_POLICIES:
        animals = [{"animal": animal, **{key: float(np.mean([s["panels"][policy][key]
                    for s in sessions if s["animal"] == animal])) for key in keys}} for animal in ("Chewie", "Mihili")]
        panels.append({"policy": policy, "animals": animals,
                       **{key: float(np.mean([a[key] for a in animals])) for key in keys}})
    return panels
