"""Three frozen initializer conditions with qualified true-Y decoders."""
import copy

import numpy as np

from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import later_fit, relative_initial_folds
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from initializer_target_label_control import CONTROL as PREVIOUS_CONTROL
from acquired_only_reader import CHECKPOINTS, Trajectory, balanced_ids, read_trials
from initializer_robustness import BRANCHES, CONTROL, MC_SEEDS, POLICY, mean_control_cells, order_bank
from residual_selection_training import fit_selected_rank
from prepare_development_trial_roles import DEVELOPMENT

from initializer_robustness_roles import ORDERS
from development_learned_electrodes import role_record

ALL_POLICIES = (POLICY, CONTROL)


def fit_target(path, source, role, records, progress, check, exposure, order):
    channels, ids = source["common_physical_channels"], balanced_ids(role, 32)
    if (order not in ORDERS or role.get("initializer_order") != order
            or len(channels) != 96 or channels != sorted(set(channels))
            or ids != role["nested_training_sets"]["32"]["training_trials"]):
        raise ValueError("Original96-electrode/32-row initializer required")
    x, y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[trial]["direction_index"] for trial in ids])
    folds = relative_initial_folds(role)
    banks, selections = order_bank(x, y, labels, ids, channels, folds, role["filename"], order, progress, check)
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


def add_contrasts(panels):
    indices = {p: {(c["training_trials"], c["electrodes"]): c for c in panels[p]["cells"]}
               for p in ALL_POLICIES}
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    if any(len(panels[p]["cells"]) != 20 or set(index) != expected for p, index in indices.items()):
        raise ValueError("All twenty matching cells required")
    for policy in ALL_POLICIES:
        panel = panels[policy]
        for key, cell in indices[policy].items():
            for other in ALL_POLICIES:
                cell["delta_Q_from_"+other] = cell["Q"]-indices[other][key]["Q"]
        for other in ALL_POLICIES:
            panel["mean_delta_Q_from_"+other] = float(np.mean([c["delta_Q_from_"+other] for c in panel["cells"]]))
    return panels


def validate_order_panels(panels, seed_panels, target_unit_count):
    if [s["mc_seed"] for s in seed_panels] != list(MC_SEEDS):
        raise ValueError("All five fixed seeds required within each initializer order")
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    candidate = {(c["training_trials"], c["electrodes"]): c for c in panels[POLICY]["cells"]}
    reference = {r["training_trials"]: r for r in panels[POLICY]["fitted_models"] if r["electrodes"] == 96}
    branches = [panels[POLICY], *(s["panel"] for s in seed_panels)]
    for panel in [*branches, panels[CONTROL]]:
        if len(panel["cells"]) != 20 or {(r["training_trials"], r["electrodes"]) for r in panel["cells"]} != expected:
            raise ValueError("Every within-order branch requires twenty cells")
        for row in panel["cells"]:
            n, e = row["training_trials"], row["electrodes"]
            before = candidate[n, e]
            if (not np.isfinite(row["null_SSE"]) or row["null_SSE"] <= 0
                    or row["null_SSE"] != before["null_SSE"]
                    or row["direction_only_Q"] != before["direction_only_Q"]
                    or row["target_numeric_elements"] != n*target_unit_count*10
                    or row["LFP_electrode_trials"] != 32*96+(n-32)*e
                    or row["LFP_numeric_elements"] != (32*96+(n-32)*e)*30):
                raise ArithmeticError("Common within-order target denominator/direction null/exposure required")
            for key in ("Q", "Q_wrong", "pairing_increment", "standardized_SSE", "standardized_wrong_SSE"):
                if not np.isfinite(row[key]):
                    raise ArithmeticError("Finite complete per-order scores required")
    for panel in branches:
        rows = panel["fitted_models"]
        if len(rows) != 20 or {(r["training_trials"], r["electrodes"]) for r in rows} != expected:
            raise ValueError("All twenty actual within-order model records required")
        for row in rows:
            if row["electrodes"] == 96:
                before = reference[row["training_trials"]]
                if any(row[key] != before[key] for key in
                       ("channels", "rank", "alpha", "candidate_count", "Q", "Q_wrong", "pairing_increment",
                        "null_SSE", "standardized_SSE", "standardized_wrong_SSE")):
                    raise ArithmeticError("Exact full96 identity required within the current initializer order")
    return 0.


def average_orders(order_results):
    if [r["initializer_order"] for r in order_results] != list(ORDERS):
        raise ValueError("All three fixed initializer orders required without exclusion")
    score_fields = ("Q", "Q_wrong", "pairing_increment", "direction_only_Q", "delta_Q_from_direction_only")
    cost_fields = ("training_trials", "electrodes", "LFP_electrode_trials", "LFP_numeric_elements",
                   "target_numeric_elements", "initial_trials", "initial_LFP_electrodes",
                   "input_coordinates_per_electrode")
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    def average(panels):
        indices = [{(c["training_trials"], c["electrodes"]): c for c in p["cells"]} for p in panels]
        if any(len(p["cells"]) != 20 or set(index) != expected for p, index in zip(panels, indices)):
            raise ValueError("Every order needs twenty cells before mean-Q aggregation")
        cells = []
        for key in sorted(expected):
            rows = [index[key] for index in indices]
            if any(any(row[field] != rows[0][field] for field in cost_fields) for row in rows):
                raise ValueError("Averaged order cells must have matched scientific acquisition costs")
            if any(not np.isfinite(row[field]) for row in rows for field in score_fields):
                raise ArithmeticError("Nonfinite order score; no order dropping")
            cells.append({**{field: rows[0][field] for field in cost_fields},
                          **{field: float(np.mean([row[field] for row in rows])) for field in score_fields}})
        return {"cells": cells, "initializer_orders": list(ORDERS),
                "aggregation": "equal mean of three order-specific Q scores; no common SSE denominator",
                **{"mean_"+field: float(np.mean([c[field] for c in cells])) for field in score_fields}}
    panels = {policy: average([r["panels"][policy] for r in order_results]) for policy in ALL_POLICIES}
    seeds = []
    for seed in MC_SEEDS:
        if any([entry["mc_seed"] for entry in r["seed_panels"]] != list(MC_SEEDS) for r in order_results):
            raise ValueError("All five seeds in every order required")
        seeds.append({"mc_seed": seed, "panel": average([r["seed_panels"][seed]["panel"] for r in order_results])})
    return add_contrasts(panels), seeds


def attach_original_context(sessions, previous, original_sources):
    if (previous["status"] != "completed_one_initializer_target_label_control"
            or previous["recipe"] != "outputs/initializer_target_label_control_proposal.md"
            or previous["held_session_accessed"] or previous["control"] != PREVIOUS_CONTROL
            or previous["MC_seeds"] != list(MC_SEEDS) or len(previous["sessions"]) != 8
            or {s["filename"] for s in previous["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete original initializer reference required for separate context")
    original = {s["filename"]: s for s in previous["sessions"]}
    expected = {(n, e) for n in CHECKPOINTS for e in (8, 16, 32, 96)}
    for session in sessions:
        old = original[session["filename"]]
        source, role, _ = original_sources[session["filename"]]
        if (old["source_roles"] != role_record(source, role) or old["animal"] != session["animal"]
                or old["target_unit_count"] != session["target_unit_count"]):
            raise ValueError("Original context must retain its original roles and native target support")
        context = {}
        for policy in ("initial_marginal", PREVIOUS_CONTROL):
            panel = old["panels"][policy]
            if len(panel["cells"]) != 20 or {(c["training_trials"], c["electrodes"]) for c in panel["cells"]} != expected:
                raise ValueError("Complete original twenty-cell context required")
            for cell in panel["cells"]:
                n, e = cell["training_trials"], cell["electrodes"]
                if (cell["target_numeric_elements"] != n*session["target_unit_count"]*10
                        or cell["LFP_electrode_trials"] != 32*96+(n-32)*e
                        or cell["LFP_numeric_elements"] != (32*96+(n-32)*e)*30
                        or not all(np.isfinite(cell[k]) for k in ("Q", "Q_wrong", "pairing_increment"))):
                    raise ValueError("Original context target support/exposure/score invalid")
            context[policy] = {"cells": copy.deepcopy(panel["cells"]),
                               **{k: v for k, v in panel.items() if k.startswith("mean_")}}
        session["original_initializer_context"] = context
    # Deliberately no old-versus-new score, capacity, denominator or source-role equality.


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
