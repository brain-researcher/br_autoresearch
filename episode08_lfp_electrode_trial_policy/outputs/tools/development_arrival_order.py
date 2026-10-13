#!/usr/bin/env python3
"""One three-order robustness comparison for the fixed EP08 trial allocators."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from acquired_only_reader import Exposure, balanced_ids, read_trials
from allocation_reader import FittingSource
from arrival_order_reader import ORDER_REPETITIONS, OrderTrajectory, order_queues
from capacity_training import fit_capacity
from development_acquired_only import relative_initial_folds
from development_design import candidate_designs
from development_reader import selected_metadata
from development_trial_allocation import aggregate, score_session
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress
from trial_allocation import CHECKPOINTS, POLICIES, choose_direction, freeze_initial_model, fit_frozen_rank


def fit_session(path, source, role, records, progress, check, exposure):
    queues = {rep: order_queues(role, rep) for rep in ORDER_REPETITIONS}
    models = {rep: [] for rep in ORDER_REPETITIONS}
    histories = {rep: [] for rep in ORDER_REPETITIONS}
    initial_ids = balanced_ids(role, 32)
    if initial_ids != role["nested_training_sets"]["32"]["training_trials"]:
        raise ValueError("Preserve the exact original initializer and canonical order")
    labels = np.asarray([records[t]["direction_index"] for t in initial_ids])
    folds = relative_initial_folds(role)
    with FittingSource(path, source, role, records, check, exposure) as reader:
        for design in candidate_designs(role, source["common_physical_channels"], 5):
            if design["training_trials"] != 32:
                continue
            check()
            channels, count, seed = tuple(design["channels"]), design["electrodes"], design["seed"]
            initial_x, initial_y = reader.read(initial_ids, list(channels))
            selected = fit_capacity(initial_x, initial_y, labels, folds, "reduced_rank", progress)
            initial, rank = freeze_initial_model(selected, initial_x.shape[1], initial_y.shape[1])
            fixed_mean, fixed_sd = initial.statistics.target_mean.copy(), initial.statistics.target_sd.copy()
            final_model, final_arrays = None, None
            for repetition in ORDER_REPETITIONS:
                for policy in POLICIES:
                    history = OrderTrajectory(reader, channels, initial_x, initial_y, queues[repetition])
                    trace = []
                    models[repetition].append((policy, 32, count, seed, channels, initial, fixed_mean, fixed_sd, tuple(history.counts)))
                    for total in range(33, 97):
                        check()
                        _, yy, direction, _ = history.observed()
                        chosen, values = choose_direction(yy.copy(), direction.copy(), fixed_sd.copy(), history.counts.copy(), policy)
                        trial = history.acquire(chosen)
                        trace.append({"acquired_total": total, "chosen_direction": chosen, "chosen_trial": trial,
                                      "counts_before": values["counts_before"], "pre_choice_utilities": values["utility"]})
                        if total not in CHECKPOINTS:
                            continue
                        xx, yy, direction, ids = history.observed()
                        if total == 96 and final_model is not None:
                            if (ids != role["nested_training_sets"]["96"]["training_trials"]
                                    or not all(np.array_equal(a, b) for a, b in zip((xx, yy, direction), final_arrays))):
                                raise ArithmeticError("Final canonical histories differ across policies/orders")
                            fitted = final_model
                        else:
                            fitted = fit_frozen_rank(xx, yy, direction, initial.alpha, rank, progress)
                            if total == 96:
                                if ids != role["nested_training_sets"]["96"]["training_trials"]:
                                    raise ArithmeticError("The full original96trial pool is required")
                                final_model, final_arrays = fitted, (xx.copy(), yy.copy(), direction.copy())
                        models[repetition].append((policy, total, count, seed, channels, fitted, fixed_mean, fixed_sd, tuple(history.counts)))
                    histories[repetition].append({"policy": policy, "electrodes": count, "seed": seed,
                        "channels": list(channels), "initial_selected_rank": selected.rank,
                        "frozen_numeric_rank": rank, "frozen_alpha": initial.alpha,
                        "initial_candidate_count": selected.candidate_count, "choices": trace})
        if any(len(models[r]) != 240 or len(histories[r]) != 48 for r in ORDER_REPETITIONS):
            raise RuntimeError("All720references/144trajectories required before evaluation")
        guide = reader.reference
    return models, histories, queues, guide


def summary_keys():
    return tuple(prefix+name for prefix in ("mean_", "intermediate_mean_") for name in
                 ("Q_common", "Q_wrong_common", "pairing_increment", "delta_Q_common_from_balanced"))


def average_order_panels(orders):
    """Equal three-order summaries; original order never enters this average."""
    result = {}
    for policy in POLICIES:
        rows = [order["panels"][policy] for order in orders]
        cells = []
        for index, first in enumerate(rows[0]["cells"]):
            keys = tuple(key for key in first if key not in ("training_trials", "electrodes"))
            cells.append({"training_trials": first["training_trials"], "electrodes": first["electrodes"],
                          **{key: float(np.mean([row["cells"][index][key] for row in rows])) for key in keys}})
        result[policy] = {"cells": cells, **{key: float(np.mean([row[key] for row in rows])) for key in summary_keys()}}
    return result


def endpoint_replay(sessions, reference):
    if (reference["status"] != "completed_two_trial_allocators_with_balanced_reference"
            or {s["filename"] for s in reference["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete original-order reference required")
    original = {s["filename"]: s for s in reference["sessions"]}
    maximum = 0.
    for session in sessions:
        old = original[session["filename"]]
        for order in session["orders"]:
            for policy in POLICIES:
                reference_rows = {(r["training_trials"], r["electrodes"], r["seed"]): r for r in old["panels"][policy]["fitted_models"]}
                for row in order["panels"][policy]["fitted_models"]:
                    previous = reference_rows[row["training_trials"], row["electrodes"], row["seed"]]
                    if row["rank"] != previous["rank"] or row["alpha"] != previous["alpha"]:
                        raise ArithmeticError("Frozen initial numeric rank/alpha differs from original attempt")
                    if row["training_trials"] not in (32, 96):
                        continue
                    for key in ("Q_common", "Q_wrong_common", "pairing_increment"):
                        error = abs(row[key]-previous[key])
                        maximum = max(maximum, error)
                        if not np.isfinite(error) or error > 1e-10:
                            raise ArithmeticError("Shared endpoint replay failed; no score release or retuning")
    return maximum


def influence(sessions):
    """Remove one session's fixed predictions; never refit or retune."""
    rows = []
    for excluded in DEVELOPMENT:
        retained = [s for s in sessions if s["filename"] != excluded]
        panels = []
        for policy in POLICIES:
            keys = ("mean_delta_Q_common_from_balanced", "intermediate_mean_delta_Q_common_from_balanced")
            animals = [{"animal": animal, **{key: float(np.mean([s["panels"][policy][key] for s in retained if s["animal"] == animal])) for key in keys}} for animal in ("Chewie", "Mihili")]
            panels.append({"policy": policy, "animals": animals,
                           **{key: float(np.mean([animal[key] for animal in animals])) for key in keys}})
        rows.append({"excluded_session": excluded, "panels": panels})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "reference-report", "output-dir"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if not 0 < a.max_seconds <= 840 or a.output_dir.exists():
        raise ValueError("Fresh output and finite bound<=840seconds required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Exactly the eight development sessions required")
    sources = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, block_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected arrival-order robustness deadline reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress.update(capacity_inner_candidate_scores=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/trial_arrival_order_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "new_method_families": 0, "held_openings_charged": 0,
        "global_search_completed": False, "named_explanatory_amendment_activated": False,
        "raw_arrays_or_coefficients_written": False,
        "orders": list(ORDER_REPETITIONS), "seed_rule": "ep08-postinitial-order-v1|20261003|filename|direction|repetition; random.Random.seed(version=2), one shuffle",
        "only_changed_factor": "postinitializer eight-row queue ordering within each direction",
        "canonical_fitting_order": "original frozen96trial direction/ID order restricted to observed IDs",
        "boundary": "trusted chosen-next-trial/channel slices; original96fitting/40evaluation roles; copied acquired histories; same-UID is not an OS seal"}
    exposure, pending = Exposure(report), []
    try:
        for filename, source, role, records in sources:
            check()
            t = time.monotonic()
            models, histories, queues, guide = fit_session(a.source_root/filename, source, role, records, progress, check, exposure)
            fit_end = time.monotonic()
            ex, ey, _ = read_trials(a.source_root/filename, source, role, records, role["evaluation_trials"],
                                   source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.asarray([records[t]["direction_index"] for t in role["evaluation_trials"]])
            orders = [{"repetition": rep, "queues": [list(q) for q in queues[rep]],
                       "panels": score_session(models[rep], ex, ey, labels, source["common_physical_channels"]),
                       "trajectories": histories[rep]} for rep in ORDER_REPETITIONS]
            pending.append({"filename": filename, "animal": role["animal"], "orders": orders,
                "panels": average_order_panels(orders),
                "all464_actual_models_and144_trajectories_fixed_before_evaluation": True,
                "initial_final_models_shared_across_all_policies_and_orders": True,
                "training_read_and_fit_seconds": fit_end-t, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t}), flush=True)
            del models, histories, orders, ex, ey, guide
        # Previously exposed original scores enter only after every new fit.
        reference = json.loads(a.reference_report.read_text())
        report["endpoint_replay_max_absolute_difference"] = endpoint_replay(pending, reference)
        report["original_order_reference"] = {"path": str(a.reference_report), "status": reference["status"],
            "panels": reference["panels"], "fixed_prediction_session_omission": influence(reference["sessions"]),
            "fresh_confirmation": False, "refitted": False}
        report["by_order"] = []
        for rep in ORDER_REPETITIONS:
            selected = [{"filename": s["filename"], "animal": s["animal"], "panels": s["orders"][rep]["panels"]} for s in pending]
            report["by_order"].append({"repetition": rep, "panels": aggregate(selected), "fixed_prediction_session_omission": influence(selected)})
        report["panels"], report["sessions"] = aggregate(pending), pending
        report["fixed_prediction_session_omission"] = influence(pending)
        report["valid_evaluations_charged"] = 1
        report["status"] = "completed_three_order_robustness_comparison"
    except Exception as error:
        report["status"] = "technical_failure_arrival_order_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations_cache_dependent"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock-block_start
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (a.output_dir/"arrival_order.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 1, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
