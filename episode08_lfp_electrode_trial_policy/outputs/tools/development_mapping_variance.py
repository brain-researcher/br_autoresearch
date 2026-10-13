#!/usr/bin/env python3
"""One four-order acquired-LFP design-index comparison; references reused."""
import argparse
import copy
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
from arrival_order_reader import OrderTrajectory, order_queues
from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import relative_initial_folds
from development_design import ELECTRODE_COUNTS, candidate_designs
from development_reader import selected_metadata
from direction_control_metrics import prediction_scores
from mapping_variance import MappingDesign, POLICY
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress
from trial_allocation import CHECKPOINTS, POLICIES, freeze_initial_model, fit_frozen_rank

ORDERS = ("original", "0", "1", "2")
ALL_POLICIES = (*POLICIES, POLICY)
SCORE_KEYS = ("Q_common", "Q_wrong_common", "pairing_increment", "native_target_SSE",
              "initial_SD_standardized_SSE", "initial_mean_null_SSE")


def fit_session(path, source, role, records, progress, check, exposure):
    queues = {"original": tuple(tuple(p[:12]) for p in role["fitting_pool_by_direction"]),
              **{str(r): order_queues(role, r) for r in range(3)}}
    models, histories = {r: [] for r in ORDERS}, {r: [] for r in ORDERS}
    initial_ids = balanced_ids(role, 32)
    if initial_ids != role["nested_training_sets"]["32"]["training_trials"]:
        raise ValueError("Unchanged initializer IDs/order required")
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
            for repetition in ORDERS:
                history = OrderTrajectory(reader, channels, initial_x, initial_y, queues[repetition])
                proxy = MappingDesign(initial_x.copy(), labels.copy(), initial.statistics, initial.alpha)
                trace = []
                models[repetition].append((32, count, seed, channels, initial, fixed_mean, fixed_sd, tuple(history.counts)))
                for total in range(33, 97):
                    check()
                    chosen, values = proxy.choose(history.counts.copy())
                    trial = history.acquire(chosen)
                    # No source handle or future row enters the proxy. Only the
                    # row just returned by the chosen queue is copied here.
                    proxy.observe(history.x[trial].copy(), chosen)
                    progress["mapping_prechoice_decisions"] += 1
                    trace.append({"acquired_total": total, "chosen_direction": chosen, "chosen_trial": trial,
                                  "counts_before": values["counts_before"], "pre_choice_utilities": values["utility"],
                                  "design_index_before": values["design_index_before"]})
                    if total not in CHECKPOINTS:
                        continue
                    xx, yy, directions, ids = history.observed()
                    if total == 96 and final_model is not None:
                        if (ids != role["nested_training_sets"]["96"]["training_trials"]
                                or not all(np.array_equal(a, b) for a, b in zip((xx, yy, directions), final_arrays))):
                            raise ArithmeticError("Canonical final arrays differ across orders")
                        fitted = final_model
                    else:
                        fitted = fit_frozen_rank(xx, yy, directions, initial.alpha, rank, progress)
                        if total == 96:
                            if ids != role["nested_training_sets"]["96"]["training_trials"]:
                                raise ArithmeticError("Complete unchanged96trial pool required")
                            final_model, final_arrays = fitted, (xx.copy(), yy.copy(), directions.copy())
                    models[repetition].append((total, count, seed, channels, fitted, fixed_mean, fixed_sd, tuple(history.counts)))
                histories[repetition].append({"policy": POLICY, "electrodes": count, "seed": seed,
                    "channels": list(channels), "initial_selected_rank": selected.rank,
                    "frozen_numeric_rank": rank, "frozen_alpha": initial.alpha,
                    "proxy_eta": proxy.eta, "initial_candidate_count": selected.candidate_count, "choices": trace})
        if any(len(models[r]) != 80 or len(histories[r]) != 16 for r in ORDERS):
            raise RuntimeError("All320model references and64trajectories required before evaluation")
        guide = reader.reference
    return models, histories, queues, guide


def score_mapping(models, ex, ey, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = []
    for total, count, seed, selected, fitted, mean, sd, counts in models:
        columns = [3*positions[c]+band for c in selected for band in range(3)]
        prediction = predict_capacity(fitted, ex[:, columns], labels)
        raw = prediction_scores(prediction, ey, labels, mean, sd)
        rows.append({"training_trials": total, "electrodes": count, "seed": seed,
            "rank": fitted.rank, "alpha": fitted.alpha, "direction_counts": list(map(int, counts)),
            "LFP_electrode_trials": total*count, "LFP_numeric_elements": total*count*3*10,
            "target_numeric_elements": total*ey.shape[1]*10,
            "Q_common": raw["Q"], "Q_wrong_common": raw["Q_wrong"], "pairing_increment": raw["pairing_increment"],
            "native_target_SSE": float(np.square(ey-prediction).sum()),
            "initial_SD_standardized_SSE": float(np.square((ey-prediction)/sd).sum()),
            "initial_mean_null_SSE": float(np.square((ey-mean)/sd).sum())})
    cells = []
    for total in CHECKPOINTS:
        for count in ELECTRODE_COUNTS:
            selected = [r for r in rows if r["training_trials"] == total and r["electrodes"] == count]
            if len(selected) != (1 if count == 96 else 5):
                raise ValueError("Incomplete mapping resource cell")
            cells.append({"training_trials": total, "electrodes": count,
                **{key: float(np.mean([r[key] for r in selected])) for key in SCORE_KEYS},
                **{key: selected[0][key] for key in ("LFP_electrode_trials", "LFP_numeric_elements", "target_numeric_elements")}})
    return {"fitted_models": rows, "cells": cells}


def model_key(row):
    return row["training_trials"], row["electrodes"], row["seed"]


def attach_references(sessions, original, arrival):
    """Trusted reporting only, invoked after all new source reads and fits."""
    if (original["status"] != "completed_two_trial_allocators_with_balanced_reference"
            or arrival["status"] != "completed_three_order_robustness_comparison"
            or original["held_session_accessed"] or arrival["held_session_accessed"]
            or {s["filename"] for s in original["sessions"]} != set(DEVELOPMENT)
            or {s["filename"] for s in arrival["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("Complete development-only matching references required")
    originals = {s["filename"]: s for s in original["sessions"]}
    arrivals = {s["filename"]: s for s in arrival["sessions"]}
    maximum = 0.
    for session in sessions:
        old, variants = originals[session["filename"]], arrivals[session["filename"]]
        if old["animal"] != session["animal"] or variants["animal"] != session["animal"]:
            raise ValueError("Reference animal identity differs")
        for order in session["orders"]:
            reference = old if order["order"] == "original" else next(
                r for r in variants["orders"] if str(r["repetition"]) == order["order"])
            if order["order"] != "original" and order["queues"] != reference["queues"]:
                raise ValueError("Prespecified order differs from reference")
            reference_histories = {(t["electrodes"], t["seed"]): t for t in reference["trajectories"] if t["policy"] == "balanced"}
            if len(reference_histories) != 16:
                raise ValueError("Every fixed reference electrode set required")
            for trace in order["trajectories"]:
                previous = reference_histories[trace["electrodes"], trace["seed"]]
                for key in ("channels", "frozen_numeric_rank", "frozen_alpha"):
                    if trace[key] != previous[key]:
                        raise ArithmeticError("Reference electrode identity or initializer capacity differs")
                # Authenticate the original queue too, from its complete
                # recorded balanced choices; never infer order from scores.
                for choice in previous["choices"]:
                    d = choice["chosen_direction"]
                    if order["queues"][d][choice["counts_before"][d]] != choice["chosen_trial"]:
                        raise ValueError("Reference chosen-trial queue differs")
            current = order["panels"][POLICY]
            for policy in POLICIES:
                prior = reference["panels"][policy]
                indexed = {model_key(r): r for r in prior["fitted_models"]}
                if len(indexed) != 80 or len(prior["cells"]) != 20:
                    raise ValueError("All reference cells and subset seeds required")
                for row in current["fitted_models"]:
                    previous = indexed[model_key(row)]
                    if row["rank"] != previous["rank"] or row["alpha"] != previous["alpha"]:
                        raise ArithmeticError("Reference numeric rank/alpha differs")
                    # Same fixed initializer normalization for all checkpoints.
                    keys = SCORE_KEYS if row["training_trials"] in (32, 96) else ("initial_mean_null_SSE",)
                    for key in keys:
                        error = abs(row[key]-previous[key])
                        # Score replay has the adopted absolute1e-10 bound;
                        # native SSE identities are exact same arrays/decoder
                        # but use a scale-aware roundoff check.
                        limit = 1e-10 if key in SCORE_KEYS[:3] else 1e-10*max(1., abs(previous[key]))
                        if not np.isfinite(error) or error > limit:
                            raise ArithmeticError("Shared endpoint/normalization replay failed")
                        if key in SCORE_KEYS[:3]:
                            maximum = max(maximum, error)
                order["panels"][policy] = copy.deepcopy(prior)
            complete_panel_summaries(order["panels"])
        session["panels"] = average_orders(session["orders"])
    return maximum


def summary_keys():
    names = ("Q_common", "Q_wrong_common", "pairing_increment",
             *("delta_Q_common_from_"+p for p in POLICIES))
    return tuple(prefix+n for prefix in ("mean_", "intermediate_mean_") for n in names)


def complete_panel_summaries(panels):
    for policy in ALL_POLICIES:
        for index, cell in enumerate(panels[policy]["cells"]):
            for other in POLICIES:
                ref = panels[other]["cells"][index]
                if (cell["training_trials"], cell["electrodes"]) != (ref["training_trials"], ref["electrodes"]):
                    raise ValueError("Matched resource cell identities required")
                cell["delta_Q_common_from_"+other] = cell["Q_common"]-ref["Q_common"]
        for key in summary_keys():
            intermediate = key.startswith("intermediate_mean_")
            name = key.removeprefix("intermediate_mean_") if intermediate else key.removeprefix("mean_")
            rows = [c for c in panels[policy]["cells"] if not intermediate or c["training_trials"] not in (32, 96)]
            panels[policy][key] = float(np.mean([c[name] for c in rows]))


def average_orders(orders):
    if tuple(o["order"] for o in orders) != ORDERS:
        raise ValueError("All four fixed orders in declared order required")
    panels = {}
    for policy in ALL_POLICIES:
        rows = [o["panels"][policy] for o in orders]
        cells = []
        for i, first in enumerate(rows[0]["cells"]):
            cells.append({k: (first[k] if k in ("training_trials", "electrodes") else float(np.mean([r["cells"][i][k] for r in rows]))) for k in first})
        panels[policy] = {"cells": cells, **{k: float(np.mean([r[k] for r in rows])) for k in summary_keys()}}
    return panels


def aggregate(sessions, complete=True):
    if complete and (len(sessions) != 8 or any(sum(s["animal"] == a for s in sessions) != 4 for a in ("Chewie", "Mihili"))):
        raise ValueError("All eight development sessions required")
    result = []
    for policy in ALL_POLICIES:
        animals = [{"animal": a, **{k: float(np.mean([s["panels"][policy][k] for s in sessions if s["animal"] == a])) for k in summary_keys()}} for a in ("Chewie", "Mihili")]
        result.append({"policy": policy, "animals": animals, **{k: float(np.mean([a[k] for a in animals])) for k in summary_keys()}})
    return result


def influence(sessions):
    return [{"excluded_session": name, "panels": aggregate([s for s in sessions if s["filename"] != name], complete=False)} for name in DEVELOPMENT]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "original-report", "arrival-report", "output-dir"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if not 0 < a.max_seconds <= 840 or a.output_dir.exists():
        raise ValueError("Fresh output and finite bound<=840seconds required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Exactly eight development sessions required")
    sources = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, block_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected mapping comparison deadline reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress.update(capacity_inner_candidate_scores=0, mapping_prechoice_decisions=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/mapping_variance_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "global_search_completed": False,
        "named_explanatory_amendment_activated": False, "raw_arrays_or_coefficients_written": False,
        "orders": list(ORDERS), "policy": POLICY, "proxy_unit": "dimensionless_unit_noise_design_index",
        "proxy_limit": "frozen full-linear design approximation, not calibrated uncertainty or actual reduced-rank decoder variance",
        "evaluation_normalization": "fixed common32trial rawY_SD and global_mean_null",
        "boundary": "trusted chosen next fitting trial/channels only; copied acquired proxy; same UID is not OS isolation"}
    exposure, pending = Exposure(report), []
    try:
        for filename, source, role, records in sources:
            check()
            start = time.monotonic()
            models, histories, queues, guide = fit_session(a.source_root/filename, source, role, records, progress, check, exposure)
            fit_end = time.monotonic()
            ex, ey, _ = read_trials(a.source_root/filename, source, role, records, role["evaluation_trials"],
                source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.asarray([records[t]["direction_index"] for t in role["evaluation_trials"]])
            orders = [{"order": r, "queues": [list(q) for q in queues[r]],
                "panels": {POLICY: score_mapping(models[r], ex, ey, labels, source["common_physical_channels"])},
                "trajectories": histories[r]} for r in ORDERS]
            pending.append({"filename": filename, "animal": role["animal"], "orders": orders,
                "all224_actual_models_and64_trajectories_fixed_before_evaluation": True,
                "initial_final_models_shared_across_orders": True,
                "training_read_and_fit_seconds": fit_end-start, "elapsed_seconds": time.monotonic()-start})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-start}), flush=True)
            del models, histories, orders, ex, ey, guide
        # Reference scores/choices never enter any fitter or proxy. No new
        # fit or action is permitted after these reads.
        original, arrival = json.loads(a.original_report.read_text()), json.loads(a.arrival_report.read_text())
        report["endpoint_replay_max_absolute_difference"] = attach_references(pending, original, arrival)
        report["reference_reuse"] = [{"path": str(path), "status": old["status"], "refitted": False,
            "logical_neural_bytes_previously_charged": old["logical_neural_bytes"], "elapsed_seconds_previously_charged": old["elapsed_seconds"]} for path, old in ((a.original_report, original), (a.arrival_report, arrival))]
        report["by_order"] = []
        for r in ORDERS:
            selected = [{"filename": s["filename"], "animal": s["animal"], "panels": next(o["panels"] for o in s["orders"] if o["order"] == r)} for s in pending]
            report["by_order"].append({"order": r, "panels": aggregate(selected), "fixed_prediction_session_omission": influence(selected)})
        report["panels"], report["sessions"] = aggregate(pending), pending
        report["fixed_prediction_session_omission"] = influence(pending)
        if (progress["inner_ridge_fits"] != 2560 or progress["outer_training_refits"] != 1792
                or progress["mapping_prechoice_decisions"] != 32768
                or report["fitting_trial_read_calls"] != 36864 or report["evaluation_trials_read"] != 320):
            raise ArithmeticError("Complete adopted model/read count differs; no completed comparison")
        report["valid_evaluations_charged"], report["valid_evaluations_total"] = 1, 20
        report["status"] = "completed_mapping_design_variance_comparison"
    except Exception as error:
        report["status"] = "technical_failure_mapping_comparison_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations_cache_dependent"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock-block_start
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (a.output_dir/"mapping_variance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 1, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
