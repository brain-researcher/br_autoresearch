#!/usr/bin/env python3
"""Trusted finite-pool allocator comparison with fixed random electrodes."""
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
from allocation_reader import AllocationTrajectory, FittingSource
from capacity_prediction import predict_capacity
from capacity_training import fit_capacity
from development_acquired_only import relative_initial_folds
from development_design import ELECTRODE_COUNTS, candidate_designs
from development_reader import selected_metadata
from direction_control_metrics import prediction_scores
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress
from trial_allocation import CHECKPOINTS, POLICIES, choose_direction, freeze_initial_model, fit_frozen_rank


def fit_session(path, source, role, records, progress, check, exposure):
    models, trajectories = [], []
    with FittingSource(path, source, role, records, check, exposure) as reader:
        initial_ids = balanced_ids(role, 32)
        if initial_ids != role["nested_training_sets"]["32"]["training_trials"]:
            raise ValueError("The unchanged initializer order is required")
        labels = np.asarray([records[t]["direction_index"] for t in initial_ids])
        folds = relative_initial_folds(role)
        for design in candidate_designs(role, source["common_physical_channels"], 5):
            if design["training_trials"] != 32:
                continue
            check()
            channels, count, seed = tuple(design["channels"]), design["electrodes"], design["seed"]
            initial_x, initial_y = reader.read(initial_ids, list(channels))
            selected = fit_capacity(initial_x, initial_y, labels, folds, "reduced_rank", progress)
            initial_model, rank = freeze_initial_model(selected, initial_x.shape[1], initial_y.shape[1])
            fixed_mean = initial_model.statistics.target_mean.copy()
            fixed_sd = initial_model.statistics.target_sd.copy()
            final_model, final_arrays = None, None
            for policy in POLICIES:
                history = AllocationTrajectory(reader, channels, initial_x, initial_y)
                trace = []
                models.append((policy, 32, count, seed, channels, initial_model, fixed_mean, fixed_sd, tuple(history.counts)))
                for total in range(33, 97):
                    check()
                    observed_x, observed_y, observed_labels, _ = history.observed()
                    chosen, values = choose_direction(observed_y.copy(), observed_labels.copy(), fixed_sd.copy(), history.counts.copy(), policy)
                    trial = history.acquire(chosen)
                    trace.append({"acquired_total": total, "chosen_direction": chosen, "chosen_trial": trial,
                                  "counts_before": values["counts_before"], "pre_choice_utilities": values["utility"]})
                    if total not in CHECKPOINTS:
                        continue
                    xx, yy, direction, ids = history.observed()
                    if total == 96 and final_model is not None:
                        if (ids != role["nested_training_sets"]["96"]["training_trials"]
                                or not all(np.array_equal(left, right) for left, right in zip((xx, yy, direction), final_arrays))):
                            raise ArithmeticError("Final acquired histories differ; no score release")
                        fitted = final_model
                    else:
                        fitted = fit_frozen_rank(xx, yy, direction, initial_model.alpha, rank, progress)
                        if total == 96:
                            if ids != role["nested_training_sets"]["96"]["training_trials"]:
                                raise ArithmeticError("The final pool must exactly equal the original96trials")
                            final_model, final_arrays = fitted, (xx.copy(), yy.copy(), direction.copy())
                    models.append((policy, total, count, seed, channels, fitted, fixed_mean, fixed_sd, tuple(history.counts)))
                trajectories.append({"policy": policy, "electrodes": count, "seed": seed, "channels": list(channels),
                    "initial_selected_rank": selected.rank, "frozen_numeric_rank": rank, "frozen_alpha": initial_model.alpha,
                    "initial_candidate_count": selected.candidate_count, "choices": trace})
        if len(models) != 240 or len(trajectories) != 48:
            raise RuntimeError("All240model references/48trajectories required before evaluation")
        guide = reader.reference
    return models, trajectories, guide


def score_session(models, ex, ey, labels, channels):
    positions = {channel: index for index, channel in enumerate(channels)}
    rows = {policy: [] for policy in POLICIES}
    shared_scores = {}
    for policy, total, count, seed, selected, fitted, fixed_mean, fixed_sd, counts in models:
        key = total, count, seed
        if total in (32, 96) and key in shared_scores:
            score = shared_scores[key]
        else:
            columns = [3*positions[c]+band for c in selected for band in range(3)]
            prediction = predict_capacity(fitted, ex[:, columns], labels)
            raw = prediction_scores(prediction, ey, labels, fixed_mean, fixed_sd)
            score = {"Q_common": raw["Q"], "Q_wrong_common": raw["Q_wrong"], "pairing_increment": raw["pairing_increment"],
                     "native_target_SSE": float(np.square(ey-prediction).sum()),
                     "initial_SD_standardized_SSE": float(np.square((ey-prediction)/fixed_sd).sum()),
                     "initial_mean_null_SSE": float(np.square((ey-fixed_mean)/fixed_sd).sum())}
            if total in (32, 96):
                shared_scores[key] = score
        rows[policy].append({"training_trials": total, "electrodes": count, "seed": seed,
            "rank": fitted.rank, "alpha": fitted.alpha, "direction_counts": list(map(int, counts)),
            "LFP_electrode_trials": total*count, "LFP_numeric_elements": total*count*3*10,
            "target_numeric_elements": total*ey.shape[1]*10, **score})
    keys = ("Q_common", "Q_wrong_common", "pairing_increment", "native_target_SSE", "initial_SD_standardized_SSE", "initial_mean_null_SSE")
    panels = {}
    for policy in POLICIES:
        cells = []
        for total in CHECKPOINTS:
            for count in ELECTRODE_COUNTS:
                selected = [r for r in rows[policy] if r["training_trials"] == total and r["electrodes"] == count]
                if len(selected) != (1 if count == 96 else 5):
                    raise RuntimeError("Incomplete allocator resource cell")
                cells.append({"training_trials": total, "electrodes": count,
                    **{key: float(np.mean([r[key] for r in selected])) for key in keys},
                    **{key: selected[0][key] for key in ("LFP_electrode_trials", "LFP_numeric_elements", "target_numeric_elements")}})
        panels[policy] = {"cells": cells, "fitted_models": rows[policy]}
    for policy in POLICIES:
        for cell, reference in zip(panels[policy]["cells"], panels["balanced"]["cells"]):
            cell["delta_Q_common_from_balanced"] = cell["Q_common"]-reference["Q_common"]
            if cell["training_trials"] in (32, 96) and cell["delta_Q_common_from_balanced"] != 0:
                raise ArithmeticError("Shared endpoint models produced unequal scores")
        for name in ("Q_common", "Q_wrong_common", "pairing_increment", "delta_Q_common_from_balanced"):
            panels[policy]["mean_"+name] = float(np.mean([c[name] for c in panels[policy]["cells"]]))
            panels[policy]["intermediate_mean_"+name] = float(np.mean([c[name] for c in panels[policy]["cells"] if c["training_trials"] not in (32, 96)]))
    return panels


def aggregate(sessions):
    animals = ("Chewie", "Mihili")
    if len(sessions) != 8 or any(sum(s["animal"] == a for s in sessions) != 4 for a in animals):
        raise ValueError("Both complete four-session animal panels required")
    keys = tuple(prefix+name for prefix in ("mean_", "intermediate_mean_")
                 for name in ("Q_common", "Q_wrong_common", "pairing_increment", "delta_Q_common_from_balanced"))
    panels = []
    for policy in POLICIES:
        rows = [{"animal": animal, **{key: float(np.mean([s["panels"][policy][key] for s in sessions if s["animal"] == animal])) for key in keys}} for animal in animals]
        panels.append({"policy": policy, "animals": rows, **{key: float(np.mean([r[key] for r in rows])) for key in keys}})
    return panels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "output-dir"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if not 0 < a.max_seconds <= 540 or a.output_dir.exists():
        raise ValueError("Fresh output and finite bound<=540seconds required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Exactly the eight development sessions are required")
    sources = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, block_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected trial-allocation deadline reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress.update(capacity_inner_candidate_scores=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/trial_allocation_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "global_search_completed": False,
        "named_explanatory_amendment_activated": False, "raw_arrays_or_coefficients_written": False,
        "electrode_branch": "fixed_random_from_trial1_five_existing_scarce_set_repetitions",
        "evaluation_normalization": "common32trial_initial_rawY_SD_and_global_mean_null_for_every_policy_and_checkpoint",
        "rank_freeze": "selected_integer_or_full_resolved_to_min(3E,U,24); no later rank or alpha tuning",
        "aggregation": "20cells/session; intermediate12cells reported separately; not directly comparable to old normalization",
        "boundary": "trusted chosen-next-trial/channel slices; copied acquired history only; same-UID is not an OS seal"}
    exposure, pending = Exposure(report), []
    try:
        for filename, source, role, records in sources:
            check()
            t = time.monotonic()
            models, histories, guide = fit_session(a.source_root/filename, source, role, records, progress, check, exposure)
            fit_end = time.monotonic()
            ex, ey, _ = read_trials(a.source_root/filename, source, role, records, role["evaluation_trials"],
                                   source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.array([records[t]["direction_index"] for t in role["evaluation_trials"]])
            pending.append({"filename": filename, "animal": role["animal"],
                "panels": score_session(models, ex, ey, labels, source["common_physical_channels"]), "trajectories": histories,
                "all176_actual_models_and48_trajectories_fixed_before_evaluation": True,
                "shared_initial_and_final_model_identity": True,
                "training_read_and_fit_seconds": fit_end-t, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t}), flush=True)
            del models, histories, ex, ey, guide
        report["panels"], report["sessions"] = aggregate(pending), pending
        report["valid_evaluations_charged"] = 2
        report["status"] = "completed_two_trial_allocators_with_balanced_reference"
    except Exception as error:
        report["status"] = "technical_failure_trial_allocation_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations_cache_dependent"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock-block_start
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (a.output_dir/"trial_allocation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 2, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
