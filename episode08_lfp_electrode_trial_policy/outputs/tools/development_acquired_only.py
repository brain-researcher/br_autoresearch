#!/usr/bin/env python3
"""Trusted initial-selection versus fixed-random acquired-only comparison."""
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

from acquired_only_reader import CHECKPOINTS, Exposure, Trajectory, balanced_ids, read_trials
from acquired_rank_training import conditioned_folds, fit_conditioned_rank
from capacity_training import fit_capacity
from capacity_prediction import predict_capacity
from development_design import ELECTRODE_COUNTS, candidate_designs
from development_reader import selected_metadata
from direction_control_metrics import prediction_scores
from direction_training import fit_direction_null
from prepare_development_trial_roles import DEVELOPMENT
from residual_selection_training import order_bank, fit_selected_rank
from ridge_training import new_progress

BRANCHES = ("initial_marginal", "fixed_random")


def relative_initial_folds(role):
    item = role["nested_training_sets"]["32"]
    relative = {t: i for i, t in enumerate(item["training_trials"])}
    return [(np.asarray([relative[t] for t in f["fitting_trials"]]), np.asarray([relative[t] for t in f["validation_trials"]])) for f in item["inner_folds"]]


def later_fit(trajectory, total, records, progress, make_null=False):
    trajectory.acquire_to(total)
    x, y, ids = trajectory.observed()
    labels = np.asarray([records[t]["direction_index"] for t in ids])
    initial = set(balanced_ids(trajectory.role, 32))
    initial_mask = np.asarray([t in initial for t in ids])
    # Within-direction queue position is metadata only and fixed before outcomes.
    positions = {t: i-4 for pool in trajectory.role["fitting_pool_by_direction"] for i, t in enumerate(pool[:12])}
    initial_rows, folds = conditioned_folds(labels, initial_mask, np.asarray([positions[t] for t in ids]))
    fitted = fit_conditioned_rank(x, y, labels, initial_rows, folds, progress)
    null = fit_direction_null(y, labels) if make_null else None
    if make_null:
        progress["direction_only_models"] += 1
    return fitted, null


def fit_session(path, source, role, records, progress, check, exposure):
    channels = source["common_physical_channels"]
    ids = balanced_ids(role, 32)
    if ids != role["nested_training_sets"]["32"]["training_trials"]:
        raise ValueError("Initializer must match the frozen32-trial prefix")
    initial_x, initial_y, guide = read_trials(path, source, role, records, ids, channels, check, exposure)
    labels = np.asarray([records[t]["direction_index"] for t in ids])
    folds = relative_initial_folds(role)
    inner, outer = order_bank(initial_x, initial_y, labels, folds, "residual_marginal",
                              f"ep08-acquired-initial-marginal-v1|{role['filename']}", progress)
    models, nulls, shared_full = [], {32: fit_direction_null(initial_y, labels)}, {}
    progress["direction_only_models"] += 1
    for count in ELECTRODE_COUNTS:
        check()
        if count == 96:
            selected_channels = channels
            fitted = fit_capacity(initial_x.copy(), initial_y.copy(), labels.copy(), folds, "reduced_rank", progress)
        else:
            fitted, columns = fit_selected_rank(initial_x, initial_y, labels, folds, inner, outer, count, progress)
            selected_channels = [channels[int(c//3)] for c in columns[::3]]
        trajectory = Trajectory(path, source, role, records, selected_channels, check, exposure)
        trajectory.initialize_selected(initial_x, initial_y, channels, guide)
        for total in CHECKPOINTS:
            check()
            if total > 32:
                fitted, null = later_fit(trajectory, total, records, progress, total not in nulls)
                if null is not None:
                    nulls[total] = null
            models.append(("initial_marginal", total, count, 0, tuple(selected_channels), fitted))
            if count == 96:
                shared_full[total] = fitted
    for design in candidate_designs(role, channels, 5):
        if design["training_trials"] != 32:
            continue
        count, seed = design["electrodes"], design["seed"]
        if count == 96:
            for total in CHECKPOINTS:
                models.append(("fixed_random", total, count, 0, tuple(channels), shared_full[total]))
            continue
        check()
        selected_channels = design["channels"]
        trajectory = Trajectory(path, source, role, records, selected_channels, check, exposure)
        trajectory.initialize_random()
        x, y, _ = trajectory.observed()
        fitted = fit_capacity(x, y, labels.copy(), folds, "reduced_rank", progress)
        for total in CHECKPOINTS:
            check()
            if total > 32:
                fitted, _null = later_fit(trajectory, total, records, progress)
            models.append(("fixed_random", total, count, seed, tuple(selected_channels), fitted))
    if len(models) != 100 or len(nulls) != 5:
        raise RuntimeError("Complete100references/95actual trajectory models required before evaluation")
    return models, nulls, guide


def score_session(models, nulls, ex, ey, labels, channels):
    positions = {c: i for i, c in enumerate(channels)}
    null_scores = {n: prediction_scores(m.direction_target_mean[labels], ey, labels, m.target_mean, m.target_sd)["Q"] for n, m in nulls.items()}
    rows, full = {b: [] for b in BRANCHES}, {}
    for branch, total, count, seed, selected, fitted in models:
        if count == 96 and total in full:
            score = full[total]
        else:
            cols = [3*positions[c]+band for c in selected for band in range(3)]
            prediction = predict_capacity(fitted, ex[:, cols], labels)
            s = fitted.statistics
            score = prediction_scores(prediction, ey, labels, s.target_mean, s.target_sd)
            if count == 96:
                full[total] = score
        electrode_trials = 32*96+(total-32)*count if branch == "initial_marginal" else total*count
        rows[branch].append({"training_trials": total, "electrodes": count, "seed": seed, "channels": list(selected),
            "alpha": fitted.alpha, "rank": fitted.rank, "candidate_count": fitted.candidate_count,
            "LFP_electrode_trials": electrode_trials, "LFP_numeric_elements": electrode_trials*3*10,
            "target_numeric_elements": total*ey.shape[1]*10, "initial_trials": 32,
            "initial_LFP_electrodes": 96 if branch == "initial_marginal" else count, **score})
    panels = {}
    for branch in BRANCHES:
        cells = []
        for total in CHECKPOINTS:
            for count in ELECTRODE_COUNTS:
                selected = [r for r in rows[branch] if r["training_trials"] == total and r["electrodes"] == count]
                if len(selected) != (5 if branch == "fixed_random" and count != 96 else 1):
                    raise RuntimeError("Incomplete acquired-only resource cell")
                cell = {"training_trials": total, "electrodes": count,
                    **{k: float(np.mean([r[k] for r in selected])) for k in ("Q", "Q_wrong", "pairing_increment")},
                    **{k: selected[0][k] for k in ("LFP_electrode_trials", "LFP_numeric_elements", "target_numeric_elements")}}
                cell["direction_only_Q"] = null_scores[total]
                cell["delta_Q_from_direction_only"] = cell["Q"]-null_scores[total]
                cells.append(cell)
        panels[branch] = {"cells": cells, "fitted_models": rows[branch],
            **{"mean_"+k: float(np.mean([c[k] for c in cells])) for k in ("Q", "Q_wrong", "pairing_increment", "delta_Q_from_direction_only")}}
    for a, b in zip(panels["initial_marginal"]["cells"], panels["fixed_random"]["cells"]):
        a["delta_Q_from_fixed_random"] = a["Q"]-b["Q"]
        a["extra_initial_LFP_electrode_trials"] = a["LFP_electrode_trials"]-b["LFP_electrode_trials"]
    panels["initial_marginal"]["mean_delta_Q_from_fixed_random"] = float(np.mean([c["delta_Q_from_fixed_random"] for c in panels["initial_marginal"]["cells"]]))
    return panels


def initial_replay_and_aggregate(sessions, selection, capacity):
    if selection["status"] != "completed_three_residual_selector_panels" or capacity["status"] != "completed_temporal_rank_kernel_development_panels":
        raise ValueError("Completed frozen reference reports required")
    references = {"initial_marginal": ({s["filename"]: s for s in selection["sessions"]}, "residual_marginal"),
                  "fixed_random": ({s["filename"]: s for s in capacity["sessions"]}, "reduced_rank")}
    maximum = 0.
    for reference, _ in references.values():
        if set(reference) != set(DEVELOPMENT):
            raise ValueError("Initial reference sessions differ")
    for session in sessions:
        for branch, (reference, old_panel) in references.items():
            old = {c["electrodes"]: c["Q"] for c in reference[session["filename"]]["panels"][old_panel]["cells"] if c["training_trials"] == 32}
            for cell in session["panels"][branch]["cells"]:
                if cell["training_trials"] == 32:
                    difference = abs(cell["Q"]-old[cell["electrodes"]])
                    maximum = max(maximum, difference)
                    if not np.isfinite(difference) or difference > 1e-10:
                        raise ArithmeticError("Initial nested reference replay failed; no retuning")
    animals = sorted({s["animal"] for s in sessions})
    if len(animals) != 2 or any(sum(s["animal"] == a for s in sessions) != 4 for a in animals):
        raise ValueError("Four sessions per biological animal required")
    panels = []
    for branch in BRANCHES:
        keys = ["mean_Q", "mean_Q_wrong", "mean_pairing_increment", "mean_delta_Q_from_direction_only"]
        if branch == "initial_marginal":
            keys.append("mean_delta_Q_from_fixed_random")
        rows = [{"animal": a, **{k: float(np.mean([s["panels"][branch][k] for s in sessions if s["animal"] == a])) for k in keys}} for a in animals]
        panels.append({"branch": branch, "animals": rows, **{k: float(np.mean([r[k] for r in rows])) for k in keys}})
    return maximum, panels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "selection-report", "capacity-report", "output-dir"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if a.max_seconds <= 0 or a.output_dir.exists():
        raise ValueError("Fresh output path and finite positive bound required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Only the fixed eight development sessions are allowed")
    selected = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, blocks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected acquired-only bound reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress.update(selector_fits=0, selection_seconds=0., capacity_inner_candidate_scores=0, direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/acquired_only_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "logical_neural_bytes": 0, "unique_logical_neural_bytes": 0,
        "fitting_trial_read_calls": 0, "evaluation_trials_read": 0, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "global_search_completed": False,
        "named_explanatory_amendment_activated": False, "raw_arrays_or_coefficients_written": False,
        "aggregation": "20cells per session; compare within this protocol, not prior12cell rankings",
        "exposure": "initial marginal32*96+(n-32)*E; random n*E; not equal initial acquisition cost",
        "boundary": "trusted chosen-channel source views and copied observed arrays; same-UID filesystem is not an OS seal"}
    exposure, pending = Exposure(report), []
    try:
        for filename, source, role, records in selected:
            check()
            t = time.monotonic()
            models, nulls, guide = fit_session(a.source_root/filename, source, role, records, progress, check, exposure)
            fit_end = time.monotonic()
            ex, ey, _ = read_trials(a.source_root/filename, source, role, records, role["evaluation_trials"],
                                   source["common_physical_channels"], check, exposure, guide, evaluation=True)
            labels = np.asarray([records[i]["direction_index"] for i in role["evaluation_trials"]])
            pending.append({"filename": filename, "animal": role["animal"],
                "panels": score_session(models, nulls, ex, ey, labels, source["common_physical_channels"]),
                "all95_actual_models_fixed_before_evaluation": True,
                "training_read_and_fit_seconds": fit_end-t, "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t}), flush=True)
            del models, nulls, ex, ey, guide
        replay, panels = initial_replay_and_aggregate(pending, json.loads(a.selection_report.read_text()), json.loads(a.capacity_report.read_text()))
        report["initial_replay_max_absolute_difference"], report["panels"], report["sessions"] = replay, panels, pending
        report["valid_evaluations_charged"] = 1
        report["status"] = "completed_one_acquired_only_comparison_with_random_reference"
    except Exception as error:
        report["status"] = "technical_failure_acquired_only_comparison_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations_cache_dependent"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock-blocks
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (a.output_dir/"acquired_only.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 1, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
