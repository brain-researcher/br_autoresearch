#!/usr/bin/env python3
"""Trusted T/L/K evaluator for fixed development roles and source windows."""
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

from capacity_training import fit_capacity
from capacity_prediction import predict_capacity
from development_design import candidate_designs, TRAINING_TOTALS, ELECTRODE_COUNTS
from development_temporal_reader import selected_metadata, read_training_arrays, read_evaluation_arrays
from direction_training import fit_residual_candidate
from direction_control_metrics import predict_residual, prediction_scores
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress

METHODS = ("temporal_ridge", "reduced_rank", "rbf")


def fit_all(tensor, y, channels, role, records, progress, check):
    if tensor.ndim != 3 or tensor.shape[1:] != (288, 10):
        raise ValueError("One trial per row,96 electrodes,three bands,ten ordered bins required")
    means = tensor.mean(axis=2)
    training = role["nested_training_sets"]["96"]["training_trials"]
    labels = np.asarray([records[t]["direction_index"] for t in training])
    models = []
    for design in candidate_designs(role, channels, 5):
        rows, columns = design["rows"], design["columns"]
        for method in METHODS:
            check()
            if method == "temporal_ridge":
                # np.ix_ selects trial/feature axes and preserves the10-bin last axis.
                x = tensor[np.ix_(rows, columns)].reshape(len(rows), -1).copy()
                fitted = fit_residual_candidate(x, y[rows].copy(), labels[rows].copy(), design["folds"], progress)
            else:
                fitted = fit_capacity(means[np.ix_(rows, columns)].copy(), y[rows].copy(), labels[rows].copy(), design["folds"], method, progress)
            models.append((method, {k: v for k, v in design.items() if k not in ("rows", "folds")}, fitted))
    if len(models) != 144:
        raise RuntimeError("All three complete session model panels required before evaluation read")
    return models


def score_all(models, tensor, y, labels):
    means, rows = tensor.mean(axis=2), {method: [] for method in METHODS}
    for method, design, fitted in models:
        columns = design["columns"]
        if method == "temporal_ridge":
            x = tensor[:, columns, :].reshape(len(tensor), -1)
            prediction = predict_residual(fitted, x, labels)
            tuning = {"candidate_count": 5, "rank": "full", "legal_ranks": ["full"]}
        else:
            prediction = predict_capacity(fitted, means[:, columns], labels)
            tuning = {"candidate_count": fitted.candidate_count, "rank": fitted.rank, "legal_ranks": list(fitted.legal_ranks)}
            if method == "rbf":
                tuning["sigma_squared"] = fitted.kernel_state.sigma_squared
                tuning["constant_feature_kernel"] = fitted.kernel_state.sigma_squared == 0
        stats = fitted.statistics
        score = prediction_scores(prediction, y, labels, stats.target_mean, stats.target_sd)
        rows[method].append({"training_trials": design["training_trials"], "electrodes": design["electrodes"],
            "seed": design["seed"], "channels": design["channels"], "alpha": fitted.alpha,
            "alpha_at_grid_boundary": fitted.alpha in (.0001, 10000.), "tuning": tuning, **score})
    panels = {}
    for method in METHODS:
        cells = []
        for total in TRAINING_TOTALS:
            for count in ELECTRODE_COUNTS:
                selected = [r for r in rows[method] if r["training_trials"] == total and r["electrodes"] == count]
                if len(selected) != (1 if count == 96 else 5):
                    raise RuntimeError("Incomplete resource cell")
                cells.append({"training_trials": total, "electrodes": count,
                    **{key: float(np.mean([r[key] for r in selected])) for key in ("Q", "Q_wrong", "pairing_increment")}})
        panels[method] = {"cells": cells, "fitted_models": rows[method],
            **{"mean_"+key: float(np.mean([c[key] for c in cells])) for key in ("Q", "Q_wrong", "pairing_increment")}}
    return panels


def append_reference_and_aggregate(report, reference):
    if reference["status"] != "completed_direction_null_pairing_and_residual_panels":
        raise ValueError("Completed fixed D/W/R reference required")
    original = {s["filename"]: s for s in reference["sessions"]}
    if set(original) != set(DEVELOPMENT):
        raise ValueError("Reference must contain the same eight development sessions")
    for session in report["sessions"]:
        old = original[session["filename"]]["panels"]
        direction = {c["training_trials"]: c["Q"] for c in old["direction_only"]["cells"]}
        ridge = {(c["training_trials"], c["electrodes"]): c["Q"] for c in old["direction_residual_random"]["cells"]}
        for panel in session["panels"].values():
            for cell in panel["cells"]:
                cell["delta_Q_from_direction_only"] = cell["Q"]-direction[cell["training_trials"]]
                cell["delta_Q_from_mean_residual_ridge"] = cell["Q"]-ridge[(cell["training_trials"], cell["electrodes"])]
            for key in ("delta_Q_from_direction_only", "delta_Q_from_mean_residual_ridge"):
                panel["mean_"+key] = float(np.mean([c[key] for c in panel["cells"]]))
    animals = sorted({s["animal"] for s in report["sessions"]})
    if len(animals) != 2 or any(sum(s["animal"] == a for s in report["sessions"]) != 4 for a in animals):
        raise ValueError("Four sessions per biological animal required")
    report["panels"] = []
    keys = ("mean_Q", "mean_Q_wrong", "mean_pairing_increment", "mean_delta_Q_from_direction_only", "mean_delta_Q_from_mean_residual_ridge")
    for method in METHODS:
        records = [{"animal": a, **{k: float(np.mean([s["panels"][method][k] for s in report["sessions"] if s["animal"] == a])) for k in keys}} for a in animals]
        report["panels"].append({"method": method, "animals": records, **{k: float(np.mean([r[k] for r in records])) for k in keys}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "direction-report", "output-dir"):
        p.add_argument("--"+flag, required=True, type=Path)
    p.add_argument("--max-seconds", required=True, type=int)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    if a.max_seconds <= 0 or a.output_dir.exists():
        raise ValueError("Finite positive deadline and fresh output path required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Only the fixed eight development sessions are allowed")
    selected = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, blocks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected T/L/K time bound reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress["capacity_inner_candidate_scores"] = 0
    report = {"analysis_id": "development_search", "recipe": "outputs/representation_capacity_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "fit_progress": progress,
        "sessions": [], "fitting_trials_read": 0, "evaluation_trials_read": 0, "logical_neural_bytes": 0,
        "evaluation_neural_elements_read": False, "held_session_accessed": False, "valid_evaluations_charged": 0,
        "held_openings_charged": 0, "named_explanatory_amendment_activated": False, "global_search_completed": False,
        "raw_arrays_or_coefficients_written": False, "temporal_trial_is_single_row": True,
        "tensor_order": "trial,electrode-band,time; electrode-major band-major time-major flatten",
        "actual_exposure": "offline all96 training electrodes, unchanged ten-bin slices",
        "boundary": "fixed copied-training-array API, trusted evaluator; same-UID filesystem is not an OS seal",
        "tuning_opportunity": {"temporal_ridge": "5alpha", "reduced_rank": "5alpha x common legal ranks1/2/4/8/full", "rbf": "5alpha; fixed fitting-only median bandwidth"}}
    try:
        for filename, source, role, records in selected:
            check()
            t = time.monotonic()
            x, y, channels, guides = read_training_arrays(a.source_root/filename, source, role, records, check, report)
            models = fit_all(x, y, channels, role, records, progress, check)
            fit_end = time.monotonic()
            ex, ey, _, _ = read_evaluation_arrays(a.source_root/filename, source, role, records, check, report, guides)
            labels = np.asarray([records[i]["direction_index"] for i in role["evaluation_trials"]])
            panels = score_all(models, ex, ey, labels)
            report["sessions"].append({"filename": filename, "animal": role["animal"], "panels": panels,
                "all144_models_fitted_before_evaluation_read": True, "training_read_and_fit_seconds": fit_end-t,
                "elapsed_seconds": time.monotonic()-t})
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t}), flush=True)
            del x, y, ex, ey, models, guides
        # Historical numerical scores are reporting-only and enter after fits.
        append_reference_and_aggregate(report, json.loads(a.direction_report.read_text()))
        report["valid_evaluations_charged"] = 3
        report["status"] = "completed_temporal_rank_kernel_development_panels"
    except Exception as error:
        report["status"] = "technical_failure_three_panel_comparison_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations_cache_dependent"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock-blocks
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (a.output_dir/"capacity_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 3, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
