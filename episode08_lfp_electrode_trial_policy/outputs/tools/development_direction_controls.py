#!/usr/bin/env python3
"""Trusted EP08 D/W/R evaluator; no changes to completed baseline/selector code."""
from __future__ import annotations

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

from development_design import TRAINING_TOTALS, ELECTRODE_COUNTS, candidate_designs
from development_reader import selected_metadata, read_training_arrays, read_evaluation_arrays
from direction_control_metrics import prediction_scores, predict_residual
from direction_training import fit_direction_null, fit_residual_candidate
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import fit_candidate, new_progress
from selector_training import (SelectedModel, build_order_bank, fit_with_orders, validate_folds)

REFERENCES = ("direction_repeatability", "conditional_target")


def fit_all(x, y, channels, role, records, progress, check):
    positions = {trial: i for i, trial in enumerate(role["nested_training_sets"]["96"]["training_trials"])}
    nulls, references, residuals = {}, [], []
    for total in TRAINING_TOTALS:
        check()
        training = role["nested_training_sets"][str(total)]
        ids = training["training_trials"]
        rows = np.asarray([positions[t] for t in ids])
        relative = {trial: i for i, trial in enumerate(ids)}
        folds = [(np.asarray([relative[t] for t in fold["fitting_trials"]]),
                  np.asarray([relative[t] for t in fold["validation_trials"]])) for fold in training["inner_folds"]]
        xx, yy = x[rows].copy(), y[rows].copy()
        labels = np.asarray([records[t]["direction_index"] for t in ids])
        validate_folds(len(xx), folds)
        nulls[total] = fit_direction_null(yy.copy(), labels.copy())
        progress["direction_only_models"] += 1
        full = fit_candidate(xx.copy(), yy.copy(), folds, progress)
        for method in REFERENCES:
            seed = f"ep08-selector-target-permutation-v1|20261003|{role['filename']}|{total}|0"
            inner, outer = build_order_bank(xx, yy, labels, folds, method, 32, seed, progress)
            for count in ELECTRODE_COUNTS[:-1]:
                fitted, columns = fit_with_orders(xx, yy, folds, inner, outer, count, progress)
                references.append((total, SelectedModel(method, count, 0, columns, fitted)))
            references.append((total, SelectedModel(method, 96, 0, np.arange(x.shape[1]), full)))
    training_ids = role["nested_training_sets"]["96"]["training_trials"]
    all_labels = np.asarray([records[t]["direction_index"] for t in training_ids])
    for design in candidate_designs(role, channels, 5):
        check()
        rows, columns = design["rows"], design["columns"]
        fitted = fit_residual_candidate(x[np.ix_(rows, columns)].copy(), y[rows].copy(),
                                        all_labels[rows].copy(), design["folds"], progress)
        residuals.append(({k: v for k, v in design.items() if k not in ("rows", "folds")}, fitted))
    if len(nulls) != 3 or len(references) != 24 or len(residuals) != 48:
        raise RuntimeError("Complete D/W/R session fits required before evaluation read")
    return nulls, references, residuals


def collapse(records, null_scores, residual=False):
    cells = []
    for total in TRAINING_TOTALS:
        for count in ELECTRODE_COUNTS:
            selected = [r for r in records if r["training_trials"] == total and r["electrodes"] == count]
            expected = 5 if residual and count != 96 else 1
            if len(selected) != expected:
                raise RuntimeError("Required fixed resource cell incomplete")
            cell = {"training_trials": total, "electrodes": count,
                    **{key: float(np.mean([r[key] for r in selected]))
                       for key in ("Q", "Q_wrong", "pairing_increment")}}
            cell["delta_Q_from_direction_only"] = cell["Q"] - null_scores[total]["Q"]
            cells.append(cell)
    return {"cells": cells, "fitted_models": records,
            **{"mean_"+key: float(np.mean([c[key] for c in cells]))
               for key in ("Q", "Q_wrong", "pairing_increment", "delta_Q_from_direction_only")}}


def score_all(models, ex, ey, evaluation_labels, channels):
    nulls, references, residuals = models
    null_scores = {}
    for total, fitted in nulls.items():
        null_scores[total] = prediction_scores(fitted.direction_target_mean[evaluation_labels], ey,
                                               evaluation_labels, fitted.target_mean, fitted.target_sd)
    panels = {"direction_only": {
        "cells": [{"training_trials": n, "electrodes": 0, **null_scores[n]} for n in TRAINING_TOTALS],
        "mean_Q": float(np.mean([null_scores[n]["Q"] for n in TRAINING_TOTALS])),
        "actual_models": 3, "LFP_electrodes_used": 0}}
    rows = {method: [] for method in REFERENCES}
    shared_full_scores = {}
    for total, model in references:
        fit = model.fitted
        if model.electrodes == 96 and total in shared_full_scores:
            score = shared_full_scores[total]
        else:
            standardized = ((ex[:, model.columns]-fit.input_mean)/fit.input_sd) @ fit.coefficients
            prediction = fit.target_mean + fit.target_sd * standardized
            score = prediction_scores(prediction, ey, evaluation_labels, fit.target_mean, fit.target_sd)
            if model.electrodes == 96:
                shared_full_scores[total] = score
        rows[model.selector].append({"training_trials": total, "electrodes": model.electrodes,
            "channels": [channels[int(c//3)] for c in model.columns[::3]], "alpha": fit.alpha, **score})
    for method in REFERENCES:
        panels["raw_"+method] = collapse(rows[method], null_scores)
    rows = []
    for design, fitted in residuals:
        prediction = predict_residual(fitted, ex[:, design["columns"]], evaluation_labels)
        s = fitted.statistics
        score = prediction_scores(prediction, ey, evaluation_labels, s.target_mean, s.target_sd)
        rows.append({"training_trials": design["training_trials"], "electrodes": design["electrodes"],
                     "seed": design["seed"], "channels": design["channels"], "alpha": fitted.alpha, **score})
    panels["direction_residual_random"] = collapse(rows, null_scores, residual=True)
    return panels


def check_replay(sessions, original):
    if original["status"] != "completed_five_development_selector_panels":
        raise ValueError("The original fixed five-panel report is required")
    old = {s["filename"]: s for s in original["sessions"]}
    if set(old) != set(DEVELOPMENT):
        raise ValueError("Replay reference must contain all eight development sessions")
    maximum = 0.
    for session in sessions:
        for method in REFERENCES:
            cells = {(c["training_trials"], c["electrodes"]): c["mean_Q"]
                     for c in old[session["filename"]]["panels"][method]["cells"]}
            for cell in session["panels"]["raw_"+method]["cells"]:
                difference = abs(cell["Q"]-cells[(cell["training_trials"], cell["electrodes"])] )
                maximum = max(maximum, difference)
                if not np.isfinite(difference) or difference > 1e-10:
                    raise ArithmeticError("Correct-pair replay differs; no W result is released or recipe retuned")
    return maximum


def aggregate(sessions):
    animals = sorted({s["animal"] for s in sessions})
    if len(sessions) != 8 or len(animals) != 2 or any(sum(s["animal"] == a for s in sessions) != 4 for a in animals):
        raise ValueError("Preserve four development sessions per biological animal")
    result = []
    for panel in ("direction_only", "raw_direction_repeatability", "raw_conditional_target", "direction_residual_random"):
        keys = ("mean_Q",) if panel == "direction_only" else (
            "mean_Q", "mean_Q_wrong", "mean_pairing_increment", "mean_delta_Q_from_direction_only")
        records = [{"animal": animal, **{key: float(np.mean([
            s["panels"][panel][key] for s in sessions if s["animal"] == animal])) for key in keys}}
                   for animal in animals]
        result.append({"panel": panel, "animals": records,
                       **{key: float(np.mean([r[key] for r in records])) for key in keys}})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "selector-report", "output-dir"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if a.max_seconds <= 0 or a.output_dir.exists():
        raise ValueError("Positive finite bound and fresh output path required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Only the fixed eight-session development panel is authorized")
    selected = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    blocks = resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected D/W/R comparison time bound reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress.update(selector_fits=0, selection_seconds=0., direction_only_models=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/direction_confounding_proposal.md",
        "status": "running", "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "fit_progress": progress, "completed_session_ids": [], "sessions": [],
        "fitting_trials_read": 0, "evaluation_trials_read": 0, "logical_neural_bytes": 0,
        "evaluation_neural_elements_read": False, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "held_openings_charged": 0,
        "named_explanatory_amendment_activated": False, "global_search_completed": False,
        "raw_arrays_or_coefficients_written": False,
        "actual_exposure": "Grouped offline worker reads all96 training electrodes; direction-only predictor uses zero LFP electrodes",
        "boundary": "fixed fitting-only API and trusted evaluator; same-UID ambient filesystem is not an OS seal",
        "pairing_control": "exact within-direction off-diagonal mean equals derangement mean; no p-value"}
    pending = []
    try:
        for filename, source, role, records in selected:
            check()
            session_start = time.monotonic()
            x, y, channels, guides = read_training_arrays(a.source_root/filename, source, role, records, check, report)
            models = fit_all(x, y, channels, role, records, progress, check)
            fit_end = time.monotonic()
            ex, ey, _, _ = read_evaluation_arrays(a.source_root/filename, source, role, records, check, report, guides)
            labels = np.asarray([records[t]["direction_index"] for t in role["evaluation_trials"]])
            panels = score_all(models, ex, ey, labels, channels)
            pending.append({"filename": filename, "animal": role["animal"], "panels": panels,
                "training_read_and_fit_seconds": fit_end-session_start,
                "elapsed_seconds": time.monotonic()-session_start,
                "all_D_W_R_models_fitted_before_evaluation_read": True})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-session_start}), flush=True)
            del x, y, models, ex, ey, guides
        # The historical numerical report enters only reporting after all fits.
        # No W score is written until every correct-pair replay check passes.
        report["correct_pair_replay_max_absolute_difference"] = check_replay(pending, json.loads(a.selector_report.read_text()))
        report["panels"] = aggregate(pending)
        report["sessions"] = pending
        report["valid_evaluations_charged"] = 3
        report["panel_charge_breakdown"] = {"direction_only": 1, "pairing_control_two_reference_branches": 1,
                                           "direction_residual_random": 1}
        report["status"] = "completed_direction_null_pairing_and_residual_panels"
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
        (a.output_dir/"direction_controls.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 3,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
