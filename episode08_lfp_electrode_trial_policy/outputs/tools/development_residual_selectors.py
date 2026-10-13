#!/usr/bin/env python3
"""Trusted residual-selector comparison; fixed global reduced-rank decoder."""
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
from development_design import TRAINING_TOTALS, ELECTRODE_COUNTS
from development_reader import selected_metadata, read_training_arrays, read_evaluation_arrays
from direction_control_metrics import prediction_scores
from prepare_development_trial_roles import DEVELOPMENT
from residual_selection_training import METHODS, SelectedRank, order_bank, fit_selected_rank
from ridge_training import new_progress


def fit_all(x, y, role, records, progress, check):
    positions = {t: i for i, t in enumerate(role["nested_training_sets"]["96"]["training_trials"])}
    result = []
    for total in TRAINING_TOTALS:
        training = role["nested_training_sets"][str(total)]
        ids = training["training_trials"]
        rows, relative = np.asarray([positions[t] for t in ids]), {t: i for i, t in enumerate(ids)}
        folds = [(np.asarray([relative[t] for t in f["fitting_trials"]]), np.asarray([relative[t] for t in f["validation_trials"]])) for f in training["inner_folds"]]
        xx, yy = x[rows].copy(), y[rows].copy()
        labels = np.asarray([records[t]["direction_index"] for t in ids])
        check()
        full = fit_capacity(xx.copy(), yy.copy(), labels.copy(), folds, "reduced_rank", progress)
        for method in METHODS:
            for replicate in range(5 if method == "permuted_residual_conditional" else 1):
                check()
                seed = f"ep08-residual-selector-permutation-v1|20261003|{role['filename']}|{total}|{replicate}"
                inner, outer = order_bank(xx, yy, labels, folds, method, seed, progress)
                for count in ELECTRODE_COUNTS[:-1]:
                    fitted, columns = fit_selected_rank(xx, yy, labels, folds, inner, outer, count, progress)
                    result.append((total, SelectedRank(method, count, replicate, columns, fitted)))
            result.append((total, SelectedRank(method, 96, 0, np.arange(x.shape[1]), full)))
    if len(result) != 72:
        raise RuntimeError("All residual-selector session fits required before evaluation access")
    return result


def score_all(models, x, y, labels, channels):
    rows, full_scores = {method: [] for method in METHODS}, {}
    for total, model in models:
        if model.electrodes == 96 and total in full_scores:
            score = full_scores[total]
        else:
            fitted = model.fitted
            prediction = predict_capacity(fitted, x[:, model.columns], labels)
            stats = fitted.statistics
            score = prediction_scores(prediction, y, labels, stats.target_mean, stats.target_sd)
            if model.electrodes == 96:
                full_scores[total] = score
        fit = model.fitted
        rows[model.method].append({"training_trials": total, "electrodes": model.electrodes,
            "replicate": model.replicate, "channels": [channels[int(c//3)] for c in model.columns[::3]],
            "alpha": fit.alpha, "rank": fit.rank, "legal_ranks": list(fit.legal_ranks),
            "candidate_count": fit.candidate_count, **score})
    panels = {}
    for method in METHODS:
        cells = []
        for total in TRAINING_TOTALS:
            for count in ELECTRODE_COUNTS:
                records = [r for r in rows[method] if r["training_trials"] == total and r["electrodes"] == count]
                expected = 5 if method == "permuted_residual_conditional" and count != 96 else 1
                if len(records) != expected:
                    raise RuntimeError("Incomplete selection resource cell")
                cells.append({"training_trials": total, "electrodes": count,
                    **{k: float(np.mean([r[k] for r in records])) for k in ("Q", "Q_wrong", "pairing_increment")}})
        panels[method] = {"cells": cells, "fitted_models": rows[method],
            **{"mean_"+k: float(np.mean([c[k] for c in cells])) for k in ("Q", "Q_wrong", "pairing_increment")}}
    return panels


def reference_and_aggregate(sessions, capacity, direction):
    if (capacity["status"] != "completed_temporal_rank_kernel_development_panels"
            or direction["status"] != "completed_direction_null_pairing_and_residual_panels"):
        raise ValueError("Completed fixed capacity/direction reports required")
    old = {s["filename"]: s for s in capacity["sessions"]}
    null = {s["filename"]: s for s in direction["sessions"]}
    if set(old) != set(DEVELOPMENT) or set(null) != set(DEVELOPMENT):
        raise ValueError("Reference development identities differ")
    maximum = 0.
    for session in sessions:
        random = {(c["training_trials"], c["electrodes"]): c["Q"] for c in old[session["filename"]]["panels"]["reduced_rank"]["cells"]}
        d = {c["training_trials"]: c["Q"] for c in null[session["filename"]]["panels"]["direction_only"]["cells"]}
        for panel in session["panels"].values():
            for cell in panel["cells"]:
                key = cell["training_trials"], cell["electrodes"]
                if cell["electrodes"] == 96:
                    difference = abs(cell["Q"]-random[key])
                    maximum = max(maximum, difference)
                    if not np.isfinite(difference) or difference > 1e-10:
                        raise ArithmeticError("Shared all96 replay failed; no scientific packet or retuning")
                cell["delta_Q_from_random_reduced_rank"] = cell["Q"]-random[key]
                cell["delta_Q_from_direction_only"] = cell["Q"]-d[cell["training_trials"]]
            for key in ("delta_Q_from_random_reduced_rank", "delta_Q_from_direction_only"):
                panel["mean_"+key] = float(np.mean([c[key] for c in panel["cells"]]))
    animals = sorted({s["animal"] for s in sessions})
    if len(animals) != 2 or any(sum(s["animal"] == a for s in sessions) != 4 for a in animals):
        raise ValueError("Four sessions per biological animal required")
    keys = ("mean_Q", "mean_Q_wrong", "mean_pairing_increment", "mean_delta_Q_from_random_reduced_rank", "mean_delta_Q_from_direction_only")
    result = []
    for method in METHODS:
        records = [{"animal": a, **{k: float(np.mean([s["panels"][method][k] for s in sessions if s["animal"] == a])) for k in keys}} for a in animals]
        result.append({"method": method, "animals": records, **{k: float(np.mean([r[k] for r in records])) for k in keys}})
    return maximum, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "metadata-report", "trial-roles", "capacity-report", "direction-report", "output-dir"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if a.max_seconds <= 0 or a.output_dir.exists():
        raise ValueError("Fresh output and positive finite deadline required")
    metadata, roles = json.loads(a.metadata_report.read_text()), json.loads(a.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {s["filename"] for s in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Only fixed development roles allowed")
    selected = [(name, *selected_metadata(metadata, roles, name)) for name in DEVELOPMENT]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, blocks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    def timeout(*_):
        raise TimeoutError("Selected residual-selection time bound reached")
    def check():
        if time.monotonic()-started >= a.max_seconds:
            timeout()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(a.max_seconds)
    progress = new_progress()
    progress.update(selector_fits=0, selection_seconds=0., capacity_inner_candidate_scores=0)
    report = {"analysis_id": "development_search", "recipe": "outputs/residual_selection_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "fit_progress": progress,
        "sessions": [], "completed_session_ids": [], "fitting_trials_read": 0, "evaluation_trials_read": 0,
        "logical_neural_bytes": 0, "evaluation_neural_elements_read": False, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "held_openings_charged": 0, "named_explanatory_amendment_activated": False,
        "global_search_completed": False, "raw_arrays_or_coefficients_written": False,
        "actual_exposure": "offline all96 fitting electrodes; no acquisition-efficiency claim",
        "boundary": "fixed fitting-array selector/decoder; trusted evaluator; same-UID filesystem is not an OS seal",
        "control": "ordinary within-direction whole-target permutations including fixed points affect selector only; true-Y final decoder"}
    pending = []
    try:
        for filename, source, role, records in selected:
            check()
            t = time.monotonic()
            x, y, channels, guides = read_training_arrays(a.source_root/filename, source, role, records, check, report)
            models = fit_all(x, y, role, records, progress, check)
            fit_end = time.monotonic()
            ex, ey, _, _ = read_evaluation_arrays(a.source_root/filename, source, role, records, check, report, guides)
            labels = np.asarray([records[i]["direction_index"] for i in role["evaluation_trials"]])
            pending.append({"filename": filename, "animal": role["animal"], "panels": score_all(models, ex, ey, labels, channels),
                "all66_actual_models_fitted_before_evaluation_read": True, "training_read_and_fit_seconds": fit_end-t,
                "elapsed_seconds": time.monotonic()-t})
            report["completed_session_ids"].append(filename)
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-t}), flush=True)
            del x, y, ex, ey, models, guides
        # Past outcomes enter reporting only after all new fits and scores.
        replay, panels = reference_and_aggregate(pending, json.loads(a.capacity_report.read_text()), json.loads(a.direction_report.read_text()))
        report["all96_replay_max_absolute_difference"] = replay
        report["sessions"], report["panels"] = pending, panels
        report["valid_evaluations_charged"] = 3
        report["status"] = "completed_three_residual_selector_panels"
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
        (a.output_dir/"residual_selectors.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 3, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
