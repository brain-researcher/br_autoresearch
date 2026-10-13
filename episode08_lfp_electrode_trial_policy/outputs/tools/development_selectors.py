#!/usr/bin/env python3
"""Trusted evaluator for five prospectively selected EP08 selector panels.

No changes to the completed baseline, its fitter or its source-role artifacts.
Candidate components receive training copies only. Same-UID ambient filesystem
access is disclosed; this is not hostile-code isolation.
"""
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

from development_baseline import score_frozen_model
from development_reader import selected_metadata, read_training_arrays, read_evaluation_arrays
from development_design import TRAINING_TOTALS, ELECTRODE_COUNTS
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress
from selector_training import SELECTORS, fit_selection_panels


def fit_session(x, y, role, records, progress, check_deadline):
    source_rows = role["nested_training_sets"]["96"]["training_trials"]
    positions = {trial: index for index, trial in enumerate(source_rows)}
    fitted = []
    for total in TRAINING_TOTALS:
        check_deadline()
        training = role["nested_training_sets"][str(total)]
        ids = training["training_trials"]
        rows = np.asarray([positions[trial] for trial in ids])
        relative = {trial: index for index, trial in enumerate(ids)}
        folds = [(np.asarray([relative[i] for i in fold["fitting_trials"]]),
                  np.asarray([relative[i] for i in fold["validation_trials"]]))
                 for fold in training["inner_folds"]]
        labels = np.asarray([records[trial]["direction_index"] for trial in ids])
        prefix = f"ep08-selector-target-permutation-v1|20261003|{role['filename']}|{total}"
        models = fit_selection_panels(x[rows].copy(), y[rows].copy(), labels, folds, prefix, progress)
        fitted.extend((total, model) for model in models)
    # 27 scarce models plus five references to one shared all-channel model / total.
    if len(fitted) != 96:
        raise RuntimeError("Five-panel session fit incomplete; no evaluation read permitted")
    return fitted


def score_session(fitted, x, y, channels):
    panel = {method: {"cells": [], "fitted_models": []} for method in SELECTORS}
    values = {}
    full_scores = {}
    for total, model in fitted:
        if model.electrodes == 96:
            if total not in full_scores:
                full_scores[total] = score_frozen_model(model.fitted, x[:, model.columns], y)
            q = full_scores[total]
        else:
            q = score_frozen_model(model.fitted, x[:, model.columns], y)
        values.setdefault((model.selector, total, model.electrodes), []).append(q)
        panel[model.selector]["fitted_models"].append({
            "training_trials": total, "electrodes": model.electrodes,
            "replicate": model.replicate,
            "channels": [channels[int(col // 3)] for col in model.columns[::3]],
            "alpha": model.fitted.alpha, "Q": q,
            "constant_input_columns_in_selected_order": list(model.fitted.constant_input_columns),
            "constant_target_columns_in_native_order": list(model.fitted.constant_target_columns)})
    for method in SELECTORS:
        for total in TRAINING_TOTALS:
            for count in ELECTRODE_COUNTS:
                scores = values[(method, total, count)]
                expected = 5 if method == "permuted_conditional_target" and count != 96 else 1
                if len(scores) != expected:
                    raise RuntimeError("Selected replication panel incomplete")
                panel[method]["cells"].append({"training_trials": total, "electrodes": count,
                    "replicate_Q": scores, "mean_Q": float(np.mean(scores)),
                    "full_electrode_model_and_score_shared": count == 96})
        panel[method]["mean_Q"] = float(np.mean([cell["mean_Q"] for cell in panel[method]["cells"]]))
    return panel


def append_baseline_contrasts(report, baseline):
    """Reporting only, after every selector fit and evaluation has finished."""
    if baseline["status"] != "completed_one_development_baseline_no_scientific_winner":
        raise ValueError("Use the completed fixed baseline only")
    reference = {item["filename"]: item for item in baseline["sessions"]}
    if set(reference) != set(DEVELOPMENT):
        raise ValueError("Baseline must contain the same complete development panel")
    for session in report["sessions"]:
        old = reference[session["filename"]]
        old_cells = {(cell["training_trials"], cell["electrodes"]): cell["mean_Q"] for cell in old["cells"]}
        for result in session["panels"].values():
            result["delta_Q_from_random_baseline"] = result["mean_Q"] - old["mean_Q"]
            for cell in result["cells"]:
                cell["delta_Q_from_random_baseline"] = cell["mean_Q"] - old_cells[(cell["training_trials"], cell["electrodes"])]
    animals = sorted({session["animal"] for session in report["sessions"]})
    if len(animals) != 2 or any(sum(s["animal"] == a for s in report["sessions"]) != 4 for a in animals):
        raise ValueError("Preserve four sessions in each primary animal")
    report["panels"] = []
    for method in SELECTORS:
        animal_records = [{"animal": animal, "mean_Q": float(np.mean([
            s["panels"][method]["mean_Q"] for s in report["sessions"] if s["animal"] == animal]))}
            for animal in animals]
        q = float(np.mean([record["mean_Q"] for record in animal_records]))
        report["panels"].append({"selector": method, "animals": animal_records,
                                "mean_Q": q, "delta_Q_from_random_baseline": q - baseline["mean_Q"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--trial-roles", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if args.max_seconds <= 0 or args.output_dir.exists():
        raise ValueError("Positive finite bound and fresh output path required")
    metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {item["filename"] for item in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Fixed eight-session panel required")
    selected = [(filename, *selected_metadata(metadata, roles, filename)) for filename in DEVELOPMENT]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before_blocks = resource.getrusage(resource.RUSAGE_SELF).ru_inblock

    def timeout_handler(*_):
        raise TimeoutError("Selected selector-comparison bound reached")

    def check_deadline():
        if time.monotonic() - started >= args.max_seconds:
            timeout_handler()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(selector_fits=0, selection_seconds=0.)
    report = {"analysis_id": "development_search", "phase": "five_fixed_selector_panels",
        "recipe": "outputs/selector_comparison_proposal.md", "status": "running",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "sessions": [], "fit_progress": progress, "fitting_trials_read": 0,
        "evaluation_trials_read": 0, "logical_neural_bytes": 0,
        "evaluation_neural_elements_read": False, "held_session_accessed": False,
        "valid_evaluations_charged": 0, "held_openings_charged": 0,
        "named_explanatory_amendment_activated": False, "global_search_completed": False,
        "source_root": str(args.source_root), "baseline_report": str(args.baseline_report),
        "boundary": "fixed training-only selection API; trusted evaluator; same-UID ambient filesystem is not an OS seal",
        "actual_exposure": "96 training rows on all96 electrodes; offline selection, no reduced-acquisition claim",
        "raw_arrays_or_coefficients_written": False}
    try:
        for filename, source, role, records in selected:
            check_deadline()
            session_start = time.monotonic()
            x, y, channels, guides = read_training_arrays(args.source_root / filename, source, role, records, check_deadline, report)
            models = fit_session(x, y, role, records, progress, check_deadline)
            fit_end = time.monotonic()
            ex, ey, _, _ = read_evaluation_arrays(args.source_root / filename, source, role, records, check_deadline, report, guides)
            panels = score_session(models, ex, ey, channels)
            report["sessions"].append({"filename": filename, "animal": role["animal"],
                "panels": panels, "training_read_and_fit_seconds": fit_end-session_start,
                "elapsed_seconds": time.monotonic()-session_start,
                "all_five_panel_fits_completed_before_evaluation_read": True})
            print(json.dumps({"session_completed": filename, "elapsed_seconds": time.monotonic()-session_start}), flush=True)
            del x, y, models, ex, ey, guides
        append_baseline_contrasts(report, json.loads(args.baseline_report.read_text()))
        report["valid_evaluations_charged"] = 5
        report["status"] = "completed_five_development_selector_panels"
    except Exception as error:
        report["status"] = "technical_failure_comparison_incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations_cache_dependent"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock-before_blocks
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "development_selectors.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "valid_evaluations_charged": 5,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
