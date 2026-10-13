#!/usr/bin/env python3
"""Trusted evaluator for the adopted fixed EP08 development baseline panel.

The source reader and evaluation arrays belong here. The only candidate is
ridge_training.fit_candidate, which receives copies of training arrays plus
relative training folds. There is no candidate plugin or held-session route.
This enforces the callable/dataflow boundary, not hostile-code containment:
ordinary same-UID filesystem access remains ambient.
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

from prepare_development_trial_roles import DEVELOPMENT
from development_reader import selected_metadata, read_training_arrays, read_evaluation_arrays
from development_design import candidate_designs, TRAINING_TOTALS, ELECTRODE_COUNTS
from ridge_training import fit_candidate, new_progress


def fit_session_models(training_x, training_y, channels, role, check_deadline, fit_progress):
    """Fixed trusted adapter: convert metadata to training-only fitter inputs."""
    models = []
    for design in candidate_designs(role, channels, replicates=5):
        check_deadline()
        fitted = fit_candidate(
            training_x[np.ix_(design["rows"], design["columns"])].copy(),
            training_y[design["rows"]].copy(), design["folds"], fit_progress)
        # Rows/folds are no longer needed. No evaluation information enters fit_candidate.
        models.append(({key: value for key, value in design.items() if key not in ("rows", "folds")}, fitted))
    if len(models) != 48:
        raise RuntimeError("Fixed session panel is incomplete; no scoring permitted")
    return models


def score_frozen_model(fitted, evaluation_x, evaluation_y):
    """Trusted score: training-mean null, never the evaluation mean."""
    predicted = ((evaluation_x - fitted.input_mean) / fitted.input_sd) @ fitted.coefficients
    target = (evaluation_y - fitted.target_mean) / fitted.target_sd
    model_sse = float(np.sum((target - predicted) ** 2))
    null_sse = float(np.sum(target ** 2))
    if not np.isfinite(model_sse) or not np.isfinite(null_sse) or null_sse <= 0:
        raise ArithmeticError("Nonfinite or zero training-mean-null SSE; retain the unresolved full comparison")
    value = 1. - model_sse / null_sse
    if not np.isfinite(value):
        raise ArithmeticError("Nonfinite development score")
    return value


def score_session(models, evaluation_x, evaluation_y, check_deadline):
    score_by_cell = {}
    fitted_records = []
    for design, fitted in models:
        check_deadline()
        value = score_frozen_model(fitted, evaluation_x[:, design["columns"]], evaluation_y)
        key = (design["training_trials"], design["electrodes"])
        score_by_cell.setdefault(key, []).append(value)
        fitted_records.append({"training_trials": design["training_trials"],
                               "electrodes": design["electrodes"], "seed": design["seed"],
                               "channels": design["channels"], "alpha": fitted.alpha,
                               "constant_input_columns_in_selected_order": list(fitted.constant_input_columns),
                               "constant_target_columns_in_native_order": list(fitted.constant_target_columns),
                               "Q": value})
    cells = []
    means = {}
    for total in TRAINING_TOTALS:
        for count in ELECTRODE_COUNTS:
            values = score_by_cell[(total, count)]
            if count == 96:
                if len(values) != 1:
                    raise RuntimeError("All-electrode model must be fitted and scored once")
                values = values * 5
            if len(values) != 5:
                raise RuntimeError("Missing random subset replicate")
            means[(total, count)] = float(np.mean(values))
            cells.append({"training_trials": total, "electrodes": count,
                          "subset_Q": values, "mean_Q": means[(total, count)]})
    electrode_steps = [{"training_trials": total, "from_electrodes": a, "to_electrodes": b,
                        "mean_delta_Q": means[(total, b)] - means[(total, a)],
                        "paired_subset_delta_Q": [v - u for u, v in zip(
                            next(cell["subset_Q"] for cell in cells if cell["training_trials"] == total and cell["electrodes"] == a),
                            next(cell["subset_Q"] for cell in cells if cell["training_trials"] == total and cell["electrodes"] == b))]}
                       for total in TRAINING_TOTALS for a, b in zip(ELECTRODE_COUNTS[:-1], ELECTRODE_COUNTS[1:])]
    trial_steps = [{"electrodes": count, "from_training_trials": a, "to_training_trials": b,
                    "mean_delta_Q": means[(b, count)] - means[(a, count)],
                    "paired_subset_delta_Q": [v - u for u, v in zip(
                        next(cell["subset_Q"] for cell in cells if cell["training_trials"] == a and cell["electrodes"] == count),
                        next(cell["subset_Q"] for cell in cells if cell["training_trials"] == b and cell["electrodes"] == count))]}
                   for count in ELECTRODE_COUNTS for a, b in zip(TRAINING_TOTALS[:-1], TRAINING_TOTALS[1:])]
    return {"cells": cells, "fitted_models": fitted_records,
            "electrode_steps": electrode_steps, "trial_steps": trial_steps,
            "mean_Q": float(np.mean(list(means.values())))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--trial-roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True,
                        help="Coordinator-selected internal bound after the training-only pilot")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if args.max_seconds <= 0 or args.output_dir.exists():
        raise ValueError("Positive selected wall bound and fresh output directory required")
    metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
    if len(roles["sessions"]) != 8 or {item["filename"] for item in roles["sessions"]} != set(DEVELOPMENT):
        raise ValueError("Preserve the fixed eight-session development panel")
    # Resolve and validate all roles before any numeric payload is opened.
    selected = [(filename, *selected_metadata(metadata, roles, filename)) for filename in DEVELOPMENT]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    input_blocks_before = resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    deadline = started + args.max_seconds

    def check_deadline(*_):
        if time.monotonic() >= deadline:
            raise TimeoutError("Selected development-comparison bound reached")

    def timeout_handler(*_):
        raise TimeoutError("Selected development-comparison bound reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_seconds)
    report = {"analysis_id": "simple_comparisons", "phase": "fixed_random_development_baseline",
              "recipe": "outputs/development_comparison_proposal.md",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "source_root": str(args.source_root), "trial_roles": str(args.trial_roles),
              "metadata_report": str(args.metadata_report), "selected_max_seconds": args.max_seconds,
              "status": "running", "sessions": [], "fit_progress": new_progress(),
              "fitting_trials_read": 0, "evaluation_trials_read": 0, "logical_neural_bytes": 0,
              "evaluation_neural_elements_read": False, "held_session_accessed": False,
              "valid_evaluations_charged": 0, "held_openings_charged": 0,
              "named_explanatory_amendment_activated": False, "global_search_completed": False,
              "boundary": "fixed training-only fitter API; trusted evaluator retains evaluation arrays; same-UID ambient filesystem access is not an OS seal",
              "source_population_units": "all native M1_spikes columns, ten-bin mean in source per-bin units; no raw-integer-count or Hz conversion claim",
              "raw_neural_arrays_or_coefficients_written": False}
    try:
        for filename, source, role, records in selected:
            check_deadline()
            path = args.source_root / filename
            session_start = time.monotonic()
            training_x, training_y, channels, guides = read_training_arrays(path, source, role, records, check_deadline, report)
            # This function has no evaluation array or source path argument.
            models = fit_session_models(training_x, training_y, channels, role, check_deadline, report["fit_progress"])
            fitted_time = time.monotonic()
            # First evaluation-neural access occurs only after all 48 models are frozen.
            evaluation_x, evaluation_y, _, _ = read_evaluation_arrays(path, source, role, records, check_deadline, report, guides)
            session = score_session(models, evaluation_x, evaluation_y, check_deadline)
            session.update({"filename": filename, "animal": role["animal"],
                            "source_file_bytes": path.stat().st_size, "population_columns": int(training_y.shape[1]),
                            "training_read_and_fit_seconds": fitted_time - session_start,
                            "elapsed_seconds": time.monotonic() - session_start,
                            "all_session_fits_completed_before_evaluation_read": True})
            report["sessions"].append(session)
            # The evaluator never calls the fitter with this session's evaluation arrays.
            del training_x, training_y, models, evaluation_x, evaluation_y, guides
        animals = sorted({item["animal"] for item in report["sessions"]})
        if len(animals) != 2 or any(sum(item["animal"] == animal for item in report["sessions"]) != 4 for animal in animals):
            raise RuntimeError("Required four-session/two-animal aggregation unavailable")
        report["animals"] = [{"animal": animal, "mean_Q": float(np.mean([
            item["mean_Q"] for item in report["sessions"] if item["animal"] == animal]))} for animal in animals]
        report["mean_Q"] = float(np.mean([item["mean_Q"] for item in report["animals"]]))
        report["valid_evaluations_charged"] = 1
        report["status"] = "completed_one_development_baseline_no_scientific_winner"
    except Exception as error:
        report["status"] = "technical_failure_full_comparison_unresolved"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock - input_blocks_before
        report["physical_IO_note"] = "Own-process ru_inblock is cache-dependent block-input operations, not exact storage or HDF5 logical bytes"
        report["actual_CPU_allocation_charge"] = "Coordinator records terminal Slurm AllocCPUS times elapsed, including failed attempts"
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "development_baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "slurm_job_id", "elapsed_seconds", "valid_evaluations_charged")}), flush=True)


if __name__ == "__main__":
    main()
