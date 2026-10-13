#!/usr/bin/env python3
"""EP08 selected training-only I/O and ridge cost pilot; no outer scoring.

Only the first development session's 96 selected fitting trials are read.
Inner CV uses fitting-pool trials, but its errors, selected penalties and
coefficients are not returned. No evaluation LFP or target is accessed.
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

from development_reader import selected_metadata, read_training_arrays
from development_design import candidate_designs
from ridge_training import fit_candidate, new_progress

SESSION = "Mihili_CO_20140217.mat"


def fit_training_curve(x, y, channels, role, check_deadline, progress):
    for design in candidate_designs(role, channels, replicates=1):
        check_deadline()
        fit_candidate(x[np.ix_(design["rows"], design["columns"])], y[design["rows"]],
                      design["folds"], progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--trial-roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if args.source_file.name != SESSION:
        raise ValueError("Only the selected first development session is permitted")
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior attempts")
    metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
    source, role, records = selected_metadata(metadata, roles, SESSION)
    started = time.monotonic()
    input_blocks_before = resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    fitting_start = None
    deadline = started + 540.

    def check_deadline(*_):
        if time.monotonic() >= deadline:
            raise TimeoutError("Selected540-second internal cost-pilot bound reached")

    def timeout_handler(*_):
        raise TimeoutError("Selected540-second internal cost-pilot bound reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(540)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"analysis_id": "source_support", "phase": "training_only_cost_pilot",
              "session": SESSION, "source_file": str(args.source_file),
              "source_file_bytes": args.source_file.stat().st_size,
              "trial_roles": str(args.trial_roles), "metadata_report": str(args.metadata_report),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "evaluation_neural_elements_read": False, "held_session_accessed": False,
              "outer_evaluation_predictions_or_scores_computed": False,
              "inner_training_CV_started": False, "inner_training_CV_completed": False,
              "valid_evaluations_charged": 0,
              "fitting_trials_read": 0, "logical_neural_bytes": 0,
              "coefficient_or_neural_value_artifacts_written": False, "status": "running"}
    fit_progress = new_progress()
    report["fit_progress"] = fit_progress
    try:
        x, y, channels, _guides = read_training_arrays(args.source_file, source, role, records, check_deadline, report)
        report["input_shape"] = list(x.shape)
        report["target_shape"] = list(y.shape)
        report["source_read_elapsed_seconds"] = time.monotonic() - started
        report["source_read_OS_block_input_operations"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock - input_blocks_before
        fitting_start = time.monotonic()
        report["inner_training_CV_started"] = True
        fit_training_curve(x, y, channels, role, check_deadline, fit_progress)
        report["inner_training_CV_completed"] = True
        report["status"] = "completed_training_only_no_scientific_score"
    except Exception as error:
        report["status"] = "technical_failure"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        if fitting_start is not None:
            report["fit_elapsed_seconds"] = time.monotonic() - fitting_start
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["OS_block_input_operations"] = resource.getrusage(resource.RUSAGE_SELF).ru_inblock - input_blocks_before
        report["physical_IO_note"] = "ru_inblock reports own-process block-input operations, depends on OS/filesystem cache, and is not HDF5 logical bytes; exact physical storage bytes unavailable"
        report["observed_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "training_cost_pilot.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
