#!/usr/bin/env python3
"""One finite sixteen-session timing control from preserved prediction arrays."""
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

from primary_reference import LAMBDAS, SESSIONS
from primary_temporal_offset import ANCHOR, RECIPE, frozen_selection, score_session, summarize_sessions
from run_primary_reference import trial_records
from source_adapter import compute_only


def load_frozen_predictions(directory, session, report, selection, rows):
    source = report["source"]
    if (report["status"] != "complete_three_session_banks" or report["session_id"] != session
            or source["session_id"] != session or source["native_offsets"] != [0, 14]
            or source["native_bin_seconds"] != .03 or source["complete_trials"] != len(rows)
            or not source["common_LMP_electrodes"]
            or source["common_LMP_electrodes"] != sorted(set(source["common_LMP_electrodes"]))):
        raise ValueError("Preserved complete native current-bin LMP bank required")
    bank = report["banks"][ANCHOR]
    chosen = frozen_selection(session, selection, bank)
    path = directory/(ANCHOR+"_predictions.npz")
    with np.load(path, allow_pickle=False) as saved:
        actual, baseline = saved["actual_xy"], saved["baseline_xy"]
        directions, ids = saved["directions"], saved["trial_ids"]
        predictions = saved["candidate_xy_by_lambda"]
    if (not np.array_equal(ids, np.array([r["trial_id"] for r in rows]))
            or not np.array_equal(directions, np.array([int(r["direction"]) for r in rows]))
            or actual.shape != (len(rows), 15, 2) or baseline.shape != actual.shape
            or predictions.shape != (len(LAMBDAS), *actual.shape)):
        raise ValueError("Preserved prediction arrays differ from original complete trial identities")
    # NPZ stores all five rows together; only the original selected row is scored.
    return actual, baseline, predictions[chosen].copy(), directions, bank, str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("anchor-bank-root", "anchor-summary", "trial-table", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=150)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if not 0 < args.max_seconds <= 150:
        raise ValueError("Selected report-only comparison bound is at most150seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "max_seconds": args.max_seconds, "new_fits": 0, "new_source_values_read": False,
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False,
        "anchor_summary": str(args.anchor_summary), "anchor_bank_root": str(args.anchor_bank_root),
        "trial_table": str(args.trial_table), "completed_sessions": []}
    sessions = {}
    def deadline(*_):
        raise TimeoutError("Selected full timing-control report deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        reference = json.loads(args.anchor_summary.read_text())
        anchors = [p for p in reference["panels"] if p["recipe"] == ANCHOR]
        if (reference["status"] != "complete_three_multichannel_comparisons" or len(anchors) != 1
                or len(anchors[0]["sessions"]) != len(SESSIONS)
                or {r["session_id"] for r in anchors[0]["sessions"]} != set(SESSIONS)):
            raise ValueError("Complete preserved independently selected LMP panel required")
        selected = {r["session_id"]: r for r in anchors[0]["sessions"]}
        for session in SESSIONS:
            check()
            t = time.monotonic()
            directory = args.anchor_bank_root/session
            old = json.loads((directory/"score_banks.json").read_text())
            rows = trial_records(args.trial_table, session)
            actual, baseline, prediction, labels, bank, path = load_frozen_predictions(
                directory, session, old, selected[session], rows)
            result = score_session(actual, baseline, prediction, labels, bank, selected[session], check)
            result.update(prediction_path=path, original_trial_ids=[r["trial_id"] for r in rows],
                original_directions=labels.tolist(), physical_electrodes=old["source"]["common_LMP_electrodes"],
                native_target_offsets=[0, 14], native_prediction_offsets=[0, 14],
                elapsed_seconds=time.monotonic()-t)
            sessions[session] = result
            report["completed_sessions"].append(session)
            print(json.dumps({"session": session, "status": "complete_timing_session", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        check()
        report.update(summarize_sessions(sessions))
        report.update(status="complete_fixed_temporal_offset_comparison", original_anchor_panel=anchors[0],
            fitted_model_selection_changed=False, input_lambda_prediction_rows_read_per_session=5,
            input_lambda_prediction_rows_scored_per_session=1, offset_Monte_Carlo_draws=0)
    except Exception as error:
        report.update(status="incomplete_preserved", error=f"{type(error).__name__}: {error}",
                      partial_session_reports=sessions)
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.out/"primary_temporal_offset.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
