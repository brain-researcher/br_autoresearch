#!/usr/bin/env python3
"""Run the fixed HW105 resource pilot or a separately bounded remainder."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import threading
import time

from primary_reference import SESSIONS
from run_elastic_net_panel import run_sessions, validate_workers
from run_primary_history_rbf import PANEL_STATUS, SUMMARY_FILE, SUMMARY_OPTIONS, read_json, validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 600
CHILD_EXTERNAL_SECONDS = 900
RESOURCE_KEYS = ("elapsed_seconds", "source_read_seconds", "candidate_and_control_fit_score_seconds",
                 "original_HW_replay_seconds", "prediction_write_seconds", "maximum_RSS_KiB")


def session_command(args, session):
    command = [sys.executable, str(Path(__file__).with_name("run_primary_history_rbf.py")),
        "session", "--session", session, "--out", str(args.bank_root/session),
        "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
    for option in ("source", "trial_table", "scalar_bank_root", "hw_bank_root"):
        command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("pilot", "remainder"):
        child = sub.add_parser(mode)
        for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root", "bank-root", "out"):
            child.add_argument("--"+name, type=Path, required=True)
        if mode == "remainder":
            for name in ("k-bank-root", "blend-bank-root", "hw-summary", "k-summary", "blend-summary", "summary-out"):
                child.add_argument("--"+name, type=Path, required=True)
            child.add_argument("--workers", type=int, required=True)
            child.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("One numerical thread per session worker required")
    allocated = int(os.environ["SLURM_CPUS_PER_TASK"])
    workers = 1 if args.mode == "pilot" else args.workers
    validate_workers(workers, allocated)
    if args.mode == "pilot":
        if (args.bank_root/PILOT).exists():
            raise FileExistsError("Fresh fixed pilot path required; no automatic retry")
        sessions = [PILOT]
    else:
        if args.max_seconds <= CHILD_EXTERNAL_SECONDS:
            raise ValueError("Root-selected aggregate bound must exceed a full 900-second child reserve")
        pilot = read_json(args.bank_root/PILOT/"score_banks.json")
        validate_session(pilot, PILOT)
        sessions = [s for s in SESSIONS if s != PILOT]
        if args.summary_out.exists() or any((args.bank_root/s).exists() for s in sessions):
            raise FileExistsError("Fresh fifteen-session remainder and summary required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "mode": args.mode, "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "selected_workers": workers, "allocated_CPUs": allocated,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS,
        "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "scientific_trial_count_increment": 0, "promotion_authorized": False,
        "search_complete": False, "completed_sessions": []}
    stop_event = threading.Event()
    def stop_dispatch(*_):
        stop_event.set()
    if args.mode == "remainder":
        report.update(pilot_session_reused=PILOT, pilot_elapsed_seconds=pilot["elapsed_seconds"],
            aggregate_internal_bound_seconds=args.max_seconds, required_remaining_sessions=sessions)
        signal.signal(signal.SIGALRM, stop_dispatch)
        signal.alarm(args.max_seconds)
    try:
        def run_one(s):
            # The inherited dispatcher reserves 300 seconds. This additional
            # check enforces this recipe's actual 900-second child envelope.
            if args.mode == "remainder" and (
                    stop_event.is_set() or args.max_seconds-(time.monotonic()-started) < CHILD_EXTERNAL_SECONDS):
                raise TimeoutError("Insufficient aggregate time for the complete HW105 child reserve")
            subprocess.run(session_command(args, s), check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = read_json(args.bank_root/s/"score_banks.json")
            validate_session(item, s)
            return {"session_id": s, **{k: item[k] for k in RESOURCE_KEYS},
                    "complete_trials": item["source"]["complete_trials"],
                    "physical_electrodes": len(item["source"]["common_LMP_electrodes"]),
                    "all_original_HW_replays_passed": True}
        def completed(s, item):
            report["completed_sessions"].append(item)
            print(json.dumps({"completed_session": s, "elapsed_seconds": item["elapsed_seconds"],
                "completed_count": len(report["completed_sessions"])}), flush=True)
        if args.mode == "pilot":
            resource_report = run_one(PILOT)
            completed(PILOT, resource_report)
            report.update(status="complete_history_rbf_resource_pilot", scientific_effects_released=False)
        else:
            report["completed_sessions"] = run_sessions(sessions, workers, run_one,
                started=started, max_seconds=args.max_seconds, stop_event=stop_event, on_complete=completed)
            available = args.max_seconds-(time.monotonic()-started)
            if stop_event.is_set() or available <= 0:
                raise TimeoutError("Root-selected HW105 aggregate deadline reached; children settled")
            command = [sys.executable, str(Path(__file__).with_name("run_primary_history_rbf.py")),
                       "summarize", "--out", str(args.summary_out)]
            for option in SUMMARY_OPTIONS:
                command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
            subprocess.run(command, check=True, timeout=available)
            path = args.summary_out/SUMMARY_FILE
            summary = read_json(path)
            if (summary["status"] != PANEL_STATUS or summary["scientific_trial_count_increment"] not in (0, 1, 2)
                    or summary["promotion_authorized"] is not False
                    or summary["all_new_saved_scores_replayed"] is not True
                    or summary["all_original_anchor_scores_replayed"] is not True
                    or summary["original_selectors_independently_replayed"] is not True):
                raise ValueError("Complete verified HW105 summary required")
            report.update(status="complete_history_rbf_remaining_panel", summary_path=str(path),
                scientific_trial_count_increment=summary["scientific_trial_count_increment"])
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "pilot_resources.json" if args.mode == "pilot" else "remaining_panel.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
