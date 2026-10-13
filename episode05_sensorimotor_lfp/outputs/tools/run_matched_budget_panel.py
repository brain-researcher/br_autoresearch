#!/usr/bin/env python3
"""Bounded remaining C/T/CT sessions after the fixed complete metadata-selected pilot."""
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
from run_primary_matched_budget import validate_session
from source_adapter import compute_only

PILOT = "Chewie_20161014"
CHILD_INTERNAL_SECONDS = 270
CHILD_EXTERNAL_SECONDS = 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "metadata", "scalar-bank-root", "hw-bank-root", "hw-summary",
                 "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread settings required")
    try:
        allocated_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Actual SLURM_CPUS_PER_TASK must explicitly name the allocation") from error
    validate_workers(args.workers, allocated_cpus)
    if args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Selected aggregate bound must exceed one full child reserve")
    pilot = json.loads((args.bank_root/PILOT/"score_banks.json").read_text())
    validate_session(pilot, PILOT)
    remaining = [s for s in SESSIONS if s != PILOT]
    if len(remaining) != 15 or args.summary_out.exists() or any((args.bank_root/s).exists() for s in remaining):
        raise ValueError("Exactly fifteen fresh remaining sessions and a fresh summary required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot_session_reused": PILOT,
        "pilot_elapsed_seconds": pilot["elapsed_seconds"], "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS, "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "selected_workers": args.workers, "allocated_CPUs": allocated_cpus,
        "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    runner = Path(__file__).with_name("run_primary_matched_budget.py")
    stop_event = threading.Event()
    def deadline(*_):
        stop_event.set()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        def run_one(session):
            command = [sys.executable, str(runner), "session", "--session", session,
                "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            for option in ("source", "trial_table", "metadata", "scalar_bank_root", "hw_bank_root"):
                command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            return {"session_id": session,
                **{key: item[key] for key in ("elapsed_seconds", "global_metadata_seconds", "source_read_seconds",
                    "reduced_fit_seconds", "three_recipe_fit_and_score_seconds", "maximum_RSS_KiB", "new_unique_fit_count", "exact_full_HW_view_reuses")}}

        def completed(session, item):
            report["completed_remaining_sessions"].append(item)
            order = {value: index for index, value in enumerate(remaining)}
            report["completed_remaining_sessions"].sort(key=lambda row: order[row["session_id"]])
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)

        report["completed_remaining_sessions"] = run_sessions(remaining, args.workers, run_one,
            started=started, max_seconds=args.max_seconds, stop_event=stop_event, on_complete=completed)
        available = args.max_seconds-(time.monotonic()-started)
        if available <= 0 or stop_event.is_set():
            raise TimeoutError("Selected matched-budget aggregate deadline reached; all children settled")
        subprocess.run([sys.executable, str(runner), "summarize", "--bank-root", str(args.bank_root),
            "--hw-summary", str(args.hw_summary), "--out", str(args.summary_out)], check=True, timeout=available)
        path = args.summary_out/"primary_matched_budget.json"
        summary = json.loads(path.read_text())
        if (summary["status"] != "complete_matched_budget_sensitivity_panel"
                or summary["scientific_trial_count_increment"] not in (0, 1, 2, 3)
                or summary["null_control_trial_count_increment"] != 0):
            raise ValueError("Complete three-condition panel with exact stress-only accounting required")
        report.update(status="complete_matched_budget_remaining_panel", summary_path=str(path),
            scientific_trial_count_increment=summary["scientific_trial_count_increment"],
            total_valid_primary_trials_including_prior=summary["total_valid_primary_trials_including_prior"])
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.out/"remaining_panel.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
