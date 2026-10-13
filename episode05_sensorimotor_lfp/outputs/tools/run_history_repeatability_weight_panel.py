#!/usr/bin/env python3
"""Complete fifteen fresh HW sessions after the fixed completed resource pilot."""
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
import time

from primary_reference import SESSIONS
from run_primary_history_repeatability_weight import validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 270
CHILD_EXTERNAL_SECONDS = 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "weight-bank-root",
                 "weight-summary", "history-bank-root", "history-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True,
                        help="Aggregate internal bound selected from the complete pilot's resource facts")
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Aggregate bound must exceed one complete 300-second child reserve")
    pilot = json.loads((args.bank_root/PILOT/"score_banks.json").read_text())
    validate_session(pilot, PILOT)
    remaining = [s for s in SESSIONS if s != PILOT]
    if len(remaining) != 15:
        raise ValueError("Exactly fifteen remaining original sessions required")
    if args.summary_out.exists() or any((args.bank_root/s).exists() for s in remaining):
        raise FileExistsError("Fresh remaining session and summary paths required; no retries")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_session_reused": PILOT, "pilot_elapsed_seconds": pilot["elapsed_seconds"],
        "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS, "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    runner = Path(__file__).with_name("run_primary_history_repeatability_weight.py")
    def deadline(*_):
        raise TimeoutError("Selected HW panel aggregate deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        for session in remaining:
            if args.max_seconds-(time.monotonic()-started) < CHILD_EXTERNAL_SECONDS:
                raise TimeoutError("Insufficient aggregate time for a full 300-second session reserve")
            command = [sys.executable, str(runner), "session", "--session", session,
                       "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            for option in ("source", "trial_table", "scalar_bank_root", "reference_bank_root", "weight_bank_root"):
                command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
            report["active_session"] = session
            # subprocess.run kills and reaps its direct child on timeout or
            # an aggregate-deadline exception; no background workers exist.
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            report["completed_remaining_sessions"].append({"session_id": session,
                **{key: item[key] for key in ("elapsed_seconds", "source_read_seconds",
                                             "two_recipe_fit_and_score_seconds", "maximum_RSS_KiB")}})
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                              "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        report.pop("active_session", None)
        available = args.max_seconds-(time.monotonic()-started)
        if available <= 0:
            deadline()
        command = [sys.executable, str(runner), "summarize", "--out", str(args.summary_out)]
        for option in ("bank_root", "history_bank_root", "history_summary", "weight_bank_root", "weight_summary"):
            command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/"primary_history_repeatability_weight.json"
        summary = json.loads(path.read_text())
        if (summary["status"] != "complete_history_repeatability_weight_control_panel"
                or summary["scientific_trial_count_increment"] not in (0, 1, 2)):
            raise ValueError("Complete sixteen-session HW panel with exact per-recipe accounting required")
        report.update(status="complete_history_repeatability_weight_remaining_panel", summary_path=str(path),
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
