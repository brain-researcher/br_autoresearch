#!/usr/bin/env python3
"""Complete fifteen fresh covariance-SVD sessions after the fixed completed resource pilot."""
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
from run_primary_covariance_svd import validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 270
CHILD_EXTERNAL_SECONDS = 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root",
                 "reference-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True,
                        help="Aggregate internal bound chosen from pilot resource facts")
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Selected aggregate bound must exceed one complete 300s child reserve")
    pilot = json.loads((args.bank_root/PILOT/"score_banks.json").read_text())
    validate_session(pilot, PILOT)
    remaining = [session for session in SESSIONS if session != PILOT]
    if len(remaining) != 15:
        raise ValueError("Exactly the fifteen remaining original sessions are required")
    if args.summary_out.exists() or any((args.bank_root/session).exists() for session in remaining):
        raise FileExistsError("Remaining session paths and complete summary must be fresh; no retries")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "pilot_session_reused": PILOT, "pilot_elapsed_seconds": pilot["elapsed_seconds"],
              "aggregate_internal_bound_seconds": args.max_seconds,
              "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS,
              "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
              "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
              "scientific_trial_count_increment": 0, "promotion_authorized": False,
              "search_complete": False}
    runner = Path(__file__).with_name("run_primary_covariance_svd.py")

    def deadline(*_):
        raise TimeoutError("Selected complete covariance_svd-panel aggregate deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        for session in remaining:
            available = args.max_seconds-(time.monotonic()-started)
            if available < CHILD_EXTERNAL_SECONDS:
                raise TimeoutError("Insufficient aggregate time for a complete 300s session reserve")
            command = [sys.executable, str(runner), "session", "--source", str(args.source),
                       "--trial-table", str(args.trial_table), "--scalar-bank-root", str(args.scalar_bank_root),
                       "--reference-bank-root", str(args.reference_bank_root), "--session", session,
                       "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            report["active_session"] = session
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            report["completed_remaining_sessions"].append({"session_id": session,
                "elapsed_seconds": item["elapsed_seconds"],
                "source_read_seconds": item["source_read_seconds"],
                "two_recipe_fit_and_score_seconds": item["two_recipe_fit_and_score_seconds"],
                "maximum_RSS_KiB": item["maximum_RSS_KiB"]})
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                              "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        report.pop("active_session", None)
        available = args.max_seconds-(time.monotonic()-started)
        if available <= 0:
            deadline()
        subprocess.run([sys.executable, str(runner), "summarize", "--bank-root", str(args.bank_root),
                        "--reference-bank-root", str(args.reference_bank_root),
                        "--reference-summary", str(args.reference_summary), "--out", str(args.summary_out)],
                       check=True, timeout=available)
        summary_path = args.summary_out/"primary_covariance_svd.json"
        summary = json.loads(summary_path.read_text())
        if (summary["status"] != "complete_covariance_svd_control_panel"
                or summary["scientific_trial_count_increment"] not in (0, 2)):
            raise ValueError("Complete two-panel summary or explicit structural duplicate required")
        report.update(status="complete_covariance_svd_remaining_panel", summary_path=str(summary_path),
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
