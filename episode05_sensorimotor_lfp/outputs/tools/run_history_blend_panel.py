#!/usr/bin/env python3
"""Finish fifteen fresh artifact-only blend sessions after the preserved pilot."""
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
from run_primary_history_blend import validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 60
CHILD_EXTERNAL_SECONDS = 90


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("w-bank-root", "hw-bank-root", "w-summary", "hw-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1 or args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("An actual CPU allocation and a finite bound exceeding one full 90-second child reserve required")
    pilot = json.loads((args.bank_root/PILOT/"score_banks.json").read_text())
    validate_session(pilot, PILOT)
    remaining = [session for session in SESSIONS if session != PILOT]
    if len(remaining) != 15 or args.summary_out.exists() or any((args.bank_root/session).exists() for session in remaining):
        raise ValueError("Exactly fifteen fresh remaining paths and a fresh final summary required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot_session_reused": PILOT,
        "pilot_elapsed_seconds": pilot["elapsed_seconds"], "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS, "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "workers": 1, "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "neural_refits": 0,
        "provider_source_values_read": False, "promotion_authorized": False, "search_complete": False}
    runner = Path(__file__).with_name("run_primary_history_blend.py")
    stopped = False
    def deadline(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        for session in remaining:
            available = args.max_seconds-(time.monotonic()-started)
            if stopped or available < CHILD_EXTERNAL_SECONDS:
                raise TimeoutError("Insufficient complete 90-second artifact-child reserve")
            command = [sys.executable, str(runner), "session", "--session", session,
                "--w-bank-root", str(args.w_bank_root), "--hw-bank-root", str(args.hw_bank_root),
                "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            details = {"session_id": session, **{key: item[key] for key in
                ("elapsed_seconds", "artifact_read_and_replay_seconds", "mixture_score_seconds", "maximum_RSS_KiB")}}
            report["completed_remaining_sessions"].append(details)
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        available = args.max_seconds-(time.monotonic()-started)
        if stopped or available <= 0 or len(report["completed_remaining_sessions"]) != 15:
            raise TimeoutError("Complete artifact panel exceeded its selected finite deadline")
        command = [sys.executable, str(runner), "summarize", "--out", str(args.summary_out)]
        for option in ("bank_root", "w_bank_root", "hw_bank_root", "w_summary", "hw_summary"):
            command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/"primary_history_blend.json"
        summary = json.loads(path.read_text())
        count = summary["scientific_trial_count_increment"]
        if (summary["status"] != "complete_history_blend_panel" or type(count) is not int or count not in (0, 1)
                or count != int(not summary["structural_duplicate_of"])
                or summary["literal_null_control_count_increment"] != 0
                or summary["alternative_explanation_count_increment"] != 0 or summary["neural_refits"] != 0
                or summary["provider_source_values_read"] is not False
                or summary["original_selectors_independently_replayed"] is not True
                or summary["all_selected_prediction_scores_replayed"] is not True
                or summary["promotion_authorized"] is not False):
            raise ValueError("Complete selected-prediction panel and exact candidate/duplicate accounting required")
        report.update(status="complete_history_blend_remaining_panel", summary_path=str(path),
            scientific_trial_count_increment=count, structural_duplicate_of=summary["structural_duplicate_of"])
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.out/"remaining_panel.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
