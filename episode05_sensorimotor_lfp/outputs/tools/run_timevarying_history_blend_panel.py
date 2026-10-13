#!/usr/bin/env python3
"""Finish fifteen fresh saved-artifact time schedule sessions after the fixed pilot."""
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
from run_primary_timevarying_history_blend import (PATH_OPTIONS, PANEL_STATUS, REPLAY_FLAGS,
    SUMMARY_FILE, read_json, validate_session)
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 60
CHILD_EXTERNAL_SECONDS = 90


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PATH_OPTIONS, "bank_root", "summary_out", "out"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1 or args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Actual allocated CPU and finite aggregate bound exceeding one full child reserve required")
    pilot = read_json(args.bank_root/PILOT/"score_banks.json")
    validate_session(pilot, PILOT)
    remaining = [s for s in SESSIONS if s != PILOT]
    if args.summary_out.exists() or any((args.bank_root/session).exists() for session in remaining):
        raise ValueError("Preserved pilot, fifteen fresh sessions and a fresh summary path required")
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
    runner = Path(__file__).with_name("run_primary_timevarying_history_blend.py")
    common = []
    for name in PATH_OPTIONS:
        common.extend(["--"+name.replace("_", "-"), str(getattr(args, name))])
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
                raise TimeoutError("Insufficient full 90-second artifact child reserve")
            command = [sys.executable, str(runner), "session", *common, "--session", session,
                "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = read_json(args.bank_root/session/"score_banks.json")
            validate_session(item, session)
            report["completed_remaining_sessions"].append({"session_id": session, **{key: item[key] for key in
                ("elapsed_seconds", "artifact_read_and_replay_seconds", "mixture_score_seconds", "maximum_RSS_KiB")}})
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        available = args.max_seconds-(time.monotonic()-started)
        if stopped or available <= 0 or len(report["completed_remaining_sessions"]) != 15:
            raise TimeoutError("Complete time schedule panel exceeded its selected finite bound")
        command = [sys.executable, str(runner), "summarize", *common,
            "--bank-root", str(args.bank_root), "--out", str(args.summary_out)]
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/SUMMARY_FILE
        summary = read_json(path)
        count = summary["scientific_trial_count_increment"]
        if (summary["status"] != PANEL_STATUS or type(count) is not int or count not in (0, 1)
                or count != int(not summary["structural_duplicate_of"])
                or summary["literal_null_control_count_increment"] != 0
                or summary["alternative_explanation_count_increment"] != 0 or summary["neural_refits"] != 0
                or summary["provider_source_values_read"] is not False or summary["promotion_authorized"] is not False
                or any(summary[flag] is not True for flag in (*REPLAY_FLAGS, "all_selected_prediction_scores_replayed",
                    "all_selected_reversal_scores_replayed", "all_exact_duplicate_flags_replayed"))
                or set(summary["selected_prediction_paths"]) != set(SESSIONS)
                or summary["reversal_descriptive_only"] is not True
                or summary["reversal_scientific_trial_count_increment"] != 0):
            raise ValueError("Complete exact-replay panel and independent candidate/duplicate accounting required")
        report.update(status="complete_timevarying_history_blend_remaining_panel", summary_path=str(path),
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
