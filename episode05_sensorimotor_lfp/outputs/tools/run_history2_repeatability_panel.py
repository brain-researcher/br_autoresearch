#!/usr/bin/env python3
"""Finish fifteen H2W sessions sequentially under a root-selected finite bound."""
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

from primary_history2_repeatability_weight import RECIPES
from primary_reference import SESSIONS
from run_primary_history2_repeatability_weight import validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 270
CHILD_EXTERNAL_SECONDS = 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "w-bank-root", "hw-bank-root", "capacity-bank-root",
                 "w-summary", "hw-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--blend-summary", type=Path)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1 or args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Actual CPU allocation and finite bound beyond full300second child reserve required")
    pilot = json.loads((args.bank_root/PILOT/"score_banks.json").read_text())
    validate_session(pilot, PILOT)
    remaining = [session for session in SESSIONS if session != PILOT]
    if len(remaining) != 15 or args.summary_out.exists() or any((args.bank_root/session).exists() for session in remaining):
        raise ValueError("Preserved pilot, fifteen fresh remaining paths and fresh summary required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot_session_reused": PILOT,
        "pilot_elapsed_seconds": pilot["elapsed_seconds"], "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS, "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "workers": 1, "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False, "original_W_refitted": False,
        "promotion_authorized": False, "search_complete": False, "new_promotion_gate": False}
    runner = Path(__file__).with_name("run_primary_history2_repeatability_weight.py")
    stopped = False
    def deadline(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        for session in remaining:
            if stopped or args.max_seconds-(time.monotonic()-started) < CHILD_EXTERNAL_SECONDS:
                raise TimeoutError("Insufficient complete300second H2W child reserve")
            command = [sys.executable, str(runner), "session", "--session", session,
                "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            for option in ("source", "trial_table", "scalar_bank_root", "w_bank_root", "hw_bank_root", "capacity_bank_root"):
                command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            report["completed_remaining_sessions"].append({"session_id": session,
                **{key: item[key] for key in ("elapsed_seconds", "source_read_seconds", "candidate_and_control_fit_score_seconds", "maximum_RSS_KiB")}})
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        available = args.max_seconds-(time.monotonic()-started)
        if stopped or available <= 0 or len(report["completed_remaining_sessions"]) != 15:
            raise TimeoutError("H2W finite aggregate deadline reached after bounded children settled")
        command = [sys.executable, str(runner), "summarize", "--out", str(args.summary_out)]
        for option in ("bank_root", "w_bank_root", "hw_bank_root", "w_summary", "hw_summary"):
            command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
        if args.blend_summary is not None:
            command.extend(["--blend-summary", str(args.blend_summary)])
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/"primary_history2_repeatability_weight.json"
        summary = json.loads(path.read_text())
        counts = summary["scientific_trial_count_by_recipe"]
        if (summary["status"] != "complete_history2_repeatability_panel" or summary["recipe_order"] != list(RECIPES)
                or set(counts) != set(RECIPES) or any(type(value) is not int or value not in (0, 1) for value in counts.values())
                or summary["scientific_trial_count_increment"] != sum(counts.values())
                or summary["literal_null_control_count_increment"] != counts[RECIPES[1]]
                or summary["alternative_explanation_count_increment"] != counts[RECIPES[1]]
                or any(summary[key] is not True for key in ("all_original_replays", "all_new_saved_scores_replayed", "original_selectors_independently_replayed"))
                or summary["original_HW_refitted"] is not False or summary["original_W_refitted"] is not False
                or summary["new_promotion_gate"] is not False or summary["promotion_authorized"] is not False):
            raise ValueError("Complete H2W summary and candidate-then-control accounting required")
        report.update(status="complete_history2_repeatability_remaining_panel", summary_path=str(path),
            **{key: summary[key] for key in ("scientific_trial_count_increment", "scientific_trial_count_by_recipe",
                "literal_null_control_count_increment", "alternative_explanation_count_increment")})
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
