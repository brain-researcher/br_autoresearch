#!/usr/bin/env python3
"""Finish seven fresh sessions and summarize exactly eight old plus eight native banks."""
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

from primary_elastic_net_native import BACKEND, RECIPES
from primary_reference import SESSIONS
from run_elastic_net_panel import (CHILD_EXTERNAL_SECONDS, CHILD_INTERNAL_SECONDS,
                                   run_sessions, validate_workers)
from run_primary_elastic_net_native import validate_session
from source_adapter import compute_only

PILOT = "Mihili_20140307"
PRESERVED = ("Chewie_20150313", "Mihili_20140217", "Mihili_20140218", "Mihili_20140303",
             "Mihili_20140304", "Mihili_20140306", "Chewie_20160929", "Chewie_20161005")
NEW = (PILOT, "Chewie_20161006", "Chewie_20161007", "Chewie_20161014", "Chewie_20161021",
       "Chewie_20150319", "Chewie_20150629", "Chewie_20150630")


def selected_bank_sources(preserved_root, native_root):
    if (len(PRESERVED) != 8 or len(NEW) != 8 or set(PRESERVED) & set(NEW)
            or set(PRESERVED) | set(NEW) != set(SESSIONS)
            or preserved_root.resolve() == native_root.resolve()):
        raise ValueError("Exactly the selected disjoint eight preserved/eight native original sessions required")
    return {session: str((preserved_root if session in PRESERVED else native_root)/session/"score_banks.json")
            for session in SESSIONS}


def validate_native(item, session):
    validate_session(item, session)
    if item.get("solver_backend") != BACKEND or item["banks"][RECIPES[0]].get("solver_backend") != BACKEND:
        raise ValueError("New bank must use the adopted C99 single-sweep backend")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root",
                 "reference-summary", "preserved-bank-root", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Every scientific worker requires explicit single-thread settings")
    allocated_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    validate_workers(args.workers, allocated_cpus)
    if args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Selected remaining aggregate bound must exceed a complete 300s child reserve")
    mapping = selected_bank_sources(args.preserved_bank_root, args.bank_root)
    # Read exactly the adopted completed sources, never failed old Mi0307 or
    # an alternate path chosen by which artifact happens to exist.
    for session in PRESERVED:
        validate_session(json.loads(Path(mapping[session]).read_text()), session)
    pilot = json.loads(Path(mapping[PILOT]).read_text())
    validate_native(pilot, PILOT)
    remaining = [session for session in SESSIONS if session in NEW and session != PILOT]
    if len(remaining) != 7:
        raise ValueError("Exactly seven previously unlaunched sessions required")
    if args.summary_out.exists() or any((args.bank_root/session).exists() for session in remaining):
        raise FileExistsError("Remaining native banks and summary must be fresh; no fallback or refit")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "solver_backend_for_new_sessions": BACKEND,
              "preserved_completed_sessions": list(PRESERVED), "pilot_session_reused": PILOT,
              "pilot_elapsed_seconds": pilot["elapsed_seconds"], "selected_score_bank_sources": mapping,
              "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
              "selected_workers": args.workers, "allocated_CPUs": allocated_cpus,
              "aggregate_internal_bound_seconds": args.max_seconds,
              "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS,
              "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
              "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    runner = Path(__file__).with_name("run_primary_elastic_net_native.py")
    stop_event = threading.Event()

    def deadline(*_):
        stop_event.set()

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        def run_one(session):
            command = [sys.executable, str(runner), "session", "--source", str(args.source),
                "--trial-table", str(args.trial_table), "--scalar-bank-root", str(args.scalar_bank_root),
                "--reference-bank-root", str(args.reference_bank_root), "--session", session,
                "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = json.loads(Path(mapping[session]).read_text())
            validate_native(item, session)
            return {"session_id": session, "elapsed_seconds": item["elapsed_seconds"],
                    "source_read_seconds": item["source_read_seconds"],
                    "two_recipe_fit_and_score_seconds": item["two_recipe_fit_and_score_seconds"],
                    "maximum_RSS_KiB": item["maximum_RSS_KiB"]}

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
            raise TimeoutError("Selected native recovery aggregate deadline reached")
        selection_path = args.out/"selected_bank_sources.json"
        selection_path.write_text(json.dumps(mapping, indent=2)+"\n")
        subprocess.run([sys.executable, str(runner), "summarize", "--session-bank-map", str(selection_path),
            "--reference-bank-root", str(args.reference_bank_root), "--reference-summary", str(args.reference_summary),
            "--out", str(args.summary_out)], check=True, timeout=available)
        summary_path = args.summary_out/"primary_elastic_net.json"
        summary = json.loads(summary_path.read_text())
        if (summary["status"] != "complete_elastic_net_control_panel"
                or summary["scientific_trial_count_increment"] not in (0, 2)
                or summary["selected_score_bank_sources"] != mapping):
            raise ValueError("Complete scientifically unchanged mixed-source panel required")
        report.update(status="complete_elastic_net_recovery_remaining_panel", summary_path=str(summary_path),
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
