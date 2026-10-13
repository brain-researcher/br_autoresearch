#!/usr/bin/env python3
"""Bounded sequential completion of the prequalified intersection support panel."""
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
from run_primary_intersection import qualify_global, validate_session
from source_adapter import compute_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "metadata", "scalar-bank-root", "hw-bank-root", "hw-summary",
                 "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("max-seconds", "prior-valid", "prior-patience", "prior-alternatives", "prior-nulls"):
        parser.add_argument("--"+name, type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread settings required")
    validate_workers(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    if args.max_seconds <= 300:
        raise ValueError("Finite selected aggregate bound must retain a complete300-second child reserve")
    metadata, descriptions, _ = qualify_global(args.metadata, args.trial_table, args.scalar_bank_root, args.hw_bank_root)
    full_i = all(row["support_sets"][0]["full_support"] for row in descriptions.values())
    same_r = all(s["electrode_ids"] == row["intersection_electrode_ids"]
                 for row in descriptions.values() for s in row["support_sets"][1:])
    charges = int(not full_i)+int(not same_r)
    if (not 28 <= args.prior_valid <= 64 or args.prior_patience < 0
            or args.prior_valid+charges > 64
            or min(args.prior_alternatives, args.prior_nulls) < 0):
        raise ValueError("Live depth must accommodate both metadata-known distinct recipes before dispatch")
    pilot_session = metadata["pilot_session"]
    if pilot_session is None:
        raise ValueError("A globally full-support duplicate requires artifact-only closeout, not this numerical remainder")
    pilot = json.loads((args.bank_root/pilot_session/"score_banks.json").read_text())
    validate_session(pilot, pilot_session)
    if pilot["qualified_support"] != descriptions[pilot_session] or pilot["metadata_pilot_session"] != pilot_session:
        raise ValueError("Preserved complete pilot must be the frozen metadata-workload choice")
    remaining = [s for s in SESSIONS if s != pilot_session]
    if args.summary_out.exists() or any((args.bank_root/s).exists() for s in remaining):
        raise ValueError("Preserve the complete pilot; all remaining outputs and summary must be fresh")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot_session_reused": pilot_session,
        "pilot_elapsed_seconds": pilot["elapsed_seconds"], "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": 270, "child_external_bound_seconds": 300, "selected_workers": 1,
        "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False,
        "frozen_live_accounting": {k: getattr(args, k) for k in
            ("prior_valid", "prior_patience", "prior_alternatives", "prior_nulls")}}
    runner = Path(__file__).with_name("run_primary_intersection.py")
    stop = threading.Event()
    signal.signal(signal.SIGALRM, lambda *_: stop.set())
    signal.alarm(args.max_seconds)
    try:
        def run_one(session):
            command = [sys.executable, str(runner), "session", "--session", session,
                       "--out", str(args.bank_root/session), "--max-seconds", "270"]
            for option in ("source", "trial_table", "metadata", "scalar_bank_root", "hw_bank_root"):
                command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
            subprocess.run(command, check=True, timeout=300)
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            if item["qualified_support"] != descriptions[session]:
                raise ValueError("Every remaining bank must preserve the same complete guide qualification")
            return {"session_id": session, **{key: item[key] for key in
                ("elapsed_seconds", "source_read_seconds", "reduced_fit_seconds", "maximum_RSS_KiB", "new_unique_fit_count")}}
        def complete(session, item):
            report["completed_remaining_sessions"].append(item)
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                              "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        report["completed_remaining_sessions"] = run_sessions(remaining, 1, run_one, started=started,
            max_seconds=args.max_seconds, stop_event=stop, on_complete=complete)
        available = args.max_seconds-(time.monotonic()-started)
        if available <= 0 or stop.is_set():
            raise TimeoutError("Selected remainder deadline reached with all children settled")
        command = [sys.executable, str(runner), "summarize", "--bank-root", str(args.bank_root),
                   "--hw-summary", str(args.hw_summary), "--out", str(args.summary_out)]
        for option in ("prior_valid", "prior_patience", "prior_alternatives", "prior_nulls"):
            command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/"primary_intersection.json"
        summary = json.loads(path.read_text())
        if (summary["status"] != "complete_intersection_sensitivity_panel"
                or summary["scientific_trial_count_increment"] != charges
                or summary["literal_null_control_count_increment"] != 0):
            raise ValueError("Both complete support sensitivities and predetermined charges required")
        report.update(status="complete_intersection_remaining_panel", summary_path=str(path),
                      scientific_trial_count_increment=charges)
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
