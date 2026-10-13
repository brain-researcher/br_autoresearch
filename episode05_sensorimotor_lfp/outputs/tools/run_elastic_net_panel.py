#!/usr/bin/env python3
"""Complete fifteen fresh elastic-net sessions after the fixed completed resource pilot."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
from run_primary_elastic_net import validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 270
CHILD_EXTERNAL_SECONDS = 300


def validate_workers(workers, allocated_cpus):
    if (type(workers) is not int or type(allocated_cpus) is not int
            or not 1 <= workers <= 4 or workers > allocated_cpus):
        raise ValueError("One to four workers must fit the actual SLURM_CPUS_PER_TASK allocation")


def run_sessions(sessions, workers, run_one, *, started, max_seconds, stop_event,
                 clock=time.monotonic, on_complete=lambda session, item: None):
    """Fixed waves; every active child settles before another wave or return."""
    if type(workers) is not int or not 1 <= workers <= 4 or len(set(sessions)) != len(sessions):
        raise ValueError("Unique ordered sessions and at most four workers required")
    completed = {}
    positions = {session: index for index, session in enumerate(sessions)}

    class DispatchStopped(RuntimeError):
        pass

    def guarded_run(session):
        try:
            if stop_event.is_set():
                raise DispatchStopped("Panel dispatch already stopped")
            if max_seconds-(clock()-started) < CHILD_EXTERNAL_SECONDS:
                raise TimeoutError("Insufficient aggregate time for a complete 300s session reserve")
            return run_one(session)
        except BaseException:
            stop_event.set()
            raise

    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        try:
            for first in range(0, len(sessions), workers):
                if stop_event.is_set():
                    raise TimeoutError("Panel dispatch stopped after aggregate deadline")
                if max_seconds-(clock()-started) < CHILD_EXTERNAL_SECONDS:
                    raise TimeoutError("Insufficient aggregate time for a complete 300s session reserve")
                errors = []
                for session in sessions[first:first+workers]:
                    if stop_event.is_set():
                        break
                    active[executor.submit(guarded_run, session)] = session
                # A wave is a barrier: no replacement or next-wave work is
                # submitted while any of its children remain active.
                while active:
                    done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda f: positions[active[f]]):
                        session = active.pop(future)
                        try:
                            item = future.result()
                            completed[session] = item
                            on_complete(session, item)
                        except BaseException as error:
                            errors.append(error)
                            stop_event.set()
                if errors:
                    concrete = [error for error in errors if not isinstance(error, DispatchStopped)]
                    raise (concrete or errors)[0]
                if stop_event.is_set():
                    raise TimeoutError("Panel stopped after aggregate deadline; all children settled")
            if len(completed) != len(sessions):
                raise RuntimeError("Every required remaining session must complete")
        except BaseException:
            stop_event.set()
            for future in active:
                future.cancel()
            # The executor context joins running workers. Each worker uses a
            # bounded subprocess.run, which kills and reaps a timed-out child.
            raise
    return [completed[session] for session in sessions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root",
                 "reference-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True,
                        help="Aggregate internal bound chosen from pilot resource facts")
    parser.add_argument("--workers", type=int, required=True,
                        help="Selected simultaneous single-thread session workers, at most four")
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    try:
        allocated_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Actual SLURM_CPUS_PER_TASK must explicitly name the allocation") from error
    validate_workers(args.workers, allocated_cpus)
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
              "selected_workers": args.workers, "allocated_CPUs": allocated_cpus,
              "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
              "scientific_trial_count_increment": 0, "promotion_authorized": False,
              "search_complete": False}
    runner = Path(__file__).with_name("run_primary_elastic_net.py")
    stop_event = threading.Event()

    def deadline(*_):
        # Never unwind the executor asynchronously and strand active children.
        # Stop dispatch; existing per-child timeouts settle every subprocess.
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
            item = json.loads((args.bank_root/session/"score_banks.json").read_text())
            validate_session(item, session)
            return {"session_id": session,
                "elapsed_seconds": item["elapsed_seconds"],
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
            raise TimeoutError("Selected complete elastic-net-panel aggregate deadline reached")
        subprocess.run([sys.executable, str(runner), "summarize", "--bank-root", str(args.bank_root),
                        "--reference-bank-root", str(args.reference_bank_root),
                        "--reference-summary", str(args.reference_summary), "--out", str(args.summary_out)],
                       check=True, timeout=available)
        summary_path = args.summary_out/"primary_elastic_net.json"
        summary = json.loads(summary_path.read_text())
        if (summary["status"] != "complete_elastic_net_control_panel"
                or summary["scientific_trial_count_increment"] not in (0, 2)):
            raise ValueError("Complete two-panel summary or explicit structural duplicate required")
        report.update(status="complete_elastic_net_remaining_panel", summary_path=str(summary_path),
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
