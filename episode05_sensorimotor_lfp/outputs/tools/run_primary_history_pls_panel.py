#!/usr/bin/env python3
"""Run the fixed PLS2 pilot or a separately selected, explicitly bounded remainder."""
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
from run_primary_history_pls import PANEL_STATUS, RECIPES, SUMMARY_FILE, SUMMARY_OPTIONS, read_json, validate_session
from source_adapter import compute_only

PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 600
CHILD_EXTERNAL_SECONDS = 900
RESOURCE_KEYS = ("elapsed_seconds", "source_read_seconds", "candidate_and_control_fit_score_seconds",
                 "projection_fit_seconds", "ridge_head_seconds", "original_HW_replay_seconds",
                 "prediction_write_seconds", "maximum_RSS_KiB")


def session_command(args, session=PILOT):
    command = [sys.executable, str(Path(__file__).with_name("run_primary_history_pls.py")),
               "session", "--session", session, "--out", str(args.bank_root/session),
               "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
    for option in ("source", "trial_table", "scalar_bank_root", "hw_bank_root"):
        command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
    return command


def run_remainder(args):
    """Prepare no resource choices implicitly; settle each fixed wave before the next."""
    allocated = int(os.environ["SLURM_CPUS_PER_TASK"])
    validate_workers(args.workers, allocated)
    if args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Selected aggregate bound must exceed a full 900-second child reserve")
    pilot = read_json(args.bank_root/PILOT/"score_banks.json")
    validate_session(pilot, PILOT)
    sessions = [session for session in SESSIONS if session != PILOT]
    if len(sessions) != 15:
        raise ValueError("Exactly fifteen remaining original sessions are required")
    if args.summary_out.exists() or any((args.bank_root/session).exists() for session in sessions):
        raise FileExistsError("Fresh fifteen-session remainder and summary required; no automatic retries")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "mode": "remainder", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "selected_workers": args.workers, "allocated_CPUs": allocated,
        "pilot_session_reused": PILOT, "pilot_elapsed_seconds": pilot["elapsed_seconds"],
        "required_remaining_sessions": sessions, "completed_sessions": [],
        "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS,
        "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "scientific_trial_count_increment": 0, "scientific_effects_released": False,
        "promotion_authorized": False, "search_complete": False}
    stop_event = threading.Event()
    def deadline(*_):
        # Stop dispatch, without asynchronously stranding active bounded children.
        stop_event.set()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        def run_one(session):
            # The inherited generic dispatcher reserves 300 s. Enforce the
            # actual PLS child envelope here before every new source launch.
            if (stop_event.is_set()
                    or args.max_seconds-(time.monotonic()-started) < CHILD_EXTERNAL_SECONDS):
                raise TimeoutError("Insufficient aggregate time for the complete 900-second PLS child reserve")
            subprocess.run(session_command(args, session), check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = read_json(args.bank_root/session/"score_banks.json")
            validate_session(item, session)
            return {"session_id": session, **{key: item[key] for key in RESOURCE_KEYS},
                "complete_trials": item["source"]["complete_trials"],
                "physical_electrodes": len(item["source"]["common_LMP_electrodes"]),
                "all_original_HW_replays_passed": True}
        def completed(session, item):
            report["completed_sessions"].append(item)
            order = {value: index for index, value in enumerate(sessions)}
            report["completed_sessions"].sort(key=lambda row: order[row["session_id"]])
            print(json.dumps({"completed_session": session, "elapsed_seconds": item["elapsed_seconds"],
                "completed_count": len(report["completed_sessions"])}), flush=True)
        report["completed_sessions"] = run_sessions(sessions, args.workers, run_one,
            started=started, max_seconds=args.max_seconds, stop_event=stop_event, on_complete=completed)
        available = args.max_seconds-(time.monotonic()-started)
        if stop_event.is_set() or available <= 0:
            raise TimeoutError("Selected PLS aggregate deadline reached; all children settled")
        command = [sys.executable, str(Path(__file__).with_name("run_primary_history_pls.py")),
                   "summarize", "--out", str(args.summary_out)]
        for option in SUMMARY_OPTIONS:
            command.extend(["--"+option.replace("_", "-"), str(getattr(args, option))])
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/SUMMARY_FILE
        summary = read_json(path)
        if (summary["status"] != PANEL_STATUS or summary["scientific_trial_count_increment"] not in (0, 1, 2)
                or summary["promotion_authorized"] is not False
                or any(summary[key] is not True for key in ("all_new_saved_scores_replayed",
                    "all_original_anchor_scores_replayed", "original_selectors_independently_replayed"))):
            raise ValueError("Complete replayed sixteen-session PLS summary required")
        report.update(status="complete_history_pls_remaining_panel", summary_path=str(path),
            scientific_trial_count_increment=summary["scientific_trial_count_increment"],
            scientific_effects_released=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = ("source", "trial_table", "scalar_bank_root", "hw_bank_root", "bank_root", "out")
    for mode in ("pilot", "remainder"):
        child = sub.add_parser(mode)
        for name in common:
            child.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
        if mode == "remainder":
            for name in (*[option for option in SUMMARY_OPTIONS if option not in common], "summary_out"):
                child.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
            child.add_argument("--workers", type=int, required=True,
                help="Coordinator-selected single-thread workers after the measured pilot, at most four")
            child.add_argument("--max-seconds", type=int, required=True,
                help="Coordinator-selected aggregate internal deadline after the measured pilot")
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("One numerical thread per selected PLS2 worker required")
    if int(os.environ["SLURM_CPUS_PER_TASK"]) < 1:
        raise RuntimeError("At least one allocated CPU is required")
    if args.mode == "remainder":
        run_remainder(args)
        return
    if (args.bank_root/PILOT).exists():
        raise FileExistsError("Fresh fixed PLS2 pilot required; no automatic retry")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "mode": "pilot", "session_id": PILOT,
        "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "selected_workers": 1,
        "allocated_CPUs": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS,
        "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "scientific_trial_count_increment": 0, "scientific_effects_released": False,
        "promotion_authorized": False, "search_complete": False, "remainder_selected": False}
    try:
        subprocess.run(session_command(args), check=True, timeout=CHILD_EXTERNAL_SECONDS)
        item = read_json(args.bank_root/PILOT/"score_banks.json")
        validate_session(item, PILOT)
        metadata = item["banks"][RECIPES[0]]["direction_fit_metadata"]
        report.update(status="complete_history_pls_resource_pilot",
            session_resources={key: item[key] for key in RESOURCE_KEYS},
            complete_trials=item["source"]["complete_trials"],
            physical_electrodes=len(item["source"]["common_LMP_electrodes"]),
            all_original_HW_replays_passed=True,
            candidate_retained_ranks=[row["pls"]["candidate_transform"]["retained_components"] for row in metadata],
            control_retained_ranks=[[state["retained_components"] for state in row["pls"]["control_transforms"]]
                                    for row in metadata])
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.out/"pilot_resources.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
