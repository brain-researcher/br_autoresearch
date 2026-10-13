#!/usr/bin/env python3
"""Finish fifteen fresh reduced-W/saved-HW blend sessions after the fixed pilot."""
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
from run_primary_history_blend_intersection import (
    PATH_OPTIONS, PANEL_STATUS, REPLAY_FLAGS, SUMMARY_FILE, read_json, validate_session,
)
from source_adapter import compute_only


PILOT = "Chewie_20150313"
CHILD_INTERNAL_SECONDS = 270
CHILD_EXTERNAL_SECONDS = 300
COUNT_OPTIONS = ("prior_valid", "prior_patience", "prior_alternatives", "prior_nulls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PATH_OPTIONS, "source", "bank_root", "summary_out", "out"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    for name in (*COUNT_OPTIONS, "max_seconds"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1 or args.max_seconds <= CHILD_EXTERNAL_SECONDS:
        raise ValueError("Actual allocated CPU and finite selected aggregate bound exceeding a full300-second reserve required")
    if not 28 <= args.prior_valid <= 64 or min(getattr(args, key) for key in COUNT_OPTIONS[1:]) < 0:
        raise ValueError("Explicit nonnegative live accounting and original scientific depth required")
    metadata = read_json(args.metadata)
    if metadata["pilot_session"] != PILOT or Path(metadata["source"]).resolve() != args.source.resolve():
        raise ValueError("Preserve the fixed metadata-selected pilot and original read-only source")
    pilot = read_json(args.bank_root/PILOT/"score_banks.json")
    validate_session(pilot, PILOT)
    remaining = [session for session in SESSIONS if session != PILOT]
    if len(remaining) != 15 or args.summary_out.exists() or any((args.bank_root/session).exists() for session in remaining):
        raise ValueError("Preserved complete pilot, fifteen fresh session paths and a fresh summary required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pilot_session_reused": PILOT,
        "pilot_elapsed_seconds": pilot["elapsed_seconds"], "aggregate_internal_bound_seconds": args.max_seconds,
        "child_internal_bound_seconds": CHILD_INTERNAL_SECONDS, "child_external_bound_seconds": CHILD_EXTERNAL_SECONDS,
        "workers": 1, "required_remaining_sessions": remaining, "completed_remaining_sessions": [],
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "sensitivity_only": True,
        "incumbent_reset": False, "promotion_authorized": False, "search_complete": False,
        "frozen_live_accounting": {key: getattr(args, key) for key in COUNT_OPTIONS}}
    runner = Path(__file__).with_name("run_primary_history_blend_intersection.py")
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
                raise TimeoutError("Insufficient full300-second child reserve; current child is settled")
            command = [sys.executable, str(runner), "session", *common, "--source", str(args.source),
                "--session", session, "--out", str(args.bank_root/session), "--max-seconds", str(CHILD_INTERNAL_SECONDS)]
            # Sequential run waits for each child; timeout kills and waits for
            # that child before raising. The parent alarm only stops dispatch.
            subprocess.run(command, check=True, timeout=CHILD_EXTERNAL_SECONDS)
            item = read_json(args.bank_root/session/"score_banks.json")
            validate_session(item, session)
            details = {"session_id": session, **{key: item[key] for key in
                ("elapsed_seconds", "global_metadata_and_selector_seconds", "artifact_read_and_replay_seconds",
                 "source_read_seconds", "reduced_W_fit_seconds", "mixture_score_seconds", "maximum_RSS_KiB")}}
            report["completed_remaining_sessions"].append(details)
            print(json.dumps({**details, "completed_remaining_count": len(report["completed_remaining_sessions"])}), flush=True)
        available = args.max_seconds-(time.monotonic()-started)
        if stopped or available <= 0 or len(report["completed_remaining_sessions"]) != 15:
            raise TimeoutError("Complete reduced-support blend panel exceeded its selected finite bound")
        command = [sys.executable, str(runner), "summarize", *common,
                   "--bank-root", str(args.bank_root), "--out", str(args.summary_out)]
        for name in COUNT_OPTIONS:
            command.extend(["--"+name.replace("_", "-"), str(getattr(args, name))])
        subprocess.run(command, check=True, timeout=available)
        path = args.summary_out/SUMMARY_FILE
        summary = read_json(path)
        count = summary["scientific_trial_count_increment"]
        if (summary["status"] != PANEL_STATUS or type(count) is not int or count not in (0, 1, 2)
                or summary["literal_null_control_count_increment"] != 0
                or summary["alternative_explanation_count_increment"] != count
                or summary["sensitivity_only"] is not True
                or any(summary[key] is not False for key in ("incumbent_reset", "promotion_authorized",
                    "eligible_for_comparable_primary_incumbent", "original_HW_refitted", "full_W_refitted"))
                or any(summary[flag] is not True for flag in (*REPLAY_FLAGS, "all_new_saved_W_scores_replayed",
                    "all_selected_mixture_scores_replayed", "all_exact_duplicate_flags_replayed"))
                or set(summary["selected_prediction_paths"]) != set(SESSIONS)
                or summary["total_valid_primary_trials_including_prior"] != args.prior_valid+count
                or args.prior_valid+count > 64):
            raise ValueError("Complete independently charged I/R support stress and exact replays required")
        report.update(status="complete_history_blend_intersection_remaining_panel", summary_path=str(path),
            scientific_trial_count_increment=count, alternative_explanation_count_increment=count,
            scientific_trial_count_by_recipe=summary["scientific_trial_count_by_recipe"],
            structural_duplicate_of=summary["structural_duplicate_of"])
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
