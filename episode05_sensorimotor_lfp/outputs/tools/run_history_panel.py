#!/usr/bin/env python3
"""Finite fixed-history/channel-label panel; preserve every completed or failed attempt."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from primary_history import LAGS, RECIPES, REPLICATES
from primary_reference import LAMBDAS, SESSIONS
from source_adapter import compute_only


def check_channel_maps(paired, control, source):
    """Validate all trial/replicate electrode maps without choosing on scores."""
    complete = source.get("complete_trials")
    channels = source.get("common_LMP_electrodes", [])
    if (type(complete) is not int or complete < 16 or not channels
            or channels != sorted(set(channels)) or any(type(c) is not int or c < 1 for c in channels)):
        raise ValueError("Complete original trials and unique physical LMP electrodes required")
    metadata = paired.get("direction_fit_metadata", [])
    counts = paired.get("direction_counts", [])
    if (len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts) or sum(counts) != complete
            or counts != control.get("direction_counts") or len(metadata) != 8
            or [item.get("held_direction") for item in metadata] != list(range(8))
            or any(item.get("fitting_bin_rows") != 15*(complete-counts[item["held_direction"]])
                   for item in metadata)):
        raise ValueError("Every complete held-direction split must preserve original trial support")
    maps = paired.get("channel_label_trial_maps", {})
    ids = maps.get("trial_ids", [])
    assignments = maps.get("source_channel_positions_by_replicate", [])
    if (len(ids) != complete or len(set(map(str, ids))) != complete
            or maps.get("replicate_ids") != list(REPLICATES) or len(assignments) != 8
            or control.get("channel_label_replicate_ids") != list(REPLICATES)
            or control.get("channel_label_Monte_Carlo_draws") != 8):
        raise ValueError("Every original trial needs all eight fixed label-control maps")
    positions = list(range(len(channels)))
    for replicate in assignments:
        if len(replicate) != complete:
            raise ValueError("Incomplete original trial map set")
        for assignment in replicate:
            if len(assignment) != len(channels) or any(type(p) is not int for p in assignment) or sorted(assignment) != positions:
                raise ValueError("Each trial requires a complete physical-electrode permutation")
    seed_rows = control.get("channel_label_seed_rows", [])
    if (len(seed_rows) != 8 or [row.get("replicate") for row in seed_rows] != list(REPLICATES)
            or any([r.get("lambda") for r in row.get("lambda_rows", [])] != list(LAMBDAS) for row in seed_rows)):
        raise ValueError("Complete five-penalty score rows for each fixed label-control seed required")


def completed_bank(directory, session):
    path = directory/"score_banks.json"
    if not path.is_file():
        raise ValueError(f"Existing incomplete directory is preserved, never retried: {directory}")
    report = json.loads(path.read_text())
    source = report.get("source", {})
    if (report.get("status") != "complete_two_history_control_session_banks"
            or report.get("session_id") != session or source.get("session_id") != session
            or set(report.get("banks", {})) != set(RECIPES)
            or source.get("native_LMP_offsets") != [-3, 14]
            or source.get("native_target_offsets") != [0, 14]
            or source.get("lag_offsets") != list(LAGS) or source.get("native_bin_seconds") != .03
            or report.get("exact_original_LMP_baseline_target_trial_replay") is not True
            or report.get("distinct_history_input_and_primary_target_windows_verified") is not True):
        raise ValueError(f"Existing failed/partial/mismatched history bank is preserved: {directory}")
    for recipe in RECIPES:
        bank = report["banks"][recipe]
        if (bank.get("recipe") != recipe or bank.get("lambdas") != list(LAMBDAS)
                or bank.get("lag_offsets") != list(LAGS)
                or [row.get("lambda") for row in bank.get("lambda_rows", [])] != list(LAMBDAS)
                or not (directory/(recipe+"_predictions.npz")).is_file()):
            raise ValueError(f"Existing recipe bank/predictions incomplete: {directory}")
    check_channel_maps(report["banks"][RECIPES[0]], report["banks"][RECIPES[1]], source)
    return {"session_id": session, "report": str(path), "elapsed_seconds": report["elapsed_seconds"],
            "complete_trials": source["complete_trials"], "complete_channel_label_replicates": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root",
                 "reference-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.max_seconds <= 0 or args.out.exists() or args.summary_out.exists():
        raise ValueError("Positive coordinator-selected bound and fresh aggregate/summary paths required")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    # Completion/support inspection only; every complete bank is reused
    # regardless of its score. Reject partial directories before new work.
    prior = {s: completed_bank(args.bank_root/s, s) for s in SESSIONS if (args.bank_root/s).exists()}
    if "Chewie_20150313" not in prior:
        raise ValueError("The selected complete representative pilot must precede scaling")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"job_id": os.environ["SLURM_JOB_ID"], "started_utc": started_utc,
              "max_seconds": args.max_seconds, "session_internal_seconds": 270,
              "session_external_seconds": 300, "sessions": [], "status": "running",
              "scientific_trial_count_increment": 0, "automatic_retries": 0,
              "reference_summary": str(args.reference_summary)}
    runner = Path(__file__).with_name("run_primary_history.py")
    def save():
        report["elapsed_seconds"] = time.monotonic()-started
        (args.out/"panel_execution.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    try:
        for session in SESSIONS:
            if session in prior:
                report["sessions"].append({**prior[session], "reused": True})
                save()
                continue
            remaining = args.max_seconds-(time.monotonic()-started)
            if remaining < 300:
                raise TimeoutError("Insufficient aggregate time for another unchanged 270-second session")
            command = [sys.executable, str(runner), "session", "--source", str(args.source),
                       "--trial-table", str(args.trial_table), "--scalar-bank-root", str(args.scalar_bank_root),
                       "--reference-bank-root", str(args.reference_bank_root), "--session", session,
                       "--out", str(args.bank_root/session), "--max-seconds", "270"]
            report["current_session"] = session
            save()
            result = subprocess.run(command, check=False, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Session {session} exited {result.returncode}; no retry or deletion")
            report["sessions"].append({**completed_bank(args.bank_root/session, session), "reused": False})
            save()
        remaining = args.max_seconds-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError("Aggregate deadline reached before complete-panel summary")
        result = subprocess.run([sys.executable, str(runner), "summarize", "--bank-root", str(args.bank_root),
                                 "--reference-summary", str(args.reference_summary), "--out", str(args.summary_out)],
                                check=False, timeout=min(60, remaining))
        if result.returncode != 0:
            raise RuntimeError("Complete-panel summary failed; all history banks preserved")
        report.update(status="complete_fixed_history_control_panel", scientific_trial_count_increment=2,
                      summary=str(args.summary_out/"primary_history_control.json"))
        report.pop("current_session", None)
    except Exception as error:
        report.update(status="incomplete_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
