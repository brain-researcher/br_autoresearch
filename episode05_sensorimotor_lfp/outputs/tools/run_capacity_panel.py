#!/usr/bin/env python3
"""Finite EP05 two-capacity panel; preserve completed banks, never retry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from primary_capacity import RECIPES
from primary_reference import LAMBDAS, SESSIONS
from source_adapter import compute_only


def completed_bank(directory, session):
    path = directory / "score_banks.json"
    if not path.is_file():
        raise ValueError(f"Existing incomplete directory is preserved, never retried: {directory}")
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_two_capacity_session_banks"
            or report.get("session_id") != session
            or set(report.get("banks", {})) != set(RECIPES)
            or report.get("source", {}).get("session_id") != session
            or report.get("exact_original_LMP_baseline_target_trial_replay") is not True):
        raise ValueError(f"Existing failed/partial/mismatched bank is preserved: {directory}")
    for recipe in RECIPES:
        bank = report["banks"][recipe]
        if (bank.get("recipe") != recipe or bank.get("lambdas") != list(LAMBDAS)
                or not (directory / (recipe+"_predictions.npz")).is_file()):
            raise ValueError(f"Existing recipe bank/predictions incomplete: {directory}")
    return {"session_id": session, "report": str(path), "elapsed_seconds": report["elapsed_seconds"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root",
                 "reference-summary", "bank-root", "summary-out", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    args = p.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.max_seconds <= 0 or args.out.exists() or args.summary_out.exists():
        raise ValueError("Positive coordinator-selected deadline and fresh aggregate/summary paths required")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    # Status/identity only: a completed pilot is reused regardless of score.
    # Reject every existing partial attempt before starting a missing session.
    prior = {s: completed_bank(args.bank_root/s, s) for s in SESSIONS if (args.bank_root/s).exists()}
    if "Chewie_20150313" not in prior:
        raise ValueError("The selected complete representative pilot must precede scaling")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"job_id": os.environ["SLURM_JOB_ID"], "started_utc": started_utc,
              "max_seconds": args.max_seconds, "sessions": [], "status": "running",
              "scientific_trial_count_increment": 0, "automatic_retries": 0,
              "reference_summary": str(args.reference_summary)}
    runner = Path(__file__).with_name("run_primary_capacity.py")
    def save():
        report["elapsed_seconds"] = time.monotonic()-started
        (args.out/"panel_execution.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        for session in SESSIONS:
            if session in prior:
                report["sessions"].append({**prior[session], "reused": True})
                save()
                continue
            # Preserve the selected full270-second session envelope. A short
            # remaining allocation never changes scientific work or trials.
            remaining = args.max_seconds-(time.monotonic()-started)
            if remaining < 300:
                raise TimeoutError("Insufficient aggregate time for another unchanged270s session")
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
            raise RuntimeError("Complete-panel summary failed; all session banks preserved")
        report.update(status="complete_fixed_capacity_panel", scientific_trial_count_increment=2,
                      summary=str(args.summary_out/"primary_capacity.json"))
        report.pop("current_session", None)
    except Exception as error:
        report.update(status="incomplete_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
