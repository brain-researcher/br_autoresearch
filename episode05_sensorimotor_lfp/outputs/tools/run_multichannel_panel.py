#!/usr/bin/env python3
"""Sequential fixed sixteen-session panel; reuse complete banks, never retry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from primary_multichannel import BLOCKS
from primary_reference import LAMBDAS, SESSIONS
from source_adapter import compute_only


def completed_bank(directory, session):
    path = directory / "score_banks.json"
    if not path.is_file():
        raise ValueError(f"Existing incomplete directory is preserved, never retried: {directory}")
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_three_session_banks" or report.get("session_id") != session
            or set(report.get("banks", {})) != set(BLOCKS)
            or report.get("source", {}).get("session_id") != session):
        raise ValueError(f"Existing failed/partial/mismatched bank is preserved: {directory}")
    for recipe in BLOCKS:
        bank = report["banks"][recipe]
        if (bank.get("recipe") != recipe or bank.get("lambdas") != list(LAMBDAS)
                or not (directory / (recipe+"_predictions.npz")).is_file()):
            raise ValueError(f"Existing recipe bank/predictions incomplete: {directory}")
    return {"session_id": session, "report": str(path), "elapsed_seconds": report["elapsed_seconds"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "bank-root", "summary-out", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    args = p.parse_args()
    compute_only()
    if args.max_seconds <= 0 or args.out.exists() or args.summary_out.exists():
        raise ValueError("Positive coordinator-selected deadline and fresh aggregate/summary paths required")
    # Read status/identity only. Reject every existing partial attempt before
    # starting any missing session; do not inspect scientific score values.
    prior = {s: completed_bank(args.bank_root/s, s) for s in SESSIONS if (args.bank_root/s).exists()}
    if "Chewie_20150313" not in prior:
        raise ValueError("The selected complete representative pilot must precede scaling")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"job_id": os.environ["SLURM_JOB_ID"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "max_seconds": args.max_seconds, "sessions": [], "status": "running",
              "scientific_trial_count_increment": 0, "automatic_retries": 0}
    runner = Path(__file__).with_name("run_primary_multichannel.py")
    def save():
        report["elapsed_seconds"] = time.monotonic()-started
        (args.out/"panel_execution.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        for session in SESSIONS:
            if session in prior:
                report["sessions"].append({**prior[session], "reused": True})
                save()
                continue
            # Preserve the full per-session scientific/cost envelope; do not
            # shrink it when the aggregate allocation is nearly exhausted.
            remaining = args.max_seconds-(time.monotonic()-started)
            if remaining < 300:
                raise TimeoutError("Insufficient aggregate time for another unchanged270s session")
            command = [sys.executable, str(runner), "session", "--source", str(args.source),
                       "--trial-table", str(args.trial_table), "--scalar-bank-root", str(args.scalar_bank_root),
                       "--session", session, "--out", str(args.bank_root/session), "--max-seconds", "270"]
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
                                 "--out", str(args.summary_out)], check=False, timeout=min(60, remaining))
        if result.returncode != 0:
            raise RuntimeError("Complete-panel summary failed; all session banks are preserved")
        report.update(status="complete_fixed_multichannel_panel", scientific_trial_count_increment=3,
                      summary=str(args.summary_out/"primary_multichannel.json"))
    except Exception as error:
        report.update(status="incomplete_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
