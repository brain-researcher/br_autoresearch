#!/usr/bin/env python3
"""Finite remaining-fifteen PCA90 panel following the complete selected pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from primary_pca import RECIPES, REPLICATES
from primary_reference import LAMBDAS, SESSIONS
from run_primary_pca import validate_session
from source_adapter import compute_only


def completed_bank(directory, session):
    path = directory/"score_banks.json"
    if not path.is_file():
        raise ValueError(f"Existing incomplete directory is preserved, never retried: {directory}")
    report = json.loads(path.read_text())
    validate_session(report, session)
    source = report["source"]
    candidate, control = (report["banks"][r] for r in RECIPES)
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    if (any(type(c) is not int or c < 1 for c in channels)
            or any(type(n) is not int or n < 2 for n in counts)):
        raise ValueError("Original positive physical IDs and complete trial counts required")
    for recipe in RECIPES:
        bank = report["banks"][recipe]
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                or not (directory/(recipe+"_predictions.npz")).is_file()):
            raise ValueError("All fixed score/prediction banks must be complete")
    seed_rows = control["loading_label_seed_rows"]
    if ([r["replicate"] for r in seed_rows] != list(REPLICATES)
            or control["loading_label_Monte_Carlo_draws"] != 8
            or any([r["lambda"] for r in seed["lambda_rows"]] != list(LAMBDAS) for seed in seed_rows)):
        raise ValueError("Eight complete seed-level five-lambda scores required")
    return {"session_id": session, "report": str(path), "elapsed_seconds": report["elapsed_seconds"],
            "complete_trials": source["complete_trials"], "complete_loading_label_replicates": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root",
                 "reference-summary", "bank-root", "summary-out", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.max_seconds <= 0 or args.out.exists() or args.summary_out.exists():
        raise ValueError("Positive root-selected bound and fresh execution/summary paths required")
    started = time.monotonic()
    prior = {s: completed_bank(args.bank_root/s, s) for s in SESSIONS if (args.bank_root/s).exists()}
    if set(prior) != {"Chewie_20150313"}:
        raise ValueError("Exactly the selected complete pilot must precede this fresh remaining-fifteen wrapper")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"job_id": os.environ["SLURM_JOB_ID"], "started_utc": datetime.now(timezone.utc).isoformat(),
        "max_seconds": args.max_seconds, "session_internal_seconds": 270, "session_external_seconds": 300,
        "sessions": [], "status": "running", "scientific_trial_count_increment": 0,
        "automatic_retries": 0, "reference_summary": str(args.reference_summary)}
    runner = Path(__file__).with_name("run_primary_pca.py")
    def save():
        report["elapsed_seconds"] = time.monotonic()-started
        (args.out/"panel_execution.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    try:
        for session in SESSIONS:
            if session in prior:
                report["sessions"].append({**prior[session], "reused": True})
                save()
                continue
            if args.max_seconds-(time.monotonic()-started) < 300:
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
            "--reference-bank-root", str(args.reference_bank_root), "--reference-summary", str(args.reference_summary),
            "--out", str(args.summary_out)], check=False, timeout=min(60, remaining))
        if result.returncode != 0:
            raise RuntimeError("Complete-panel summary failed; all PCA/control banks preserved")
        summary_path = args.summary_out/"primary_pca.json"
        summary = json.loads(summary_path.read_text())
        count = summary["scientific_trial_count_increment"]
        if (summary["status"] != "complete_pca_control_panel"
                or count != (0 if summary["structural_full_span_duplicate"] else 2)):
            raise ValueError("Complete-panel structural-duplicate accounting must be preserved")
        report.update(status="complete_fixed_pca_panel", scientific_trial_count_increment=count,
                      structural_full_span_duplicate=summary["structural_full_span_duplicate"],
                      summary=str(summary_path))
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
