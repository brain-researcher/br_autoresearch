#!/usr/bin/env python3
"""Finite EP05 temporal/control panel; reuse complete banks and never retry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from primary_reference import LAMBDAS, SESSIONS
from primary_temporal import RECIPES, REPLICATES
from source_adapter import compute_only


def check_trial_maps(paired, control, complete_trials):
    """Check preserved original-ID pairing support without selecting on scores."""
    metadata = paired.get("direction_fit_metadata", [])
    if (len(metadata) != 8 or [m.get("held_direction") for m in metadata] != list(range(8))
            or control.get("training_shuffle_replicate_ids") != list(REPLICATES)
            or control.get("training_shuffle_Monte_Carlo_draws") != len(REPLICATES)):
        raise ValueError("Eight complete held-direction maps and eight fixed control replicates required")
    identities = {}
    for split in metadata:
        ids = split.get("target_trial_ids_in_fitting_order", [])
        labels = split.get("target_directions_in_fitting_order", [])
        held = split["held_direction"]
        if (len(ids) != len(labels) or len(set(ids)) != len(ids)
                or set(labels) != set(range(8))-{held}
                or split.get("fitting_bin_rows") != 15*len(ids)):
            raise ValueError("Incomplete or repeated fitting trial identities")
        for trial_id, direction in zip(ids, labels):
            if trial_id in identities and identities[trial_id] != direction:
                raise ValueError("Original trial direction changed between held-direction banks")
            identities[trial_id] = direction
        maps = split.get("training_feature_source_trial_ids_by_replicate", [])
        if len(maps) != 8 or [m.get("replicate") for m in maps] != list(REPLICATES):
            raise ValueError("Every held direction must retain all eight original-ID derangements")
        direction_by_id = dict(zip(ids, labels))
        for mapping in maps:
            sources = mapping.get("source_trial_ids", [])
            if (len(sources) != len(ids) or set(sources) != set(ids)
                    or any(a == b or direction_by_id[a] != direction_by_id[b]
                           for a, b in zip(ids, sources))):
                raise ValueError("Training map must be a complete within-direction whole-trial derangement")
    counts = [sum(d == direction for d in identities.values()) for direction in range(8)]
    if (len(identities) != complete_trials or counts != paired.get("direction_counts")
            or counts != control.get("direction_counts") or any(n < 2 for n in counts)):
        raise ValueError("Complete original session trial/direction support required")
    for split in metadata:
        expected = {trial_id for trial_id, direction in identities.items()
                    if direction != split["held_direction"]}
        if set(split["target_trial_ids_in_fitting_order"]) != expected:
            raise ValueError("Held-direction fitting map omits an original eligible trial")
    seed_rows = control.get("training_shuffle_seed_rows", [])
    if (len(seed_rows) != 8 or [r.get("replicate") for r in seed_rows] != list(REPLICATES)
            or any([row.get("lambda") for row in seed.get("lambda_rows", [])] != list(LAMBDAS)
                   for seed in seed_rows)):
        raise ValueError("Eight complete five-penalty control score rows required")


def completed_bank(directory, session):
    path = directory/"score_banks.json"
    if not path.is_file():
        raise ValueError(f"Existing incomplete directory is preserved, never retried: {directory}")
    report = json.loads(path.read_text())
    source = report.get("source", {})
    if (report.get("status") != "complete_two_temporal_control_session_banks"
            or report.get("session_id") != session or set(report.get("banks", {})) != set(RECIPES)
            or source.get("session_id") != session or source.get("native_offsets") != [0, 14]
            or source.get("native_bin_seconds") != .03
            or report.get("exact_original_LMP_baseline_target_trial_replay") is not True):
        raise ValueError(f"Existing failed/partial/mismatched bank is preserved: {directory}")
    for recipe in RECIPES:
        bank = report["banks"][recipe]
        if (bank.get("recipe") != recipe or bank.get("lambdas") != list(LAMBDAS)
                or [r.get("lambda") for r in bank.get("lambda_rows", [])] != list(LAMBDAS)
                or not (directory/(recipe+"_predictions.npz")).is_file()):
            raise ValueError(f"Existing recipe bank/predictions incomplete: {directory}")
    check_trial_maps(report["banks"][RECIPES[0]], report["banks"][RECIPES[1]], source.get("complete_trials"))
    return {"session_id": session, "report": str(path), "elapsed_seconds": report["elapsed_seconds"],
            "complete_trials": source["complete_trials"], "complete_training_control_replicates": 8}


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
        raise ValueError("Positive coordinator-selected deadline and fresh aggregate/summary paths required")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    # Read only to validate completion/identities; no score-dependent branch.
    # Stop before launching any missing session if an existing attempt is partial.
    prior = {s: completed_bank(args.bank_root/s, s) for s in SESSIONS if (args.bank_root/s).exists()}
    if "Chewie_20150313" not in prior:
        raise ValueError("The selected complete representative pilot must precede scaling")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"job_id": os.environ["SLURM_JOB_ID"], "started_utc": started_utc,
              "max_seconds": args.max_seconds, "session_internal_seconds": 270,
              "session_external_seconds": 300, "sessions": [], "status": "running",
              "scientific_trial_count_increment": 0, "automatic_retries": 0,
              "reference_summary": str(args.reference_summary)}
    runner = Path(__file__).with_name("run_primary_temporal.py")
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
            raise RuntimeError("Complete-panel summary failed; all session banks preserved")
        report.update(status="complete_fixed_temporal_control_panel", scientific_trial_count_increment=2,
                      summary=str(args.summary_out/"primary_temporal_control.json"))
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
