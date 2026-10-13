#!/usr/bin/env python3
"""One fixed EP05 session bank; root must first accept the complete benchmark."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from primary_reference import RECIPE, SESSIONS, fit_bank, summarize_banks
from source_adapter import M1_SESSIONS, M1Session, compute_only


def trial_records(path, session_id):
    with path.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if {r["session_id"] for r in all_rows} != set(SESSIONS):
        raise ValueError("The fixed sixteen-session qualified trial table is required")
    rows = [row for row in all_rows if row["session_id"] == session_id]
    keys = [int(r["source_trial_index_zero_based"]) for r in rows]
    if not rows or len(keys) != len(set(keys)) or keys != sorted(keys):
        raise ValueError("Original unique source trial order required")
    if any(row["publication_execution_support"] != "True" or row["result"] != "R" for row in rows):
        raise ValueError("Every frozen qualified whole trial must be supported")
    return rows


def load_session(source, table, session_id, check):
    specification = next(item for item in M1_SESSIONS if f"{item[0]}_{item[2]}" == session_id)
    rows = trial_records(table, session_id)
    channels, available, metadata = None, [], []
    read_bytes = 0
    with M1Session(source, *specification) as session:
        # All-electrode guide/identity metadata, no LFP/velocity values yet.
        for row in rows:
            check()
            trial = int(row["source_trial_index_zero_based"])
            meta = session.metadata(trial)
            if (meta["trial_id"] != row["trial_id"] or meta["direction"] != int(row["direction"])
                    or meta["direction_circular_error"] > .01 or meta["result"] != "R"
                    or meta["idx_movement_on"] != int(row["idx_movement_on"])):
                raise ValueError("Frozen source identity/direction/event changed")
            guide = np.asarray(session.field(trial, "M1_lfp_guide")).T
            if guide.ndim != 2 or guide.shape[1] != 3 or not np.isfinite(guide).all():
                raise ValueError("Finite physical LMP guide required")
            indices = np.flatnonzero((guide[:, 1] == 0) & (guide[:, 2] == 0))
            identities = guide[indices, 0]
            if (len(indices) == 0 or len(np.unique(identities)) != len(indices)
                    or np.any(identities < 1) or np.any(identities != np.floor(identities))):
                raise ValueError("Unique positive integral LMP electrode IDs required")
            mapping = {int(c): int(index) for c, index in zip(identities, indices)}
            channels = set(mapping) if channels is None else channels & set(mapping)
            available.append(mapping)
            metadata.append(meta)
        if not channels:
            raise ValueError("No common metadata-qualified LMP electrodes")
        channel_ids = sorted(channels)
        x, y = [], []
        for row, meta, mapping in zip(rows, metadata, available):
            check()
            trial = int(row["source_trial_index_zero_based"])
            start, end = meta["idx_movement_on"], meta["idx_movement_on"]+15
            lfp, velocity = session.field(trial, "M1_lfp"), session.field(trial, "vel")
            selected = sorted(mapping[c] for c in channel_ids)
            if (lfp.ndim != 2 or velocity.shape != (2, lfp.shape[1])
                    or start < 0 or end > lfp.shape[1] or max(selected) >= lfp.shape[0]):
                raise ValueError("Fixed LMP/velocity window is unavailable")
            # Only selected LMP channels and the fixed execution velocity bins.
            values, target = lfp[selected, start:end], velocity[:, start:end]
            if not np.isfinite(values).all() or not np.isfinite(target).all():
                raise ValueError("Nonfinite whole trial; no post-score exclusions")
            read_bytes += int(values.nbytes+target.nbytes)
            x.append(values.astype(np.float64).mean(axis=0)[:, None])
            y.append(target.astype(np.float64).T)
    return np.stack(x), np.stack(y), np.array([m["direction"] for m in metadata]), rows, {
        "animal": specification[0], "implant": specification[1], "session_id": session_id,
        "complete_trials": len(rows), "common_LMP_electrodes": channel_ids,
        "guide_intersection_removed_per_trial": [len(m)-len(channels) for m in available],
        "native_offsets": [0, 14], "native_bin_seconds": .03,
        "selected_logical_source_bytes": read_bytes,
        "source_fields_with_values_read": ["M1_lfp:LMP_only", "vel:XY"],
        "HDF_chunk_boundary": "Storage chunks may contain other elements; only selected LMP/velocity values returned to Python"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    fit.add_argument("--source", type=Path, required=True)
    fit.add_argument("--trial-table", type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    fit.add_argument("--out", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--bank-root", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numeric settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Selected session bound must be <=270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_trial_count_increment": 0, "promotion_authorized": False}
    def check(*_):
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            raise TimeoutError("Selected whole-session reference bound reached")
    def deadline(*_):
        raise TimeoutError("Selected whole-session reference bound reached")
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, rows, source = load_session(args.source, args.trial_table, args.session, check)
            report.update(session_id=args.session, source=source,
                          source_read_seconds=time.monotonic()-started)
            bank, baseline, predictions = fit_bank(x, y, labels, check)
            report.update(bank)
            np.savez(args.out/"predictions.npz", actual_xy=y, baseline_xy=baseline,
                     candidate_xy_by_lambda=predictions, directions=labels,
                     trial_ids=np.asarray([r["trial_id"] for r in rows]))
            report["status"] = "complete_session_score_bank"
        else:
            banks = {}
            for session_id in SESSIONS:
                path = args.bank_root/session_id/"score_bank.json"
                bank = json.loads(path.read_text())
                if bank["status"] != "complete_session_score_bank" or bank["session_id"] != session_id:
                    raise ValueError("All fixed completed session banks required")
                banks[session_id] = bank
            report.update(summarize_banks(banks))
            report["bank_root"] = str(args.bank_root)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        name = "score_bank.json" if args.mode == "session" else "primary_reference.json"
        (args.out/name).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    # A partial session bank does not print scientific scores or choose a winner.
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "scientific_trial_count_increment": report["scientific_trial_count_increment"]}), flush=True)


if __name__ == "__main__":
    main()
