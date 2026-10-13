#!/usr/bin/env python3
"""Three adopted fixed feature banks; complete panels only for score release."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from multichannel_reader import load_session
from primary_multichannel import BLOCKS, fit_bank, summarize_banks
from primary_reference import SESSIONS
from source_adapter import compute_only


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    fit.add_argument("--out", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--bank-root", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Selected complete three-bank session must fit<=270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(BLOCKS), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected three-bank session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            blocks, y, labels, rows, source = load_session(args.source, args.trial_table, args.scalar_bank_root, args.session, check)
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started, banks={})
            common_baseline = None
            for recipe in BLOCKS:
                t = time.monotonic()
                bank, baseline, predictions = fit_bank(blocks[recipe], y, labels, recipe, check)
                if common_baseline is not None and not np.array_equal(common_baseline, baseline):
                    raise ArithmeticError("Shared baseline differs across unchanged trial/target panel")
                common_baseline = baseline
                bank["fit_and_score_seconds"] = time.monotonic()-t
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         candidate_xy_by_lambda=predictions, directions=labels,
                         trial_ids=np.asarray([r["trial_id"] for r in rows]))
                report["banks"][recipe] = bank
            report["status"] = "complete_three_session_banks"
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                if (item["status"] != "complete_three_session_banks" or item["session_id"] != session
                        or set(item["banks"]) != set(BLOCKS)):
                    raise ValueError("All three complete banks in every fixed session required")
                sessions[session] = item
            report["panels"] = [summarize_banks({s: sessions[s]["banks"][r] for s in SESSIONS}, r) for r in BLOCKS]
            report["source_scale_diagnostics"] = {s: sessions[s]["source"]["LMP_scale_diagnostic"] for s in SESSIONS}
            report.update(status="complete_three_multichannel_comparisons", scientific_trial_count_increment=3,
                          total_valid_primary_trials_including_scalar=4, bank_root=str(args.bank_root))
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        name = "score_banks.json" if args.mode == "session" else "primary_multichannel.json"
        (args.out/name).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
