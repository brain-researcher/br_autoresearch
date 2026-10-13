#!/usr/bin/env python3
"""One full LMP rank-one/Nyström session bank or a complete16-session summary."""
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

from capacity_lmp_reader import load_lmp_session
from primary_capacity import RECIPES, fit_banks, summarize_banks
from primary_reference import SESSIONS
from source_adapter import compute_only


def check_reference_after_fit(root, session, source, actual, baseline, labels, trial_ids):
    """Only called after every new model is fixed; does not return old scores."""
    directory = root/session
    report = json.loads((directory/"score_banks.json").read_text())
    if (report["status"] != "complete_three_session_banks" or report["session_id"] != session
            or report["source"]["common_LMP_electrodes"] != source["common_LMP_electrodes"]
            or report["source"]["complete_trials"] != source["complete_trials"]
            or report["source"]["native_offsets"] != [0, 14]):
        raise ValueError("Full-channel reference identity/opportunity changed")
    with np.load(directory/"lmp_full_channel_ridge_v1_predictions.npz", allow_pickle=False) as reference:
        for key, values in (("actual_xy", actual), ("baseline_xy", baseline),
                            ("directions", labels), ("trial_ids", trial_ids)):
            if not np.array_equal(reference[key], values):
                raise ArithmeticError(f"Exact reference {key} replay differs; no score release")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    session = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "out"):
        session.add_argument("--"+name, type=Path, required=True)
    session.add_argument("--session", choices=SESSIONS, required=True)
    session.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "reference-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Selected complete capacity pilot/session bound is<=270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete capacity session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, rows, source = load_lmp_session(args.source, args.trial_table, args.scalar_bank_root, args.session, check)
            ids = np.array([r["trial_id"] for r in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting_start = time.monotonic()
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check)
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting_start
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            for recipe in RECIPES:
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         candidate_xy_by_lambda=predictions[recipe], directions=labels, trial_ids=ids)
            report.update(status="complete_two_capacity_session_banks", banks=banks)
        else:
            sessions = {}
            for session_id in SESSIONS:
                item = json.loads((args.bank_root/session_id/"score_banks.json").read_text())
                if (item["status"] != "complete_two_capacity_session_banks" or item["session_id"] != session_id
                        or set(item["banks"]) != set(RECIPES) or item["exact_original_LMP_baseline_target_trial_replay"] is not True):
                    raise ValueError("Complete two-recipe banks and exact reference replay required in all16 sessions")
                sessions[session_id] = item
            panels = [summarize_banks({s: sessions[s]["banks"][r] for s in SESSIONS}, r) for r in RECIPES]
            reference = json.loads(args.reference_summary.read_text())
            anchor = [p for p in reference["panels"] if p["recipe"] == "lmp_full_channel_ridge_v1"]
            if (reference["status"] != "complete_three_multichannel_comparisons" or len(anchor) != 1
                    or {r["session_id"] for r in anchor[0]["sessions"]} != set(SESSIONS)):
                raise ValueError("Complete preserved global LMP anchor required")
            report.update(status="complete_two_capacity_comparisons", panels=panels,
                reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchor[0]},
                scientific_trial_count_increment=2, total_valid_primary_trials_including_prior=6,
                bank_root=str(args.bank_root))
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_capacity.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
