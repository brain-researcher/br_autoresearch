#!/usr/bin/env python3
"""Complete EP05 temporal/control session banks or the fixed sixteen-session summary."""
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
from primary_reference import SESSIONS
from primary_temporal import RECIPES, aggregate_rows, fit_banks, summarize_banks
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    session = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "out"):
        session.add_argument("--"+name, type=Path, required=True)
    session.add_argument("--session", choices=SESSIONS, required=True)
    session.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "reference-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Selected complete temporal/control session bound is <=270 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete temporal/control session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, rows, source = load_lmp_session(args.source, args.trial_table,
                                                       args.scalar_bank_root, args.session, check)
            ids = np.array([r["trial_id"] for r in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting_start = time.monotonic()
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check)
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting_start
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_temporal_control_session_banks", banks=banks)
        else:
            sessions = {}
            for session_id in SESSIONS:
                item = json.loads((args.bank_root/session_id/"score_banks.json").read_text())
                if (item["status"] != "complete_two_temporal_control_session_banks"
                        or item["session_id"] != session_id or set(item["banks"]) != set(RECIPES)
                        or item["exact_original_LMP_baseline_target_trial_replay"] is not True):
                    raise ValueError("Both complete banks and exact replay required in all sixteen sessions")
                sessions[session_id] = item["banks"]
            panels, control = summarize_banks(sessions)
            reference = json.loads(args.reference_summary.read_text())
            anchors = [p for p in reference["panels"] if p["recipe"] == "lmp_full_channel_ridge_v1"]
            if (reference["status"] != "complete_three_multichannel_comparisons" or len(anchors) != 1
                    or {r["session_id"] for r in anchors[0]["sessions"]} != set(SESSIONS)):
                raise ValueError("Complete preserved global stationary LMP anchor required")
            anchor = anchors[0]
            anchor_rows = {r["session_id"]: r for r in anchor["sessions"]}
            contrasts = []
            for panel in panels:
                rows = [{"session_id": r["session_id"], "animal": r["animal"],
                         "delta_R2": r["delta_R2"]-anchor_rows[r["session_id"]]["delta_R2"],
                         "identity_increment": r["identity_increment"]-anchor_rows[r["session_id"]]["identity_increment"]}
                        for r in panel["sessions"]]
                contrasts.append({"recipe": panel["recipe"], **aggregate_rows(rows)})
            report.update(status="complete_two_temporal_control_comparisons", panels=panels,
                          paired_training_control=control,
                          reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchor},
                          contrasts_against_stationary_anchor=contrasts,
                          scientific_trial_count_increment=2, total_valid_primary_trials_including_prior=8,
                          post_initial_comparisons=2, post_initial_controls=1,
                          bank_root=str(args.bank_root))
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_temporal_control.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
