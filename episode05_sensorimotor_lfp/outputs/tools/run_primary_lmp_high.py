#!/usr/bin/env python3
"""Complete fixed LMP/high joint-control banks; release only full panels."""
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
from primary_lmp_high import ANCHOR, RECIPES, REPLICATES, fit_banks, summarize_banks
from primary_reference import SESSIONS
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_two_lmp_high_session_banks" or item["session_id"] != session
            or set(item["banks"]) != set(RECIPES)
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or item["source"]["native_offsets"] != [0, 14]
            or item["source"]["native_bin_seconds"] != .03):
        raise ValueError("Both complete joint/control banks and original native target replay required")
    source, joint, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    if (joint["feature_width"] != 3*len(source["common_LMP_electrodes"])
            or joint["high_bands_hz"] != [[100, 200], [200, 400]]
            or control["high_trial_replicate_ids"] != list(REPLICATES)
            or [r["replicate"] for r in joint["high_trial_maps"]] != list(REPLICATES)
            or len(joint["direction_fit_metadata"]) != 8):
        raise ValueError("Exact joint coordinates, eight held fits and eight high-trial maps required")
    all_ids = None
    for replicate in joint["high_trial_maps"]:
        records = replicate["directions"]
        if [r["direction"] for r in records] != list(range(8)):
            raise ValueError("All eight direction maps required")
        ids = []
        for record in records:
            recipient, donor = record["recipient_trial_ids"], record["donor_trial_ids"]
            d, r = record["direction"], replicate["replicate"]
            literal = f"ep05-lmp-high-trial-label-v1|20261003|{session}|{d}|{r}"
            if (record["replicate"] != r or record["seed_literal"] != literal
                    or not 1 <= record["draw_count"] <= 10000 or recipient != sorted(set(recipient))
                    or sorted(donor) != recipient or any(a == b for a, b in zip(recipient, donor))
                    or len(recipient) != joint["direction_counts"][d]):
                raise ValueError("Original high-trial derangement identity/count/seed differs")
            ids.extend(recipient)
        if len(ids) != source["complete_trials"] or len(set(ids)) != len(ids):
            raise ValueError("All original complete trials required in every map")
        if all_ids is not None and sorted(ids) != all_ids:
            raise ValueError("Replicate trial identities differ")
        all_ids = sorted(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "reference-bank-root", "reference-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Complete joint-control session bound must be at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete LMP/high session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            blocks, y, labels, rows, source = load_session(args.source, args.trial_table,
                                                          args.scalar_bank_root, args.session, check)
            lmp, high = blocks[ANCHOR], blocks["high_full_channel_ridge_v1"]
            del blocks  # Discard the qualified reader's low-band block without fitting/scoring.
            source["low_block_discarded_without_fit_or_score"] = True
            ids = np.array([r["trial_id"] for r in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fit_started = time.monotonic()
            banks, baseline, predictions = fit_banks(lmp, high, y, labels, ids, args.session, check)
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fit_started
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_lmp_high_session_banks", banks=banks)
            validate_session(report, args.session)
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            # All new session fits must be complete before these reference-score reads.
            reference = json.loads(args.reference_summary.read_text())
            anchors = [p for p in reference["panels"] if p["recipe"] == ANCHOR]
            if reference["status"] != "complete_three_multichannel_comparisons" or len(anchors) != 1:
                raise ValueError("Complete preserved multichannel summary required")
            anchor_banks = {}
            for session in SESSIONS:
                old = json.loads((args.reference_bank_root/session/"score_banks.json").read_text())
                if old["status"] != "complete_three_session_banks" or old["session_id"] != session:
                    raise ValueError("Preserved five-lambda LMP banks required")
                anchor_banks[session] = old["banks"][ANCHOR]
            report.update(summarize_banks(sessions, anchor_banks, anchors[0]))
            report.update(status="complete_two_lmp_high_comparisons", scientific_trial_count_increment=2,
                total_valid_primary_trials_including_prior=12, post_initial_comparisons=6,
                post_initial_controls=3, bank_root=str(args.bank_root),
                reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchors[0]},
                interpretation="Aligned added high-block information; not isolated cross-block interaction")
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_lmp_high.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
