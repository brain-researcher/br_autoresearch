#!/usr/bin/env python3
"""Complete PCA90/loading-label session banks and finite full-panel summary."""
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
from primary_pca import ANCHOR, RECIPES, REPLICATES, fit_banks, summarize_banks
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_two_pca_session_banks" or item["session_id"] != session
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or set(item["banks"]) != set(RECIPES) or item["source"]["native_offsets"] != [0, 14]):
        raise ValueError("Both complete native current-bin PCA/control banks and exact replay required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or channels != sorted(set(channels)) or not channels
            or len(counts) != 8 or min(counts) < 2 or sum(counts) != source["complete_trials"]
            or control["direction_counts"] != counts or candidate["physical_electrode_count"] != len(channels)
            or candidate["retained_variance_target"] != .9 or candidate["whitened"] is not False
            or [r["replicate"] for r in candidate["loading_maps"]] != list(REPLICATES)
            or control["loading_label_replicate_ids"] != list(REPLICATES)):
        raise ValueError("Original physical/trial support and fixed PCA/seed identities required")
    for mapping in candidate["loading_maps"]:
        positions, r = mapping["source_channel_positions"], mapping["replicate"]
        if (sorted(positions) != list(range(len(channels)))
                or mapping["seed_literal"] != f"ep05-lmp-pca90-loading-label-v1|20261004|{session}|{r}"
                or mapping["source_channel_ids"] != [channels[i] for i in positions]):
            raise ValueError("Complete frozen physical loading-label maps required")
    metadata = candidate["direction_fit_metadata"]
    if [r["held_direction"] for r in metadata] != list(range(8)):
        raise ValueError("All eight held-direction PCA fits required")
    for row in metadata:
        k = row["pca"]["retained_components"]
        diagnostics = [row["paired_projected_covariance"], *row["control_projected_covariance_by_replicate"]]
        if (not 1 <= k <= len(channels) or row["fitting_bin_rows"] != 15*(source["complete_trials"]-counts[row["held_direction"]])
                or len(diagnostics) != 9 or row["pca"]["full_span"] != (k == len(channels))
                or any(len(d["eigenvalues"]) != k or not 0 <= d["numerical_rank"] <= k
                       or [r["lambda"] for r in d["ridge_df_by_lambda"]] != list(LAMBDAS) for d in diagnostics)):
            raise ValueError("Complete candidate/control rank-energy diagnostics required; zero-rank controls remain valid")
    if candidate["all_folds_full_span"] != all(r["pca"]["full_span"] for r in metadata):
        raise ValueError("Structural full-span identity differs")


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
        raise RuntimeError("Explicit single-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Complete PCA/control session bound must be at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete PCA/control session deadline reached")
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
            fitting = time.monotonic()
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check)
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting
            for mapping in banks[RECIPES[0]]["loading_maps"]:
                mapping["source_channel_ids"] = [source["common_LMP_electrodes"][i] for i in mapping["source_channel_positions"]]
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_pca_session_banks", banks=banks)
            validate_session(report, args.session)
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            reference = json.loads(args.reference_summary.read_text())
            anchors = [p for p in reference["panels"] if p["recipe"] == ANCHOR]
            if reference["status"] != "complete_three_multichannel_comparisons" or len(anchors) != 1:
                raise ValueError("Complete preserved multichannel reference required")
            anchor_banks = {}
            for session in SESSIONS:
                old = json.loads((args.reference_bank_root/session/"score_banks.json").read_text())
                if old["status"] != "complete_three_session_banks" or old["session_id"] != session:
                    raise ValueError("Complete preserved original LMP banks required")
                anchor_banks[session] = old["banks"][ANCHOR]
            report.update(summarize_banks(sessions, anchor_banks, anchors[0]))
            count = report["scientific_trial_count_increment"]
            report.update(status="complete_pca_control_panel", total_valid_primary_trials_including_prior=13+count,
                post_initial_comparisons=7+count, post_initial_controls=4+count//2,
                bank_root=str(args.bank_root), reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchors[0]})
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_pca.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
