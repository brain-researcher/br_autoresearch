#!/usr/bin/env python3
"""Complete native-target covariance-SVD/basis-control banks and full summary."""
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
from primary_covariance_svd import ANCHOR, RECIPES, REPLICATES, fit_banks, numeric_trial_ids, summarize_banks
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_two_covariance_svd_session_banks" or item["session_id"] != session
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or set(item["banks"]) != set(RECIPES) or item["source"]["native_offsets"] != [0, 14]):
        raise ValueError("Both complete native current-bin covariance_svd/control banks and exact replay required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or channels != sorted(set(channels)) or not channels
            or any(type(c) is not int or c < 1 for c in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or control["direction_counts"] != counts
            or candidate["physical_electrode_count"] != len(channels)
            or candidate["covariance_SVD_fits"] != 72 or candidate["nominal_component_cap"] != 2
            or candidate["whitened"] is not False or candidate["native_XY_scaled"] is not False
            or control["basis_label_replicate_ids"] != list(REPLICATES)
            or control["ridge_heads_use_true_targets"] is not True
            or control["realized_ranks_independent"] is not True):
        raise ValueError("Original physical/trial support and fixed native-target SVD/seed identities required")
    metadata = candidate["direction_fit_metadata"]
    if [r["held_direction"] for r in metadata] != list(range(8)):
        raise ValueError("All eight outer held-direction fits required")
    for row in metadata:
        held = row["held_direction"]
        allowed = sorted(set(range(8))-{held})
        fitting_rows = 15*(source["complete_trials"]-counts[held])
        if (row["fitting_bin_rows"] != fitting_rows
                or len(row["physical_mean"]) != len(channels) or len(row["physical_sd"]) != len(channels)
                or [r["replicate"] for r in row["basis_trial_maps_by_replicate"]] != list(REPLICATES)
                or len(row["projector_overlap_by_replicate"]) != len(REPLICATES)):
            raise ValueError("Complete physical scaling and eight fixed basis maps required")
        reference_ids = None
        for mapping in row["basis_trial_maps_by_replicate"]:
            rep, maps = mapping["replicate"], mapping["directions"]
            if [m["direction"] for m in maps] != allowed:
                raise ValueError("Only the seven fitting directions may enter a basis map")
            current_ids = []
            for m in maps:
                d, recipients, donors = m["direction"], m["recipient_trial_ids"], m["donor_trial_ids"]
                numbers = numeric_trial_ids(recipients, session)
                if (not np.array_equal(numbers, np.sort(numbers)) or len(recipients) != counts[d]
                        or sorted(donors) != recipients
                        or m["seed_literal"] != f"ep05-lmp-covariance-svd-basis-label-v1|20261004|{session}|{held}|{rep}|{d}"):
                    raise ValueError("Complete canonical original-ID within-direction basis maps required")
                current_ids.extend(recipients)
            if len(set(current_ids)) != len(current_ids) or (reference_ids is not None and current_ids != reference_ids):
                raise ValueError("Basis-map support must be identical in all eight replicates")
            reference_ids = current_ids
        bases = [row["paired_basis"], *row["control_bases_by_replicate"]]
        diagnostics = [row["paired_projected_covariance"], *row["control_projected_covariance_by_replicate"]]
        if len(bases) != 9 or len(diagnostics) != 9:
            raise ValueError("Candidate plus all eight independent bases and diagnostics required")
        for basis, diagnostic in zip(bases, diagnostics):
            k = basis["retained_components"]
            values = np.asarray(basis["singular_values"])
            tolerance = 64*np.finfo(np.float64).eps*max(len(channels), 2)*max(1., float(values[0]))
            if (not 0 <= k <= min(2, len(channels)) or values.shape != (min(2, len(channels)),)
                    or not np.isfinite(values).all() or np.any(values < 0)
                    or basis["singular_value_tolerance"] != tolerance or k != int(np.count_nonzero(values > tolerance))
                    or np.asarray(basis["input_basis"]).shape != (len(channels), k)
                    or np.asarray(basis["cross_covariance"]).shape != (len(channels), 2)
                    or basis["full_span"] != (k == len(channels))
                    or len(diagnostic["eigenvalues"]) != k or not 0 <= diagnostic["numerical_rank"] <= k
                    or [r["lambda"] for r in diagnostic["ridge_df_by_lambda"]] != list(LAMBDAS)):
                raise ValueError("Independent numerical-null ranks and complete projected diagnostics required")
    all_bases = [b for row in metadata for b in [row["paired_basis"], *row["control_bases_by_replicate"]]]
    if (candidate["all_arms_all_folds_full_span"] != all(b["full_span"] for b in all_bases)
            or candidate["all_arms_all_folds_zero_rank"] != all(b["retained_components"] == 0 for b in all_bases)):
        raise ValueError("Structural duplicate identity differs")


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
        raise ValueError("Complete covariance_svd/control session bound must be at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete covariance_svd/control session deadline reached")
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
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_covariance_svd_session_banks", banks=banks)
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
            report.update(status="complete_covariance_svd_control_panel", total_valid_primary_trials_including_prior=17+count,
                post_initial_comparisons=11+count, post_initial_controls=6+count//2,
                bank_root=str(args.bank_root), reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchors[0]})
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_covariance_svd.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
