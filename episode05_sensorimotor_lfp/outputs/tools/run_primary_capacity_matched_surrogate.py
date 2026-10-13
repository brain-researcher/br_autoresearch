#!/usr/bin/env python3
"""Fit only fixed HW training surrogates, then replay saved full-original HW."""
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

from history_lmp_reader import load_lmp_history_session
from primary_capacity_matched_surrogate import ANCHOR, RECIPE, REPLICATES, fit_banks, summarize_banks, trial_map
from primary_covariance_svd import numeric_trial_ids
from primary_reference import LAMBDAS, SESSIONS
from run_primary_elapsed_only import check_anchor_after_fit
from run_primary_history_repeatability_weight import validate_session as validate_hw_session
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_capacity_matched_surrogate_session_bank"
            or item["session_id"] != session or item["original_HW_refitted"] is not False
            or item["exact_original_HW_neural_state_replay"] is not True
            or item["exact_original_HW_saved_arrays_and_scoring_replay"] is not True
            or item["exact_original_HW_weighted_covariance_replay"] is not True
            or item["promotion_authorized"] is not False):
        raise ValueError("Complete surrogate bank and exact original HW replays required")
    bank, source = item["bank"], item["source"]
    labels, ids = np.asarray(bank["original_direction_labels"]), np.asarray(bank["original_trial_ids"])
    numeric_trial_ids(ids, session)
    counts, channels = bank["direction_counts"], source["common_LMP_electrodes"]
    if (bank["recipe"] != RECIPE or bank["lambdas"] != list(LAMBDAS)
            or bank["surrogate_replicate_ids"] != list(REPLICATES)
            or source["session_id"] != session or source["native_bin_seconds"] != .03
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != [0, -1, -2, -3] or bank["lag_offsets"] != [0, -1, -2, -3]
            or not channels or channels != sorted(set(channels))
            or any(type(c) is not int or c < 1 for c in channels)
            or bank["physical_electrode_ids"] != channels or bank["physical_electrode_count"] != len(channels)
            or bank["history_column_count"] != 4*len(channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or len(ids) != source["complete_trials"]
            or labels.shape != ids.shape or set(labels.tolist()) != set(range(8))
            or counts != [int(np.count_nonzero(labels == d)) for d in range(8)]
            or bank["native_XY_scaled"] is not False or bank["weights_use_velocity"] is not False
            or bank["positive_floor"] != .1 or bank["post_weight_standardization"] is not False
            or bank["post_weight_centering"] is not False or bank["original_HW_refitted"] is not False
            or bank["shared_Gram_eigendecompositions"] != 8
            or bank["surrogate_specific_Gram_eigendecompositions"] != 0
            or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0):
        raise ValueError("Unchanged full-support HW and exactly shared surrogate capacity required")
    seeds = bank["surrogate_seed_rows"]
    if [row["replicate"] for row in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed seed banks required")
    for rows in (bank["lambda_rows"], bank["original_HW_saved_prediction_rows"],
                 *(seed["lambda_rows"] for seed in seeds)):
        if ([row["lambda"] for row in rows] != list(LAMBDAS)
                or not all(np.isfinite(row[k]) for row in rows for k in
                    ("candidate_SSE", "candidate_R2", "wrong_pair_SSE", "delta_R2", "identity_increment"))):
            raise ValueError("Complete finite five-penalty score rows required")
    for index in range(len(LAMBDAS)):
        for key in ("candidate_SSE", "wrong_pair_SSE"):
            if bank["lambda_rows"][index][key] != float(np.mean([seed["lambda_rows"][index][key] for seed in seeds])):
                raise ValueError("Surrogate must average all eight SSEs, not predictions")
    metadata = bank["direction_fit_metadata"]
    if [row["held_direction"] for row in metadata] != list(range(8)):
        raise ValueError("All eight original held-direction fits required")
    for row in metadata:
        held, capacity = row["held_direction"], row["capacity_identity"]
        train = labels != held
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        if (row["fitting_bin_rows"] != 15*int(np.count_nonzero(train))
                or row["surrogate_trial_maps"] != [records for _, records in maps]
                or row["repeatability"]["fitting_directions"] != sorted(set(range(8))-{held})
                or row["repeatability"]["fitting_direction_counts"] != [counts[d] for d in range(8) if d != held]
                or capacity["fitting_design_rows"] != row["fitting_bin_rows"]
                or capacity["feature_columns"] != 4*len(channels)
                or capacity["shared_Gram_eigendecompositions"] != 1
                or capacity["surrogate_specific_Gram_eigendecompositions"] != 0
                or capacity["original_HW_refitted"] is not False):
            raise ValueError("Exact predetermined whole-trial maps and shared fitting Gram required")
        for key in ("surrogate_cross_product_equals_original_by_replicate",
                    "surrogate_design_equals_original_by_replicate", "whole_fitting_map_identity_by_replicate"):
            if len(capacity[key]) != 8 or any(type(value) is not bool for value in capacity[key]):
                raise ValueError("All eight exact map/design/cross-product identity flags required")
        if capacity["whole_fitting_map_identity_by_replicate"] != [
                bool(np.array_equal(mapping, np.arange(len(mapping)))) for mapping, _ in maps]:
            raise ValueError("Whole-map identities must agree with the retained fixed maps")
        covariance = row["paired_weighted_covariance"]
        if ([r["lambda"] for r in covariance["ridge_df_by_lambda"]] != list(LAMBDAS)
                or len(covariance["eigenvalues"]) != 4*len(channels)
                or not np.isfinite(covariance["eigenvalues"]).all()
                or not all(np.isfinite(r["effective_df"]) for r in covariance["ridge_df_by_lambda"])):
            raise ValueError("Original complete shared capacity diagnostics required")
    duplicate = all(all(row["capacity_identity"]["surrogate_cross_product_equals_original_by_replicate"])
                    for row in metadata)
    if type(bank["structural_original_HW_duplicate"]) is not bool or bank["structural_original_HW_duplicate"] != duplicate:
        raise ValueError("Duplicate accounting must cover every seed and direction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "hw-bank-root", "hw-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted session internal cap is at most 270 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False,
        "promotion_authorized": False, "search_complete": False, "new_promotion_gate": False}
    def deadline(*_):
        raise TimeoutError("Adopted capacity-matched surrogate session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, rows, source = load_lmp_history_session(args.source, args.trial_table,
                args.scalar_bank_root, args.session, check)
            ids = np.asarray([row["trial_id"] for row in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting = time.monotonic()
            bank, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check,
                electrode_ids=source["common_LMP_electrodes"])
            report["surrogate_fit_and_score_seconds"] = time.monotonic()-fitting
            check_anchor_after_fit(args.hw_bank_root, args.session, source, y, labels, ids, baseline, bank)
            previous = json.loads((args.hw_bank_root/args.session/"score_banks.json").read_text())["banks"][ANCHOR]
            for new, old in zip(bank["direction_fit_metadata"], previous["direction_fit_metadata"]):
                if new["paired_weighted_covariance"] != old["paired_weighted_covariance"]:
                    raise ValueError("Original weighted-HW capacity diagnostics did not replay exactly")
            report.update(exact_original_HW_neural_state_replay=True,
                exact_original_HW_saved_arrays_and_scoring_replay=True,
                exact_original_HW_weighted_covariance_replay=True)
            check()
            np.savez(args.out/(RECIPE+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                directions=labels, trial_ids=ids, candidate_xy_by_replicate_lambda=predictions)
            report.update(status="complete_capacity_matched_surrogate_session_bank", bank=bank)
            validate_session(report, args.session)
        else:
            sessions, anchors = {}, {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                old = json.loads((args.hw_bank_root/session/"score_banks.json").read_text())
                validate_hw_session(old, session)
                sessions[session], anchors[session] = item["bank"], old["banks"][ANCHOR]
            reference = json.loads(args.hw_summary.read_text())
            panels = [p for p in reference["panels"] if p["recipe"] == ANCHOR]
            if reference["status"] != "complete_history_repeatability_weight_control_panel" or len(panels) != 1:
                raise ValueError("Complete preserved original HW selection report required")
            report.update(summarize_banks(sessions, anchors, panels[0]), status="complete_capacity_matched_surrogate_panel",
                bank_root=str(args.bank_root), original_HW_anchor={"bank_root": str(args.hw_bank_root),
                    "summary": str(args.hw_summary), "refitted": False},
                all_original_HW_neural_state_replay=True,
                all_original_HW_saved_arrays_and_scoring_replay=True,
                all_original_HW_weighted_covariance_replay=True)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_capacity_matched_surrogate.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
