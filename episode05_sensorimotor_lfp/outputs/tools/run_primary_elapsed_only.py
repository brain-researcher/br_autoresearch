#!/usr/bin/env python3
"""Fit the adopted elapsed-only HW sensitivity; replay original saved HW anchors."""
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
from primary_capacity import score_bank
from primary_elapsed_only import ANCHOR, RECIPE, fit_banks, summarize_banks
from primary_reference import LAMBDAS, SESSIONS
from run_primary_history_repeatability_weight import validate_session as validate_hw_session
from source_adapter import compute_only

STATE_KEYS = ("held_direction", "fitting_bin_rows", "physical_mean", "physical_sd",
    "physical_zero_SD_columns", "history_physical_mean", "history_physical_sd",
    "history_zero_SD_coordinates", "history_fitting_column_mean", "repeatability", "paired_weights")


def check_anchor_after_fit(root, session, source, y, labels, ids, original_baseline, bank):
    old = json.loads((root/session/"score_banks.json").read_text())
    validate_hw_session(old, session)
    anchor = old["banks"][ANCHOR]
    if (source["common_LMP_electrodes"] != old["source"]["common_LMP_electrodes"]
            or source["complete_trials"] != old["source"]["complete_trials"]
            or bank["physical_electrode_ids"] != anchor["physical_electrode_ids"]):
        raise ValueError("Exact full-original HW physical/trial support required")
    for new, previous in zip(bank["direction_fit_metadata"], anchor["direction_fit_metadata"]):
        for key in STATE_KEYS:
            if new[key] != previous[key]:
                raise ValueError(f"Original HW neural state did not replay: {key}")
    path = root/session/(ANCHOR+"_predictions.npz")
    with np.load(path, allow_pickle=False) as saved:
        for key, actual in (("actual_xy", y), ("directions", labels), ("trial_ids", ids),
                            ("baseline_xy", original_baseline)):
            if not np.array_equal(saved[key], actual):
                raise ValueError(f"Original saved full-support HW arrays did not replay: {key}")
        predictions = saved["candidate_xy_by_lambda"]
        if predictions.shape != (len(LAMBDAS), *y.shape) or not np.isfinite(predictions).all():
            raise ValueError("All five complete saved original HW prediction arrays required")
        replay = score_bank(y, labels, original_baseline, predictions, ANCHOR)
    for key in ("lambdas", "lambda_rows", "baseline_SSE", "baseline_R2", "denominator", "direction_counts"):
        if replay[key] != anchor[key]:
            raise ValueError(f"Saved original HW full prediction scoring did not replay: {key}")
    if bank["denominator"] != anchor["denominator"] or bank["direction_counts"] != anchor["direction_counts"]:
        raise ValueError("Original full native-XY denominator and direction support required")
    bank.update(original_baseline_SSE=anchor["baseline_SSE"], original_baseline_R2=anchor["baseline_R2"],
        original_HW_saved_prediction_rows=anchor["lambda_rows"], original_HW_predictions_path=str(path))


def validate_session(item, session):
    if (item["status"] != "complete_elapsed_only_HW_session_bank" or item["session_id"] != session
            or item["exact_original_HW_neural_state_replay"] is not True
            or item["exact_original_HW_saved_arrays_and_scoring_replay"] is not True
            or item["original_HW_refitted"] is not False or item["sensitivity_only"] is not True
            or item["promotion_authorized"] is not False):
        raise ValueError("Complete nonpromotable elapsed-only bank and original HW replays required")
    bank, source = item["bank"], item["source"]
    counts, channels = bank["direction_counts"], source["common_LMP_electrodes"]
    if (bank["recipe"] != RECIPE or bank["lambdas"] != list(LAMBDAS)
            or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
            or source["session_id"] != session or source["native_bin_seconds"] != .03
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != [0, -1, -2, -3] or bank["lag_offsets"] != [0, -1, -2, -3]
            or not channels or channels != sorted(set(channels))
            or any(type(c) is not int or c < 1 for c in channels)
            or bank["physical_electrode_ids"] != channels or bank["physical_electrode_count"] != len(channels)
            or bank["history_column_count"] != 4*len(channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"]
            or bank["native_XY_scaled"] is not False or bank["weights_use_velocity"] is not False
            or bank["elapsed_harmonics"] != 0 or bank["positive_floor"] != .1
            or bank["baseline_direction_weighting"] != "equal seven fitting directions"
            or bank["post_weight_standardization"] is not False or bank["post_weight_centering"] is not False
            or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0):
        raise ValueError("Original support, fixed elapsed model and unchanged HW feature recipe required")
    for rows in (bank["lambda_rows"], bank["original_HW_saved_prediction_rows"]):
        if ([r["lambda"] for r in rows] != list(LAMBDAS)
                or not all(np.isfinite(r[k]) for r in rows for k in
                    ("candidate_SSE", "candidate_R2", "wrong_pair_SSE", "delta_R2", "identity_increment"))):
            raise ValueError("Complete finite five-penalty sensitivity and saved HW rows required")
    metadata = bank["direction_fit_metadata"]
    if [r["held_direction"] for r in metadata] != list(range(8)):
        raise ValueError("All eight held-direction fits required")
    for row in metadata:
        held = row["held_direction"]
        if (row["fitting_bin_rows"] != 15*(source["complete_trials"]-counts[held])
                or row["repeatability"]["fitting_directions"] != sorted(set(range(8))-{held})
                or row["repeatability"]["fitting_direction_counts"] != [counts[d] for d in range(8) if d != held]):
            raise ValueError("Exactly seven original fitting directions required")
        for key, shape in (("elapsed_template_coefficients", (1, 15, 2)),
                           ("elapsed_common_residual_intercept", (2,)),
                           ("original_common_residual_intercept", (2,))):
            value = np.asarray(row[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Finite complete elapsed template and common intercept required")
        for key in ("exact_elapsed_equals_original_baseline_on_fitting_and_evaluation",
                    "exact_zero_weighted_design_response_cross_product"):
            if type(row[key]) is not bool:
                raise ValueError("Exact duplicate flags must be explicit booleans")
    for key, detail in (("structural_original_HW_duplicate",
                         "exact_elapsed_equals_original_baseline_on_fitting_and_evaluation"),
                        ("structural_time_only_duplicate", "exact_zero_weighted_design_response_cross_product")):
        if type(bank[key]) is not bool or bank[key] != all(r[detail] for r in metadata):
            raise ValueError("Duplicate accounting must cover every original direction fold")


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
        raise RuntimeError("Explicit single-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted complete single-session internal bound is at most 270 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "alternative_explanation_stress_count_increment": 0,
        "null_control_trial_count_increment": 0, "sensitivity_only": True,
        "promotion_authorized": False, "search_complete": False, "original_HW_refitted": False}
    def deadline(*_):
        raise TimeoutError("Adopted elapsed-only HW session deadline reached")
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
            ids = np.array([r["trial_id"] for r in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting = time.monotonic()
            bank, baseline, original_baseline, predictions = fit_banks(x, y, labels, ids, args.session,
                check, electrode_ids=source["common_LMP_electrodes"])
            report["single_recipe_fit_and_score_seconds"] = time.monotonic()-fitting
            check_anchor_after_fit(args.hw_bank_root, args.session, source, y, labels, ids, original_baseline, bank)
            report.update(exact_original_HW_neural_state_replay=True,
                exact_original_HW_saved_arrays_and_scoring_replay=True)
            check()
            np.savez(args.out/(RECIPE+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                original_baseline_xy=original_baseline, directions=labels, trial_ids=ids,
                candidate_xy_by_lambda=predictions)
            report.update(status="complete_elapsed_only_HW_session_bank", bank=bank)
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
                raise ValueError("Complete original HW summary required")
            report.update(summarize_banks(sessions, anchors, panels[0]), status="complete_elapsed_only_HW_panel",
                bank_root=str(args.bank_root), original_HW_anchor={"bank_root": str(args.hw_bank_root),
                    "summary": str(args.hw_summary), "refitted": False},
                all_original_HW_neural_state_replay=True,
                all_original_HW_saved_arrays_and_scoring_replay=True)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_elapsed_only.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
