#!/usr/bin/env python3
"""Run the complete native full-target sensitivity and paired training-label control."""
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
from primary_full_target_sensitivity import ANCHOR, RECIPES, REPLICATES, fit_banks, summarize_banks, trial_map
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_two_full_target_sensitivity_session_banks"
            or item["session_id"] != session or set(item["banks"]) != set(RECIPES)
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or item["sensitivity_only"] is not True or item["promotion_authorized"] is not False):
        raise ValueError("Complete nonpromotable sensitivity banks and exact original replay required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    ids, labels = np.asarray(candidate["fitting_trial_ids"]), np.asarray(candidate["fitting_trial_directions"])
    if (source["session_id"] != session or source["native_offsets"] != [0, 14] or source["native_bin_seconds"] != .03
            or not channels or channels != sorted(set(channels)) or any(type(e) is not int or e < 1 for e in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or control["direction_counts"] != counts
            or ids.shape != (source["complete_trials"],) or labels.shape != ids.shape
            or [int(np.count_nonzero(labels == d)) for d in range(8)] != counts
            or candidate["physical_electrode_ids"] != channels or candidate["physical_electrode_count"] != len(channels)
            or control["training_label_replicate_ids"] != list(REPLICATES)):
        raise ValueError("Original physical, trial and direction support required")
    for bank in (candidate, control):
        if bank["native_XY_scaled"] is not False or bank["baseline_added_to_prediction"] is not False or bank["sensitivity_only"] is not True or bank["promotion_authorized"] is not False:
            raise ValueError("Direct native XY with no added B and no primary promotion required")
    for key in ("training_targets_permuted_as_whole_trajectories", "within_direction_only",
                "physical_Gram_shared_exactly", "true_training_mu_shared_exactly", "evaluation_unpermuted"):
        if control[key] is not True:
            raise ValueError("Fixed complete-trajectory training null required: "+key)
    metadata = candidate["direction_fit_metadata"]
    if [row["held_direction"] for row in metadata] != list(range(8)):
        raise ValueError("Eight complete original held-direction fits required")
    for row in metadata:
        held, width = row["held_direction"], len(channels)
        train = labels != held
        if row["fitting_bin_rows"] != 15*int(np.count_nonzero(train)) or type(row["baseline_equals_mu"]) is not bool:
            raise ValueError("Original fitting support and exact constant-B flag required")
        for key, shape in (("physical_mean", (width,)), ("physical_sd", (width,)), ("mu", (2,)),
                           ("physical_Gram", (width, width)), ("paired_cross", (width, 2)),
                           ("control_cross_by_replicate", (8, width, 2))):
            array = np.asarray(row[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("Finite complete fitting state and own target cross-products required")
        sd, gram = np.asarray(row["physical_sd"]), np.asarray(row["physical_Gram"])
        covariance = row["physical_covariance"]
        if (np.any(sd < 0) or row["physical_zero_SD_columns"] != np.flatnonzero(sd == 0).tolist()
                or np.any(np.diag(gram) < 0) or not np.allclose(gram, gram.T, rtol=1e-13, atol=1e-13)
                or len(covariance["eigenvalues"]) != width or not np.isfinite(covariance["eigenvalues"]).all()
                or [d["lambda"] for d in covariance["ridge_df_by_lambda"]] != list(LAMBDAS)
                or len(row["controls_equal_candidate_by_replicate"]) != 8):
            raise ValueError("Original feature normalization, Gram and complete diagnostics required")
        expected = [{"replicate": r, "directions": trial_map(ids[train], labels[train], session, held, r)[1]} for r in REPLICATES]
        if row["training_target_maps"] != expected:
            raise ValueError("Exact native-ID fitting-only whole-trajectory maps required")
        equal = [bool(np.array_equal(cross, row["paired_cross"])) for cross in row["control_cross_by_replicate"]]
        if row["controls_equal_candidate_by_replicate"] != equal:
            raise ValueError("Control duplication requires exact target cross-product equality")
    same = all(all(row["controls_equal_candidate_by_replicate"]) for row in metadata)
    if (candidate["structural_residual_anchor_duplicate"] != all(row["baseline_equals_mu"] for row in metadata)
            or candidate["all_controls_equal_candidate"] != same or control["structural_paired_duplicate"] != same):
        raise ValueError("Separate paired/control duplicate accounting must match every fitted arm")


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
        raise RuntimeError("Explicit one-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted complete full-target session bound is at most 270 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "sensitivity_only": True,
        "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete full-target sensitivity session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, rows, source = load_lmp_session(args.source, args.trial_table, args.scalar_bank_root, args.session, check)
            ids = np.array([row["trial_id"] for row in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting = time.monotonic()
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check, electrode_ids=source["common_LMP_electrodes"])
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            check()
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_full_target_sensitivity_session_banks", banks=banks)
            validate_session(report, args.session)
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            reference = json.loads(args.reference_summary.read_text())
            panels = [p for p in reference["panels"] if p["recipe"] == ANCHOR]
            if reference["status"] != "complete_three_multichannel_comparisons" or len(panels) != 1:
                raise ValueError("Complete preserved residual-LMP reference required")
            anchor_banks = {}
            for session in SESSIONS:
                old = json.loads((args.reference_bank_root/session/"score_banks.json").read_text())
                if old["status"] != "complete_three_session_banks" or old["session_id"] != session:
                    raise ValueError("Complete original residual-LMP banks required")
                anchor_banks[session] = old["banks"][ANCHOR]
            report.update(summarize_banks(sessions, anchor_banks, panels[0]))
            count = report["scientific_trial_count_increment"]
            report.update(status="complete_full_target_sensitivity_control_panel", total_valid_primary_trials_including_prior=23+count,
                post_initial_comparisons=17+count, post_initial_controls=9+report["control_trial_count_increment"],
                bank_root=str(args.bank_root), reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": panels[0]})
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_full_target_sensitivity.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
