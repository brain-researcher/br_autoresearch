#!/usr/bin/env python3
"""Complete repeatability-weight/physical-label-control banks and full summary."""
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
from primary_repeatability_weight import ANCHOR, RECIPES, REPLICATES, fit_banks, numeric_trial_ids, summarize_banks
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_two_repeatability_weight_session_banks" or item["session_id"] != session
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or set(item["banks"]) != set(RECIPES) or item["source"]["native_offsets"] != [0, 14]):
        raise ValueError("Both complete native current-bin weight/control banks and exact replay required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or channels != sorted(set(channels)) or not channels
            or any(type(c) is not int or c < 1 for c in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or control["direction_counts"] != counts
            or candidate["physical_electrode_count"] != len(channels) or candidate["physical_electrode_ids"] != channels
            or candidate["native_XY_scaled"] is not False or candidate["weights_use_velocity"] is not False
            or candidate["positive_floor"] != .1 or candidate["post_weight_standardization"] is not False
            or control["weight_label_replicate_ids"] != list(REPLICATES)
            or control["controls_have_own_Gram"] is not True or control["true_native_targets"] is not True):
        raise ValueError("Original physical/trial support and fixed repeatability-weight identities required")
    maps = candidate["weight_label_maps"]
    if [r["replicate"] for r in maps] != list(REPLICATES):
        raise ValueError("All eight fixed physical label maps required")
    for mapping in maps:
        rep = mapping["replicate"]
        if (mapping["recipient_electrode_ids"] != channels or sorted(mapping["donor_electrode_ids"]) != channels
                or mapping["seed_literal"] != f"ep05-lmp-repeatability-weight-label-v1|20261004|{session}|{rep}"):
            raise ValueError("Original physical electrode identities and fixed map seed required")
    metadata = candidate["direction_fit_metadata"]
    if [r["held_direction"] for r in metadata] != list(range(8)):
        raise ValueError("All eight outer held-direction fits required")
    for row in metadata:
        held = row["held_direction"]
        allowed = sorted(set(range(8))-{held})
        statistics = row["repeatability"]
        if (row["fitting_bin_rows"] != 15*(source["complete_trials"]-counts[held])
                or statistics["fitting_directions"] != allowed
                or statistics["fitting_direction_counts"] != [counts[d] for d in allowed]
                or len(row["physical_mean"]) != len(channels) or len(row["physical_sd"]) != len(channels)
                or len(row["controls_equal_candidate_by_replicate"]) != 8):
            raise ValueError("Complete fitting-only support and normalization required")
        for key in ("T", "P", "r", "a"):
            if len(statistics[key]) != len(channels) or not np.isfinite(statistics[key]).all():
                raise ValueError("Finite full-channel repeatability statistics required")
        t, p, r, a = (np.asarray(statistics[key]) for key in ("T", "P", "r", "a"))
        if np.any(t < 0) or np.any(r < 0) or np.any(r > 1) or not np.array_equal(a, .1+r):
            raise ValueError("Fixed clipped repeatability ratio and positive floor required")
        raw = statistics["raw_ratio"]
        if len(raw) != len(channels) or len(statistics["clip_category"]) != len(channels):
            raise ValueError("Unclipped signed-ratio diagnostics required")
        for e, value in enumerate(raw):
            if t[e] == 0:
                if value is not None or r[e] != 0:
                    raise ValueError("Zero-energy ratio must remain undefined with applied r=0")
            elif value is None or not np.isfinite(value) or value != p[e]/t[e] or r[e] != np.clip(value, 0, 1):
                raise ValueError("Exact recorded signed ratio and clipping required")
        weights = [row["paired_weights"], *row["control_weights_by_replicate"]]
        covariances = [row["paired_weighted_covariance"], *row["control_weighted_covariance_by_replicate"]]
        if len(weights) != 9 or len(covariances) != 9:
            raise ValueError("Candidate and eight distinct weighted Grams required")
        active = row["paired_weights"]["positive_energy_columns"]
        for arm, (weight, covariance) in enumerate(zip(weights, covariances)):
            if (weight["positive_energy_columns"] != active or type(weight["structural_identity"]) is not bool
                    or len(weight["weights"]) != len(channels) or not np.isfinite(weight["weights"]).all()
                    or np.any(np.asarray(weight["weights"]) <= 0)
                    or weight["original_trace"] != weights[0]["original_trace"]
                    or len(covariance["eigenvalues"]) != len(channels)
                    or [d["lambda"] for d in covariance["ridge_df_by_lambda"]] != list(LAMBDAS)):
                raise ValueError("Full positive applied weights and own covariance diagnostics required")
            expected_a = a if arm == 0 else a[[channels.index(donor) for donor in maps[arm-1]["donor_electrode_ids"]]]
            if not np.array_equal(weight["base_weights"], expected_a):
                raise ValueError("Each control must assign the fixed physical labels before trace normalization")
            identity = not active or bool(np.all(expected_a[active] == expected_a[active][0]))
            if weight["structural_identity"] != identity or (identity and weight["weights"] != [1.]*len(channels)):
                raise ValueError("Independent exact active-column structural identity required")
            if arm:
                equal = np.array_equal(np.asarray(weight["weights"])[active], np.asarray(weights[0]["weights"])[active])
                if row["controls_equal_candidate_by_replicate"][arm-1] != equal:
                    raise ValueError("Control/candidate identity must use exact applied active weights")
    if (candidate["structural_identity"] != all(d["paired_weights"]["structural_identity"] for d in metadata)
            or control["structural_identity"] != all(w["structural_identity"] for d in metadata for w in d["control_weights_by_replicate"])
            or candidate["all_controls_equal_candidate"] != all(all(d["controls_equal_candidate_by_replicate"]) for d in metadata)):
        raise ValueError("Separate candidate/control duplicate accounting identities differ")


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
        raise ValueError("Complete repeatability_weight/control session bound must be at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete repeatability_weight/control session deadline reached")
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
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check, electrode_ids=source["common_LMP_electrodes"])
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            report["exact_original_LMP_baseline_target_trial_replay"] = True
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_repeatability_weight_session_banks", banks=banks)
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
            report.update(status="complete_repeatability_weight_control_panel", total_valid_primary_trials_including_prior=19+count,
                post_initial_comparisons=13+count, post_initial_controls=7+report["control_trial_count_increment"],
                bank_root=str(args.bank_root), reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchors[0]})
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_repeatability_weight.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
