#!/usr/bin/env python3
"""Complete native-XY elastic-net/training-control banks with a C99 single-sweep kernel and full summary."""
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
from primary_elastic_net_native import ANCHOR, BACKEND, RECIPES, REPLICATES, fit_banks, numeric_trial_ids, summarize_banks
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity import check_reference_after_fit
from source_adapter import compute_only


def validate_session(item, session):
    if (item["status"] != "complete_two_elastic_net_session_banks" or item["session_id"] != session
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or set(item["banks"]) != set(RECIPES) or item["source"]["native_offsets"] != [0, 14]):
        raise ValueError("Both complete native current-bin elastic-net/control banks and exact replay required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or channels != sorted(set(channels)) or not channels
            or any(type(c) is not int or c < 1 for c in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or control["direction_counts"] != counts
            or candidate["physical_electrode_count"] != len(channels)
            or candidate["elastic_net_two_response_fits"] != 40
            or candidate["native_XY_scaled"] is not False or candidate["coordinatewise_L1"] is not True
            or candidate["common_velocity_penalty_scale"] is not True or candidate["maximum_sweeps"] != 10000
            or control["training_label_replicate_ids"] != list(REPLICATES)
            or control["elastic_net_two_response_fits"] != 320
            or control["fitting_targets_unpermuted"] is not True
            or control["physical_Gram_shared_exactly"] is not True
            or control["physical_normalization_preserved"] is not True
            or control["training_LMP_permuted_as_whole_trials"] is not True):
        raise ValueError("Original physical/trial support and fixed native-target elastic-net identities required")
    metadata = candidate["direction_fit_metadata"]
    if [r["held_direction"] for r in metadata] != list(range(8)):
        raise ValueError("All eight outer held-direction fits required")
    for row in metadata:
        held = row["held_direction"]
        allowed = sorted(set(range(8))-{held})
        fitting_rows = 15*(source["complete_trials"]-counts[held])
        scale = row["common_velocity_penalty_scale"]
        if (row["fitting_bin_rows"] != fitting_rows
                or len(row["physical_mean"]) != len(channels) or len(row["physical_sd"]) != len(channels)
                or not np.isfinite(row["physical_mean"]).all() or not np.isfinite(row["physical_sd"]).all()
                or np.any(np.asarray(row["physical_sd"]) < 0)
                or not np.isfinite(scale) or scale < 0 or row["physical_Gram_shared_exactly"] is not True
                or [r["replicate"] for r in row["training_trial_maps_by_replicate"]] != list(REPLICATES)):
            raise ValueError("Complete fitting-only physical scaling and eight fixed training maps required")
        reference_ids = None
        for mapping in row["training_trial_maps_by_replicate"]:
            rep, maps = mapping["replicate"], mapping["directions"]
            if [m["direction"] for m in maps] != allowed:
                raise ValueError("Only seven fitting directions may enter a training map")
            current_ids = []
            for m in maps:
                d, recipients, donors = m["direction"], m["recipient_trial_ids"], m["donor_trial_ids"]
                numbers = numeric_trial_ids(recipients, session)
                if (not np.array_equal(numbers, np.sort(numbers)) or len(recipients) != counts[d]
                        or sorted(donors) != recipients
                        or m["seed_literal"] != f"ep05-lmp-elastic-net-training-label-v1|20261004|{session}|{held}|{rep}|{d}"):
                    raise ValueError("Complete canonical original-ID within-direction training maps required")
                current_ids.extend(recipients)
            if len(set(current_ids)) != len(current_ids) or (reference_ids is not None and current_ids != reference_ids):
                raise ValueError("Training-map support must be identical in all eight replicates")
            reference_ids = current_ids
        diagnostics = [row["paired_fit_diagnostics_by_lambda"], *row["control_fit_diagnostics_by_replicate_lambda"]]
        if len(diagnostics) != 9:
            raise ValueError("Candidate plus all eight control fits required")
        for arm in diagnostics:
            if [d["lambda"] for d in arm] != list(LAMBDAS):
                raise ValueError("All five original penalties required for every arm")
            for d in arm:
                finite_keys = ("maximum_KKT_residual", "KKT_residual_Frobenius_norm", "KKT_tolerance",
                               "objective_error_upper_bound", "coefficient_L1_norm", "coefficient_Frobenius_norm")
                if (d["converged"] is not True or not all(np.isfinite(d[k]) and d[k] >= 0 for k in finite_keys)
                        or d["maximum_KKT_residual"] > d["KKT_tolerance"]
                        or d["maximum_sweeps"] != 10000 or not 1 <= d["completed_sweeps"] <= 10000
                        or d["penalty_scale"] != scale or d["L2_penalty_coefficient"] != d["lambda"]
                        or d["L1_penalty_coefficient"] != d["lambda"]*scale
                        or not 0 <= d["active_X"] <= len(channels) or not 0 <= d["active_Y"] <= len(channels)
                        or d["zero_coefficients"] != 2*len(channels)-d["active_X"]-d["active_Y"]
                        or d["active_channel_union"]+d["active_channel_intersection"] != d["active_X"]+d["active_Y"]
                        or not 0 <= d["active_channel_intersection"] <= min(d["active_X"], d["active_Y"])
                        or not max(d["active_X"], d["active_Y"]) <= d["active_channel_union"] <= len(channels)):
                    raise ValueError("Complete converged elastic-net fits and exact support diagnostics required")
    if candidate["all_folds_zero_residual_scale"] != all(row["common_velocity_penalty_scale"] == 0 for row in metadata):
        raise ValueError("Structural zero-residual duplicate identity differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("session-bank-map", "reference-bank-root", "reference-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Complete elastic_net/control session bound must be at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False, "solver_backend": BACKEND}
    def deadline(*_):
        raise TimeoutError("Selected complete elastic_net/control session deadline reached")
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
            report.update(status="complete_two_elastic_net_session_banks", banks=banks)
            validate_session(report, args.session)
        else:
            selected_sources = json.loads(args.session_bank_map.read_text())
            if set(selected_sources) != set(SESSIONS) or any(not isinstance(p, str) for p in selected_sources.values()):
                raise ValueError("Explicit exact sixteen-session score-bank mapping required")
            sessions = {}
            for session in SESSIONS:
                item = json.loads(Path(selected_sources[session]).read_text())
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
            report.update(status="complete_elastic_net_control_panel", total_valid_primary_trials_including_prior=19+count,
                post_initial_comparisons=13+count, post_initial_controls=7+count//2,
                session_bank_map=str(args.session_bank_map), selected_score_bank_sources=selected_sources,
                solver_backend="summary_only_mixed_preserved_python_and_native_banks", reference_anchor={"path": str(args.reference_summary), "refitted": False, "panel": anchors[0]})
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_elastic_net.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
