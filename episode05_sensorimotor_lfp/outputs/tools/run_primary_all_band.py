#!/usr/bin/env python3
"""Run fixed all-band session banks and the complete J/LMP anchor comparison."""
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

from all_band_reader import load_session
from primary_all_band import ALL_BANDS, ANCHORS, BLOCKS, RECIPES, REPLICATES, fit_banks, summarize_banks, trial_map
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity import check_reference_after_fit
from run_primary_lmp_high import validate_session as validate_J_session
from source_adapter import compute_only


def check_J_reference_after_fit(root, session, source, y, baseline, labels, ids, banks):
    old = json.loads((root/session/"score_banks.json").read_text())
    validate_J_session(old, session)
    for key in ("common_LMP_electrodes", "complete_trials", "native_offsets", "native_bin_seconds"):
        if old["source"][key] != source[key]:
            raise ValueError("Retained J source support changed: "+key)
    for before, after in zip(old["banks"][ANCHORS["joint"]]["direction_fit_metadata"], banks[RECIPES[0]]["direction_fit_metadata"]):
        if before["held_direction"] != after["held_direction"] or before["fitting_bin_rows"] != after["fitting_bin_rows"]:
            raise ValueError("Original J fitting direction support changed")
        for name, key in (("lmp", "LMP_zero_SD_columns"), ("high", "high_zero_SD_columns")):
            if before[key] != np.flatnonzero(np.asarray(after["physical_states"][name]["physical_sd"]) == 0).tolist():
                raise ValueError("Original J physical zero-SD support did not replay")
    with np.load(root/session/(ANCHORS["joint"]+"_predictions.npz"), allow_pickle=False) as original:
        for key, actual in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
            if not np.array_equal(original[key], actual):
                raise ValueError("Original J target/baseline/trial replay differs: "+key)


def validate_session(item, session):
    if (item["status"] != "complete_two_all_band_session_banks" or item["session_id"] != session
            or set(item["banks"]) != set(RECIPES) or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or item["original_J_support_zero_SD_and_target_replay"] is not True):
        raise ValueError("Both complete all-band banks and original reference replays required")
    source, paired, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], paired["direction_counts"]
    width = len(channels)
    bands = {name: [list(b) for b in values] for name, values in BLOCKS.items()}
    ids, labels = np.asarray(paired["original_trial_ids"]), np.asarray(paired["original_trial_directions"])
    if (source["session_id"] != session or source["native_offsets"] != [0, 14] or source["native_bin_seconds"] != .03
            or source["bands_hz"] != bands or source["all_bands_hz"] != [list(b) for b in ALL_BANDS]
            or source["all_trial_guides_qualified_before_values"] is not True or source["source_spike_values_read"] is not False
            or not channels or channels != sorted(set(channels)) or any(type(e) is not int or e < 1 for e in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or control["direction_counts"] != counts
            or ids.shape != (source["complete_trials"],) or labels.shape != ids.shape
            or [int(np.count_nonzero(labels == d)) for d in range(8)] != counts
            or paired["physical_electrode_ids"] != channels or paired["bands_hz"] != bands
            or any(item["banks"][r]["feature_width"] != 9*width for r in RECIPES)
            or paired["native_XY_scaled"] is not False or paired["block_rescaling"] is not False
            or control["added_training_replicate_ids"] != list(REPLICATES)):
        raise ValueError("Exact nine-band original physical/trial/target support required")
    for key in ("training_added_only_permuted", "evaluation_unpermuted", "true_fitting_targets",
                "physical_normalization_preserved", "controls_have_own_full_Gram"):
        if control[key] is not True:
            raise ValueError("Fixed added-block training control required: "+key)
    metadata = paired["direction_fit_metadata"]
    if [row["held_direction"] for row in metadata] != list(range(8)):
        raise ValueError("All original held-direction fits required")
    for row in metadata:
        held = row["held_direction"]
        train = labels != held
        if (row["fitting_bin_rows"] != 15*int(np.count_nonzero(train))
                or not np.isfinite(row["added_fitting_energy"]) or row["added_fitting_energy"] < 0
                or row["structural_J_duplicate"] != (row["added_fitting_energy"] == 0)
                or len(row["controls_equal_candidate_by_replicate"]) != 8
                or any(type(value) is not bool for value in row["controls_equal_candidate_by_replicate"])):
            raise ValueError("Fixed fitting support and exact design/zero-energy identities required")
        for name, bands_for_block in BLOCKS.items():
            for key in ("physical_mean", "physical_sd"):
                values = np.asarray(row["physical_states"][name][key])
                if values.shape != (width*len(bands_for_block),) or not np.isfinite(values).all() or (key == "physical_sd" and np.any(values < 0)):
                    raise ValueError("Complete fitting-only physical normalization required")
        means = np.asarray([row["paired_fitting_column_mean"], *row["control_fitting_column_means"]])
        if means.shape != (9, 9*width) or not np.isfinite(means).all():
            raise ValueError("Separate finite centering for candidate and all controls required")
        covariances = [row["paired_covariance"], *row["control_covariances"]]
        if len(covariances) != 9:
            raise ValueError("Nine separate complete physical Grams required")
        for covariance in covariances:
            if (len(covariance["eigenvalues"]) != 9*width or not np.isfinite(covariance["eigenvalues"]).all()
                    or not np.isfinite(covariance["trace"]) or covariance["trace"] < 0
                    or [d["lambda"] for d in covariance["ridge_df_by_lambda"]] != list(LAMBDAS)):
                raise ValueError("Own-arm full covariance/df diagnostics required")
        expected = [{"replicate": r, "directions": trial_map(ids[train], labels[train], session, held, r)[1]} for r in REPLICATES]
        if row["training_added_maps"] != expected:
            raise ValueError("Exact original native-ID fitting-only added-block maps required")
    duplicate = all(row["structural_J_duplicate"] for row in metadata)
    same = all(all(row["controls_equal_candidate_by_replicate"]) for row in metadata)
    if (paired["structural_J_duplicate"] != duplicate or control["structural_J_duplicate"] != duplicate
            or paired["all_controls_equal_candidate"] != same or control["structural_paired_duplicate"] != same):
        raise ValueError("Complete-recipe duplicate flags must match every fold and seed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "joint-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "reference-bank-root", "reference-summary", "joint-bank-root", "joint-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted complete all-band session bound is at most 270 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete all-band session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            blocks, y, labels, rows, source = load_session(args.source, args.trial_table, args.scalar_bank_root, args.session, check)
            ids = np.array([row["trial_id"] for row in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting = time.monotonic()
            banks, baseline, predictions = fit_banks(blocks["lmp"], blocks["high"], blocks["added"], y, labels, ids,
                args.session, check, electrode_ids=source["common_LMP_electrodes"])
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting
            check_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            check_J_reference_after_fit(args.joint_bank_root, args.session, source, y, baseline, labels, ids, banks)
            report.update(exact_original_LMP_baseline_target_trial_replay=True, original_J_support_zero_SD_and_target_replay=True)
            check()
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline, directions=labels,
                         trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_all_band_session_banks", banks=banks)
            validate_session(report, args.session)
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            anchor_banks, anchor_panels, sources = {}, {}, {}
            for name, root, path, status, bank_status in (
                ("joint", args.joint_bank_root, args.joint_summary, "complete_two_lmp_high_comparisons", "complete_two_lmp_high_session_banks"),
                ("lmp", args.reference_bank_root, args.reference_summary, "complete_three_multichannel_comparisons", "complete_three_session_banks")):
                reference = json.loads(path.read_text())
                panels = [p for p in reference["panels"] if p["recipe"] == ANCHORS[name]]
                if reference["status"] != status or len(panels) != 1:
                    raise ValueError("Complete preserved J/LMP summaries required")
                anchor_panels[name], anchor_banks[name] = panels[0], {}
                sources[name] = {"path": str(path), "bank_root": str(root), "refitted": False}
                for session in SESSIONS:
                    old = json.loads((root/session/"score_banks.json").read_text())
                    if old["status"] != bank_status or old["session_id"] != session:
                        raise ValueError("Complete original J/LMP banks required")
                    if name == "joint":
                        validate_J_session(old, session)
                    anchor_banks[name][session] = old["banks"][ANCHORS[name]]
            report.update(summarize_banks(sessions, anchor_banks, anchor_panels))
            count = report["scientific_trial_count_increment"]
            report.update(status="complete_all_band_control_panel", total_valid_primary_trials_including_prior=25+count,
                post_initial_comparisons=19+count, post_initial_controls=10+report["control_trial_count_increment"],
                bank_root=str(args.bank_root), reference_anchors=sources)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_all_band.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
