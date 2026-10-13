#!/usr/bin/env python3
"""Complete fixed HW/control session banks and H/W anchor comparisons."""
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
from primary_history_repeatability_weight import ANCHORS, LAGS, RECIPES, REPLICATES, fit_banks, summarize_banks, weight_map
from primary_reference import LAMBDAS, SESSIONS
from run_primary_history import check_history_reference_after_fit
from run_primary_repeatability_weight import validate_session as validate_weight_session
from source_adapter import compute_only


def check_weight_reference_after_fit(root, session, banks):
    old = json.loads((root/session/"score_banks.json").read_text())
    validate_weight_session(old, session)
    previous, current = old["banks"][ANCHORS["repeatability"]], banks[RECIPES[0]]
    for key in ("physical_electrode_ids", "weight_label_maps"):
        if previous[key] != current[key]:
            raise ValueError("Original W physical state or fixed weight-label maps did not replay")
    for before, after in zip(previous["direction_fit_metadata"], current["direction_fit_metadata"]):
        for key in ("held_direction", "fitting_bin_rows", "physical_mean", "physical_sd",
                    "physical_zero_SD_columns", "repeatability"):
            if before[key] != after[key]:
                raise ValueError(f"Original current-only W fitting state did not replay: {key}")


def validate_session(item, session):
    if (item["status"] != "complete_two_history_repeatability_weight_session_banks"
            or item["session_id"] != session or set(item["banks"]) != set(RECIPES)
            or item["exact_original_LMP_baseline_target_trial_replay"] is not True
            or item["exact_original_current_weight_state_and_maps_replay"] is not True
            or item["distinct_history_input_and_primary_target_windows_verified"] is not True):
        raise ValueError("Both complete HW banks and all original replay checks required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    electrodes = len(channels)
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != list(LAGS) or not channels or channels != sorted(set(channels))
            or any(type(e) is not int or e < 1 for e in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or control["direction_counts"] != counts
            or candidate["physical_electrode_ids"] != channels or candidate["physical_electrode_count"] != electrodes
            or candidate["history_column_count"] != 4*electrodes
            or candidate["native_XY_scaled"] is not False or candidate["weights_use_velocity"] is not False
            or candidate["positive_floor"] != .1 or candidate["post_weight_standardization"] is not False
            or candidate["post_weight_centering"] is not False
            or any(item["banks"][r]["lag_offsets"] != list(LAGS) for r in RECIPES)
            or control["weight_label_replicate_ids"] != list(REPLICATES)
            or control["controls_have_own_Gram"] is not True or control["true_native_targets"] is not True):
        raise ValueError("Fixed physical/history/target support and native ridge identities required")
    maps = candidate["weight_label_maps"]
    if maps != [weight_map(channels, session, r)[1] for r in REPLICATES]:
        raise ValueError("Exact original W physical-label maps required")
    metadata = candidate["direction_fit_metadata"]
    if [d["held_direction"] for d in metadata] != list(range(8)):
        raise ValueError("Eight original held-direction fits required")
    for row in metadata:
        held = row["held_direction"]
        allowed = sorted(set(range(8))-{held})
        statistics = row["repeatability"]
        if (row["fitting_bin_rows"] != 15*(source["complete_trials"]-counts[held])
                or statistics["fitting_directions"] != allowed
                or statistics["fitting_direction_counts"] != [counts[d] for d in allowed]
                or len(row["controls_equal_candidate_by_replicate"]) != 8):
            raise ValueError("Fitting-only seven-direction support required")
        for key, shape in (("physical_mean", (electrodes,)), ("physical_sd", (electrodes,)),
                           ("history_physical_mean", (4, electrodes)), ("history_physical_sd", (4, electrodes)),
                           ("history_fitting_column_mean", (4*electrodes,))):
            array = np.asarray(row[key])
            if array.shape != shape or not np.isfinite(array).all() or (key.endswith("sd") and np.any(array < 0)):
                raise ValueError("Separate complete current and history normalization states required")
        for key in ("T", "P", "r", "a"):
            if len(statistics[key]) != electrodes or not np.isfinite(statistics[key]).all():
                raise ValueError("Finite full physical-channel repeatability statistics required")
        t, p, r, a = (np.asarray(statistics[key]) for key in ("T", "P", "r", "a"))
        if np.any(t < 0) or np.any(r < 0) or np.any(r > 1) or not np.array_equal(a, .1+r):
            raise ValueError("Fixed positive floor and clipped current-only ratios required")
        if len(statistics["raw_ratio"]) != electrodes or len(statistics["clip_category"]) != electrodes:
            raise ValueError("Original signed and undefined ratio diagnostics required")
        for e, raw in enumerate(statistics["raw_ratio"]):
            if t[e] == 0:
                if raw is not None or r[e] != 0:
                    raise ValueError("Zero current energy requires undefined ratio and r=0")
            elif raw is None or not np.isfinite(raw) or raw != p[e]/t[e] or r[e] != np.clip(raw, 0, 1):
                raise ValueError("Exact signed current repeatability ratio required")
        arms = [row["paired_weights"], *row["control_weights_by_replicate"]]
        covariances = [row["paired_weighted_covariance"], *row["control_weighted_covariance_by_replicate"]]
        if len(arms) != 9 or len(covariances) != 9:
            raise ValueError("One candidate and eight independently fitted controls required")
        diagonal = np.asarray(arms[0]["history_energy_by_lag_electrode"])
        if diagonal.shape != (4, electrodes) or not np.isfinite(diagonal).all() or np.any(diagonal < 0):
            raise ValueError("Finite nonnegative full history energy required")
        q, active = diagonal.sum(axis=0), np.flatnonzero(diagonal.sum(axis=0) > 0)
        if row["all_past_lags_zero"] != bool(np.all(diagonal[1:] == 0)):
            raise ValueError("No-past duplicate requires exact zero energy for all three past lags")
        for arm, (info, covariance) in enumerate(zip(arms, covariances)):
            expected_a = a if arm == 0 else a[weight_map(channels, session, arm-1)[0]]
            weights = np.asarray(info["weights"])
            identity = not len(active) or bool(np.all(expected_a[active] == expected_a[active][0]))
            if (info["positive_energy_columns"] != active.tolist()
                    or not np.array_equal(info["history_energy_by_electrode"], q)
                    or not np.array_equal(info["history_energy_by_lag_electrode"], diagonal)
                    or not np.array_equal(info["base_weights"], expected_a)
                    or info["structural_identity"] != identity or type(info["structural_identity"]) is not bool
                    or weights.shape != (electrodes,) or not np.isfinite(weights).all() or np.any(weights <= 0)
                    or (identity and not np.array_equal(weights, np.ones(electrodes)))
                    or info["original_trace"] != arms[0]["original_trace"]
                    or len(covariance["eigenvalues"]) != 4*electrodes
                    or not np.isfinite(covariance["eigenvalues"]).all()
                    or [d["lambda"] for d in covariance["ridge_df_by_lambda"]] != list(LAMBDAS)):
                raise ValueError("Own history Gram, physical weighting and exact structural identities required")
            if arm and row["controls_equal_candidate_by_replicate"][arm-1] != bool(np.array_equal(
                    weights[active], np.asarray(arms[0]["weights"])[active])):
                raise ValueError("Control equivalence requires exact applied weights on full history support")
    no_past = all(d["all_past_lags_zero"] for d in metadata)
    if (candidate["structural_identity"] != all(d["paired_weights"]["structural_identity"] for d in metadata)
            or control["structural_identity"] != all(a["structural_identity"] for d in metadata for a in d["control_weights_by_replicate"])
            or candidate["all_controls_equal_candidate"] != all(all(d["controls_equal_candidate_by_replicate"]) for d in metadata)
            or any(item["banks"][recipe]["all_past_lags_zero"] != no_past for recipe in RECIPES)):
        raise ValueError("Per-recipe duplicate accounting must match every fold and control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "weight-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "history-bank-root", "history-summary", "weight-bank-root", "weight-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted complete HW session bound is at most 270 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete HW session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, rows, source = load_lmp_history_session(args.source, args.trial_table, args.scalar_bank_root, args.session, check)
            ids = np.array([r["trial_id"] for r in rows])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting = time.monotonic()
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check, electrode_ids=source["common_LMP_electrodes"])
            report["two_recipe_fit_and_score_seconds"] = time.monotonic()-fitting
            check_history_reference_after_fit(args.reference_bank_root, args.session, source, y, baseline, labels, ids)
            check_weight_reference_after_fit(args.weight_bank_root, args.session, banks)
            report.update(exact_original_LMP_baseline_target_trial_replay=True,
                          distinct_history_input_and_primary_target_windows_verified=True,
                          exact_original_current_weight_state_and_maps_replay=True)
            check()
            for recipe in RECIPES:
                key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                         directions=labels, trial_ids=ids, **{key: predictions[recipe]})
            report.update(status="complete_two_history_repeatability_weight_session_banks", banks=banks)
            validate_session(report, args.session)
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            anchor_banks, anchor_panels, sources = {}, {}, {}
            for name, root, path, status, bank_status in (
                ("history", args.history_bank_root, args.history_summary, "complete_two_history_control_comparisons", "complete_two_history_control_session_banks"),
                ("repeatability", args.weight_bank_root, args.weight_summary, "complete_repeatability_weight_control_panel", "complete_two_repeatability_weight_session_banks")):
                reference = json.loads(path.read_text())
                panels = [p for p in reference["panels"] if p["recipe"] == ANCHORS[name]]
                if reference["status"] != status or len(panels) != 1:
                    raise ValueError("Complete preserved H and W summaries required")
                anchor_panels[name], anchor_banks[name] = panels[0], {}
                sources[name] = {"path": str(path), "bank_root": str(root), "refitted": False}
                for session in SESSIONS:
                    old = json.loads((root/session/"score_banks.json").read_text())
                    if (old["status"] != bank_status or old["session_id"] != session
                            or old["exact_original_LMP_baseline_target_trial_replay"] is not True):
                        raise ValueError("Complete preserved original H/W banks required")
                    if name == "repeatability":
                        validate_weight_session(old, session)
                    anchor_banks[name][session] = old["banks"][ANCHORS[name]]
            report.update(summarize_banks(sessions, anchor_banks, anchor_panels))
            count = report["scientific_trial_count_increment"]
            report.update(status="complete_history_repeatability_weight_control_panel",
                total_valid_primary_trials_including_prior=21+count, post_initial_comparisons=15+count,
                post_initial_controls=8+report["control_trial_count_increment"], bank_root=str(args.bank_root), reference_anchors=sources)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_history_repeatability_weight.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
