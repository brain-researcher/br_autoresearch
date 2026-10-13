#!/usr/bin/env python3
"""Fit fixed H2W and its training surrogate; replay immutable original anchors."""
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
from primary_capacity_matched_surrogate import RECIPE as OLD_NULL
from primary_covariance_svd import numeric_trial_ids
from primary_history_blend import SCORE_KEYS, prediction_scores
from primary_history2_repeatability_weight import ANCHORS, RECIPES, REPLICATES, fit_banks, summarize_banks, trial_map
from primary_reference import LAMBDAS, SESSIONS
from run_primary_capacity_matched_surrogate import validate_session as validate_capacity_session
from run_primary_elapsed_only import check_anchor_after_fit
from run_primary_history_blend import SUMMARY_STATUSES, load_components, restore_target_layout
from source_adapter import compute_only

REPLAY_FLAGS = ("exact_original_HW_neural_state_replay", "exact_original_W_weight_state_replay",
    "exact_original_component_arrays_and_scores_replay", "exact_original_capacity_maps_replay",
    "exact_original_capacity_arrays_and_scores_replay")


def check_saved_anchors(w_root, hw_root, capacity_root, session, source, y, baseline, labels, ids, banks, predictions):
    candidate = banks[RECIPES[0]]
    check_anchor_after_fit(hw_root, session, source, y, labels, ids, baseline, candidate)
    old_y, old_b, old_labels, old_ids, old_predictions, old_banks, _ = load_components(w_root, hw_root, session)
    for actual, original in ((y, old_y), (baseline, old_b), (labels, old_labels), (ids, old_ids)):
        if not np.array_equal(actual, original):
            raise ValueError("Both original component supports must exactly replay source support")
    for new, old in zip(candidate["direction_fit_metadata"], old_banks["W"]["direction_fit_metadata"]):
        for key in ("held_direction", "fitting_bin_rows", "physical_mean", "physical_sd", "physical_zero_SD_columns", "repeatability"):
            if new[key] != old[key]:
                raise ValueError("Unchanged original W current-only weight state required: "+key)
    old = json.loads((capacity_root/session/"score_banks.json").read_text())
    validate_capacity_session(old, session)
    old_null = old["bank"]
    if old_null["physical_electrode_ids"] != candidate["physical_electrode_ids"]:
        raise ValueError("Original capacity physical support required")
    for new, previous in zip(candidate["direction_fit_metadata"], old_null["direction_fit_metadata"]):
        if new["surrogate_trial_maps"] != previous["surrogate_trial_maps"]:
            raise ValueError("All original capacity maps must replay exactly")
    path = capacity_root/session/(OLD_NULL+"_predictions.npz")
    with np.load(path, allow_pickle=False) as saved:
        for key, actual in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
            if not np.array_equal(saved[key], actual):
                raise ValueError("Original capacity saved support changed: "+key)
        original_null_predictions = saved["candidate_xy_by_replicate_lambda"]
    if original_null_predictions.shape != (8, 5, *y.shape) or not np.isfinite(original_null_predictions).all():
        raise ValueError("Complete original eight-by-five surrogate predictions required")
    for replicate in REPLICATES:
        replay = score_bank(y, labels, baseline, original_null_predictions[replicate], OLD_NULL)
        if replay["lambda_rows"] != old_null["surrogate_seed_rows"][replicate]["lambda_rows"]:
            raise ValueError("Original surrogate seed score replay failed")
    candidate.update(original_component_score_rows={name: bank["lambda_rows"] for name, bank in old_banks.items()},
        exact_prediction_equals_components={name: [[bool(np.array_equal(predictions[RECIPES[0]][i], values[j]))
            for j in range(5)] for i in range(5)] for name, values in old_predictions.items()},
        exact_surrogate_equals_original_HW_surrogate_by_lambda=[[
            bool(np.array_equal(predictions[RECIPES[1]][:, i], original_null_predictions[:, j]))
            for j in range(5)] for i in range(5)])


def validate_session(item, session):
    if (item["status"] != "complete_history2_repeatability_session_banks" or item["session_id"] != session
            or any(item[key] is not True for key in REPLAY_FLAGS)
            or item["original_HW_refitted"] is not False or item["original_W_refitted"] is not False
            or item["promotion_authorized"] is not False or item["scientific_trial_count_increment"] != 0
            or set(item["banks"]) != set(RECIPES)):
        raise ValueError("Both complete H2W recipes and exact original replays required")
    source = item["source"]
    candidate, control = (item["banks"][recipe] for recipe in RECIPES)
    channels, counts = source["common_LMP_electrodes"], candidate["direction_counts"]
    labels, ids = np.asarray(candidate["original_direction_labels"]), np.asarray(candidate["original_trial_ids"])
    numeric_trial_ids(ids, session)
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != [0, -1, -2, -3] or not channels or channels != sorted(set(channels))
            or any(type(e) is not int or e < 1 for e in channels)
            or len(counts) != 8 or any(type(n) is not int or n < 2 for n in counts)
            or sum(counts) != source["complete_trials"] or labels.shape != ids.shape or len(ids) != sum(counts)
            or counts != [int(np.count_nonzero(labels == d)) for d in range(8)]
            or candidate["shared_Gram_eigendecompositions"] != 8
            or candidate["surrogate_specific_Gram_eigendecompositions"] != 0
            or candidate["zero_energy_is_duplicate_proof"] is not False):
        raise ValueError("Exact original reader support and explicit two-lag model required")
    for recipe, bank in item["banks"].items():
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or bank["physical_electrode_ids"] != channels or bank["physical_electrode_count"] != len(channels)
                or bank["history_column_count"] != 2*len(channels) or bank["lag_offsets"] != [0, -1]
                or bank["reader_LMP_offsets"] != [-3, 14] or bank["effective_LMP_offsets"] != [-1, 14]
                or bank["native_XY_scaled"] is not False or bank["weights_use_velocity"] is not False
                or bank["post_weight_standardization"] is not False or bank["post_weight_centering"] is not False
                or bank["positive_floor"] != .1 or bank["original_HW_refitted"] is not False
                or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0
                or bank["original_direction_labels"] != labels.tolist() or bank["original_trial_ids"] != ids.tolist()
                or any(bank[key] != candidate[key] for key in ("baseline_R2", "baseline_SSE", "denominator", "direction_counts"))):
            raise ValueError("Complete same-support native two-lag banks required")
    seeds = control["surrogate_seed_rows"]
    if control["surrogate_replicate_ids"] != list(REPLICATES) or [row["replicate"] for row in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed surrogate banks required")
    for rows in (candidate["lambda_rows"], control["lambda_rows"], *candidate["original_component_score_rows"].values(),
                 *(seed["lambda_rows"] for seed in seeds)):
        if [row["lambda"] for row in rows] != list(LAMBDAS) or not all(np.isfinite(row[key]) for row in rows for key in SCORE_KEYS):
            raise ValueError("Complete finite original penalty rows required")
    for index in range(5):
        for key in ("candidate_SSE", "wrong_pair_SSE"):
            if control["lambda_rows"][index][key] != float(np.mean([row["lambda_rows"][index][key] for row in seeds])):
                raise ValueError("Control must average all eight correct/wrong SSEs")
    for flags in (*candidate["exact_prediction_equals_components"].values(), candidate["exact_surrogate_equals_original_HW_surrogate_by_lambda"]):
        if np.asarray(flags).shape != (5, 5) or any(type(value) is not bool for row in flags for value in row):
            raise ValueError("Literal selected-array equality evidence required")
    metadata = candidate["direction_fit_metadata"]
    if [row["held_direction"] for row in metadata] != list(range(8)):
        raise ValueError("Every original direction fit required")
    for row in metadata:
        held, capacity = row["held_direction"], row["capacity_identity"]
        train = labels != held
        if (row["fitting_bin_rows"] != 15*int(np.count_nonzero(train))
                or row["surrogate_trial_maps"] != [trial_map(ids[train], labels[train], session, held, r)[1] for r in REPLICATES]
                or row["repeatability"]["fitting_directions"] != sorted(set(range(8))-{held})
                or row["repeatability"]["fitting_direction_counts"] != [counts[d] for d in range(8) if d != held]
                or capacity["feature_columns"] != 2*len(channels) or capacity["fitting_design_rows"] != row["fitting_bin_rows"]
                or capacity["shared_Gram_eigendecompositions"] != 1 or capacity["surrogate_specific_Gram_eigendecompositions"] != 0
                or row["h2_weighted_covariance"] != capacity["shared_covariance"]):
            raise ValueError("Preserved fitting-only maps and one shared two-lag Gram required")
        for key in ("surrogate_cross_product_equals_original_by_replicate", "surrogate_design_equals_original_by_replicate",
                    "whole_fitting_map_identity_by_replicate"):
            if len(capacity[key]) != 8 or any(type(flag) is not bool for flag in capacity[key]):
                raise ValueError("Complete exact capacity identity flags required")
        info, covariance = row["h2_weights"], row["h2_weighted_covariance"]
        if (np.asarray(info["history_energy_by_lag_electrode"]).shape != (2, len(channels))
                or info["base_weights"] != row["repeatability"]["a"]
                or len(covariance["eigenvalues"]) != 2*len(channels) or not np.isfinite(covariance["eigenvalues"]).all()
                or [entry["lambda"] for entry in covariance["ridge_df_by_lambda"]] != list(LAMBDAS)):
            raise ValueError("Two-lag current-prior metric and shared covariance required")
    expected = all(all(row["capacity_identity"]["surrogate_cross_product_equals_original_by_replicate"]) for row in metadata)
    if type(candidate["structural_control_equals_candidate"]) is not bool or candidate["structural_control_equals_candidate"] != expected:
        raise ValueError("Control duplicate must cover all directions and all draws")


def attach_blend_and_replay_new(bank_root, sessions, blend_report):
    blend_rows = None if blend_report is None else {row["session_id"]: row for row in blend_report["blend"]["sessions"]}
    if blend_report is not None and (blend_report["status"] != "complete_history_blend_panel"
            or blend_report["all_selected_prediction_scores_replayed"] is not True
            or blend_report["original_selectors_independently_replayed"] is not True
            or set(blend_report["selected_prediction_paths"]) != set(SESSIONS)
            or len(blend_report["blend"]["sessions"]) != 16 or set(blend_rows) != set(SESSIONS)):
        raise ValueError("Only complete original selected blend may be used")
    for session in SESSIONS:
        candidate, control = (sessions[session][recipe] for recipe in RECIPES)
        with np.load(bank_root/session/(RECIPES[0]+"_predictions.npz"), allow_pickle=False) as saved:
            y, baseline = restore_target_layout(saved["actual_xy"]), restore_target_layout(saved["baseline_xy"])
            labels, ids, values = saved["directions"], saved["trial_ids"], saved["candidate_xy_by_lambda"]
        if labels.tolist() != candidate["original_direction_labels"] or ids.tolist() != candidate["original_trial_ids"]:
            raise ValueError("New saved trial identities changed")
        replay = score_bank(y, labels, baseline, values, RECIPES[0])
        if any(replay[key] != candidate[key] for key in ("lambda_rows", "baseline_R2", "baseline_SSE", "denominator", "direction_counts")):
            raise ValueError("New candidate saved score bank did not replay")
        with np.load(bank_root/session/(RECIPES[1]+"_predictions.npz"), allow_pickle=False) as saved:
            for key, actual in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
                if not np.array_equal(saved[key], actual):
                    raise ValueError("New candidate/control saved supports differ")
            null_values = saved["candidate_xy_by_replicate_lambda"]
        for r in REPLICATES:
            if score_bank(y, labels, baseline, null_values[r], RECIPES[1])["lambda_rows"] != control["surrogate_seed_rows"][r]["lambda_rows"]:
                raise ValueError("New control saved seed scores did not replay")
        if blend_report is not None:
            with np.load(Path(blend_report["selected_prediction_paths"][session]), allow_pickle=False) as saved:
                for key, actual in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
                    if not np.array_equal(saved[key], actual):
                        raise ValueError("Original selected blend support must match exactly")
                prediction = saved["candidate_xy"]
            row = blend_rows[session]
            replay = prediction_scores(y, baseline, labels, prediction)
            if any(replay[key] != row[key] for key in SCORE_KEYS):
                raise ValueError("Original selected blend score must replay")
            candidate.update(blend_reference_row=row,
                exact_prediction_equals_blend=[bool(np.array_equal(value, prediction)) for value in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "w-bank-root", "hw-bank-root", "capacity-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "w-bank-root", "hw-bank-root", "w-summary", "hw-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    summary.add_argument("--blend-summary", type=Path)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted H2W session bound is at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False,
        "original_W_refitted": False, "promotion_authorized": False, "search_complete": False, "new_promotion_gate": False}
    def deadline(*_):
        raise TimeoutError("Adopted H2W session deadline reached")
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
            fitted = time.monotonic()
            banks, baseline, predictions = fit_banks(x, y, labels, ids, args.session, check,
                electrode_ids=source["common_LMP_electrodes"])
            report["candidate_and_control_fit_score_seconds"] = time.monotonic()-fitted
            check_saved_anchors(args.w_bank_root, args.hw_bank_root, args.capacity_bank_root,
                args.session, source, y, baseline, labels, ids, banks, predictions)
            check()
            for recipe in RECIPES:
                prediction_key = "candidate_xy_by_lambda" if recipe == RECIPES[0] else "candidate_xy_by_replicate_lambda"
                np.savez(args.out/(recipe+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                    directions=labels, trial_ids=ids, **{prediction_key: predictions[recipe]})
            report.update({key: True for key in REPLAY_FLAGS})
            report.update(banks=banks, status="complete_history2_repeatability_session_banks")
            validate_session(report, args.session)
        else:
            sessions, anchors, panels = {}, {name: {} for name in ANCHORS}, {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
                for name, root in (("W", args.w_bank_root), ("HW", args.hw_bank_root)):
                    original = json.loads((root/session/"score_banks.json").read_text())
                    anchors[name][session] = original["banks"][ANCHORS[name]]
                    if sessions[session][RECIPES[0]]["original_component_score_rows"][name] != anchors[name][session]["lambda_rows"]:
                        raise ValueError("Original component scores changed since fitting")
            for name, path in (("W", args.w_summary), ("HW", args.hw_summary)):
                original = json.loads(path.read_text())
                matching = [panel for panel in original["panels"] if panel["recipe"] == ANCHORS[name]]
                if original["status"] != SUMMARY_STATUSES[name] or len(matching) != 1:
                    raise ValueError("Complete original component summary required")
                panels[name] = matching[0]
            blend = None if args.blend_summary is None else json.loads(args.blend_summary.read_text())
            attach_blend_and_replay_new(args.bank_root, sessions, blend)
            report.update(summarize_banks(sessions, anchors, panels, None if blend is None else blend["blend"]),
                status="complete_history2_repeatability_panel", bank_root=str(args.bank_root),
                original_component_summaries={"W": str(args.w_summary), "HW": str(args.hw_summary)},
                original_blend_summary=None if args.blend_summary is None else str(args.blend_summary),
                all_original_replays=True, all_new_saved_scores_replayed=True, original_selectors_independently_replayed=True)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_history2_repeatability_weight.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
