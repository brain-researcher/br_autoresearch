#!/usr/bin/env python3
"""Bounded HW105 session bank and complete sixteen-session release only."""
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
from primary_capacity import landmark_rows, score_bank
from primary_capacity_matched_surrogate import trial_map
from primary_covariance_svd import numeric_trial_ids
from primary_history_blend import SCORE_KEYS, prediction_scores, select_cell, validate_original_selections
from primary_history_rbf import ANCHORS, RECIPES, SUPPORT_KEYS, duplicate_accounting, fit_banks, summarize_banks
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from run_primary_elapsed_only import check_anchor_after_fit
from run_primary_history_blend import restore_target_layout, validate_session as validate_blend_session
from run_primary_history_repeatability_weight import validate_session as validate_hw_session
from source_adapter import compute_only

SESSION_STATUS = "complete_history_rbf_session_banks"
PANEL_STATUS = "complete_history_rbf_panel"
SUMMARY_FILE = "primary_history_rbf.json"
SUMMARY_OPTIONS = ("bank_root", "hw_bank_root", "k_bank_root", "blend_bank_root",
                   "hw_summary", "k_summary", "blend_summary")


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_session(item, session):
    if (item["status"] != SESSION_STATUS or item["session_id"] != session
            or item["exact_original_HW_neural_state_replay"] is not True
            or item["exact_original_HW_saved_arrays_and_scoring_replay"] is not True
            or item["exact_original_HW_weighted_covariance_replay"] is not True
            or item["original_HW_refitted"] is not False or item["promotion_authorized"] is not False
            or item["scientific_trial_count_increment"] != 0 or set(item["banks"]) != set(RECIPES)):
        raise ValueError("Complete uncharged candidate/control session and exact HW replay required")
    source, candidate, control = item["source"], item["banks"][RECIPES[0]], item["banks"][RECIPES[1]]
    ids, labels = np.asarray(candidate["original_trial_ids"]), np.asarray(candidate["original_direction_labels"])
    numeric_trial_ids(ids, session)
    channels = source["common_LMP_electrodes"]
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != [0, -1, -2, -3] or not channels
            or channels != sorted(set(channels)) or any(type(c) is not int or c < 1 for c in channels)
            or source["complete_trials"] != len(ids) or labels.shape != ids.shape
            or set(labels.tolist()) != set(range(8))):
        raise ValueError("Complete original native source support required")
    for recipe, bank in item["banks"].items():
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or bank["physical_electrode_ids"] != channels or bank["physical_electrode_count"] != len(channels)
                or bank["history_column_count"] != 4*len(channels) or bank["kernel_feature_columns"] != 105
                or bank["original_trial_ids"] != ids.tolist() or bank["original_direction_labels"] != labels.tolist()
                or bank["direction_counts"] != [int(np.count_nonzero(labels == d)) for d in range(8)]
                or any(n < 2 for n in bank["direction_counts"])
                or bank["lag_offsets"] != [0, -1, -2, -3] or bank["native_XY_scaled"] is not False
                or bank["weights_use_velocity"] is not False or bank["positive_floor"] != .1
                or bank["post_weight_standardization"] is not False or bank["post_weight_centering"] is not False
                or bank["original_HW_refitted"] is not False
                or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0):
            raise ValueError("Fixed HW/RBF representation and native scoring required")
        if any(bank[k] != candidate[k] for k in SUPPORT_KEYS):
            raise ValueError("Identical candidate/control native support required")
        if ([r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(row[k]) for row in bank["lambda_rows"] for k in SCORE_KEYS)):
            raise ValueError("Complete finite five-penalty bank required")
    if (control["surrogate_replicate_ids"] != list(range(8))
            or [r["replicate"] for r in control["surrogate_seed_rows"]] != list(range(8))
            or control["direction_fit_metadata"] != candidate["direction_fit_metadata"]):
        raise ValueError("All eight fixed controls with the same frozen feature map required")
    for index in range(5):
        for key in ("candidate_SSE", "wrong_pair_SSE"):
            if control["lambda_rows"][index][key] != float(np.mean([
                    r["lambda_rows"][index][key] for r in control["surrogate_seed_rows"]])):
                raise ValueError("Mean control SSE must include all eight draws")
    metadata = candidate["direction_fit_metadata"]
    if [r["held_direction"] for r in metadata] != list(range(8)):
        raise ValueError("All eight whole-direction folds required")
    for row in metadata:
        held, capacity = row["held_direction"], row["capacity_identity"]
        train = labels != held
        _, landmark_ids = landmark_rows(ids[train], labels[train], session, held)
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in range(8)]
        sd = np.asarray(row["history_physical_sd"])
        if (sd.shape != (4, len(channels)) or not np.isfinite(sd).all() or np.any(sd < 0)
                or row["active_RBF_columns"] != int(np.count_nonzero(sd > 0))
                or row["RBF_sigma_squared"] != row["active_RBF_columns"]
                or row["landmark_trial_ids"] != landmark_ids or row["landmark_count"] != 105
                or row["landmark_bins_per_trial"] != 15 or row["landmark_diagonal_regularizer"] != 1e-8
                or row["fitting_bin_rows"] != 15*int(train.sum())
                or row["repeatability"]["fitting_directions"] != sorted(set(range(8))-{held})
                or row["surrogate_trial_maps"] != [records for _, records in maps]
                or capacity["fitting_design_rows"] != row["fitting_bin_rows"] or capacity["feature_columns"] != 105
                or capacity["shared_head_Gram_eigendecompositions"] != 1
                or capacity["surrogate_specific_Gram_eigendecompositions"] != 0):
            raise ValueError("Inherited landmark/constant-column rules and shared feature Gram required")
        for key in ("surrogate_cross_product_equals_candidate_by_replicate",
                    "surrogate_design_equals_candidate_by_replicate", "whole_fitting_map_identity_by_replicate"):
            if len(capacity[key]) != 8 or any(type(value) is not bool for value in capacity[key]):
                raise ValueError("All eight explicit identity flags required")
        if capacity["whole_fitting_map_identity_by_replicate"] != [
                bool(np.array_equal(mapping, np.arange(len(mapping)))) for mapping, _ in maps]:
            raise ValueError("Recorded map identities changed")


def load_original_context(args):
    banks = {name: {} for name in ("HW", "K")}
    blend_banks, sources = {}, {name: {} for name in ("HW", "K")}
    for s in SESSIONS:
        for name, root in (("HW", args.hw_bank_root), ("K", args.k_bank_root)):
            item = read_json(root/s/"score_banks.json")
            if name == "HW":
                validate_hw_session(item, s)
            elif (item["status"] != "complete_two_capacity_session_banks" or item["session_id"] != s
                    or item["exact_original_LMP_baseline_target_trial_replay"] is not True):
                raise ValueError("Complete original stationary K bank required")
            banks[name][s], sources[name][s] = item["banks"][ANCHORS[name]], item["source"]
        item = read_json(args.blend_bank_root/s/"score_banks.json")
        validate_blend_session(item, s)
        blend_banks[s] = item["bank"]
    originals, rows = {}, {}
    for name, status in (("HW", "complete_history_repeatability_weight_control_panel"),
                         ("K", "complete_two_capacity_comparisons")):
        original = read_json(getattr(args, name.lower()+"_summary"))
        matches = [p for p in original["panels"] if p["recipe"] == ANCHORS[name]]
        if original["status"] != status or len(matches) != 1:
            raise ValueError("Complete original independent selection summary required")
        rows[name] = validate_original_selections(banks[name], matches[0])
        originals[name] = original
    blend = read_json(args.blend_summary)
    if (blend["status"] != "complete_history_blend_panel" or blend["recipe"] != ANCHORS["blend"]
            or any(blend[k] is not True for k in ("original_selectors_independently_replayed",
                "all_selected_prediction_scores_replayed", "all_original_component_arrays_support_replay",
                "all_original_component_scores_replay"))
            or set(blend["selected_prediction_paths"]) != set(SESSIONS)
            or len(blend["blend"]["sessions"]) != 16):
        raise ValueError("Complete original blend summary and selected arrays required")
    rows["blend"] = {r["session_id"]: r for r in blend["blend"]["sessions"]}
    if set(rows["blend"]) != set(SESSIONS):
        raise ValueError("Every original blend outer session required")
    for s in SESSIONS:
        index, training, _ = select_cell(blend_banks, s)
        old, cell = rows["blend"][s], blend_banks[s]["cells"][index]
        if (old["animal"] != ANIMALS[s] or old["cell_index"] != index or old["outer_training_sessions"] != training
                or any(old[k] != cell[k] for k in ("lambda", "alpha", *SCORE_KEYS))):
            raise ValueError("Original blend selector and selected cell did not replay")
    originals["blend"] = blend
    return banks, rows, blend_banks, sources, originals


def check_saved_support(saved, y, baseline, labels, ids):
    for key, actual in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
        if not np.array_equal(saved[key], actual):
            raise ValueError("Original native saved support differs: "+key)


def validate_prediction_array(values, shape):
    if (values.shape != shape or values.dtype != np.dtype("float64")
            or not values.flags.c_contiguous or not np.isfinite(values).all()):
        raise ValueError("Literal complete finite float64 C-order predictions required")


def release_complete_panel(args, sessions):
    banks, rows, blend_banks, sources, originals = load_original_context(args)
    result = summarize_banks({s: sessions[s]["banks"] for s in SESSIONS}, rows, banks)
    selected_rows = {r["session_id"]: r for r in result["candidate"]["sessions"]}
    selected_dir = args.out/"selected_predictions"
    selected_dir.mkdir(exist_ok=False)
    candidate_flags, control_flags, paths = {}, {}, {}
    for s in SESSIONS:
        item, selected = sessions[s], selected_rows[s]
        with np.load(args.bank_root/s/(RECIPES[0]+"_predictions.npz"), allow_pickle=False) as saved:
            y, baseline = restore_target_layout(saved["actual_xy"]), restore_target_layout(saved["baseline_xy"])
            labels, ids = saved["directions"], saved["trial_ids"]
            candidate, controls = saved["candidate_xy_by_lambda"], saved["control_xy_by_replicate_lambda"]
        validate_prediction_array(candidate, (5, *y.shape))
        validate_prediction_array(controls, (8, 5, *y.shape))
        if (ids.tolist() != item["banks"][RECIPES[0]]["original_trial_ids"]
                or labels.tolist() != item["banks"][RECIPES[0]]["original_direction_labels"]):
            raise ValueError("New saved native trial identities changed")
        replay = score_bank(y, labels, baseline, candidate, RECIPES[0])
        if any(replay[k] != item["banks"][RECIPES[0]][k] for k in (*SUPPORT_KEYS, "lambda_rows")):
            raise ValueError("Exact new candidate saved score replay failed")
        for r in range(8):
            replay = score_bank(y, labels, baseline, controls[r], RECIPES[1])
            if replay["lambda_rows"] != item["banks"][RECIPES[1]]["surrogate_seed_rows"][r]["lambda_rows"]:
                raise ValueError("Exact new control saved score replay failed")
        anchor_arrays = {}
        for name, root in (("HW", args.hw_bank_root), ("K", args.k_bank_root)):
            source = sources[name][s]
            if (source["session_id"] != s or source["native_bin_seconds"] != .03
                    or source["complete_trials"] != len(y)
                    or source["common_LMP_electrodes"] != item["source"]["common_LMP_electrodes"]
                    or (name == "K" and source["native_offsets"] != [0, 14])):
                raise ValueError("Original anchor physical/target source support differs")
            with np.load(root/s/(ANCHORS[name]+"_predictions.npz"), allow_pickle=False) as saved:
                check_saved_support(saved, y, baseline, labels, ids)
                old = saved["candidate_xy_by_lambda"]
            validate_prediction_array(old, (5, *y.shape))
            replay = score_bank(y, labels, baseline, old, ANCHORS[name])
            if any(replay[k] != banks[name][s][k] for k in (*SUPPORT_KEYS, "lambda_rows")):
                raise ValueError("Original anchor all-penalty score replay failed: "+name)
            if any(replay[k] != item["banks"][RECIPES[0]][k] for k in SUPPORT_KEYS):
                raise ValueError("Original and new score support differs")
            anchor_arrays[name] = old[rows[name][s]["lambda_index"]]
        with np.load(originals["blend"]["selected_prediction_paths"][s], allow_pickle=False) as saved:
            check_saved_support(saved, y, baseline, labels, ids)
            blend_array = saved["candidate_xy"]
            if (float(saved["selected_lambda"]) != rows["blend"][s]["lambda"]
                    or float(saved["selected_alpha"]) != rows["blend"][s]["alpha"]):
                raise ValueError("Original selected blend labels changed")
        validate_prediction_array(blend_array, y.shape)
        if any(prediction_scores(y, baseline, labels, blend_array)[k] != rows["blend"][s][k] for k in SCORE_KEYS):
            raise ValueError("Original selected blend score did not replay")
        if any(blend_banks[s][k] != item["banks"][RECIPES[0]][k] for k in SUPPORT_KEYS if k != "lambdas"):
            raise ValueError("Original blend scoring support differs")
        anchor_arrays.update(blend=blend_array, baseline=baseline)
        chosen, nulls = candidate[selected["lambda_index"]], controls[:, selected["lambda_index"]]
        candidate_flags[s] = {n: bool(np.array_equal(chosen, a)) for n, a in anchor_arrays.items()}
        control_flags[s] = {n: [bool(np.array_equal(p, a)) for p in nulls]
                            for n, a in {**anchor_arrays, "candidate": chosen}.items()}
        path = selected_dir/(s+".npz")
        np.savez(path, actual_xy=y, baseline_xy=baseline, directions=labels, trial_ids=ids,
            candidate_xy=chosen, control_xy_by_replicate=nulls, selected_lambda=selected["lambda"])
        paths[s] = str(path)
    result.update(duplicate_accounting(candidate_flags, control_flags), selected_prediction_paths=paths,
        all_new_saved_scores_replayed=True, all_original_anchor_scores_replayed=True,
        original_selectors_independently_replayed=True, bank_root=str(args.bank_root))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    session = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root", "out"):
        session.add_argument("--"+name, type=Path, required=True)
    session.add_argument("--session", choices=SESSIONS, required=True)
    session.add_argument("--max-seconds", type=int, default=600)
    summary = sub.add_parser("summarize")
    for name in (*SUMMARY_OPTIONS, "out"):
        summary.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 600:
        raise ValueError("Adopted complete-session cap is at most 600 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False,
        "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Adopted HW105 complete-session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            x, y, labels, records, source = load_lmp_history_session(args.source, args.trial_table,
                args.scalar_bank_root, args.session, check)
            ids = np.asarray([row["trial_id"] for row in records])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            fitting = time.monotonic()
            banks, baseline, candidate, controls = fit_banks(x, y, labels, ids, args.session, check,
                electrode_ids=source["common_LMP_electrodes"])
            report["candidate_and_control_fit_score_seconds"] = time.monotonic()-fitting
            replaying = time.monotonic()
            check_anchor_after_fit(args.hw_bank_root, args.session, source, y, labels, ids, baseline, banks[RECIPES[0]])
            old = read_json(args.hw_bank_root/args.session/"score_banks.json")["banks"][ANCHORS["HW"]]
            for new, previous in zip(banks[RECIPES[0]]["direction_fit_metadata"], old["direction_fit_metadata"]):
                if new["paired_weighted_covariance"] != previous["paired_weighted_covariance"]:
                    raise ValueError("Original HW weighted covariance did not replay exactly")
            report.update(exact_original_HW_neural_state_replay=True,
                exact_original_HW_saved_arrays_and_scoring_replay=True,
                exact_original_HW_weighted_covariance_replay=True,
                original_HW_replay_seconds=time.monotonic()-replaying)
            check()
            writing = time.monotonic()
            np.savez(args.out/(RECIPES[0]+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                directions=labels, trial_ids=ids, candidate_xy_by_lambda=candidate,
                control_xy_by_replicate_lambda=controls)
            report.update(status=SESSION_STATUS, banks=banks, prediction_write_seconds=time.monotonic()-writing)
            validate_session(report, args.session)
        else:
            sessions = {}
            for s in SESSIONS:
                item = read_json(args.bank_root/s/"score_banks.json")
                validate_session(item, s)
                sessions[s] = item
            report.update(release_complete_panel(args, sessions), status=PANEL_STATUS)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        name = "score_banks.json" if args.mode == "session" else SUMMARY_FILE
        (args.out/name).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "maximum_RSS_KiB")}), flush=True)


if __name__ == "__main__":
    main()
