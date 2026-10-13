#!/usr/bin/env python3
"""Bounded adopted PLS2 session fitting and complete-panel release only."""
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
from primary_covariance_svd import numeric_trial_ids, trial_map
from primary_history_blend import prediction_scores, validate_original_selections
from primary_history_pls import (ANCHORS, COMPONENT_CAP, DUPLICATE_ANCHORS, RECIPES, REPLICATES,
                                 SCORE_KEYS, SUPPORT_KEYS, duplicate_accounting, fit_banks,
                                 summarize_banks, validate_pls_state)
from primary_reference import LAMBDAS, SESSIONS
from run_primary_covariance_svd import validate_session as validate_svd_session
from run_primary_elapsed_only import check_anchor_after_fit
from run_primary_history_blend import restore_target_layout
from run_primary_history_rbf import (PANEL_STATUS as HW105_PANEL_STATUS, check_saved_support,
                                   load_original_context as load_hw_k_blend_context,
                                   read_json, validate_prediction_array,
                                   validate_session as validate_hw105_session)
from source_adapter import compute_only

SESSION_STATUS = "complete_history_pls_session_banks"
PANEL_STATUS = "complete_history_pls_panel"
SUMMARY_FILE = "primary_history_pls.json"
SUMMARY_OPTIONS = ("bank_root", "hw_bank_root", "k_bank_root", "blend_bank_root", "svd_bank_root", "rbf_bank_root",
                   "hw_summary", "k_summary", "blend_summary", "svd_summary", "rbf_summary")


def validate_score_covariance(value, state):
    k = state["retained_components"]
    eigenvalues = np.asarray(value["eigenvalues"])
    if (eigenvalues.shape != (k,) or not np.isfinite(eigenvalues).all() or np.any(eigenvalues < 0)
            or not np.isfinite(value["trace"]) or value["trace"] < 0
            or not 0 <= value["numerical_rank"] <= k
            or value["diagnostic_only"] is not True
            or [row["lambda"] for row in value["ridge_df_by_lambda"]] != list(LAMBDAS)
            or any(not np.isfinite(row["effective_df"]) or row["effective_df"] < 0
                   for row in value["ridge_df_by_lambda"])):
        raise ValueError("Complete diagnostic-only PLS score covariance required")


def validate_session(item, session):
    if (item["status"] != SESSION_STATUS or item["session_id"] != session
            or item["exact_original_HW_neural_state_replay"] is not True
            or item["exact_original_HW_saved_arrays_and_scoring_replay"] is not True
            or item["exact_original_HW_weighted_covariance_replay"] is not True
            or item["original_HW_refitted"] is not False or item["promotion_authorized"] is not False
            or item["scientific_trial_count_increment"] != 0 or set(item["banks"]) != set(RECIPES)):
        raise ValueError("Complete uncharged PLS2/control session and exact original HW replay required")
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
        raise ValueError("Complete original native history/target support required")
    for recipe, bank in item["banks"].items():
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or bank["physical_electrode_ids"] != channels or bank["physical_electrode_count"] != len(channels)
                or bank["history_column_count"] != 4*len(channels) or bank["component_cap"] != COMPONENT_CAP
                or bank["original_trial_ids"] != ids.tolist() or bank["original_direction_labels"] != labels.tolist()
                or bank["direction_counts"] != [int(np.count_nonzero(labels == d)) for d in range(8)]
                or any(n < 2 for n in bank["direction_counts"])
                or bank["lag_offsets"] != [0, -1, -2, -3] or bank["native_XY_scaled"] is not False
                or bank["weights_use_velocity"] is not False or bank["positive_floor"] != .1
                or bank["post_weight_standardization"] is not False or bank["post_weight_centering"] is not False
                or bank["score_standardization"] is not False or bank["exact_capacity_match"] is not False
                or bank["target_shuffle_scope"] != "projection_and_head" or bank["supervised_projection_fits"] != 72
                or bank["original_HW_refitted"] is not False
                or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0):
            raise ValueError("Fixed original HW state, native scoring and PLS2 operator required")
        if any(bank[key] != candidate[key] for key in SUPPORT_KEYS):
            raise ValueError("Identical candidate/control native target support required")
        if ([row["lambda"] for row in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(row[key]) for row in bank["lambda_rows"] for key in SCORE_KEYS)):
            raise ValueError("Complete finite original five-penalty score bank required")
    if (control["surrogate_replicate_ids"] != list(REPLICATES)
            or [row["replicate"] for row in control["surrogate_seed_rows"]] != list(REPLICATES)
            or control["direction_fit_metadata"] != candidate["direction_fit_metadata"]):
        raise ValueError("All eight fixed target-label controls with independently learned transforms required")
    for seed in control["surrogate_seed_rows"]:
        if ([row["lambda"] for row in seed["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(row[key]) for row in seed["lambda_rows"] for key in SCORE_KEYS)):
            raise ValueError("Every target-label draw needs all five finite score rows")
    for index in range(len(LAMBDAS)):
        correct = float(np.mean([row["lambda_rows"][index]["candidate_SSE"] for row in control["surrogate_seed_rows"]]))
        wrong = float(np.mean([row["lambda_rows"][index]["wrong_pair_SSE"] for row in control["surrogate_seed_rows"]]))
        denominator = control["denominator"]
        expected = {"candidate_SSE": correct, "wrong_pair_SSE": wrong, "candidate_R2": 1-correct/denominator,
                    "delta_R2": (control["baseline_SSE"]-correct)/denominator,
                    "identity_increment": (wrong-correct)/denominator}
        if any(control["lambda_rows"][index][key] != value for key, value in expected.items()):
            raise ValueError("Mean-control score must use all eight correct/wrong SSEs")
    metadata = candidate["direction_fit_metadata"]
    if [row["held_direction"] for row in metadata] != list(range(8)):
        raise ValueError("All eight held-direction fits are required")
    for row in metadata:
        held, details = row["held_direction"], row["pls"]
        train = labels != held
        maps = [trial_map(ids[train], labels[train], session, held, r) for r in REPLICATES]
        if (row["fitting_bin_rows"] != 15*int(train.sum())
                or row["repeatability"]["fitting_directions"] != sorted(set(range(8))-{held})
                or row["target_trial_maps"] != [records for _, records in maps]
                or details["supervised_projection_fits"] != 9 or details["exact_capacity_match"] is not False
                or details["target_shuffle_scope"] != "projection_and_head"
                or len(details["control_transforms"]) != len(REPLICATES)
                or len(details["control_score_covariances"]) != len(REPLICATES)):
            raise ValueError("Exactly seven fitting directions and eight whole-trial supervised refits required")
        states = [details["candidate_transform"], *details["control_transforms"]]
        diagnostics = [details["candidate_score_covariance"], *details["control_score_covariances"]]
        for state, diagnostic in zip(states, diagnostics):
            validate_pls_state(state, columns=4*len(channels), rows=row["fitting_bin_rows"])
            validate_score_covariance(diagnostic, state)
        for key in ("whole_fitting_map_identity_by_replicate", "control_targets_equal_candidate_by_replicate"):
            if len(details[key]) != len(REPLICATES) or any(type(flag) is not bool for flag in details[key]):
                raise ValueError("All eight explicit target-map identity flags required")
        identity = [bool(np.array_equal(mapping, np.arange(len(mapping)))) for mapping, _ in maps]
        if details["whole_fitting_map_identity_by_replicate"] != identity:
            raise ValueError("Recorded target map identity changed")
        for r in REPLICATES:
            if identity[r] and not details["control_targets_equal_candidate_by_replicate"][r]:
                raise ValueError("Identity target map cannot change the native fitting target")
            if (details["control_targets_equal_candidate_by_replicate"][r]
                    and (states[r+1] != states[0] or diagnostics[r+1] != diagnostics[0])):
                raise ValueError("Identical targets must reproduce the exact PLS map and score covariance")


def load_references(args):
    """Reuse qualified selectors; additional old recipes serve duplicate checks."""
    banks, rows, blend_banks, sources, originals = load_hw_k_blend_context(args)
    for name, root, validator in (("covariance_SVD", args.svd_bank_root, validate_svd_session),
                                  ("HW105", args.rbf_bank_root, validate_hw105_session)):
        banks[name], sources[name] = {}, {}
        for session in SESSIONS:
            item = read_json(root/session/"score_banks.json")
            validator(item, session)
            banks[name][session] = item["banks"][ANCHORS[name]]
            sources[name][session] = item["source"]
    svd = read_json(args.svd_summary)
    matches = [p for p in svd["panels"] if p["recipe"] == ANCHORS["covariance_SVD"]]
    if svd["status"] != "complete_covariance_svd_control_panel" or len(matches) != 1:
        raise ValueError("Complete preserved original covariance-SVD panel required")
    rows["covariance_SVD"] = validate_original_selections(banks["covariance_SVD"], matches[0])
    rbf = read_json(args.rbf_summary)
    if (rbf["status"] != HW105_PANEL_STATUS or rbf["promotion_authorized"] is not False
            or any(rbf[key] is not True for key in ("all_new_saved_scores_replayed",
                "all_original_anchor_scores_replayed", "original_selectors_independently_replayed"))):
        raise ValueError("Complete replayed HW105 panel required")
    rows["HW105"] = validate_original_selections(banks["HW105"], rbf["candidate"])
    return banks, rows, blend_banks, sources, originals


def release_complete_panel(args, sessions):
    if set(sessions) != set(SESSIONS):
        raise ValueError("No scientific release before all sixteen complete sessions")
    banks, rows, blend_banks, sources, originals = load_references(args)
    result = summarize_banks({s: sessions[s]["banks"] for s in SESSIONS}, rows, banks["HW"])
    selected_rows = {row["session_id"]: row for row in result["candidate"]["sessions"]}
    selected_dir = args.out/"selected_predictions"
    selected_dir.mkdir(exist_ok=False)
    candidate_flags, control_flags, paths = {}, {}, {}
    roots = {"HW": args.hw_bank_root, "K": args.k_bank_root,
             "covariance_SVD": args.svd_bank_root, "HW105": args.rbf_bank_root}
    for session in SESSIONS:
        item, selected = sessions[session], selected_rows[session]
        with np.load(args.bank_root/session/(RECIPES[0]+"_predictions.npz"), allow_pickle=False) as saved:
            # Original saved Y/B require their time-fast reduction layout;
            # model prediction arrays remain literal C-order as originally fit.
            y, baseline = restore_target_layout(saved["actual_xy"]), restore_target_layout(saved["baseline_xy"])
            labels, ids = saved["directions"], saved["trial_ids"]
            candidate, controls = saved["candidate_xy_by_lambda"], saved["control_xy_by_replicate_lambda"]
        validate_prediction_array(candidate, (len(LAMBDAS), *y.shape))
        validate_prediction_array(controls, (len(REPLICATES), len(LAMBDAS), *y.shape))
        if (ids.tolist() != item["banks"][RECIPES[0]]["original_trial_ids"]
                or labels.tolist() != item["banks"][RECIPES[0]]["original_direction_labels"]):
            raise ValueError("New saved native trial identities changed")
        replay = score_bank(y, labels, baseline, candidate, RECIPES[0])
        if any(replay[key] != item["banks"][RECIPES[0]][key] for key in (*SUPPORT_KEYS, "lambda_rows")):
            raise ValueError("Exact new PLS candidate saved scores did not replay")
        for r in REPLICATES:
            replay = score_bank(y, labels, baseline, controls[r], RECIPES[1])
            if (any(replay[key] != item["banks"][RECIPES[1]][key] for key in SUPPORT_KEYS)
                    or replay["lambda_rows"] != item["banks"][RECIPES[1]]["surrogate_seed_rows"][r]["lambda_rows"]):
                raise ValueError("Exact new target-label saved scores did not replay")
        anchor_arrays = {}
        for name, root in roots.items():
            source = sources[name][session]
            if (source["session_id"] != session or source["native_bin_seconds"] != .03
                    or source["complete_trials"] != len(y)
                    or source["common_LMP_electrodes"] != item["source"]["common_LMP_electrodes"]
                    or (name in ("K", "covariance_SVD") and source["native_offsets"] != [0, 14])):
                raise ValueError("Original reference physical and target support differs")
            with np.load(root/session/(ANCHORS[name]+"_predictions.npz"), allow_pickle=False) as saved:
                check_saved_support(saved, y, baseline, labels, ids)
                old = saved["candidate_xy_by_lambda"]
            validate_prediction_array(old, (len(LAMBDAS), *y.shape))
            replay = score_bank(y, labels, baseline, old, ANCHORS[name])
            if any(replay[key] != banks[name][session][key] for key in (*SUPPORT_KEYS, "lambda_rows")):
                raise ValueError("Original all-penalty reference replay failed: "+name)
            if any(replay[key] != item["banks"][RECIPES[0]][key] for key in SUPPORT_KEYS):
                raise ValueError("Original/new native score support differs")
            anchor_arrays[name] = old[rows[name][session]["lambda_index"]]
        with np.load(originals["blend"]["selected_prediction_paths"][session], allow_pickle=False) as saved:
            check_saved_support(saved, y, baseline, labels, ids)
            blend_array = saved["candidate_xy"]
            if (float(saved["selected_lambda"]) != rows["blend"][session]["lambda"]
                    or float(saved["selected_alpha"]) != rows["blend"][session]["alpha"]):
                raise ValueError("Original selected blend labels changed")
        validate_prediction_array(blend_array, y.shape)
        if any(prediction_scores(y, baseline, labels, blend_array)[key] != rows["blend"][session][key] for key in SCORE_KEYS):
            raise ValueError("Original selected blend scores did not replay")
        if any(blend_banks[session][key] != item["banks"][RECIPES[0]][key]
               for key in SUPPORT_KEYS if key != "lambdas"):
            raise ValueError("Original blend native scoring support differs")
        anchor_arrays.update(blend=blend_array, baseline=baseline)
        chosen, nulls = candidate[selected["lambda_index"]], controls[:, selected["lambda_index"]]
        candidate_flags[session] = {name: bool(np.array_equal(chosen, values)) for name, values in anchor_arrays.items()}
        control_flags[session] = {name: [bool(np.array_equal(prediction, values)) for prediction in nulls]
                                  for name, values in {**anchor_arrays, "candidate": chosen}.items()}
        path = selected_dir/(session+".npz")
        np.savez(path, actual_xy=y, baseline_xy=baseline, directions=labels, trial_ids=ids,
                 candidate_xy=chosen, control_xy_by_replicate=nulls, selected_lambda=selected["lambda"])
        paths[session] = str(path)
    result.update(duplicate_accounting(candidate_flags, control_flags), selected_prediction_paths=paths,
        all_new_saved_scores_replayed=True, all_original_anchor_scores_replayed=True,
        original_selectors_independently_replayed=True, bank_root=str(args.bank_root),
        duplicate_reference_names=list(DUPLICATE_ANCHORS),
        duplicate_only_reference_panels=["HW105", "covariance_SVD", "K"])
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
    if any(os.environ.get(key) != "1" for key in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 600:
        raise ValueError("Adopted complete PLS session cap is at most 600 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False,
        "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Adopted PLS2 complete-session deadline reached")
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
            metadata = banks[RECIPES[0]]["direction_fit_metadata"]
            report["projection_fit_seconds"] = sum(row["pls"]["transform_seconds"] for row in metadata)
            report["ridge_head_seconds"] = sum(row["pls"]["ridge_head_seconds"] for row in metadata)
            replaying = time.monotonic()
            check_anchor_after_fit(args.hw_bank_root, args.session, source, y, labels, ids, baseline, banks[RECIPES[0]])
            old = read_json(args.hw_bank_root/args.session/"score_banks.json")["banks"][ANCHORS["HW"]]
            for new, previous in zip(metadata, old["direction_fit_metadata"]):
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
        filename = "score_banks.json" if args.mode == "session" else SUMMARY_FILE
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "maximum_RSS_KiB")}), flush=True)


if __name__ == "__main__":
    main()
