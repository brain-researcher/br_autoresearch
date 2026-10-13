#!/usr/bin/env python3
"""Bounded stored-projection/true-head fitting and complete-panel release."""
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
from primary_capacity_matched_surrogate import average_banks
from primary_covariance_svd import RECIPES as SVD_RECIPES, numeric_trial_ids, trial_map
from primary_history_blend import prediction_scores, validate_original_selections
from primary_history_pls import ANCHORS, RECIPES as PLS_RECIPES, REPLICATES, SCORE_KEYS, SUPPORT_KEYS
from primary_history_pls_basis import (ENSEMBLE_REFERENCES, RECIPE, SINGLE_REFERENCES,
    duplicate_accounting, fit_banks, summarize_banks)
from primary_reference import LAMBDAS, SESSIONS
from run_primary_elapsed_only import check_anchor_after_fit
from run_primary_history_blend import restore_target_layout
from run_primary_history_pls import (PANEL_STATUS as PLS_PANEL_STATUS, SUMMARY_OPTIONS as PLS_SUMMARY_OPTIONS,
    load_references, validate_score_covariance, validate_session as validate_pls_session)
from run_primary_history_rbf import check_saved_support, read_json, validate_prediction_array
from source_adapter import compute_only

SESSION_STATUS = "complete_history_pls_basis_session_bank"
PANEL_STATUS = "complete_history_pls_basis_panel"
SUMMARY_FILE = "primary_history_pls_basis.json"
SUMMARY_OPTIONS = (*PLS_SUMMARY_OPTIONS, "pls_bank_root", "pls_summary")
REPLAY_FLAGS = ("all_new_saved_scores_replayed", "all_original_anchor_scores_replayed",
                "original_selectors_independently_replayed", "all_stored_transform_heads_replayed")


def validate_score_bank(bank):
    if (bank["recipe"] != RECIPE or bank["lambdas"] != list(LAMBDAS)
            or bank["surrogate_replicate_ids"] != list(REPLICATES)
            or [row["replicate"] for row in bank["surrogate_seed_rows"]] != list(REPLICATES)
            or not np.isfinite(bank["denominator"]) or bank["denominator"] <= 0):
        raise ValueError("Complete native-target eight-draw basis-only bank required")
    for rows in [bank["lambda_rows"], *[row["lambda_rows"] for row in bank["surrogate_seed_rows"]]]:
        if ([row["lambda"] for row in rows] != list(LAMBDAS)
                or not all(np.isfinite(row[key]) for row in rows for key in SCORE_KEYS)):
            raise ValueError("All five finite scores for each of eight true-head draws required")
    seeds = [{**{key: bank[key] for key in ("recipe", *SUPPORT_KEYS)}, "lambda_rows": row["lambda_rows"]}
             for row in bank["surrogate_seed_rows"]]
    expected = average_banks(seeds)
    if expected["lambda_rows"] != bank["lambda_rows"]:
        raise ValueError("Mean diagnostic must average correct/wrong SSE, not predictions")


def validate_session(item, session):
    if (item["status"] != SESSION_STATUS or item["session_id"] != session
            or item["scientific_trial_count_increment"] != 0 or item["literal_null_control_count_increment"] != 0
            or item["alternative_explanation_count_increment"] != 0
            or item["promotion_authorized"] is not False or item["original_HW_refitted"] is not False
            or any(item[key] is not True for key in ("exact_original_HW_neural_state_replay",
                "exact_original_HW_saved_arrays_and_scoring_replay", "exact_original_HW_weighted_covariance_replay",
                "original_paired_and_full_target_heads_replayed", "original_PLS_saved_scores_replayed"))):
        raise ValueError("Complete uncharged stored-transform session and all original replay required")
    bank, source = item["bank"], item["source"]
    validate_score_bank(bank)
    ids, labels = np.asarray(bank["original_trial_ids"]), np.asarray(bank["original_direction_labels"])
    numeric_trial_ids(ids, session)
    channels = source["common_LMP_electrodes"]
    if (source["session_id"] != session or source["native_bin_seconds"] != .03
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != [0, -1, -2, -3] or not channels or channels != sorted(set(channels))
            or any(type(channel) is not int or channel < 1 for channel in channels)
            or source["complete_trials"] != len(ids) or labels.shape != ids.shape
            or set(labels.tolist()) != set(range(8))
            or bank["direction_counts"] != [int(np.count_nonzero(labels == direction)) for direction in range(8)]
            or any(n < 2 for n in bank["direction_counts"])
            or bank["physical_electrode_ids"] != channels or bank["physical_electrode_count"] != len(channels)
            or bank["history_column_count"] != 4*len(channels) or bank["component_cap"] != 2
            or bank["lag_offsets"] != [0, -1, -2, -3] or bank["positive_floor"] != .1
            or any(bank[key] is not False for key in ("native_XY_scaled", "weights_use_velocity", "post_weight_standardization",
                "post_weight_centering", "score_standardization", "original_HW_refitted", "candidate_and_basis_capacity_matched"))
            or bank["new_projection_fits"] != 0 or bank["stored_transforms_reused"] != 72
            or bank["ridge_heads_use_true_targets"] is not True
            or bank["target_shuffle_scope"] != "stored_projection_only"
            or bank["lambda_selection"] != "preserved_PLS_candidate_outer_session_selection_only"):
        raise ValueError("Fixed original HW support and stored PLS-only head change required")
    metadata = bank["direction_fit_metadata"]
    if [row["held_direction"] for row in metadata] != list(range(8)):
        raise ValueError("All eight original held-direction fits required")
    for row in metadata:
        held, details = row["held_direction"], row["basis_diagnostic"]
        train = labels != held
        if (row["fitting_bin_rows"] != 15*int(train.sum())
                or row["target_trial_maps"] != [trial_map(ids[train], labels[train], session, held, r)[1] for r in REPLICATES]
                or row["original_paired_and_full_target_heads_replayed"] is not True
                or details["stored_transforms_reused"] != 9 or details["new_projection_fits"] != 0
                or details["basis_and_full_target_score_Grams_identical"] is not True
                or details["candidate_and_basis_capacity_matched"] is not False
                or len(details["basis_components"]) != 8 or len(details["basis_score_covariances"]) != 8):
            raise ValueError("Eight fixed maps and literal stored projection reuse required")
        for k, covariance in zip([details["paired_candidate_components"], *details["basis_components"]],
                [details["paired_candidate_score_covariance"], *details["basis_score_covariances"]]):
            if type(k) is not int or k not in (0, 1, 2):
                raise ValueError("Original zero/one/two component states required")
            validate_score_covariance(covariance, {"retained_components": k})


def replay_scores(values, y, baseline, labels, bank, recipe, seed_key=None):
    if seed_key is None:
        replay = score_bank(y, labels, baseline, values, recipe)
        if any(replay[key] != bank[key] for key in (*SUPPORT_KEYS, "lambda_rows")):
            raise ValueError("Saved complete score bank did not replay: "+recipe)
    else:
        if [row["replicate"] for row in bank[seed_key]] != list(REPLICATES):
            raise ValueError("All eight original seed rows required")
        seed_banks = []
        for r in REPLICATES:
            replay = score_bank(y, labels, baseline, values[r], recipe)
            if (any(replay[key] != bank[key] for key in SUPPORT_KEYS)
                    or replay["lambda_rows"] != bank[seed_key][r]["lambda_rows"]):
                raise ValueError("Saved individual draw scores did not replay: "+recipe)
            seed_banks.append(replay)
        if average_banks(seed_banks)["lambda_rows"] != bank["lambda_rows"]:
            raise ValueError("Saved mean-loss panel did not replay: "+recipe)


def original_pls_context(args):
    old = {}
    for session in SESSIONS:
        item = read_json(args.pls_bank_root/session/"score_banks.json")
        validate_pls_session(item, session)
        old[session] = item
    summary = read_json(args.pls_summary)
    if (summary["status"] != PLS_PANEL_STATUS or summary["promotion_authorized"] is not False
            or any(summary[key] is not True for key in REPLAY_FLAGS[:3])):
        raise ValueError("Complete original replayed PLS panel required")
    p = {s: old[s]["banks"][PLS_RECIPES[0]] for s in SESSIONS}
    f = {s: old[s]["banks"][PLS_RECIPES[1]] for s in SESSIONS}
    selected_p = validate_original_selections(p, summary["candidate"])
    full_rows = summary["control"]["sessions"]
    if len(full_rows) != 16 or {row["session_id"] for row in full_rows} != set(SESSIONS):
        raise ValueError("Complete original full-target control selections required")
    selected_f = {row["session_id"]: row for row in full_rows}
    for s in SESSIONS:
        chosen, row = selected_p[s], selected_f[s]
        # Lambda10 is the frozen completed choice, not a new parameter search.
        if (chosen["lambda"] != 10 or row["lambda_index"] != chosen["lambda_index"]
                or row["outer_training_sessions"] != chosen["outer_training_sessions"]
                or any(row[key] != f[s]["lambda_rows"][chosen["lambda_index"]][key] for key in ("lambda", *SCORE_KEYS))):
            raise ValueError("Original candidate-selected lambda10/full-target rows changed")
    return old, p, f, selected_p, selected_f


def release_complete_panel(args, sessions):
    if set(sessions) != set(SESSIONS):
        raise ValueError("No basis-only effects before all sixteen complete sessions")
    old, p, f, selected_p, selected_f = original_pls_context(args)
    banks, rows, blend_banks, sources, originals = load_references(args)
    result = summarize_banks({s: sessions[s]["bank"] for s in SESSIONS}, p, f, selected_p, selected_f,
                            {name: rows[name] for name in ("HW", "blend")})
    single, ensembles, paths = {}, {}, {}
    selected_dir = args.out/"selected_predictions"
    selected_dir.mkdir(exist_ok=False)
    roots = {"HW": args.hw_bank_root, "K": args.k_bank_root,
             "covariance_SVD": args.svd_bank_root, "HW105": args.rbf_bank_root}
    for session in SESSIONS:
        item, selected = sessions[session], selected_p[session]
        with np.load(args.bank_root/session/(RECIPE+"_predictions.npz"), allow_pickle=False) as saved:
            y, baseline = restore_target_layout(saved["actual_xy"]), restore_target_layout(saved["baseline_xy"])
            labels, ids, new = saved["directions"], saved["trial_ids"], saved["candidate_xy_by_replicate_lambda"]
        validate_prediction_array(new, (8, 5, *y.shape))
        if ids.tolist() != item["bank"]["original_trial_ids"] or labels.tolist() != item["bank"]["original_direction_labels"]:
            raise ValueError("New saved original trial/direction identities changed")
        replay_scores(new, y, baseline, labels, item["bank"], RECIPE, "surrogate_seed_rows")
        with np.load(args.pls_bank_root/session/(PLS_RECIPES[0]+"_predictions.npz"), allow_pickle=False) as saved:
            check_saved_support(saved, y, baseline, labels, ids)
            old_p, old_f = saved["candidate_xy_by_lambda"], saved["control_xy_by_replicate_lambda"]
        validate_prediction_array(old_p, (5, *y.shape))
        validate_prediction_array(old_f, (8, 5, *y.shape))
        replay_scores(old_p, y, baseline, labels, p[session], PLS_RECIPES[0])
        replay_scores(old_f, y, baseline, labels, f[session], PLS_RECIPES[1], "surrogate_seed_rows")
        anchor_arrays = {"baseline": baseline, "PLS": old_p[selected["lambda_index"]]}
        for name, root in roots.items():
            source = sources[name][session]
            if (source["session_id"] != session or source["complete_trials"] != len(y)
                    or source["native_bin_seconds"] != .03
                    or source["common_LMP_electrodes"] != item["source"]["common_LMP_electrodes"]
                    or (name in ("K", "covariance_SVD") and source["native_offsets"] != [0, 14])):
                raise ValueError("Original duplicate-reference support changed")
            with np.load(root/session/(ANCHORS[name]+"_predictions.npz"), allow_pickle=False) as saved:
                check_saved_support(saved, y, baseline, labels, ids)
                values = saved["candidate_xy_by_lambda"]
            validate_prediction_array(values, (5, *y.shape))
            replay_scores(values, y, baseline, labels, banks[name][session], ANCHORS[name])
            anchor_arrays[name] = values[rows[name][session]["lambda_index"]]
        with np.load(originals["blend"]["selected_prediction_paths"][session], allow_pickle=False) as saved:
            check_saved_support(saved, y, baseline, labels, ids)
            blend = saved["candidate_xy"]
            if float(saved["selected_lambda"]) != rows["blend"][session]["lambda"] or float(saved["selected_alpha"]) != rows["blend"][session]["alpha"]:
                raise ValueError("Original blend selected parameters changed")
        validate_prediction_array(blend, y.shape)
        if any(prediction_scores(y, baseline, labels, blend)[key] != rows["blend"][session][key] for key in SCORE_KEYS):
            raise ValueError("Original blend scores did not replay")
        anchor_arrays["blend"] = blend
        svd_item = read_json(args.svd_bank_root/session/"score_banks.json")
        svd_control = svd_item["banks"][SVD_RECIPES[1]]
        with np.load(args.svd_bank_root/session/(SVD_RECIPES[1]+"_predictions.npz"), allow_pickle=False) as saved:
            check_saved_support(saved, y, baseline, labels, ids)
            svd_ensemble = saved["candidate_xy_by_replicate_lambda"]
        validate_prediction_array(svd_ensemble, (8, 5, *y.shape))
        replay_scores(svd_ensemble, y, baseline, labels, svd_control, SVD_RECIPES[1], "basis_label_seed_rows")
        selected_new = new[:, selected["lambda_index"]]
        selected_ensembles = {"PLS_full_target": old_f[:, selected["lambda_index"]],
            "covariance_SVD_basis": svd_ensemble[:, rows["covariance_SVD"][session]["lambda_index"]]}
        single[session] = {name: [bool(np.array_equal(value, anchor)) for value in selected_new]
                           for name, anchor in anchor_arrays.items()}
        ensembles[session] = {name: [bool(np.array_equal(selected_new[r], values[r])) for r in REPLICATES]
                              for name, values in selected_ensembles.items()}
        path = selected_dir/(session+".npz")
        np.savez(path, actual_xy=y, baseline_xy=baseline, directions=labels, trial_ids=ids,
                 candidate_xy_by_replicate=selected_new, selected_lambda=selected["lambda"])
        paths[session] = str(path)
    result.update(duplicate_accounting(single, ensembles), selected_prediction_paths=paths,
        single_duplicate_reference_names=list(SINGLE_REFERENCES), ensemble_duplicate_reference_names=list(ENSEMBLE_REFERENCES),
        **{key: True for key in REPLAY_FLAGS}, bank_root=str(args.bank_root), original_pls_bank_root=str(args.pls_bank_root))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    session = sub.add_parser("session")
    for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root", "pls-bank-root", "out"):
        session.add_argument("--"+name, type=Path, required=True)
    session.add_argument("--session", choices=SESSIONS, required=True)
    session.add_argument("--max-seconds", type=int, default=600)
    summary = sub.add_parser("summarize")
    for name in (*SUMMARY_OPTIONS, "out"):
        summary.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("One numerical thread per worker required")
    if args.mode == "session" and not 0 < args.max_seconds <= 600:
        raise ValueError("Adopted session internal cap is at most 600 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False,
        "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Complete stored-projection session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            old = read_json(args.pls_bank_root/args.session/"score_banks.json")
            validate_pls_session(old, args.session)
            x, y, labels, records, source = load_lmp_history_session(args.source, args.trial_table,
                args.scalar_bank_root, args.session, check)
            ids = np.asarray([row["trial_id"] for row in records])
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-started)
            with np.load(args.pls_bank_root/args.session/(PLS_RECIPES[0]+"_predictions.npz"), allow_pickle=False) as saved:
                old_baseline = restore_target_layout(saved["baseline_xy"])
                check_saved_support(saved, y, old_baseline, labels, ids)
                p, f = saved["candidate_xy_by_lambda"], saved["control_xy_by_replicate_lambda"]
            validate_prediction_array(p, (5, *y.shape))
            validate_prediction_array(f, (8, 5, *y.shape))
            fitting = time.monotonic()
            bank, baseline, predictions = fit_banks(x, y, labels, ids, args.session, old["banks"], p, f, check,
                electrode_ids=source["common_LMP_electrodes"])
            if not np.array_equal(baseline, old_baseline):
                raise ValueError("Original PLS baseline did not replay")
            report["stored_projection_and_heads_seconds"] = time.monotonic()-fitting
            report["old_head_reconstruction_seconds"] = sum(row["basis_diagnostic"]["old_head_reconstruction_seconds"] for row in bank["direction_fit_metadata"])
            report["new_true_head_seconds"] = sum(row["basis_diagnostic"]["new_true_head_seconds"] for row in bank["direction_fit_metadata"])
            replaying = time.monotonic()
            check_anchor_after_fit(args.hw_bank_root, args.session, source, y, labels, ids, baseline, bank)
            original_hw = read_json(args.hw_bank_root/args.session/"score_banks.json")["banks"][ANCHORS["HW"]]
            if any(new["paired_weighted_covariance"] != previous["paired_weighted_covariance"]
                   for new, previous in zip(bank["direction_fit_metadata"], original_hw["direction_fit_metadata"])):
                raise ValueError("Original HW weighted covariance did not replay literally")
            report.update(exact_original_HW_neural_state_replay=True,
                exact_original_HW_saved_arrays_and_scoring_replay=True, exact_original_HW_weighted_covariance_replay=True,
                original_paired_and_full_target_heads_replayed=True, original_PLS_saved_scores_replayed=True,
                original_HW_replay_seconds=time.monotonic()-replaying, original_pls_bank_root=str(args.pls_bank_root))
            check()
            writing = time.monotonic()
            np.savez(args.out/(RECIPE+"_predictions.npz"), actual_xy=y, baseline_xy=baseline,
                directions=labels, trial_ids=ids, candidate_xy_by_replicate_lambda=predictions)
            check()
            report.update(status=SESSION_STATUS, bank=bank, prediction_write_seconds=time.monotonic()-writing)
            validate_session(report, args.session)
            check()
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
