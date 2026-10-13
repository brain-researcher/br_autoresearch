#!/usr/bin/env python3
"""Run the fixed HW C/T/CT resource diagnostic and complete-panel summary."""
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
from primary_matched_budget import (ELECTRODE_BUDGET, RECIPES, REFERENCE, TRIAL_BUDGET, VIEWS,
    finalize_banks, fit_reduced, metadata_direction_counts, summarize_banks, support_views, validate_global_metadata)
from primary_reference import LAMBDAS, RECIPE as SCALAR_RECIPE, SESSIONS
from run_primary_history_repeatability_weight import validate_session as validate_HW_session
from run_primary_reference import trial_records
from source_adapter import compute_only


def qualify_global_support(metadata_path, table, scalar_root):
    metadata = json.loads(metadata_path.read_text())
    records = {session: trial_records(table, session) for session in SESSIONS}
    plans = validate_global_metadata(metadata, records)
    original = {row["session_id"]: row for row in metadata["sessions"]}
    for session in SESSIONS:
        scalar = json.loads((scalar_root/session/"score_bank.json").read_text())
        row = original[session]
        if (scalar["status"] != "complete_session_score_bank" or scalar["recipe"] != SCALAR_RECIPE
                or scalar["session_id"] != session or scalar["source"]["session_id"] != session
                or scalar["source"]["common_LMP_electrodes"] != row["physical_LMP_electrode_ids"]
                or scalar["source"]["complete_trials"] != row["complete_trials"]
                or scalar["direction_counts"] != metadata_direction_counts(row)):
            raise ValueError("Global released metadata differs from original scalar source support: "+session)
    return plans, original, records


def load_reference_after_fit(root, session, source):
    item = json.loads((root/session/"score_banks.json").read_text())
    validate_HW_session(item, session)
    for key in ("common_LMP_electrodes", "complete_trials", "native_LMP_offsets", "native_target_offsets",
                "native_bin_seconds", "lag_offsets"):
        if item["source"][key] != source[key]:
            raise ValueError("Original complete HW source support differs: "+key)
    path = root/session/(REFERENCE+"_predictions.npz")
    with np.load(path, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in ("actual_xy", "baseline_xy", "directions", "trial_ids", "candidate_xy_by_lambda")}
    reference["fold_metadata"] = item["banks"][REFERENCE]["direction_fit_metadata"]
    return reference, str(path)


def validate_session(item, session):
    if (item["status"] != "complete_three_matched_budget_session_banks" or item["session_id"] != session
            or item["global_metadata_qualified_before_source"] is not True
            or item["exact_original_HW_target_trial_source_replay"] is not True
            or item["sensitivity_only"] is not True or item["promotion_authorized"] is not False
            or item["matched_electrode_budget"] != ELECTRODE_BUDGET or item["matched_direction_trial_budget"] != TRIAL_BUDGET
            or set(item["banks"]) != set(RECIPES)):
        raise ValueError("Complete fixed matched-resource banks and qualified support required")
    source = item["source"]
    ids, labels = np.asarray(item["original_trial_ids"]), np.asarray(item["original_trial_directions"])
    electrodes = source["common_LMP_electrodes"]
    if (source["session_id"] != session or source["native_LMP_offsets"] != [-3, 14]
            or source["native_target_offsets"] != [0, 14] or source["lag_offsets"] != [0, -1, -2, -3]
            or source["native_bin_seconds"] != .03 or source["spike_values_read"] is not False
            or ids.shape != (source["complete_trials"],) or labels.shape != ids.shape):
        raise ValueError("Original complete native history/XY source support required")
    planned = support_views(ids, labels, electrodes, session)
    for recipe in RECIPES:
        bank = item["banks"][recipe]
        if (bank["recipe"] != recipe or bank["lambdas"] != list(LAMBDAS)
                or [view["view"] for view in bank["views"]] != list(VIEWS)
                or [row["lambda"] for row in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(row[key]) for row in bank["lambda_rows"]
                    for key in ("delta_R2", "identity_increment", "candidate_normalized_SSE", "wrong_pair_normalized_SSE"))):
            raise ValueError("Eight complete views and all original penalty scores required")
        for view, support in zip(bank["views"], planned[recipe]):
            if view["support"] != support:
                raise ValueError("Exact metadata-only support queues and original ordering required")
            key = view["fit_key"]
            if (key == "full_HW") != support["full_support"]:
                raise ValueError("Only exact full support can reuse original HW")
            if key != "full_HW":
                fit = item["new_fit_metadata"][key]
                if fit["trial_ids"] != support["trial_ids"] or fit["electrode_ids"] != support["electrode_ids"]:
                    raise ValueError("Exact within-attempt support reuse required")
            counts = [int(np.count_nonzero(labels[support["trial_positions"]] == d)) for d in range(8)]
            if (view["bank"]["direction_counts"] != counts
                    or view["bank"]["recipe"] != recipe or view["reference_bank"]["recipe"] != REFERENCE
                    or view["reference_own_baseline_bank"]["recipe"] != REFERENCE
                    or [d["held_direction"] for d in view["fit_diagnostics"]] != list(range(8))):
                raise ValueError("Retained whole-direction support and correct prediction roles required")
            for scored in (view["bank"], view["reference_bank"], view["reference_own_baseline_bank"]):
                if (scored["direction_counts"] != counts or scored["denominator"] != view["bank"]["denominator"]
                        or scored["denominator"] <= 0 or not np.isfinite(scored["denominator"])
                        or scored["lambdas"] != list(LAMBDAS)
                        or not all(np.isfinite(row[k]) for row in scored["lambda_rows"]
                            for k in ("candidate_SSE", "wrong_pair_SSE", "delta_R2", "identity_increment"))):
                    raise ValueError("Reference and reduced predictions need the same actual per-view denominator")
            if view["bank"]["baseline_SSE"] != view["reference_bank"]["baseline_SSE"]:
                raise ValueError("Reference comparison must use reduced baseline for the common contrast")
            if view["baseline_change"] != (view["bank"]["baseline_SSE"]-view["reference_own_baseline_bank"]["baseline_SSE"])/view["bank"]["denominator"]:
                raise ValueError("Exact separate baseline-change contribution required")
            for fold in view["fit_diagnostics"]:
                if (fold["fitting_bin_rows"] != 15*(sum(counts)-counts[fold["held_direction"]])
                        or not np.isfinite(fold["trace"]) or fold["trace"] < 0
                        or [r["lambda"] for r in fold["ridge_df_by_lambda"]] != list(LAMBDAS)):
                    raise ValueError("Reduced fitting-row support and full original-penalty diagnostics required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("session")
    for name in ("source", "trial-table", "metadata", "scalar-bank-root", "hw-bank-root", "out"):
        fit.add_argument("--"+name, type=Path, required=True)
    fit.add_argument("--session", choices=SESSIONS, required=True)
    fit.add_argument("--max-seconds", type=int, default=270)
    summary = sub.add_parser("summarize")
    for name in ("bank-root", "hw-summary", "out"):
        summary.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted complete matched-budget session bound is at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "recipes": list(RECIPES),
        "sensitivity_only": True, "scientific_trial_count_increment": 0, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Selected complete matched-budget session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            plans, metadata, original_rows = qualify_global_support(args.metadata, args.trial_table, args.scalar_bank_root)
            report.update(global_metadata_qualified_before_source=True, support_metadata_path=str(args.metadata),
                matched_electrode_budget=ELECTRODE_BUDGET, matched_direction_trial_budget=TRIAL_BUDGET,
                global_metadata_seconds=time.monotonic()-started)
            check()
            reading = time.monotonic()
            x, y, labels, rows, source = load_lmp_history_session(args.source, args.trial_table, args.scalar_bank_root, args.session, check)
            ids = np.array([row["trial_id"] for row in rows])
            if (rows != original_rows[args.session] or source["common_LMP_electrodes"] != metadata[args.session]["physical_LMP_electrode_ids"]
                    or source["complete_trials"] != metadata[args.session]["complete_trials"]):
                raise ValueError("Source support changed after global queue qualification")
            report.update(session_id=args.session, source=source, source_read_seconds=time.monotonic()-reading,
                original_trial_ids=ids.tolist(), original_trial_directions=labels.tolist())
            fitting = time.monotonic()
            supports = plans[args.session]
            fits, routes = fit_reduced(x, y, labels, ids, supports, check)
            report["reduced_fit_seconds"] = time.monotonic()-fitting
            reference, reference_path = load_reference_after_fit(args.hw_bank_root, args.session, source)
            check()
            banks, payload = finalize_banks(y, labels, ids, supports, fits, routes, reference)
            report.update(exact_original_HW_target_trial_source_replay=True, HW_prediction_reference_path=reference_path,
                reference_refitted=False, new_unique_fit_count=len(fits),
                exact_full_HW_view_reuses=sum(k == "full_HW" for keys in routes.values() for k in keys),
                new_fit_metadata={key: {k: fit[k] for k in ("metadata", "trial_ids", "electrode_ids")} for key, fit in fits.items()},
                banks=banks, three_recipe_fit_and_score_seconds=time.monotonic()-fitting,
                status="complete_three_matched_budget_session_banks")
            validate_session(report, args.session)
            check()
            np.savez(args.out/"new_fitted_predictions.npz", actual_xy=y, directions=labels, trial_ids=ids, **payload)
        else:
            sessions = {}
            for session in SESSIONS:
                check()
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            original = json.loads(args.hw_summary.read_text())
            panels = [p for p in original["panels"] if p["recipe"] == REFERENCE]
            if original["status"] != "complete_history_repeatability_weight_control_panel" or len(panels) != 1:
                raise ValueError("Original complete HW panel required")
            report.update(summarize_banks(sessions, panels[0]))
            report.update(status="complete_matched_budget_sensitivity_panel", bank_root=str(args.bank_root),
                original_HW_summary_path=str(args.hw_summary), reference_refitted=False)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_matched_budget.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
