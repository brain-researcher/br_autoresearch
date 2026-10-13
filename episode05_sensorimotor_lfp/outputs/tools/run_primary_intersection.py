#!/usr/bin/env python3
"""Fit fixed guide-intersection HW supports after complete metadata qualification."""
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
from primary_intersection import (RECIPES, REFERENCE, average_banks, finalize_banks, fit_supports,
                                  summarize_banks, validate_description, validate_metadata)
from primary_reference import LAMBDAS, SESSIONS
from qualify_primary_intersection import original_metadata
from run_primary_history_blend import restore_target_layout
from run_primary_history_repeatability_weight import validate_session as validate_HW_session
from run_primary_reference import trial_records
from source_adapter import compute_only


def qualify_global(metadata_path, table, scalar_root, hw_root):
    metadata = json.loads(metadata_path.read_text())
    records = {s: trial_records(table, s) for s in SESSIONS}
    descriptions = validate_metadata(metadata, records)
    for session in SESSIONS:
        scalar = json.loads((scalar_root/session/"score_bank.json").read_text())
        history = json.loads((hw_root/session/"score_banks.json").read_text())
        expected = original_metadata(scalar, history, session, records[session])
        if any(descriptions[session][key] != value for key, value in expected.items()):
            raise ValueError("Qualified intersection metadata differs from original scalar/HW identities")
    return metadata, descriptions, records


def load_reference(root, session, description, source=None):
    item = json.loads((root/session/"score_banks.json").read_text())
    validate_HW_session(item, session)
    old_source = item["source"]
    if (old_source["common_LMP_electrodes"] != description["original_electrode_ids"]
            or old_source["complete_trials"] != description["complete_trials"]):
        raise ValueError("Original HW physical/trial support must match the qualified guide join")
    if source is not None:
        for key in ("common_LMP_electrodes", "complete_trials", "native_LMP_offsets", "native_target_offsets",
                    "native_bin_seconds", "lag_offsets"):
            if source[key] != old_source[key]:
                raise ValueError("Original complete HW source support differs: "+key)
    path = root/session/(REFERENCE+"_predictions.npz")
    with np.load(path, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in
                     ("actual_xy", "baseline_xy", "directions", "trial_ids", "candidate_xy_by_lambda")}
    reference.update(fold_metadata=item["banks"][REFERENCE]["direction_fit_metadata"],
                     score_bank=item["banks"][REFERENCE])
    return reference, old_source, str(path)


def validate_session(item, session):
    if (item["status"] != "complete_intersection_sensitivity_session_banks" or item["session_id"] != session
            or item["global_metadata_qualified_before_source"] is not True
            or item["exact_original_HW_target_baseline_trial_score_replay"] is not True
            or item["reference_refitted"] is not False or item["sensitivity_only"] is not True
            or item["promotion_authorized"] is not False or set(item["banks"]) != set(RECIPES)):
        raise ValueError("Complete guide-defined support sensitivity and exact original HW replay required")
    description, source = item["qualified_support"], item["source"]
    records = [{"trial_id": i, "source_trial_index_zero_based": t, "direction": d}
        for i, t, d in zip(description["trial_ids"], description["source_trial_indices_zero_based"], description["trial_directions"])]
    validate_description(description, records)
    if (source["session_id"] != session or source["common_LMP_electrodes"] != description["original_electrode_ids"]
            or source["complete_trials"] != description["complete_trials"]
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["lag_offsets"] != [0, -1, -2, -3] or source["native_bin_seconds"] != .03
            or source["spike_values_read"] is not False):
        raise ValueError("Unchanged original source windows, physical IDs and trials required")
    views = item["banks"][RECIPES[0]]["views"]+item["banks"][RECIPES[1]]["views"]
    if (len(item["banks"][RECIPES[0]]["views"]) != 1 or len(item["banks"][RECIPES[1]]["views"]) != 8
            or [v["support"] for v in views] != description["support_sets"]):
        raise ValueError("Intersection then eight immutable ordinary supports required")
    full = item["banks"][RECIPES[0]]["full_HW_bank"]
    if item["banks"][RECIPES[1]]["full_HW_bank"] != full:
        raise ValueError("Both recipes must use identical saved full-HW reference")
    for bank in (full, *item["banks"].values(), *(v["bank"] for v in views)):
        if (bank["lambdas"] != list(LAMBDAS) or bank["direction_counts"] != description["direction_counts"]
                or bank["denominator"] != full["denominator"] or not np.isfinite(bank["denominator"])
                or bank["denominator"] <= 0 or bank["baseline_SSE"] != full["baseline_SSE"]
                or bank["baseline_R2"] != full["baseline_R2"]
                or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)
                or not all(np.isfinite(row[k]) for row in bank["lambda_rows"] for k in
                           ("candidate_SSE", "wrong_pair_SSE", "delta_R2", "identity_increment"))):
            raise ValueError("Original B/D/direction support and complete finite five-penalty rows required")
    averaged = average_banks([v["bank"] for v in views[1:]])
    if (item["banks"][RECIPES[0]]["lambda_rows"] != views[0]["bank"]["lambda_rows"]
            or item["banks"][RECIPES[1]]["lambda_rows"] != averaged["lambda_rows"]
            or item["banks"][RECIPES[1]]["support_seed_rows"] != averaged["support_seed_rows"]):
        raise ValueError("Candidate and exact eight-SSE aggregation must replay")
    seen = {}
    for view in views:
        support, key = view["support"], view["fit_key"]
        if (key == "full_HW") != support["full_support"]:
            raise ValueError("Only full physical support may reuse original HW")
        signature = tuple(support["electrode_ids"])
        if signature in seen and seen[signature] != key:
            raise ValueError("Identical physical sets must reuse the exact fit")
        seen[signature] = key
        if key != "full_HW" and item["new_fit_metadata"][key]["electrode_ids"] != support["electrode_ids"]:
            raise ValueError("Saved fitted support differs from frozen physical IDs")
        folds = view["fit_diagnostics"]
        if [d["held_direction"] for d in folds] != list(range(8)):
            raise ValueError("Every support must include eight original held-direction fits")
        for fold in folds:
            if (fold["fitting_bin_rows"] != 15*(description["complete_trials"]-description["direction_counts"][fold["held_direction"]])
                    or not np.isfinite(fold["trace"]) or fold["trace"] < 0
                    or [r["lambda"] for r in fold["ridge_df_by_lambda"]] != list(LAMBDAS)):
                raise ValueError("Original fitting rows and selected-support covariance diagnostics required")
    keys = {v["fit_key"] for v in views if v["fit_key"] != "full_HW"}
    if (keys != set(item["new_fit_metadata"]) or len(keys) != description["unique_new_fit_supports"]
            or item["new_unique_fit_count"] != len(keys)):
        raise ValueError("Exact distinct-support fit count must match qualified metadata")


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
    for name in ("prior-valid", "prior-patience", "prior-alternatives", "prior-nulls"):
        summary.add_argument("--"+name, type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted complete intersection session cap is at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "recipes": list(RECIPES),
        "scientific_trial_count_increment": 0, "sensitivity_only": True, "reference_refitted": False,
        "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Adopted complete intersection session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            metadata, descriptions, records = qualify_global(args.metadata, args.trial_table, args.scalar_bank_root, args.hw_bank_root)
            description = descriptions[args.session]
            if Path(metadata["source"]).resolve() != args.source.resolve():
                raise ValueError("Exact qualified read-only source root required")
            report.update(global_metadata_qualified_before_source=True, metadata_path=str(args.metadata),
                metadata_pilot_session=metadata["pilot_session"], qualified_support=description,
                global_metadata_seconds=time.monotonic()-started)
            reading = time.monotonic()
            if description["unique_new_fit_supports"]:
                x, y, labels, rows, source = load_lmp_history_session(args.source, args.trial_table,
                    args.scalar_bank_root, args.session, check)
                ids = np.asarray([r["trial_id"] for r in rows])
                if (rows != records[args.session] or source["common_LMP_electrodes"] != description["original_electrode_ids"]
                        or source["complete_trials"] != description["complete_trials"]):
                    raise ValueError("Original source support changed after complete guide qualification")
                report["source_read_seconds"] = time.monotonic()-reading
                fitting = time.monotonic()
                fits, routes = fit_supports(x, y, labels, ids, description, check)
                report["reduced_fit_seconds"] = time.monotonic()-fitting
                reference, _, path = load_reference(args.hw_bank_root, args.session, description, source)
            else:
                reference, source, path = load_reference(args.hw_bank_root, args.session, description)
                y, labels, ids = restore_target_layout(reference["actual_xy"]), reference["directions"], reference["trial_ids"]
                fits, routes = {}, ["full_HW"]*9
                report.update(source_read_seconds=0., reduced_fit_seconds=0.)
            check()
            banks, payload = finalize_banks(y, labels, ids, description, fits, routes, reference)
            report.update(session_id=args.session, source=source, banks=banks,
                exact_original_HW_target_baseline_trial_score_replay=True, HW_predictions_path=path,
                provider_source_values_read=bool(description["unique_new_fit_supports"]),
                new_unique_fit_count=len(fits), exact_full_HW_support_reuses=routes.count("full_HW"),
                new_fit_metadata={key: {k: fit[k] for k in ("metadata", "electrode_ids")} for key, fit in fits.items()},
                status="complete_intersection_sensitivity_session_banks")
            validate_session(report, args.session)
            check()
            np.savez(args.out/"new_fitted_predictions.npz", actual_xy=y, directions=labels, trial_ids=ids, **payload)
        else:
            sessions = {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["banks"]
            original = json.loads(args.hw_summary.read_text())
            panels = [p for p in original["panels"] if p["recipe"] == REFERENCE]
            if original["status"] != "complete_history_repeatability_weight_control_panel" or len(panels) != 1:
                raise ValueError("Complete preserved original HW selection report required")
            report.update(summarize_banks(sessions, panels[0], **{k: getattr(args, k) for k in
                ("prior_valid", "prior_patience", "prior_alternatives", "prior_nulls")}),
                status="complete_intersection_sensitivity_panel", bank_root=str(args.bank_root),
                original_HW_summary_path=str(args.hw_summary))
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_intersection.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
