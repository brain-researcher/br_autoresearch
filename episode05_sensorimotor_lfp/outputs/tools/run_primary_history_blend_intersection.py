#!/usr/bin/env python3
"""Refit only reduced current W and combine with every preserved intersection HW view."""
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
from primary_history_blend import (COMPONENTS, SCORE_KEYS, blend_prediction, prediction_scores,
    select_cell as select_original_blend)
from primary_history_blend_intersection import (OLD_RECIPES, RECIPES, SUPPORT_KEYS,
    fit_current_view, score_views, summarize_banks, validate_bank, validate_prior_hw_selections)
from primary_intersection import average_banks
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from run_primary_history_blend import (load_components, restore_target_layout,
    validate_session as validate_original_blend_session)
from run_primary_intersection import qualify_global, validate_session as validate_old_session
from source_adapter import compute_only

PATH_OPTIONS = ("metadata", "trial_table", "scalar_bank_root", "w_bank_root", "hw_bank_root",
                "blend_bank_root", "blend_summary", "intersection_bank_root", "intersection_summary")
REPLAY_FLAGS = ("global_metadata_qualified_before_source", "original_selectors_independently_replayed",
    "exact_original_target_baseline_trial_replay", "exact_all_nine_HW_scores_replay",
    "exact_reduced_W_current_state_replay", "exact_original_full_blend_prediction_replay")
SESSION_STATUS = "complete_history_blend_intersection_session_bank"
PANEL_STATUS = "complete_history_blend_intersection_panel"
SUMMARY_FILE = "primary_history_blend_intersection.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def load_context(args):
    """All fixed joins and both historical selectors precede source values."""
    metadata, descriptions, records = qualify_global(args.metadata, args.trial_table, args.scalar_bank_root, args.hw_bank_root)
    if hasattr(args, "source") and Path(metadata["source"]).resolve() != args.source.resolve():
        raise ValueError("Only the exact metadata-qualified read-only source may be used")
    old_report = read_json(args.intersection_summary)
    if (old_report["status"] != "complete_intersection_sensitivity_panel"
            or old_report["reference_refitted"] is not False or old_report["sensitivity_only"] is not True
            or {p["recipe"] for p in old_report["panels"]} != set(OLD_RECIPES)):
        raise ValueError("Complete original I/R-HW summary required")
    old_panels = {name: next(p for p in old_report["panels"] if p["recipe"] == recipe)
                  for name, recipe in zip(("I", "R"), OLD_RECIPES)}
    index_rows = {row["session_id"]: row for row in old_panels["I"]["sessions"]}
    if len(old_panels["I"]["sessions"]) != 16 or set(index_rows) != set(SESSIONS):
        raise ValueError("Every original selected I-HW session required")
    old_sessions, old_views, blend_banks = {}, {}, {}
    for session in SESSIONS:
        item = read_json(args.intersection_bank_root/session/"score_banks.json")
        validate_old_session(item, session)
        if item["qualified_support"] != descriptions[session]:
            raise ValueError("Completed HW views must use the exact metadata002 description")
        views = item["banks"][OLD_RECIPES[0]]["views"]+item["banks"][OLD_RECIPES[1]]["views"]
        old_views[session] = views
        old_sessions[session] = {"views": [{"support": view["support"], "HW_bank": view["bank"]} for view in views],
                                 "original_I_HW_lambda_index": index_rows[session]["lambda_index"]}
        original = read_json(args.blend_bank_root/session/"score_banks.json")
        validate_original_blend_session(original, session)
        blend_banks[session] = original["bank"]
    old_rows = validate_prior_hw_selections(old_sessions, old_panels)
    blend_report = read_json(args.blend_summary)
    if (blend_report["status"] != "complete_history_blend_panel"
            or blend_report["recipe"] != "lmp_current_history_repeatability_convex_blend_v1"
            or any(blend_report[key] is not True for key in ("original_selectors_independently_replayed",
                "all_selected_prediction_scores_replayed", "all_original_component_arrays_support_replay",
                "all_original_component_scores_replay"))
            or set(blend_report["selected_prediction_paths"]) != set(SESSIONS)
            or len(blend_report["blend"]["sessions"]) != 16):
        raise ValueError("Complete exact-replay original full-support blend required")
    blend_rows = {row["session_id"]: row for row in blend_report["blend"]["sessions"]}
    if set(blend_rows) != set(SESSIONS):
        raise ValueError("All original selected full-support blend sessions required")
    for session in SESSIONS:
        index, training, scores = select_original_blend(blend_banks, session)
        row, cell = blend_rows[session], blend_banks[session]["cells"][index]
        if (row["animal"] != ANIMALS[session] or row["cell_index"] != index
                or row["outer_training_sessions"] != training or row["training_primary_scores"] != scores
                or any(row[key] != cell[key] for key in ("lambda_index", "alpha_index", "lambda", "alpha", *SCORE_KEYS))):
            raise ValueError("Exact original full-blend outer selector changed")
    return {"metadata": metadata, "descriptions": descriptions, "records": records,
        "old_hw_panels": old_panels, "old_hw_rows": old_rows, "old_views": old_views,
        "blend_banks": blend_banks, "blend_report": blend_report, "blend_rows": blend_rows}


def same_saved_support(saved, y, baseline, labels, ids):
    for key, expected in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
        if not np.array_equal(saved[key], expected):
            raise ValueError("Saved original support differs: "+key)


def check_predictions(values, shape):
    if (values.shape != shape or values.dtype != np.dtype("float64")
            or not values.flags.c_contiguous or not np.isfinite(values).all()):
        raise ValueError("Literal finite float64 C-order complete saved predictions required")


def load_saved_views(args, session, context):
    """No source reads or refits: exact reduced HW and full-blend array replay."""
    y, baseline, labels, ids, full_components, full_banks, full_source = load_components(args.w_bank_root, args.hw_bank_root, session)
    description = context["descriptions"][session]
    if (ids.tolist() != description["trial_ids"] or labels.tolist() != description["trial_directions"]
            or full_source["common_LMP_electrodes"] != description["original_electrode_ids"]):
        raise ValueError("Original full-component support differs from metadata002")
    item = read_json(args.intersection_bank_root/session/"score_banks.json")
    validate_old_session(item, session)
    views = item["banks"][OLD_RECIPES[0]]["views"]+item["banks"][OLD_RECIPES[1]]["views"]
    if views != context["old_views"][session] or item["qualified_support"] != description:
        raise ValueError("Old HW support banks changed after original selector replay")
    path = args.intersection_bank_root/session/"new_fitted_predictions.npz"
    hw_predictions, hw_metadata = [], []
    with np.load(path, allow_pickle=False) as saved:
        same_saved_support(saved, y, baseline, labels, ids)
        restore_target_layout(saved["actual_xy"])
        restore_target_layout(saved["baseline_xy"])
        expected_keys = {view["fit_key"]+"_candidate_xy_by_lambda" for view in views if view["fit_key"] != "full_HW"}
        if set(saved.files) != expected_keys|{"actual_xy", "baseline_xy", "directions", "trial_ids"}:
            raise ValueError("Exact saved HW fit-key routing required")
        cache = {}
        for view in views:
            key = view["fit_key"]
            if key == "full_HW":
                prediction = full_components["HW"]
                folds = full_banks["HW"]["direction_fit_metadata"]
            else:
                if key not in cache:
                    cache[key] = saved[key+"_candidate_xy_by_lambda"]
                prediction = cache[key]
                folds = item["new_fit_metadata"][key]["metadata"]
            check_predictions(prediction, (5, *y.shape))
            replay = score_bank(y, labels, baseline, prediction, view["support"]["recipe"])
            if any(replay[name] != view["bank"][name] for name in ("lambdas", *SUPPORT_KEYS, "lambda_rows")):
                raise ValueError("Exact saved reduced HW view all-five-score replay failed")
            hw_predictions.append(prediction)
            hw_metadata.append(folds)
    old_mean = average_banks([score_bank(y, labels, baseline, p, OLD_RECIPES[1]) for p in hw_predictions[1:]])
    if any(old_mean[key] != item["banks"][OLD_RECIPES[1]][key] for key in (*SUPPORT_KEYS, "lambda_rows", "support_seed_rows")):
        raise ValueError("Original mean-eight HW losses must replay exactly")
    row = context["blend_rows"][session]
    with np.load(Path(context["blend_report"]["selected_prediction_paths"][session]), allow_pickle=False) as saved:
        same_saved_support(saved, y, baseline, labels, ids)
        full_blend = saved["candidate_xy"]
        if float(saved["selected_lambda"]) != row["lambda"] or float(saved["selected_alpha"]) != row["alpha"]:
            raise ValueError("Original full-blend selected parameters changed")
    check_predictions(full_blend, y.shape)
    rebuilt = blend_prediction(full_components["W"][row["lambda_index"]], full_components["HW"][row["lambda_index"]], row["alpha"])
    scores = prediction_scores(y, baseline, labels, full_blend)
    if not np.array_equal(full_blend, rebuilt) or any(scores[key] != row[key] for key in SCORE_KEYS):
        raise ValueError("Exact original selected full-blend array/score replay failed")
    return {"y": y, "baseline": baseline, "labels": labels, "ids": ids,
        "HW_predictions": hw_predictions, "HW_metadata": hw_metadata, "full_components": full_components,
        "full_banks": full_banks, "full_blend": full_blend, "source": item["source"], "HW_path": str(path)}


def fit_reduced_W(x, y, labels, description, saved, check=lambda: None):
    if x.shape != (len(y), 18, len(description["original_electrode_ids"])):
        raise ValueError("Exact original eighteen-bin source layout required")
    fits, routes, predictions, metadata, seen = {}, [], [], [], {}
    for vi, support in enumerate(description["support_sets"]):
        check()
        signature = tuple(support["electrode_ids"])
        if support["full_support"]:
            key = "full_W"
            prediction, folds = saved["full_components"]["W"], saved["full_banks"]["W"]["direction_fit_metadata"]
        elif signature in seen:
            key = seen[signature]
            prediction, folds = fits[key]["predictions"], fits[key]["metadata"]
        else:
            key = f"wfit{len(fits):03d}"
            selected_x = x[:, :, support["electrode_positions"]]
            baseline, prediction, folds = fit_current_view(selected_x, y, labels, saved["HW_metadata"][vi], check)
            if not np.array_equal(baseline, saved["baseline"]):
                raise ValueError("Reduced W must preserve the original full and reduced HW baseline exactly")
            fits[key] = {"predictions": prediction, "metadata": folds, "electrode_ids": support["electrode_ids"]}
            seen[signature] = key
        for new, old in zip(folds, saved["HW_metadata"][vi]):
            if any(new[name] != old[name] for name in ("held_direction", "fitting_bin_rows", "physical_mean", "physical_sd", "repeatability")):
                raise ValueError("Every reused W/HW current-state view must replay exactly")
        routes.append(key)
        predictions.append(prediction)
        metadata.append(folds)
    return fits, routes, predictions, metadata


def attach_metadata(bank, routes, w_metadata, hw_metadata):
    for vi, view in enumerate(bank["views"]):
        view.update(W_fit_key=routes[vi], W_fit_metadata=w_metadata[vi], HW_fit_metadata=hw_metadata[vi])


def validate_session(item, session):
    if (item["status"] != SESSION_STATUS or item["session_id"] != session or item["recipes"] != list(RECIPES)
            or any(item[key] is not True for key in REPLAY_FLAGS)
            or item["original_HW_refitted"] is not False or item["full_W_refitted"] is not False
            or item["sensitivity_only"] is not True or item["promotion_authorized"] is not False
            or item["scientific_trial_count_increment"] != 0 or item["literal_null_control_count_increment"] != 0
            or item["alternative_explanation_count_increment"] != 0):
        raise ValueError("Complete source-bounded W-only support mixture and every exact replay required")
    bank, description, source = item["bank"], item["qualified_support"], item["source"]
    validate_bank(bank)
    if (description["session_id"] != session or source["session_id"] != session
            or source["common_LMP_electrodes"] != description["original_electrode_ids"]
            or source["complete_trials"] != description["complete_trials"]
            or source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
            or source["native_bin_seconds"] != .03 or source["spike_values_read"] is not False
            or source["lag_offsets"] != [0, -1, -2, -3]
            or item["effective_W_offsets"] != [0, 14] or item["W_reader_offsets"] != [-3, 14]
            or [view["support"] for view in bank["views"]] != description["support_sets"]
            or bank["direction_counts"] != description["direction_counts"]):
        raise ValueError("Exact original source/current/target windows and metadata support routing required")
    seen = {}
    for view in bank["views"]:
        support, key = view["support"], view["W_fit_key"]
        if (key == "full_W") != support["full_support"]:
            raise ValueError("Only a full physical set may reuse original W")
        signature = tuple(support["electrode_ids"])
        if signature in seen and seen[signature] != key:
            raise ValueError("Identical physical sets must reuse one literal reduced W fit")
        seen[signature] = key
        if key != "full_W" and item["new_W_fit_metadata"][key]["electrode_ids"] != support["electrode_ids"]:
            raise ValueError("Saved W fit physical support differs")
        if ([r["held_direction"] for r in view["W_fit_metadata"]] != list(range(8))
                or [r["held_direction"] for r in view["HW_fit_metadata"]] != list(range(8))):
            raise ValueError("Every W and preserved HW direction fold required")
        for w, hw in zip(view["W_fit_metadata"], view["HW_fit_metadata"]):
            if (any(w[key] != hw[key] for key in ("held_direction", "fitting_bin_rows", "physical_mean", "physical_sd", "repeatability"))
                    or w["fitting_bin_rows"] != 15*(description["complete_trials"]-description["direction_counts"][w["held_direction"]])
                    or len(w["paired_weights"]["weights"]) != len(support["electrode_ids"])
                    or [r["lambda"] for r in w["paired_weighted_covariance"]["ridge_df_by_lambda"]] != list(LAMBDAS)):
                raise ValueError("Every reduced W current state and native fitting rows must replay preserved HW")
    keys = {view["W_fit_key"] for view in bank["views"] if view["W_fit_key"] != "full_W"}
    if (keys != set(item["new_W_fit_metadata"]) or len(keys) != item["new_unique_W_fit_count"]
            or len(keys) != description["unique_new_fit_supports"] or item["new_W_ridge_heads"] != 40*len(keys)):
        raise ValueError("Exactly one five-penalty eight-fold W fit per new physical support required")


def replay_new_session(args, session, context, item):
    """Re-score all saved new W/HW mixtures, including every literal reuse flag."""
    saved = load_saved_views(args, session, context)
    predictions, cache = [], {}
    with np.load(args.bank_root/session/"new_W_predictions.npz", allow_pickle=False) as archive:
        same_saved_support(archive, saved["y"], saved["baseline"], saved["labels"], saved["ids"])
        expected = {key+"_candidate_xy_by_lambda" for key in item["new_W_fit_metadata"]}
        if set(archive.files) != expected|{"actual_xy", "baseline_xy", "directions", "trial_ids"}:
            raise ValueError("Saved new W fit-key arrays changed")
        for view in item["bank"]["views"]:
            key = view["W_fit_key"]
            if key == "full_W":
                values = saved["full_components"]["W"]
            else:
                if key not in cache:
                    cache[key] = archive[key+"_candidate_xy_by_lambda"]
                values = cache[key]
            check_predictions(values, (5, *saved["y"].shape))
            predictions.append(values)
    replay = score_views(saved["y"], saved["baseline"], saved["labels"], context["descriptions"][session],
        predictions, saved["HW_predictions"], context["old_hw_rows"]["I"][session]["lambda_index"], saved["full_blend"])
    for key in (*SUPPORT_KEYS, "original_I_HW_lambda_index", "full_blend_scores", "mean_random_cells"):
        if replay[key] != item["bank"][key]:
            raise ValueError("Saved new support-mixture scoring replay failed: "+key)
    for vi, (new, old) in enumerate(zip(replay["views"], item["bank"]["views"])):
        if (any(new[key] != old[key] for key in ("support", "blend_bank", "W_bank", "HW_bank", "error_cross_products_by_lambda"))
                or old["HW_fit_metadata"] != saved["HW_metadata"][vi]):
            raise ValueError("Every new W score, mixture cell, error cross-term and duplicate flag must replay exactly")
    return saved, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("session", "summarize"):
        child = sub.add_parser(mode)
        for name in (*PATH_OPTIONS, "out"):
            child.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
        if mode == "session":
            child.add_argument("--source", type=Path, required=True)
            child.add_argument("--session", choices=SESSIONS, required=True)
            child.add_argument("--max-seconds", type=int, default=270)
        else:
            child.add_argument("--bank-root", type=Path, required=True)
            for name in ("prior-valid", "prior-patience", "prior-alternatives", "prior-nulls"):
                child.add_argument("--"+name, type=int, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1:
        raise ValueError("An actual allocated CPU is required")
    if args.mode == "session" and not 0 < args.max_seconds <= 270:
        raise ValueError("Adopted W-only support pilot bound is at most270seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipes": list(RECIPES), "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "original_HW_refitted": False,
        "full_W_refitted": False, "sensitivity_only": True, "eligible_for_comparable_primary_incumbent": False,
        "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Adopted reduced-W blend-support session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        context = load_context(args)
        report["global_metadata_and_selector_seconds"] = time.monotonic()-started
        if args.mode == "session":
            session, description = args.session, context["descriptions"][args.session]
            replay_start = time.monotonic()
            saved = load_saved_views(args, session, context)
            report["artifact_read_and_replay_seconds"] = time.monotonic()-replay_start
            reading = time.monotonic()
            if description["unique_new_fit_supports"]:
                x, y, labels, records, source = load_lmp_history_session(args.source, args.trial_table, args.scalar_bank_root, session, check)
                ids = np.asarray([row["trial_id"] for row in records])
                if (records != context["records"][session] or source["common_LMP_electrodes"] != description["original_electrode_ids"]
                        or any(not np.array_equal(new, old) for new, old in
                            ((y, saved["y"]), (labels, saved["labels"]), (ids, saved["ids"])))
                        or y.strides != saved["y"].strides):
                    raise ValueError("Qualified source targets/IDs/layout differ from complete saved anchors")
                report["source_read_seconds"] = time.monotonic()-reading
                fitting = time.monotonic()
                fits, routes, predictions, w_metadata = fit_reduced_W(x, y, labels, description, saved, check)
                report["reduced_W_fit_seconds"] = time.monotonic()-fitting
            else:
                y, labels, ids, source = saved["y"], saved["labels"], saved["ids"], saved["source"]
                fits, routes = {}, ["full_W"]*9
                predictions = [saved["full_components"]["W"]]*9
                w_metadata = [saved["full_banks"]["W"]["direction_fit_metadata"]]*9
                report.update(source_read_seconds=0., reduced_W_fit_seconds=0.)
            scoring = time.monotonic()
            bank = score_views(y, saved["baseline"], labels, description, predictions, saved["HW_predictions"],
                context["old_hw_rows"]["I"][session]["lambda_index"], saved["full_blend"], check)
            attach_metadata(bank, routes, w_metadata, saved["HW_metadata"])
            report.update(session_id=session, source=source, qualified_support=description, bank=bank,
                effective_W_offsets=[0, 14], W_reader_offsets=[-3, 14],
                provider_source_values_read=bool(description["unique_new_fit_supports"]),
                new_unique_W_fit_count=len(fits), new_W_ridge_heads=40*len(fits),
                new_W_fit_metadata={key: {name: value[name] for name in ("metadata", "electrode_ids")} for key, value in fits.items()},
                mixture_score_seconds=time.monotonic()-scoring, status=SESSION_STATUS,
                **{key: True for key in REPLAY_FLAGS})
            validate_session(report, session)
            check()
            np.savez(args.out/"new_W_predictions.npz", actual_xy=y, baseline_xy=saved["baseline"], directions=labels, trial_ids=ids,
                **{key+"_candidate_xy_by_lambda": value["predictions"] for key, value in fits.items()})
        else:
            sessions, items = {}, {}
            for session in SESSIONS:
                item = read_json(args.bank_root/session/"score_banks.json")
                validate_session(item, session)
                if item["qualified_support"] != context["descriptions"][session]:
                    raise ValueError("Final new banks must match the immutable metadata002 support")
                sessions[session], items[session] = item["bank"], item
            result = summarize_banks(sessions, context["old_hw_panels"], context["blend_report"]["blend"],
                **{name: getattr(args, name) for name in ("prior_valid", "prior_patience", "prior_alternatives", "prior_nulls")})
            selected_dir = args.out/"selected_predictions"
            selected_dir.mkdir(exist_ok=False)
            selected_rows = {row["session_id"]: row for row in result["panels"][0]["sessions"]}
            paths = {}
            for session in SESSIONS:
                saved, w_predictions = replay_new_session(args, session, context, items[session])
                row = selected_rows[session]
                li, alpha = row["lambda_index"], row["alpha"]
                predictions = [blend_prediction(w[li], hw[li], alpha) for w, hw in zip(w_predictions, saved["HW_predictions"])]
                path = selected_dir/(session+".npz")
                np.savez(path, actual_xy=saved["y"], baseline_xy=saved["baseline"], directions=saved["labels"], trial_ids=saved["ids"],
                    intersection_xy=predictions[0], random_support_xy_by_replicate=np.stack(predictions[1:]),
                    selected_lambda=row["lambda"], selected_alpha=alpha)
                paths[session] = str(path)
            report.update(result, status=PANEL_STATUS, bank_root=str(args.bank_root), selected_prediction_paths=paths,
                metadata_path=str(args.metadata), original_intersection_summary=str(args.intersection_summary),
                original_full_blend_summary=str(args.blend_summary), provider_source_values_read=False,
                all_new_saved_W_scores_replayed=True, all_selected_mixture_scores_replayed=True,
                all_exact_duplicate_flags_replayed=True, **{key: True for key in REPLAY_FLAGS})
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else SUMMARY_FILE
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "maximum_RSS_KiB")}), flush=True)


if __name__ == "__main__":
    main()
