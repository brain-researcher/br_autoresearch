#!/usr/bin/env python3
"""Score preserved W/HW mixtures and publish a complete outer-session selection."""
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

from primary_capacity import score_bank
from primary_covariance_svd import numeric_trial_ids
from primary_history_blend import (ALPHAS, COMPONENTS, RECIPE, SCORE_KEYS, blend_prediction,
    prediction_scores, score_session, summarize_banks, validate_bank, validate_original_selections)
from primary_reference import LAMBDAS, SESSIONS
from source_adapter import compute_only

COMPLETE_STATUSES = {"W": "complete_two_repeatability_weight_session_banks",
                     "HW": "complete_two_history_repeatability_weight_session_banks"}
SUMMARY_STATUSES = {"W": "complete_repeatability_weight_control_panel",
                    "HW": "complete_history_repeatability_weight_control_panel"}


def restore_target_layout(values):
    """Restore the original target.T.astype(float64), then stack, memory order."""
    values = np.asarray(values)
    if values.dtype != np.dtype("float64") or values.ndim != 3 or values.shape[1:] != (15, 2) or not np.isfinite(values).all():
        raise ValueError("Complete saved native float64 XY trajectories required")
    return np.stack([trajectory.T.copy(order="C").T for trajectory in values])


def load_components(w_root, hw_root, session):
    banks, arrays, paths, sources = {}, {}, {}, {}
    for name, root in (("W", Path(w_root)), ("HW", Path(hw_root))):
        item = json.loads((root/session/"score_banks.json").read_text())
        source, recipe = item["source"], COMPONENTS[name]
        if (item["status"] != COMPLETE_STATUSES[name] or item["session_id"] != session
                or item["exact_original_LMP_baseline_target_trial_replay"] is not True
                or source["session_id"] != session or source["native_bin_seconds"] != .03):
            raise ValueError("Complete original component session and target replay required")
        if name == "W":
            if source["native_offsets"] != [0, 14]:
                raise ValueError("Original W current-bin target interval required")
        elif (source["native_LMP_offsets"] != [-3, 14] or source["native_target_offsets"] != [0, 14]
                or source["lag_offsets"] != [0, -1, -2, -3]
                or item["exact_original_current_weight_state_and_maps_replay"] is not True
                or item["distinct_history_input_and_primary_target_windows_verified"] is not True):
            raise ValueError("Original HW history interval and unchanged primary target required")
        old = item["banks"][recipe]
        channels = source["common_LMP_electrodes"]
        if (old["recipe"] != recipe or old["lambdas"] != list(LAMBDAS)
                or [row["lambda"] for row in old["lambda_rows"]] != list(LAMBDAS)
                or not channels or channels != sorted(set(channels))
                or any(type(channel) is not int or channel < 1 for channel in channels)
                or old["physical_electrode_ids"] != channels or old["physical_electrode_count"] != len(channels)
                or old["native_XY_scaled"] is not False):
            raise ValueError("Original physical electrodes, native target and five-penalty candidate required")
        path = root/session/(recipe+"_predictions.npz")
        with np.load(path, allow_pickle=False) as saved:
            y = restore_target_layout(saved["actual_xy"])
            baseline = restore_target_layout(saved["baseline_xy"])
            labels, ids = saved["directions"], saved["trial_ids"]
            predictions = saved["candidate_xy_by_lambda"]
        numbers = numeric_trial_ids(ids, session)
        if (ids.dtype.kind not in "US" or labels.shape != (len(y),) or ids.shape != (len(y),)
                or np.any(np.diff(numbers) <= 0) or set(labels.tolist()) != set(range(8))
                or any(np.count_nonzero(labels == d) < 2 for d in range(8))
                or len(y) != source["complete_trials"] or baseline.shape != y.shape
                or predictions.shape != (len(LAMBDAS), *y.shape) or predictions.dtype != np.dtype("float64")
                or not predictions.flags.c_contiguous or not np.isfinite(predictions).all()):
            raise ValueError("Complete ordered original native IDs/targets and literal saved prediction layout required")
        replay = score_bank(y, labels, baseline, predictions, recipe)
        for key in ("lambdas", "baseline_R2", "baseline_SSE", "denominator", "direction_counts", "lambda_rows"):
            if replay[key] != old[key]:
                raise ValueError(f"Exact saved {name} component score replay failed: {key}")
        banks[name], arrays[name], paths[name], sources[name] = old, (y, baseline, labels, ids, predictions), str(path), source
    for index, key in enumerate(("actual_xy", "baseline_xy", "directions", "trial_ids")):
        if not np.array_equal(arrays["W"][index], arrays["HW"][index]):
            raise ValueError(f"Original W/HW saved support differs: {key}")
    if sources["W"]["common_LMP_electrodes"] != sources["HW"]["common_LMP_electrodes"]:
        raise ValueError("Original W/HW physical support differs")
    y, baseline, labels, ids, _ = arrays["W"]
    source = {"session_id": session, "complete_trials": len(y), "native_bin_seconds": .03,
        "native_target_offsets": [0, 14], "common_LMP_electrodes": sources["W"]["common_LMP_electrodes"],
        "component_input_offsets": {"W": [0, 14], "HW": [-3, 14]}, "component_prediction_paths": paths,
        "target_and_baseline_layout": "fixed original time-fast trial trajectories restored from saved values",
        "provider_source_values_read": False}
    return y, baseline, labels, ids, {name: values[4] for name, values in arrays.items()}, banks, source


def validate_session(item, session):
    if (item["status"] != "complete_history_blend_session_bank" or item["session_id"] != session
            or item["exact_original_component_arrays_support_replay"] is not True
            or item["exact_original_component_scores_replay"] is not True
            or item["provider_source_values_read"] is not False or item["neural_refits"] != 0
            or item["scientific_trial_count_increment"] != 0):
        raise ValueError("Complete artifact-only mixture session and exact endpoint replays required")
    bank, source = item["bank"], item["source"]
    validate_bank(bank)
    if (source["session_id"] != session or source["native_target_offsets"] != [0, 14]
            or source["native_bin_seconds"] != .03 or source["provider_source_values_read"] is not False
            or source["component_input_offsets"] != {"W": [0, 14], "HW": [-3, 14]}
            or len(bank["direction_counts"]) != 8 or any(type(n) is not int or n < 2 for n in bank["direction_counts"])
            or sum(bank["direction_counts"]) != source["complete_trials"]
            or set(bank["component_score_rows"]) != set(COMPONENTS)):
        raise ValueError("Full original scoring support required")
    for name, alpha_index in (("W", 0), ("HW", 4)):
        rows = bank["component_score_rows"][name]
        if [row["lambda"] for row in rows] != list(LAMBDAS):
            raise ValueError("All five original endpoint rows required")
        for li in range(5):
            if any(bank["cells"][5*li+alpha_index][key] != rows[li][key] for key in SCORE_KEYS):
                raise ValueError("Literal endpoint score must reproduce its original bank")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    session_parser = sub.add_parser("session")
    for name in ("w-bank-root", "hw-bank-root", "out"):
        session_parser.add_argument("--"+name, type=Path, required=True)
    session_parser.add_argument("--session", choices=SESSIONS, required=True)
    session_parser.add_argument("--max-seconds", type=int, default=60)
    summary_parser = sub.add_parser("summarize")
    for name in ("bank-root", "w-bank-root", "hw-bank-root", "w-summary", "hw-summary", "out"):
        summary_parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit single-thread numerical settings required")
    if args.mode == "session" and not 0 < args.max_seconds <= 60:
        raise ValueError("Adopted artifact-only session bound is at most 60 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "neural_refits": 0,
        "provider_source_values_read": False, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Adopted history-blend artifact session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        if args.mode == "session":
            y, baseline, labels, ids, predictions, banks, source = load_components(args.w_bank_root, args.hw_bank_root, args.session)
            report.update(session_id=args.session, source=source, artifact_read_and_replay_seconds=time.monotonic()-started)
            check()
            scoring = time.monotonic()
            bank = score_session(y, baseline, labels, predictions["W"], predictions["HW"], check)
            bank["component_score_rows"] = {name: banks[name]["lambda_rows"] for name in COMPONENTS}
            report.update(mixture_score_seconds=time.monotonic()-scoring,
                exact_original_component_arrays_support_replay=True, exact_original_component_scores_replay=True,
                bank=bank, status="complete_history_blend_session_bank")
            validate_session(report, args.session)
        else:
            sessions, anchors, anchor_panels = {}, {name: {} for name in COMPONENTS}, {}
            for session in SESSIONS:
                item = json.loads((args.bank_root/session/"score_banks.json").read_text())
                validate_session(item, session)
                sessions[session] = item["bank"]
                for name, root in (("W", args.w_bank_root), ("HW", args.hw_bank_root)):
                    old = json.loads((root/session/"score_banks.json").read_text())
                    if old["status"] != COMPLETE_STATUSES[name] or old["session_id"] != session:
                        raise ValueError("Both complete preserved component banks required")
                    anchors[name][session] = old["banks"][COMPONENTS[name]]
                    if sessions[session]["component_score_rows"][name] != anchors[name][session]["lambda_rows"]:
                        raise ValueError("Original component score rows changed since mixture scoring")
            for name, path in (("W", args.w_summary), ("HW", args.hw_summary)):
                old = json.loads(path.read_text())
                panels = [panel for panel in old["panels"] if panel["recipe"] == COMPONENTS[name]]
                if old["status"] != SUMMARY_STATUSES[name] or len(panels) != 1:
                    raise ValueError("Complete original endpoint selection summaries required")
                validate_original_selections(anchors[name], panels[0])
                anchor_panels[name] = panels[0]
            result = summarize_banks(sessions, anchors, anchor_panels)
            selected_dir = args.out/"selected_predictions"
            selected_dir.mkdir(exist_ok=False)
            selected_paths = {}
            for row in result["blend"]["sessions"]:
                session = row["session_id"]
                y, baseline, labels, ids, predictions, _, _ = load_components(args.w_bank_root, args.hw_bank_root, session)
                prediction = blend_prediction(predictions["W"][row["lambda_index"]], predictions["HW"][row["lambda_index"]], row["alpha"])
                replay = prediction_scores(y, baseline, labels, prediction)
                if any(replay[key] != row[key] for key in SCORE_KEYS):
                    raise ValueError("Selected final prediction must replay its complete mixture cell")
                for name in COMPONENTS:
                    original_index = next(old["lambda_index"] for old in anchor_panels[name]["sessions"] if old["session_id"] == session)
                    if bool(np.array_equal(prediction, predictions[name][original_index])) != row["exactly_matches_original_selected_components"][name]:
                        raise ValueError("Selected-array exact duplicate evidence changed")
                path = selected_dir/(session+".npz")
                np.savez(path, actual_xy=y, baseline_xy=baseline, directions=labels, trial_ids=ids,
                    candidate_xy=prediction, selected_lambda=row["lambda"], selected_alpha=row["alpha"])
                selected_paths[session] = str(path)
            report.update(result, status="complete_history_blend_panel", selected_prediction_paths=selected_paths,
                bank_root=str(args.bank_root), original_component_summaries={"W": str(args.w_summary), "HW": str(args.hw_summary)},
                original_selectors_independently_replayed=True, all_selected_prediction_scores_replayed=True,
                all_original_component_arrays_support_replay=True, all_original_component_scores_replay=True)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        filename = "score_banks.json" if args.mode == "session" else "primary_history_blend.json"
        (args.out/filename).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
