#!/usr/bin/env python3
"""Score the adopted five linear time schedules using only complete saved artifacts."""
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
from primary_elapsed_only import full_model_contrast
from primary_history_blend import SCORE_KEYS, blend_prediction, prediction_scores
from primary_reference import LAMBDAS, SESSIONS
from primary_timevarying_history_blend import (ANCHORS, RECIPE, SUPPORT_KEYS, score_session,
    summarize_banks, timevarying_prediction, validate_anchor_selections, validate_bank)
from run_primary_history_blend import (COMPLETE_STATUSES, SUMMARY_STATUSES, load_components,
    restore_target_layout, validate_session as validate_blend_session)
from run_primary_history2_repeatability_weight import REPLAY_FLAGS as H2_REPLAY_FLAGS
from source_adapter import compute_only

PATH_OPTIONS = ("w_bank_root", "hw_bank_root", "blend_bank_root", "h2_bank_root",
                "w_summary", "hw_summary", "blend_summary", "h2_summary")
REPLAY_FLAGS = ("original_selectors_independently_replayed", "exact_original_component_arrays_support_replay",
    "exact_original_component_scores_replay", "exact_original_blend_selected_and_half_cells_replay",
    "exact_original_H2_arrays_and_scores_replay", "exact_all_four_selected_anchor_scores_replay")
SESSION_STATUS = "complete_timevarying_history_blend_session_bank"
PANEL_STATUS = "complete_timevarying_history_blend_panel"
SUMMARY_FILE = "primary_timevarying_history_blend.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def load_anchor_context(args):
    """Replay all four original outer selectors before evaluating any new session."""
    banks = {name: {} for name in ("W", "HW", "H2")}
    blends, reports, panels, sources = {}, {}, {}, {name: {} for name in banks}
    for session in SESSIONS:
        for name in ("W", "HW", "H2"):
            item = read_json(getattr(args, name.lower()+"_bank_root")/session/"score_banks.json")
            if name in COMPLETE_STATUSES:
                if (item["status"] != COMPLETE_STATUSES[name]
                        or item["exact_original_LMP_baseline_target_trial_replay"] is not True):
                    raise ValueError("Complete original W/HW banks and target replay required")
            elif (item["status"] != "complete_history2_repeatability_session_banks"
                    or any(item[flag] is not True for flag in H2_REPLAY_FLAGS)
                    or item["original_HW_refitted"] is not False or item["original_W_refitted"] is not False):
                raise ValueError("Complete original H2 bank and all preserved original replay flags required")
            if item["session_id"] != session or item["source"]["session_id"] != session:
                raise ValueError("Original session identity changed")
            old = item["banks"][ANCHORS[name]]
            if (old["recipe"] != ANCHORS[name] or old["lambdas"] != list(LAMBDAS)
                    or [r["lambda"] for r in old["lambda_rows"]] != list(LAMBDAS)
                    or not all(np.isfinite(row[key]) for row in old["lambda_rows"] for key in SCORE_KEYS)):
                raise ValueError("Complete finite original five-penalty banks required")
            banks[name][session], sources[name][session] = old, item["source"]
        item = read_json(args.blend_bank_root/session/"score_banks.json")
        validate_blend_session(item, session)
        blends[session] = item["bank"]
    for name in ("W", "HW", "H2", "blend"):
        original = read_json(getattr(args, name.lower()+"_summary"))
        reports[name] = original
        if name in SUMMARY_STATUSES:
            matching = [row for row in original["panels"] if row["recipe"] == ANCHORS[name]]
            if original["status"] != SUMMARY_STATUSES[name] or len(matching) != 1:
                raise ValueError("Complete original W/HW selection summaries required")
            panels[name] = matching[0]
        elif name == "H2":
            if (original["status"] != "complete_history2_repeatability_panel"
                    or any(original[key] is not True for key in
                        ("all_original_replays", "all_new_saved_scores_replayed", "original_selectors_independently_replayed"))
                    or original["original_HW_refitted"] is not False or original["original_W_refitted"] is not False
                    or original["recipes"][0] != ANCHORS[name]):
                raise ValueError("Complete replayed original H2 selection summary required")
            panels[name] = {**original["candidate"], "recipe": ANCHORS[name]}
        else:
            if (original["status"] != "complete_history_blend_panel" or original["recipe"] != ANCHORS[name]
                    or any(original[key] is not True for key in ("original_selectors_independently_replayed",
                        "all_selected_prediction_scores_replayed", "all_original_component_arrays_support_replay",
                        "all_original_component_scores_replay"))
                    or set(original["selected_prediction_paths"]) != set(SESSIONS)):
                raise ValueError("Complete original blend selected arrays and selector replay required")
            panels[name] = original["blend"]
    rows = validate_anchor_selections(banks, panels, blends)
    return {"banks": banks, "panels": panels, "blend_banks": blends, "original_rows": rows,
            "reports": reports, "sources": sources}


def check_support(saved, y, baseline, labels, ids):
    for key, expected in (("actual_xy", y), ("baseline_xy", baseline), ("directions", labels), ("trial_ids", ids)):
        if not np.array_equal(saved[key], expected):
            raise ValueError("Original anchor saved support differs: "+key)


def load_session_anchors(args, session, context):
    """Return Y, B, labels, IDs, W/HW banks, four selected predictions, source."""
    y, baseline, labels, ids, components, old_banks, source = load_components(args.w_bank_root, args.hw_bank_root, session)
    for name in ("W", "HW"):
        if old_banks[name] != context["banks"][name][session]:
            raise ValueError("Original W/HW score bank changed after selector replay")
    old = context["banks"]["H2"][session]
    old_source = context["sources"]["H2"][session]
    if (old_source["native_bin_seconds"] != .03 or old_source["native_LMP_offsets"] != [-3, 14]
            or old_source["native_target_offsets"] != [0, 14] or old_source["lag_offsets"] != [0, -1, -2, -3]
            or old_source["complete_trials"] != len(y)
            or old_source["common_LMP_electrodes"] != source["common_LMP_electrodes"]
            or old["physical_electrode_ids"] != source["common_LMP_electrodes"]
            or old["physical_electrode_count"] != len(source["common_LMP_electrodes"])
            or old["reader_LMP_offsets"] != [-3, 14] or old["effective_LMP_offsets"] != [-1, 14]
            or old["lag_offsets"] != [0, -1] or old["native_XY_scaled"] is not False
            or old["original_trial_ids"] != ids.tolist() or old["original_direction_labels"] != labels.tolist()):
        raise ValueError("Original H2 reader/effective history and physical/target support required")
    path = args.h2_bank_root/session/(ANCHORS["H2"]+"_predictions.npz")
    with np.load(path, allow_pickle=False) as saved:
        check_support(saved, y, baseline, labels, ids)
        # Fixed restoration also verifies saved native precision and shape.
        restore_target_layout(saved["actual_xy"])
        restore_target_layout(saved["baseline_xy"])
        h2 = saved["candidate_xy_by_lambda"]
    if (h2.shape != (5, *y.shape) or h2.dtype != np.dtype("float64")
            or not h2.flags.c_contiguous or not np.isfinite(h2).all()):
        raise ValueError("Literal finite C-order original H2 five-penalty predictions required")
    replay = score_bank(y, labels, baseline, h2, ANCHORS["H2"])
    if any(replay[key] != old[key] for key in ("lambdas", *SUPPORT_KEYS, "lambda_rows")):
        raise ValueError("Exact original H2 all-penalty score replay failed")
    original_rows = {name: rows[session] for name, rows in context["original_rows"].items()}
    selected = {name: components[name][original_rows[name]["lambda_index"]] for name in ("W", "HW")}
    selected["H2"] = h2[original_rows["H2"]["lambda_index"]]
    blend_path = Path(context["reports"]["blend"]["selected_prediction_paths"][session])
    with np.load(blend_path, allow_pickle=False) as saved:
        check_support(saved, y, baseline, labels, ids)
        selected["blend"] = saved["candidate_xy"]
        if (float(saved["selected_lambda"]) != original_rows["blend"]["lambda"]
                or float(saved["selected_alpha"]) != original_rows["blend"]["alpha"]):
            raise ValueError("Original selected blend parameters changed")
    blend_row = original_rows["blend"]
    exact_blend = blend_prediction(components["W"][blend_row["lambda_index"]],
                                  components["HW"][blend_row["lambda_index"]], blend_row["alpha"])
    if not np.array_equal(exact_blend, selected["blend"]):
        raise ValueError("Original selected blend array failed literal reconstruction")
    for li in range(5):
        half = blend_prediction(components["W"][li], components["HW"][li], .5)
        scores = prediction_scores(y, baseline, labels, half)
        if any(scores[key] != context["blend_banks"][session]["cells"][5*li+2][key] for key in SCORE_KEYS):
            raise ValueError("Exact original half-blend score replay failed")
    for name, prediction in selected.items():
        if (prediction.shape != y.shape or prediction.dtype != np.dtype("float64")
                or not prediction.flags.c_contiguous or not np.isfinite(prediction).all()):
            raise ValueError("Literal finite original selected prediction required")
        scores = prediction_scores(y, baseline, labels, prediction)
        if any(scores[key] != original_rows[name][key] for key in SCORE_KEYS):
            raise ValueError("Exact selected original anchor score replay failed: "+name)
    for bank in (old, context["blend_banks"][session]):
        if any(bank[key] != old_banks["W"][key] for key in SUPPORT_KEYS):
            raise ValueError("All four anchors must retain original full target/baseline support")
    source = {**source, "anchor_effective_input_offsets": {"W": [0, 14], "HW": [-3, 14], "H2": [-1, 14]},
        "H2_reader_offsets": [-3, 14], "H2_prediction_path": str(path), "blend_selected_prediction_path": str(blend_path)}
    return y, baseline, labels, ids, components, selected, source


def attach_references(bank, session, context):
    bank["original_selected_score_rows"] = {name: rows[session] for name, rows in context["original_rows"].items()}
    bank["shared_lambda_score_rows"] = {name: context["banks"][name][session]["lambda_rows"] for name in ("W", "HW")}
    bank["shared_lambda_score_rows"]["half_blend"] = [context["blend_banks"][session]["cells"][5*i+2] for i in range(5)]
    # Pilot includes the adopted descriptive comparison arithmetic without selecting on that session.
    for cell in bank["cells"]:
        references = {name+"_independently_selected": row for name, row in bank["original_selected_score_rows"].items()}
        references.update({name+"_schedule_lambda_matched": rows[cell["lambda_index"]]
                           for name, rows in bank["shared_lambda_score_rows"].items()})
        cell["fixed_reference_contrasts"] = {name: full_model_contrast(cell, row, bank["denominator"])
                                             for name, row in references.items()}
        reverse = bank["cells"][5*cell["lambda_index"]+4-cell["slope_index"]]
        cell["same_lambda_reversal_contrast"] = full_model_contrast(cell, reverse, bank["denominator"])


def validate_session(item, session):
    if (item["status"] != SESSION_STATUS or item["session_id"] != session
            or any(item[flag] is not True for flag in REPLAY_FLAGS)
            or item["provider_source_values_read"] is not False or item["neural_refits"] != 0
            or item["scientific_trial_count_increment"] != 0 or item["literal_null_control_count_increment"] != 0
            or item["alternative_explanation_count_increment"] != 0 or item["promotion_authorized"] is not False):
        raise ValueError("Complete artifact-only session with all four exact original anchor replays required")
    bank, source = item["bank"], item["source"]
    validate_bank(bank)
    if (source["session_id"] != session or source["native_target_offsets"] != [0, 14]
            or source["native_bin_seconds"] != .03 or source["provider_source_values_read"] is not False
            or source["component_input_offsets"] != {"W": [0, 14], "HW": [-3, 14]}
            or source["anchor_effective_input_offsets"] != {"W": [0, 14], "HW": [-3, 14], "H2": [-1, 14]}
            or source["H2_reader_offsets"] != [-3, 14] or sum(bank["direction_counts"]) != source["complete_trials"]
            or set(bank["original_selected_score_rows"]) != set(ANCHORS)
            or set(bank["shared_lambda_score_rows"]) != {"W", "HW", "half_blend"}):
        raise ValueError("Exact full original support and all fixed anchor rows required")
    for li in range(5):
        if any(bank["cells"][5*li+2][key] != bank["shared_lambda_score_rows"]["half_blend"][li][key] for key in SCORE_KEYS):
            raise ValueError("Zero-slope cell must exactly replay original half-blend")
    for cell in bank["cells"]:
        references = {name+"_independently_selected": row for name, row in bank["original_selected_score_rows"].items()}
        references.update({name+"_schedule_lambda_matched": rows[cell["lambda_index"]]
                           for name, rows in bank["shared_lambda_score_rows"].items()})
        expected = {name: full_model_contrast(cell, row, bank["denominator"]) for name, row in references.items()}
        reverse = bank["cells"][5*cell["lambda_index"]+4-cell["slope_index"]]
        if (cell["fixed_reference_contrasts"] != expected
                or cell["same_lambda_reversal_contrast"] != full_model_contrast(cell, reverse, bank["denominator"])):
            raise ValueError("Prespecified direct anchor and fixed-lambda reversal contrasts must replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("session", "summarize"):
        child = sub.add_parser(mode)
        for name in (*PATH_OPTIONS, "out"):
            child.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
        if mode == "session":
            child.add_argument("--session", choices=SESSIONS, required=True)
            child.add_argument("--max-seconds", type=int, default=60)
        else:
            child.add_argument("--bank-root", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread numerical settings required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1:
        raise ValueError("An actual allocated CPU is required")
    if args.mode == "session" and not 0 < args.max_seconds <= 60:
        raise ValueError("Adopted artifact session bound is at most 60 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"recipe": RECIPE, "status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_trial_count_increment": 0, "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": 0, "neural_refits": 0,
        "provider_source_values_read": False, "promotion_authorized": False, "search_complete": False}
    def deadline(*_):
        raise TimeoutError("Adopted timevarying blend artifact session deadline reached")
    def check():
        if args.mode == "session" and time.monotonic()-started >= args.max_seconds:
            deadline()
    if args.mode == "session":
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
    try:
        context = load_anchor_context(args)
        if args.mode == "session":
            y, baseline, labels, ids, components, originals, source = load_session_anchors(args, args.session, context)
            report.update(session_id=args.session, source=source, artifact_read_and_replay_seconds=time.monotonic()-started)
            scoring = time.monotonic()
            bank = score_session(y, baseline, labels, components["W"], components["HW"], originals, check)
            attach_references(bank, args.session, context)
            report.update(bank=bank, status=SESSION_STATUS, mixture_score_seconds=time.monotonic()-scoring,
                          **{flag: True for flag in REPLAY_FLAGS})
            validate_session(report, args.session)
        else:
            sessions = {}
            for session in SESSIONS:
                item = read_json(args.bank_root/session/"score_banks.json")
                validate_session(item, session)
                bank = item["bank"]
                if bank["original_selected_score_rows"] != {name: rows[session] for name, rows in context["original_rows"].items()}:
                    raise ValueError("Fixed original selected rows changed since schedule scoring")
                expected_shared = {name: context["banks"][name][session]["lambda_rows"] for name in ("W", "HW")}
                expected_shared["half_blend"] = [context["blend_banks"][session]["cells"][5*i+2] for i in range(5)]
                if bank["shared_lambda_score_rows"] != expected_shared:
                    raise ValueError("Fixed shared-lambda diagnostic rows changed since schedule scoring")
                sessions[session] = bank
            result = summarize_banks(sessions, context["banks"], context["panels"], context["blend_banks"])
            selected_dir = args.out/"selected_predictions"
            selected_dir.mkdir(exist_ok=False)
            paths = {}
            for row in result["candidate"]["sessions"]:
                session, li = row["session_id"], row["lambda_index"]
                y, baseline, labels, ids, components, originals, _ = load_session_anchors(args, session, context)
                prediction = timevarying_prediction(components["W"][li], components["HW"][li], row["slope"])
                scores = prediction_scores(y, baseline, labels, prediction)
                if any(scores[key] != row[key] for key in SCORE_KEYS):
                    raise ValueError("Final selected prediction score did not replay exactly")
                for name in ANCHORS:
                    if bool(np.array_equal(prediction, originals[name])) != row["exact_prediction_equals_originals"][name]:
                        raise ValueError("Exact global duplicate evidence changed at final prediction replay")
                reverse = timevarying_prediction(components["W"][li], components["HW"][li], -row["slope"])
                reverse_row = sessions[session]["cells"][5*li+4-row["slope_index"]]
                reverse_scores = prediction_scores(y, baseline, labels, reverse)
                if any(reverse_scores[key] != reverse_row[key] for key in SCORE_KEYS):
                    raise ValueError("Same-lambda reversed prediction score did not replay exactly")
                path = selected_dir/(session+".npz")
                np.savez(path, actual_xy=y, baseline_xy=baseline, directions=labels, trial_ids=ids,
                    candidate_xy=prediction, reversed_slope_xy=reverse, selected_lambda=row["lambda"],
                    selected_slope=row["slope"], selected_alpha_by_bin=np.asarray(row["alpha_by_bin"]))
                paths[session] = str(path)
            report.update(result, status=PANEL_STATUS, bank_root=str(args.bank_root), selected_prediction_paths=paths,
                original_anchor_summaries={name: str(getattr(args, name.lower()+"_summary")) for name in ANCHORS},
                all_selected_prediction_scores_replayed=True, all_selected_reversal_scores_replayed=True,
                all_exact_duplicate_flags_replayed=True, **{flag: True for flag in REPLAY_FLAGS})
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
