#!/usr/bin/env python3
"""Read only a complete sixteen-session elapsed-only HW summary."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from primary_elapsed_only import RECIPE
from primary_reference import ANIMALS, SESSIONS
from source_adapter import compute_only

PANEL_NAMES = ("elapsed_sensitivity", "original_HW_versus_elapsed_baseline",
    "original_B_versus_elapsed_baseline", "ET_HW_versus_original_B_descriptive",
    "original_HW_versus_original_B_replayed")
CONTRAST_NAMES = ("HW_independently_selected", "HW_ET_HW_lambda_matched")


def _panel(item):
    rows = item["sessions"]
    if (len(rows) != 16 or {r["session_id"] for r in rows} != set(SESSIONS)
            or any(r["animal"] != ANIMALS[r["session_id"]] for r in rows)
            or any(not math.isfinite(r[k]) for r in rows for k in ("delta_R2", "identity_increment"))):
        raise ValueError("All sixteen finite original session estimates required")
    omissions = item["leave_one_session_within_animal"]
    ranges = {}
    for animal in ("Mihili", "Chewie"):
        expected = {s for s in SESSIONS if ANIMALS[s] == animal}
        values = omissions[animal]
        if (len(values) != len(expected) or {r["omitted_session"] for r in values} != expected
                or any(not math.isfinite(r[k]) for r in values for k in ("primary_increment", "identity_increment"))):
            raise ValueError("All fixed-prediction within-animal omissions required")
        ranges[animal] = {k: {"minimum": min(r[k] for r in values),
            "maximum": max(r[k] for r in values),
            "nonpositive_omissions": [r["omitted_session"] for r in values if r[k] <= 0]}
            for k in ("primary_increment", "identity_increment")}
    keys = ("session_id", "animal", "lambda", "lambda_index", "ET_HW_lambda", "HW_lambda",
            "delta_R2", "identity_increment", "candidate_SSE", "wrong_pair_SSE", "candidate_R2")
    return {**{key: item[key] for key in ("primary_increment", "identity_increment",
        "primary_by_animal", "identity_by_animal", "primary_positive_session_counts")},
        "identity_positive_session_counts": {a: sum(r["identity_increment"] > 0 for r in rows if r["animal"] == a)
                                             for a in ("Mihili", "Chewie")},
        "lambda_counts": dict(Counter(str(r["lambda"]) for r in rows if "lambda" in r)),
        "within_animal_omission_ranges": ranges,
        "leave_one_session_within_animal": omissions,
        "sessions": [{key: r[key] for key in keys if key in r} for r in rows]}


def extract(path):
    report = json.loads(Path(path).read_text())
    if (report["status"] != "complete_elapsed_only_HW_panel" or report["recipe"] != RECIPE
            or report["sensitivity_only"] is not True or report["promotion_authorized"] is not False
            or report["updates_primary_incumbent"] is not False or report["original_HW_refitted"] is not False
            or report["all_original_HW_neural_state_replay"] is not True
            or report["all_original_HW_saved_arrays_and_scoring_replay"] is not True
            or report["original_HW_anchor"]["refitted"] is not False
            or report["scientific_trial_count_increment"] not in (0, 1)
            or report["alternative_explanation_stress_count_increment"] != report["scientific_trial_count_increment"]
            or report["null_control_trial_count_increment"] != 0
            or set(report["full_model_anchor_contrasts"]) != set(CONTRAST_NAMES)):
        raise ValueError("Only a complete nonpromotable elapsed-only panel with exact original replays is permitted")
    panels = {name: _panel(report[name]) for name in PANEL_NAMES}
    contrasts = {name: _panel(report["full_model_anchor_contrasts"][name]) for name in CONTRAST_NAMES}
    by_session = {name: {r["session_id"]: r for r in report[name]["sessions"]} for name in PANEL_NAMES}
    decomposition = []
    for session in SESSIONS:
        hw_b = by_session["original_HW_versus_original_B_replayed"][session]["delta_R2"]
        b_t = by_session["original_B_versus_elapsed_baseline"][session]["delta_R2"]
        hw_t = by_session["original_HW_versus_elapsed_baseline"][session]["delta_R2"]
        et_b = by_session["ET_HW_versus_original_B_descriptive"][session]["delta_R2"]
        et_t = by_session["elapsed_sensitivity"][session]["delta_R2"]
        decomposition.append({"session_id": session, "animal": ANIMALS[session],
            "HW_versus_B": hw_b, "B_versus_T": b_t, "HW_versus_T": hw_t,
            "HW_decomposition_roundoff": hw_b+b_t-hw_t,
            "ET_HW_versus_B": et_b, "ET_HW_versus_T": et_t,
            "ET_HW_decomposition_roundoff": et_b+b_t-et_t})
    return {"status": report["status"], "recipe": RECIPE, "path": str(path),
        "job_id": report["job_id"], "elapsed_seconds": report["elapsed_seconds"],
        "maximum_RSS_KiB": report["maximum_RSS_KiB"],
        "accounting": {key: report[key] for key in ("scientific_trial_count_increment",
            "alternative_explanation_stress_count_increment", "null_control_trial_count_increment",
            "structural_original_HW_duplicate", "structural_time_only_duplicate",
            "updates_primary_incumbent", "sensitivity_only", "promotion_authorized")},
        "exact_replay": {key: report[key] for key in ("all_original_HW_neural_state_replay",
            "all_original_HW_saved_arrays_and_scoring_replay", "original_HW_refitted")},
        "screens": {key: report[key] for key in ("elapsed_sensitivity_screens",
            "all_elapsed_sensitivity_screens_pass", "original_HW_elapsed_baseline_screens",
            "all_original_HW_elapsed_baseline_screens_pass")},
        "panels": panels, "direct_full_prediction_contrasts": contrasts,
        "baseline_decomposition": decomposition,
        "maximum_absolute_decomposition_roundoff": {prefix: max(abs(r[prefix+"_decomposition_roundoff"])
            for r in decomposition) for prefix in ("HW", "ET_HW")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps(extract(args.summary), indent=2, allow_nan=False))
