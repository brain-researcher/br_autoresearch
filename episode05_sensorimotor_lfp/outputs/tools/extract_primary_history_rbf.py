#!/usr/bin/env python3
"""Release compact evidence only from a complete verified HW105 panel."""
import argparse
import json
from pathlib import Path

from primary_history_rbf import ANCHORS, RECIPES, REPLICATES, duplicate_accounting
from primary_reference import ANIMALS, SESSIONS
from run_primary_history_rbf import PANEL_STATUS
from source_adapter import compute_only


ANIMAL_NAMES = ("Mihili", "Chewie")
SESSION_FIELDS = ("session_id", "animal", "lambda_index", "lambda", "alpha", "cell_index",
                  "reference_lambda", "baseline_R2", "candidate_R2", "candidate_SSE",
                  "wrong_pair_SSE", "delta_R2", "identity_increment")
OMISSION_FIELDS = ("excluded_session", "omitted_session", "primary_increment", "identity_increment")
SCREEN_FIELDS = ("primary_at_least_0p01", "identity_at_least_0p005", "both_primary_animals_positive",
                 "both_identity_animals_positive", "strict_session_majorities",
                 "every_within_animal_primary_omission_positive",
                 "every_within_animal_identity_omission_positive")
CONTRAST_NAMES = ("candidate_minus_control", "candidate_minus_HW", "candidate_minus_K",
                  "candidate_minus_blend", "candidate_minus_HW_lambda_matched", "candidate_minus_K_lambda_matched")
MAX_EXPORT_BYTES = 512 * 1024


def scalars(item, fields):
    """Copy only named scalar values, never a nested saved bank or prediction."""
    result = {}
    for key in fields:
        if key not in item:
            continue
        value = item[key]
        if type(value) not in (str, bool, int, float) and value is not None:
            raise ValueError("A compact evidence field is not scalar: " + key)
        result[key] = value
    return result


def compact_panel(value):
    """Project existing effects without recalculating or expanding old anchors."""
    rows = value["sessions"]
    if len(rows) != len(SESSIONS) or {row["session_id"] for row in rows} != set(SESSIONS):
        raise ValueError("Compact evidence requires all sixteen session rows")
    if any(row["animal"] != ANIMALS[row["session_id"]] for row in rows):
        raise ValueError("Compact evidence session/animal labels differ")
    result = scalars(value, ("primary_increment", "identity_increment"))
    for field in ("primary_by_animal", "identity_by_animal", "primary_positive_session_counts"):
        result[field] = scalars(value[field], ANIMAL_NAMES)
    result["sessions"] = [scalars(row, SESSION_FIELDS) for row in rows]
    influence = value["leave_one_session_fixed_prediction_influence"]
    if len(influence) != len(SESSIONS):
        raise ValueError("All fixed-prediction omission metrics are required")
    result["leave_one_session_fixed_prediction_influence"] = [scalars(row, OMISSION_FIELDS) for row in influence]
    result["leave_one_session_within_animal"] = {}
    for animal in ANIMAL_NAMES:
        omissions = value["leave_one_session_within_animal"][animal]
        if len(omissions) != sum(ANIMALS[s] == animal for s in SESSIONS):
            raise ValueError("All within-animal omission metrics are required")
        result["leave_one_session_within_animal"][animal] = [scalars(row, OMISSION_FIELDS) for row in omissions]
    return result


def extract(path):
    report = json.loads(Path(path).read_text())
    if (report["status"] != PANEL_STATUS or report["promotion_authorized"] is not False
            or report["all_new_saved_scores_replayed"] is not True
            or report["all_original_anchor_scores_replayed"] is not True
            or report["original_selectors_independently_replayed"] is not True):
        raise ValueError("Complete original/new replayed sixteen-session summary required")
    accounting = duplicate_accounting(report["selected_candidate_exact_equality"], report["selected_control_exact_equality"])
    if any(report[k] != v for k, v in accounting.items()):
        raise ValueError("Literal global duplicate accounting differs")
    names = ("all_reference_margin_screens_pass", "candidate_trial_count_increment", "control_trial_count_increment",
             "scientific_trial_count_increment", "literal_null_control_count_increment",
             "alternative_explanation_count_increment", "eligible_for_comparable_primary_incumbent")
    seeds = report["candidate_minus_each_control"]
    if len(seeds) != len(REPLICATES) or [row["replicate"] for row in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed control contrasts are required")
    return {"source": str(path), "status": report["status"], **scalars(report, names),
        "recipes": list(RECIPES),
        "candidate": compact_panel(report["candidate"]),
        "control": compact_panel(report["control"]),
        "reference_contrasts": {name: compact_panel(report["reference_contrasts"][name]) for name in CONTRAST_NAMES},
        "candidate_minus_each_control": [{"replicate": row["replicate"], **compact_panel(row)} for row in seeds],
        "original_anchors": {name: compact_panel(report["original_anchors"][name]) for name in ANCHORS},
        "reference_margin_screens": scalars(report["reference_margin_screens"], SCREEN_FIELDS),
        "candidate_exact_duplicate_of": accounting["candidate_exact_duplicate_of"],
        "control_exact_duplicate_of": accounting["control_exact_duplicate_of"],
        "count_order": list(RECIPES),
        "all_new_saved_scores_replayed": True, "all_original_anchor_scores_replayed": True,
        "original_selectors_independently_replayed": True,
        "interpretation": "Fixed nonlinear weighted history and training-pair correspondence; no mechanism or isolated cross-lag claim",
        "promotion_authorized": False, "search_complete": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    rendered = json.dumps(extract(args.summary), indent=2, allow_nan=False)
    if len(rendered.encode("utf-8")) > MAX_EXPORT_BYTES:
        raise ValueError("Compact evidence exceeds the fixed 512 KiB export bound")
    print(rendered)
