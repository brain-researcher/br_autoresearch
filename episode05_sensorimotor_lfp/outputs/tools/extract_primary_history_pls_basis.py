#!/usr/bin/env python3
"""Release bounded scalar evidence from one complete stored-projection diagnostic."""
import argparse
import json
import math
from pathlib import Path

from extract_primary_history_rbf import MAX_EXPORT_BYTES, SCREEN_FIELDS, compact_panel, scalars
from primary_history_pls_basis import (CONTRAST_NAMES, ENSEMBLE_REFERENCES, RECIPE, REPLICATES,
                                      SINGLE_REFERENCES, duplicate_accounting)
from primary_reference import SESSIONS
from run_primary_history_pls_basis import PANEL_STATUS, REPLAY_FLAGS
from source_adapter import compute_only


def compact_diagnostics(rows):
    if len(rows) != 16 or {row["session_id"] for row in rows} != set(SESSIONS):
        raise ValueError("Complete original-session diagnostics required")
    result = []
    for row in rows:
        if len(row["folds"]) != 8 or [fold["held_direction"] for fold in row["folds"]] != list(range(8)):
            raise ValueError("Exactly eight original direction fits required")
        folds = []
        for fold in row["folds"]:
            value = scalars(fold, ("held_direction", "paired_components", "paired_trace", "paired_effective_df"))
            if type(value["paired_components"]) is not int or value["paired_components"] not in (0, 1, 2):
                raise ValueError("Original retained paired rank required")
            for key in ("basis_components", "basis_traces", "basis_effective_df"):
                vector = fold[key]
                if (len(vector) != 8 or any(type(item) not in (int, float) or not math.isfinite(item) for item in vector)
                        or (key == "basis_components" and any(type(item) is not int or item not in (0, 1, 2) for item in vector))):
                    raise ValueError("Exactly eight finite scalar projection diagnostics required")
                value[key] = list(vector)
            folds.append(value)
        result.append({**scalars(row, ("session_id", "lambda")), "folds": folds})
    return result


def extract(path):
    report = json.loads(Path(path).read_text())
    if (report["status"] != PANEL_STATUS or report["recipe"] != RECIPE
            or report["promotion_authorized"] is not False or report["search_complete"] is not False
            or report["new_selector_fitted"] is not False
            or report["candidate_and_basis_capacity_matched"] is not False
            or report["basis_and_full_target_score_Grams_identical"] is not True
            or any(report[key] is not True for key in REPLAY_FLAGS)
            or report["single_duplicate_reference_names"] != list(SINGLE_REFERENCES)
            or report["ensemble_duplicate_reference_names"] != list(ENSEMBLE_REFERENCES)):
        raise ValueError("Complete preserved-selector/all-replay basis-only panel required")
    accounting = duplicate_accounting(report["selected_exact_single_equality"], report["selected_exact_ensemble_equality"])
    if any(report[key] != value for key, value in accounting.items()):
        raise ValueError("Global single-panel and entire-ensemble duplicate accounting differs")
    result = {"source": str(path), "status": PANEL_STATUS, "recipe": RECIPE,
        **scalars(accounting, ("scientific_trial_count_increment", "alternative_explanation_count_increment",
            "literal_null_control_count_increment", "candidate_trial_count_increment",
            "eligible_for_comparable_primary_incumbent", "updates_primary_incumbent")),
        "basis_only": compact_panel(report["basis_only"]),
        "reference_contrasts": {name: compact_panel(report["reference_contrasts"][name]) for name in CONTRAST_NAMES},
        "original_anchors": {name: compact_panel(report["original_anchors"][name])
                             for name in ("paired_PLS", "full_target_control", "HW", "blend")},
        "comparison_screens": {name: scalars(report["comparison_screens"][name], SCREEN_FIELDS) for name in CONTRAST_NAMES},
        "projection_diagnostics": compact_diagnostics(report["projection_diagnostics"]),
        "exact_duplicate_single_panels": accounting["exact_duplicate_single_panels"],
        "exact_duplicate_ensembles": accounting["exact_duplicate_ensembles"],
        "single_duplicate_reference_names": list(SINGLE_REFERENCES),
        "ensemble_duplicate_reference_names": list(ENSEMBLE_REFERENCES),
        "count_order": [RECIPE], **{key: True for key in REPLAY_FLAGS},
        "new_selector_fitted": False, "candidate_and_basis_capacity_matched": False,
        "basis_and_full_target_score_Grams_identical": True,
        "interpretation": "Conditional projection/head correspondence diagnostic; true heads preserve target signal; not a literal null or promotable candidate",
        "promotion_authorized": False, "search_complete": False}
    for name in ("paired_PLS_minus_each_basis", "each_basis_minus_matched_full_target"):
        seeds = report[name]
        if len(seeds) != 8 or [row["replicate"] for row in seeds] != list(REPLICATES):
            raise ValueError("All eight original matched draw contrasts required")
        result[name] = [{"replicate": row["replicate"], **compact_panel(row)} for row in seeds]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    rendered = json.dumps(extract(args.summary), indent=2, allow_nan=False)
    if len(rendered.encode("utf-8")) > MAX_EXPORT_BYTES:
        raise ValueError("Compact evidence exceeds the fixed 512 KiB export bound")
    print(rendered)
