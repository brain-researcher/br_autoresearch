#!/usr/bin/env python3
"""Release bounded scalar evidence only after the complete PLS2 panel replays."""
import argparse
import json
from pathlib import Path

from extract_primary_history_rbf import MAX_EXPORT_BYTES, SCREEN_FIELDS, compact_panel, scalars
from primary_history_pls import CONTRAST_NAMES, DUPLICATE_ANCHORS, RECIPES, REPLICATES, duplicate_accounting
from primary_reference import SESSIONS
from run_primary_history_pls import PANEL_STATUS
from source_adapter import compute_only


COMPARISON_NAMES = ("candidate_minus_control", "candidate_minus_HW", "candidate_minus_blend")
DIAGNOSTIC_FIELDS = ("retained_components", "stop_reason", "trace", "numerical_rank", "effective_df")
REPLAY_FIELDS = ("all_new_saved_scores_replayed", "all_original_anchor_scores_replayed",
                 "original_selectors_independently_replayed")


def compact_projection_diagnostics(rows):
    """Release only fixed fold counts and scalar covariance diagnostics, never states."""
    if len(rows) != len(SESSIONS) or {row["session_id"] for row in rows} != set(SESSIONS):
        raise ValueError("Projection diagnostics require all sixteen sessions")
    result = []
    for row in rows:
        folds = row["folds"]
        if len(folds) != 8 or [fold["held_direction"] for fold in folds] != list(range(8)):
            raise ValueError("Exactly eight original held directions required")
        projected = []
        for fold in folds:
            if len(fold["controls"]) != len(REPLICATES):
                raise ValueError("All eight separately refitted control diagnostics required")
            diagnostics = [fold["candidate"], *fold["controls"]]
            for item in diagnostics:
                if (any(key not in item for key in DIAGNOSTIC_FIELDS)
                        or type(item["retained_components"]) is not int
                        or item["retained_components"] not in (0, 1, 2)):
                    raise ValueError("Complete fixed-cap projection diagnostics required")
            projected.append({"held_direction": fold["held_direction"],
                "candidate": scalars(fold["candidate"], DIAGNOSTIC_FIELDS),
                "controls": [scalars(item, DIAGNOSTIC_FIELDS) for item in fold["controls"]]})
        result.append({**scalars(row, ("session_id", "lambda")), "folds": projected})
    return result


def extract(path):
    report = json.loads(Path(path).read_text())
    if (report["status"] != PANEL_STATUS or report["promotion_authorized"] is not False
            or report["search_complete"] is not False
            or any(report[key] is not True for key in REPLAY_FIELDS)):
        raise ValueError("Complete independently selected and replayed sixteen-session PLS panel required")
    if (report["recipes"] != list(RECIPES)
            or report["duplicate_reference_names"] != list(DUPLICATE_ANCHORS)
            or report["duplicate_only_reference_panels"] != ["HW105", "covariance_SVD", "K"]):
        raise ValueError("Frozen PLS recipes and duplicate references required")
    accounting = duplicate_accounting(report["selected_candidate_exact_equality"],
                                     report["selected_control_exact_equality"])
    if any(report[key] != value for key, value in accounting.items()):
        raise ValueError("Literal complete-panel duplicate accounting differs")
    seeds = report["candidate_minus_each_control"]
    if len(seeds) != len(REPLICATES) or [row["replicate"] for row in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed target-label contrasts required")
    scalar_names = ("all_reference_margin_screens_pass", "candidate_trial_count_increment",
        "control_trial_count_increment", "scientific_trial_count_increment",
        "literal_null_control_count_increment", "alternative_explanation_count_increment",
        "eligible_for_comparable_primary_incumbent", "control_updates_primary_incumbent",
        "original_HW_refitted")
    return {"source": str(path), "status": report["status"], **scalars(report, scalar_names),
        "recipes": list(RECIPES),
        "candidate": compact_panel(report["candidate"]),
        "control": compact_panel(report["control"]),
        "reference_contrasts": {name: compact_panel(report["reference_contrasts"][name])
                                for name in CONTRAST_NAMES},
        "candidate_minus_each_control": [{"replicate": row["replicate"], **compact_panel(row)} for row in seeds],
        "original_anchors": {name: compact_panel(report["original_anchors"][name]) for name in ("HW", "blend")},
        "reference_margin_screens": scalars(report["reference_margin_screens"], SCREEN_FIELDS),
        "comparison_screens": {name: scalars(report["comparison_screens"][name], SCREEN_FIELDS)
                               for name in COMPARISON_NAMES},
        "projection_diagnostics": compact_projection_diagnostics(report["projection_diagnostics"]),
        "candidate_exact_duplicate_of": accounting["candidate_exact_duplicate_of"],
        "control_exact_duplicate_of": accounting["control_exact_duplicate_of"],
        "count_order": list(RECIPES),
        "duplicate_reference_names": list(DUPLICATE_ANCHORS),
        "duplicate_only_reference_panels": ["HW105", "covariance_SVD", "K"],
        **{key: True for key in REPLAY_FIELDS},
        "interpretation": "Fixed supervised compression versus full target-label refits; ranks and effective capacity can differ; no mechanism attribution",
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
