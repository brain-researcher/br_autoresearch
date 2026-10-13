#!/usr/bin/env python3
"""Extract only a complete replayed original-support linear time blend panel."""
import argparse
from collections import Counter
import json
from pathlib import Path

from extract_primary_capacity_matched_surrogate import brief_panel
from primary_reference import SESSIONS
from primary_timevarying_history_blend import ANCHORS, RECIPE
from run_primary_timevarying_history_blend import PANEL_STATUS, REPLAY_FLAGS
from source_adapter import compute_only


def extract(path):
    report = json.loads(Path(path).read_text())
    count = report["scientific_trial_count_increment"]
    expected = {name+"_independently_selected" for name in ANCHORS}
    expected.update({"W_schedule_lambda_matched", "HW_schedule_lambda_matched", "half_blend_schedule_lambda_matched"})
    if (report["status"] != PANEL_STATUS or report["recipe"] != RECIPE
            or type(count) is not int or count not in (0, 1) or count != int(not report["structural_duplicate_of"])
            or report["eligible_for_comparable_primary_incumbent"] != bool(count)
            or report["literal_null_control_count_increment"] != 0 or report["alternative_explanation_count_increment"] != 0
            or report["neural_refits"] != 0 or report["provider_source_values_read"] is not False
            or report["promotion_authorized"] is not False or report["reversal_descriptive_only"] is not True
            or report["reversal_scientific_trial_count_increment"] != 0
            or any(report[flag] is not True for flag in (*REPLAY_FLAGS, "all_selected_prediction_scores_replayed",
                "all_selected_reversal_scores_replayed", "all_exact_duplicate_flags_replayed"))
            or set(report["selected_prediction_paths"]) != set(SESSIONS)
            or set(report["original_anchors"]) != set(ANCHORS) or set(report["reference_contrasts"]) != expected):
        raise ValueError("Only the complete exact-replay sixteen-session time schedule panel is permitted")
    rows = report["candidate"]["sessions"]
    accounting = ("scientific_trial_count_increment", "literal_null_control_count_increment",
        "alternative_explanation_count_increment", "eligible_for_comparable_primary_incumbent",
        "structural_duplicate_of", "reference_margin_screens", "all_reference_margin_screens_pass",
        "reversal_descriptive_only", "reversal_scientific_trial_count_increment", "neural_refits",
        "provider_source_values_read", "promotion_authorized", "search_complete")
    return {"source_summary": str(path), "status": report["status"], "recipe": RECIPE,
        **{key: report[key] for key in ("job_id", "elapsed_seconds", "maximum_RSS_KiB")},
        "accounting": {key: report[key] for key in accounting},
        "exact_replay": {key: report[key] for key in (*REPLAY_FLAGS, "all_selected_prediction_scores_replayed",
            "all_selected_reversal_scores_replayed", "all_exact_duplicate_flags_replayed")},
        "candidate": brief_panel(report["candidate"], sessions=True),
        "slope_counts": dict(Counter(str(row["slope"]) for row in rows)),
        "slope_sign_counts": dict(Counter("negative" if row["slope"] < 0 else "positive" if row["slope"] > 0 else "zero" for row in rows)),
        "joint_lambda_slope_counts": dict(Counter(f"lambda={row['lambda']},slope={row['slope']}" for row in rows)),
        "selected_sessions": [{key: row[key] for key in ("session_id", "animal", "lambda", "slope", "alpha_by_bin",
            "delta_R2", "identity_increment", "exact_prediction_equals_originals")} for row in rows],
        "reference_contrasts": {name: brief_panel(value, sessions=True) for name, value in report["reference_contrasts"].items()},
        "original_anchors": {name: brief_panel(value, sessions=True) for name, value in report["original_anchors"].items()},
        "reversed_slope": brief_panel(report["reversed_slope"], sessions=True),
        "selected_minus_reversed": brief_panel(report["selected_minus_reversed"], sessions=True),
        "original_anchor_summaries": report["original_anchor_summaries"],
        "selected_prediction_paths": report["selected_prediction_paths"],
        "interpretation": "Reversal is descriptive only; mixed selected signs do not identify a stable timing pattern. Mean coefficient does not establish mean contribution or equal capacity."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps(extract(args.summary), indent=2, allow_nan=False))
