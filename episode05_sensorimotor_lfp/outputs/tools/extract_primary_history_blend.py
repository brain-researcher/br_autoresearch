#!/usr/bin/env python3
"""Extract only a complete sixteen-session original-support history blend."""
import argparse
from collections import Counter
import json
from pathlib import Path

from extract_primary_relevance import thin_panel
from plot_repeatability_weight import validate_rows
from primary_history_blend import COMPONENTS, RECIPE
from primary_reference import SESSIONS
from source_adapter import compute_only


def extract(path):
    report = json.loads(Path(path).read_text())
    count = report["scientific_trial_count_increment"]
    if (report["status"] != "complete_history_blend_panel" or report["recipe"] != RECIPE
            or type(count) is not int or count not in (0, 1)
            or count != int(not report["structural_duplicate_of"])
            or report["literal_null_control_count_increment"] != 0
            or report["alternative_explanation_count_increment"] != 0
            or report["neural_refits"] != 0 or report["provider_source_values_read"] is not False
            or report["promotion_authorized"] is not False
            or report["original_selectors_independently_replayed"] is not True
            or report["all_selected_prediction_scores_replayed"] is not True
            or report["all_original_component_arrays_support_replay"] is not True
            or report["all_original_component_scores_replay"] is not True
            or set(report["selected_prediction_paths"]) != set(SESSIONS)
            or set(report["original_components"]) != set(COMPONENTS)
            or set(report["reference_contrasts"]) != {name+"_"+kind for name in COMPONENTS
                for kind in ("independently_selected", "blend_lambda_matched")}):
        raise ValueError("Only a complete original-support mixture with exact endpoint/selected-array replays is permitted")
    candidate = report["blend"]
    for panel in (candidate, *report["reference_contrasts"].values(), *report["original_components"].values()):
        validate_rows(panel["sessions"])
    keys = ("status", "recipe", "scientific_trial_count_increment", "literal_null_control_count_increment",
        "alternative_explanation_count_increment", "eligible_for_comparable_primary_incumbent",
        "structural_duplicate_of", "reference_margin_screens", "all_reference_margin_screens_pass",
        "neural_refits", "provider_source_values_read", "promotion_authorized", "search_complete",
        "original_component_summaries", "original_selectors_independently_replayed",
        "all_selected_prediction_scores_replayed", "all_original_component_arrays_support_replay",
        "all_original_component_scores_replay", "elapsed_seconds", "maximum_RSS_KiB", "job_id")
    return {"source_summary": str(path), **{key: report[key] for key in keys},
        "candidate": thin_panel(candidate),
        "alpha_counts": dict(Counter(str(row["alpha"]) for row in candidate["sessions"])),
        "joint_lambda_alpha_counts": dict(Counter(f"lambda={row['lambda']},alpha={row['alpha']}"
            for row in candidate["sessions"])),
        "selected_sessions": [{key: row[key] for key in ("session_id", "animal", "lambda", "alpha",
            "delta_R2", "identity_increment", "literal_endpoint_reused", "exactly_matches_original_selected_components")}
            for row in candidate["sessions"]],
        "reference_contrasts": {name: thin_panel(panel) for name, panel in report["reference_contrasts"].items()},
        "original_components": {name: thin_panel(panel) for name, panel in report["original_components"].items()},
        "selected_prediction_paths": report["selected_prediction_paths"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps(extract(args.summary), indent=2, allow_nan=False))
