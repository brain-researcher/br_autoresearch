#!/usr/bin/env python3
"""Compact complete guide-intersection sensitivity; never source or partial banks."""
import argparse
import json
from pathlib import Path

from extract_primary_relevance import thin_panel
from plot_repeatability_weight import validate_rows
from primary_intersection import RECIPES, REPLICATES
from source_adapter import compute_only


def extract(report):
    if (report["status"] != "complete_intersection_sensitivity_panel"
            or report["sensitivity_only"] is not True or report["promotion_authorized"] is not False
            or report["incumbent_reset"] is not False or report["spike_contamination_identified"] is not False
            or report["reference_refitted"] is not False or report["recipe_order"] != list(RECIPES)
            or report["literal_null_control_count_increment"] != 0):
        raise ValueError("Only complete nonpromotable metadata-defined support sensitivities may be extracted")
    panels = {p["recipe"]: p for p in report["panels"]}
    if (len(report["panels"]) != 2 or set(panels) != set(RECIPES)
            or [r["replicate"] for r in report["random_support_panels"]] != list(REPLICATES)
            or set(report["reference_contrasts"]) != {"independently_selected", "intersection_lambda_matched"}):
        raise ValueError("Both recipes, all eight support draws and both full-HW anchors required")
    for panel in (*panels.values(), report["intersection_minus_random_support"],
                  *report["random_support_panels"], *report["reference_contrasts"].values()):
        validate_rows(panel["sessions"])
    charges = report["scientific_trial_count_by_recipe"]
    count = report["scientific_trial_count_increment"]
    if (any(type(charges[r]) is not int or charges[r] not in (0, 1) for r in RECIPES)
            or count != sum(charges.values()) or report["stress_diagnostic_count_increment"] != count
            or report["alternative_explanation_count_increment"] != count
            or report["total_valid_primary_trials_including_prior"] != report["prior_valid_comparisons"]+count
            or report["post_initial_comparisons"] != report["total_valid_primary_trials_including_prior"]-6
            or report["post_initial_null_controls"] != report["prior_literal_null_controls"]
            or report["post_initial_alternative_checks"] != report["prior_alternative_checks"]+count):
        raise ValueError("Ordered support-stress charges must not become literal null-control trials")
    keys = ("status", "scientific_trial_count_increment", "scientific_trial_count_by_recipe",
        "structural_duplicate_of", "stress_diagnostic_count_increment", "literal_null_control_count_increment",
        "alternative_explanation_count_increment", "prior_valid_comparisons", "prior_patience",
        "total_valid_primary_trials_including_prior", "post_initial_comparisons", "post_initial_null_controls",
        "post_initial_alternative_checks", "recipe_order", "prospective_patience_transitions", "incumbent_reset",
        "sensitivity_only", "promotion_authorized", "search_complete", "spike_contamination_identified", "limitation")
    def compact(panel):
        return {**thin_panel(panel), "sessions": [{k: row[k] for k in
            ("session_id", "animal", "lambda", "delta_R2", "identity_increment")}
            for row in panel["sessions"]]}
    support = []
    for irow, rrow in zip(panels[RECIPES[0]]["sessions"], panels[RECIPES[1]]["sessions"]):
        if irow["session_id"] != rrow["session_id"] or len(irow["selected_supports"]) != 1 or len(rrow["selected_supports"]) != 8:
            raise ValueError("Ordered full-cohort candidate and support draws required")
        chosen = irow["selected_supports"][0]
        random = rrow["selected_supports"]
        support.append({"session_id": irow["session_id"], "animal": irow["animal"],
            "intersection_electrodes": chosen["support"]["electrode_ids"],
            "electrode_count": len(chosen["support"]["electrode_ids"]), "lambda": irow["lambda"],
            "candidate_fit_diagnostics": chosen["selected_fit_diagnostics"],
            "random_supports": [r["support"] for r in random],
            "seed_ranges": {metric: [min(r[metric] for r in random), max(r[metric] for r in random)]
                            for metric in ("delta_R2", "identity_increment")},
            "finite_support_ranges_are_confidence_intervals": False})
    return {**{key: report[key] for key in keys}, "panels": [compact(panels[r]) for r in RECIPES],
        "intersection_minus_random_support": compact(report["intersection_minus_random_support"]),
        "reference_contrasts": {k: compact(v) for k, v in report["reference_contrasts"].items()},
        "random_support_panels": [{"replicate": r["replicate"], **compact(r)} for r in report["random_support_panels"]],
        "selected_support": support}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    compute_only()
    print(json.dumps({"source_summary": str(args.summary), **extract(json.loads(args.summary.read_text()))},
                     indent=2, allow_nan=False))
