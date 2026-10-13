#!/usr/bin/env python3
"""Compact evidence from the complete covariance-SVD panel, without refits."""
import argparse
import json
from pathlib import Path

from extract_primary_relevance import bounds, seed_summaries, thin_panel
from primary_covariance_svd import RECIPES


def diagnostics(rows):
    folds = [f for r in rows for f in r["selected_projection_diagnostics"]]
    out = {"fold_count": len(folds)}
    for arm, values in (("candidate", [f["paired"] for f in folds]),
                        ("control", [c for f in folds for c in f["controls"]])):
        out[arm] = {k + "_range": bounds(v[k] for v in values)
                    for k in ("retained_components", "trace", "numerical_rank", "effective_df")}
        out[arm]["zero_rank_count"] = sum(v["retained_components"] == 0 for v in values)
        out[arm]["singular_value_range"] = bounds(s for v in values for s in v["singular_values"])
    out["projector_overlap_range"] = bounds(v for f in folds for v in f["projector_overlap_by_replicate"])
    out["control_to_candidate_trace_ratio"] = bounds(c["trace"] / f["paired"]["trace"]
        for f in folds if f["paired"]["trace"] > 0 for c in f["controls"]) if any(f["paired"]["trace"] > 0 for f in folds) else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    if not report.get("status", "").startswith("complete_"):
        raise ValueError("Only a complete scientific summary may be extracted")
    panels = {p["recipe"]: p for p in report["panels"]}
    if set(panels) != set(RECIPES) or any(len(p["sessions"]) != 16 for p in panels.values()):
        raise ValueError("Both full original sixteen-session panels required")
    candidate, control = (panels[k] for k in RECIPES)
    keys = ("status", "scientific_trial_count_increment", "total_valid_primary_trials_including_prior",
            "post_initial_comparisons", "post_initial_controls", "structural_duplicate",
            "structural_full_span_duplicate", "structural_baseline_duplicate", "compression_screens",
            "all_compression_screens_pass")
    result = {"source_summary": str(args.summary), **{k: report[k] for k in keys},
        "panels": [thin_panel(candidate), thin_panel(control)],
        "candidate_minus_mean_control": thin_panel(report["basis_label_control"]),
        "anchor_contrasts": {k: thin_panel(v) for k, v in report["anchor_contrasts"].items()},
        "projection_diagnostics": diagnostics(candidate["sessions"]),
        "projection_diagnostics_by_animal": {a: diagnostics([r for r in candidate["sessions"] if r["animal"] == a])
                                             for a in ("Mihili", "Chewie")},
        "control_seeds": seed_summaries(candidate, control)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
