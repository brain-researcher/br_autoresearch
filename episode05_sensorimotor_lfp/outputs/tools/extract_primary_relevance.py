#!/usr/bin/env python3
"""Print compact descriptive evidence from the complete EP05 relevance panel."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

ANIMALS = ("Mihili", "Chewie")
CANDIDATE = "lmp_inner_lodo_top16_ridge_v1"
CONTROL = "lmp_inner_lodo_top16_rank_labels8_v1"
METRICS = ("delta_R2", "identity_increment")


def bounds(values):
    values = list(values)
    return [min(values), max(values)]


def animal_values(rows, metric):
    return {animal: mean(row[metric] for row in rows if row["animal"] == animal)
            for animal in ANIMALS}


def omission_summary(rows):
    out = {}
    for metric in METRICS:
        original = animal_values(rows, metric)
        omitted = []
        for excluded in rows:
            remaining = [row for row in rows if row["session_id"] != excluded["session_id"]]
            animals = animal_values(remaining, metric)
            omitted.append({"excluded_session": excluded["session_id"],
                            "overall": mean(animals.values()), "animals": animals})
        original_overall = mean(original.values())
        out[metric] = {"overall_range": bounds(row["overall"] for row in omitted),
            "animal_ranges": {animal: bounds(row["animals"][animal] for row in omitted)
                              for animal in ANIMALS},
            "overall_positive_sign_changes": [row["excluded_session"] for row in omitted
                if (row["overall"] > 0) != (original_overall > 0)],
            "animal_positive_sign_changes": {animal: [row["excluded_session"] for row in omitted
                if (row["animals"][animal] > 0) != (original[animal] > 0)] for animal in ANIMALS}}
    return out


def thin_panel(panel):
    keys = ("recipe", "contrast", "primary_increment", "identity_increment", "primary_by_animal",
            "identity_by_animal", "primary_positive_session_counts", "reference_margin_screens",
            "all_reference_margin_screens_pass", "directional_screens", "all_directional_screens_pass")
    out = {key: panel[key] for key in keys if key in panel}
    out["fixed_prediction_omissions"] = omission_summary(panel["sessions"])
    out["lambda_counts"] = dict(Counter(str(row["lambda"]) for row in panel["sessions"]))
    return out


def diagnostics(rows):
    folds = [fold for row in rows for fold in row["selected_retention_diagnostics"]]
    paired = [fold["paired"] for fold in folds]
    controls = [control for fold in folds for control in fold["controls"]]
    ratios = [control["trace"]/fold["paired"]["trace"] for fold in folds
              if fold["paired"]["trace"] > 0 for control in fold["controls"]]
    ranking = {"fold_count": len(folds),
        "positive_gain_channel_range": bounds(fold["ranking_positive_channels"] for fold in folds),
        "no_positive_gain_fold_count": sum(fold["ranking_positive_channels"] == 0 for fold in folds),
        "all_raw_gains_tied_fold_count": sum(fold["ranking_all_raw_gains_tied"] for fold in folds),
        "cutoff_tie_multiplicity_range": bounds(fold["ranking_cutoff_tied_channels"] for fold in folds),
        "multiple_cutoff_tie_fold_count": sum(fold["ranking_cutoff_tied_channels"] > 1 for fold in folds),
        "raw_signed_gain_range": [min(fold["ranking_raw_gain_range"][0] for fold in folds),
                                  max(fold["ranking_raw_gain_range"][1] for fold in folds)],
        "retained_K_range": bounds(fold["retained_channels"] for fold in folds),
        "control_overlap_with_candidate_range": bounds(value for fold in folds
                                                       for value in fold["control_overlap_with_candidate"])}
    capacity = {"paired": {key+"_range": bounds(value[key] for value in paired)
                           for key in ("trace", "numerical_rank", "effective_df")},
                "control": {key+"_range": bounds(value[key] for value in controls)
                            for key in ("trace", "numerical_rank", "effective_df")},
                "control_trace_over_paired_range": bounds(ratios) if ratios else None,
                "zero_paired_trace_fold_count": sum(fold["paired"]["trace"] == 0 for fold in folds),
                "zero_control_trace_count": sum(value["trace"] == 0 for value in controls),
                "capacity_diagnostics_do_not_change_selection_or_scoring": True}
    return {"ranking": ranking, "capacity": capacity}


def seed_summaries(candidate, control):
    paired = {row["session_id"]: row for row in candidate["sessions"]}
    summaries = []
    for replicate in range(8):
        rows = []
        for row in control["sessions"]:
            matches = [seed for seed in row["selected_seed_scores"] if seed["replicate"] == replicate]
            if len(matches) != 1:
                raise ValueError("One complete score for every fixed replicate/session required")
            rows.append({**matches[0], "session_id": row["session_id"], "animal": row["animal"]})
        contrast = [{"session_id": row["session_id"], "animal": row["animal"],
                     **{metric: paired[row["session_id"]][metric]-row[metric] for metric in METRICS}}
                    for row in rows]
        item = {"replicate": replicate, "control": {}, "candidate_minus_control": {}}
        for label, values in (("control", rows), ("candidate_minus_control", contrast)):
            for metric in METRICS:
                animals = animal_values(values, metric)
                item[label][metric] = {"overall": mean(animals.values()), "animals": animals}
        summaries.append(item)
    ranges = {label: {metric: {"overall_range": bounds(row[label][metric]["overall"] for row in summaries),
               "animal_ranges": {animal: bounds(row[label][metric]["animals"][animal] for row in summaries)
                                 for animal in ANIMALS}} for metric in METRICS}
              for label in ("control", "candidate_minus_control")}
    return {"seeds": summaries, "ranges": ranges,
            "description": "Finite seed spread, not independent animals or a confidence interval"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    if report.get("status") != "complete_relevance_control_panel":
        raise ValueError("Only the completed full-cohort summary may be extracted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if set(panels) != {CANDIDATE, CONTROL} or any(len(panel["sessions"]) != 16 for panel in panels.values()):
        raise ValueError("Both complete sixteen-session panels are required")
    candidate, control = panels[CANDIDATE], panels[CONTROL]
    out = {"source_summary": str(args.summary),
           **{key: report[key] for key in ("status", "scientific_trial_count_increment",
               "total_valid_primary_trials_including_prior", "post_initial_comparisons",
               "post_initial_controls", "structural_full_retention_duplicate", "relevance_screens",
               "all_relevance_screens_pass")},
           "panels": [thin_panel(candidate), thin_panel(control)],
           "candidate_minus_mean_control": thin_panel(report["rank_label_control"]),
           "anchor_contrasts": {key: thin_panel(value) for key, value in report["anchor_contrasts"].items()},
           "retention_diagnostics": diagnostics(candidate["sessions"]),
           "retention_diagnostics_by_animal": {animal: diagnostics([row for row in candidate["sessions"]
                                                                    if row["animal"] == animal])
                                               for animal in ANIMALS},
           "control_seeds": seed_summaries(candidate, control)}
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
