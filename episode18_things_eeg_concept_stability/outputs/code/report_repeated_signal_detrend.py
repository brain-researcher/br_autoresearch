"""Complete-cohort paired report of a fixed polynomial-drift diagnostic.

Only released numeric records enter this consumer. Whole-sequence EEG
residualization is descriptive preprocessing, not out-of-sample prediction;
this report neither changes original R nor introduces a replacement pass rule.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from feature_extractors import require_compute
from report_repeated_image_control import DEVELOPMENT_PARTICIPANTS, _read, require
from report_repeated_signal_diagnostic import _describe, _format, _scalar
import repeated_signal_detrend as model


TEMPLATE_METRICS = ("matched_cosine", "shift37_cosine", "matched_minus_shift37_cosine")
DATA_ROLE = "development_repeated_signal_recovery_diagnostic_only"


def _templates(structure, person, *, adjusted, negative=False):
    require(structure.get("schema") == "ep18_repeated_signal_structure_v1"
            and structure.get("status") == "complete_repeated_signal_structure"
            and structure.get("participant_id") == person and structure.get("data_role") == DATA_ROLE
            and structure.get("physical_sequence_count") == 12
            and structure.get("image_identity_count") == 200
            and structure.get("recorded_channel_count") == 62
            and structure.get("sensor_contrast_count") == 61
            and structure.get("raw_EEG_detrending") is adjusted
            and structure.get("model_fit_performed") is False
            and structure.get("inference_performed") is False
            and structure.get("original_control_threshold_changed") is False
            and structure.get("primary_rescue") is False
            and structure.get("audit_access") is False
            and structure.get("concept_or_false_scores_computed") is False,
            "same_person_complete_fixed_support_structure_required")
    sequences = structure["sequence_metrics"]
    require(len(sequences) == 12
            and [item["ordinal"] for item in sequences] == list(range(12))
            and len({item["sequence_id"] for item in sequences}) == 12,
            "all_twelve_chronological_sequences_required")
    definition = structure["template_definition"]
    require(definition.get("lag_bin_edges_ms") == (list(range(-800, 1, 20)) if negative else list(range(0, 801, 20)))
            and definition.get("final_bin_includes_0ms" if negative else "final_bin_includes_800ms") is True
            and definition.get("interpolation") is False
            and definition.get("event_snapping") is False
            and definition.get("prestimulus_correction") is False,
            "original_acquisition_grid_template_bins_required")
    values = structure["template_metrics"]
    if values["cosine_defined"] is False:
        require(all(values[key] is None for key in TEMPLATE_METRICS),
                "undefined_cosines_must_remain_explicit")
        return dict.fromkeys(TEMPLATE_METRICS)
    require(values["cosine_defined"] is True, "explicit_cosine_validity_required")
    metrics = {key: _scalar(values[key]) for key in TEMPLATE_METRICS}
    require(abs(metrics["matched_cosine"]) <= 1 + 1e-10
            and abs(metrics["shift37_cosine"]) <= 1 + 1e-10
            and np.isclose(metrics["matched_cosine"] - metrics["shift37_cosine"],
                           metrics["matched_minus_shift37_cosine"], rtol=1e-10, atol=1e-12),
            "matched_minus_fixed_shift37_cosine_required")
    return metrics


def _participant(row, original, person):
    require(row.get("status") == "complete_repeated_signal_detrend"
            and row.get("participant_id") == person
            and row.get("audit_access") is False
            and row.get("concept_or_false_scores_computed") is False,
            "complete_matching_development_detrend_required")
    component = row
    require(component.get("schema") == model.SCHEMA
            and component.get("status") == "complete_repeated_signal_detrend"
            and component.get("participant_id") == person
            and component.get("data_role") == DATA_ROLE
            and component.get("projection_recipe") == model.PROJECTION_RECIPE
            and component.get("original_R_changed") is False
            and component.get("original_control_threshold_changed") is False
            and component.get("primary_rescue") is False
            and component.get("out_of_sample_prediction") is False
            and component.get("audit_access") is False
            and component.get("concept_or_false_scores_computed") is False,
            "fixed_degree_three_diagnostic_without_primary_change_required")
    require(original.get("participant_id") == person
            and original.get("status") == "complete_repeated_signal_diagnostic"
            and original.get("audit_access") is False
            and original.get("concept_or_false_scores_computed") is False,
            "same_person_released_original_diagnostic_required")
    raw_structure, adjusted_structure = original["structure"], component["structure"]
    negative_structure = component["negative_lag_structure"]
    raw = _templates(raw_structure, person, adjusted=False)
    adjusted = _templates(adjusted_structure, person, adjusted=True)
    negative = _templates(negative_structure, person, adjusted=True, negative=True)
    sequence_keys = lambda structure: [(item["sequence_id"], item["ordinal"], item["retained_samples"])
                                      for item in structure["sequence_metrics"]]
    require(sequence_keys(raw_structure) == sequence_keys(adjusted_structure)
            and sequence_keys(raw_structure) == sequence_keys(negative_structure)
            and all(adjusted_structure["template_definition"].get(key) == value
                    for key, value in raw_structure["template_definition"].items())
            and adjusted_structure["template_definition"].get("lag_window_relative_to_original_event_ms") == [0, 800]
            and adjusted_structure["template_definition"].get("diagnostic_onset_shift_ms") == 0
            and raw_structure["image_order"] == adjusted_structure["image_order"]
            and raw_structure["image_order"] == negative_structure["image_order"]
            and negative_structure["template_definition"].get("diagnostic_onset_shift_ms") == -800
            and negative_structure["template_definition"].get("lag_window_relative_to_original_event_ms") == [-800, 0]
            and negative_structure["template_definition"].get("final_bin_includes_800ms") is False
            and len(component["projection_accounting"]["sequence_rows"]) == 12,
            "paired_diagnostic_must_preserve_sequences_bins_and_image_order")
    values = {}
    for key in TEMPLATE_METRICS:
        values["raw_" + key] = raw[key]
        values["adjusted_" + key] = adjusted[key]
        values["paired_change_" + key] = (None if raw[key] is None or adjusted[key] is None
                                          else adjusted[key] - raw[key])
        values["negative_lag_" + key] = negative[key]
        values["positive_minus_negative_" + key] = (None if adjusted[key] is None or negative[key] is None
                                                    else adjusted[key] - negative[key])
    return values


def _figure(values, estimates, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    panels = (("raw_matched_minus_shift37_cosine", "Original 0–800 ms image alignment"),
              ("adjusted_matched_minus_shift37_cosine", "Adjusted 0–800 ms alignment"),
              ("negative_lag_matched_minus_shift37_cosine", "Adjusted −800–0 ms alignment"),
              ("positive_minus_negative_matched_minus_shift37_cosine", "Paired positive − negative lag alignment"))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for axis, (key, title) in zip(axes.flat, panels):
        estimate = estimates[key]
        axis.set(title=title, xlabel="Matched − cyclic37 cosine" if not key.startswith("positive_minus")
                 else "Adjusted positive-lag alignment − negative-lag alignment", yticks=[], ylim=(-.32, .75))
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        axis.axvline(0., color=".7", linewidth=.8)
        if estimate["status"] != "complete":
            axis.text(.5, .5, f"Undefined in {estimate['undefined_participant_count']}/30 people\n"
                      "Complete-cohort estimate unavailable", ha="center", va="center",
                      transform=axis.transAxes, fontsize=9)
            axis.set_xticks([])
            continue
        mean = estimate["mean"]
        lower, upper = estimate["descriptive_95_percent_interval"]
        axis.scatter(values[key], np.linspace(-.2, .2, 30), s=18, color="#54779c", alpha=.8,
                     label="30 participant values")
        axis.errorbar(mean, .43, xerr=[[mean - lower], [upper - mean]], fmt="o", color="black",
                      capsize=4, label="Mean and descriptive 95% CI")
    axes.flat[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), fontsize=8, frameon=False)
    figure.suptitle("EP18: paired diagnostic of slow-drift masking", fontsize=12)
    figure.supxlabel("Degree 0–3 fit uses each whole sequence's EEG; descriptive participant-t intervals, df=29.\n"
                     "This is not out-of-sample prediction and does not change the failed original control.", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(root / ("repeated_signal_detrend." + extension), dpi=180)
    plt.close(figure)


def report(*, development_manifest, diagnostic_root, original_report, output_root, figure_root):
    require_compute()
    manifest = _read(development_manifest)
    people = tuple(manifest["participants"])
    require(manifest.get("role") == "development" and people == DEVELOPMENT_PARTICIPANTS
            and manifest.get("event_count") == 739440
            and manifest.get("frozen_event_join") == "development_event_image_join.tsv.gz",
            "original_complete_development_manifest_required")
    paths = [Path(diagnostic_root) / person / "complete.json" for person in people]
    require(all(path.is_file() for path in paths), "all30_completed_detrend_diagnostics_required")
    released = _read(original_report)
    require(released.get("status") == "complete_30_person_repeated_signal_diagnostic_report"
            and tuple(released["participants"]) == people
            and released.get("participant_count") == 30
            and set(released["participant_diagnostics"]) == set(people)
            and set(released["participant_metrics"]) == set(people)
            and released.get("audit_access") is False
            and released.get("concept_or_false_scores_computed") is False
            and released.get("primary_result_rescued") is False,
            "same30_released_original_signal_diagnostic_required")
    rows = [_read(path) for path in paths]
    require(all(row["configuration"] == rows[0]["configuration"] for row in rows),
            "same_fixed_detrend_configuration_across_all30_required")
    per_person = {person: _participant(row, released["participant_diagnostics"][person], person)
                  for person, row in zip(people, rows)}
    # Bind the released raw summaries to their source templates before pairing.
    for person in people:
        for key in TEMPLATE_METRICS:
            actual, expected = per_person[person]["raw_" + key], released["participant_metrics"][person][key]
            require((actual is None and expected is None) or (actual is not None and expected is not None
                    and np.isclose(actual, _scalar(expected), rtol=1e-12, atol=1e-14)),
                    "raw_metrics_must_match_the_released_participant_report")
    values = {key: [per_person[person][key] for person in people] for key in per_person[people[0]]}
    estimates = {key: _describe(value) for key, value in values.items()}
    record = {
        "status": "complete_30_person_repeated_signal_detrend_report",
        "participants": people, "participant_count": 30, "inference_unit": "participant",
        "degrees_of_freedom": 29, "interval_role": "descriptive_two_sided_95_percent_participant_t",
        "paired_change_definition": "adjusted_minus_original_within_the_same_participant",
        "lag_comparison_definition": "adjusted_0_to_800ms_minus_adjusted_minus800_to_0ms_within_the_same_participant",
        "negative_lag_role": "fixed_prospective_control_for_this_diagnostic_not_original_registered_control_or_pass_gate",
        "simultaneous_or_confirmatory_inference": False, "new_pass_rule": None,
        "estimates": estimates, "participant_metrics": per_person,
        "participant_diagnostics": dict(zip(people, rows)),
        "original_R_estimate_unchanged": released["estimates"]["original_R"],
        "projection_recipe": model.PROJECTION_RECIPE,
        "scientific_role": "exploratory_post_hoc_diagnostic_of_slow_drift_masking",
        "analysis_chosen_after": str(original_report), "diagnostic_root": str(diagnostic_root),
        "original_R_changed": False, "original_control_threshold_changed": False,
        "primary_result_rescued": False, "out_of_sample_prediction": False,
        "whole_sequence_EEG_used_for_projection": True,
        "concept_signal_absence_inferred": False, "audit_access": False,
        "concept_or_false_scores_computed": False,
        "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "interpretation": "The paired change examines whether fixed slow polynomial components obscure image-aligned structure. The fixed negative-lag comparison probes temporal-order/slow-structure alternatives; it is not the original registered negative control. Residualization uses each complete sequence's EEG, so this is not out-of-sample prediction. Overlapping neighboring-image responses and noncausal whole-sequence projection still prevent a uniquely causal interpretation. No new pass rule, primary rescue, mechanism, or concept-signal absence follows.",
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    output, figures = Path(output_root), Path(figure_root)
    require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
            "fresh_distinct_report_and_figure_directories_required")
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(values, estimates, figures)
    summary = (
        "All 30 original development people, all 12 repeated sequences, 200 image identities and 62 channels: "
        f"raw matched-minus-cyclic37 cosine {_format(estimates['raw_matched_minus_shift37_cosine'])}; "
        f"after fixed degree-0–3 removal {_format(estimates['adjusted_matched_minus_shift37_cosine'])}.\n"
        f"Paired raw-to-adjusted alignment change {_format(estimates['paired_change_matched_minus_shift37_cosine'])}; "
        f"adjusted matched cosine {_format(estimates['adjusted_matched_cosine'])}; "
        f"adjusted negative-lag alignment {_format(estimates['negative_lag_matched_minus_shift37_cosine'])}; "
        f"paired positive-minus-negative alignment {_format(estimates['positive_minus_negative_matched_minus_shift37_cosine'])}; "
        "all intervals are descriptive participant-t(df29), with no available-case exclusions or new pass rule.\n"
        "This post hoc slow-drift diagnostic uses each whole sequence's EEG and is not out-of-sample prediction; "
        "the fixed negative-lag comparison is diagnostic, not the registered control, and cannot uniquely separate "
        "temporal-order/neighbor effects or noncausal projection. Original R and its failed control remain unchanged: "
        "no primary rescue, mechanism, or concept-signal absence is established; concept/false scores and audit remain "
        "closed under same-account procedural separation.\n"
    )
    require(len(summary.splitlines()) == 3, "exact_three_line_summary_required")
    (output / "summary.md").write_text(summary)
    with (output / "repeated_signal_detrend.complete.json").open("x") as destination:
        json.dump(record, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-manifest", "diagnostic-root", "original-report", "output-root", "figure-root"):
        parser.add_argument("--" + name, required=True)
    report(**vars(parser.parse_args()))
