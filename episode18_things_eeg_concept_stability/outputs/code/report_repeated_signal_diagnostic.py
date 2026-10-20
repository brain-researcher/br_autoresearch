"""Describe the complete development diagnosis of the failed repeated control.

Consumes numeric diagnostic records and the already released control report;
never reads EEG, images or closed outcomes. This is a post hoc diagnostic,
not a new positive-control test or a corrected primary score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

from feature_extractors import require_compute
from report_repeated_image_control import DEVELOPMENT_PARTICIPANTS, _read, _vector, require
import repeated_signal_diagnostic as decomposition_model
from repeated_signal_structure import BANDS


DIRECTIONS = ("odd_to_even", "even_to_odd")
DECOMPOSITION_METRICS = (
    "gain_original_R", "gain_recomputed_R", "polynomial_gain_contribution",
    "orthogonal_gain_contribution", "nuisance_polynomial_energy_fraction",
)
TEMPLATE_METRICS = ("matched_cosine", "shift37_cosine", "matched_minus_shift37_cosine")
COVARIANCE_METRICS = ("matched_cross_covariance_V2", "shift37_cross_covariance_V2",
                      "matched_minus_shift37_cross_covariance_V2", "odd_energy_V2", "even_energy_V2")
RAW_METRICS = ("mean_square_dc_V2", "centered_variance_V2", "total_mean_square_V2")


def _scalar(value):
    return float(_vector(value, ()))


def _same_gain(left, right):
    # The model checks near-zero gains on their original loss-ratio scale.
    return bool(np.isclose(1 - left, 1 - right,
        rtol=decomposition_model.MATCH_RTOL, atol=decomposition_model.MATCH_ATOL))


def _mean_optional(values):
    return None if any(value is None for value in values) else float(np.mean([_scalar(value) for value in values]))


def _describe(values):
    """No smaller-cohort inference when a cosine is undefined for a person."""
    require(len(values) == 30, "exact30_participant_values_required")
    undefined = sum(value is None for value in values)
    if undefined:
        return {"status": "unavailable_for_complete_cohort", "participant_count": 30,
                "undefined_participant_count": undefined, "mean": None,
                "standard_error": None, "descriptive_95_percent_interval": None}
    array = _vector(values, (30,))
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(30))
    width = float(t.ppf(.975, 29) * se)
    return {"status": "complete", "participant_count": 30, "undefined_participant_count": 0,
            "mean": mean, "standard_error": se,
            "descriptive_95_percent_interval": [mean - width, mean + width]}


def _participant(row, person, original_r):
    require(row.get("status") == "complete_repeated_signal_diagnostic"
            and row.get("participant_id") == person and row.get("audit_access") is False
            and row.get("concept_or_false_scores_computed") is False,
            "complete_matching_development_diagnostic_required")
    decomposition, structure = row["decomposition"], row["structure"]
    require(decomposition.get("participant_id") == person
            and decomposition.get("status") == "complete_repeated_signal_diagnostic"
            and decomposition.get("schema") == decomposition_model.SCHEMA
            and decomposition.get("diagnostic_recipe") == decomposition_model.DIAGNOSTIC_RECIPE
            and decomposition.get("data_role") == "development"
            and decomposition.get("primary_R_changed") is False
            and decomposition.get("inner_retuning_performed") is False
            and decomposition.get("audit_access") is False
            and decomposition.get("concept_or_false_models_computed") is False
            and structure.get("participant_id") == person
            and structure.get("schema") == "ep18_repeated_signal_structure_v1"
            and structure.get("status") == "complete_repeated_signal_structure"
            and structure.get("data_role") == "development_repeated_signal_recovery_diagnostic_only"
            and structure.get("audit_access") is False
            and structure.get("concept_or_false_scores_computed") is False
            and structure.get("raw_EEG_detrending") is False
            and structure.get("model_fit_performed") is False
            and structure.get("inference_performed") is False
            and structure.get("original_control_threshold_changed") is False
            and structure.get("primary_rescue") is False,
            "same_person_complete_decomposition_and_structure_required")
    directions = decomposition["directions"]
    require(len(directions) == 2
            and tuple(direction.get("direction") for direction in directions) == DIRECTIONS,
            "both_original_diagnostic_directions_required")
    for direction in directions:
        require(direction.get("released_losses_reproduced") is True
                and len(direction["sequence_rows"]) == 6
                and _scalar(direction["original_total_nuisance_denominator"]) > 0
                and _same_gain(_scalar(direction["gain_recomputed_R"]), _scalar(direction["gain_original_R"]))
                and _same_gain(_scalar(direction["polynomial_gain_contribution"])
                               + _scalar(direction["orthogonal_gain_contribution"]),
                               _scalar(direction["gain_original_R"])),
                "each_direction_requires_original_denominator_and_component_accounting")
    metrics = {}
    for key in DECOMPOSITION_METRICS:
        metrics[key] = float(np.mean([_scalar(direction[key]) for direction in directions]))
        require(np.isclose(metrics[key], _scalar(decomposition[key]), rtol=1e-8, atol=1e-10),
                "participant_decomposition_must_average_two_directions")
    require(_same_gain(metrics["gain_original_R"], original_r)
            and _same_gain(metrics["gain_recomputed_R"], original_r)
            and _same_gain(metrics["polynomial_gain_contribution"]
                           + metrics["orthogonal_gain_contribution"], original_r),
            "decomposition_must_reproduce_unchanged_original_R")
    require(0 <= metrics["nuisance_polynomial_energy_fraction"] <= 1 + 1e-10,
            "nuisance_polynomial_fraction_must_be_bounded")
    metrics["original_R"] = original_r
    templates = structure["template_metrics"]
    metrics.update({key: _scalar(templates[key]) for key in COVARIANCE_METRICS})
    if templates["cosine_defined"] is True:
        metrics.update({key: _scalar(templates[key]) for key in TEMPLATE_METRICS})
        require(abs(metrics["matched_cosine"]) <= 1 + 1e-10
                and abs(metrics["shift37_cosine"]) <= 1 + 1e-10
                and np.isclose(metrics["matched_cosine"] - metrics["shift37_cosine"],
                               metrics["matched_minus_shift37_cosine"], rtol=1e-10, atol=1e-12),
                "matched_minus_fixed_shift37_cosine_required")
    else:
        require(templates["cosine_defined"] is False
                and all(templates[key] is None for key in TEMPLATE_METRICS),
                "undefined_cosines_must_remain_explicit")
        metrics.update({key: None for key in TEMPLATE_METRICS})
    require(len(structure["sequence_metrics"]) == 12,
            "all12_sequence_spectral_and_DC_summaries_required")
    sequences = structure["sequence_metrics"]
    for key in RAW_METRICS:
        metrics["equal_sequence_mean_" + key] = float(np.mean([_scalar(item[key]) for item in sequences]))
    dc_fractions = [(_scalar(item["mean_square_dc_V2"]) / _scalar(item["total_mean_square_V2"]))
                    if _scalar(item["total_mean_square_V2"]) > 0 else None for item in sequences]
    metrics["equal_sequence_mean_DC_fraction"] = _mean_optional(dc_fractions)
    for band in BANDS:
        metrics["equal_sequence_mean_power_V2_" + band] = float(np.mean([
            _scalar(item["spectral_power_V2"][band]) for item in sequences]))
        metrics["equal_sequence_mean_centered_power_fraction_" + band] = _mean_optional([
            item["spectral_fraction_centered"][band] for item in sequences])
    return metrics


def _figure(values, estimates, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    panels = (
        ("original_R", "Original repeated-image gain R", "Reduction in original normalized loss"),
        ("nuisance_polynomial_energy_fraction", "Nuisance error in polynomial component", "Fraction of original nuisance error"),
        ("matched_minus_shift37_cosine", "Template similarity: matched − cyclic 37", "Difference in odd/even template cosine"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    for ax, (key, title, label) in zip(axes, panels):
        estimate = estimates[key]
        ax.set(title=title, xlabel=label, yticks=[], ylim=(-.32, .75))
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        if estimate["status"] != "complete":
            ax.text(.5, .5, f"Undefined in {estimate['undefined_participant_count']}/30 people\n"
                    "Complete-cohort estimate unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            continue
        mean = estimate["mean"]
        low, high = estimate["descriptive_95_percent_interval"]
        if key != "nuisance_polynomial_energy_fraction":
            ax.axvline(0, color=".7", linewidth=.8)
        ax.scatter(values[key], np.linspace(-.2, .2, 30), s=18, color="#54779c", alpha=.8,
                   label="30 participant points")
        ax.errorbar(mean, .43, xerr=[[mean - low], [high - mean]], fmt="o", color="black",
                    capsize=4, label="Mean and descriptive 95% CI")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), fontsize=8, frameon=False)
    fig.suptitle("EP18 exploratory diagnosis of the failed repeated-image control", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(root / ("repeated_signal_diagnostic." + extension), dpi=180)
    plt.close(fig)


def _format(estimate):
    if estimate["status"] != "complete":
        return f"unavailable ({estimate['undefined_participant_count']}/30 undefined; nobody excluded)"
    low, high = estimate["descriptive_95_percent_interval"]
    return f"{estimate['mean']:.6g} (descriptive 95% CI {low:.6g} to {high:.6g})"


def report(*, development_manifest, diagnostic_root, repeated_report, output_root, figure_root):
    require_compute()
    manifest = _read(development_manifest)
    people = tuple(manifest["participants"])
    require(manifest.get("role") == "development" and people == DEVELOPMENT_PARTICIPANTS
            and manifest.get("event_count") == 739440
            and manifest.get("frozen_event_join") == "development_event_image_join.tsv.gz",
            "original_complete_development_manifest_required")
    paths = [Path(diagnostic_root) / person / "complete.json" for person in people]
    require(all(path.is_file() for path in paths), "all30_completed_signal_diagnostics_required")
    released = _read(repeated_report)
    require(released.get("status") == "complete_30_person_repeated_image_control_report"
            and tuple(released["participants"]) == people
            and set(released["participant_G_T_R"]) == set(people)
            and released.get("participant_count") == 30
            and released.get("directions_per_participant") == 2
            and released.get("audit_access") is False
            and released.get("concept_or_false_scores_computed") is False
            and released.get("primary_result_rescued") is False
            and released.get("all_three_positive_controls_pass") is False
            and released.get("endpoint_control_pass", {}).get("R") is False,
            "same30_released_failed_control_required")
    original = _vector([released["participant_G_T_R"][person] for person in people], (30, 3))[:, 2]
    rows = [_read(path) for path in paths]
    require(all(row["configuration"] == rows[0]["configuration"] and row["recipe"] == rows[0]["recipe"]
                for row in rows), "same_diagnostic_and_original_recipe_across_all30_required")
    per_person = {person: _participant(row, person, float(original[index]))
                  for index, (person, row) in enumerate(zip(people, rows))}
    keys = tuple(per_person[people[0]])
    values = {key: [per_person[person][key] for person in people] for key in keys}
    estimates = {key: _describe(value) for key, value in values.items()}
    record = {
        "status": "complete_30_person_repeated_signal_diagnostic_report",
        "participants": people, "participant_count": 30, "inference_unit": "participant",
        "degrees_of_freedom": 29, "interval_role": "descriptive_two_sided_95_percent_participant_t",
        "simultaneous_or_confirmatory_inference": False, "new_pass_rule": None,
        "estimates": estimates, "participant_metrics": per_person,
        "participant_diagnostics": dict(zip(people, rows)),
        "decomposition_normalization": "original_total_nuisance_error_in_each_direction_then_equal_direction_average",
        "template_comparison": "all_200_image_by_lag_by_sensor_values_centered_across_images_per_sequence; odd_even_matched_cosine_minus_fixed_even_image_shift37",
        "raw_summaries": "all12_sequence_DC_variance_and_spectral_band_summaries_preserved_in_participant_diagnostics.structure.sequence_metrics; cohort_summaries_use_equal_sequence_means_within_each_person; mean_fractions_are_not_ratios_of_pooled_energies",
        "scientific_role": "exploratory_post_hoc_diagnosis_of_failed_control",
        "analysis_chosen_after": str(repeated_report), "diagnostic_root": str(diagnostic_root),
        "preprocessing_or_primary_score_changed": False, "primary_result_rescued": False,
        "concept_signal_absence_inferred": False, "audit_access": False,
        "concept_or_false_scores_computed": False,
        "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "interpretation": "Polynomial error concentration and image-aligned template similarity diagnose measurement/readout sensitivity. Similarity may also reflect temporal order or neighboring-image responses; fixed shift37 does not rule these out. Neither diagnostic identifies a biological mechanism, changes the failed registered control nor establishes signal absence.",
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    output, figures = Path(output_root), Path(figure_root)
    require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
            "fresh_distinct_report_and_figure_directories_required")
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(values, estimates, figures)
    summary = (
        f"Exploratory diagnosis of all 30 original development people: original R={_format(estimates['original_R'])}; "
        f"nuisance polynomial fraction={_format(estimates['nuisance_polynomial_energy_fraction'])}; "
        f"matched-minus-cyclic37 template cosine={_format(estimates['matched_minus_shift37_cosine'])}.\n"
        f"Directions average within people: polynomial contribution {estimates['polynomial_gain_contribution']['mean']:.6g} "
        f"plus orthogonal contribution {estimates['orthogonal_gain_contribution']['mean']:.6g} reconstruct original R using the original nuisance denominator; "
        "intervals are descriptive t(df29), with raw DC/band and lag summaries in the complete numeric report.\n"
        "This post hoc diagnosis changes no preprocessing, primary score or pass rule; it cannot rescue the failed control or establish concept-signal absence; concept/false scores and audit remain closed under same-account procedural separation.\n"
    )
    (output / "summary.md").write_text(summary)
    with (output / "repeated_signal_diagnostic.complete.json").open("x") as destination:
        json.dump(record, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-manifest", "diagnostic-root", "repeated-report", "output-root", "figure-root"):
        parser.add_argument("--" + name, required=True)
    report(**vars(parser.parse_args()))
