"""Report all thirty development repeated-image controls with frozen G/T.

Only completed numeric records enter this consumer. It never reads EEG,
images, concept/false scores or audit data. Both directions are averaged
within each participant; sequence/direction counts never increase n.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

from feature_extractors import require_compute
from participant_endpoint_inference import POSITIVE, _bounds
import repeated_image_control as repeated


THRESHOLDS = np.array([.002, .001, 0.])
KEYS = ("G", "T", "R")
# The original released development assignment, not a caller-selected cohort.
DEVELOPMENT_PARTICIPANTS = tuple("sub-" + suffix for suffix in (
    "07", "08", "09", "10", "11", "12", "13", "14", "20", "21", "22", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "34", "35", "38", "43", "44", "45", "46", "47", "48", "49"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    path = Path(path)
    require(path.stat().st_size <= 16 * 1024**2, "bounded_numeric_record_required")
    with path.open() as source:
        return json.load(source)


def _vector(value, shape):
    raw = np.asarray(value)
    require(raw.shape == shape and raw.dtype.kind in "iuf" and np.isfinite(raw).all(),
            "complete_finite_numeric_vector_required")
    return raw.astype(float)


def _participant(row, person):
    require(row.get("schema") == repeated.SCHEMA
            and row.get("status") == "complete_repeated_image_control"
            and row.get("participant_id") == person and row.get("data_role") == "development"
            and row.get("image_identity_count") == 200 and row.get("physical_sequence_count") == 12
            and row.get("inference_unit") == "participant"
            and row.get("cohort_inference_performed") is False
            and row.get("audit_access") is False and row.get("concept_or_false_models_computed") is False,
            "complete_matching_development_control_required")
    binding = row["binding"]
    require(binding.get("schema") == repeated.SCHEMA and binding.get("participant_id") == person
            and binding.get("data_role") == "development"
            and binding.get("identity_recipe") == repeated.IDENTITY_RECIPE,
            "original_repeated_control_recipe_required")
    repeated.validate_recipe(binding["recipe"])
    require(len(binding["image_ids"]) == len(set(binding["image_ids"])) == 200
            and len(binding["channel_order"]) == len(set(binding["channel_order"])) == 62,
            "complete_identity_and_channel_support_required")
    support = binding["sequence_support"]
    require(len(support) == 12 and [item["ordinal"] for item in support] == list(range(12))
            and len({item["sequence_id"] for item in support}) == 12
            and all(type(item["samples"]) is int and type(item["retained_samples"]) is int
                    and 0 < item["retained_samples"] <= item["samples"] for item in support),
            "twelve_complete_chronological_sequences_required")
    directions = row["directions"]
    require(len(directions) == 2 and tuple(item.get("direction") for item in directions) == repeated.DIRECTIONS,
            "both_original_directions_required")
    gains = []
    for parity, direction in enumerate(directions):
        train = [item["sequence_id"] for item in support if item["ordinal"] % 2 == parity]
        heldout = [item["sequence_id"] for item in support if item["ordinal"] % 2 != parity]
        require(direction.get("status") == "complete_direction" and direction.get("binding") == binding
                and direction.get("training_sequence_ids") == train
                and direction.get("heldout_sequence_ids") == heldout
                and direction["training_whitening"].get("sequence_ids") == train
                and direction.get("inner_validations_completed") == 6,
                "original_disjoint_outer_and_training_only_whitening_required")
        losses = _vector(direction["identity_sequence_losses"], (6,))
        nuisance = _vector(direction["nuisance_sequence_losses"], (6,))
        means = _vector(direction["inner_mean_normalized_identity_losses"], (13,))
        require(np.all(losses >= 0) and np.all(nuisance >= 0) and np.mean(nuisance) > 0
                and np.all(means >= 0), "positive_nuisance_mean_and_nonnegative_losses_required")
        selected = min(range(13), key=lambda index: (means[index], -repeated.RIDGE_GRID[index], index))
        require(direction.get("selected_penalty_index") == selected
                and direction.get("selected_ridge_penalty") == repeated.RIDGE_GRID[selected],
                "original_training_only_penalty_selection_required")
        gain = float(1 - np.mean(losses) / np.mean(nuisance))
        require(np.isfinite(gain) and _vector(direction["gain"], ()) == gain,
                "direction_gain_must_match_saved_sequence_losses")
        gains.append(gain)
    average = float(np.mean(gains))
    require(_vector(row["gain"], ()) == average, "participant_gain_must_average_both_directions")
    return average


def _evidence(data):
    data = _vector(data, (30, 3))
    family = _bounds(data, POSITIVE, two_sided=False)
    means, se = np.asarray(family.mean), np.asarray(family.standard_error)
    widths = t.ppf(.975, 29) * se
    omitted = np.array([np.delete(data, index, axis=0).mean(axis=0) for index in range(30)])
    lower_pass = np.asarray(family.lower) > THRESHOLDS
    influence_pass = np.all(omitted > 0, axis=0)
    return {"simultaneous_one_sided_positive_family": asdict(family),
        "thresholds": dict(zip(KEYS, THRESHOLDS.tolist())),
        "lower_bound_exceeds_threshold": dict(zip(KEYS, lower_pass.tolist())),
        "leave_one_participant_means": omitted.tolist(),
        "every_leave_one_participant_mean_positive": dict(zip(KEYS, influence_pass.tolist())),
        "endpoint_control_pass": dict(zip(KEYS, (lower_pass & influence_pass).tolist())),
        "all_three_positive_controls_pass": bool(np.all(lower_pass & influence_pass)),
        "descriptive_95_percent_intervals": dict(zip(KEYS, np.column_stack((means-widths, means+widths)).tolist()))}


def _figure(data, record, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    labels = ("Image gain G", "Task-nuisance gain T", "Repeated-image gain R")
    family = record["simultaneous_one_sided_positive_family"]
    for index, (key, ax) in enumerate(zip(KEYS, axes)):
        mean = family["mean"][index]
        low, high = record["descriptive_95_percent_intervals"][key]
        ax.axvline(0, color=".7", linewidth=.8)
        ax.axvline(THRESHOLDS[index], color="#b14e32", linestyle="--", linewidth=1,
                   label="Registered control threshold")
        ax.scatter(data[:, index], np.linspace(-.2, .2, 30), s=18, color="#54779c", alpha=.8,
                   label="30 participant points")
        ax.errorbar(mean, .43, xerr=[[mean-low], [high-mean]], fmt="o", color="black", capsize=4,
                    label="Mean and descriptive 95% CI")
        ax.scatter(family["lower"][index], .62, marker=">", color="#8b4a80", s=35,
                   label="Simultaneous one-sided lower bound")
        ax.set(title=labels[index], xlabel="Reduction in normalized held-out loss", yticks=[], ylim=(-.32, .82))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.spines[["left", "right", "top"]].set_visible(False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.04), fontsize=7, frameon=False)
    fig.suptitle("EP18 development controls: repeated images cannot rescue the main comparison", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(root / ("repeated_image_G_T_R." + extension), dpi=180)
    plt.close(fig)


def report(*, development_manifest, repeated_root, core_report, output_root, figure_root):
    require_compute()
    manifest = _read(development_manifest)
    people = tuple(manifest["participants"])
    require(manifest.get("role") == "development" and people == DEVELOPMENT_PARTICIPANTS
            and manifest.get("event_count") == 739440
            and manifest.get("frozen_event_join") == "development_event_image_join.tsv.gz",
            "original_complete_development_manifest_required")
    paths = [Path(repeated_root) / person / "complete.json" for person in people]
    require(all(path.is_file() for path in paths), "all30_completed_repeated_records_required")
    core = _read(core_report)
    require(core.get("status") == "complete_30_person_core_calibration"
            and tuple(core["participants"]) == people and set(core["participant_G_T"]) == set(people)
            and core.get("participant_count") == 30 and core.get("outer_splits") == 180
            and core.get("inner_splits") == 900 and core.get("audit_access") is False
            and core.get("concept_or_false_scores_computed") is False
            and core.get("scientific_role") == "registered_development_core_calibration",
            "same30_released_core_calibration_required")
    gt = _vector([core["participant_G_T"][person] for person in people], (30, 2))
    require(np.allclose(gt.mean(axis=0), _vector([core["G"], core["T"]], (2,)), rtol=0, atol=1e-14)
            and core.get("delta_Q") == .002 and core.get("delta_T") == .001
            and core.get("reference_underidentified") is True and core["G"] <= 0,
            "unchanged_released_core_means_and_margins_required")
    rows = [_read(path) for path in paths]
    repeated_gains = [_participant(row, person) for row, person in zip(rows, people)]
    first = rows[0]["binding"]
    require(all(all(row["binding"][key] == first[key] for key in ("recipe", "identity_recipe", "image_ids", "channel_order"))
                for row in rows), "same_control_recipe_images_and_channels_across_people_required")
    data = np.column_stack((gt, repeated_gains))
    record = _evidence(data)
    require(record["all_three_positive_controls_pass"] is False, "failed_core_reference_cannot_be_rescued")
    record.update({"status": "complete_30_person_repeated_image_control_report", "participants": people,
        "participant_count": 30, "directions_per_participant": 2, "inference_unit": "participant",
        "participant_G_T_R": dict(zip(people, data.tolist())), "core_report": str(core_report),
        "repeated_root": str(repeated_root), "leave_one_participant_omission_order": people,
        "scientific_role": "prespecified_development_signal_recovery_control",
        "core_reference_remains_underidentified": True, "primary_result_rescued": False,
        "concept_signal_absence_inferred": False, "audit_access": False, "concept_or_false_scores_computed": False,
        "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "interpretation": "R can establish recovery of repeated image-specific responses under this pipeline, not concept-specific generalization; failed G/T controls and the original concept comparison remain unresolved.",
        "job_id": os.environ["SLURM_JOB_ID"]})
    output, figures = Path(output_root), Path(figure_root)
    require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
            "fresh_distinct_report_and_figure_directories_required")
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(data, record, figures)
    family = record["simultaneous_one_sided_positive_family"]
    descriptions = [f"{key}={family['mean'][index]:.6g} (descriptive 95% CI {record['descriptive_95_percent_intervals'][key][0]:.6g} to {record['descriptive_95_percent_intervals'][key][1]:.6g})"
                    for index, key in enumerate(KEYS)]
    summary = ("Across the original 30 development participants, " + "; ".join(descriptions) + ".\n"
        "Both repeated-image directions are averaged within each person; simultaneous one-sided t bounds use alpha=.05/3, thresholds G>.002/T>.001/R>0 and every leave-one-person mean>0; R control "
        + ("passes" if record["endpoint_control_pass"]["R"] else "does not pass") + ".\n"
        "G/T keep the control family failed; repeated-image recovery cannot rescue the main concept comparison or establish concept-signal absence; audit/concept/false scores remain closed under same-account procedural separation.\n")
    (output / "summary.md").write_text(summary)
    with (output / "repeated_image_control.complete.json").open("x") as destination:
        json.dump(record, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development-manifest", "repeated-root", "core-report", "output-root", "figure-root"):
        parser.add_argument("--" + name, required=True)
    report(**vars(parser.parse_args()))
