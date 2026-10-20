"""Inactive exploratory reporting for a separately adopted common-support route."""
import json
from pathlib import Path

import numpy as np

from calibration_falsifiers import TaskBObservation
from observed_map_audit_context import (
    AUDIT_IDS, CONDITIONS, DEVELOPMENT_IDS, PARTITION_RUN_IDS, VARIANTS,
    validate_prediction_bundle,
)
from observed_map_common_support import (
    FULL_SUPPORT, ROUTE_ID, require_common_adoption, validate_support_record,
)
from report_observed_map_audit import _jsonable, _require, _score
from registered_endpoint_metrics import condition_weights


PREDICTION_KEYS = ("M0a", "M0b", "predictions", "condition_control_predictions",
                   "donor_control_predictions", "random_basis_predictions", "matched_warp_predictions")


def score_common_support(predictions, observations, *, policy, support_record, check):
    require_common_adoption(policy)
    full = validate_prediction_bundle(predictions, policy=policy, require_sampling_geometry=True)
    keep = validate_support_record(support_record, full["voxel_ids"])
    size = int(keep.sum())
    _require(size >= 2, "fewer_than_two_common_locations_cannot_support_centered_prediction_loss")
    observations = tuple(observations)
    _require(len(observations) == 12 and all(type(item) is TaskBObservation for item in observations)
             and tuple(item.participant_id for item in observations) == AUDIT_IDS
             and all(tuple(item.task_b_condition_ids) == CONDITIONS
                     and tuple(item.voxel_ids) == full["voxel_ids"] for item in observations),
             "all12_original_observations_on_the_full188_axis_required")
    selected_ids = tuple(np.asarray(full["voxel_ids"])[keep].tolist())
    selected = []
    for item in observations:
        check()
        values = np.asarray(item.task_b_partitions)
        _require(values.shape == (4, 18, FULL_SUPPORT) and values.dtype.kind in "fiu",
                 "complete_four_partition_eighteen_condition_arrays_required")
        selected.append(TaskBObservation(item.participant_id, CONDITIONS, selected_ids, values[..., keep]))
    restricted = dict(full, voxel_ids=selected_ids)
    for key in PREDICTION_KEYS:
        restricted[key] = full[key][..., keep]
    # Existing _score centers observations; existing _loss centers each fixed
    # prediction after this restriction. No model or control is fitted again.
    result = _score(restricted, tuple(selected), check=check, support_size=size)
    primary = result["primary"]
    primary.pop("benefit_threshold")
    primary["interval"] = "exploratory_Bonferroni_participant_t_formula_not_postselection_calibrated"
    primary["conditional_on"] = "fixed188_trained_predictors_joint_cohort_availability_selected_spatial_support"
    primary["coverage_limit"] = "no_calibrated_postselection_or_simultaneous_coverage_claim"
    primary["confirmatory_decision"] = None
    primary["working_assumptions"].append("joint_mask_selection_does_not_establish_independent_mask_conditioned_gains")
    for entry in primary["variants"].values():
        bounds = entry.pop("simultaneous_95_percent_interval")
        entry.pop("decision")
        entry["exploratory_95_percent_formula_interval"] = bounds
        entry["mean_sign"] = "positive" if entry["mean_gain"] > 0 else "negative" if entry["mean_gain"] < 0 else "zero"
        entry["formula_interval_spans_zero"] = bounds[0] <= 0 <= bounds[1]
    return result


def _figure(report, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(13, 10), layout="constrained")
    if "primary" in report:
        for index, name in enumerate(VARIANTS):
            item = report["primary"]["variants"][name]
            mean = item["mean_gain"]
            low, high = item["exploratory_95_percent_formula_interval"]
            axis.scatter(item["participant_gain"], index + np.linspace(-.19, .19, 12),
                         s=17, alpha=.5, color="#70869b", label="12 participant gains" if index == 0 else None)
            axis.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o", color="#455b73",
                          capsize=4, label="Mean and exploratory interval formula" if index == 0 else None)
        axis.axvline(0, color=".55", linewidth=.9)
        axis.set(yticks=range(16), yticklabels=VARIANTS,
                 xlabel="Observed-map gain: M0b loss − model loss")
        axis.invert_yaxis()
        axis.legend(fontsize=8)
    else:
        keep = np.asarray(report["support"]["keep"], dtype=bool)
        axis.scatter(np.arange(FULL_SUPPORT), keep.astype(int), s=12, color="#455b73")
        axis.set(xlabel="Original frozen location index", ylabel="Valid in every required audit map",
                 yticks=[0, 1], yticklabels=["No", "Yes"])
    axis.set_title(f"EP15 exploratory common support: {report['support_size']}/188 locations")
    figure.supxlabel("Previously frozen predictors; renewed Task-B access; same-account procedural separation.\n"
                     "Joint availability selection is not covered by the interval formula; original 188-location audit remains unresolved.", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(root / ("observed_map_common_support." + extension), dpi=180)
    plt.close(figure)


def _three_lines(report):
    first = (f"All 12 audit participants, 18 B-only conditions, four partitions: complete strict-availability intersection retains "
             f"{report['support_size']}/188 locations; predictors trained on the frozen 188 locations, all 16 variants and all controls retained.\n")
    if "primary" not in report:
        second = "Fewer than two common locations: centered prediction loss is uninformative; no gains or intervals computed.\n"
    else:
        entries = []
        for name in VARIANTS:
            value = report["primary"]["variants"][name]
            low, high = value["exploratory_95_percent_formula_interval"]
            entries.append(f"{name}: {value['mean_gain']:.6g} [{low:.6g}, {high:.6g}]")
        second = "Exploratory participant mean gains and Bonferroni-t formula intervals (12 units, df=11): " + "; ".join(entries) + ".\n"
    third = ("Availability-conditioned exploratory followup with prior and renewed Task-B access; joint mask/cohort selection and training uncertainty are unaccounted for. "
             "Same-account procedural separation. No original 188-location audit pass, calibrated post-selection coverage, winner, equivalence, absence, mechanism or whole-cerebellum claim.\n")
    return first + second + third


def report_common_support(predictions, observations, *, support_record, policy, output_root, figure_root, check):
    require_common_adoption(policy)
    full = validate_prediction_bundle(predictions, policy=policy, require_sampling_geometry=True)
    keep = validate_support_record(support_record, full["voxel_ids"])
    output, figures = Path(output_root), Path(figure_root)
    _require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
             "fresh_distinct_followup_report_and_figure_directories_required")
    size = int(keep.sum())
    report = {
        "status": "complete_exploratory_common_support_report" if size >= 2 else "scientific_stop_insufficient_common_support",
        "route_id": ROUTE_ID, "participant_ids": AUDIT_IDS, "development_ids": DEVELOPMENT_IDS,
        "condition_ids": CONDITIONS, "partition_run_ids": PARTITION_RUN_IDS, "variant_ids": VARIANTS,
        "support": dict(support_record), "support_size": size,
        "voxel_ids": tuple(np.asarray(full["voxel_ids"])[keep].tolist()),
        "condition_weights": condition_weights(), "partition_weight": .25, "participant_weight": 1 / 12,
        "voxel_weight": 1 / size if size else None,
        "original188_audit_completed": False, "confirmatory_decision": None,
        "uncertainty_calibrated_after_selection": False, "training_cohort_uncertainty_included": False,
        "prediction_refitting_or_geometry_rebuilding": False,
        "original_geometric_norm_rank_matching_asserted_after_restriction": False,
        "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "prior_and_renewed_Task_B_access_disclosed": True,
        "shared14_Task_B_or_private_reliability_used": False,
        "interpretation": "spatially_restrict_then_recenter_all_fixed_predictions_and_observations_equal_location_weights",
        "claim_limit": "availability_conditioned_description_no_original188_or_whole_cerebellum_or_mechanism_claim",
    }
    if size >= 2:
        report.update(score_common_support(full, observations, policy=policy, support_record=support_record, check=check))
    summary = _three_lines(report)
    check()
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(report, figures)
    check()
    with (output / "observed_map_common_support.complete.json").open("x", encoding="utf-8") as destination:
        json.dump(_jsonable(report), destination, indent=2, allow_nan=False)
        destination.write("\n")
    for root in (output, figures):
        (root / "3lines.md").write_text(summary, encoding="utf-8")
    return report
