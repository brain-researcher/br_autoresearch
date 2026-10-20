"""Inactive, adoption-gated reporting for the proposed complete EP15 audit.

Consumes already frozen predictions and the one permitted in-memory Task-B
observation bundle. This module reads no input file, opens no audit container,
fits no model, and cannot adopt its own proposal. The runner owns prediction
locking and the recorded single Task-B opening. All controls are descriptive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from calibration_falsifiers import TaskBObservation
from development_lopo import (
    MATCHED_WARP_DRAWS, M2_VARIANTS, M3_VARIANTS, RANDOM_BASIS_DRAWS, _array,
)
from observed_beta_summary import _difference, _draw_summary, _loss
from observed_map_audit_context import (
    AUDIT_IDS, CONDITIONS, DEVELOPMENT_IDS, PARTITION_RUN_IDS, ROUTE_ID,
    VARIANTS, require_adoption, validate_prediction_bundle,
)
from registered_endpoint_metrics import condition_weights


FAMILY_SIZE = 16
PARTICIPANTS = 12
SUPPORT_SIZE = 188
ALPHA = .05
DF = 11


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _primary_family(gains):
    """The sole inferential family: one paired gain per person and variant."""
    gains = _array(gains, (PARTICIPANTS, FAMILY_SIZE), "audit participant gains")
    critical = float(t.ppf(1 - ALPHA / (2 * FAMILY_SIZE), DF))
    mean = gains.mean(axis=0)
    standard_deviation = gains.std(axis=0, ddof=1)
    standard_error = standard_deviation / np.sqrt(PARTICIPANTS)
    lower, upper = mean - critical * standard_error, mean + critical * standard_error
    _require(np.isfinite(np.concatenate((mean, standard_error, lower, upper))).all(),
             "all sixteen finite participant-t estimates are required")
    entries = {}
    for index, name in enumerate(VARIANTS):
        decision = ("positive_heldout_observed_map_prediction_benefit" if lower[index] > 0
                    else "negative_mean_gain" if upper[index] < 0 else "inconclusive")
        entries[name] = {
            "participant_gain": gains[:, index], "mean_gain": float(mean[index]),
            "participant_standard_deviation": float(standard_deviation[index]),
            "standard_error": float(standard_error[index]),
            "simultaneous_95_percent_interval": [float(lower[index]), float(upper[index])],
            "decision": decision,
        }
    return {
        "definition": "D[i,m] = observed_loss[i,M0b] - observed_loss[i,m]",
        "family": VARIANTS, "family_size": FAMILY_SIZE, "participant_count": PARTICIPANTS,
        "degrees_of_freedom": DF, "family_alpha": ALPHA,
        "interval": "two_sided_Bonferroni_participant_t",
        "critical_t": critical, "benefit_threshold": 0., "variants": entries,
        "independent_unit": "participant_not_condition_partition_voxel_or_control_draw",
        "conditional_on": "fixed_development_fit_selected_cohort_and_188_location_support",
        "working_assumptions": [
            "independent_exchangeable_participant_gains_under_the_participant_t_model",
            "the_jointly_balanced_24_person_split_does_not_guarantee_that_model",
        ],
        "coverage_limit": "model_based_small_sample_intervals_not_exact_design_based_guarantees",
        "training_cohort_uncertainty_included": False,
        "general_population_or_cross_site_transport_established": False,
        "winner_or_class_ranking": None, "equivalence_or_absence_inference": False,
    }


def _score(bundle, observations, *, check):
    """Pure scoring after adoption, complete prediction lock and axis checks."""
    observations = tuple(observations)
    # Reject every identity/axis omission before converting any observed value.
    _require(len(observations) == PARTICIPANTS
             and all(type(item) is TaskBObservation for item in observations)
             and tuple(item.participant_id for item in observations) == AUDIT_IDS,
             "all original twelve audit observations are required in canonical order")
    _require(all(tuple(item.task_b_condition_ids) == CONDITIONS
                 and tuple(item.voxel_ids) == bundle["voxel_ids"] for item in observations),
             "audit observations must retain all eighteen conditions and the frozen 188-location axis")
    check()
    observed = np.stack([_array(item.task_b_partitions, (4, 18, SUPPORT_SIZE), "audit observed Task B")
                         for item in observations])
    centered = observed - observed.mean(axis=-1, keepdims=True)
    losses = {baseline: _loss(centered, [bundle[baseline]] * PARTICIPANTS)
              for baseline in ("M0a", "M0b")}
    for index, name in enumerate(VARIANTS):
        check()
        losses[name] = _loss(centered, bundle["predictions"][:, index])
    gains = np.column_stack([losses["M0b"]["participant_mean_observed_mse"]
                            - losses[name]["participant_mean_observed_mse"] for name in VARIANTS])
    calibration = {}
    for label, field in (("condition_permutation", "condition_control_predictions"),
                         ("identity_derangement", "donor_control_predictions")):
        variants = {}
        for index, name in enumerate(VARIANTS):
            check()
            control = _loss(centered, bundle[field][:, index])
            variants[name] = {"control_loss": control,
                              "control_minus_true": _difference(control, losses[name])}
        calibration[label] = {"variants": variants,
            "true_loss_reference": "ordinary.losses_by_model",
            "source_training_ids": DEVELOPMENT_IDS,
            "true_packet_ids": AUDIT_IDS,
            "control_packet_ids": bundle["donor_ids"] if label == "identity_derangement" else AUDIT_IDS,
            "inference": "descriptive_only_no_intervals",
            "identity_cycle_couples_recipients": label == "identity_derangement"}
    basis_draws = []
    for draw in range(RANDOM_BASIS_DRAWS):
        check()
        basis_draws.append(_loss(centered, bundle["random_basis_predictions"][:, draw]))
    basis = _draw_summary(basis_draws, losses[M3_VARIANTS[0]])
    basis.update(matched_true_variant=M3_VARIANTS[0], inference="descriptive_only_no_intervals")
    warps = {}
    for index, name in enumerate(M2_VARIANTS):
        draws = []
        for draw in range(MATCHED_WARP_DRAWS):
            check()
            draws.append(_loss(centered, bundle["matched_warp_predictions"][:, index, draw]))
        warps[name] = _draw_summary(draws, losses[name])
        warps[name]["inference"] = "descriptive_only_no_intervals"
    check()
    return {
        "ordinary": {"source_training_ids": DEVELOPMENT_IDS, "losses_by_model": losses},
        "primary": _primary_family(gains),
        "controls": {
            "calibration": calibration, "random_local_basis": basis,
            "matched_random_warp": warps, "condition_mapping": bundle["condition_mapping"],
            "draw_aggregation": "square_each_fixed_draw_before_averaging_losses_not_predictions",
            "inference": "descriptive_only_outside_primary_family",
            "selects_variants_or_rescues_failure": False,
        },
    }


def _figure(primary, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 10), layout="constrained")
    offsets = np.linspace(-.19, .19, PARTICIPANTS)
    colors = {"positive_heldout_observed_map_prediction_benefit": "#236a40",
              "negative_mean_gain": "#a14435", "inconclusive": "#455b73"}
    for index, name in enumerate(VARIANTS):
        entry = primary["variants"][name]
        mean = entry["mean_gain"]
        lower, upper = entry["simultaneous_95_percent_interval"]
        axis.scatter(entry["participant_gain"], index + offsets, s=17, alpha=.5, color="#70869b",
                     label="12 participant gains" if index == 0 else None)
        axis.errorbar(mean, index, xerr=[[mean - lower], [upper - mean]], fmt="o",
                      color=colors[entry["decision"]], capsize=4, linewidth=1.5,
                      label="Mean and simultaneous 95% interval" if index == 0 else None)
    axis.axvline(0., color=".55", linewidth=.9)
    axis.set(yticks=range(FAMILY_SIZE), yticklabels=VARIANTS,
             xlabel="Observed-map gain: M0b loss − model loss (positive favors model)",
             title="EP15: all 16 predictors in the 12 held-out audit participants")
    axis.invert_yaxis()
    axis.spines[["top", "right"]].set_visible(False)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    axis.legend(loc="lower right", fontsize=8)
    figure.supxlabel("Bonferroni participant-t intervals, df=11; conditional on the fixed development fit and 188 locations.\n"
                     "Same-account procedural separation; observed-map benefit does not identify a biological mechanism.", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(root / ("observed_map_audit." + extension), dpi=180)
    plt.close(figure)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _three_lines(report):
    descriptions = []
    for name in VARIANTS:
        entry = report["primary"]["variants"][name]
        low, high = entry["simultaneous_95_percent_interval"]
        descriptions.append(f"{name}: {entry['mean_gain']:.6g} [{low:.6g}, {high:.6g}], {entry['decision']}")
    return (
        "All 12 original audit participants, 18 Task-B-only conditions, four partitions and the development-frozen 188 locations: all 16 Task-A-calibrated predictors evaluated against M0b; sources use all 12 development participants.\n"
        "Observed-map gains with simultaneous two-sided 95% Bonferroni participant-t intervals (df=11): "
        + "; ".join(descriptions) + ".\n"
        "Conditional on the fixed development fit, selected cohort and support; participant-t assumptions are not guaranteed by the balanced split. Controls are descriptive; no winner, equivalence, absence, biology-versus-measurement distinction, general-population or cross-site claim. Same-account procedural separation, not technical isolation or blinded evaluation; this audit is consumed for the narrower observed-map question.\n"
    )


def report_audit_predictions(predictions, observations, *, output_root, figure_root, policy, check):
    """Report the complete proposed audit only after its explicit adoption.

    No command-line reader is supplied. The separately gated runner must first
    load the frozen complete prediction bundle, then open Task B once. The
    reporter never changes support, cohort, model, threshold, or prediction.
    """
    require_adoption(policy)  # No input conversion or output side effect before authority.
    _require(callable(check), "an explicit resource callback is required")
    check()
    output, figures = Path(output_root), Path(figure_root)
    _require(output.resolve() != figures.resolve() and not output.exists() and not figures.exists(),
             "fresh distinct numeric-report and figure directories are required")
    bundle = validate_prediction_bundle(predictions, policy=policy, require_sampling_geometry=True)
    scores = _score(bundle, observations, check=check)
    report = {
        "status": "complete_observed_map_audit_report", "route_id": ROUTE_ID,
        "participant_ids": AUDIT_IDS, "participant_count": PARTICIPANTS,
        "development_ids": DEVELOPMENT_IDS, "condition_ids": CONDITIONS,
        "variant_ids": VARIANTS, "partition_run_ids": PARTITION_RUN_IDS,
        "voxel_ids": bundle["voxel_ids"], "support_size": SUPPORT_SIZE,
        "condition_weights": condition_weights(), "participant_weight": 1 / PARTICIPANTS,
        "partition_weight": 1 / 4, "voxel_weight": 1 / SUPPORT_SIZE,
        "loss_definition": "separately_spatially_centered_observation_and_prediction_then_mean_squared_voxel_error",
        "aggregation": "equal_four_partitions_equal_seven_domains_equal_conditions_within_domain_equal_people",
        "observation_includes_measurement_error": True,
        "source_fit_uses_audit_people": False, "prediction_bundle_status": bundle["status"],
        "prediction_bundle_Task_B_accessed": bundle["audit_Task_B_accessed"],
        "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "audit_Task_B_evaluated": True, "audit_remains_untouched_for_original_mechanism_question": False,
        "shared_Task_B_conditions_and_private_reliability_used": False,
        "biological_mechanism_or_reliable_signal_recovery_established": False,
        "independent_replication": False,
        "claim_limit": "observed_map_prediction_benefit_does_not_distinguish_biology_from_stable_measurement_structure",
        **scores,
    }
    summary = _three_lines(report)
    _require(len(summary.splitlines()) == 3, "exactly three summary lines are required")
    check()
    output.mkdir(parents=True, mode=0o700)
    figures.mkdir(parents=True, mode=0o700)
    _figure(report["primary"], figures)
    check()
    with (output / "observed_map_audit.complete.json").open("x", encoding="utf-8") as destination:
        json.dump(_jsonable(report), destination, indent=2, allow_nan=False)
        destination.write("\n")
    (output / "3lines.md").write_text(summary, encoding="utf-8")
    (figures / "3lines.md").write_text(summary, encoding="utf-8")
    return report
