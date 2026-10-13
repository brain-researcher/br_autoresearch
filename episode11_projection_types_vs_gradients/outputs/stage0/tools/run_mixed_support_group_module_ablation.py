#!/usr/bin/env python3
"""Synthetic-only attribution of the EP11 finite-group estimator failure.

This diagnostic uses the fixed 56-coordinate mixed-support score and the
outcome-blind 206-cell/36-unit metadata layout.  It never reads a real regional
projection outcome.  The group-module panel holds the known synthetic
group-free backbone fixed, integrates every audit label, and changes one
development-side module at a time: discovery, signature estimation, and
position-dependent gating.  A small finite-D12 arm reruns the current fitted
H-minus-C pipeline after the generator subset-invariance repair.

The effect calibration remains the existing direction-aligned projected-C
proxy.  Therefore this program is an estimator attribution diagnostic, not
full-family population calibration or registered recovery qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t

import run_mixed_support_56d_qualification as core
import run_mixed_support_design_sensitivity as design


ARM_NAMES = (
    "MOD_DISC",
    "MOD_LABEL",
    "MOD_SIGNATURE",
    "MOD_GATE_FIT",
    "MOD_GATE_TRUTH",
)
CONTRASTS = {
    "label_discovery_rescue_including_downstream_signature_and_intercept_refit": (
        "MOD_DISC",
        "MOD_LABEL",
    ),
    "signature_rescue_given_true_labels": ("MOD_LABEL", "MOD_SIGNATURE"),
    "position_gating_need": ("MOD_SIGNATURE", "MOD_GATE_TRUTH"),
    "smooth_gate_remaining_regret": ("MOD_GATE_FIT", "MOD_GATE_TRUTH"),
}


def equal_unit_cell_weights(
    layout: core.MetadataLayout, indices: np.ndarray
) -> np.ndarray:
    units = layout.units[np.asarray(indices, dtype=int)]
    unique, counts = np.unique(units, return_counts=True)
    count_by_unit = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_unit[unit] for unit in units], dtype=float)
    return weights * (len(weights) / weights.sum())


def equal_unit_prevalence(
    layout: core.MetadataLayout, indices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    units = layout.units[np.asarray(indices, dtype=int)]
    per_unit = [float(labels[units == unit].mean()) for unit in sorted(set(units.tolist()))]
    probability_one = float(np.clip(np.mean(per_unit), 1.0e-4, 1.0 - 1.0e-4))
    return np.array([1.0 - probability_one, probability_one])


def smooth_gate_basis(layout: core.MetadataLayout, indices: np.ndarray) -> np.ndarray:
    # The first global soma PC contains the planted gate; depth and a quadratic
    # term give the diagnostic the registered smooth position/depth form
    # without using an outcome or audit prevalence.
    pc0 = core.standardized(layout.pc[:, 0])[indices]
    depth = core.standardized(layout.layer_depth[:, None])[:, 0][indices]
    return np.column_stack([np.ones(len(indices)), pc0, pc0**2 - 1.0, depth])


def fit_equal_unit_smooth_gate(
    layout: core.MetadataLayout,
    development: np.ndarray,
    labels: np.ndarray,
    ridge: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = np.asarray(labels, dtype=float)
    design_matrix = smooth_gate_basis(layout, development)
    weights = equal_unit_cell_weights(layout, development)
    penalty = np.array([0.0, ridge, ridge, ridge])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(design_matrix @ beta, -25.0, 25.0)
        value = np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta))
        value /= weights.sum()
        value += 0.5 * float(np.sum(penalty * beta**2))
        probability = expit(eta)
        gradient = design_matrix.T @ (weights * (probability - labels))
        gradient /= weights.sum()
        gradient += penalty * beta
        return float(value), gradient

    prevalence = float(np.clip(np.average(labels, weights=weights), 1.0e-4, 1.0 - 1.0e-4))
    intercept = math.log(prevalence / (1.0 - prevalence))
    starts = (np.array([intercept, 0.0, 0.0, 0.0]), np.zeros(4))
    fits = [
        minimize(
            lambda beta: objective(beta)[0],
            start,
            jac=lambda beta: objective(beta)[1],
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0)] * 4,
            options={"maxiter": 200, "ftol": 1.0e-11, "gtol": 1.0e-8},
        )
        for start in starts
    ]
    converged = [fit for fit in fits if bool(fit.success)]
    if not converged:
        messages = "; ".join(str(fit.message) for fit in fits)
        raise RuntimeError(f"Smooth gate optimization did not converge: {messages}")
    best = min(converged, key=lambda fit: float(fit.fun))
    if not np.isfinite(best.fun) or not np.isfinite(best.x).all():
        raise RuntimeError("Smooth gate optimization returned nonfinite values")
    return np.asarray(best.x, dtype=float), {
        "success": bool(best.success),
        "objective": float(best.fun),
        "iterations": int(getattr(best, "nit", -1)),
        "starts": len(starts),
    }


def gate_probabilities(
    layout: core.MetadataLayout, indices: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    probability_one = expit(np.clip(smooth_gate_basis(layout, indices) @ beta, -25.0, 25.0))
    return np.column_stack([1.0 - probability_one, probability_one])


def fit_current_conditional_signatures(
    outcomes: np.ndarray,
    truth: core.SyntheticTruth,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the current pooled-cell conditional signature regression."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2).astype(float)
    if min(counts) < 2:
        raise ValueError("Both development groups need at least two cells")
    pooled_weights = (counts + 1.0) / (counts.sum() + 2.0)
    codes = np.array(
        [
            -math.sqrt(pooled_weights[1] / pooled_weights[0]),
            math.sqrt(pooled_weights[0] / pooled_weights[1]),
        ]
    )
    latent_design = codes[labels, None]
    support_coef, value_coef = core.fit_adjustments(
        outcomes,
        truth.base_support_logits,
        truth.base_value_mean,
        latent_design,
    )
    return codes[:, None] @ support_coef, codes[:, None] @ value_coef


def mixture_scores(
    outcomes: np.ndarray,
    base_support_logits: np.ndarray,
    base_value_mean: np.ndarray,
    support_signatures: np.ndarray,
    value_signatures: np.ndarray,
    group_probabilities: np.ndarray,
    sigma: float,
    return_components: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(group_probabilities, dtype=float)
    if probabilities.shape != (len(outcomes), 2):
        raise ValueError("Group probabilities must be n-by-2")
    probabilities = np.clip(probabilities, 1.0e-300, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    components = np.empty((len(outcomes), 2), dtype=float)
    variance = float(sigma**2)
    for row, face in enumerate(core.make_face_data(value) for value in outcomes):
        mask = np.zeros(core.N_COORDINATES, dtype=bool)
        mask[face.active] = True
        dimension = len(face.active) - 1
        normal_constant = dimension * math.log(2.0 * math.pi * variance)
        for group in range(2):
            support_logits = base_support_logits[row] + support_signatures[group]
            support_logp = core.active_set_log_probability(mask, support_logits)
            mean_active = (
                base_value_mean[row, face.active]
                + value_signatures[group, face.active]
            )
            mean_ilr = face.helmert.T @ mean_active
            residual = face.ilr - mean_ilr
            value_logp = -0.5 * (
                normal_constant + float(residual @ residual) / variance
            )
            value_logp += face.log_jacobian
            components[row, group] = (
                math.log(probabilities[row, group]) + support_logp + value_logp
            )
    score = logsumexp(components, axis=1)
    return score, components if return_components else None


def module_accuracy(components: np.ndarray, truth_labels: np.ndarray) -> float:
    inferred = np.argmax(components, axis=1)
    direct = float(np.mean(inferred == truth_labels))
    flipped = float(np.mean((1 - inferred) == truth_labels))
    return max(direct, flipped)


def one_replicate(
    layout: core.MetadataLayout,
    roles: core.RoleSplit,
    fixture_name: str,
    amplitude: float,
    replicate: int,
    seed: int,
    projected_c_parameters: Sequence[float],
    run_current_integration: bool,
    margin: float,
) -> dict[str, object]:
    rng = np.random.default_rng(
        core.stable_u64(seed, "module_full36", fixture_name, replicate) % (2**32)
    )
    outcomes, truth = core.generate_synthetic_outcomes(
        layout,
        np.arange(len(layout.frame), dtype=int),
        "reusable_groups",
        amplitude,
        rng,
    )
    development = roles.development
    audit = roles.audit
    development_outcomes = outcomes[development]
    audit_outcomes = outcomes[audit]
    development_truth = core.subset_synthetic_truth(truth, development)
    audit_truth = core.subset_synthetic_truth(truth, audit)

    oracle_score = core.oracle_mixed_support_scores(
        audit_outcomes, audit_truth, include_groups=True
    )
    projected_c_score = core.projected_continuous_oracle_scores(
        audit_outcomes, audit_truth, projected_c_parameters
    )

    residual = core.residual_features(
        development_outcomes,
        development_truth.base_support_logits,
        development_truth.base_value_mean,
    )
    discovered_labels = core.two_means(residual)
    true_labels = development_truth.groups

    discovered_support, discovered_value = fit_current_conditional_signatures(
        development_outcomes, development_truth, discovered_labels
    )
    discovered_prevalence = equal_unit_prevalence(
        layout, development, discovered_labels
    )
    discovered_probability = np.tile(discovered_prevalence, (len(audit), 1))

    if set(true_labels.tolist()) != {0, 1} or min(np.bincount(true_labels, minlength=2)) < 2:
        raise RuntimeError("Strong diagnostic replicate lacks both planted development groups")
    label_support, label_value = fit_current_conditional_signatures(
        development_outcomes, development_truth, true_labels
    )
    true_prevalence = equal_unit_prevalence(layout, development, true_labels)
    constant_probability = np.tile(true_prevalence, (len(audit), 1))
    gate_beta, gate_fit = fit_equal_unit_smooth_gate(layout, development, true_labels)
    fitted_gate_probability = gate_probabilities(layout, audit, gate_beta)

    arm_inputs = {
        "MOD_DISC": (
            discovered_support,
            discovered_value,
            discovered_probability,
        ),
        "MOD_LABEL": (label_support, label_value, constant_probability),
        "MOD_SIGNATURE": (
            audit_truth.group_support_shift,
            audit_truth.group_value_shift,
            constant_probability,
        ),
        "MOD_GATE_FIT": (
            audit_truth.group_support_shift,
            audit_truth.group_value_shift,
            fitted_gate_probability,
        ),
        "MOD_GATE_TRUTH": (
            audit_truth.group_support_shift,
            audit_truth.group_value_shift,
            audit_truth.group_probabilities,
        ),
    }
    arm_regrets: dict[str, float] = {}
    arm_accuracy: dict[str, float] = {}
    arm_scores: dict[str, np.ndarray] = {}
    for arm, (support_signature, value_signature, probabilities) in arm_inputs.items():
        arm_score, components = mixture_scores(
            audit_outcomes,
            audit_truth.base_support_logits,
            audit_truth.base_value_mean,
            support_signature,
            value_signature,
            probabilities,
            audit_truth.noise_sigma,
            return_components=True,
        )
        assert components is not None
        arm_regrets[arm] = core.equal_unit_mean(
            layout, audit, oracle_score - arm_score
        )
        arm_scores[arm] = arm_score
        arm_accuracy[arm] = (
            module_accuracy(components, audit_truth.groups)
            if amplitude > 0.0
            else math.nan
        )

    truth_gate_score, _ = mixture_scores(
        audit_outcomes,
        audit_truth.base_support_logits,
        audit_truth.base_value_mean,
        audit_truth.group_support_shift,
        audit_truth.group_value_shift,
        audit_truth.group_probabilities,
        audit_truth.noise_sigma,
    )
    np.testing.assert_allclose(truth_gate_score, oracle_score, atol=1.0e-10, rtol=0.0)
    if amplitude == 0.0:
        for arm in ("MOD_SIGNATURE", "MOD_GATE_FIT", "MOD_GATE_TRUTH"):
            np.testing.assert_allclose(
                arm_scores[arm], oracle_score, atol=1.0e-10, rtol=0.0
            )

    row: dict[str, object] = {
        "fixture": fixture_name,
        "replicate": replicate,
        "generator_amplitude": amplitude,
        "oracle_H_minus_projected_C_proxy": core.equal_unit_mean(
            layout, audit, oracle_score - projected_c_score
        ),
        "smooth_gate_fit_success": bool(gate_fit["success"]),
        "smooth_gate_beta_json": json.dumps(gate_beta.tolist(), separators=(",", ":")),
        "discovered_label_accuracy_development": (
            max(
                float(np.mean(discovered_labels == true_labels)),
                float(np.mean((1 - discovered_labels) == true_labels)),
            )
            if amplitude > 0.0
            else math.nan
        ),
        "current_H_minus_C": math.nan,
        "current_H_minus_C_halfwidth": math.nan,
        "current_gain_detected": math.nan,
        "current_group_accuracy": math.nan,
        "real_projection_outcomes_read": False,
    }
    for arm in ARM_NAMES:
        row[f"regret_{arm}"] = arm_regrets[arm]
        row[f"accuracy_{arm}"] = arm_accuracy[arm]

    if run_current_integration:
        models, _, _ = design.fit_with_inner_unit_cv(
            layout,
            development,
            development_outcomes,
            seed=design.development_fit_seed(
                layout, development, analysis_seed=seed, replicate=replicate
            ),
        )
        c_score, _ = core.score_model(
            models["C"], core.design_for(models["C"], layout, audit), audit_outcomes
        )
        h_score, h_components = core.score_model(
            models["H"],
            core.design_for(models["H"], layout, audit),
            audit_outcomes,
            return_components=True,
        )
        assert h_components is not None
        statistics = design.audit_unit_statistics(
            layout, audit, h_score - c_score, margin
        )
        row["current_H_minus_C"] = statistics["mean"]
        row["current_H_minus_C_halfwidth"] = statistics["one_sided95_halfwidth"]
        row["current_gain_detected"] = statistics["predictive_gain_detected"]
        row["current_group_accuracy"] = design.h_group_accuracy(
            h_components, models["H"], audit_truth.groups, amplitude > 0.0
        )
    return row


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.nan
    return mean, se


def sequential_mcse_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    strong = frame[frame["fixture"].eq("strong_projected_proxy_target")]
    zero = frame[frame["fixture"].eq("zero_group")]
    diagnostics: dict[str, float] = {}
    for name, (before, after) in CONTRASTS.items():
        values = (
            strong[f"regret_{before}"].to_numpy()
            - strong[f"regret_{after}"].to_numpy()
        )
        diagnostics[f"strong_{name}"] = mean_se(values)[1]
    # The exact-signature zero arms are checked cellwise. These two estimated
    # arms remain stochastic and join the sequential precision requirement.
    for arm in ("MOD_DISC", "MOD_LABEL"):
        diagnostics[f"zero_regret_{arm}"] = mean_se(
            zero[f"regret_{arm}"].to_numpy()
        )[1]
    return diagnostics


def summarize(
    rows: list[dict[str, object]],
    calibration: core.OracleCalibration,
    batch_size: int,
    target_mcse: float,
    integration_replicates: int,
    maximum_replicates: int,
) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    fixtures: dict[str, object] = {}
    for fixture, group in frame.groupby("fixture", sort=True):
        arms: dict[str, object] = {}
        for arm in ARM_NAMES:
            mean, se = mean_se(group[f"regret_{arm}"].to_numpy())
            accuracy, accuracy_se = mean_se(group[f"accuracy_{arm}"].to_numpy())
            arms[arm] = {
                "mean_oracle_regret_nats_per_neuron": mean,
                "MC_SE": se,
                "mean_pooled_cell_permutation_invariant_accuracy": accuracy,
                "accuracy_MC_SE": accuracy_se,
            }
        current = group[np.isfinite(group["current_H_minus_C"])]
        current_mean, current_se = mean_se(current["current_H_minus_C"].to_numpy())
        proxy_mean, proxy_se = mean_se(
            group["oracle_H_minus_projected_C_proxy"].to_numpy()
        )
        discovery_mean, discovery_se = mean_se(
            group["discovered_label_accuracy_development"].to_numpy()
        )
        fixtures[str(fixture)] = {
            "replicates": int(len(group)),
            "arms": arms,
            "oracle_H_minus_projected_C_proxy": {
                "mean_nats_per_neuron": proxy_mean,
                "MC_SE": proxy_se,
            },
            "development_discovery_accuracy_pooled_cell": {
                "mean": discovery_mean,
                "MC_SE": discovery_se,
            },
            "smooth_gate_optimizer_success_rate": float(
                group["smooth_gate_fit_success"].mean()
            ),
            "current_finite_D12": {
                "replicates": int(len(current)),
                "mean_H_minus_C_nats_per_neuron": current_mean,
                "MC_SE": current_se,
                "predictive_gain_detection_rate": (
                    float(current["current_gain_detected"].mean()) if len(current) else None
                ),
                "mean_one_sided95_halfwidth": (
                    float(current["current_H_minus_C_halfwidth"].mean())
                    if len(current)
                    else None
                ),
                "mean_group_accuracy": (
                    float(current["current_group_accuracy"].mean())
                    if len(current) and np.isfinite(current["current_group_accuracy"]).any()
                    else None
                ),
            },
        }

    strong = frame[frame["fixture"].eq("strong_projected_proxy_target")]
    paired: dict[str, object] = {}
    for name, (before, after) in CONTRASTS.items():
        # Rescue gain = regret before - regret after.  For the final entry the
        # tuple is ordered so this is fitted-gate regret above truth-gate.
        values = (
            strong[f"regret_{before}"].to_numpy()
            - strong[f"regret_{after}"].to_numpy()
        )
        mean, se = mean_se(values)
        critical = float(student_t.ppf(0.95, len(values) - 1)) if len(values) > 1 else math.nan
        paired[name] = {
            "mean_nats_per_neuron": mean,
            "MC_SE": se,
            "one_sided95_lower": mean - critical * se,
            "one_sided95_upper": mean + critical * se,
            "lower_exceeds_0_005": bool(mean - critical * se > 0.005),
        }

    stopping_mcses = sequential_mcse_diagnostics(frame)
    maximum_achieved_mcse = max(stopping_mcses.values())
    strong_replicates = int(len(strong))
    sequential_target_met = bool(maximum_achieved_mcse <= target_mcse)
    for result in paired.values():
        result["material_bottleneck_status"] = (
            "yes"
            if sequential_target_met and result["lower_exceeds_0_005"]
            else "no"
            if sequential_target_met
            else "unresolved_MC_precision"
        )
    calibration_dict = {
        "protocol_id": calibration.protocol_id,
        "role": "direction_aligned_projected_C_proxy_not_full_family_calibration",
        "fit_and_verification_replicates_each": calibration.replicates,
        "amplitude_grid": list(calibration.amplitude_grid),
        "projection_family": calibration.projection_family,
        "raw_mean_gain_by_scenario": {
            key: list(value) for key, value in calibration.raw_mean_gain_by_scenario.items()
        },
        "raw_MC_SE_by_scenario": {
            key: list(value) for key, value in calibration.raw_se_by_scenario.items()
        },
        "points": [
            {
                "target_margin_multiplier": point.target_margin_multiplier,
                "target_proxy_gain_nats_per_neuron": point.target_gain_nats_per_neuron,
                "generator_amplitude": point.generator_amplitude,
                "independent_verification_mean_gain": point.independent_verification_mean_gain,
                "independent_verification_MC_SE": point.independent_verification_se,
                "independent_verification_lower95": point.independent_verification_lower95,
                "independent_verification_upper95": point.independent_verification_upper95,
                "projected_C_parameters": list(point.projected_C_parameters),
            }
            for point in calibration.points
        ],
    }
    return {
        "schema_version": "ep11_group_module_ablation_v1",
        "analysis_class": "synthetic_group_module_attribution_not_qualification",
        "real_projection_outcomes_read": False,
        "outcome_bearing_search_authorized": False,
        "fixed_role_count": "D12_A8_16_unassigned_metadata_only_not_locked",
        "generator_subset_invariance_repair": True,
        "module_backbone": "known_synthetic_group_free_truth",
        "audit_labels_integrated": True,
        "current_finite_D12_reference_role": (
            "historical_overconstrained_frozen_C_inside_H_not_production_contract"
        ),
        "batch_size": batch_size,
        "target_paired_contrast_MC_SE": target_mcse,
        "sequential_MC_SE_diagnostics": stopping_mcses,
        "maximum_achieved_sequential_MC_SE": maximum_achieved_mcse,
        "sequential_target_met": sequential_target_met,
        "stopped_at_replicate_cap": bool(
            strong_replicates >= maximum_replicates and not sequential_target_met
        ),
        "component_interpretation_allowed": sequential_target_met,
        "integration_replicates": integration_replicates,
        "fixtures": fixtures,
        "paired_attribution": paired,
        "projected_proxy_calibration": calibration_dict,
        "remaining_required_next": (
            "population-optimal complete-current-family C/H calibration, then an "
            "equal-unit jointly optimized finite-D12 estimator with smooth gating"
        ),
    }


def write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "group_module_ablation_replicates.csv", index=False)
    with (output_dir / "group_module_ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(design.json_ready(summary), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    lines = [
        "# EP11 synthetic group-module attribution",
        "",
        "- Real projection outcomes read: **no**",
        "- Outcome-bearing search authorized: **no**",
        "- Common backbone: known synthetic group-free density",
        "- Audit labels: integrated for every score",
        "- Effect scale: repaired subset-invariant generator, but still the projected-C proxy rather than full-family calibration",
        "",
        "| Fixture | Arm | Replicates | Mean oracle regret | MC SE | Mean group accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fixture, fixture_summary in summary["fixtures"].items():
        for arm, arm_summary in fixture_summary["arms"].items():
            accuracy = arm_summary[
                "mean_pooled_cell_permutation_invariant_accuracy"
            ]
            accuracy_text = (
                f"{accuracy:.3f}"
                if accuracy is not None and np.isfinite(accuracy)
                else "NA"
            )
            lines.append(
                f"| {fixture} | {arm} | {fixture_summary['replicates']} | "
                f"{arm_summary['mean_oracle_regret_nats_per_neuron']:.4f} | "
                f"{arm_summary['MC_SE']:.4f} | "
                f"{accuracy_text} |"
            )
    lines.extend(
        [
            "",
            "## Strong-fixture paired attribution",
            "",
            "| Contrast | Mean rescue/regret | MC SE | Paired one-sided 95% bounds (central 90% interval) | Material (>0.005 lower bound) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in summary["paired_attribution"].items():
        lines.append(
            f"| {name} | {result['mean_nats_per_neuron']:.4f} | "
            f"{result['MC_SE']:.4f} | "
            f"[{result['one_sided95_lower']:.4f}, {result['one_sided95_upper']:.4f}] | "
            f"{result['material_bottleneck_status']} |"
        )
    lines.extend(
        [
            "",
            "## Fixture diagnostics",
            "",
            "| Fixture | Oracle H - projected-C proxy | MC SE | Development discovery accuracy | Gate success rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fixture, fixture_summary in summary["fixtures"].items():
        proxy = fixture_summary["oracle_H_minus_projected_C_proxy"]
        discovery = fixture_summary["development_discovery_accuracy_pooled_cell"]
        discovery_text = (
            f"{discovery['mean']:.3f}"
            if discovery["mean"] is not None and np.isfinite(discovery["mean"])
            else "NA"
        )
        lines.append(
            f"| {fixture} | {proxy['mean_nats_per_neuron']:.4f} | "
            f"{proxy['MC_SE']:.4f} | {discovery_text} | "
            f"{fixture_summary['smooth_gate_optimizer_success_rate']:.3f} |"
        )
    strong_calibration = next(
        point
        for point in summary["projected_proxy_calibration"]["points"]
        if point["target_margin_multiplier"] == 2.0
    )
    realized_proxy = summary["fixtures"]["strong_projected_proxy_target"][
        "oracle_H_minus_projected_C_proxy"
    ]
    lines.extend(
        [
            "",
            "The nominal strong-fixture target was 0.100 nats/neuron, but the",
            "independent projected-proxy calibration realized "
            f"{strong_calibration['independent_verification_mean_gain']:.4f} "
            f"(MC SE {strong_calibration['independent_verification_MC_SE']:.4f}); "
            f"the diagnostic replicates realized {realized_proxy['mean_nats_per_neuron']:.4f}. "
            "The proxy is not the",
            "complete fitted C family, so neither value qualifies the production effect scale.",
            "",
            "## Current finite-D12 reference",
            "",
            "| Fixture | Replicates | Mean H - C | MC SE | Detection rate | Mean one-sided 95% half-width | Mean group accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fixture, fixture_summary in summary["fixtures"].items():
        current = fixture_summary["current_finite_D12"]
        accuracy = current["mean_group_accuracy"]
        accuracy_text = f"{accuracy:.3f}" if accuracy is not None else "NA"
        lines.append(
            f"| {fixture} | {current['replicates']} | "
            f"{current['mean_H_minus_C_nats_per_neuron']:.4f} | "
            f"{current['MC_SE']:.4f} | "
            f"{current['predictive_gain_detection_rate']:.3f} | "
            f"{current['mean_one_sided95_halfwidth']:.4f} | {accuracy_text} |"
        )
    lines.extend(
        [
            "",
            "This reference is the historical overconstrained fitter that freezes the",
            "fitted `C` backbone into `H`; it is not the production-contract `H - C`",
            "comparison.",
        ]
    )
    lines.extend(
        [
            "",
            "`MOD_GATE_TRUTH` is required to reproduce the exact oracle density",
            "cellwise. The attribution is conditional and ordered; it is not a variance",
            f"decomposition. Sequential MC-SE target met: **{summary['sequential_target_met']}**; ",
            f"maximum monitored MC SE: **{summary['maximum_achieved_sequential_MC_SE']:.4f}**.",
            "Component-level interpretation is disabled if that target is false.",
            "All reported group accuracies are pooled-cell, truth-assisted",
            "permutation-invariant diagnostics; they are not equal-unit evidence that a",
            "fixed development label transports to audit units.",
            "Before any recovery suite, EP11 still needs population-optimal",
            "complete-family C/H calibration and a finite-D12 estimator that uses equal-unit",
            "objectives, smooth development-only gating, marginal group likelihood, and",
            "separate but budget-matched C and H optimization.",
            "",
        ]
    )
    (output_dir / "group_module_ablation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def executable_self_test(layout: core.MetadataLayout) -> dict[str, object]:
    roles = design.nested_role_partitions(layout, seed=711)[
        "nested_baseline_12_dev_8_audit_16_unassigned"
    ]
    outcomes, truth = core.generate_synthetic_outcomes(
        layout,
        np.arange(len(layout.frame), dtype=int),
        "reusable_groups",
        0.25,
        np.random.default_rng(712),
    )
    development_truth = core.subset_synthetic_truth(truth, roles.development)
    audit_truth = core.subset_synthetic_truth(truth, roles.audit)
    discovered = core.two_means(
        core.residual_features(
            outcomes[roles.development],
            development_truth.base_support_logits,
            development_truth.base_value_mean,
        )
    )
    support, value = fit_current_conditional_signatures(
        outcomes[roles.development], development_truth, discovered
    )
    beta, diagnostics = fit_equal_unit_smooth_gate(
        layout, roles.development, development_truth.groups
    )
    probabilities = gate_probabilities(layout, roles.audit, beta)
    score, _ = mixture_scores(
        outcomes[roles.audit],
        audit_truth.base_support_logits,
        audit_truth.base_value_mean,
        support,
        value,
        probabilities,
        audit_truth.noise_sigma,
    )
    oracle = core.oracle_mixed_support_scores(
        outcomes[roles.audit], audit_truth, include_groups=True
    )
    exact, _ = mixture_scores(
        outcomes[roles.audit],
        audit_truth.base_support_logits,
        audit_truth.base_value_mean,
        audit_truth.group_support_shift,
        audit_truth.group_value_shift,
        audit_truth.group_probabilities,
        audit_truth.noise_sigma,
    )
    np.testing.assert_allclose(exact, oracle, atol=1.0e-10, rtol=0.0)
    if not np.isfinite(score).all() or not np.isfinite(probabilities).all():
        raise AssertionError("Module self-test produced nonfinite values")
    return {
        "status": "pass",
        "real_projection_outcomes_read": False,
        "outcome_bearing_search_authorized": False,
        "exact_truth_gate_matches_oracle_cellwise": True,
        "smooth_gate_fit_finite": True,
        "smooth_gate_optimizer_success": bool(diagnostics["success"]),
        "audit_labels_integrated": True,
    }


def main() -> None:
    episode_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=episode_root / "outputs/stage0/whole_cortex_ssptr_inventory.csv",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=episode_root / "outputs/stage0/allen_target_vocabulary.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maximum-replicates", type=int, default=256)
    parser.add_argument("--integration-replicates", type=int, default=32)
    parser.add_argument("--target-contrast-mcse", type=float, default=0.005)
    parser.add_argument("--calibration-replicates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    core.validate_vocabulary(args.vocabulary)
    layout = core.load_metadata(args.inventory)
    if args.self_test:
        print(json.dumps(executable_self_test(layout), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    if args.batch_size < 2 or args.maximum_replicates < args.batch_size:
        raise ValueError("Invalid sequential replicate limits")
    if args.maximum_replicates % args.batch_size:
        raise ValueError("Maximum replicates must be a multiple of batch size")
    if args.integration_replicates < 0 or args.integration_replicates > args.maximum_replicates:
        raise ValueError("Integration replicate count is outside the run range")

    calibration = core.calibrate_oracle_group_effects(
        layout,
        scenarios=["reusable_groups"],
        target_margin_multipliers=[0.0, 2.0],
        margin=args.margin,
        seed=core.stable_u64(args.seed, "repaired_proxy_calibration") % (2**32),
        replicates=args.calibration_replicates,
        amplitude_grid=core.ORACLE_CALIBRATION_AMPLITUDES,
    )
    # The projected proxy is conditional on its provisional audit covariate
    # distribution. Reuse that exact fixed D12/A8/16 metadata split in the
    # module panel rather than silently transporting its amplitude and C
    # parameters to a different role split.
    roles = core.RoleSplit(
        development=np.flatnonzero(
            np.isin(layout.units, calibration.fixed_role_manifest["development"])
        ),
        audit=np.flatnonzero(
            np.isin(layout.units, calibration.fixed_role_manifest["audit"])
        ),
        unassigned=np.flatnonzero(
            np.isin(layout.units, calibration.fixed_role_manifest["unassigned"])
        ),
    )
    if (
        len(set(layout.units[roles.development].tolist())) != 12
        or len(set(layout.units[roles.audit].tolist())) != 8
        or len(set(layout.units[roles.unassigned].tolist())) != 16
    ):
        raise AssertionError("Calibration role manifest did not reconstruct D12/A8/16")
    points = {
        point.target_margin_multiplier: point for point in calibration.points
    }
    fixtures = (
        ("zero_group", points[0.0]),
        ("strong_projected_proxy_target", points[2.0]),
    )

    rows: list[dict[str, object]] = []
    for batch_start in range(0, args.maximum_replicates, args.batch_size):
        batch_stop = batch_start + args.batch_size
        for replicate in range(batch_start, batch_stop):
            for fixture_name, point in fixtures:
                rows.append(
                    one_replicate(
                        layout,
                        roles,
                        fixture_name,
                        point.generator_amplitude,
                        replicate,
                        args.seed,
                        point.projected_C_parameters,
                        run_current_integration=replicate < args.integration_replicates,
                        margin=args.margin,
                    )
                )
        monitored_mcses = sequential_mcse_diagnostics(pd.DataFrame(rows))
        if max(monitored_mcses.values()) <= args.target_contrast_mcse:
            break

    summary = summarize(
        rows,
        calibration,
        args.batch_size,
        args.target_contrast_mcse,
        args.integration_replicates,
        args.maximum_replicates,
    )
    write_outputs(args.output_dir, rows, summary)
    print(json.dumps(design.json_ready(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
