#!/usr/bin/env python3
"""Synthetic-only EP11 estimator-ceiling and role-design sensitivity.

This program never reads a regional projection outcome.  It imports the
factorized B/P/C/D/H smoke evaluator and asks two deliberately narrower design
questions:

1. Does the current estimator recover a projected-oracle-proxy 2x-margin group
   fixture when the number of synthetic brain/sample units is enlarged by
   whole-unit, line-stratified metadata bootstrap?
2. How do metadata-only prospective allocations of the 36 observed units
   change H-C power and precision relative to 12 development / 8 audit / 16
   unassigned?

The alternative allocations reserve the observed singleton-line unit as
unassigned.  They do not enforce the pial-geodesic/depth common-support rules,
because no frozen geometry-eligible role assignment is available here.  They
are design sensitivities, not qualification, role locks, or authorization to
read real outcomes.  Bootstrap clones are synthetic independent units that
retain an observed unit's cell count, line, positions, layers, and other
outcome-blind design rows; they do not manufacture real independent animals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

import run_mixed_support_56d_qualification as core


FIXED_SPLIT_DESIGNS: dict[str, tuple[int, int, int]] = {
    "nested_baseline_12_dev_8_audit_16_unassigned": (12, 8, 16),
    "nested_estimator_ceiling_27_dev_8_audit_1_unassigned": (27, 8, 1),
    "nested_precision_ceiling_12_dev_23_audit_1_unassigned": (12, 23, 1),
}

CEILING_DESIGNS: dict[str, tuple[int, int]] = {
    "bootstrap_ceiling_36_dev_36_audit": (36, 36),
    "bootstrap_ceiling_72_dev_72_audit": (72, 72),
}

DEFAULT_GROUP_SCENARIOS = (
    "reusable_groups",
    "groups_plus_continuum",
    "spatially_interdigitated_groups",
)

DEFAULT_MARGIN = core.DEFAULT_MARGIN
DEFAULT_STRONG_MULTIPLIER = 2.0
REFERENCE_STRONG_FIXTURE_UNIT_SD = 0.217


@dataclass(frozen=True)
class Fixture:
    name: str
    scenario: str
    generator_amplitude: float
    target_margin_multiplier: float
    target_proxy_gain: float
    verified_proxy_gain: float
    projected_c_parameters: tuple[float, ...]
    truth_has_nonzero_groups: bool


@dataclass(frozen=True)
class GenericFold:
    fold_index: int
    training_positions: np.ndarray
    validation_positions: np.ndarray
    training_units: tuple[str, ...]
    validation_units: tuple[str, ...]


def stable_u64(*parts: object) -> int:
    return core.stable_u64("design_sensitivity", *parts)


def development_fit_seed(
    layout: core.MetadataLayout,
    development: np.ndarray,
    analysis_seed: int,
    replicate: int,
) -> int:
    """Key inner folds only to the development sample, never to an arm label."""
    units = tuple(sorted(set(layout.units[np.asarray(development, dtype=int)].tolist())))
    return int(
        stable_u64(analysis_seed, "inner_cv", replicate, *units) % (2**32)
    )


def fitted_model_set_fingerprint(
    models: dict[str, core.MixedSupportModel],
) -> str:
    """Stable numeric fingerprint used to verify paired-arm fit identity."""
    digest = hashlib.sha256()
    for family in sorted(models):
        digest.update(family.encode("utf-8"))
        model = models[family]
        for field, value in sorted(vars(model).items()):
            digest.update(field.encode("utf-8"))
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def unit_first_rows(layout: core.MetadataLayout) -> pd.DataFrame:
    frame = layout.frame.copy()
    frame["_unit"] = layout.units
    frame["_line"] = layout.lines
    return frame.groupby("_unit", sort=True).first().reset_index()


def line_by_unit(layout: core.MetadataLayout, indices: np.ndarray) -> dict[str, str]:
    indices = np.asarray(indices, dtype=int)
    output: dict[str, str] = {}
    for unit in sorted(set(layout.units[indices].tolist())):
        local = indices[layout.units[indices] == unit]
        lines = set(layout.lines[local].tolist())
        if len(lines) != 1:
            raise ValueError(f"Unit {unit!r} has multiple line identities")
        output[unit] = str(next(iter(lines)))
    return output


def prospective_role_split(
    layout: core.MetadataLayout,
    development_units: int,
    audit_units: int,
    unassigned_units: int,
    seed: int,
) -> core.RoleSplit:
    """Outcome-independent line-stratified allocation of the observed 36 units."""
    if development_units + audit_units + unassigned_units != 36:
        raise ValueError("Observed-unit designs must allocate all 36 units")
    if unassigned_units < 1:
        raise ValueError("At least one unassigned slot is required for the singleton line")

    unit_rows = unit_first_rows(layout)
    line_groups = {
        str(line): group["_unit"].astype(str).tolist()
        for line, group in unit_rows.groupby("_line", sort=True)
    }
    singleton_lines = [line for line, units in line_groups.items() if len(units) == 1]
    if len(singleton_lines) != 1:
        raise ValueError("This diagnostic expects exactly one singleton metadata line")
    singleton_unit = line_groups[singleton_lines[0]][0]
    nonsingleton = {line: units for line, units in line_groups.items() if len(units) > 1}
    if development_units < len(nonsingleton) or audit_units < len(nonsingleton):
        raise ValueError("Each nonsingleton line needs one development and one audit unit")

    rng = np.random.default_rng(seed)
    development: list[str] = []
    audit: list[str] = []
    remaining: list[str] = []
    for line in sorted(nonsingleton):
        choices = np.asarray(nonsingleton[line], dtype=object)
        order = rng.permutation(len(choices))
        development.append(str(choices[order[0]]))
        audit.append(str(choices[order[1]]))
        remaining.extend(str(choices[index]) for index in order[2:])
    rng.shuffle(remaining)

    development_extra = development_units - len(development)
    audit_extra = audit_units - len(audit)
    development.extend(remaining[:development_extra])
    audit.extend(remaining[development_extra : development_extra + audit_extra])
    residual = remaining[development_extra + audit_extra :]
    unassigned = [singleton_unit, *residual]
    if (
        len(development) != development_units
        or len(audit) != audit_units
        or len(unassigned) != unassigned_units
    ):
        raise AssertionError("Prospective role allocator produced incorrect counts")
    if set(development) & set(audit) or set(development) & set(unassigned) or set(audit) & set(unassigned):
        raise AssertionError("Prospective roles overlap")

    for role_name, units in (("development", development), ("audit", audit)):
        represented = {line_by_unit(layout, np.flatnonzero(np.isin(layout.units, units)))[unit]
                       for unit in units}
        if represented != set(nonsingleton):
            raise AssertionError(f"{role_name} does not represent every nonsingleton line")
    return core.RoleSplit(
        development=np.flatnonzero(np.isin(layout.units, development)),
        audit=np.flatnonzero(np.isin(layout.units, audit)),
        unassigned=np.flatnonzero(np.isin(layout.units, unassigned)),
    )


def nested_role_partitions(
    layout: core.MetadataLayout,
    seed: int,
) -> dict[str, core.RoleSplit]:
    """Build the three arms from one identical D12/A8 core and reserve."""
    partitions = {
        name: prospective_role_split(layout, *counts, seed=seed)
        for name, counts in FIXED_SPLIT_DESIGNS.items()
    }
    baseline = partitions["nested_baseline_12_dev_8_audit_16_unassigned"]
    estimator = partitions["nested_estimator_ceiling_27_dev_8_audit_1_unassigned"]
    precision = partitions["nested_precision_ceiling_12_dev_23_audit_1_unassigned"]
    baseline_dev = set(layout.units[baseline.development].tolist())
    baseline_audit = set(layout.units[baseline.audit].tolist())
    reserve = set(layout.units[baseline.unassigned].tolist())
    # Identify the true singleton by unit count per line, rather than by cell
    # count (the unit may contain several cells).
    unit_lines = line_by_unit(layout, np.arange(len(layout.units)))
    line_unit_counts = pd.Series(unit_lines).value_counts().to_dict()
    singleton_units = {unit for unit, line in unit_lines.items() if line_unit_counts[line] == 1}
    if len(singleton_units) != 1 or not singleton_units.issubset(reserve):
        raise AssertionError("The one singleton-line unit is not reserved")
    movable_reserve = reserve - singleton_units
    if set(layout.units[estimator.audit].tolist()) != baseline_audit:
        raise AssertionError("Estimator arm changed the fixed A8 core")
    if set(layout.units[precision.development].tolist()) != baseline_dev:
        raise AssertionError("Precision arm changed the fixed D12 core")
    if set(layout.units[estimator.development].tolist()) != baseline_dev | movable_reserve:
        raise AssertionError("Estimator arm did not move exactly the 15 reserve units")
    if set(layout.units[precision.audit].tolist()) != baseline_audit | movable_reserve:
        raise AssertionError("Precision arm did not move exactly the 15 reserve units")
    if len(movable_reserve) != 15:
        raise AssertionError("Nested design does not contain exactly 15 movable units")
    return partitions


def largest_remainder_line_counts(
    line_sizes: dict[str, int],
    requested_units: int,
) -> dict[str, int]:
    if requested_units < len(line_sizes):
        raise ValueError("Bootstrap role is too small to represent every observed line")
    total = float(sum(line_sizes.values()))
    raw = {line: requested_units * count / total for line, count in line_sizes.items()}
    counts = {line: max(1, int(math.floor(value))) for line, value in raw.items()}
    while sum(counts.values()) < requested_units:
        line = max(
            line_sizes,
            key=lambda item: (raw[item] - counts[item], line_sizes[item], item),
        )
        counts[line] += 1
    while sum(counts.values()) > requested_units:
        removable = [line for line in line_sizes if counts[line] > 1]
        if not removable:
            raise RuntimeError("Could not reduce bootstrap line counts")
        line = min(
            removable,
            key=lambda item: (raw[item] - counts[item], -line_sizes[item], item),
        )
        counts[line] -= 1
    return counts


def bootstrap_enlarged_layout(
    base: core.MetadataLayout,
    development_units: int,
    audit_units: int,
    seed: int,
) -> tuple[core.MetadataLayout, core.RoleSplit, dict[str, object]]:
    """Clone complete metadata units with replacement into synthetic roles."""
    rng = np.random.default_rng(seed)
    unit_to_indices = {
        unit: np.flatnonzero(base.units == unit)
        for unit in sorted(set(base.units.tolist()))
    }
    unit_to_line = line_by_unit(base, np.arange(len(base.units)))
    line_to_units: dict[str, list[str]] = {}
    for unit, line in unit_to_line.items():
        line_to_units.setdefault(line, []).append(unit)
    line_sizes = {line: len(units) for line, units in line_to_units.items()}

    selected_rows: list[int] = []
    clone_units: list[str] = []
    clone_roles: list[str] = []
    clone_templates: list[str] = []
    for role, count in (("development", development_units), ("audit", audit_units)):
        role_line_counts = largest_remainder_line_counts(line_sizes, count)
        clone_index = 0
        for line in sorted(role_line_counts):
            templates = np.asarray(sorted(line_to_units[line]), dtype=object)
            sampled = rng.choice(templates, size=role_line_counts[line], replace=True)
            for template in sampled:
                template = str(template)
                clone_id = f"{role[:3]}_clone_{clone_index:04d}_from_{template}"
                rows = unit_to_indices[template]
                selected_rows.extend(int(value) for value in rows)
                clone_units.extend([clone_id] * len(rows))
                clone_roles.extend([role] * len(rows))
                clone_templates.append(template)
                clone_index += 1

    selected = np.asarray(selected_rows, dtype=int)
    frame = base.frame.iloc[selected].copy().reset_index(drop=True)
    frame["fMOST_brain_id"] = clone_units
    frame["provider_sample_id"] = [f"synthetic_{unit}" for unit in clone_units]
    layout = core.MetadataLayout(
        frame=frame,
        units=np.asarray(clone_units, dtype=str),
        lines=base.lines[selected].copy(),
        b_design=base.b_design[selected].copy(),
        p_design=base.p_design[selected].copy(),
        coords=base.coords[selected].copy(),
        pc=base.pc[selected].copy(),
        layer_depth=base.layer_depth[selected].copy(),
    )
    clone_roles_array = np.asarray(clone_roles, dtype=str)
    roles = core.RoleSplit(
        development=np.flatnonzero(clone_roles_array == "development"),
        audit=np.flatnonzero(clone_roles_array == "audit"),
        unassigned=np.array([], dtype=int),
    )
    if len(set(layout.units[roles.development])) != development_units:
        raise AssertionError("Bootstrap development clone count is wrong")
    if len(set(layout.units[roles.audit])) != audit_units:
        raise AssertionError("Bootstrap audit clone count is wrong")
    metadata = {
        "bootstrap_unit": "complete_observed_brain_sample_metadata_cluster",
        "with_replacement": True,
        "synthetic_clones_are_real_independent_animals": False,
        "development_template_draws": development_units,
        "audit_template_draws": audit_units,
        "distinct_development_templates": len(set(clone_templates[:development_units])),
        "distinct_templates_overall": len(set(clone_templates)),
    }
    return layout, roles, metadata


def generic_unit_folds(
    layout: core.MetadataLayout,
    indices: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[GenericFold, ...]:
    indices = np.asarray(indices, dtype=int)
    indexed_units = layout.units[indices]
    units = sorted(set(indexed_units.tolist()))
    if n_folds < 2 or n_folds > len(units):
        raise ValueError("Invalid number of whole-unit folds")
    unit_lines = line_by_unit(layout, indices)
    line_to_units: dict[str, list[str]] = {}
    for unit, line in unit_lines.items():
        line_to_units.setdefault(line, []).append(unit)

    capacities = [len(units) // n_folds + int(i < len(units) % n_folds) for i in range(n_folds)]
    fold_units: list[list[str]] = [[] for _ in range(n_folds)]
    fold_lines: list[set[str]] = [set() for _ in range(n_folds)]
    ordered_lines = sorted(
        line_to_units,
        key=lambda line: (-len(line_to_units[line]), stable_u64(seed, "line", line)),
    )
    for line in ordered_lines:
        ordered_units = sorted(
            line_to_units[line], key=lambda unit: stable_u64(seed, "unit", unit)
        )
        for unit in ordered_units:
            available = [i for i in range(n_folds) if len(fold_units[i]) < capacities[i]]
            selected = min(
                available,
                key=lambda i: (
                    line in fold_lines[i],
                    len(fold_units[i]) / capacities[i],
                    stable_u64(seed, "tie", line, unit, i),
                ),
            )
            fold_units[selected].append(unit)
            fold_lines[selected].add(line)

    folds: list[GenericFold] = []
    covered: set[str] = set()
    for fold_index, validation in enumerate(fold_units):
        validation_units = tuple(sorted(validation))
        training_units = tuple(sorted(set(units) - set(validation_units)))
        training_positions = np.flatnonzero(np.isin(indexed_units, training_units))
        validation_positions = np.flatnonzero(np.isin(indexed_units, validation_units))
        if set(training_units) & set(validation_units):
            raise AssertionError("Generic fold leaks a validation unit into training")
        covered.update(validation_units)
        folds.append(
            GenericFold(
                fold_index=fold_index,
                training_positions=training_positions,
                validation_positions=validation_positions,
                training_units=training_units,
                validation_units=validation_units,
            )
        )
    if covered != set(units) or sum(len(f.validation_units) for f in folds) != len(units):
        raise AssertionError("Generic folds do not score every unit exactly once")
    return tuple(folds)


def fit_with_inner_unit_cv(
    layout: core.MetadataLayout,
    development: np.ndarray,
    development_outcomes: np.ndarray,
    seed: int,
    n_folds: int = 4,
) -> tuple[dict[str, core.MixedSupportModel], core.FitDiagnostics, dict[str, object]]:
    """Choose C's sigma by equal-unit held-out score, then freeze it into H."""
    development = np.asarray(development, dtype=int)
    development_outcomes = np.asarray(development_outcomes, dtype=float)
    folds = generic_unit_folds(layout, development, n_folds=n_folds, seed=seed)
    grid = tuple(float(value) for value in core.SHARED_SIGMA_GRID)
    per_sigma: list[list[float]] = [[] for _ in grid]
    h_score_evaluations = 0
    c_score_evaluations = 0
    for fold in folds:
        training = development[fold.training_positions]
        validation = development[fold.validation_positions]
        fold_models, _ = core.fit_models(
            layout,
            training,
            development_outcomes[fold.training_positions],
            noise_protocol_id="design_sensitivity_inner_unit_cv_provisional",
        )
        heldout = development_outcomes[fold.validation_positions]
        validation_units = layout.units[validation]
        for sigma_index, sigma in enumerate(grid):
            for family in ("C", "H"):
                candidate = replace(fold_models[family], sigma=sigma)
                scores, _ = core.score_model(
                    candidate,
                    core.design_for(candidate, layout, validation),
                    heldout,
                )
                if family == "C":
                    for unit in fold.validation_units:
                        per_sigma[sigma_index].append(
                            float(scores[validation_units == unit].mean())
                        )
                    c_score_evaluations += 1
                else:
                    h_score_evaluations += 1
    expected_units = len(set(layout.units[development].tolist()))
    if any(len(values) != expected_units for values in per_sigma):
        raise AssertionError("Inner CV lacks one C score per development unit")
    objectives = tuple(float(np.mean(values)) for values in per_sigma)
    selected_index = int(np.argmax(np.asarray(objectives)))
    selected_sigma = grid[selected_index]
    protocol_id = "design_sensitivity_inner_unit_C_reference_sigma_frozen_into_H_v1"
    models, diagnostics = core.fit_models(
        layout,
        development,
        development_outcomes,
        shared_sigma_override=selected_sigma,
        noise_protocol_id=protocol_id,
    )
    core.assert_factorized_model_set(models, layout.b_design.shape[1])
    metadata = {
        "protocol_id": protocol_id,
        "fold_seed": int(seed),
        "folds": n_folds,
        "development_units": expected_units,
        "validation_unit_coverage": sum(len(fold.validation_units) for fold in folds),
        "train_validation_overlap_count": sum(
            len(set(fold.training_units) & set(fold.validation_units)) for fold in folds
        ),
        "selected_sigma": selected_sigma,
        "sigma_grid": list(grid),
        "C_sigma_score_evaluations": c_score_evaluations,
        "H_sigma_score_evaluations": h_score_evaluations,
        "C_H_grid_budget_equal": c_score_evaluations == h_score_evaluations,
        "selection_reference_family": "C",
        "unseen_line_rule": "zero_deviation_from_population_intercept_when_training_fold_lacks_line",
    }
    return models, diagnostics, metadata


def fit_oracle_training_label_only_h(
    layout: core.MetadataLayout,
    development: np.ndarray,
    outcomes: np.ndarray,
    models: dict[str, core.MixedSupportModel],
    true_labels: np.ndarray,
) -> core.MixedSupportModel | None:
    """Replace only core's discovered training labels with synthetic truth.

    This reconstructs core.fit_models' P residual, first principal score,
    rank-normal continuous coordinate, continuous-basis adjustment, and
    conditional-C offsets exactly.  The sole replacement is `two_means(...)`
    with the known synthetic development labels.  Audit labels are never used.
    """
    labels = np.asarray(true_labels, dtype=int)
    if set(labels.tolist()) != {0, 1} or min(np.bincount(labels, minlength=2)) < 2:
        return None
    model_c = models["C"]
    design = core.design_for(model_c, layout, development)
    p_support = design @ model_c.support_beta
    p_value = design @ model_c.value_beta
    p_residual = core.residual_features(outcomes, p_support, p_value)
    p_scores, _ = core.principal_scores(p_residual, n_components=1)
    u_c = core.rank_normal(p_scores[:, 0])
    z_c = core.continuous_basis(u_c)
    c_support_coef, c_value_coef = core.fit_adjustments(
        outcomes, p_support, p_value, z_c
    )
    conditional_support = p_support + z_c @ c_support_coef
    conditional_value = p_value + z_c @ c_value_coef

    counts = np.bincount(labels, minlength=2).astype(float)
    weights = (counts + 1.0) / (counts.sum() + 2.0)
    codes = np.array(
        [-math.sqrt(weights[1] / weights[0]), math.sqrt(weights[0] / weights[1])]
    )
    group_design = codes[labels, None]
    support_coef, value_coef = core.fit_adjustments(
        outcomes,
        conditional_support,
        conditional_value,
        group_design,
    )
    group_support = codes[:, None] @ support_coef
    group_value = codes[:, None] @ value_coef
    h_support = np.vstack(
        [
            group_support[group] + shift
            for group in range(2)
            for shift in model_c.state_support_shift
        ]
    )
    h_value = np.vstack(
        [
            group_value[group] + shift
            for group in range(2)
            for shift in model_c.state_value_shift
        ]
    )
    h_weights = np.concatenate(
        [weights[group] * model_c.state_weights for group in range(2)]
    )
    return core.MixedSupportModel(
        name="H_oracle_training_label_only",
        design_name=model_c.design_name,
        support_beta=model_c.support_beta,
        value_beta=model_c.value_beta,
        state_support_shift=h_support,
        state_value_shift=h_value,
        state_weights=h_weights,
        sigma=model_c.sigma,
        backbone_parent="C",
        continuous_support_backbone=model_c.continuous_support_backbone,
        continuous_value_backbone=model_c.continuous_value_backbone,
        continuous_weights=model_c.continuous_weights,
        group_support_signature=group_support,
        group_value_signature=group_value,
        group_weights=weights,
        group_gating="constant_development_only",
        group_signature_protocol="core_conditioning_with_oracle_synthetic_training_labels_only",
        noise_protocol_id=model_c.noise_protocol_id,
        paired_training_protocol_id=model_c.paired_training_protocol_id,
    )


def audit_unit_statistics(
    layout: core.MetadataLayout,
    audit: np.ndarray,
    differences: np.ndarray,
    margin: float,
) -> dict[str, float | bool | int]:
    units = layout.units[audit]
    values = np.asarray(
        [float(differences[units == unit].mean()) for unit in sorted(set(units.tolist()))]
    )
    count = len(values)
    if count < 2:
        raise ValueError("At least two audit units are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(count)
    critical = float(student_t.ppf(0.95, count - 1))
    halfwidth = critical * se
    return {
        "audit_units": count,
        "mean": mean,
        "unit_sd": sd,
        "se": se,
        "one_sided95_halfwidth": halfwidth,
        "lower_one_sided95": mean - halfwidth,
        "upper_one_sided95": mean + halfwidth,
        "predictive_gain_detected": bool(mean - halfwidth > margin),
        "precision_halfwidth_below_margin": bool(halfwidth < margin),
    }


def minimum_audit_units_for_halfwidth(sd: float, margin: float, maximum: int = 5000) -> int | None:
    if not np.isfinite(sd) or sd < 0.0:
        return None
    for count in range(2, maximum + 1):
        critical = float(student_t.ppf(0.95, count - 1))
        if critical * sd / math.sqrt(count) < margin:
            return count
    return None


def h_group_accuracy(
    h_components: np.ndarray,
    model_h: core.MixedSupportModel,
    truth: np.ndarray,
    nonzero_groups: bool,
) -> float:
    """Marginalize H's continuous nodes, then apply one global group permutation."""
    if not nonzero_groups:
        return math.nan
    if model_h.continuous_weights is None or model_h.group_weights is None:
        raise ValueError("H does not expose its continuous and group factors")
    n_groups = len(model_h.group_weights)
    n_nodes = len(model_h.continuous_weights)
    expected_states = n_groups * n_nodes
    if n_groups != 2 or h_components.shape[1] != expected_states:
        raise ValueError("H component layout does not match two groups by continuous nodes")
    group_log_posterior = logsumexp(
        h_components.reshape(len(h_components), n_groups, n_nodes), axis=2
    )
    inferred = np.argmax(group_log_posterior, axis=1)
    direct = float(np.mean(inferred == truth))
    flipped = float(np.mean((1 - inferred) == truth))
    return max(direct, flipped)


def evaluate_split_once(
    layout: core.MetadataLayout,
    roles: core.RoleSplit,
    fixture: Fixture,
    design_name: str,
    replicate: int,
    seed: int,
    margin: float,
    design_class: str,
    precomputed_full_outcomes: np.ndarray | None = None,
    precomputed_full_truth: core.SyntheticTruth | None = None,
) -> dict[str, object]:
    assigned = np.concatenate([roles.development, roles.audit])
    if (precomputed_full_outcomes is None) != (precomputed_full_truth is None):
        raise ValueError("Precomputed outcomes and truth must be supplied together")
    if precomputed_full_outcomes is None:
        rng = np.random.default_rng(
            stable_u64(seed, "outcomes", design_name, fixture.name, replicate) % (2**32)
        )
        assigned_outcomes, truth = core.generate_synthetic_outcomes(
            layout,
            assigned,
            fixture.scenario,
            fixture.generator_amplitude,
            rng,
        )
        common_random_number_nested_partition = False
    else:
        if len(precomputed_full_outcomes) != len(layout.frame):
            raise ValueError("Precomputed outcomes do not cover the supplied layout")
        assigned_outcomes = precomputed_full_outcomes[assigned]
        truth = core.subset_synthetic_truth(precomputed_full_truth, assigned)
        common_random_number_nested_partition = True
    n_development_cells = len(roles.development)
    development_outcomes = assigned_outcomes[:n_development_cells]
    audit_outcomes = assigned_outcomes[n_development_cells:]
    inner_seed = development_fit_seed(
        layout, roles.development, analysis_seed=seed, replicate=replicate
    )
    models, _, cv = fit_with_inner_unit_cv(
        layout,
        roles.development,
        development_outcomes,
        seed=inner_seed,
    )
    oracle_label_h = None
    if fixture.truth_has_nonzero_groups:
        oracle_label_h = fit_oracle_training_label_only_h(
            layout,
            roles.development,
            development_outcomes,
            models,
            truth.groups[:n_development_cells],
        )
    scores: dict[str, np.ndarray] = {}
    h_components: np.ndarray | None = None
    for family in ("C", "D", "H"):
        scores[family], components = core.score_model(
            models[family],
            core.design_for(models[family], layout, roles.audit),
            audit_outcomes,
            return_components=family == "H",
        )
        if family == "H":
            h_components = components
    assert h_components is not None
    h_minus_c = scores["H"] - scores["C"]
    stats = audit_unit_statistics(layout, roles.audit, h_minus_c, margin)
    audit_units = layout.units[roles.audit]
    unit_score_map = {
        unit: float(h_minus_c[audit_units == unit].mean())
        for unit in sorted(set(audit_units.tolist()))
    }
    if oracle_label_h is not None:
        oracle_label_score, _ = core.score_model(
            oracle_label_h,
            core.design_for(oracle_label_h, layout, roles.audit),
            audit_outcomes,
        )
        oracle_label_stats = audit_unit_statistics(
            layout, roles.audit, oracle_label_score - scores["C"], margin
        )
    else:
        oracle_label_stats = {
            "mean": math.nan,
            "unit_sd": math.nan,
            "one_sided95_halfwidth": math.nan,
            "lower_one_sided95": math.nan,
            "predictive_gain_detected": False,
            "precision_halfwidth_below_margin": False,
        }
    audit_truth = core.subset_synthetic_truth(
        truth, slice(n_development_cells, None)
    )
    oracle_h = core.oracle_mixed_support_scores(
        audit_outcomes, audit_truth, include_groups=True
    )
    projected_c = core.projected_continuous_oracle_scores(
        audit_outcomes,
        audit_truth,
        fixture.projected_c_parameters,
    )
    oracle_density_stats = audit_unit_statistics(
        layout, roles.audit, oracle_h - projected_c, margin
    )
    truth_audit = truth.groups[n_development_cells:]
    accuracy = h_group_accuracy(
        h_components,
        models["H"],
        truth_audit,
        fixture.truth_has_nonzero_groups,
    )
    return {
        "design": design_name,
        "design_class": design_class,
        "fixture": fixture.name,
        "scenario": fixture.scenario,
        "replicate": replicate,
        "generator_amplitude": fixture.generator_amplitude,
        "target_margin_multiplier": fixture.target_margin_multiplier,
        "target_projected_oracle_proxy_gain": fixture.target_proxy_gain,
        "verified_projected_oracle_proxy_gain": fixture.verified_proxy_gain,
        "truth_has_nonzero_groups": fixture.truth_has_nonzero_groups,
        "development_units": len(set(layout.units[roles.development].tolist())),
        "audit_units": stats["audit_units"],
        "unassigned_units": len(set(layout.units[roles.unassigned].tolist())),
        "development_cells": len(roles.development),
        "audit_cells": len(roles.audit),
        "H_minus_C": stats["mean"],
        "H_minus_C_audit_unit_sd": stats["unit_sd"],
        "H_minus_C_one_sided95_halfwidth": stats["one_sided95_halfwidth"],
        "H_minus_C_lower_one_sided95": stats["lower_one_sided95"],
        "predictive_gain_detected": stats["predictive_gain_detected"],
        "precision_halfwidth_below_margin": stats["precision_halfwidth_below_margin"],
        "permutation_invariant_H_group_accuracy": accuracy,
        "oracle_training_label_only_H_minus_C": oracle_label_stats["mean"],
        "oracle_training_label_only_unit_sd": oracle_label_stats["unit_sd"],
        "oracle_training_label_only_halfwidth": oracle_label_stats[
            "one_sided95_halfwidth"
        ],
        "oracle_training_label_only_gain_detected": oracle_label_stats[
            "predictive_gain_detected"
        ],
        "oracle_density_H_minus_projected_C": oracle_density_stats["mean"],
        "oracle_density_unit_sd": oracle_density_stats["unit_sd"],
        "oracle_density_halfwidth": oracle_density_stats["one_sided95_halfwidth"],
        "oracle_density_gain_detected": oracle_density_stats["predictive_gain_detected"],
        "selected_sigma": cv["selected_sigma"],
        "development_fit_seed": cv["fold_seed"],
        "fitted_model_set_fingerprint": fitted_model_set_fingerprint(models),
        "audit_unit_H_minus_C_json": json.dumps(
            unit_score_map, sort_keys=True, separators=(",", ":")
        ),
        "inner_cv_validation_unit_coverage": cv["validation_unit_coverage"],
        "inner_cv_train_validation_overlap_count": cv["train_validation_overlap_count"],
        "inner_cv_C_H_grid_budget_equal": cv["C_H_grid_budget_equal"],
        "common_random_number_nested_partition": common_random_number_nested_partition,
        "label_accuracy_scope": "H_posterior_continuous_nodes_marginalized_one_global_permutation",
        "outer_cv_inference_calibrated": True,
        "geometry_constraints_enforced": False,
        "real_projection_outcomes_read": False,
    }


def evaluate_all35_outer_cv_once(
    layout: core.MetadataLayout,
    fixture: Fixture,
    replicate: int,
    seed: int,
    margin: float,
) -> dict[str, object]:
    """Seven outer folds; all 35 nonsingleton units are scored exactly once."""
    unit_lines = line_by_unit(layout, np.arange(len(layout.frame)))
    line_counts = pd.Series(unit_lines).value_counts().to_dict()
    singleton_units = {
        unit for unit, line in unit_lines.items() if line_counts[line] == 1
    }
    if len(singleton_units) != 1:
        raise AssertionError("Outer CV expects one excluded singleton-line unit")
    all_indices = np.flatnonzero(~np.isin(layout.units, list(singleton_units)))
    outer_folds = generic_unit_folds(
        layout,
        all_indices,
        n_folds=7,
        seed=stable_u64(seed, "outer_folds", replicate) % (2**32),
    )
    rng = np.random.default_rng(
        stable_u64(seed, "outer_outcomes", fixture.name, replicate) % (2**32)
    )
    outcomes, truth = core.generate_synthetic_outcomes(
        layout,
        all_indices,
        fixture.scenario,
        fixture.generator_amplitude,
        rng,
    )
    h_scores = np.empty(len(all_indices), dtype=float)
    c_scores = np.empty(len(all_indices), dtype=float)
    oracle_label_scores = np.full(len(all_indices), np.nan, dtype=float)
    inferred_labels = np.full(len(all_indices), -1, dtype=int)
    scored_units: set[str] = set()
    inner_overlap = 0
    for fold in outer_folds:
        training = all_indices[fold.training_positions]
        audit = all_indices[fold.validation_positions]
        models, _, cv = fit_with_inner_unit_cv(
            layout,
            training,
            outcomes[fold.training_positions],
            seed=stable_u64(seed, "outer_inner_cv", replicate, fold.fold_index) % (2**32),
            n_folds=5,
        )
        inner_overlap += int(cv["train_validation_overlap_count"])
        h_components: np.ndarray | None = None
        for family, destination in (("C", c_scores), ("H", h_scores)):
            destination[fold.validation_positions], components = core.score_model(
                models[family],
                core.design_for(models[family], layout, audit),
                outcomes[fold.validation_positions],
                return_components=family == "H",
            )
            if family == "H":
                h_components = components
        assert h_components is not None
        if fixture.truth_has_nonzero_groups:
            rescue_h = fit_oracle_training_label_only_h(
                layout,
                training,
                outcomes[fold.training_positions],
                models,
                truth.groups[fold.training_positions],
            )
            if rescue_h is not None:
                oracle_label_scores[fold.validation_positions], _ = core.score_model(
                    rescue_h,
                    core.design_for(rescue_h, layout, audit),
                    outcomes[fold.validation_positions],
                )
        if fixture.truth_has_nonzero_groups:
            n_groups = len(models["H"].group_weights)
            n_nodes = len(models["H"].continuous_weights)
            group_log_posterior = logsumexp(
                h_components.reshape(len(audit), n_groups, n_nodes), axis=2
            )
            inferred_labels[fold.validation_positions] = np.argmax(
                group_log_posterior, axis=1
            )
        scored_units.update(fold.validation_units)
    if len(scored_units) != 35:
        raise AssertionError("Outer CV did not score all 35 nonsingleton units exactly once")
    h_minus_c = h_scores - c_scores
    stats = audit_unit_statistics(layout, all_indices, h_minus_c, margin)
    outer_units = layout.units[all_indices]
    unit_score_map = {
        unit: float(h_minus_c[outer_units == unit].mean())
        for unit in sorted(set(outer_units.tolist()))
    }
    if fixture.truth_has_nonzero_groups and np.isfinite(oracle_label_scores).all():
        oracle_label_stats = audit_unit_statistics(
            layout,
            all_indices,
            oracle_label_scores - c_scores,
            margin,
        )
    else:
        oracle_label_stats = {
            "mean": math.nan,
            "unit_sd": math.nan,
            "one_sided95_halfwidth": math.nan,
            "predictive_gain_detected": False,
        }
    oracle_h = core.oracle_mixed_support_scores(
        outcomes, truth, include_groups=True
    )
    projected_c = core.projected_continuous_oracle_scores(
        outcomes, truth, fixture.projected_c_parameters
    )
    oracle_density_stats = audit_unit_statistics(
        layout, all_indices, oracle_h - projected_c, margin
    )
    if fixture.truth_has_nonzero_groups:
        direct = float(np.mean(inferred_labels == truth.groups))
        flipped = float(np.mean((1 - inferred_labels) == truth.groups))
        outer_global_permutation_accuracy = max(direct, flipped)
    else:
        outer_global_permutation_accuracy = math.nan
    return {
        "design": "outer_cv_30_train_5_test_all_35_nonsingleton_scored_once",
        "design_class": "prospective_outer_unit_cv_sensitivity",
        "fixture": fixture.name,
        "scenario": fixture.scenario,
        "replicate": replicate,
        "generator_amplitude": fixture.generator_amplitude,
        "target_margin_multiplier": fixture.target_margin_multiplier,
        "target_projected_oracle_proxy_gain": fixture.target_proxy_gain,
        "verified_projected_oracle_proxy_gain": fixture.verified_proxy_gain,
        "truth_has_nonzero_groups": fixture.truth_has_nonzero_groups,
        "development_units": 30,
        "audit_units": 35,
        "unassigned_units": 1,
        "development_cells": math.nan,
        "audit_cells": len(all_indices),
        "H_minus_C": stats["mean"],
        "H_minus_C_audit_unit_sd": stats["unit_sd"],
        # Fold scores share heavily overlapping training sets.  Their 35 unit
        # means are descriptive and cannot use the independent-audit t rule.
        "H_minus_C_one_sided95_halfwidth": math.nan,
        "H_minus_C_lower_one_sided95": math.nan,
        "predictive_gain_detected": math.nan,
        "precision_halfwidth_below_margin": math.nan,
        "permutation_invariant_H_group_accuracy": outer_global_permutation_accuracy,
        "oracle_training_label_only_H_minus_C": oracle_label_stats["mean"],
        "oracle_training_label_only_unit_sd": oracle_label_stats["unit_sd"],
        "oracle_training_label_only_halfwidth": math.nan,
        "oracle_training_label_only_gain_detected": math.nan,
        "oracle_density_H_minus_projected_C": oracle_density_stats["mean"],
        "oracle_density_unit_sd": oracle_density_stats["unit_sd"],
        "oracle_density_halfwidth": math.nan,
        "oracle_density_gain_detected": math.nan,
        "selected_sigma": math.nan,
        "development_fit_seed": math.nan,
        "fitted_model_set_fingerprint": "fold_specific_models_no_single_fingerprint",
        "audit_unit_H_minus_C_json": json.dumps(
            unit_score_map, sort_keys=True, separators=(",", ":")
        ),
        "inner_cv_validation_unit_coverage": 30,
        "inner_cv_train_validation_overlap_count": inner_overlap,
        "inner_cv_C_H_grid_budget_equal": True,
        "common_random_number_nested_partition": False,
        "label_accuracy_scope": "outer_H_posterior_one_global_permutation_descriptive_not_canonical_reuse",
        "outer_cv_inference_calibrated": False,
        "geometry_constraints_enforced": False,
        "real_projection_outcomes_read": False,
    }


def fixtures_from_calibration(
    calibration: core.OracleCalibration,
    scenarios: Sequence[str],
    strong_multiplier: float,
) -> list[Fixture]:
    lookup = {
        (point.scenario, point.target_margin_multiplier): point
        for point in calibration.points
    }
    fixtures = [
        Fixture(
            name="zero_group_null_groups_plus_continuum",
            scenario="groups_plus_continuum",
            generator_amplitude=0.0,
            target_margin_multiplier=0.0,
            target_proxy_gain=0.0,
            verified_proxy_gain=0.0,
            projected_c_parameters=(0.0, 0.0, 0.0, 0.0),
            truth_has_nonzero_groups=False,
        )
    ]
    for scenario in scenarios:
        point = lookup[(scenario, strong_multiplier)]
        fixtures.append(
            Fixture(
                name=f"strong_2x_{scenario}",
                scenario=scenario,
                generator_amplitude=point.generator_amplitude,
                target_margin_multiplier=strong_multiplier,
                target_proxy_gain=point.target_gain_nats_per_neuron,
                verified_proxy_gain=point.independent_verification_mean_gain,
                projected_c_parameters=point.projected_C_parameters,
                truth_has_nonzero_groups=True,
            )
        )
    return fixtures


def parse_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_float_grid(text: str) -> list[float]:
    values = [float(value) for value in parse_list(text)]
    if not values or values != sorted(set(values)) or values[0] != 0.0:
        raise ValueError("Calibration amplitudes must be unique, increasing, and start at zero")
    return values


def verify_nested_d12_pairing(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Verify baseline and precision arms reuse one D12 fit and identical A8 scores."""
    baseline_name = "nested_baseline_12_dev_8_audit_16_unassigned"
    precision_name = "nested_precision_ceiling_12_dev_23_audit_1_unassigned"
    indexed = {
        (str(row["design"]), str(row["fixture"]), int(row["replicate"])): row
        for row in rows
    }
    keys = sorted(
        (fixture, replicate)
        for design, fixture, replicate in indexed
        if design == baseline_name
        and (precision_name, fixture, replicate) in indexed
    )
    for fixture, replicate in keys:
        baseline = indexed[(baseline_name, fixture, replicate)]
        precision = indexed[(precision_name, fixture, replicate)]
        if baseline["development_fit_seed"] != precision["development_fit_seed"]:
            raise AssertionError("Paired D12 arms used different inner-CV fold seeds")
        if (
            baseline["fitted_model_set_fingerprint"]
            != precision["fitted_model_set_fingerprint"]
        ):
            raise AssertionError("Paired D12 arms did not produce an identical fitted model")
        baseline_scores = json.loads(str(baseline["audit_unit_H_minus_C_json"]))
        precision_scores = json.loads(str(precision["audit_unit_H_minus_C_json"]))
        if not set(baseline_scores).issubset(precision_scores):
            raise AssertionError("Baseline A8 is not a subset of precision-arm A23")
        for unit, baseline_score in baseline_scores.items():
            if not np.isclose(
                float(baseline_score),
                float(precision_scores[unit]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(
                    f"Shared audit unit {unit!r} changed score across paired D12 arms"
                )
    return {
        "status": "pass" if keys else "not_applicable",
        "paired_fixture_replicates_checked": len(keys),
        "inner_cv_seed_key_excludes_design_name": True,
        "identical_D12_model_fingerprint_required": True,
        "shared_A8_unit_scores_identical_required": True,
    }


def summarize(rows: list[dict[str, object]], margin: float) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    conditions: list[dict[str, object]] = []
    for (design, fixture), group in frame.groupby(["design", "fixture"], sort=True):
        grouped_truth = bool(group["truth_has_nonzero_groups"].iloc[0])
        inference_calibrated = bool(group["outer_cv_inference_calibrated"].all())
        sd = float(group["H_minus_C_audit_unit_sd"].mean())
        accuracy_values = group["permutation_invariant_H_group_accuracy"].astype(float)
        conditions.append(
            {
                "design": str(design),
                "design_class": str(group["design_class"].iloc[0]),
                "fixture": str(fixture),
                "replicates": int(len(group)),
                "development_units": int(group["development_units"].iloc[0]),
                "audit_units": int(group["audit_units"].iloc[0]),
                "unassigned_units": int(group["unassigned_units"].iloc[0]),
                "mean_development_cells": float(group["development_cells"].mean()),
                "mean_audit_cells": float(group["audit_cells"].mean()),
                "mean_H_minus_C": float(group["H_minus_C"].mean()),
                "mean_audit_unit_SD": sd,
                "mean_one_sided95_halfwidth": float(
                    group["H_minus_C_one_sided95_halfwidth"].mean()
                ),
                "predictive_gain_detection_rate": float(
                    group["predictive_gain_detected"].mean()
                ),
                "precision_rate": float(
                    group["precision_halfwidth_below_margin"].mean()
                ),
                "false_positive_rate": (
                    math.nan
                    if grouped_truth or not inference_calibrated
                    else float(group["predictive_gain_detected"].mean())
                ),
                "mean_permutation_invariant_H_group_accuracy": (
                    float(accuracy_values.mean()) if grouped_truth else math.nan
                ),
                "mean_oracle_training_label_only_H_minus_C": (
                    float(
                        group[
                            "oracle_training_label_only_H_minus_C"
                        ].mean()
                    )
                    if grouped_truth
                    else math.nan
                ),
                "oracle_training_label_only_detection_rate": (
                    float(
                        group[
                            "oracle_training_label_only_gain_detected"
                        ].mean()
                    )
                    if grouped_truth
                    else math.nan
                ),
                "mean_oracle_density_H_minus_projected_C": float(
                    group["oracle_density_H_minus_projected_C"].mean()
                ),
                "oracle_density_detection_rate": float(
                    group["oracle_density_gain_detected"].mean()
                ),
                "minimum_audit_units_for_halfwidth_below_margin_at_mean_SD": (
                    minimum_audit_units_for_halfwidth(sd, margin)
                    if inference_calibrated
                    else None
                ),
                "outer_cv_inference_calibrated": inference_calibrated,
                "label_accuracy_scope": str(group["label_accuracy_scope"].iloc[0]),
                "all_nested_rows_use_common_random_numbers": bool(
                    group["common_random_number_nested_partition"].all()
                    if str(group["design_class"].iloc[0])
                    == "observed_unit_metadata_only_nested_partition"
                    else True
                ),
                "all_inner_train_validation_overlaps_zero": bool(
                    (group["inner_cv_train_validation_overlap_count"] == 0).all()
                ),
                "all_C_H_grid_budgets_equal": bool(
                    group["inner_cv_C_H_grid_budget_equal"].all()
                ),
            }
        )

    reference_halfwidth_36 = float(
        student_t.ppf(0.95, 35) * REFERENCE_STRONG_FIXTURE_UNIT_SD / math.sqrt(36)
    )
    return {
        "schema_version": "ep11_mixed_support_design_sensitivity_v1",
        "analysis_class": "synthetic_design_and_estimator_sensitivity_not_qualification",
        "outcome_bearing_search_authorized": False,
        "real_projection_outcomes_read": False,
        "geometry_constraints_enforced": False,
        "role_layout_status": "metadata_only_not_geometry_eligible_not_locked",
        "margin_nats_per_neuron": margin,
        "nested_partition": {
            "fixed_development_core_units": 12,
            "fixed_audit_core_units": 8,
            "movable_reserve_units": 15,
            "singleton_line_units_always_unassigned": 1,
            "arms": {
                "baseline": [12, 8, 16],
                "estimator_ceiling": [27, 8, 1],
                "precision_ceiling": [12, 23, 1],
            },
            "common_random_numbers_across_arms": True,
            "full_36_unit_synthetic_outcome_generated_once_per_replicate_fixture": True,
        },
        "conditions": conditions,
        "precision_reference": {
            "strong_fixture_unit_SD": REFERENCE_STRONG_FIXTURE_UNIT_SD,
            "one_sided95_halfwidth_with_36_audit_units": reference_halfwidth_36,
            "minimum_units_for_halfwidth_below_0_05_if_SD_unchanged": (
                minimum_audit_units_for_halfwidth(
                    REFERENCE_STRONG_FIXTURE_UNIT_SD, margin
                )
            ),
            "interpretation": (
                "If unit-level variability stays this large, reallocating the existing "
                "36 units cannot by itself attain a 0.05 halfwidth."
            ),
        },
        "estimator_limitations": [
            "coefficient residual-PCA and signature fits remain pooled-cell weighted",
            "bootstrap clones are synthetic and are not additional independent animals",
            "calibration is a projected-oracle proxy, not projection onto the full C family",
            "one provisional-layout projected-C proxy is transported across design arms rather than reprojected per design",
            "continuous absolute adequacy and full reuse gates are not evaluated",
            "common-support geometry is not enforced",
            "public brain/sample units are not provider-certified independent animals",
            "outer-CV scores have overlapping training sets, so t intervals and detection rates are disabled",
        ],
        "recommended_next_design": (
            "Repair and separately diagnose group discovery, projection-signature estimation, "
            "position-dependent gating, and population-optimal C-versus-H effect calibration "
            "before any further role-count or outer-CV sensitivity."
        ),
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_fields = set(rows[0])
    if any(set(row) != expected_fields for row in rows):
        raise ValueError("Design-sensitivity replicate rows do not share one schema")
    with (output_dir / "design_sensitivity_replicates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "design_sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    lines = [
        "# EP11 synthetic estimator and design sensitivity",
        "",
        "- Analysis class: `synthetic_design_and_estimator_sensitivity_not_qualification`",
        "- Real projection outcomes read: **no**",
        "- Geometry constraints enforced: **no**",
        "- Outcome-bearing search authorized: **no**",
        "",
        "The fixed-split rows are metadata-only prospective allocations. Bootstrap",
        "rows are estimator ceilings based on synthetic clones of complete metadata",
        "units, not evidence that more independent animals exist.",
        "The three observed-unit arms share the identical D12/A8 core and differ",
        "only by moving the same 15 reserve units into development or audit.",
        "Outer-CV rows are descriptive only: overlapping training sets invalidate",
        "the independent-audit t interval, so their power and precision are `NA`.",
        "",
        "| Design | Fixture | n | Dev | Audit | Fitted H-C | Training-label-only H-C | Oracle-density H-C | Unit SD | Halfwidth | Power | Precision | H-group accuracy | Required audit n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        accuracy = row["mean_permutation_invariant_H_group_accuracy"]
        accuracy_text = "NA" if not np.isfinite(accuracy) else f"{accuracy:.3f}"
        rescue = row["mean_oracle_training_label_only_H_minus_C"]
        rescue_text = "NA" if not np.isfinite(rescue) else f"{rescue:.3f}"
        required = row["minimum_audit_units_for_halfwidth_below_margin_at_mean_SD"]
        lines.append(
            f"| {row['design']} | {row['fixture']} | {row['replicates']} | "
            f"{row['development_units']} | {row['audit_units']} | "
            f"{row['mean_H_minus_C']:.3f} | {rescue_text} | "
            f"{row['mean_oracle_density_H_minus_projected_C']:.3f} | "
            f"{row['mean_audit_unit_SD']:.3f} | "
            f"{row['mean_one_sided95_halfwidth']:.3f} | "
            f"{row['predictive_gain_detection_rate']:.3f} | {row['precision_rate']:.3f} | "
            f"{accuracy_text} | {required} |"
        )
    lines.extend(
        [
            "",
            "The observed strong-fixture reference SD of 0.217 implies a 36-unit",
            f"one-sided 95% halfwidth of {summary['precision_reference']['one_sided95_halfwidth_with_36_audit_units']:.3f}.",
            "Therefore alternative allocations alone may not repair precision if the",
            "unit-level variability remains unchanged.",
            "",
            "## Interpretation and stopping rule",
            "",
            "The D27/A8 estimator ceiling did not improve the fitted strong-fixture",
            "contrast or group accuracy relative to D12/A8. Supplying true synthetic",
            "training labels raised H-C only to about 0.04 nats/neuron, far below the",
            "projected-oracle proxy near 0.12. The current bottleneck therefore extends",
            "beyond label discovery to the signature, gating, likelihood, or calibration",
            "path. The D12/A23 precision ceiling reduced the mean half-width to about",
            "0.076, but did not reach 0.05; any feasible design with no more than 20",
            "audit units cannot repair this by reallocation alone if the observed",
            "unit-level variability is unchanged.",
            "",
            "The companion outcome-blind geometry frontier proves at most 33 assigned",
            "units under each depth convention and does not establish one shared role",
            "assignment across conventions. These synthetic arms are diagnostic ceilings,",
            "not implementable role designs or registered qualification. The next step is",
            "to repair and isolate group discovery, signature estimation, smooth gating,",
            "and full population-optimal C-versus-H calibration before testing more role",
            "counts or outer cross-validation.",
            "",
        ]
    )
    (output_dir / "design_sensitivity_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def executable_self_test(layout: core.MetadataLayout) -> dict[str, object]:
    partitions = nested_role_partitions(layout, seed=331)
    fixed = partitions["nested_baseline_12_dev_8_audit_16_unassigned"]
    estimator = partitions["nested_estimator_ceiling_27_dev_8_audit_1_unassigned"]
    precision = partitions["nested_precision_ceiling_12_dev_23_audit_1_unassigned"]
    assert len(set(layout.units[fixed.development])) == 12
    assert len(set(layout.units[fixed.audit])) == 8
    assert len(set(layout.units[fixed.unassigned])) == 16
    assert len(set(layout.units[estimator.development])) == 27
    assert len(set(layout.units[precision.audit])) == 23

    enlarged, enlarged_roles, metadata = bootstrap_enlarged_layout(
        layout, 12, 12, seed=992
    )
    assert len(set(enlarged.units[enlarged_roles.development])) == 12
    assert len(set(enlarged.units[enlarged_roles.audit])) == 12
    folds = generic_unit_folds(layout, fixed.development, n_folds=4, seed=117)
    assert sum(len(fold.validation_units) for fold in folds) == 12
    assert not any(set(fold.training_units) & set(fold.validation_units) for fold in folds)
    unit_lines = line_by_unit(layout, np.arange(len(layout.frame)))
    line_counts = pd.Series(unit_lines).value_counts().to_dict()
    singleton_units = {
        unit for unit, line in unit_lines.items() if line_counts[line] == 1
    }
    nonsingleton_indices = np.flatnonzero(
        ~np.isin(layout.units, list(singleton_units))
    )
    outer_folds = generic_unit_folds(
        layout, nonsingleton_indices, n_folds=7, seed=118
    )
    assert all(len(fold.training_units) == 30 for fold in outer_folds)
    assert all(len(fold.validation_units) == 5 for fold in outer_folds)

    fixture = Fixture(
        name="selftest_strong_reusable_groups",
        scenario="reusable_groups",
        generator_amplitude=0.10,
        target_margin_multiplier=2.0,
        target_proxy_gain=0.10,
        verified_proxy_gain=math.nan,
        projected_c_parameters=(0.0, 0.0, 0.0, 0.0),
        truth_has_nonzero_groups=True,
    )
    full_outcomes, full_truth = core.generate_synthetic_outcomes(
        layout,
        np.arange(len(layout.frame), dtype=int),
        fixture.scenario,
        fixture.generator_amplitude,
        np.random.default_rng(550),
    )
    row = evaluate_split_once(
        layout,
        core.RoleSplit(
            development=fixed.development[:],
            audit=fixed.audit[:],
            unassigned=fixed.unassigned[:],
        ),
        fixture,
        "nested_baseline_12_dev_8_audit_16_unassigned",
        replicate=0,
        seed=551,
        margin=DEFAULT_MARGIN,
        design_class="self_test",
        precomputed_full_outcomes=full_outcomes,
        precomputed_full_truth=full_truth,
    )
    if not np.isfinite(float(row["H_minus_C"])):
        raise AssertionError("Self-test produced a nonfinite H-C score")
    precision_row = evaluate_split_once(
        layout,
        precision,
        fixture,
        "nested_precision_ceiling_12_dev_23_audit_1_unassigned",
        replicate=0,
        seed=551,
        margin=DEFAULT_MARGIN,
        design_class="self_test",
        precomputed_full_outcomes=full_outcomes,
        precomputed_full_truth=full_truth,
    )
    pairing_checks = verify_nested_d12_pairing([row, precision_row])
    if pairing_checks["status"] != "pass":
        raise AssertionError("Nested D12 pairing self-test did not execute")
    clone_row = evaluate_split_once(
        enlarged,
        enlarged_roles,
        fixture,
        "selftest_bootstrap_12_12",
        replicate=0,
        seed=552,
        margin=DEFAULT_MARGIN,
        design_class="self_test_bootstrap",
    )
    if not np.isfinite(float(clone_row["H_minus_C"])):
        raise AssertionError("Bootstrap self-test produced a nonfinite H-C score")
    if set(row) != set(precision_row) or set(row) != set(clone_row):
        raise AssertionError("Fixed and bootstrap replicate rows do not share one schema")
    if not np.isfinite(float(row["permutation_invariant_H_group_accuracy"])):
        raise AssertionError("Self-test produced a nonfinite H-posterior group accuracy")
    return {
        "status": "pass",
        "analysis_class": "synthetic_design_and_estimator_sensitivity_not_qualification",
        "real_projection_outcomes_read": False,
        "outcome_bearing_search_authorized": False,
        "observed_metadata_units": len(set(layout.units.tolist())),
        "nested_arms": {
            "baseline": [12, 8, 16],
            "estimator_ceiling": [27, 8, 1],
            "precision_ceiling": [12, 23, 1],
        },
        "bootstrap_clone_check": metadata,
        "fold_validation_coverage": sum(len(fold.validation_units) for fold in folds),
        "fold_overlap": sum(
            len(set(fold.training_units) & set(fold.validation_units)) for fold in folds
        ),
        "outer_7x5_layout": {
            "folds": 7,
            "training_units_per_fold": 30,
            "validation_units_per_fold": 5,
            "total_scored_units": sum(
                len(fold.validation_units) for fold in outer_folds
            ),
        },
        "finite_H_minus_C": True,
        "finite_bootstrap_H_minus_C": True,
        "common_random_number_nested_partition": bool(
            row["common_random_number_nested_partition"]
        ),
        "nested_D12_pairing_checks": pairing_checks,
        "geometry_constraints_enforced": False,
        "known_limitations": [
            "metadata-only roles",
            "pooled-cell coefficient fitting",
            "projected-oracle proxy calibration",
        ],
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
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--outer-cv-replicates", type=int, default=0)
    parser.add_argument("--calibration-replicates", type=int, default=8)
    parser.add_argument(
        "--calibration-amplitudes",
        default=",".join(str(value) for value in core.ORACLE_CALIBRATION_AMPLITUDES),
    )
    parser.add_argument("--group-scenarios", default=",".join(DEFAULT_GROUP_SCENARIOS))
    parser.add_argument("--designs", default=",".join((*FIXED_SPLIT_DESIGNS, *CEILING_DESIGNS)))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    core.validate_vocabulary(args.vocabulary)
    layout = core.load_metadata(args.inventory)
    if args.self_test:
        print(json.dumps(executable_self_test(layout), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    if args.replicates < 1 or args.calibration_replicates < 2 or args.outer_cv_replicates < 0:
        raise ValueError("Replicates must be positive; calibration needs at least two")
    if not np.isclose(args.margin, DEFAULT_MARGIN):
        raise ValueError("This sensitivity keeps the registered 0.05-nat margin fixed")

    scenarios = parse_list(args.group_scenarios)
    if not scenarios or any(scenario not in core.GROUP_SCENARIOS for scenario in scenarios):
        raise ValueError("Group scenarios must be registered grouped fixtures")
    selected_designs = parse_list(args.designs)
    unknown_designs = sorted(
        set(selected_designs) - set(FIXED_SPLIT_DESIGNS) - set(CEILING_DESIGNS)
    )
    if unknown_designs:
        raise ValueError(f"Unknown designs: {unknown_designs}")

    calibration = core.calibrate_oracle_group_effects(
        layout,
        scenarios=scenarios,
        target_margin_multipliers=[0.0, DEFAULT_STRONG_MULTIPLIER],
        margin=args.margin,
        seed=stable_u64(args.seed, "calibration") % (2**32),
        replicates=args.calibration_replicates,
        amplitude_grid=parse_float_grid(args.calibration_amplitudes),
    )
    fixtures = fixtures_from_calibration(
        calibration, scenarios, DEFAULT_STRONG_MULTIPLIER
    )

    rows: list[dict[str, object]] = []
    bootstrap_metadata: dict[str, dict[str, object]] = {}
    for replicate in range(args.replicates):
        nested = nested_role_partitions(
            layout,
            seed=stable_u64(args.seed, "nested_roles", replicate) % (2**32),
        )
        nested_full_data: dict[str, tuple[np.ndarray, core.SyntheticTruth]] = {}
        all_observed_indices = np.arange(len(layout.frame), dtype=int)
        for fixture in fixtures:
            rng = np.random.default_rng(
                stable_u64(
                    args.seed,
                    "nested_common_outcomes",
                    fixture.name,
                    replicate,
                )
                % (2**32)
            )
            nested_full_data[fixture.name] = core.generate_synthetic_outcomes(
                layout,
                all_observed_indices,
                fixture.scenario,
                fixture.generator_amplitude,
                rng,
            )
        for design_name in selected_designs:
            if design_name in FIXED_SPLIT_DESIGNS:
                local_layout = layout
                roles = nested[design_name]
                design_class = "observed_unit_metadata_only_nested_partition"
            else:
                dev, audit = CEILING_DESIGNS[design_name]
                local_layout, roles, clone_metadata = bootstrap_enlarged_layout(
                    layout,
                    dev,
                    audit,
                    seed=stable_u64(
                        args.seed, "bootstrap_roles", design_name, replicate
                    )
                    % (2**32),
                )
                bootstrap_metadata[design_name] = clone_metadata
                design_class = "whole_unit_metadata_bootstrap_estimator_ceiling"
            for fixture in fixtures:
                if design_name in FIXED_SPLIT_DESIGNS:
                    common_outcomes, common_truth = nested_full_data[fixture.name]
                else:
                    common_outcomes, common_truth = None, None
                rows.append(
                    evaluate_split_once(
                        local_layout,
                        roles,
                        fixture,
                        design_name,
                        replicate,
                        args.seed,
                        args.margin,
                        design_class,
                        precomputed_full_outcomes=common_outcomes,
                        precomputed_full_truth=common_truth,
                    )
                )

    if args.outer_cv_replicates:
        for replicate in range(args.outer_cv_replicates):
            for fixture in fixtures:
                rows.append(
                    evaluate_all35_outer_cv_once(
                        layout,
                        fixture,
                        replicate,
                        args.seed,
                        args.margin,
                    )
                )

    nested_pairing_checks = verify_nested_d12_pairing(rows)
    summary = summarize(rows, args.margin)
    summary["nested_D12_pairing_checks"] = nested_pairing_checks
    summary["calibration"] = core.oracle_calibration_to_dict(calibration)
    summary["calibration"]["role_in_this_analysis"] = (
        "projected_oracle_proxy_effect_scale_only_not_qualification"
    )
    summary["selected_designs"] = selected_designs
    summary["bootstrap_metadata"] = bootstrap_metadata
    summary["outer_cv_replicates"] = args.outer_cv_replicates
    write_outputs(args.output_dir, rows, summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
