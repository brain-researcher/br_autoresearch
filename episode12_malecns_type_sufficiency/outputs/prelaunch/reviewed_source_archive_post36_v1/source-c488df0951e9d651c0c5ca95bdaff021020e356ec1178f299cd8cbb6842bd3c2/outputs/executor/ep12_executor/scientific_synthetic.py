"""Generated biological counterexamples for the actual EP12 T/U/M fitter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    from scipy.special import softmax
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("generated scientific fixtures require numpy and scipy") from exc

from .scientific_models import seed_from


Array = np.ndarray


@dataclass(frozen=True)
class SyntheticSide:
    counts: Array
    covariates: Array
    latent_coordinate: Array
    group: Array
    endpoint_loss_fraction: Array


@dataclass(frozen=True)
class SyntheticType:
    scenario: str
    left: SyntheticSide
    right: SyntheticSide
    partner_names: tuple[str, ...]
    truth_family: str
    generator_parameters: dict[str, Any]


PARTNER_NAMES = (
    "partner_A",
    "partner_B",
    "partner_C",
    "partner_D",
    "other_typed",
    "explicit_unknown",
    "untyped",
    "fragment",
    "proofreading",
    "missing_annotation",
    "out_of_vocabulary",
)


SCENARIO_TRUTH = {
    "adequate_template": "T",
    "linear_continuum": "U_linear",
    "curved_continuum_dense": "U_curved",
    "curved_continuum_sparse_intermediates": "continuous_connected",
    "residual_groups_within_continuum": "M",
    "weak_residual_groups": "M_near_margin",
    "side_specific_endpoint_loss": "technical_artifact",
    "anatomy_quality_confound": "nuisance_explained",
    "high_strength_or_influential_cells": "concentrated_artifact",
}


def _balanced_groups(rng: np.random.Generator, size: int, components: int) -> Array:
    labels = np.arange(size) % components
    rng.shuffle(labels)
    return labels.astype(int)


def _draw_side(
    *,
    scenario: str,
    side: str,
    neurons: int,
    count_total: int,
    strength: float,
    missing_fraction: float,
    type_shift: Array,
    group_components: int,
    seed: int,
) -> SyntheticSide:
    rng = np.random.default_rng(seed)
    position = rng.uniform(-1.0, 1.0, size=neurons)
    quality = rng.normal(0.0, 1.0, size=neurons)
    covariates = np.column_stack([position, quality])
    base = np.array(
        [
            0.8,
            0.45,
            0.05,
            -0.15,
            -0.55,
            -1.15,
            -1.55,
            -1.75,
            -1.85,
            -2.0,
            -2.1,
        ]
    ) + type_shift
    anatomy = np.array(
        [
            0.35,
            -0.30,
            0.20,
            -0.18,
            0.02,
            -0.09,
            0.04,
            -0.03,
            0.01,
            -0.02,
            0.02,
        ]
    )
    quality_effect = np.array(
        [0.05, 0.02, -0.10, -0.12, 0.00, 0.18, 0.10, 0.12, 0.08, 0.11, 0.09]
    )
    logits = base[None, :] + position[:, None] * anatomy + quality[:, None] * quality_effect

    z = rng.normal(0.0, 1.0, size=neurons)
    groups = np.full(neurons, -1, dtype=int)
    linear_loading = np.array(
        [0.62, 0.38, -0.52, -0.36, -0.05, -0.07, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    curve_loading = np.array(
        [-0.22, 0.40, 0.34, -0.38, -0.09, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    group_loading = np.array(
        [0.72, 0.45, -0.62, -0.46, -0.05, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    if scenario == "linear_continuum":
        logits += strength * z[:, None] * linear_loading
    elif scenario == "curved_continuum_dense":
        logits += strength * (
            z[:, None] * linear_loading
            + 0.65 * (z**2 - 1.0)[:, None] * curve_loading
        )
    elif scenario == "curved_continuum_sparse_intermediates":
        poles = rng.choice([-1.0, 1.0], size=neurons)
        z = 1.45 * poles + rng.normal(0.0, 0.22, size=neurons)
        logits += strength * (
            z[:, None] * linear_loading
            + 0.65 * (z**2 - 1.0)[:, None] * curve_loading
        )
    elif scenario in {"residual_groups_within_continuum", "weak_residual_groups"}:
        groups = _balanced_groups(rng, neurons, group_components)
        centered = groups.astype(float) - float(
            np.mean(np.arange(group_components))
        )
        centered /= max(float(np.max(np.abs(centered))), 1.0)
        group_strength = strength if scenario == "residual_groups_within_continuum" else 0.45 * strength
        logits += 0.25 * z[:, None] * linear_loading
        logits += group_strength * centered[:, None] * group_loading
    elif scenario == "anatomy_quality_confound":
        logits += strength * (
            0.85 * position[:, None] * group_loading
            + 0.45 * quality[:, None] * curve_loading
        )
    elif scenario == "high_strength_or_influential_cells":
        if neurons:
            influential = int(rng.integers(0, neurons))
            logits[influential] += 3.0 * strength * group_loading
    elif scenario in {"adequate_template", "side_specific_endpoint_loss"}:
        pass
    else:
        raise ValueError(f"unknown generated scenario: {scenario}")

    probabilities = softmax(logits, axis=1)
    loss = np.zeros(neurons, dtype=float)
    if scenario == "side_specific_endpoint_loss" and side == "right":
        loss = np.clip(
            missing_fraction * (0.65 + 0.35 * (quality - quality.min()) / (np.ptp(quality) + 1e-9)),
            0.0,
            0.95,
        )
        removed = probabilities[:, 2:4] * loss[:, None]
        probabilities[:, 2:4] -= removed
        probabilities[:, PARTNER_NAMES.index("explicit_unknown")] += removed.sum(axis=1)

    totals = np.maximum(
        1,
        np.round(count_total * np.exp(rng.normal(0.0, 0.18, size=neurons))).astype(int),
    )
    if scenario == "high_strength_or_influential_cells" and neurons:
        totals[int(np.argmax(np.abs(logits @ group_loading)))] *= 12
    counts = np.vstack(
        [rng.multinomial(int(total), probability) for total, probability in zip(totals, probabilities)]
    ).astype(np.int64)
    return SyntheticSide(
        counts=counts,
        covariates=covariates,
        latent_coordinate=z,
        group=groups,
        endpoint_loss_fraction=loss,
    )


def generate_synthetic_type(
    scenario: str,
    *,
    left_neurons: int,
    right_neurons: int,
    count_total: int,
    strength: float,
    missing_fraction: float,
    seed: int,
) -> SyntheticType:
    if scenario not in SCENARIO_TRUTH:
        raise ValueError(f"unknown scenario: {scenario}")
    rng = np.random.default_rng(seed_from(seed, "type-shift"))
    type_shift = rng.normal(0.0, 0.18, size=len(PARTNER_NAMES))
    type_shift -= type_shift.mean()
    group_components = 2 + int(seed_from(seed, "group-components") % 2)
    left = _draw_side(
        scenario=scenario,
        side="left",
        neurons=left_neurons,
        count_total=count_total,
        strength=strength,
        missing_fraction=missing_fraction,
        type_shift=type_shift,
        group_components=group_components,
        seed=seed_from(seed, "left"),
    )
    right = _draw_side(
        scenario=scenario,
        side="right",
        neurons=right_neurons,
        count_total=count_total,
        strength=strength,
        missing_fraction=missing_fraction,
        type_shift=type_shift,
        group_components=group_components,
        seed=seed_from(seed, "right"),
    )
    return SyntheticType(
        scenario=scenario,
        left=left,
        right=right,
        partner_names=PARTNER_NAMES,
        truth_family=SCENARIO_TRUTH[scenario],
        generator_parameters={
            "left_neurons": left_neurons,
            "right_neurons": right_neurons,
            "count_total": count_total,
            "strength": strength,
            "missing_fraction": missing_fraction,
            "group_components": group_components,
            "seed": seed,
        },
    )
