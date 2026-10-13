"""Generated biological counterexamples for the actual EP12 T/U/M fitter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    from scipy.special import softmax
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("generated scientific fixtures require numpy and scipy") from exc

Array = np.ndarray


def seed_from(*parts: object) -> int:
    """Derive the historical fixture seed without importing fitted models."""

    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


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


@dataclass(frozen=True)
class SyntheticSideDGP:
    """Pre-count state and latent law for one generated side.

    ``realized_probabilities`` records the exact probabilities used by the
    historical fixture draw.  Oracle code instead uses ``latent_family`` and
    the corresponding distribution parameters, so it cannot condition on the
    realized latent coordinate or group label.
    """

    scenario: str
    side: str
    nuisance_logits: Array
    count_totals: Array
    endpoint_loss_fraction: Array
    latent_family: str
    strength: float
    group_strength: float
    group_components: int
    group_weights: Array
    group_centers: Array
    deterministic_logit_offsets: Array
    realized_probabilities: Array


@dataclass(frozen=True)
class SyntheticTypeWithDGP:
    fixture: SyntheticType
    left_dgp: SyntheticSideDGP
    right_dgp: SyntheticSideDGP


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


BASE_LOGITS = np.array(
    [0.8, 0.45, 0.05, -0.15, -0.55, -1.15, -1.55, -1.75, -1.85, -2.0, -2.1],
    dtype=float,
)
ANATOMY_LOADING = np.array(
    [0.35, -0.30, 0.20, -0.18, 0.02, -0.09, 0.04, -0.03, 0.01, -0.02, 0.02],
    dtype=float,
)
QUALITY_LOADING = np.array(
    [0.05, 0.02, -0.10, -0.12, 0.00, 0.18, 0.10, 0.12, 0.08, 0.11, 0.09],
    dtype=float,
)
LINEAR_LOADING = np.array(
    [0.62, 0.38, -0.52, -0.36, -0.05, -0.07, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
CURVE_LOADING = np.array(
    [-0.22, 0.40, 0.34, -0.38, -0.09, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
GROUP_LOADING = np.array(
    [0.72, 0.45, -0.62, -0.46, -0.05, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
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


def _group_centers(components: int) -> Array:
    centers = np.arange(components, dtype=float) - float(
        np.mean(np.arange(components))
    )
    centers /= max(float(np.max(np.abs(centers))), 1.0)
    return centers


def _apply_endpoint_loss(probabilities: Array, loss_fraction: float) -> Array:
    result = np.asarray(probabilities, dtype=float).copy()
    removed = result[2:4] * float(loss_fraction)
    result[2:4] -= removed
    result[PARTNER_NAMES.index("explicit_unknown")] += removed.sum()
    return result


def generator_probabilities(
    state: SyntheticSideDGP,
    neuron_index: int,
    *,
    latent_coordinate: float = 0.0,
    group_center: float | None = None,
) -> Array:
    """Evaluate one generator probability vector from its pre-count state.

    This function exposes the single probability path shared by oracle code.
    It deliberately accepts a component *center*, not a realized group label.
    """

    if neuron_index < 0 or neuron_index >= state.nuisance_logits.shape[0]:
        raise IndexError("synthetic DGP neuron index is out of range")
    logits = state.nuisance_logits[neuron_index].copy()
    z = float(latent_coordinate)
    if state.latent_family == "point_mass":
        pass
    elif state.latent_family == "normal_linear":
        logits += state.strength * z * LINEAR_LOADING
    elif state.latent_family in {"normal_curved", "sparse_curved"}:
        logits += state.strength * (
            z * LINEAR_LOADING + 0.65 * (z**2 - 1.0) * CURVE_LOADING
        )
    elif state.latent_family == "grouped_normal_linear":
        if group_center is None:
            raise ValueError("grouped generator probabilities require a group center")
        logits += 0.25 * z * LINEAR_LOADING
        logits += state.group_strength * float(group_center) * GROUP_LOADING
    elif state.latent_family == "deterministic_offset":
        logits += state.deterministic_logit_offsets[neuron_index]
    else:  # pragma: no cover - constructor owns the closed vocabulary
        raise ValueError(f"unknown synthetic latent family: {state.latent_family}")
    probabilities = softmax(logits)
    return _apply_endpoint_loss(
        probabilities, state.endpoint_loss_fraction[neuron_index]
    )


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
) -> tuple[SyntheticSide, SyntheticSideDGP]:
    rng = np.random.default_rng(seed)
    position = rng.uniform(-1.0, 1.0, size=neurons)
    quality = rng.normal(0.0, 1.0, size=neurons)
    covariates = np.column_stack([position, quality])
    base = BASE_LOGITS + type_shift
    logits = (
        base[None, :]
        + position[:, None] * ANATOMY_LOADING
        + quality[:, None] * QUALITY_LOADING
    )
    nuisance_logits = logits.copy()

    z = rng.normal(0.0, 1.0, size=neurons)
    groups = np.full(neurons, -1, dtype=int)
    latent_family = "point_mass"
    group_strength = 0.0
    deterministic_offsets = np.zeros_like(logits)

    if scenario == "linear_continuum":
        latent_family = "normal_linear"
        logits += strength * z[:, None] * LINEAR_LOADING
    elif scenario == "curved_continuum_dense":
        latent_family = "normal_curved"
        logits += strength * (
            z[:, None] * LINEAR_LOADING
            + 0.65 * (z**2 - 1.0)[:, None] * CURVE_LOADING
        )
    elif scenario == "curved_continuum_sparse_intermediates":
        latent_family = "sparse_curved"
        poles = rng.choice([-1.0, 1.0], size=neurons)
        z = 1.45 * poles + rng.normal(0.0, 0.22, size=neurons)
        logits += strength * (
            z[:, None] * LINEAR_LOADING
            + 0.65 * (z**2 - 1.0)[:, None] * CURVE_LOADING
        )
    elif scenario in {"residual_groups_within_continuum", "weak_residual_groups"}:
        latent_family = "grouped_normal_linear"
        groups = _balanced_groups(rng, neurons, group_components)
        centered = groups.astype(float) - float(
            np.mean(np.arange(group_components))
        )
        centered /= max(float(np.max(np.abs(centered))), 1.0)
        group_strength = strength if scenario == "residual_groups_within_continuum" else 0.45 * strength
        logits += 0.25 * z[:, None] * LINEAR_LOADING
        logits += group_strength * centered[:, None] * GROUP_LOADING
    elif scenario == "anatomy_quality_confound":
        latent_family = "deterministic_offset"
        deterministic_offsets = strength * (
            0.85 * position[:, None] * GROUP_LOADING
            + 0.45 * quality[:, None] * CURVE_LOADING
        )
        logits += deterministic_offsets
    elif scenario == "high_strength_or_influential_cells":
        latent_family = "deterministic_offset"
        if neurons:
            influential = int(rng.integers(0, neurons))
            deterministic_offsets[influential] += 3.0 * strength * GROUP_LOADING
            logits[influential] += 3.0 * strength * GROUP_LOADING
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
        totals[int(np.argmax(np.abs(logits @ GROUP_LOADING)))] *= 12
    counts = np.vstack(
        [rng.multinomial(int(total), probability) for total, probability in zip(totals, probabilities)]
    ).astype(np.int64)
    fixture = SyntheticSide(
        counts=counts,
        covariates=covariates,
        latent_coordinate=z,
        group=groups,
        endpoint_loss_fraction=loss,
    )
    if latent_family == "grouped_normal_linear":
        group_centers = _group_centers(group_components)
        group_weights = (
            np.bincount(groups, minlength=group_components).astype(float) / neurons
        )
    else:
        group_centers = np.empty(0, dtype=float)
        group_weights = np.empty(0, dtype=float)
    dgp = SyntheticSideDGP(
        scenario=scenario,
        side=side,
        nuisance_logits=nuisance_logits,
        count_totals=totals.copy(),
        endpoint_loss_fraction=loss.copy(),
        latent_family=latent_family,
        strength=float(strength),
        group_strength=float(group_strength),
        group_components=int(group_components),
        group_weights=group_weights,
        group_centers=group_centers,
        deterministic_logit_offsets=deterministic_offsets,
        realized_probabilities=probabilities.copy(),
    )
    return fixture, dgp


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
    return generate_synthetic_type_with_dgp(
        scenario,
        left_neurons=left_neurons,
        right_neurons=right_neurons,
        count_total=count_total,
        strength=strength,
        missing_fraction=missing_fraction,
        seed=seed,
    ).fixture


def generate_synthetic_type_with_dgp(
    scenario: str,
    *,
    left_neurons: int,
    right_neurons: int,
    count_total: int,
    strength: float,
    missing_fraction: float,
    seed: int,
) -> SyntheticTypeWithDGP:
    if scenario not in SCENARIO_TRUTH:
        raise ValueError(f"unknown scenario: {scenario}")
    rng = np.random.default_rng(seed_from(seed, "type-shift"))
    type_shift = rng.normal(0.0, 0.18, size=len(PARTNER_NAMES))
    type_shift -= type_shift.mean()
    group_components = 2 + int(seed_from(seed, "group-components") % 2)
    left, left_dgp = _draw_side(
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
    right, right_dgp = _draw_side(
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
    fixture = SyntheticType(
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
    return SyntheticTypeWithDGP(
        fixture=fixture,
        left_dgp=left_dgp,
        right_dgp=right_dgp,
    )
