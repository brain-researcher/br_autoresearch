#!/usr/bin/env python3
"""Finite-pool optimizer diagnostic for the EP11 synthetic C/H smoke families.

This program uses only the outcome-blind 206-cell metadata layout and fully
synthetic 56-coordinate projection compositions.  It never reads or constructs
a real projection outcome.  For each candidate planted group amplitude it fits
the complete identified-coordinate-bounded smoke-family C density by the
equal-unit marginal likelihood over a finite synthetic pool on one fixed
provisional A8 metadata support.  It then provisionally calibrates
the amplitude at which the exact, label-integrated generator H exceeds fitted C
by 0.100 nats/neuron.  At zero and at that selected amplitude it independently
fits the complete H family, including a smooth position/depth gate, and verifies
all three densities with fresh draws over the same A8 support.  This is a
finite-pool optimizer diagnostic, not a population-family ceiling. Positive-
amplitude population projection requires the separate, converged SAA ladder.
Finite-D12-to-A8 transport belongs to Panel C and is not tested here.

Unlike the earlier four-parameter projected proxy, neither fitted family holds
the generator base fixed or confines a fitted direction to the planted
signature.  C and H have separate coefficient vectors and are optimized
separately.  Their common continuous smoke family is the currently implemented
fixed standard-normal seven-node quadrature with a linear/quadratic connected
map.  H adds only a two-state symmetric signature and a smooth gate.  The
implementation is a bounded estimator diagnostic, not production-family
qualification and not authorization for outcome-bearing search.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Keep each independent optimizer single-threaded when several are run inside
# one Slurm allocation.
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


SCENARIO = "reusable_groups"
TARGET_GAIN = 0.100
DEFAULT_AMPLITUDE_GRID = (0.0, 0.10, 0.20, 0.30, 0.45, 0.65, 0.90, 1.25, 1.75, 2.50)
GATE_TERMS = 4
SUPPORT_BOUND = 12.0
VALUE_BOUND = 8.0
LATENT_BOUND = 8.0
GATE_BOUND = 8.0
SIGMA_BOUNDS = (0.18, 1.50)
PANEL_B_RUN_SEED = 20260927
PROJECTED_GRADIENT_TOLERANCE = 2.0e-5
BOUND_ACTIVE_TOLERANCE = 1.0e-8
FISHER_DIAGONAL_FLOOR = 1.0e-2
OPTIMIZER_SCALE_BOUNDS = (0.05, 10.0)
CONTINUOUS_NODES, GAUSSIAN_NODE_WEIGHTS = core.quadrature()
CONTINUOUS_NODE_BASIS = core.continuous_basis(CONTINUOUS_NODES)
VALUE_CONTRAST_BASIS = core.ordered_helmert(core.N_COORDINATES)
N_VALUE_CONTRASTS = core.N_COORDINATES - 1


@dataclass(frozen=True)
class DesignBasis:
    """A weighted-orthonormal basis for the fixed-support P column space."""

    transform: np.ndarray
    rank: int
    estimable_rank: int
    raw_term_count: int

    @classmethod
    def from_support(
        cls, design: np.ndarray, weights: np.ndarray | None = None
    ) -> "DesignBasis":
        design = np.asarray(design, dtype=float)
        if design.ndim != 2 or not np.isfinite(design).all():
            raise ValueError("Fixed-support P design must be a finite matrix")
        if weights is None:
            weights = np.full(len(design), 1.0 / len(design), dtype=float)
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != (len(design),)
            or np.any(weights <= 0.0)
            or not np.isfinite(weights).all()
        ):
            raise ValueError("Design-basis weights must be finite and positive")
        weights = weights / weights.sum()
        _, singular, right = np.linalg.svd(
            np.sqrt(weights)[:, None] * design, full_matrices=False
        )
        if not len(singular) or singular[0] <= 0.0:
            raise ValueError("Fixed-support P design has no estimable direction")
        estimable_rank = int(np.sum(singular > singular[0] * 1.0e-10))
        term_count = int(design.shape[1])
        transform = right[:estimable_rank].T / singular[:estimable_rank]
        for column in range(estimable_rank):
            pivot = int(np.argmax(np.abs(transform[:, column])))
            if transform[pivot, column] < 0.0:
                transform[:, column] *= -1.0
        transformed = design @ transform
        np.testing.assert_allclose(
            transformed.T @ (weights[:, None] * transformed),
            np.eye(estimable_rank),
            atol=5.0e-12,
            rtol=0.0,
        )
        return cls(
            transform=transform,
            rank=estimable_rank,
            estimable_rank=estimable_rank,
            raw_term_count=term_count,
        )

    def apply(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(design, dtype=float) @ self.transform


@dataclass(frozen=True)
class ScoreData:
    design: np.ndarray
    gate_design: np.ndarray
    active: np.ndarray
    log_values: np.ndarray
    log_jacobian: np.ndarray
    dimension: np.ndarray
    weights: np.ndarray
    outcomes: np.ndarray


@dataclass(frozen=True)
class PopulationPool:
    data: ScoreData
    single_design: np.ndarray
    truth: core.SyntheticTruth
    replicates: int


@dataclass(frozen=True)
class OptimizerSolution:
    theta: np.ndarray
    objective: float
    optimizer_success: bool
    converged: bool
    iterations: int
    evaluations: int
    gradient_infinity_norm: float
    projected_gradient_infinity_norm: float
    scaled_projected_gradient_infinity_norm: float
    projected_gradient_block_infinity_norms: dict[str, float]
    optimizer_scale_minimum: float
    optimizer_scale_maximum: float
    numerical_boundary_diagnostics: dict[str, object]
    at_iteration_cap: bool
    message: str


@dataclass(frozen=True)
class FamilyFit:
    family: str
    amplitude: float
    solutions: tuple[OptimizerSolution, ...]
    parameter_count: int
    design_rank: int
    embedded_c_objective: float | None = None

    @property
    def converged_solutions(self) -> tuple[OptimizerSolution, ...]:
        good = tuple(solution for solution in self.solutions if solution.converged)
        return tuple(sorted(good, key=lambda solution: solution.objective))

    @property
    def ranked_solutions(self) -> tuple[OptimizerSolution, ...]:
        return self.converged_solutions

    @property
    def best(self) -> OptimizerSolution:
        ranked = self.ranked_solutions
        if not ranked:
            details = "; ".join(
                (
                    f"start={index}: kkt_converged={solution.converged}, "
                    f"optimizer_success={solution.optimizer_success}, "
                    f"nit={solution.iterations}, cap={solution.at_iteration_cap}, "
                    f"projected_grad_inf={solution.projected_gradient_infinity_norm:.6g}, "
                    f"scaled_projected_grad_inf="
                    f"{solution.scaled_projected_gradient_infinity_norm:.6g}, "
                    f"block_projected_grad_inf="
                    f"{solution.projected_gradient_block_infinity_norms}, "
                    f"boundaries={solution.numerical_boundary_diagnostics}, "
                    f"scipy_message={solution.message!r}"
                )
                for index, solution in enumerate(self.solutions)
            )
            raise RuntimeError(
                f"No KKT-converged {self.family} optimizer solution ({details})"
            )
        return ranked[0]


class ParameterCodec:
    """Pack the complete identified coordinates of one bounded smoke family."""

    def __init__(self, family: str, design_rank: int):
        if family not in {"C", "H"}:
            raise ValueError("Family must be C or H")
        self.family = family
        self.design_rank = int(design_rank)
        self.shapes: dict[str, tuple[int, ...]] = {
            "base_support": (design_rank, core.N_COORDINATES),
            "base_value": (design_rank, N_VALUE_CONTRASTS),
            "continuous_support": (2, core.N_COORDINATES),
            "continuous_value": (2, N_VALUE_CONTRASTS),
        }
        if family == "H":
            self.shapes.update(
                {
                    "group_support": (core.N_COORDINATES,),
                    "group_value": (N_VALUE_CONTRASTS,),
                    "gate_beta": (GATE_TERMS,),
                }
            )
        self.shapes["log_sigma"] = (1,)
        self.slices: dict[str, slice] = {}
        cursor = 0
        for name, shape in self.shapes.items():
            size = int(np.prod(shape))
            self.slices[name] = slice(cursor, cursor + size)
            cursor += size
        self.size = cursor

    def blank(self) -> dict[str, np.ndarray]:
        return {name: np.zeros(shape, dtype=float) for name, shape in self.shapes.items()}

    def pack(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        if set(blocks) != set(self.shapes):
            raise ValueError("Parameter blocks do not match codec")
        theta = np.empty(self.size, dtype=float)
        for name, shape in self.shapes.items():
            value = np.asarray(blocks[name], dtype=float)
            if value.shape != shape:
                raise ValueError(f"Block {name} has shape {value.shape}, expected {shape}")
            theta[self.slices[name]] = value.ravel()
        return theta

    def unpack(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.size,):
            raise ValueError("Packed parameter vector has wrong length")
        return {
            name: theta[self.slices[name]].reshape(shape)
            for name, shape in self.shapes.items()
        }

    def bounds(self) -> list[tuple[float, float]]:
        limits = {
            "base_support": (-SUPPORT_BOUND, SUPPORT_BOUND),
            "base_value": (-VALUE_BOUND, VALUE_BOUND),
            "continuous_support": (-LATENT_BOUND, LATENT_BOUND),
            "continuous_value": (-LATENT_BOUND, LATENT_BOUND),
            "group_support": (-LATENT_BOUND, LATENT_BOUND),
            "group_value": (-LATENT_BOUND, LATENT_BOUND),
            "gate_beta": (-GATE_BOUND, GATE_BOUND),
            "log_sigma": (math.log(SIGMA_BOUNDS[0]), math.log(SIGMA_BOUNDS[1])),
        }
        output: list[tuple[float, float]] = []
        for name, shape in self.shapes.items():
            output.extend([limits[name]] * int(np.prod(shape)))
        return output


def project_value_coordinates(values: np.ndarray) -> np.ndarray:
    """Remove the unidentifiable all-coordinate value shift."""
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != core.N_COORDINATES:
        raise ValueError("Full value array must end in 56 coordinates")
    return values @ VALUE_CONTRAST_BASIS


def expand_value_coordinates(contrasts: np.ndarray) -> np.ndarray:
    """Map identifiable Helmert value contrasts back to 56 coordinates."""
    contrasts = np.asarray(contrasts, dtype=float)
    if contrasts.shape[-1] != N_VALUE_CONTRASTS:
        raise ValueError("Value-contrast array must end in 55 coordinates")
    return contrasts @ VALUE_CONTRAST_BASIS.T


def smooth_gate_basis(layout: core.MetadataLayout, indices: np.ndarray) -> np.ndarray:
    pc0 = core.standardized(layout.pc[:, 0])[indices]
    depth = core.standardized(layout.layer_depth[:, None])[:, 0][indices]
    return np.column_stack([np.ones(len(indices)), pc0, pc0**2 - 1.0, depth])


def equal_unit_weights(layout: core.MetadataLayout, indices: np.ndarray) -> np.ndarray:
    units = layout.units[np.asarray(indices, dtype=int)]
    unique, counts = np.unique(units, return_counts=True)
    count_by_unit = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray(
        [1.0 / (len(unique) * count_by_unit[unit]) for unit in units], dtype=float
    )
    if not np.isfinite(weights).all():
        raise FloatingPointError("Equal-unit weights are nonfinite")
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1.0e-14, rtol=0.0)
    return weights


def outcome_arrays(outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim != 2 or outcomes.shape[1] != core.N_COORDINATES:
        raise ValueError("Outcomes must be n-by-56")
    active = outcomes > 0.0
    counts = active.sum(axis=1)
    if np.any(counts == 0) or not np.allclose(outcomes.sum(axis=1), 1.0, atol=1.0e-10):
        raise ValueError("Every synthetic composition must occupy a nonempty face")
    log_values = np.zeros_like(outcomes)
    log_values[active] = np.log(outcomes[active])
    log_jacobian = -np.log(counts) - log_values.sum(axis=1)
    dimension = counts.astype(float) - 1.0
    return active, log_values, log_jacobian, dimension


def make_score_data(
    layout: core.MetadataLayout,
    basis: DesignBasis,
    indices: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray | None = None,
) -> ScoreData:
    active, log_values, log_jacobian, dimension = outcome_arrays(outcomes)
    if weights is None:
        weights = equal_unit_weights(layout, indices)
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (len(indices),)
        or np.any(weights < 0.0)
        or not np.isfinite(weights).all()
        or not float(weights.sum()) > 0.0
    ):
        raise ValueError("Score weights are misaligned")
    weights = weights / weights.sum()
    return ScoreData(
        design=basis.apply(layout.p_design[indices]),
        gate_design=smooth_gate_basis(layout, indices),
        active=active,
        log_values=log_values,
        log_jacobian=log_jacobian,
        dimension=dimension,
        weights=weights,
        outcomes=np.asarray(outcomes, dtype=float),
    )


def rng_for(seed: int, stream: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), int(replicate)]))


def panel_b_fixed_roles(layout: core.MetadataLayout) -> core.RoleSplit:
    """Reconstruct the completed Panel-B provisional core without exposing IDs."""
    calibration_seed = core.stable_u64(
        PANEL_B_RUN_SEED, "repaired_proxy_calibration"
    ) % (2**32)
    role_seed = core.stable_u64(calibration_seed, "oracle_fixed_roles") % (2**32)
    roles = core.role_split(layout, np.random.default_rng(role_seed))
    counts = (
        len(set(layout.units[roles.development].tolist())),
        len(set(layout.units[roles.audit].tolist())),
        len(set(layout.units[roles.unassigned].tolist())),
    )
    if counts != (12, 8, 16):
        raise AssertionError("Panel-B role reconstruction is not D12/A8/16")
    if (
        len(roles.development),
        len(roles.audit),
        len(roles.unassigned),
    ) != (68, 76, 62):
        raise AssertionError("Panel-B role reconstruction changed its cell partition")
    assigned = np.concatenate([roles.development, roles.audit, roles.unassigned])
    if len(assigned) != len(layout.frame) or len(np.unique(assigned)) != len(layout.frame):
        raise AssertionError("Panel-B role reconstruction is not a complete disjoint partition")
    return roles


def generate_full_synthetic(
    layout: core.MetadataLayout,
    amplitude: float,
    seed: int,
    stream: int,
    replicate: int,
) -> tuple[np.ndarray, core.SyntheticTruth]:
    """Generate the invariant full-layout reusable-group DGP.

    Group uniforms, support uniforms, and Gaussian noise use separate child
    streams.  A fixed 64-round support-uniform cube makes the random numbers
    common across amplitudes even when conditional-nonempty rejection would
    otherwise consume an amplitude-dependent number of draws.
    """
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("Amplitude must be finite and nonnegative")
    all_indices = np.arange(len(layout.frame), dtype=int)
    seed_sequence = np.random.SeedSequence([int(seed), int(stream), int(replicate)])
    group_seed, support_seed, noise_seed = seed_sequence.spawn(3)
    group_rng = np.random.default_rng(group_seed)
    support_rng = np.random.default_rng(support_seed)
    noise_rng = np.random.default_rng(noise_seed)

    continuous_value, continuous_support, group_value, group_support = core.truth_directions()
    features = core.continuous_truth_features(
        layout, all_indices, SCENARIO, np.random.default_rng(0)
    )
    gate_coordinate = core.standardized(layout.pc[:, 0])
    probability_one = expit(0.30 * gate_coordinate)
    groups = (group_rng.random(len(all_indices)) < probability_one).astype(int)
    group_probabilities = np.column_stack([1.0 - probability_one, probability_one])

    coordinate = np.arange(core.N_COORDINATES, dtype=float)
    support_intercept = -1.75 + 0.50 * np.cos(
        2.0 * math.pi * coordinate / core.N_COORDINATES
    )
    value_intercept = 0.25 * np.sin(
        2.0 * math.pi * coordinate / core.N_COORDINATES + 0.3
    )
    line_effect = core.fixed_line_effect(layout.lines)
    position_loading = np.vstack(
        [
            0.16 * np.sin(coordinate * 0.21),
            0.13 * np.cos(coordinate * 0.17),
            0.11 * np.sin(coordinate * 0.11 + 0.5),
        ]
    )
    base_support_logits = (
        support_intercept[None, :]
        + 0.32 * features @ continuous_support
        + 0.20 * layout.coords @ position_loading
        + 0.50 * line_effect
    )
    base_value_mean = (
        value_intercept[None, :]
        + 0.46 * features @ continuous_value
        + 0.25 * layout.coords @ position_loading
        + line_effect
    )
    group_support_shift = np.vstack(
        [-(0.58 * amplitude) * group_support, +(0.58 * amplitude) * group_support]
    )
    group_value_shift = np.vstack(
        [-(0.72 * amplitude) * group_value, +(0.72 * amplitude) * group_value]
    )
    support_logits = base_support_logits + group_support_shift[groups]
    value_mean = base_value_mean + group_value_shift[groups]

    probability = expit(support_logits)
    uniforms = support_rng.random((64, len(all_indices), core.N_COORDINATES))
    candidates = uniforms < probability[None, :, :]
    nonempty = candidates.any(axis=2)
    if np.any(~nonempty.any(axis=0)):
        raise RuntimeError("Fixed conditional-nonempty draw cube was exhausted")
    first = np.argmax(nonempty, axis=0)
    active = candidates[first, np.arange(len(all_indices))]
    noise = noise_rng.normal(0.0, core.GENERATOR_NOISE_SIGMA, size=value_mean.shape)
    noisy_log_intensity = value_mean + noise
    outcomes = np.zeros_like(noisy_log_intensity)
    for row in range(len(outcomes)):
        on = active[row]
        values = noisy_log_intensity[row, on]
        values = np.exp(values - values.max())
        outcomes[row, on] = values / values.sum()

    truth = core.SyntheticTruth(
        groups=groups,
        group_probabilities=group_probabilities,
        base_support_logits=base_support_logits,
        base_value_mean=base_value_mean,
        group_support_shift=group_support_shift,
        group_value_shift=group_value_shift,
        group_support_direction=group_support.copy(),
        group_value_direction=group_value.copy(),
        noise_sigma=core.GENERATOR_NOISE_SIGMA,
    )
    return outcomes, truth


def build_population_pool(
    layout: core.MetadataLayout,
    basis: DesignBasis,
    indices: np.ndarray,
    amplitude: float,
    replicates: int,
    seed: int,
    stream: int,
) -> PopulationPool:
    if replicates < 1:
        raise ValueError("Population pool requires at least one replicate")
    outcomes: list[np.ndarray] = []
    first_truth: core.SyntheticTruth | None = None
    for replicate in range(replicates):
        full_outcomes, full_truth = generate_full_synthetic(
            layout, amplitude, seed, stream, replicate
        )
        outcomes.append(full_outcomes[indices])
        if first_truth is None:
            first_truth = core.subset_synthetic_truth(full_truth, indices)
    assert first_truth is not None
    stacked = np.vstack(outcomes)
    expected_shape = (replicates * len(indices), core.N_COORDINATES)
    if stacked.shape != expected_shape or not np.isfinite(stacked).all():
        raise AssertionError(
            f"Population pool has shape {stacked.shape}, expected {expected_shape}"
        )
    single_weights = equal_unit_weights(layout, indices)
    weights = np.tile(single_weights / replicates, replicates)
    repeated_indices = np.tile(indices, replicates)
    data = make_score_data(layout, basis, repeated_indices, stacked, weights)
    return PopulationPool(
        data=data,
        single_design=basis.apply(layout.p_design[indices]),
        truth=first_truth,
        replicates=replicates,
    )


def vector_log1mexp(log_values: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    output = np.empty_like(values)
    lower = values < math.log(0.5)
    output[lower] = np.log1p(-np.exp(values[lower]))
    output[~lower] = np.log(-np.expm1(values[~lower]))
    return output


def log_nonempty_from_logits(logits: np.ndarray) -> np.ndarray:
    """Stable log probability that independent Bernoulli support is nonempty."""
    logits = np.asarray(logits, dtype=float)
    log_softplus = np.empty_like(logits)
    low = logits < -35.0
    high = logits > 35.0
    middle = ~(low | high)
    # log(softplus(x)) is asymptotic to x on the far negative tail and to
    # log(x) on the far positive tail.  These branches retain information that
    # expit/logaddexp alone would round away for an all-off configuration.
    log_softplus[low] = logits[low]
    log_softplus[high] = np.log(logits[high])
    log_softplus[middle] = np.log(np.logaddexp(0.0, logits[middle]))
    log_total_hazard = logsumexp(log_softplus, axis=-1)
    output = np.empty_like(log_total_hazard)
    tiny = log_total_hazard < -20.0
    tiny_hazard = np.exp(log_total_hazard[tiny])
    output[tiny] = log_total_hazard[tiny] + np.log1p(
        -0.5 * tiny_hazard + np.square(tiny_hazard) / 6.0
    )
    total_hazard = np.exp(log_total_hazard[~tiny])
    output[~tiny] = np.log(-np.expm1(-total_hazard))
    if not np.isfinite(output).all():
        raise FloatingPointError("Conditional-nonempty normalizer is nonfinite")
    return output


def family_objective_gradient(
    codec: ParameterCodec,
    theta: np.ndarray,
    data: ScoreData,
    return_scores: bool = False,
    return_responsibilities: bool = False,
) -> tuple[float, np.ndarray] | np.ndarray | tuple[np.ndarray, np.ndarray]:
    if return_responsibilities and not return_scores:
        raise ValueError("Responsibilities are available only with return_scores=True")
    blocks = codec.unpack(theta)
    n_rows = len(data.active)
    node_weights = GAUSSIAN_NODE_WEIGHTS
    if codec.family == "C":
        state_node = np.arange(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int)
        state_group = np.zeros(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int)
        group_code = np.zeros(core.CONTINUOUS_QUADRATURE_ORDER, dtype=float)
        log_mixture = np.broadcast_to(np.log(node_weights)[None, :], (n_rows, len(state_node)))
        gate_probability = None
        gate_derivative = None
    else:
        state_node = np.tile(np.arange(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int), 2)
        state_group = np.repeat(np.arange(2, dtype=int), core.CONTINUOUS_QUADRATURE_ORDER)
        group_code = np.where(state_group == 0, -1.0, 1.0)
        raw_gate = data.gate_design @ blocks["gate_beta"]
        gate_probability = expit(raw_gate)
        log_gate = np.column_stack(
            [-np.logaddexp(0.0, raw_gate), -np.logaddexp(0.0, -raw_gate)]
        )
        log_mixture = log_gate[:, state_group] + np.log(node_weights)[state_node][None, :]

    base_support = data.design @ blocks["base_support"]
    base_value = expand_value_coordinates(data.design @ blocks["base_value"])
    continuous_support = CONTINUOUS_NODE_BASIS @ blocks["continuous_support"]
    continuous_value = expand_value_coordinates(
        CONTINUOUS_NODE_BASIS @ blocks["continuous_value"]
    )
    support_shift = continuous_support[state_node]
    value_shift = continuous_value[state_node]
    if codec.family == "H":
        support_shift = support_shift + group_code[:, None] * blocks["group_support"][None, :]
        group_value = expand_value_coordinates(blocks["group_value"])
        value_shift = value_shift + group_code[:, None] * group_value[None, :]

    raw_support = base_support[:, None, :] + support_shift[None, :, :]
    support_probability = expit(raw_support)
    log_on = -np.logaddexp(0.0, -raw_support)
    log_off = -np.logaddexp(0.0, raw_support)
    unconditioned = np.where(data.active[:, None, :], log_on, log_off).sum(axis=2)
    log_empty = log_off.sum(axis=2)
    log_nonempty = log_nonempty_from_logits(raw_support)
    support_log_density = unconditioned - log_nonempty

    sigma = math.exp(float(blocks["log_sigma"][0]))
    variance = sigma**2
    mean = base_value[:, None, :] + value_shift[None, :, :]
    difference = (data.log_values[:, None, :] - mean) * data.active[:, None, :]
    active_count = data.dimension + 1.0
    face_mean = difference.sum(axis=2) / active_count[:, None]
    residual = (difference - face_mean[:, :, None]) * data.active[:, None, :]
    residual_sum = np.square(residual).sum(axis=2)
    value_log_density = -0.5 * (
        data.dimension[:, None] * math.log(2.0 * math.pi * variance)
        + residual_sum / variance
    ) + data.log_jacobian[:, None]
    components = log_mixture + support_log_density + value_log_density
    scores = logsumexp(components, axis=1)
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"{codec.family} produced a nonfinite cell score")

    responsibilities = np.exp(components - scores[:, None])
    if not np.isfinite(responsibilities).all():
        raise FloatingPointError(f"{codec.family} produced nonfinite responsibilities")
    if return_scores:
        if return_responsibilities:
            return scores, responsibilities
        return scores
    weighted_responsibilities = data.weights[:, None] * responsibilities
    log_empty_correction = (
        log_on + log_empty[:, :, None] - log_nonempty[:, :, None]
    )
    if np.max(log_empty_correction) > 1.0e-10:
        raise AssertionError("Conditional-empty gradient correction exceeded one")
    empty_correction = np.exp(np.minimum(log_empty_correction, 0.0))
    support_signal = (
        data.active[:, None, :]
        - support_probability
        - empty_correction
    )
    weighted_support = weighted_responsibilities[:, :, None] * support_signal

    value_signal = residual / variance
    weighted_value = weighted_responsibilities[:, :, None] * value_signal
    gradient = codec.blank()
    base_support_signal = weighted_support.sum(axis=1)
    gradient["base_support"] = data.design.T @ base_support_signal
    gradient["continuous_support"] = np.einsum(
        "nsj,sq->qj", weighted_support, CONTINUOUS_NODE_BASIS[state_node], optimize=True
    )
    base_value_signal = weighted_value.sum(axis=1)
    gradient["base_value"] = project_value_coordinates(
        data.design.T @ base_value_signal
    )
    continuous_value_gradient = np.einsum(
        "nsj,sq->qj", weighted_value, CONTINUOUS_NODE_BASIS[state_node], optimize=True
    )
    gradient["continuous_value"] = project_value_coordinates(
        continuous_value_gradient
    )

    if codec.family == "H":
        gradient["group_support"] = np.einsum(
            "nsj,s->j", weighted_support, group_code, optimize=True
        )
        gradient["group_value"] = project_value_coordinates(
            np.einsum("nsj,s->j", weighted_value, group_code, optimize=True)
        )
        posterior_one = responsibilities[:, state_group == 1].sum(axis=1)
        gate_signal = data.weights * (posterior_one - gate_probability)
        gradient["gate_beta"] = data.gate_design.T @ gate_signal

    sigma_signal = -data.dimension[:, None] + residual_sum / variance
    gradient["log_sigma"][0] = float(np.sum(weighted_responsibilities * sigma_signal))
    log_likelihood = float(np.sum(data.weights * scores))
    packed_gradient = codec.pack(gradient)
    if not np.isfinite(log_likelihood) or not np.isfinite(packed_gradient).all():
        raise FloatingPointError(f"{codec.family} objective or gradient is nonfinite")
    return -log_likelihood, -packed_gradient


def truth_scores(outcomes: np.ndarray, truth: core.SyntheticTruth) -> np.ndarray:
    """Vectorized exact generator H score with both labels integrated."""
    active, log_values, log_jacobian, dimension = outcome_arrays(outcomes)
    support_logits = (
        truth.base_support_logits[:, None, :]
        + truth.group_support_shift[None, :, :]
    )
    log_on = -np.logaddexp(0.0, -support_logits)
    log_off = -np.logaddexp(0.0, support_logits)
    unconditioned = np.where(active[:, None, :], log_on, log_off).sum(axis=2)
    support_log_density = unconditioned - log_nonempty_from_logits(
        support_logits
    )
    mean = truth.base_value_mean[:, None, :] + truth.group_value_shift[None, :, :]
    difference = (log_values[:, None, :] - mean) * active[:, None, :]
    active_count = dimension + 1.0
    face_mean = difference.sum(axis=2) / active_count[:, None]
    residual = (difference - face_mean[:, :, None]) * active[:, None, :]
    residual_sum = np.square(residual).sum(axis=2)
    variance = float(truth.noise_sigma**2)
    value_log_density = -0.5 * (
        dimension[:, None] * math.log(2.0 * math.pi * variance)
        + residual_sum / variance
    ) + log_jacobian[:, None]
    components = (
        np.log(np.clip(truth.group_probabilities, 1.0e-300, 1.0))
        + support_log_density
        + value_log_density
    )
    scores = logsumexp(components, axis=1)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Exact generator produced a nonfinite cell score")
    return scores


def truth_based_blocks(
    codec: ParameterCodec,
    pool: PopulationPool,
    amplitude: float,
    mode: str,
) -> dict[str, np.ndarray]:
    blocks = codec.blank()
    truth = pool.truth
    blocks["base_support"] = np.linalg.lstsq(
        pool.single_design, truth.base_support_logits, rcond=None
    )[0]
    blocks["base_value"] = project_value_coordinates(
        np.linalg.lstsq(pool.single_design, truth.base_value_mean, rcond=None)[0]
    )
    blocks["log_sigma"][0] = math.log(truth.noise_sigma)

    if mode == "adjacent_node_scale":
        middle = core.CONTINUOUS_QUADRATURE_ORDER // 2
        node_scale = abs(float(CONTINUOUS_NODES[middle + 1]))
    else:
        node_scale = 1.0

    if codec.family == "C":
        blocks["continuous_support"][0] = truth.group_support_shift[1] / node_scale
        blocks["continuous_value"][0] = project_value_coordinates(
            truth.group_value_shift[1] / node_scale
        )
    else:
        blocks["group_support"] = truth.group_support_shift[1]
        blocks["group_value"] = project_value_coordinates(
            truth.group_value_shift[1]
        )
        logits = np.log(
            np.clip(truth.group_probabilities[:, 1], 1.0e-12, 1.0 - 1.0e-12)
            / np.clip(truth.group_probabilities[:, 0], 1.0e-12, 1.0)
        )
        blocks["gate_beta"] = np.linalg.lstsq(
            pool.data.gate_design[: len(logits)], logits, rcond=None
        )[0]
    return blocks


def fit_gate_initialization(
    design: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    prevalence = float(np.clip(np.sum(weights * labels) / weights.sum(), 1.0e-4, 1.0 - 1.0e-4))
    start = np.array([math.log(prevalence / (1.0 - prevalence)), 0.0, 0.0, 0.0])
    penalty = np.array([0.0, 0.05, 0.05, 0.05])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        value = float(
            np.sum(weights * (np.logaddexp(0.0, eta) - labels * eta))
            + 0.5 * np.sum(penalty * beta**2)
        )
        gradient = design.T @ (weights * (expit(eta) - labels)) + penalty * beta
        return value, gradient

    result = minimize(
        lambda beta: objective(beta),
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-GATE_BOUND, GATE_BOUND)] * GATE_TERMS,
        options={"maxiter": 100, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    return np.asarray(result.x if np.isfinite(result.x).all() else start, dtype=float)


def empirical_blocks(codec: ParameterCodec, pool: PopulationPool) -> dict[str, np.ndarray]:
    """Outcome-derived start only; the final objective reoptimizes every block."""
    data = pool.data
    base = core.fit_base_model(
        "population_start",
        "P",
        data.design,
        data.outcomes,
        support_ridge=1.0,
        value_ridge=1.0,
    )
    base_support = data.design @ base.support_beta
    base_value = data.design @ base.value_beta
    residual = core.residual_features(data.outcomes, base_support, base_value)
    scores, _ = core.principal_scores(residual, n_components=1)
    coordinate = core.rank_normal(scores[:, 0])
    latent_design = core.continuous_basis(coordinate)
    continuous_support, continuous_value = core.fit_adjustments(
        data.outcomes,
        base_support,
        base_value,
        latent_design,
        support_ridge=1.0,
        value_ridge=1.0,
    )
    blocks = codec.blank()
    blocks["base_support"] = base.support_beta
    blocks["base_value"] = project_value_coordinates(base.value_beta)
    blocks["continuous_support"] = continuous_support
    blocks["continuous_value"] = project_value_coordinates(continuous_value)
    blocks["log_sigma"][0] = math.log(base.sigma)
    if codec.family == "H":
        conditional_support = base_support + latent_design @ continuous_support
        conditional_value = base_value + latent_design @ continuous_value
        conditional_residual = core.residual_features(
            data.outcomes, conditional_support, conditional_value
        )
        labels = core.two_means(conditional_residual)
        signed = (2.0 * labels - 1.0)[:, None]
        group_support, group_value = core.fit_adjustments(
            data.outcomes,
            conditional_support,
            conditional_value,
            signed,
            support_ridge=1.0,
            value_ridge=1.0,
        )
        blocks["group_support"] = group_support[0]
        blocks["group_value"] = project_value_coordinates(group_value[0])
        blocks["gate_beta"] = fit_gate_initialization(
            data.gate_design, labels, data.weights
        )
    return blocks


def clip_to_bounds(theta: np.ndarray, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    output = np.asarray(theta, dtype=float).copy()
    for index, (lower, upper) in enumerate(bounds):
        output[index] = np.clip(output[index], lower + 1.0e-7, upper - 1.0e-7)
    return output


def embed_c_theta_in_h(c_theta: np.ndarray, design_rank: int) -> np.ndarray:
    codec_c = ParameterCodec("C", design_rank)
    codec_h = ParameterCodec("H", design_rank)
    blocks_c = codec_c.unpack(np.asarray(c_theta, dtype=float))
    if not np.isfinite(c_theta).all():
        raise FloatingPointError("Cannot embed a nonfinite fitted-C parameter vector")
    blocks_h = codec_h.blank()
    for name in codec_c.shapes:
        blocks_h[name] = blocks_c[name].copy()
    return codec_h.pack(blocks_h)


def projected_gradient(
    theta: np.ndarray,
    gradient: np.ndarray,
    bounds: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return the first-order KKT residual for box-constrained minimization."""
    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if theta.shape != gradient.shape or len(bounds) != len(theta):
        raise ValueError("Projected-gradient inputs are misaligned")
    if not np.isfinite(theta).all() or not np.isfinite(gradient).all():
        raise FloatingPointError("Projected-gradient inputs are nonfinite")
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    if np.any(theta < lower - BOUND_ACTIVE_TOLERANCE) or np.any(
        theta > upper + BOUND_ACTIVE_TOLERANCE
    ):
        raise AssertionError("Optimizer returned a point outside its bounds")
    residual = gradient.copy()
    lower_kkt = (theta <= lower + BOUND_ACTIVE_TOLERANCE) & (gradient >= 0.0)
    upper_kkt = (theta >= upper - BOUND_ACTIVE_TOLERANCE) & (gradient <= 0.0)
    residual[lower_kkt | upper_kkt] = 0.0
    return residual


def fisher_jacobi_scale(
    codec: ParameterCodec, theta: np.ndarray, data: ScoreData
) -> np.ndarray:
    """Return a positive diagonal optimizer scale in identifiable coordinates.

    The value, gate, and noise entries use complete-data Fisher diagonals under
    the current state responsibilities; the support entries use the positive
    unconditioned-Bernoulli diagonal as a curvature proxy because the scored
    support law is conditional on a nonempty face.  This is only a numerical
    change of variables: the fitted density, physical coefficient bounds, and
    final physical-coordinate KKT check are unchanged.
    """
    blocks = codec.unpack(theta)
    _, responsibilities = family_objective_gradient(
        codec,
        theta,
        data,
        return_scores=True,
        return_responsibilities=True,
    )
    if codec.family == "C":
        state_node = np.arange(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int)
        state_group = np.zeros(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int)
    else:
        state_node = np.tile(
            np.arange(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int), 2
        )
        state_group = np.repeat(
            np.arange(2, dtype=int), core.CONTINUOUS_QUADRATURE_ORDER
        )
    group_code = np.where(state_group == 0, -1.0, 1.0)
    base_support = data.design @ blocks["base_support"]
    continuous_support = CONTINUOUS_NODE_BASIS @ blocks["continuous_support"]
    support_shift = continuous_support[state_node]
    if codec.family == "H":
        support_shift = (
            support_shift
            + group_code[:, None] * blocks["group_support"][None, :]
        )
    raw_support = base_support[:, None, :] + support_shift[None, :, :]
    support_probability = expit(raw_support)
    support_information = support_probability * (1.0 - support_probability)
    weighted_responsibilities = data.weights[:, None] * responsibilities
    curvature = codec.blank()
    curvature["base_support"] = np.einsum(
        "ns,nsj,nk->kj",
        weighted_responsibilities,
        support_information,
        np.square(data.design),
        optimize=True,
    )
    curvature["continuous_support"] = np.einsum(
        "ns,nsj,sq->qj",
        weighted_responsibilities,
        support_information,
        np.square(CONTINUOUS_NODE_BASIS[state_node]),
        optimize=True,
    )
    if codec.family == "H":
        curvature["group_support"] = np.einsum(
            "ns,nsj,s->j",
            weighted_responsibilities,
            support_information,
            np.square(group_code),
            optimize=True,
        )

    active = data.active.astype(float)
    active_count = data.dimension + 1.0
    face_contrast_norm = (
        active @ np.square(VALUE_CONTRAST_BASIS)
        - np.square(active @ VALUE_CONTRAST_BASIS) / active_count[:, None]
    )
    if np.any(face_contrast_norm < -1.0e-12):
        raise AssertionError("Face-projected value curvature became negative")
    face_contrast_norm = np.maximum(face_contrast_norm, 0.0)
    variance = math.exp(2.0 * float(blocks["log_sigma"][0]))
    curvature["base_value"] = np.einsum(
        "n,nk,nq->kq",
        data.weights,
        np.square(data.design),
        face_contrast_norm,
        optimize=True,
    ) / variance
    expected_continuous_square = responsibilities @ np.square(
        CONTINUOUS_NODE_BASIS[state_node]
    )
    curvature["continuous_value"] = np.einsum(
        "n,np,nq->pq",
        data.weights,
        expected_continuous_square,
        face_contrast_norm,
        optimize=True,
    ) / variance
    if codec.family == "H":
        curvature["group_value"] = (
            np.einsum(
                "n,nq->q", data.weights, face_contrast_norm, optimize=True
            )
            / variance
        )
        raw_gate = data.gate_design @ blocks["gate_beta"]
        gate_probability = expit(raw_gate)
        curvature["gate_beta"] = np.einsum(
            "n,nk->k",
            data.weights * gate_probability * (1.0 - gate_probability),
            np.square(data.gate_design),
            optimize=True,
        )
    curvature["log_sigma"][0] = float(
        2.0 * np.sum(data.weights * data.dimension)
    )
    packed = codec.pack(curvature)
    if not np.isfinite(packed).all() or np.any(packed < -1.0e-12):
        raise FloatingPointError("Fisher-Jacobi curvature is invalid")
    scale = 1.0 / np.sqrt(np.maximum(packed, FISHER_DIAGONAL_FLOOR))
    scale = np.clip(scale, *OPTIMIZER_SCALE_BOUNDS)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise FloatingPointError("Fisher-Jacobi optimizer scale is invalid")
    return scale


def numerical_boundary_diagnostics(
    codec: ParameterCodec, theta: np.ndarray, data: ScoreData
) -> dict[str, object]:
    """Summarize active numerical boxes and untruncated predictor magnitudes."""
    theta = np.asarray(theta, dtype=float)
    bounds = codec.bounds()
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    tolerance = 1.0e-6 * np.maximum(1.0, upper - lower)
    active = (theta <= lower + tolerance) | (theta >= upper - tolerance)
    block_counts = {
        name: int(np.count_nonzero(active[parameter_slice]))
        for name, parameter_slice in codec.slices.items()
    }
    blocks = codec.unpack(theta)
    if codec.family == "C":
        state_node = np.arange(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int)
        state_group = np.zeros(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int)
    else:
        state_node = np.tile(
            np.arange(core.CONTINUOUS_QUADRATURE_ORDER, dtype=int), 2
        )
        state_group = np.repeat(
            np.arange(2, dtype=int), core.CONTINUOUS_QUADRATURE_ORDER
        )
    group_code = np.where(state_group == 0, -1.0, 1.0)
    support_shift = (
        CONTINUOUS_NODE_BASIS @ blocks["continuous_support"]
    )[state_node]
    if codec.family == "H":
        support_shift = (
            support_shift
            + group_code[:, None] * blocks["group_support"][None, :]
        )
    raw_support = (
        data.design @ blocks["base_support"]
    )[:, None, :] + support_shift[None, :, :]
    maximum_absolute_support_predictor = float(np.max(np.abs(raw_support)))
    maximum_absolute_gate_predictor = 0.0
    if codec.family == "H":
        raw_gate = data.gate_design @ blocks["gate_beta"]
        maximum_absolute_gate_predictor = float(np.max(np.abs(raw_gate)))
    return {
        "active_parameter_bound_count": int(np.count_nonzero(active)),
        "active_parameter_bound_fraction": float(np.mean(active)),
        "active_parameter_bound_count_by_block": block_counts,
        "sigma": math.exp(float(blocks["log_sigma"][0])),
        "sigma_at_numerical_bound": bool(block_counts["log_sigma"] > 0),
        "hard_predictor_clipping": False,
        "maximum_absolute_support_predictor": maximum_absolute_support_predictor,
        "maximum_absolute_gate_predictor": maximum_absolute_gate_predictor,
    }


def has_nonzero_group_signature(codec: ParameterCodec, theta: np.ndarray) -> bool:
    if codec.family != "H":
        raise ValueError("Only H has a group signature")
    blocks = codec.unpack(theta)
    return bool(
        np.any(blocks["group_support"] != 0.0)
        or np.any(blocks["group_value"] != 0.0)
    )


def make_initial_starts(
    codec: ParameterCodec,
    pool: PopulationPool,
    amplitude: float,
    count: int,
    seed: int,
    embedded_h_start: np.ndarray | None = None,
) -> list[np.ndarray]:
    if count < 2:
        raise ValueError("Multiple starts means at least two starts")
    bounds = codec.bounds()
    first_blocks = truth_based_blocks(codec, pool, amplitude, "gaussian")
    if codec.family == "H" and not (
        np.any(first_blocks["group_support"] != 0.0)
        or np.any(first_blocks["group_value"] != 0.0)
    ):
        # The exact nested start is stationary in the group direction at some
        # zero-group fixtures, so reserve a deterministic nonzero-signature slot.
        first_blocks["group_support"][0] = 0.05
        first_blocks["group_value"][1] = -0.05
    # The second start is estimated from the synthetic pool without truth
    # labels, truth signatures, or a planted direction.  It is an initializer,
    # never a frozen coefficient source.
    second_blocks = empirical_blocks(codec, pool)
    truth_start = codec.pack(first_blocks)
    empirical_start = codec.pack(second_blocks)
    row_projection = np.linalg.pinv(pool.single_design) @ pool.single_design

    def perturbed(source: np.ndarray, number: int) -> np.ndarray:
        source_blocks = codec.unpack(source)
        rng = rng_for(seed, 701 if codec.family == "C" else 702, number)
        for name, value in source_blocks.items():
            scale = 0.009 if name == "log_sigma" else 0.035
            delta = rng.normal(0.0, scale, size=value.shape)
            if name in {"base_support", "base_value"}:
                # Do not manufacture arbitrary audit behavior by perturbing a
                # coefficient direction that is null on all development rows.
                delta = row_projection @ delta
            source_blocks[name] = value + delta
        return clip_to_bounds(codec.pack(source_blocks), bounds)

    empirical_start = perturbed(empirical_start, 1)
    starts: list[np.ndarray] = []
    if embedded_h_start is not None:
        if codec.family != "H":
            raise ValueError("A fitted-C embedding can initialize only H")
        embedded_h_start = np.asarray(embedded_h_start, dtype=float)
        if embedded_h_start.shape != (codec.size,) or not np.isfinite(
            embedded_h_start
        ).all():
            raise ValueError("Embedded H start is invalid")
        lower = np.asarray([item[0] for item in bounds], dtype=float)
        upper = np.asarray([item[1] for item in bounds], dtype=float)
        if np.any(embedded_h_start < lower) or np.any(embedded_h_start > upper):
            raise ValueError("Embedded H start lies outside the H bounds")
        starts.append(embedded_h_start.copy())
    starts.extend([truth_start, empirical_start])
    while len(starts) < count:
        source = (truth_start if len(starts) % 2 == 0 else empirical_start).copy()
        starts.append(perturbed(source, len(starts)))
    starts = [
        start.copy()
        if embedded_h_start is not None and index == 0
        else clip_to_bounds(start, bounds)
        for index, start in enumerate(starts[:count])
    ]
    if embedded_h_start is not None:
        # clip_to_bounds leaves an already interior embedding unchanged.
        np.testing.assert_array_equal(starts[0], embedded_h_start)
        if not any(has_nonzero_group_signature(codec, start) for start in starts[1:]):
            raise AssertionError("Final H starts lack a nonzero group signature")
    return starts


def fit_family(
    family: str,
    pool: PopulationPool,
    amplitude: float,
    starts: int,
    max_iterations: int,
    seed: int,
    embedded_c_theta: np.ndarray | None = None,
) -> FamilyFit:
    codec = ParameterCodec(family, pool.single_design.shape[1])
    embedded_h_start: np.ndarray | None = None
    embedded_c_objective: float | None = None
    if embedded_c_theta is not None:
        if family != "H":
            raise ValueError("A fitted-C embedding is valid only for H")
        embedded_h_start = embed_c_theta_in_h(embedded_c_theta, codec.design_rank)
        codec_c = ParameterCodec("C", codec.design_rank)
        c_objective, _ = family_objective_gradient(codec_c, embedded_c_theta, pool.data)
        embedded_c_objective, _ = family_objective_gradient(
            codec, embedded_h_start, pool.data
        )
        np.testing.assert_allclose(
            embedded_c_objective, c_objective, atol=1.0e-12, rtol=0.0
        )
    initial = make_initial_starts(
        codec,
        pool,
        amplitude,
        starts,
        seed,
        embedded_h_start=embedded_h_start,
    )
    solutions: list[OptimizerSolution] = []
    for start in initial:
        optimizer_scale = fisher_jacobi_scale(codec, start, pool.data)
        scaled_start = start / optimizer_scale
        scaled_bounds = [
            (lower / scale, upper / scale)
            for (lower, upper), scale in zip(codec.bounds(), optimizer_scale)
        ]

        def scaled_objective(phi: np.ndarray) -> tuple[float, np.ndarray]:
            value, physical_gradient = family_objective_gradient(
                codec, np.asarray(phi, dtype=float) * optimizer_scale, pool.data
            )
            return value, physical_gradient * optimizer_scale

        result = minimize(
            scaled_objective,
            scaled_start,
            jac=True,
            method="L-BFGS-B",
            bounds=scaled_bounds,
            options={
                "maxiter": int(max_iterations),
                "maxls": 40,
                "ftol": 1.0e-14,
                "gtol": 5.0e-7,
                "maxcor": 20,
            },
        )
        result_theta = np.asarray(result.x, dtype=float) * optimizer_scale
        if (
            not np.isfinite(result.fun)
            or not np.isfinite(result_theta).all()
            or result_theta.shape != (codec.size,)
        ):
            raise FloatingPointError(f"{family} optimizer returned nonfinite fit values")
        final_objective, final_gradient = family_objective_gradient(
            codec, result_theta, pool.data
        )
        if not np.isclose(final_objective, float(result.fun), atol=1.0e-10, rtol=1.0e-10):
            raise AssertionError(f"{family} optimizer objective does not replay")
        projected = projected_gradient(result_theta, final_gradient, codec.bounds())
        projected_norm = float(np.max(np.abs(projected)))
        scaled_projected = projected_gradient(
            np.asarray(result.x, dtype=float),
            final_gradient * optimizer_scale,
            scaled_bounds,
        )
        scaled_projected_norm = float(np.max(np.abs(scaled_projected)))
        projected_block_norms = {
            name: float(np.max(np.abs(projected[parameter_slice])))
            for name, parameter_slice in codec.slices.items()
        }
        at_iteration_cap = int(getattr(result, "nit", -1)) >= int(max_iterations)
        solutions.append(
            OptimizerSolution(
                theta=result_theta.copy(),
                objective=float(result.fun),
                optimizer_success=bool(result.success),
                converged=bool(result.success)
                and not at_iteration_cap
                and projected_norm <= PROJECTED_GRADIENT_TOLERANCE,
                iterations=int(getattr(result, "nit", -1)),
                evaluations=int(getattr(result, "nfev", -1)),
                gradient_infinity_norm=float(np.max(np.abs(final_gradient))),
                projected_gradient_infinity_norm=projected_norm,
                scaled_projected_gradient_infinity_norm=scaled_projected_norm,
                projected_gradient_block_infinity_norms=projected_block_norms,
                optimizer_scale_minimum=float(optimizer_scale.min()),
                optimizer_scale_maximum=float(optimizer_scale.max()),
                numerical_boundary_diagnostics=numerical_boundary_diagnostics(
                    codec, result_theta, pool.data
                ),
                at_iteration_cap=at_iteration_cap,
                message=str(result.message),
            )
        )
    return FamilyFit(
        family=family,
        amplitude=float(amplitude),
        solutions=tuple(solutions),
        parameter_count=codec.size,
        design_rank=pool.single_design.shape[1],
        embedded_c_objective=embedded_c_objective,
    )


def fit_task(payload: tuple[object, ...]) -> FamilyFit:
    (
        layout,
        basis,
        indices,
        amplitude,
        family,
        fit_replicates,
        starts,
        max_iterations,
        seed,
        embedded_c_theta,
    ) = payload
    assert isinstance(layout, core.MetadataLayout)
    assert isinstance(basis, DesignBasis)
    pool = build_population_pool(
        layout,
        basis,
        np.asarray(indices, dtype=int),
        float(amplitude),
        int(fit_replicates),
        int(seed),
        101,
    )
    return fit_family(
        str(family),
        pool,
        float(amplitude),
        int(starts),
        int(max_iterations),
        int(seed),
        embedded_c_theta=(
            None
            if embedded_c_theta is None
            else np.asarray(embedded_c_theta, dtype=float)
        ),
    )


def fit_many(tasks: list[tuple[object, ...]], workers: int) -> list[FamilyFit]:
    if workers == 1:
        return [fit_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fit_task, tasks))


def equal_unit_mean(layout: core.MetadataLayout, indices: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.shape != (len(indices),) or not np.isfinite(values).all():
        raise FloatingPointError("Equal-unit mean received nonfinite or misaligned scores")
    mean = float(np.sum(equal_unit_weights(layout, indices) * values))
    if not np.isfinite(mean):
        raise FloatingPointError("Equal-unit mean is nonfinite")
    return mean


def score_fit(
    fit: FamilyFit,
    solution: OptimizerSolution,
    layout: core.MetadataLayout,
    basis: DesignBasis,
    indices: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    codec = ParameterCodec(fit.family, basis.rank)
    data = make_score_data(layout, basis, indices, outcomes)
    scores = np.asarray(
        family_objective_gradient(codec, solution.theta, data, return_scores=True),
        dtype=float,
    )
    if scores.shape != (len(indices),) or not np.isfinite(scores).all():
        raise FloatingPointError(f"{fit.family} verification scores are nonfinite")
    return scores


def curve_verify_task(payload: tuple[object, ...]) -> tuple[float, list[float]]:
    layout, basis, audit, amplitude, fit, start, stop, seed = payload
    assert isinstance(layout, core.MetadataLayout)
    assert isinstance(basis, DesignBasis)
    assert isinstance(fit, FamilyFit)
    output: list[float] = []
    audit = np.asarray(audit, dtype=int)
    for replicate in range(int(start), int(stop)):
        outcomes, truth = generate_full_synthetic(
            layout, float(amplitude), int(seed), 202, replicate
        )
        audit_outcomes = outcomes[audit]
        audit_truth = core.subset_synthetic_truth(truth, audit)
        fitted = score_fit(fit, fit.best, layout, basis, audit, audit_outcomes)
        exact = truth_scores(audit_outcomes, audit_truth)
        contrast = equal_unit_mean(layout, audit, exact - fitted)
        if not np.isfinite(contrast):
            raise FloatingPointError("Curve verification contrast is nonfinite")
        output.append(contrast)
    return float(amplitude), output


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        raise ValueError("Cannot summarize zero replicates")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise FloatingPointError("Replicate summary contains nonfinite values")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else math.inf
    return mean, se


def pava_non_decreasing(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or len(raw) == 0 or not np.isfinite(raw).all():
        raise FloatingPointError("Isotonic calibration requires finite values")
    blocks: list[list[float | int]] = []
    for index, value in enumerate(raw):
        blocks.append([float(value), 1.0, index, index + 1])
        while len(blocks) >= 2 and float(blocks[-2][0]) > float(blocks[-1][0]):
            right = blocks.pop()
            left = blocks.pop()
            weight = float(left[1]) + float(right[1])
            blocks.append(
                [
                    (float(left[0]) * float(left[1]) + float(right[0]) * float(right[1]))
                    / weight,
                    weight,
                    int(left[2]),
                    int(right[3]),
                ]
            )
    fitted = np.empty_like(raw)
    for value, _, start, stop in blocks:
        fitted[int(start) : int(stop)] = float(value)
    return np.maximum(fitted, 0.0)


def interpolate_target(
    amplitudes: Sequence[float], monotone_gain: Sequence[float], target: float
) -> tuple[float, int, int] | None:
    x = np.asarray(amplitudes, dtype=float)
    y = np.asarray(monotone_gain, dtype=float)
    if target > y[-1]:
        return None
    upper = int(np.searchsorted(y, target, side="left"))
    if upper == 0:
        return float(x[0]), 0, 0
    lower = upper - 1
    if y[upper] <= y[lower] + 1.0e-15:
        return float(x[upper]), lower, upper
    fraction = (target - y[lower]) / (y[upper] - y[lower])
    return float(x[lower] + fraction * (x[upper] - x[lower])), lower, upper


def final_verify_task(payload: tuple[object, ...]) -> list[dict[str, float | int | str]]:
    layout, basis, audit, fixtures, start, stop, seed = payload
    assert isinstance(layout, core.MetadataLayout)
    assert isinstance(basis, DesignBasis)
    audit = np.asarray(audit, dtype=int)
    rows: list[dict[str, float | int | str]] = []
    for replicate in range(int(start), int(stop)):
        for fixture_name, amplitude, fit_c, fit_h in fixtures:
            outcomes, truth = generate_full_synthetic(
                layout, float(amplitude), int(seed), 303, replicate
            )
            audit_outcomes = outcomes[audit]
            audit_truth = core.subset_synthetic_truth(truth, audit)
            truth_score = truth_scores(audit_outcomes, audit_truth)
            c_ranked = fit_c.ranked_solutions
            h_ranked = fit_h.ranked_solutions
            if len(c_ranked) < 2 or len(h_ranked) < 2:
                raise RuntimeError(
                    "Final verification requires two KKT-converged starts per family"
                )
            c_score = score_fit(fit_c, c_ranked[0], layout, basis, audit, audit_outcomes)
            h_score = score_fit(fit_h, h_ranked[0], layout, basis, audit, audit_outcomes)
            c_second = score_fit(
                fit_c, c_ranked[1], layout, basis, audit, audit_outcomes
            )
            h_second = score_fit(
                fit_h, h_ranked[1], layout, basis, audit, audit_outcomes
            )
            row: dict[str, float | int | str] = {
                "fixture": str(fixture_name),
                "replicate": replicate,
                "amplitude": float(amplitude),
                "truth_minus_C": equal_unit_mean(layout, audit, truth_score - c_score),
                "H_minus_C": equal_unit_mean(layout, audit, h_score - c_score),
                "truth_minus_H": equal_unit_mean(layout, audit, truth_score - h_score),
                "C_best_minus_second": equal_unit_mean(
                    layout, audit, c_score - c_second
                ),
                "H_best_minus_second": equal_unit_mean(
                    layout, audit, h_score - h_second
                ),
            }
            numeric = np.asarray(
                [value for key, value in row.items() if key != "fixture"], dtype=float
            )
            if not np.isfinite(numeric).all():
                raise FloatingPointError("Final verification row is nonfinite")
            rows.append(row)
    return rows


def summarize_optimizer(fit: FamilyFit) -> dict[str, object]:
    ranked = fit.ranked_solutions
    return {
        "family": fit.family,
        "amplitude": fit.amplitude,
        "parameter_count": fit.parameter_count,
        "design_rank": fit.design_rank,
        "starts": len(fit.solutions),
        "converged_starts": len(fit.converged_solutions),
        "best_training_negative_mean_log_density": fit.best.objective,
        "best_iterations": fit.best.iterations,
        "best_evaluations": fit.best.evaluations,
        "best_gradient_infinity_norm": fit.best.gradient_infinity_norm,
        "best_projected_gradient_infinity_norm": fit.best.projected_gradient_infinity_norm,
        "best_scaled_projected_gradient_infinity_norm": (
            fit.best.scaled_projected_gradient_infinity_norm
        ),
        "best_projected_gradient_block_infinity_norms": (
            fit.best.projected_gradient_block_infinity_norms
        ),
        "best_optimizer_scale_range": [
            fit.best.optimizer_scale_minimum,
            fit.best.optimizer_scale_maximum,
        ],
        "best_numerical_boundary_diagnostics": (
            fit.best.numerical_boundary_diagnostics
        ),
        "projected_gradient_KKT_tolerance": PROJECTED_GRADIENT_TOLERANCE,
        "best_at_iteration_cap": fit.best.at_iteration_cap,
        "best_two_available": len(ranked) >= 2,
        "embedded_C_training_negative_mean_log_density": fit.embedded_c_objective,
        "best_H_objective_no_worse_than_embedded_C": (
            None
            if fit.embedded_c_objective is None
            else fit.best.objective <= fit.embedded_c_objective
        ),
    }


FINAL_CONTRAST_COLUMNS = (
    "truth_minus_C",
    "H_minus_C",
    "truth_minus_H",
    "C_best_minus_second",
    "H_best_minus_second",
)


def assert_curve_replicate_completeness(
    values_by_amplitude: dict[float, list[float]],
    amplitudes: Sequence[float],
    expected_replicates: int,
) -> None:
    if set(values_by_amplitude) != set(amplitudes):
        raise AssertionError("Curve verification amplitude set is incomplete")
    for amplitude in amplitudes:
        values = np.asarray(values_by_amplitude[float(amplitude)], dtype=float)
        if values.shape != (expected_replicates,):
            raise AssertionError(
                f"Curve amplitude {amplitude} has {len(values)} of "
                f"{expected_replicates} scheduled replicates"
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"Curve amplitude {amplitude} contains a nonfinite replicate"
            )


def assert_final_replicate_completeness(
    frame: pd.DataFrame,
    fixtures: Sequence[str],
    expected_replicates: int,
) -> None:
    required = {"fixture", "replicate", "amplitude", *FINAL_CONTRAST_COLUMNS}
    if not required.issubset(frame.columns):
        raise AssertionError("Final verification columns are incomplete")
    if len(frame) != len(fixtures) * expected_replicates:
        raise AssertionError("Final verification row count is incomplete")
    if not np.isfinite(
        frame[["replicate", "amplitude", *FINAL_CONTRAST_COLUMNS]].to_numpy(
            dtype=float
        )
    ).all():
        raise FloatingPointError("Final verification contains a nonfinite value")
    expected = np.arange(expected_replicates, dtype=int)
    for fixture in fixtures:
        local = frame.loc[frame["fixture"].eq(fixture), "replicate"].to_numpy(
            dtype=int
        )
        if local.shape != (expected_replicates,) or not np.array_equal(
            np.sort(local), expected
        ):
            raise AssertionError(
                f"Fixture {fixture} does not contain each scheduled replicate exactly once"
            )
    unexpected = set(frame["fixture"].astype(str)) - set(fixtures)
    if unexpected:
        raise AssertionError("Final verification contains an unexpected fixture")


def contrast_summary(
    frame: pd.DataFrame,
    fixture: str,
    column: str,
    expected_replicates: int,
) -> dict[str, float | int]:
    values = frame.loc[frame["fixture"].eq(fixture), column].to_numpy(dtype=float)
    if values.shape != (expected_replicates,):
        raise AssertionError(f"Contrast {fixture}/{column} is replicate-incomplete")
    mean, se = mean_se(values)
    return {"replicates": int(len(values)), "mean": mean, "mcse": se}


def max_monitored_mcse(frame: pd.DataFrame, expected_replicates: int) -> float:
    values = [
        float(
            contrast_summary(frame, fixture, column, expected_replicates)["mcse"]
        )
        for fixture in ("zero_group", "selected_strong")
        for column in FINAL_CONTRAST_COLUMNS
    ]
    if not np.isfinite(values).all():
        raise FloatingPointError("A monitored Monte Carlo SE is nonfinite")
    return max(values)


def one_sided_upper(mean: float, se: float, replicates: int) -> float:
    if replicates < 2 or not np.isfinite(se):
        return math.inf
    return float(mean + student_t.ppf(0.95, replicates - 1) * se)


def write_outputs(
    output_dir: Path,
    curve_rows: list[dict[str, object]],
    final_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(output_dir / "full_family_calibration_curve.csv", index=False)
    if final_rows:
        pd.DataFrame(final_rows).to_csv(
            output_dir / "full_family_calibration_replicates.csv", index=False
        )
    (output_dir / "full_family_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# EP11 conditional-A8 full-family finite-pool diagnostic",
        "",
        f"**Status:** `{summary['status']}`.",
        "",
        "This run used only synthetic 56-coordinate outcomes on the outcome-blind",
        "metadata layout. C and H were optimized separately; no real projection",
        "outcome was read or constructed. Both use the fixed seven-node standard-",
        "normal quadrature of the implemented smoke grammar. Finite-pool fitting and",
        "independent verification both condition on the same fixed provisional A8",
        "metadata support. This tool does not establish a population projection; that",
        "requires the increasing group-stratified SAA runner. D12-to-A8 transport is",
        "reserved for Panel C. This diagnostic does not authorize outcome-bearing search.",
        "The family uses an equal-unit weighted-SVD P basis and 55 Helmert value",
        "contrasts. Its coefficient boxes are numerical bounds on those identified",
        "coordinates, not the former raw-coordinate box.",
        "",
        "## Diagnostic result",
        "",
    ]
    selected = summary.get("selected_amplitude")
    if selected is None:
        lines.append(str(summary["calibration_failure_reason"]))
    else:
        lines.extend(
            [
                f"- Selected generator amplitude: {float(selected):.6f}",
                f"- Final verification replicates: {summary['final_verification']['replicates']}",
                f"- Final maximum monitored MCSE: {summary['final_verification']['maximum_mcse']:.6f}",
                "",
                "## Acceptance gates",
                "",
            ]
        )
        for name, gate in summary["acceptance"].items():
            lines.append(f"- `{name}`: `{gate}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This calibrates the bounded implemented smoke family only. It does not",
            "qualify the richer registered production grammar, establish animal",
            "independence, lock real-data roles, or support a biological group claim.",
            "",
        ]
    )
    (output_dir / "full_family_calibration_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def explicit_h_2_by_7_enumeration(
    codec: ParameterCodec,
    theta: np.ndarray,
    data: ScoreData,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent scalar-loop reference for all two-by-seven H states."""
    if codec.family != "H":
        raise ValueError("The explicit two-by-seven enumeration is H-only")
    blocks = codec.unpack(theta)
    sigma = math.exp(float(blocks["log_sigma"][0]))
    variance = sigma**2
    scores = np.empty(len(data.active), dtype=float)
    responsibilities = np.empty(
        (len(data.active), 2 * core.CONTINUOUS_QUADRATURE_ORDER), dtype=float
    )
    for row in range(len(data.active)):
        gate_eta = float(data.gate_design[row] @ blocks["gate_beta"])
        component_values: list[float] = []
        for group in range(2):
            group_code = -1.0 if group == 0 else 1.0
            log_group = float(
                -np.logaddexp(0.0, gate_eta)
                if group == 0
                else -np.logaddexp(0.0, -gate_eta)
            )
            for node in range(core.CONTINUOUS_QUADRATURE_ORDER):
                continuous_support = (
                    CONTINUOUS_NODE_BASIS[node] @ blocks["continuous_support"]
                )
                continuous_value = (
                    expand_value_coordinates(
                        CONTINUOUS_NODE_BASIS[node] @ blocks["continuous_value"]
                    )
                )
                support_logits = (
                    data.design[row] @ blocks["base_support"]
                    + continuous_support
                    + group_code * blocks["group_support"]
                )
                log_on = -np.logaddexp(0.0, -support_logits)
                log_off = -np.logaddexp(0.0, support_logits)
                unconditioned = float(
                    np.where(data.active[row], log_on, log_off).sum()
                )
                support_log_density = unconditioned - float(
                    log_nonempty_from_logits(support_logits)
                )
                mean = (
                    expand_value_coordinates(
                        data.design[row] @ blocks["base_value"]
                    )
                    + continuous_value
                    + group_code * expand_value_coordinates(blocks["group_value"])
                )
                difference = (
                    (data.log_values[row] - mean) * data.active[row]
                )
                active_count = int(data.active[row].sum())
                face_mean = float(difference.sum() / active_count)
                residual = (difference - face_mean) * data.active[row]
                residual_sum = float(np.square(residual).sum())
                value_log_density = -0.5 * (
                    data.dimension[row] * math.log(2.0 * math.pi * variance)
                    + residual_sum / variance
                ) + data.log_jacobian[row]
                component_values.append(
                    log_group
                    + math.log(float(GAUSSIAN_NODE_WEIGHTS[node]))
                    + support_log_density
                    + float(value_log_density)
                )
        components = np.asarray(component_values, dtype=float)
        scores[row] = float(logsumexp(components))
        responsibilities[row] = np.exp(components - scores[row])
    if not np.isfinite(scores).all() or not np.isfinite(responsibilities).all():
        raise FloatingPointError("Explicit H enumeration is nonfinite")
    return scores, responsibilities


def executable_self_test(layout: core.MetadataLayout) -> dict[str, object]:
    # The conditional-nonempty support likelihood must remain informative even
    # when every Bernoulli logit is far beyond ordinary floating-point expit
    # resolution.  This is the regime that hard predictor clipping previously
    # obscured.
    far_off = np.full(core.N_COORDINATES, -1000.0)
    far_off_log_nonempty = float(log_nonempty_from_logits(far_off))
    np.testing.assert_allclose(
        far_off_log_nonempty,
        -1000.0 + math.log(core.N_COORDINATES),
        atol=1.0e-12,
        rtol=0.0,
    )
    far_on = np.full(core.N_COORDINATES, 1000.0)
    np.testing.assert_allclose(
        float(log_nonempty_from_logits(far_on)), 0.0, atol=0.0, rtol=0.0
    )
    moderate_logits = np.asarray([-3.0, -0.5, 0.25, 2.0])
    moderate_reference = math.log1p(
        -float(np.prod(expit(-moderate_logits)))
    )
    np.testing.assert_allclose(
        float(log_nonempty_from_logits(moderate_logits)),
        moderate_reference,
        atol=2.0e-15,
        rtol=0.0,
    )
    epsilon_extreme = 1.0e-4
    far_off_plus = far_off.copy()
    far_off_minus = far_off.copy()
    far_off_plus[0] += epsilon_extreme
    far_off_minus[0] -= epsilon_extreme
    numerical_log_normalizer_gradient = (
        float(log_nonempty_from_logits(far_off_plus))
        - float(log_nonempty_from_logits(far_off_minus))
    ) / (2.0 * epsilon_extreme)
    far_off_log_on = -np.logaddexp(0.0, -far_off[0])
    far_off_log_empty = float(np.sum(-np.logaddexp(0.0, far_off)))
    stable_log_normalizer_gradient = math.exp(
        far_off_log_on + far_off_log_empty - far_off_log_nonempty
    )
    np.testing.assert_allclose(
        numerical_log_normalizer_gradient,
        stable_log_normalizer_gradient,
        atol=5.0e-10,
        rtol=0.0,
    )
    singleton_mask = np.zeros(core.N_COORDINATES, dtype=bool)
    singleton_mask[0] = True
    np.testing.assert_allclose(
        core.active_set_log_probability(singleton_mask, far_off),
        -math.log(core.N_COORDINATES),
        atol=1.0e-12,
        rtol=0.0,
    )

    roles = panel_b_fixed_roles(layout)
    audit_weights = equal_unit_weights(layout, roles.audit)
    basis = DesignBasis.from_support(
        layout.p_design[roles.audit], audit_weights
    )
    transformed_audit_design = basis.apply(layout.p_design[roles.audit])
    np.testing.assert_allclose(
        transformed_audit_design.T
        @ (audit_weights[:, None] * transformed_audit_design),
        np.eye(basis.rank),
        atol=5.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        VALUE_CONTRAST_BASIS.T @ VALUE_CONTRAST_BASIS,
        np.eye(N_VALUE_CONTRASTS),
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        VALUE_CONTRAST_BASIS.sum(axis=0), 0.0, atol=1.0e-14, rtol=0.0
    )
    development_design = layout.p_design[roles.development]
    _, development_singular, development_right = np.linalg.svd(
        development_design, full_matrices=True
    )
    development_rank = int(
        np.sum(development_singular > development_singular[0] * 1.0e-10)
    )
    development_null = development_right[development_rank:].T
    audit_null_extension = layout.p_design[roles.audit] @ development_null
    if (
        development_null.shape[1] == 0
        or np.max(np.abs(audit_null_extension)) <= 0.5
    ):
        raise AssertionError(
            "D12-null directions unexpectedly became identified on A8"
        )
    all_indices = np.arange(len(layout.frame), dtype=int)

    # Exercise several genuinely different 20-unit scopes, including the exact
    # fixed D12+A8 core used by completed Panel B.  The generator itself always
    # constructs the full layout and only then subsets it.
    all_units = np.asarray(sorted(set(layout.units.tolist())), dtype=object)
    assigned_core = np.sort(np.concatenate([roles.development, roles.audit]))
    assigned_units = tuple(sorted(set(layout.units[assigned_core].tolist())))
    if len(assigned_units) != 20:
        raise AssertionError("The fixed assigned core does not contain 20 units")
    unit_subsets: list[tuple[str, ...]] = [assigned_units]
    seen_subsets = {assigned_units}
    subset_rng = np.random.default_rng(20260927)
    while len(unit_subsets) < 4:
        candidate = tuple(
            sorted(subset_rng.choice(all_units, size=20, replace=False).tolist())
        )
        if candidate not in seen_subsets:
            seen_subsets.add(candidate)
            unit_subsets.append(candidate)
    index_subsets = [
        np.flatnonzero(np.isin(layout.units, np.asarray(units, dtype=object)))
        for units in unit_subsets
    ]
    if any(
        len(set(layout.units[indices].tolist())) != 20 or len(indices) <= 20
        for indices in index_subsets
    ):
        raise AssertionError("A subset-invariance probe is not a nontrivial 20-unit set")
    for scenario in (
        "reusable_groups",
        "groups_plus_continuum",
        "animal_confounded_continuum",
    ):
        full_feature = core.continuous_truth_features(
            layout, all_indices, scenario, np.random.default_rng(819)
        )
        full_labels, full_probability = core.synthetic_groups(
            layout, all_indices, scenario, np.random.default_rng(991)
        )
        for subset_indices in index_subsets:
            subset_feature = core.continuous_truth_features(
                layout, subset_indices, scenario, np.random.default_rng(819)
            )
            np.testing.assert_allclose(
                subset_feature,
                full_feature[subset_indices],
                atol=1.0e-14,
                rtol=0.0,
            )
            subset_labels, subset_probability = core.synthetic_groups(
                layout, subset_indices, scenario, np.random.default_rng(991)
            )
            np.testing.assert_array_equal(
                subset_labels, full_labels[subset_indices]
            )
            np.testing.assert_array_equal(
                subset_probability, full_probability[subset_indices]
            )

    outcomes_zero, truth_zero = generate_full_synthetic(layout, 0.0, 41, 9, 0)
    outcomes_strong, truth_strong = generate_full_synthetic(layout, 0.7, 41, 9, 0)
    outcomes_repeat, truth_repeat = generate_full_synthetic(layout, 0.7, 41, 9, 0)
    np.testing.assert_array_equal(outcomes_strong, outcomes_repeat)
    np.testing.assert_array_equal(truth_strong.groups, truth_repeat.groups)
    np.testing.assert_array_equal(truth_zero.groups, truth_strong.groups)
    np.testing.assert_array_equal(
        truth_zero.group_probabilities, truth_strong.group_probabilities
    )
    np.testing.assert_array_equal(truth_zero.base_support_logits, truth_strong.base_support_logits)
    np.testing.assert_array_equal(truth_zero.base_value_mean, truth_strong.base_value_mean)

    audit = roles.audit
    vectorized_truth = truth_scores(
        outcomes_strong[audit], core.subset_synthetic_truth(truth_strong, audit)
    )
    reference_truth = core.oracle_mixed_support_scores(
        outcomes_strong[audit],
        core.subset_synthetic_truth(truth_strong, audit),
        include_groups=True,
    )
    np.testing.assert_allclose(vectorized_truth, reference_truth, atol=1.0e-10, rtol=0.0)
    zero_h = truth_scores(outcomes_zero[audit], core.subset_synthetic_truth(truth_zero, audit))
    zero_c = core.oracle_mixed_support_scores(
        outcomes_zero[audit],
        core.subset_synthetic_truth(truth_zero, audit),
        include_groups=False,
    )
    np.testing.assert_allclose(zero_h, zero_c, atol=1.0e-10, rtol=0.0)

    # Pack/unpack, analytic gradient, and exact H-to-C nesting.
    tiny_indices = roles.audit[: min(10, len(roles.audit))]
    tiny_outcomes = outcomes_strong[tiny_indices]
    tiny_data = make_score_data(layout, basis, tiny_indices, tiny_outcomes)
    codec_c = ParameterCodec("C", basis.rank)
    codec_h = ParameterCodec("H", basis.rank)
    if codec_c.size != 3109 or codec_h.size != 3224:
        raise AssertionError(
            f"Unexpected complete-family sizes: C={codec_c.size}, H={codec_h.size}"
        )
    pool = PopulationPool(
        data=tiny_data,
        single_design=basis.apply(layout.p_design[tiny_indices]),
        truth=core.subset_synthetic_truth(truth_strong, tiny_indices),
        replicates=1,
    )
    theta_c = codec_c.pack(
        truth_based_blocks(codec_c, pool, 0.7, "adjacent_node_scale")
    )
    repacked = codec_c.pack(codec_c.unpack(theta_c))
    np.testing.assert_array_equal(theta_c, repacked)
    objective, gradient = family_objective_gradient(codec_c, theta_c, tiny_data)
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise AssertionError("C objective or gradient is nonfinite")
    check_indices = np.linspace(0, codec_c.size - 1, 12, dtype=int)
    epsilon = 1.0e-6
    for index in check_indices:
        plus = theta_c.copy()
        minus = theta_c.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = family_objective_gradient(codec_c, plus, tiny_data)[0]
        minus_value = family_objective_gradient(codec_c, minus, tiny_data)[0]
        numerical = (plus_value - minus_value) / (2.0 * epsilon)
        np.testing.assert_allclose(numerical, gradient[index], atol=2.5e-5, rtol=8.0e-4)
    c_optimizer_scale = fisher_jacobi_scale(codec_c, theta_c, tiny_data)
    if (
        c_optimizer_scale.shape != theta_c.shape
        or np.any(c_optimizer_scale < OPTIMIZER_SCALE_BOUNDS[0])
        or np.any(c_optimizer_scale > OPTIMIZER_SCALE_BOUNDS[1])
    ):
        raise AssertionError("C Fisher-Jacobi scale is invalid")
    phi_c = theta_c / c_optimizer_scale
    scaled_gradient = gradient * c_optimizer_scale
    for index in check_indices[::3]:
        plus = phi_c.copy()
        minus = phi_c.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = family_objective_gradient(
            codec_c, plus * c_optimizer_scale, tiny_data
        )[0]
        minus_value = family_objective_gradient(
            codec_c, minus * c_optimizer_scale, tiny_data
        )[0]
        numerical = (plus_value - minus_value) / (2.0 * epsilon)
        np.testing.assert_allclose(
            numerical, scaled_gradient[index], atol=2.5e-5, rtol=8.0e-4
        )

    theta_h_gradient = codec_h.pack(
        truth_based_blocks(codec_h, pool, 0.7, "gaussian")
    )
    h_objective, h_gradient = family_objective_gradient(
        codec_h, theta_h_gradient, tiny_data
    )
    if not np.isfinite(h_objective) or not np.isfinite(h_gradient).all():
        raise AssertionError("H objective or gradient is nonfinite")
    h_check_indices = [
        codec_h.slices[name].start
        + (codec_h.slices[name].stop - codec_h.slices[name].start) // 2
        for name in codec_h.shapes
    ]
    for index in h_check_indices:
        plus = theta_h_gradient.copy()
        minus = theta_h_gradient.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = family_objective_gradient(codec_h, plus, tiny_data)[0]
        minus_value = family_objective_gradient(codec_h, minus, tiny_data)[0]
        numerical = (plus_value - minus_value) / (2.0 * epsilon)
        np.testing.assert_allclose(
            numerical, h_gradient[index], atol=3.0e-5, rtol=1.0e-3
        )
    h_optimizer_scale = fisher_jacobi_scale(
        codec_h, theta_h_gradient, tiny_data
    )
    phi_h = theta_h_gradient / h_optimizer_scale
    h_scaled_gradient = h_gradient * h_optimizer_scale
    for index in h_check_indices:
        plus = phi_h.copy()
        minus = phi_h.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = family_objective_gradient(
            codec_h, plus * h_optimizer_scale, tiny_data
        )[0]
        minus_value = family_objective_gradient(
            codec_h, minus * h_optimizer_scale, tiny_data
        )[0]
        numerical = (plus_value - minus_value) / (2.0 * epsilon)
        np.testing.assert_allclose(
            numerical, h_scaled_gradient[index], atol=3.0e-5, rtol=1.0e-3
        )
    parameterized_truth_h = family_objective_gradient(
        codec_h, theta_h_gradient, tiny_data, return_scores=True
    )
    exact_tiny_h = truth_scores(tiny_outcomes, pool.truth)
    np.testing.assert_allclose(
        parameterized_truth_h, exact_tiny_h, atol=1.0e-9, rtol=0.0
    )

    # Exercise all 2 x 7 states with both continuous and group support/value
    # shifts nonzero.  This catches state-ordering defects hidden by a
    # degenerate continuous or group block.
    nondegenerate_blocks = {
        name: value.copy()
        for name, value in codec_h.unpack(theta_h_gradient).items()
    }
    coordinate = np.arange(core.N_COORDINATES, dtype=float)
    nondegenerate_blocks["continuous_support"][0] = 0.07 * np.sin(
        0.19 * coordinate
    )
    nondegenerate_blocks["continuous_support"][1] = 0.05 * np.cos(
        0.13 * coordinate + 0.2
    )
    nondegenerate_blocks["continuous_value"][0] = project_value_coordinates(
        0.06 * np.cos(0.17 * coordinate)
    )
    nondegenerate_blocks["continuous_value"][1] = project_value_coordinates(
        0.04 * np.sin(0.11 * coordinate + 0.4)
    )
    if any(
        not np.any(nondegenerate_blocks[name] != 0.0)
        for name in (
            "continuous_support",
            "continuous_value",
            "group_support",
            "group_value",
        )
    ):
        raise AssertionError("Nondegenerate H enumeration probe has a zero shift block")
    nondegenerate_theta = codec_h.pack(nondegenerate_blocks)
    vectorized_h, vectorized_responsibilities = family_objective_gradient(
        codec_h,
        nondegenerate_theta,
        tiny_data,
        return_scores=True,
        return_responsibilities=True,
    )
    enumerated_h, enumerated_responsibilities = explicit_h_2_by_7_enumeration(
        codec_h, nondegenerate_theta, tiny_data
    )
    np.testing.assert_allclose(vectorized_h, enumerated_h, atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(
        vectorized_responsibilities,
        enumerated_responsibilities,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        vectorized_responsibilities.sum(axis=1),
        1.0,
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        enumerated_responsibilities.sum(axis=1),
        1.0,
        atol=1.0e-12,
        rtol=0.0,
    )
    latent_flipped_blocks = {
        name: value.copy() for name, value in nondegenerate_blocks.items()
    }
    latent_flipped_blocks["continuous_support"][0] *= -1.0
    latent_flipped_blocks["continuous_value"][0] *= -1.0
    latent_flipped_scores = family_objective_gradient(
        codec_h,
        codec_h.pack(latent_flipped_blocks),
        tiny_data,
        return_scores=True,
    )
    np.testing.assert_allclose(
        vectorized_h, latent_flipped_scores, atol=1.0e-10, rtol=0.0
    )
    label_flipped_blocks = {
        name: value.copy() for name, value in nondegenerate_blocks.items()
    }
    label_flipped_blocks["group_support"] *= -1.0
    label_flipped_blocks["group_value"] *= -1.0
    label_flipped_blocks["gate_beta"] *= -1.0
    label_flipped_scores = family_objective_gradient(
        codec_h,
        codec_h.pack(label_flipped_blocks),
        tiny_data,
        return_scores=True,
    )
    np.testing.assert_allclose(
        vectorized_h, label_flipped_scores, atol=1.0e-10, rtol=0.0
    )
    reference_orbit_scale = fisher_jacobi_scale(
        codec_h, nondegenerate_theta, tiny_data
    )
    np.testing.assert_allclose(
        reference_orbit_scale,
        fisher_jacobi_scale(
            codec_h, codec_h.pack(latent_flipped_blocks), tiny_data
        ),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reference_orbit_scale,
        fisher_jacobi_scale(
            codec_h, codec_h.pack(label_flipped_blocks), tiny_data
        ),
        atol=1.0e-12,
        rtol=0.0,
    )

    blocks_c = codec_c.unpack(theta_c)
    reference_c_model = core.MixedSupportModel(
        name="C_self_test",
        design_name="P",
        support_beta=blocks_c["base_support"],
        value_beta=expand_value_coordinates(blocks_c["base_value"]),
        state_support_shift=CONTINUOUS_NODE_BASIS @ blocks_c["continuous_support"],
        state_value_shift=expand_value_coordinates(
            CONTINUOUS_NODE_BASIS @ blocks_c["continuous_value"]
        ),
        state_weights=GAUSSIAN_NODE_WEIGHTS,
        sigma=math.exp(float(blocks_c["log_sigma"][0])),
    )
    reference_c_score, _ = core.score_model(
        reference_c_model, tiny_data.design, tiny_outcomes
    )
    vectorized_c_score = family_objective_gradient(
        codec_c, theta_c, tiny_data, return_scores=True
    )
    np.testing.assert_allclose(
        vectorized_c_score, reference_c_score, atol=1.0e-10, rtol=0.0
    )

    # Verify that quotienting the raw 39-term P design and the value-location
    # gauge preserves both scores and state responsibilities on fixed A8.
    equivalence_rng = np.random.default_rng(20260930)
    raw_support_beta = equivalence_rng.normal(
        0.0, 0.03, size=(basis.raw_term_count, core.N_COORDINATES)
    )
    raw_value_beta = equivalence_rng.normal(
        0.0, 0.03, size=(basis.raw_term_count, core.N_COORDINATES)
    )
    raw_continuous_support = equivalence_rng.normal(
        0.0, 0.04, size=(2, core.N_COORDINATES)
    )
    raw_continuous_value = equivalence_rng.normal(
        0.0, 0.04, size=(2, core.N_COORDINATES)
    )
    full_audit_raw_design = layout.p_design[roles.audit]
    reduced_audit_design = basis.apply(full_audit_raw_design)
    raw_to_reduced_support = reduced_audit_design.T @ (
        audit_weights[:, None] * (full_audit_raw_design @ raw_support_beta)
    )
    raw_to_reduced_value_full = reduced_audit_design.T @ (
        audit_weights[:, None] * (full_audit_raw_design @ raw_value_beta)
    )
    equivalence_blocks = codec_c.blank()
    equivalence_blocks["base_support"] = raw_to_reduced_support
    equivalence_blocks["base_value"] = project_value_coordinates(
        raw_to_reduced_value_full
    )
    equivalence_blocks["continuous_support"] = raw_continuous_support
    equivalence_blocks["continuous_value"] = project_value_coordinates(
        raw_continuous_value
    )
    equivalence_blocks["log_sigma"][0] = math.log(0.47)
    reduced_scores, reduced_responsibilities = family_objective_gradient(
        codec_c,
        codec_c.pack(equivalence_blocks),
        tiny_data,
        return_scores=True,
        return_responsibilities=True,
    )
    raw_model = core.MixedSupportModel(
        name="raw_C_equivalence",
        design_name="P",
        support_beta=raw_support_beta,
        value_beta=raw_value_beta,
        state_support_shift=CONTINUOUS_NODE_BASIS @ raw_continuous_support,
        state_value_shift=CONTINUOUS_NODE_BASIS @ raw_continuous_value,
        state_weights=GAUSSIAN_NODE_WEIGHTS,
        sigma=0.47,
    )
    raw_scores, raw_components = core.score_model(
        raw_model,
        layout.p_design[tiny_indices],
        tiny_outcomes,
        return_components=True,
    )
    assert raw_components is not None
    raw_responsibilities = np.exp(raw_components - raw_scores[:, None])
    np.testing.assert_allclose(
        reduced_scores, raw_scores, atol=2.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        reduced_responsibilities,
        raw_responsibilities,
        atol=2.0e-12,
        rtol=0.0,
    )
    c_score = family_objective_gradient(
        codec_c, theta_c, tiny_data, return_scores=True
    )
    embedded_theta_h = embed_c_theta_in_h(theta_c, basis.rank)
    blocks_h = codec_h.unpack(embedded_theta_h)
    for name in codec_c.shapes:
        np.testing.assert_array_equal(blocks_h[name], blocks_c[name])
    np.testing.assert_array_equal(blocks_h["group_support"], 0.0)
    np.testing.assert_array_equal(blocks_h["group_value"], 0.0)
    np.testing.assert_array_equal(blocks_h["gate_beta"], 0.0)
    embedded_score = family_objective_gradient(
        codec_h, embedded_theta_h, tiny_data, return_scores=True
    )
    np.testing.assert_allclose(c_score, embedded_score, atol=1.0e-10, rtol=0.0)
    blocks_h["gate_beta"] = np.array([0.4, -0.2, 0.1, 0.3])
    theta_h = codec_h.pack(blocks_h)
    h_score = family_objective_gradient(codec_h, theta_h, tiny_data, return_scores=True)
    np.testing.assert_allclose(c_score, h_score, atol=1.0e-10, rtol=0.0)
    _, nested_gradient = family_objective_gradient(codec_h, theta_h, tiny_data)
    np.testing.assert_allclose(
        nested_gradient[codec_h.slices["gate_beta"]], 0.0, atol=1.0e-12, rtol=0.0
    )
    if np.shares_memory(theta_c, theta_h):
        raise AssertionError("C and H parameter vectors unexpectedly share memory")

    kkt_probe = projected_gradient(
        np.asarray([0.0, 0.5, 1.0, 0.0, 1.0]),
        np.asarray([2.0, -0.3, -2.0, -2.0, 2.0]),
        [(0.0, 1.0)] * 5,
    )
    np.testing.assert_array_equal(kkt_probe, np.asarray([0.0, -0.3, 0.0, -2.0, 2.0]))

    return {
        "status": "pass",
        "real_projection_outcomes_read": False,
        "coordinates": core.N_COORDINATES,
        "role_counts": {"development": 12, "audit": 8, "unassigned": 16},
        "P_design_term_count": basis.raw_term_count,
        "A8_estimable_design_rank": basis.estimable_rank,
        "D12_estimable_design_rank": development_rank,
        "D12_null_extension_maximum_on_A8": float(
            np.max(np.abs(audit_null_extension))
        ),
        "twenty_unit_invariance_subsets": len(index_subsets),
        "checks": {
            "full_layout_generator_covariates_are_subset_invariant_across_four_20_unit_scopes": True,
            "fixed_Panel_B_assigned_core_is_one_invariance_scope": True,
            "common_random_generator_draws_are_amplitude_aligned": True,
            "vectorized_truth_equals_reference_oracle_cellwise": True,
            "vectorized_C_equals_reference_mixed_support_score_cellwise": True,
            "parameterized_full_H_contains_generator_density_on_fit_support": True,
            "zero_amplitude_truth_H_equals_truth_C_cellwise": True,
            "analytic_gradient_matches_finite_difference": True,
            "Fisher_Jacobi_scaled_gradient_matches_finite_difference": True,
            "H_Fisher_Jacobi_scaled_gradient_matches_finite_difference": True,
            "weighted_P_design_is_orthonormal_and_rank_reduced": True,
            "value_coordinates_use_an_orthonormal_sum_zero_basis": True,
            "raw_and_identified_C_scores_and_responsibilities_match": True,
            "D12_nullspace_is_explicitly_not_transportable_to_A8": True,
            "complete_C_H_identifiable_parameter_counts_are_3109_and_3224": True,
            "zero_group_parameterized_H_equals_C_cellwise": True,
            "fitted_C_embedding_helper_preserves_every_common_block": True,
            "zero_signature_H_has_zero_gate_gradient": True,
            "nondegenerate_H_matches_explicit_2_by_7_enumeration": True,
            "H_responsibilities_sum_to_one": True,
            "latent_sign_and_group_label_orbits_preserve_density": True,
            "latent_sign_and_group_label_orbits_preserve_optimizer_scale": True,
            "bounds_aware_projected_gradient_KKT_rule_is_exercised": True,
            "C_and_H_parameter_storage_is_independent": True,
            "conditional_nonempty_normalizer_is_stable_at_extreme_logits": True,
            "extreme_logit_normalizer_gradient_is_stable_without_clipping": True,
            "reference_active_set_score_is_stable_without_clipping": True,
        },
        "outcome_bearing_search_authorized": False,
    }


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or values[0] != 0.0 or tuple(sorted(set(values))) != values:
        raise ValueError("Amplitude grid must be unique, increasing, and start at zero")
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Amplitude grid must be finite and nonnegative")
    return values


def chunk_ranges(start: int, stop: int, workers: int) -> list[tuple[int, int]]:
    points = np.linspace(start, stop, min(workers, stop - start) + 1, dtype=int)
    return [(int(a), int(b)) for a, b in zip(points[:-1], points[1:]) if b > a]


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
    parser.add_argument("--amplitude-grid", default=",".join(map(str, DEFAULT_AMPLITUDE_GRID)))
    parser.add_argument("--fit-replicates", type=int, default=32)
    parser.add_argument("--curve-starts", type=int, default=2)
    parser.add_argument("--final-starts", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--maximum-verification-replicates", type=int, default=256)
    parser.add_argument("--target-mcse", type=float, default=0.005)
    parser.add_argument("--target-gain", type=float, default=TARGET_GAIN)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    core.validate_vocabulary(args.vocabulary)
    layout = core.load_metadata(args.inventory)
    if args.self_test:
        print(json.dumps(executable_self_test(layout), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    amplitudes = parse_grid(args.amplitude_grid)
    if args.fit_replicates < 2:
        raise ValueError("Finite-pool fitting needs at least two synthetic replicates")
    if args.curve_starts < 2 or args.final_starts < 2:
        raise ValueError("Every fitted family needs at least two starts")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    if args.batch_size < 2 or args.maximum_verification_replicates < args.batch_size:
        raise ValueError("Invalid sequential verification limits")
    if args.maximum_verification_replicates % args.batch_size:
        raise ValueError("Maximum verification replicates must be a batch multiple")
    if (
        not np.isfinite(args.target_mcse)
        or not np.isfinite(args.target_gain)
        or args.target_mcse <= 0.0
        or args.target_gain <= 0.0
    ):
        raise ValueError("Targets must be finite and positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", args.workers))
    if args.workers > allocated:
        raise ValueError("--workers exceeds SLURM_CPUS_PER_TASK")

    roles = panel_b_fixed_roles(layout)
    panel_a_indices = roles.audit
    basis = DesignBasis.from_support(
        layout.p_design[panel_a_indices],
        equal_unit_weights(layout, panel_a_indices),
    )

    print("Fitting the full C family along the bounded amplitude grid", file=sys.stderr, flush=True)
    curve_tasks = [
        (
            layout,
            basis,
            panel_a_indices,
            amplitude,
            "C",
            args.fit_replicates,
            args.curve_starts,
            args.max_iterations,
            args.seed,
            None,
        )
        for amplitude in amplitudes
    ]
    curve_fits = fit_many(curve_tasks, min(args.workers, len(curve_tasks)))
    curve_fit_by_amplitude = {fit.amplitude: fit for fit in curve_fits}

    curve_values: dict[float, list[float]] = {amplitude: [] for amplitude in amplitudes}
    curve_precision_reached = False
    curve_replicates = 0
    for batch_start in range(0, args.maximum_verification_replicates, args.batch_size):
        batch_stop = batch_start + args.batch_size
        tasks = [
            (
                layout,
                basis,
                panel_a_indices,
                amplitude,
                curve_fit_by_amplitude[amplitude],
                batch_start,
                batch_stop,
                args.seed,
            )
            for amplitude in amplitudes
        ]
        if args.workers == 1:
            batch_results = [curve_verify_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
                batch_results = list(executor.map(curve_verify_task, tasks))
        for amplitude, values in batch_results:
            curve_values[amplitude].extend(values)
        assert_curve_replicate_completeness(curve_values, amplitudes, batch_stop)
        curve_replicates = batch_stop
        curve_mcses = [mean_se(curve_values[amplitude])[1] for amplitude in amplitudes]
        maximum_curve_mcse = max(curve_mcses)
        print(
            f"Curve verification n={batch_stop}; maximum MCSE={maximum_curve_mcse:.6f}",
            file=sys.stderr,
            flush=True,
        )
        if maximum_curve_mcse <= args.target_mcse:
            curve_precision_reached = True
            break

    raw_curve = np.asarray([mean_se(curve_values[value])[0] for value in amplitudes])
    monotone_curve = pava_non_decreasing(raw_curve)
    selection = interpolate_target(amplitudes, monotone_curve, args.target_gain)
    curve_rows: list[dict[str, object]] = []
    for index, amplitude in enumerate(amplitudes):
        mean, se = mean_se(curve_values[amplitude])
        curve_rows.append(
            {
                "amplitude": amplitude,
                "replicates": len(curve_values[amplitude]),
                "mean_truth_minus_C": mean,
                "mcse_truth_minus_C": se,
                "isotonic_mean_truth_minus_C": float(monotone_curve[index]),
                **{f"optimizer_{key}": value for key, value in summarize_optimizer(curve_fit_by_amplitude[amplitude]).items()},
            }
        )
    curve_optimizer_pass = all(
        len(fit.converged_solutions) >= 2 for fit in curve_fits
    )
    curve_boundaries_inactive = all(
        fit.best.numerical_boundary_diagnostics["active_parameter_bound_count"] == 0
        for fit in curve_fits
    )

    base_summary: dict[str, object] = {
        "diagnostic": "conditional_A8_finite_pool_full_smoke_family_optimizer_check",
        "arms": {
            "CAL_C_FULL": "separately_optimized_complete_C_smoke_family",
            "CAL_H_FULL": "separately_optimized_complete_H_smoke_family_with_smooth_gate",
            "CAL_H_TRUTH": "exact_label_integrated_generator_density",
        },
        "uses_real_projection_outcomes": False,
        "outcome_bearing_search_authorized": False,
        "fixed_role_counts": {"development": 12, "audit": 8, "unassigned": 16},
        "conditional_A8_units": 8,
        "finite_fit_pool_support": "fixed_provisional_A8_metadata_support",
        "population_projection_established": False,
        "positive_amplitude_population_projection_requires_separate_SAA_ladder": True,
        "finite_fit_pool_can_overfit_high_dimensional_C": True,
        "fit_and_independent_verification_use_same_A8_metadata_support": True,
        "D12_to_A8_transfer_evaluated": False,
        "common_scoring_object": "56_coordinate_conditional_nonempty_support_plus_face_ILR_Hausdorff_density",
        "equal_unit_objective": True,
        "C_and_H_fit_separately_without_frozen_coefficients": True,
        "final_H_start_contract": "exact_fitted_C_embedding_plus_nonzero_signature_start",
        "shared_block_regularization": "none_bounded_population_likelihood",
        "coefficient_bound_contract": "numerical_boxes_on_weighted_SVD_and_Helmert_identified_coordinates",
        "globally_equivalent_to_prior_raw_coefficient_box": False,
        "hard_support_or_gate_predictor_clipping": False,
        "numerical_boundary_and_predictor_magnitudes_reported": True,
        "shared_block_bounds_and_optimizer_budget_matched": True,
        "P_design_term_count": basis.raw_term_count,
        "A8_estimable_P_design_rank": basis.estimable_rank,
        "P_design_parameterization": "equal_unit_weighted_SVD_whitened_identifiable_column_space",
        "value_parameterization": "fixed_55_dimensional_sum_zero_Helmert_contrasts",
        "fit_replicates": args.fit_replicates,
        "optimizer": {
            "algorithm": "analytic_gradient_L_BFGS_B",
            "maximum_iterations_per_start": args.max_iterations,
            "diagonal_preconditioner": "start_specific_Fisher_Jacobi_curvature_proxy",
            "final_convergence_check": "physical_identified_coordinate_projected_gradient",
            "projected_gradient_tolerance": PROJECTED_GRADIENT_TOLERANCE,
        },
        "curve_verification": {
            "replicates": curve_replicates,
            "scheduled_replicates_complete": True,
            "target_mcse": args.target_mcse,
            "maximum_mcse": max(mean_se(curve_values[value])[1] for value in amplitudes),
            "precision_reached": curve_precision_reached,
        },
        "curve_two_start_KKT_convergence_pass": curve_optimizer_pass,
        "curve_numerical_boundaries_inactive": curve_boundaries_inactive,
        "target_gain_nats_per_neuron": args.target_gain,
        "model_bounds": {
            "support_coefficient": SUPPORT_BOUND,
            "value_coefficient": VALUE_BOUND,
            "latent_and_group_coefficient": LATENT_BOUND,
            "gate_coefficient": GATE_BOUND,
            "sigma": list(SIGMA_BOUNDS),
        },
        "continuous_quadrature": {
            "nodes": core.CONTINUOUS_QUADRATURE_ORDER,
            "distribution": "fixed_standard_normal_Gauss_Hermite",
            "weights_fitted": False,
        },
        "curve_optimizer": [summarize_optimizer(fit) for fit in sorted(curve_fits, key=lambda item: item.amplitude)],
        "qualification_scope": "synthetic_smoke_family_only_not_registered_production_qualification",
    }
    if selection is None or selection[0] <= 0.0:
        if not curve_precision_reached:
            status = "unresolved_curve_monte_carlo_precision_at_cap"
            failure_reason = (
                "The curve verification reached its replicate cap before its "
                "Monte Carlo precision target, so amplitude calibration is unresolved."
            )
        elif not curve_optimizer_pass:
            status = "unresolved_curve_optimizer_multistart_failure"
            failure_reason = (
                "At least one curve fit lacked two KKT-converged starts, so "
                "amplitude calibration is unresolved."
            )
        elif not curve_boundaries_inactive:
            status = "revise_curve_numerical_boundary_active"
            failure_reason = (
                "At least one curve optimum used a numerical coefficient/sigma "
                "bound."
            )
        elif selection is None:
            status = "unresolved_target_not_bracketed"
            failure_reason = (
                f"The tested amplitude grid did not bracket the requested "
                f"{args.target_gain:.6f} nats/neuron target."
            )
        else:
            status = "unresolved_zero_amplitude_not_below_target"
            failure_reason = (
                "The fitted zero-amplitude truth-minus-C gap already met or "
                f"exceeded the requested {args.target_gain:.6f} nats/neuron "
                "target, so no nonzero strong-group amplitude can be calibrated."
            )
        base_summary.update(
            {
                "status": status,
                "selected_amplitude": None,
                "calibration_failure_reason": failure_reason,
                "maximum_isotonic_gain": float(monotone_curve[-1]),
            }
        )
        write_outputs(args.output_dir, curve_rows, [], base_summary)
        return

    selected_amplitude, lower, upper = selection
    print(
        f"Refitting C/H at zero and selected amplitude {selected_amplitude:.6f}",
        file=sys.stderr,
        flush=True,
    )
    final_specs = (
        ("zero_group", 0.0),
        ("selected_strong", selected_amplitude),
    )
    final_c_tasks = [
        (
            layout,
            basis,
            panel_a_indices,
            amplitude,
            "C",
            args.fit_replicates,
            args.final_starts,
            args.max_iterations,
            args.seed,
            None,
        )
        for _, amplitude in final_specs
    ]
    final_c_fits = fit_many(
        final_c_tasks, min(args.workers, len(final_c_tasks))
    )
    if any(len(fit.converged_solutions) < 2 for fit in final_c_fits):
        raise RuntimeError("A final C fit lacks two KKT-converged starts")
    c_fit_by_amplitude = {fit.amplitude: fit for fit in final_c_fits}
    final_h_tasks = [
        (
            layout,
            basis,
            panel_a_indices,
            amplitude,
            "H",
            args.fit_replicates,
            args.final_starts,
            args.max_iterations,
            args.seed,
            c_fit_by_amplitude[amplitude].best.theta.copy(),
        )
        for _, amplitude in final_specs
    ]
    final_h_fits = fit_many(
        final_h_tasks, min(args.workers, len(final_h_tasks))
    )
    if any(len(fit.converged_solutions) < 2 for fit in final_h_fits):
        raise RuntimeError("A final H fit lacks two KKT-converged starts")
    final_fits = [*final_c_fits, *final_h_fits]
    fit_lookup = {(fit.family, fit.amplitude): fit for fit in final_fits}
    fixtures = tuple(
        (
            name,
            amplitude,
            fit_lookup[("C", amplitude)],
            fit_lookup[("H", amplitude)],
        )
        for name, amplitude in final_specs
    )

    final_rows: list[dict[str, object]] = []
    final_precision_reached = False
    final_replicates = 0
    fixture_names = tuple(name for name, _ in final_specs)
    for batch_start in range(0, args.maximum_verification_replicates, args.batch_size):
        batch_stop = batch_start + args.batch_size
        ranges = chunk_ranges(batch_start, batch_stop, args.workers)
        tasks = [
            (layout, basis, panel_a_indices, fixtures, start, stop, args.seed)
            for start, stop in ranges
        ]
        if args.workers == 1:
            chunks = [final_verify_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
                chunks = list(executor.map(final_verify_task, tasks))
        for chunk in chunks:
            final_rows.extend(chunk)
        final_frame = pd.DataFrame(final_rows)
        assert_final_replicate_completeness(
            final_frame, fixture_names, batch_stop
        )
        final_replicates = batch_stop
        maximum_final_mcse = max_monitored_mcse(final_frame, final_replicates)
        print(
            f"Final verification n={batch_stop}; maximum MCSE={maximum_final_mcse:.6f}",
            file=sys.stderr,
            flush=True,
        )
        if maximum_final_mcse <= args.target_mcse:
            final_precision_reached = True
            break

    final_frame = pd.DataFrame(final_rows)
    assert_final_replicate_completeness(
        final_frame, fixture_names, final_replicates
    )
    contrasts = {
        fixture: {
            column: contrast_summary(
                final_frame, fixture, column, final_replicates
            )
            for column in FINAL_CONTRAST_COLUMNS
        }
        for fixture, _ in final_specs
    }
    zero_truth_c = contrasts["zero_group"]["truth_minus_C"]
    zero_h_c = contrasts["zero_group"]["H_minus_C"]
    strong_truth_c = contrasts["selected_strong"]["truth_minus_C"]
    strong_truth_h = contrasts["selected_strong"]["truth_minus_H"]
    strong_c_stability = contrasts["selected_strong"]["C_best_minus_second"]
    strong_h_stability = contrasts["selected_strong"]["H_best_minus_second"]
    truth_h_upper = one_sided_upper(
        float(strong_truth_h["mean"]),
        float(strong_truth_h["mcse"]),
        int(strong_truth_h["replicates"]),
    )
    optimizer_pass = all(len(fit.converged_solutions) >= 2 for fit in final_fits)
    final_boundaries_inactive = all(
        fit.best.numerical_boundary_diagnostics["active_parameter_bound_count"] == 0
        for fit in final_fits
    )
    h_nested_dominance_pass = all(
        fit.embedded_c_objective is not None
        and fit.best.objective <= fit.embedded_c_objective
        for fit in final_h_fits
    )
    acceptance = {
        "curve_sequential_precision_reached": curve_precision_reached,
        "final_sequential_precision_reached": final_precision_reached,
        "curve_scheduled_replicates_complete": True,
        "final_scheduled_fixture_replicates_complete": True,
        "at_least_two_converged_starts_for_every_curve_C_fit": curve_optimizer_pass,
        "at_least_two_converged_starts_for_each_final_fit": optimizer_pass,
        "curve_optima_do_not_use_numerical_bounds": curve_boundaries_inactive,
        "final_optima_do_not_use_numerical_bounds": final_boundaries_inactive,
        "every_final_H_best_training_objective_at_most_embedded_C_objective": h_nested_dominance_pass,
        "zero_truth_minus_C_absolute_mean_at_most_0.005": abs(float(zero_truth_c["mean"])) <= 0.005,
        "zero_H_minus_C_absolute_mean_at_most_0.005": abs(float(zero_h_c["mean"])) <= 0.005,
        "selected_truth_minus_C_within_0.010_of_target": abs(float(strong_truth_c["mean"]) - args.target_gain) <= 0.010,
        "selected_truth_minus_C_mcse_at_most_0.005": float(strong_truth_c["mcse"]) <= 0.005,
        "selected_truth_minus_H_one_sided_95_upper_below_0.010": truth_h_upper < 0.010,
        "selected_C_best_two_verification_scores_differ_below_0.002": abs(float(strong_c_stability["mean"])) < 0.002,
        "selected_H_best_two_verification_scores_differ_below_0.002": abs(float(strong_h_stability["mean"])) < 0.002,
    }
    if not curve_precision_reached or not final_precision_reached:
        status = "unresolved_monte_carlo_precision_at_cap"
    elif all(acceptance.values()):
        status = "diagnostic_pass_finite_pool_only_population_SAA_required"
    else:
        status = "revise_panel_A_acceptance_failed"
    base_summary.update(
        {
            "status": status,
            "selected_amplitude": selected_amplitude,
            "selected_bracket": {
                "lower_amplitude": amplitudes[lower],
                "upper_amplitude": amplitudes[upper],
                "lower_isotonic_gain": float(monotone_curve[lower]),
                "upper_isotonic_gain": float(monotone_curve[upper]),
            },
            "final_optimizer": [summarize_optimizer(fit) for fit in final_fits],
            "final_verification": {
                "replicates": final_replicates,
                "scheduled_fixture_replicates_complete": True,
                "target_mcse": args.target_mcse,
                "maximum_mcse": max_monitored_mcse(
                    final_frame, final_replicates
                ),
                "precision_reached": final_precision_reached,
                "contrasts": contrasts,
                "selected_truth_minus_H_one_sided_95_upper": truth_h_upper,
            },
            "acceptance": acceptance,
        }
    )
    write_outputs(args.output_dir, curve_rows, final_rows, base_summary)


if __name__ == "__main__":
    main()
