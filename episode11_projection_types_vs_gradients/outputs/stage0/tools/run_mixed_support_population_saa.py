#!/usr/bin/env python3
"""Synthetic-only population-SAA pilot for the EP11 C projection.

This runner approximates the population projection onto the currently
implemented continuous family C on the fixed provisional A8 metadata support.
It integrates the two planted groups by deterministic stratification rather
than by sampling their labels.  Support and value noise are still integrated
by a nested sample-average-approximation (SAA) ladder.  Its JSON status is a
population-C pilot gate only; fitted-H recovery and the rest of Panel A remain
incomplete even when every pilot check passes.

After terminal scoring succeeds, the runner also writes one validated fitted-C
model-state NPZ for later H embedding and common-draw rescoring.  This is a
model result, not a provenance or Panel-A-completion artifact.

The program never reads or constructs a real regional projection outcome.  It
also does not test D12-to-A8 transport, qualify the registered production
grammar, or authorize outcome-bearing search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

# Every optimizer worker is intentionally single-threaded.  Set these before
# NumPy/SciPy are imported, and reassert them in spawned workers before work.
NUMERICAL_LIBRARY_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def force_single_numerical_library_thread() -> None:
    for name in NUMERICAL_LIBRARY_THREAD_VARIABLES:
        os.environ[name] = "1"


force_single_numerical_library_thread()

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import run_mixed_support_full_family_calibration as full


FIT_STREAM = 5101
VALIDATION_STREAM = 5201
TERMINAL_STREAM = 5301
ANCHOR_STREAM = 5401
TRUTH_TEMPLATE_STREAM = 5501
ANCHOR_TOLERANCE = 1.0e-10
PARAMETER_EQUIVALENCE_ATOL = 1.0e-10
PARAMETER_EQUIVALENCE_RTOL = 1.0e-10
MODEL_STATE_CHECKPOINT_FILENAME = "population_saa_C_model_state.npz"
SUMMARY_FILENAME = "population_saa_pilot_summary.json"
MODEL_STATE_SCHEMA_VERSION = 1
MODEL_STATE_FAMILY_CODE_C = ord("C")
MODEL_STATE_BASIS_ATOL = 1.0e-12
MODEL_STATE_OBJECTIVE_ATOL = 1.0e-10
MODEL_STATE_OBJECTIVE_RTOL = 1.0e-10
MAX_EXACT_FLOAT_INTEGER = 2**53
MODEL_STATE_STREAM_NAMES = (
    "fit",
    "validation",
    "terminal",
    "anchor",
    "truth_template",
)
TERMINAL_CANDIDATE_SOURCE_NAMES = (
    "polished",
    "pre_polish_best",
    "original_second",
)
C_CODEC_BLOCK_NAMES = (
    "base_support",
    "base_value",
    "continuous_support",
    "continuous_value",
    "log_sigma",
)
MODEL_STATE_ARRAY_KEYS = (
    "schema_version",
    "family_code",
    "codec_size",
    "design_rank",
    "codec_block_offsets",
    "codec_block_ndim",
    "codec_block_shapes",
    "amplitude",
    "requested_saa_ladder",
    "executed_rung_replicates",
    "validation_replicates",
    "terminal_replicates",
    "anchor_replicates",
    "stream_ids",
    "seed",
    "basis_transform",
    "basis_rank",
    "basis_estimable_rank",
    "basis_raw_term_count",
    "basis_transform_shape",
    "metadata_p_design_shape",
    "support_p_design_shape",
    "support_indices",
    "support_cell_count",
    "support_unit_count",
    "coordinate_count",
    "value_contrast_count",
    "continuous_node_count",
    "continuous_node_basis_shape",
    "continuous_nodes",
    "gaussian_node_weights",
    "continuous_node_basis",
    "value_contrast_basis_shape",
    "value_contrast_basis",
    "gate_term_count",
    "rung_solution_offsets",
    "rung_solution_counts",
    "rung_solution_thetas",
    "rung_solution_objectives",
    "rung_solution_optimizer_success",
    "rung_solution_converged",
    "rung_solution_at_iteration_cap",
    "pre_polish_present",
    "pre_polish_theta",
    "pre_polish_objective",
    "pre_polish_optimizer_success",
    "pre_polish_converged",
    "pre_polish_at_iteration_cap",
    "polished_present",
    "polished_theta",
    "polished_objective",
    "polished_optimizer_success",
    "polished_converged",
    "polished_at_iteration_cap",
    "terminal_candidate_thetas",
    "terminal_candidate_objectives",
    "terminal_candidate_optimizer_success",
    "terminal_candidate_converged",
    "terminal_candidate_at_iteration_cap",
    "terminal_candidate_source_flags",
    "frozen_terminal_candidate_index",
    "frozen_theta",
    "frozen_objective",
    "frozen_source_flags",
    "frozen_training_replicates",
)


@dataclass(frozen=True)
class StreamIDs:
    fit: int
    validation: int
    terminal: int
    anchor: int

    def validate(self) -> None:
        named = {
            "fit": self.fit,
            "validation": self.validation,
            "terminal": self.terminal,
            "anchor": self.anchor,
            "truth_template": TRUTH_TEMPLATE_STREAM,
        }
        if any(value < 0 for value in named.values()):
            raise ValueError("Synthetic stream IDs must be nonnegative integers")
        if any(value > MAX_EXACT_FLOAT_INTEGER for value in named.values()):
            raise ValueError(
                "Synthetic stream IDs must be exactly representable in float64"
            )
        if len(set(named.values())) != len(named):
            raise ValueError(
                "Fit, validation, terminal, anchor, and truth-template stream "
                "IDs must be distinct"
            )

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "fit": self.fit,
            "validation": self.validation,
            "terminal": self.terminal,
            "anchor": self.anchor,
            "truth_template": TRUTH_TEMPLATE_STREAM,
            "all_distinct": True,
        }


@dataclass(frozen=True)
class ConditionalGroupDraw:
    """One conditional draw with its sampled support kept authoritative."""

    active: np.ndarray
    log_values: np.ndarray
    outcomes: np.ndarray


@dataclass(frozen=True)
class RankedTerminalCandidate:
    sources: tuple[str, ...]
    solution: full.OptimizerSolution


def support_face_arrays(
    active: np.ndarray, log_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return face Jacobians/dimensions without inferring support from values."""
    active = np.asarray(active, dtype=bool)
    log_values = np.asarray(log_values, dtype=float)
    if (
        active.ndim != 2
        or active.shape != log_values.shape
        or active.shape[1] != full.core.N_COORDINATES
    ):
        raise ValueError("Support and log values must be aligned n-by-56 arrays")
    if not np.isfinite(log_values).all():
        raise FloatingPointError("Conditional log compositions are nonfinite")
    counts = active.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every synthetic composition must occupy a nonempty face")
    log_jacobian = -np.log(counts) - log_values.sum(axis=1)
    dimension = counts.astype(float) - 1.0
    if not np.isfinite(log_jacobian).all():
        raise FloatingPointError("Conditional composition Jacobian is nonfinite")
    return log_jacobian, dimension


def outcomes_from_active_log_values(
    active: np.ndarray, log_values: np.ndarray
) -> np.ndarray:
    """Construct positive compatibility values while retaining sampled support."""
    active = np.asarray(active, dtype=bool)
    log_values = np.asarray(log_values, dtype=float)
    support_face_arrays(active, log_values)
    outcomes = np.zeros_like(log_values)
    # A normal positive floor remains nonzero after division by at most 56.
    # These values initialize empirical starts only; active/log_values remain
    # authoritative for every fit and score.
    minimum_positive = np.finfo(float).tiny
    for row in range(len(outcomes)):
        on = active[row]
        row_logs = log_values[row, on]
        values = np.exp(row_logs - row_logs.max())
        values = np.maximum(values, minimum_positive)
        outcomes[row, on] = values / values.sum()
    if not np.array_equal(outcomes > 0.0, active):
        raise AssertionError("Compatibility outcomes erased sampled support")
    np.testing.assert_allclose(
        outcomes.sum(axis=1), 1.0, atol=1.0e-10, rtol=0.0
    )
    return outcomes


def draw_from_active_log_intensity(
    active: np.ndarray, log_intensity: np.ndarray
) -> ConditionalGroupDraw:
    """Normalize on each sampled face while retaining exact log compositions."""
    active = np.asarray(active, dtype=bool)
    log_intensity = np.asarray(log_intensity, dtype=float)
    if active.shape != log_intensity.shape or active.ndim != 2:
        raise ValueError("Support and log intensity arrays are misaligned")
    if not np.isfinite(log_intensity).all():
        raise FloatingPointError("Conditional log intensities are nonfinite")
    if np.any(active.sum(axis=1) == 0):
        raise ValueError("Cannot normalize an empty sampled support")
    log_values = np.zeros_like(log_intensity)
    for row in range(len(log_values)):
        on = active[row]
        values = log_intensity[row, on]
        log_values[row, on] = values - np.logaddexp.reduce(values)
    outcomes = outcomes_from_active_log_values(active, log_values)
    return ConditionalGroupDraw(
        active=active.copy(),
        log_values=log_values,
        outcomes=outcomes,
    )


def score_data_from_draw(
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    draw: ConditionalGroupDraw,
    weights: np.ndarray | None = None,
) -> full.ScoreData:
    """Build score data from the sampled support and log values directly."""
    indices = np.asarray(indices, dtype=int)
    if draw.active.shape[0] != len(indices):
        raise ValueError("Conditional draw rows do not match metadata indices")
    log_jacobian, dimension = support_face_arrays(
        draw.active, draw.log_values
    )
    if weights is None:
        weights = full.equal_unit_weights(layout, indices)
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (len(indices),)
        or np.any(weights < 0.0)
        or not np.isfinite(weights).all()
        or not float(weights.sum()) > 0.0
    ):
        raise ValueError("Score weights are misaligned")
    weights = weights / weights.sum()
    return full.ScoreData(
        design=basis.apply(layout.p_design[indices]),
        gate_design=full.smooth_gate_basis(layout, indices),
        active=np.asarray(draw.active, dtype=bool),
        log_values=np.asarray(draw.log_values, dtype=float),
        log_jacobian=log_jacobian,
        dimension=dimension,
        weights=weights,
        outcomes=np.asarray(draw.outcomes, dtype=float),
    )


def truth_scores_from_draw(
    draw: ConditionalGroupDraw, truth: full.core.SyntheticTruth
) -> np.ndarray:
    """Exact generator-H scores using the authoritative sampled support."""
    log_jacobian, dimension = support_face_arrays(
        draw.active, draw.log_values
    )
    active = draw.active
    log_values = draw.log_values
    support_logits = (
        truth.base_support_logits[:, None, :]
        + truth.group_support_shift[None, :, :]
    )
    if support_logits.shape[0] != len(active):
        raise ValueError("Truth rows do not match the conditional draw")
    log_on = -np.logaddexp(0.0, -support_logits)
    log_off = -np.logaddexp(0.0, support_logits)
    unconditioned = np.where(active[:, None, :], log_on, log_off).sum(axis=2)
    support_log_density = unconditioned - full.log_nonempty_from_logits(
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


def maximum_absolute_finite_difference(
    fitted: np.ndarray, exact: np.ndarray
) -> float:
    fitted = np.asarray(fitted, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if fitted.shape != exact.shape:
        raise ValueError("Anchor score arrays are misaligned")
    difference = fitted - exact
    if not np.isfinite(difference).all():
        raise FloatingPointError("Exact analytic anchor difference is nonfinite")
    return float(np.max(np.abs(difference)))


def parameters_equivalent(
    left: full.OptimizerSolution, right: full.OptimizerSolution
) -> bool:
    return bool(
        np.allclose(
            left.theta,
            right.theta,
            atol=PARAMETER_EQUIVALENCE_ATOL,
            rtol=PARAMETER_EQUIVALENCE_RTOL,
        )
    )


def rank_terminal_candidates(
    candidates: Sequence[tuple[str, full.OptimizerSolution]],
) -> tuple[RankedTerminalCandidate, ...]:
    """Deduplicate and rank candidates only by the fit-stream objective."""
    checked: list[tuple[int, str, full.OptimizerSolution]] = []
    for order, (source, solution) in enumerate(candidates):
        if not source:
            raise ValueError("A terminal candidate needs a source label")
        if not np.isfinite(solution.objective) or not np.isfinite(
            solution.theta
        ).all():
            raise FloatingPointError("A terminal candidate is nonfinite")
        checked.append((order, source, solution))
    checked.sort(key=lambda item: (item[2].objective, item[0]))
    unique: list[RankedTerminalCandidate] = []
    for _, source, solution in checked:
        for index, existing in enumerate(unique):
            if parameters_equivalent(solution, existing.solution):
                unique[index] = RankedTerminalCandidate(
                    sources=existing.sources + (source,),
                    solution=existing.solution,
                )
                break
        else:
            unique.append(
                RankedTerminalCandidate(sources=(source,), solution=solution)
            )
    return tuple(unique)


@dataclass(frozen=True)
class SAAMemmapStore:
    """Disk-backed, group-stratified synthetic SAA draws.

    Only the active mask and log composition are persisted.  The objective
    reads at most ``chunk_replicates`` at a time, so its resident arrays do not
    grow with the SAA ladder maximum.
    """

    active_path: Path
    log_values_path: Path
    layout: full.core.MetadataLayout
    basis: full.DesignBasis
    indices: np.ndarray
    truth: full.core.SyntheticTruth
    maximum_replicates: int
    stream: int

    @classmethod
    def create(
        cls,
        directory: Path,
        layout: full.core.MetadataLayout,
        basis: full.DesignBasis,
        indices: np.ndarray,
        amplitude: float,
        maximum_replicates: int,
        seed: int,
        stream: int,
    ) -> "SAAMemmapStore":
        if maximum_replicates < 1:
            raise ValueError("SAA storage needs at least one replicate")
        indices = np.asarray(indices, dtype=int)
        directory.mkdir(parents=True, exist_ok=True)
        truth = truth_template(layout, indices, amplitude, seed)
        shape = (
            int(maximum_replicates),
            2,
            len(indices),
            full.core.N_COORDINATES,
        )
        active_path = directory / "saa_active.npy"
        log_values_path = directory / "saa_log_values.npy"
        active = np.lib.format.open_memmap(
            active_path, mode="w+", dtype=np.bool_, shape=shape
        )
        log_values = np.lib.format.open_memmap(
            log_values_path, mode="w+", dtype=np.float64, shape=shape
        )
        for replicate in range(maximum_replicates):
            for group in range(2):
                draw = conditional_group_draw(
                    truth, group, seed, stream, replicate
                )
                active[replicate, group] = draw.active
                log_values[replicate, group] = draw.log_values
        active.flush()
        log_values.flush()
        del active
        del log_values
        return cls(
            active_path=active_path,
            log_values_path=log_values_path,
            layout=layout,
            basis=basis,
            indices=indices,
            truth=truth,
            maximum_replicates=int(maximum_replicates),
            stream=int(stream),
        )

    def score_data(
        self,
        start: int,
        stop: int,
        include_outcomes: bool = False,
    ) -> full.ScoreData:
        if not (0 <= start < stop <= self.maximum_replicates):
            raise ValueError("Requested SAA chunk is outside the stored range")
        active_map = np.load(self.active_path, mmap_mode="r")
        log_values_map = np.load(self.log_values_path, mmap_mode="r")
        active = np.asarray(active_map[start:stop]).reshape(
            -1, full.core.N_COORDINATES
        )
        log_values = np.asarray(log_values_map[start:stop]).reshape(
            -1, full.core.N_COORDINATES
        )
        replicate_count = stop - start
        one_rep_design = np.vstack(
            [
                self.basis.apply(self.layout.p_design[self.indices]),
                self.basis.apply(self.layout.p_design[self.indices]),
            ]
        )
        one_rep_gate = np.vstack(
            [
                full.smooth_gate_basis(self.layout, self.indices),
                full.smooth_gate_basis(self.layout, self.indices),
            ]
        )
        design = np.tile(one_rep_design, (replicate_count, 1))
        gate_design = np.tile(one_rep_gate, (replicate_count, 1))
        cell_weights = full.equal_unit_weights(self.layout, self.indices)
        one_rep_weights = np.concatenate(
            [
                cell_weights * self.truth.group_probabilities[:, 0],
                cell_weights * self.truth.group_probabilities[:, 1],
            ]
        )
        np.testing.assert_allclose(
            one_rep_weights.sum(), 1.0, atol=1.0e-14, rtol=0.0
        )
        weights = np.tile(one_rep_weights / replicate_count, replicate_count)
        log_jacobian, dimension = support_face_arrays(active, log_values)
        if include_outcomes:
            outcomes = outcomes_from_active_log_values(
                active, log_values
            )
        else:
            # The likelihood and its gradient do not consume ``outcomes``.
            # Avoid a second n-by-56 floating-point array in every chunk.
            outcomes = np.empty((len(active), 0), dtype=float)
        data = full.ScoreData(
            design=design,
            gate_design=gate_design,
            active=active,
            log_values=log_values,
            log_jacobian=log_jacobian,
            dimension=dimension,
            weights=weights,
            outcomes=outcomes,
        )
        if not (
            np.isfinite(data.log_values).all()
            and np.isfinite(data.log_jacobian).all()
            and np.isfinite(data.weights).all()
        ):
            raise FloatingPointError("SAA chunk contains a nonfinite value")
        np.testing.assert_allclose(data.weights.sum(), 1.0, atol=1.0e-14, rtol=0.0)
        return data

    def iter_score_data(
        self, replicates: int, chunk_replicates: int
    ) -> Iterator[tuple[full.ScoreData, float]]:
        if not (1 <= replicates <= self.maximum_replicates):
            raise ValueError("SAA prefix is outside the stored ladder")
        if chunk_replicates < 1:
            raise ValueError("Chunk size must be positive")
        for start in range(0, replicates, chunk_replicates):
            stop = min(start + chunk_replicates, replicates)
            yield self.score_data(start, stop), (stop - start) / replicates


def truth_template(
    layout: full.core.MetadataLayout,
    indices: np.ndarray,
    amplitude: float,
    seed: int,
) -> full.core.SyntheticTruth:
    """Construct the existing full-layout DGP, then retain fixed A8 rows."""
    _, truth = full.generate_full_synthetic(
        layout, amplitude, seed, TRUTH_TEMPLATE_STREAM, 0
    )
    return full.core.subset_synthetic_truth(truth, np.asarray(indices, dtype=int))


def sample_conditional_nonempty_support(
    probability: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sample independent Bernoulli rows conditional on nonempty support.

    Rejection continues only for unresolved rows, so there is no finite-round
    whole-store failure while each returned row retains the exact conditional
    Bernoulli law.
    """
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != full.core.N_COORDINATES:
        raise ValueError("Support probabilities must be an n-by-56 array")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError("Support probabilities must be finite values in [0, 1]")
    if np.any(~np.any(probability > 0.0, axis=1)):
        raise ValueError("A support row has zero probability of being nonempty")

    active = np.zeros(probability.shape, dtype=bool)
    unresolved = np.arange(len(probability), dtype=int)
    rounds = 0
    while len(unresolved):
        candidate = rng.random(
            (len(unresolved), full.core.N_COORDINATES)
        ) < probability[unresolved]
        nonempty = candidate.any(axis=1)
        active[unresolved[nonempty]] = candidate[nonempty]
        unresolved = unresolved[~nonempty]
        rounds += 1
    if np.any(~active.any(axis=1)):
        raise AssertionError("Conditional-nonempty support sampler left an empty row")
    return active, rounds


def conditional_group_draw(
    truth: full.core.SyntheticTruth,
    group: int,
    seed: int,
    stream: int,
    replicate: int,
) -> ConditionalGroupDraw:
    """Draw one support/noise realization conditional on a fixed group."""
    if group not in (0, 1):
        raise ValueError("Group stratum must be zero or one")
    sequence = np.random.SeedSequence(
        [int(seed), int(stream), int(replicate), int(group)]
    )
    support_seed, noise_seed = sequence.spawn(2)
    support_rng = np.random.default_rng(support_seed)
    noise_rng = np.random.default_rng(noise_seed)
    support_logits = (
        truth.base_support_logits + truth.group_support_shift[group][None, :]
    )
    probability = expit(support_logits)
    active, _ = sample_conditional_nonempty_support(probability, support_rng)
    mean = truth.base_value_mean + truth.group_value_shift[group][None, :]
    noisy = mean + noise_rng.normal(0.0, truth.noise_sigma, size=mean.shape)
    return draw_from_active_log_intensity(active, noisy)


def parse_ladder(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("SAA ladder must be comma-separated integers") from exc
    if len(values) < 2 or any(value < 2 for value in values):
        raise ValueError("SAA ladder needs at least two levels, each at least two")
    if tuple(sorted(set(values))) != values:
        raise ValueError("SAA ladder must be strictly increasing without duplicates")
    return values


def resolve_fresh_start_schedule(
    later_level_fresh_starts: int,
    first_level_fresh_starts_override: int | None,
) -> tuple[int, int]:
    """Resolve first/later fresh starts while preserving the legacy default."""
    if later_level_fresh_starts < 2:
        raise ValueError("--starts must request at least two fresh starts")
    first_level_fresh_starts = (
        later_level_fresh_starts
        if first_level_fresh_starts_override is None
        else first_level_fresh_starts_override
    )
    if first_level_fresh_starts < 2:
        raise ValueError(
            "--first-level-fresh-starts must request at least two fresh starts"
        )
    return first_level_fresh_starts, later_level_fresh_starts


def start_source_counts_by_level(
    level_count: int,
    first_level_fresh_starts: int,
    later_level_fresh_starts: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return explicit fresh/warm counts for every requested ladder level."""
    if level_count < 1:
        raise ValueError("A start schedule needs at least one ladder level")
    first, later = resolve_fresh_start_schedule(
        later_level_fresh_starts,
        first_level_fresh_starts,
    )
    fresh = (first,) + (later,) * (level_count - 1)
    warm = (0,) + (1,) * (level_count - 1)
    return fresh, warm


def require_slurm_cpus_per_task() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        raise RuntimeError(
            "SLURM_CPUS_PER_TASK must be set inside a Slurm allocation"
        )
    try:
        cpus = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SLURM_CPUS_PER_TASK must be a positive integer"
        ) from exc
    if cpus < 1:
        raise RuntimeError("SLURM_CPUS_PER_TASK must be a positive integer")
    return cpus


def chunked_objective_gradient(
    codec: full.ParameterCodec,
    theta: np.ndarray,
    store: SAAMemmapStore,
    replicates: int,
    chunk_replicates: int,
) -> tuple[float, np.ndarray]:
    value = 0.0
    gradient = np.zeros(codec.size, dtype=float)
    mass_sum = 0.0
    for data, mass in store.iter_score_data(replicates, chunk_replicates):
        local_value, local_gradient = full.family_objective_gradient(
            codec, theta, data
        )
        value += mass * float(local_value)
        gradient += mass * np.asarray(local_gradient, dtype=float)
        mass_sum += mass
    np.testing.assert_allclose(mass_sum, 1.0, atol=1.0e-14, rtol=0.0)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("Chunked SAA objective or gradient is nonfinite")
    return float(value), gradient


def preconditioner_score_data(
    store: SAAMemmapStore,
    replicates: int,
    preconditioner_replicates: int,
) -> full.ScoreData:
    """Return one fixed fit prefix, independent of objective chunking."""
    if not (1 <= preconditioner_replicates <= replicates):
        raise ValueError(
            "Preconditioner replicates must fit inside every SAA ladder level"
        )
    return store.score_data(0, preconditioner_replicates)


@dataclass(frozen=True)
class CStartTask:
    """Complete immutable input for one independent C optimizer start."""

    start_index: int
    store: SAAMemmapStore
    initial_theta: np.ndarray
    replicates: int
    chunk_replicates: int
    preconditioner_replicates: int
    max_iterations: int


@dataclass(frozen=True)
class CStartResult:
    start_index: int
    solution: full.OptimizerSolution


def optimize_c_start(task: CStartTask) -> CStartResult:
    """Run one start against the same read-only nested-SAA prefix."""
    force_single_numerical_library_thread()
    if any(
        os.environ.get(name) != "1"
        for name in NUMERICAL_LIBRARY_THREAD_VARIABLES
    ):
        raise AssertionError("Optimizer worker numerical threads are not limited")

    codec = full.ParameterCodec("C", task.store.basis.rank)
    start = np.asarray(task.initial_theta, dtype=float)
    if start.shape != (codec.size,) or not np.isfinite(start).all():
        raise ValueError("Optimizer worker received an invalid start")
    preconditioner_data = preconditioner_score_data(
        task.store, task.replicates, task.preconditioner_replicates
    )
    optimizer_scale = full.fisher_jacobi_scale(
        codec, start, preconditioner_data
    )
    scaled_start = start / optimizer_scale
    scaled_bounds = [
        (lower / scale, upper / scale)
        for (lower, upper), scale in zip(codec.bounds(), optimizer_scale)
    ]

    def scaled_objective(phi: np.ndarray) -> tuple[float, np.ndarray]:
        physical = np.asarray(phi, dtype=float) * optimizer_scale
        objective, physical_gradient = chunked_objective_gradient(
            codec,
            physical,
            task.store,
            task.replicates,
            task.chunk_replicates,
        )
        return objective, physical_gradient * optimizer_scale

    result = minimize(
        scaled_objective,
        scaled_start,
        jac=True,
        method="L-BFGS-B",
        bounds=scaled_bounds,
        options={
            "maxiter": int(task.max_iterations),
            "maxls": 40,
            "ftol": 1.0e-14,
            "gtol": 5.0e-7,
            "maxcor": 20,
        },
    )
    theta = np.asarray(result.x, dtype=float) * optimizer_scale
    if theta.shape != (codec.size,) or not np.isfinite(theta).all():
        raise FloatingPointError("SAA optimizer returned invalid parameters")
    objective, gradient = chunked_objective_gradient(
        codec,
        theta,
        task.store,
        task.replicates,
        task.chunk_replicates,
    )
    if not np.isclose(objective, float(result.fun), atol=1.0e-10, rtol=1.0e-10):
        raise AssertionError("SAA optimizer objective did not replay")
    projected = full.projected_gradient(theta, gradient, codec.bounds())
    scaled_projected = full.projected_gradient(
        np.asarray(result.x, dtype=float),
        gradient * optimizer_scale,
        scaled_bounds,
    )
    projected_norm = float(np.max(np.abs(projected)))
    at_cap = int(getattr(result, "nit", -1)) >= int(task.max_iterations)
    block_norms = {
        name: float(np.max(np.abs(projected[parameter_slice])))
        for name, parameter_slice in codec.slices.items()
    }
    solution = full.OptimizerSolution(
        theta=theta,
        objective=float(objective),
        optimizer_success=bool(result.success),
        converged=bool(result.success)
        and not at_cap
        and projected_norm <= full.PROJECTED_GRADIENT_TOLERANCE,
        iterations=int(getattr(result, "nit", -1)),
        evaluations=int(getattr(result, "nfev", -1)),
        gradient_infinity_norm=float(np.max(np.abs(gradient))),
        projected_gradient_infinity_norm=projected_norm,
        scaled_projected_gradient_infinity_norm=float(
            np.max(np.abs(scaled_projected))
        ),
        projected_gradient_block_infinity_norms=block_norms,
        optimizer_scale_minimum=float(optimizer_scale.min()),
        optimizer_scale_maximum=float(optimizer_scale.max()),
        numerical_boundary_diagnostics=full.numerical_boundary_diagnostics(
            codec, theta, preconditioner_data
        ),
        at_iteration_cap=at_cap,
        message=str(result.message),
    )
    return CStartResult(start_index=task.start_index, solution=solution)


def run_c_start_tasks(
    tasks: Sequence[CStartTask],
    start_workers: int,
    progress_label: str,
) -> tuple[full.OptimizerSolution, ...]:
    """Execute starts serially or with spawn, returning original start order."""
    if start_workers < 1:
        raise ValueError("Start worker count must be positive")
    if not tasks:
        raise ValueError("At least one optimizer start task is required")
    start_indices = [task.start_index for task in tasks]
    if len(set(start_indices)) != len(start_indices):
        raise ValueError("Optimizer start indices must be unique")

    solutions: list[full.OptimizerSolution | None] = [None] * len(tasks)

    def record(position: int, result: CStartResult) -> None:
        expected_index = tasks[position].start_index
        if result.start_index != expected_index:
            raise AssertionError("Optimizer worker returned the wrong start index")
        solutions[position] = result.solution
        print(
            f"{progress_label}: completed start {position + 1}/{len(tasks)} "
            f"(objective={result.solution.objective:.12g}, "
            f"KKT_converged={result.solution.converged})",
            file=sys.stderr,
            flush=True,
        )

    effective_workers = min(int(start_workers), len(tasks))
    if effective_workers == 1:
        for position, task in enumerate(tasks):
            record(position, optimize_c_start(task))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
            initializer=force_single_numerical_library_thread,
        ) as executor:
            future_positions = {
                executor.submit(optimize_c_start, task): position
                for position, task in enumerate(tasks)
            }
            for future in as_completed(future_positions):
                position = future_positions[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Optimizer start {position + 1}/{len(tasks)} failed"
                    ) from exc
                record(position, result)

    if any(solution is None for solution in solutions):
        raise AssertionError("An optimizer start result was not collected")
    return tuple(solution for solution in solutions if solution is not None)


def fit_chunked_c(
    store: SAAMemmapStore,
    amplitude: float,
    replicates: int,
    chunk_replicates: int,
    preconditioner_replicates: int,
    initialization_replicates: int,
    starts: int,
    start_workers: int,
    max_iterations: int,
    seed: int,
    warm_start: np.ndarray | None,
    fixed_initial_starts: Sequence[np.ndarray] | None = None,
) -> full.FamilyFit:
    """Fit C with a bounded-memory objective and physical-coordinate KKT test."""
    codec = full.ParameterCodec("C", store.basis.rank)
    initialization_stop = min(initialization_replicates, replicates)
    initialization_data = store.score_data(
        0, initialization_stop, include_outcomes=True
    )
    proxy_pool = full.PopulationPool(
        data=initialization_data,
        single_design=store.basis.apply(store.layout.p_design[store.indices]),
        truth=store.truth,
        replicates=initialization_stop,
    )
    if fixed_initial_starts is None:
        initial = full.make_initial_starts(
            codec,
            proxy_pool,
            amplitude,
            starts,
            seed + replicates,
        )
        if warm_start is not None:
            warm_start = np.asarray(warm_start, dtype=float)
            if warm_start.shape != (codec.size,) or not np.isfinite(warm_start).all():
                raise ValueError("SAA warm start is invalid")
            # Later ladder levels deliberately have one more scheduled start:
            # the requested fresh slots plus the preceding level's optimum.
            initial.append(full.clip_to_bounds(warm_start, codec.bounds()))
    else:
        if warm_start is not None:
            raise ValueError("A fixed-start fit cannot also receive a warm start")
        initial = []
        for supplied in fixed_initial_starts:
            supplied = np.asarray(supplied, dtype=float)
            if supplied.shape != (codec.size,) or not np.isfinite(supplied).all():
                raise ValueError("A fixed SAA start is invalid")
            initial.append(full.clip_to_bounds(supplied, codec.bounds()))
        if not initial:
            raise ValueError("A fixed-start fit needs at least one start")

    tasks = tuple(
        CStartTask(
            start_index=index,
            store=store,
            initial_theta=np.asarray(start, dtype=float),
            replicates=int(replicates),
            chunk_replicates=int(chunk_replicates),
            preconditioner_replicates=int(preconditioner_replicates),
            max_iterations=int(max_iterations),
        )
        for index, start in enumerate(initial)
    )
    solutions = run_c_start_tasks(
        tasks,
        start_workers,
        progress_label=f"R={replicates}",
    )
    return full.FamilyFit(
        family="C",
        amplitude=float(amplitude),
        solutions=solutions,
        parameter_count=codec.size,
        design_rank=codec.design_rank,
    )


def analytic_truth_theta(
    family: str,
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    truth: full.core.SyntheticTruth,
    draw: ConditionalGroupDraw,
    amplitude: float,
) -> np.ndarray:
    data = score_data_from_draw(layout, basis, indices, draw)
    pool = full.PopulationPool(
        data=data,
        single_design=basis.apply(layout.p_design[indices]),
        truth=truth,
        replicates=1,
    )
    codec = full.ParameterCodec(family, basis.rank)
    return codec.pack(full.truth_based_blocks(codec, pool, amplitude, "gaussian"))


def exact_anchor_check(
    family: str,
    amplitude: float,
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    seed: int,
    stream: int,
    replicates: int,
) -> dict[str, object]:
    truth = truth_template(layout, indices, amplitude, seed)
    initialization_draw = conditional_group_draw(
        truth, 0, seed, stream, 0
    )
    theta = analytic_truth_theta(
        family,
        layout,
        basis,
        indices,
        truth,
        initialization_draw,
        amplitude,
    )
    codec = full.ParameterCodec(family, basis.rank)
    maximum_error = 0.0
    comparisons = 0
    for replicate in range(replicates):
        for group in range(2):
            draw = conditional_group_draw(
                truth, group, seed, stream, replicate
            )
            data = score_data_from_draw(layout, basis, indices, draw)
            fitted = np.asarray(
                full.family_objective_gradient(
                    codec, theta, data, return_scores=True
                ),
                dtype=float,
            )
            exact = truth_scores_from_draw(draw, truth)
            maximum_error = max(
                maximum_error,
                maximum_absolute_finite_difference(fitted, exact),
            )
            comparisons += len(draw.active)
    boundary = full.numerical_boundary_diagnostics(
        codec,
        theta,
        score_data_from_draw(layout, basis, indices, initialization_draw),
    )
    bounds = codec.bounds()
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    parameters_within_bounds = bool(
        np.all(theta >= lower) and np.all(theta <= upper)
    )
    boundaries_inactive = bool(
        boundary["active_parameter_bound_count"] == 0
        and not boundary["sigma_at_numerical_bound"]
        and not boundary["hard_predictor_clipping"]
    )
    return {
        "family": family,
        "amplitude": float(amplitude),
        "replicates": int(replicates),
        "cell_score_comparisons": int(comparisons),
        "maximum_absolute_score_error": maximum_error,
        "tolerance": ANCHOR_TOLERANCE,
        "parameters_within_codec_bounds": parameters_within_bounds,
        "numerical_boundaries_inactive": boundaries_inactive,
        "pass": bool(
            maximum_error <= ANCHOR_TOLERANCE
            and parameters_within_bounds
            and boundaries_inactive
        ),
        "analytic_parameter_boundary_diagnostics": boundary,
    }


def mean_mcse(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A Monte Carlo summary needs at least two finite values")
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def evaluate_solutions(
    solutions: Sequence[full.OptimizerSolution],
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    amplitude: float,
    seed: int,
    stream: int,
    replicates: int,
) -> list[dict[str, object]]:
    if not solutions:
        return []
    truth = truth_template(layout, indices, amplitude, seed)
    codec = full.ParameterCodec("C", basis.rank)
    cell_weights = full.equal_unit_weights(layout, indices)
    contrast = np.zeros((len(solutions), replicates), dtype=float)
    c_score = np.zeros((len(solutions), replicates), dtype=float)
    truth_score = np.zeros(replicates, dtype=float)
    for replicate in range(replicates):
        for group in range(2):
            draw = conditional_group_draw(
                truth, group, seed, stream, replicate
            )
            data = score_data_from_draw(layout, basis, indices, draw)
            exact = truth_scores_from_draw(draw, truth)
            stratum_weights = (
                cell_weights * truth.group_probabilities[:, group]
            )
            truth_score[replicate] += float(np.sum(stratum_weights * exact))
            for index, solution in enumerate(solutions):
                fitted = np.asarray(
                    full.family_objective_gradient(
                        codec, solution.theta, data, return_scores=True
                    ),
                    dtype=float,
                )
                c_score[index, replicate] += float(
                    np.sum(stratum_weights * fitted)
                )
                contrast[index, replicate] += float(
                    np.sum(stratum_weights * (exact - fitted))
                )
    output: list[dict[str, object]] = []
    truth_mean, truth_mcse = mean_mcse(truth_score)
    for index, solution in enumerate(solutions):
        contrast_mean, contrast_mcse = mean_mcse(contrast[index])
        score_mean, score_mcse = mean_mcse(c_score[index])
        output.append(
            {
                "training_rank": index + 1,
                "training_negative_mean_log_density": solution.objective,
                "replicates": int(replicates),
                "mean_truth_minus_C_nats_per_neuron": contrast_mean,
                "mcse_truth_minus_C_nats_per_neuron": contrast_mcse,
                "mean_C_log_density": score_mean,
                "mcse_C_log_density": score_mcse,
                "mean_truth_log_density": truth_mean,
                "mcse_truth_log_density": truth_mcse,
            }
        )
    if len(solutions) >= 2:
        paired_difference = c_score[0] - c_score[1]
        paired_mean, paired_mcse = mean_mcse(paired_difference)
        output[0]["paired_C_log_density_minus_next_solution"] = {
            "mean": paired_mean,
            "mcse": paired_mcse,
            "replicates": int(replicates),
        }
    return output


def boundary_inactive(solution: full.OptimizerSolution) -> bool:
    diagnostics = solution.numerical_boundary_diagnostics
    return bool(
        diagnostics["active_parameter_bound_count"] == 0
        and not diagnostics["sigma_at_numerical_bound"]
        and not diagnostics["hard_predictor_clipping"]
    )


def optimizer_summary(fit: full.FamilyFit) -> dict[str, object]:
    starts: list[dict[str, object]] = []
    for index, solution in enumerate(fit.solutions):
        starts.append(
            {
                "start": index,
                "objective": solution.objective,
                "optimizer_success": solution.optimizer_success,
                "physical_KKT_converged": solution.converged,
                "iterations": solution.iterations,
                "evaluations": solution.evaluations,
                "at_iteration_cap": solution.at_iteration_cap,
                "physical_projected_gradient_infinity_norm": (
                    solution.projected_gradient_infinity_norm
                ),
                "physical_projected_gradient_block_infinity_norms": (
                    solution.projected_gradient_block_infinity_norms
                ),
                "scaled_projected_gradient_infinity_norm": (
                    solution.scaled_projected_gradient_infinity_norm
                ),
                "optimizer_scale_range": [
                    solution.optimizer_scale_minimum,
                    solution.optimizer_scale_maximum,
                ],
                "boundary_diagnostics": solution.numerical_boundary_diagnostics,
                "message": solution.message,
            }
        )
    return {
        "family": fit.family,
        "parameter_count": fit.parameter_count,
        "design_rank": fit.design_rank,
        "scheduled_starts": len(fit.solutions),
        "converged_starts": len(fit.converged_solutions),
        "starts": starts,
    }


def nested_diagnostics_match(
    left: object, right: object, tolerance: float = 1.0e-12
) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return bool(
            set(left) == set(right)
            and all(
                nested_diagnostics_match(left[key], right[key], tolerance)
                for key in left
            )
        )
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        return bool(
            np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance)
        )
    return bool(left == right)


def optimizer_solution_equivalence(
    sequential: full.OptimizerSolution,
    parallel: full.OptimizerSolution,
    tolerance: float = 1.0e-12,
) -> dict[str, object]:
    scalar_fields = (
        "objective",
        "gradient_infinity_norm",
        "projected_gradient_infinity_norm",
        "scaled_projected_gradient_infinity_norm",
        "optimizer_scale_minimum",
        "optimizer_scale_maximum",
    )
    scalar_differences = {
        name: abs(float(getattr(sequential, name)) - float(getattr(parallel, name)))
        for name in scalar_fields
    }
    exact_fields = (
        "optimizer_success",
        "converged",
        "iterations",
        "evaluations",
        "at_iteration_cap",
        "message",
    )
    same_theta_shape = sequential.theta.shape == parallel.theta.shape
    theta_difference = (
        float(np.max(np.abs(sequential.theta - parallel.theta)))
        if same_theta_shape
        else math.inf
    )
    diagnostics_match = bool(
        nested_diagnostics_match(
            sequential.projected_gradient_block_infinity_norms,
            parallel.projected_gradient_block_infinity_norms,
            tolerance,
        )
        and nested_diagnostics_match(
            sequential.numerical_boundary_diagnostics,
            parallel.numerical_boundary_diagnostics,
            tolerance,
        )
    )
    exact_metadata_match = all(
        getattr(sequential, name) == getattr(parallel, name)
        for name in exact_fields
    )
    passed = bool(
        same_theta_shape
        and theta_difference <= tolerance
        and max(scalar_differences.values()) <= tolerance
        and diagnostics_match
        and exact_metadata_match
    )
    return {
        "pass": passed,
        "absolute_tolerance": tolerance,
        "maximum_absolute_theta_difference": theta_difference,
        "maximum_absolute_scalar_diagnostic_difference": max(
            scalar_differences.values()
        ),
        "structured_diagnostics_match": diagnostics_match,
        "status_iteration_evaluation_and_message_match": exact_metadata_match,
    }


def finite_float_array(value: object) -> np.ndarray:
    """Encode one checkpoint field as a finite float64 array."""
    original = np.asarray(value)
    if np.issubdtype(original.dtype, np.integer):
        if original.dtype.kind == "u":
            outside_exact_range = original > MAX_EXACT_FLOAT_INTEGER
        else:
            outside_exact_range = (original < -MAX_EXACT_FLOAT_INTEGER) | (
                original > MAX_EXACT_FLOAT_INTEGER
            )
        if np.any(outside_exact_range):
            raise ValueError("Checkpoint integers must be exact in float64")
    elif original.dtype.kind == "O":
        for item in original.flat:
            if isinstance(item, (int, np.integer)) and abs(int(item)) > (
                MAX_EXACT_FLOAT_INTEGER
            ):
                raise ValueError("Checkpoint integers must be exact in float64")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("Model-state checkpoint arrays must be finite")
    return array


def solution_checkpoint_fields(
    prefix: str,
    solution: full.OptimizerSolution | None,
    codec_size: int,
) -> dict[str, np.ndarray]:
    present = solution is not None
    if present:
        theta = np.asarray(solution.theta, dtype=float)
        if theta.shape != (codec_size,) or not np.isfinite(theta).all():
            raise ValueError(f"{prefix} solution has an invalid theta")
        objective = float(solution.objective)
        if not np.isfinite(objective):
            raise FloatingPointError(f"{prefix} solution objective is nonfinite")
    else:
        theta = np.zeros(codec_size, dtype=float)
        objective = 0.0
    return {
        f"{prefix}_present": finite_float_array([float(present)]),
        f"{prefix}_theta": finite_float_array(theta),
        f"{prefix}_objective": finite_float_array([objective]),
        f"{prefix}_optimizer_success": finite_float_array(
            [float(solution.optimizer_success) if present else 0.0]
        ),
        f"{prefix}_converged": finite_float_array(
            [float(solution.converged) if present else 0.0]
        ),
        f"{prefix}_at_iteration_cap": finite_float_array(
            [float(solution.at_iteration_cap) if present else 0.0]
        ),
    }


def build_c_model_state_arrays(
    *,
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    amplitude: float,
    requested_ladder: Sequence[int],
    executed_rung_replicates: Sequence[int],
    validation_replicates: int,
    terminal_replicates: int,
    anchor_replicates: int,
    streams: StreamIDs,
    seed: int,
    fits: Sequence[full.FamilyFit],
    pre_polish_best: full.OptimizerSolution | None,
    polished_solution: full.OptimizerSolution | None,
    terminal_candidates: Sequence[RankedTerminalCandidate],
) -> dict[str, np.ndarray]:
    """Encode the reusable fitted-C state without strings or object arrays."""
    streams.validate()
    indices = np.asarray(indices, dtype=int)
    requested_ladder = tuple(int(value) for value in requested_ladder)
    executed_rung_replicates = tuple(
        int(value) for value in executed_rung_replicates
    )
    if executed_rung_replicates != requested_ladder:
        raise ValueError("A final fitted-C checkpoint requires the complete ladder")
    if min(validation_replicates, terminal_replicates, anchor_replicates) < 2:
        raise ValueError("Checkpoint evaluation replicate counts must be at least two")
    codec = full.ParameterCodec("C", basis.rank)
    if len(fits) != len(executed_rung_replicates) or not fits:
        raise ValueError("Checkpoint rungs and executed replicate counts differ")
    if tuple(codec.shapes) != C_CODEC_BLOCK_NAMES:
        raise AssertionError("The fitted-C codec block order changed")
    for fit in fits:
        if (
            fit.family != "C"
            or fit.parameter_count != codec.size
            or fit.design_rank != basis.rank
            or not np.isclose(fit.amplitude, amplitude, atol=0.0, rtol=0.0)
            or not fit.solutions
        ):
            raise ValueError("A checkpoint rung does not match the fitted C family")
    if not terminal_candidates:
        raise ValueError("A fitted-C checkpoint needs a terminal candidate")

    rung_offsets = [0]
    rung_solutions: list[full.OptimizerSolution] = []
    for fit in fits:
        rung_solutions.extend(fit.solutions)
        rung_offsets.append(len(rung_solutions))
    rung_thetas = np.vstack(
        [np.asarray(solution.theta, dtype=float) for solution in rung_solutions]
    )
    if rung_thetas.shape != (len(rung_solutions), codec.size):
        raise ValueError("A rung solution theta has the wrong C codec size")

    terminal_thetas = np.vstack(
        [
            np.asarray(candidate.solution.theta, dtype=float)
            for candidate in terminal_candidates
        ]
    )
    if terminal_thetas.shape != (len(terminal_candidates), codec.size):
        raise ValueError("A terminal candidate theta has the wrong C codec size")
    source_flags = np.zeros(
        (len(terminal_candidates), len(TERMINAL_CANDIDATE_SOURCE_NAMES)),
        dtype=float,
    )
    source_columns = {
        source: column
        for column, source in enumerate(TERMINAL_CANDIDATE_SOURCE_NAMES)
    }
    for row, candidate in enumerate(terminal_candidates):
        if not candidate.sources:
            raise ValueError("A terminal candidate has no source")
        for source in candidate.sources:
            if source not in source_columns:
                raise ValueError(f"Unknown terminal candidate source {source!r}")
            source_flags[row, source_columns[source]] = 1.0

    stream_values = (
        streams.fit,
        streams.validation,
        streams.terminal,
        streams.anchor,
        TRUTH_TEMPLATE_STREAM,
    )
    frozen_candidate = terminal_candidates[0]
    codec_block_offsets = [codec.slices[name].start for name in C_CODEC_BLOCK_NAMES]
    codec_block_offsets.append(codec.slices[C_CODEC_BLOCK_NAMES[-1]].stop)
    maximum_codec_ndim = max(
        len(codec.shapes[name]) for name in C_CODEC_BLOCK_NAMES
    )
    codec_block_shapes = np.zeros(
        (len(C_CODEC_BLOCK_NAMES), maximum_codec_ndim), dtype=int
    )
    codec_block_ndim = np.zeros(len(C_CODEC_BLOCK_NAMES), dtype=int)
    for row, name in enumerate(C_CODEC_BLOCK_NAMES):
        shape = codec.shapes[name]
        codec_block_ndim[row] = len(shape)
        codec_block_shapes[row, : len(shape)] = shape
    arrays: dict[str, np.ndarray] = {
        "schema_version": finite_float_array([MODEL_STATE_SCHEMA_VERSION]),
        "family_code": finite_float_array([MODEL_STATE_FAMILY_CODE_C]),
        "codec_size": finite_float_array([codec.size]),
        "design_rank": finite_float_array([basis.rank]),
        "codec_block_offsets": finite_float_array(codec_block_offsets),
        "codec_block_ndim": finite_float_array(codec_block_ndim),
        "codec_block_shapes": finite_float_array(codec_block_shapes),
        "amplitude": finite_float_array([amplitude]),
        "requested_saa_ladder": finite_float_array(requested_ladder),
        "executed_rung_replicates": finite_float_array(
            executed_rung_replicates
        ),
        "validation_replicates": finite_float_array(
            [validation_replicates]
        ),
        "terminal_replicates": finite_float_array([terminal_replicates]),
        "anchor_replicates": finite_float_array([anchor_replicates]),
        "stream_ids": finite_float_array(stream_values),
        "seed": finite_float_array([seed]),
        "basis_transform": finite_float_array(basis.transform),
        "basis_rank": finite_float_array([basis.rank]),
        "basis_estimable_rank": finite_float_array([basis.estimable_rank]),
        "basis_raw_term_count": finite_float_array([basis.raw_term_count]),
        "basis_transform_shape": finite_float_array(basis.transform.shape),
        "metadata_p_design_shape": finite_float_array(layout.p_design.shape),
        "support_p_design_shape": finite_float_array(
            layout.p_design[indices].shape
        ),
        "support_indices": finite_float_array(indices),
        "support_cell_count": finite_float_array([len(indices)]),
        "support_unit_count": finite_float_array(
            [len(set(layout.units[indices].tolist()))]
        ),
        "coordinate_count": finite_float_array([full.core.N_COORDINATES]),
        "value_contrast_count": finite_float_array([full.N_VALUE_CONTRASTS]),
        "continuous_node_count": finite_float_array(
            [len(full.CONTINUOUS_NODES)]
        ),
        "continuous_node_basis_shape": finite_float_array(
            full.CONTINUOUS_NODE_BASIS.shape
        ),
        "continuous_nodes": finite_float_array(full.CONTINUOUS_NODES),
        "gaussian_node_weights": finite_float_array(
            full.GAUSSIAN_NODE_WEIGHTS
        ),
        "continuous_node_basis": finite_float_array(
            full.CONTINUOUS_NODE_BASIS
        ),
        "value_contrast_basis_shape": finite_float_array(
            full.VALUE_CONTRAST_BASIS.shape
        ),
        "value_contrast_basis": finite_float_array(
            full.VALUE_CONTRAST_BASIS
        ),
        "gate_term_count": finite_float_array([full.GATE_TERMS]),
        "rung_solution_offsets": finite_float_array(rung_offsets),
        "rung_solution_counts": finite_float_array(
            np.diff(np.asarray(rung_offsets, dtype=int))
        ),
        "rung_solution_thetas": finite_float_array(rung_thetas),
        "rung_solution_objectives": finite_float_array(
            [solution.objective for solution in rung_solutions]
        ),
        "rung_solution_optimizer_success": finite_float_array(
            [solution.optimizer_success for solution in rung_solutions]
        ),
        "rung_solution_converged": finite_float_array(
            [solution.converged for solution in rung_solutions]
        ),
        "rung_solution_at_iteration_cap": finite_float_array(
            [solution.at_iteration_cap for solution in rung_solutions]
        ),
        "terminal_candidate_thetas": finite_float_array(terminal_thetas),
        "terminal_candidate_objectives": finite_float_array(
            [candidate.solution.objective for candidate in terminal_candidates]
        ),
        "terminal_candidate_optimizer_success": finite_float_array(
            [
                candidate.solution.optimizer_success
                for candidate in terminal_candidates
            ]
        ),
        "terminal_candidate_converged": finite_float_array(
            [candidate.solution.converged for candidate in terminal_candidates]
        ),
        "terminal_candidate_at_iteration_cap": finite_float_array(
            [
                candidate.solution.at_iteration_cap
                for candidate in terminal_candidates
            ]
        ),
        "terminal_candidate_source_flags": finite_float_array(source_flags),
        "frozen_terminal_candidate_index": finite_float_array([0]),
        "frozen_theta": finite_float_array(frozen_candidate.solution.theta),
        "frozen_objective": finite_float_array(
            [frozen_candidate.solution.objective]
        ),
        "frozen_source_flags": finite_float_array(source_flags[0]),
        "frozen_training_replicates": finite_float_array(
            [executed_rung_replicates[-1]]
        ),
    }
    arrays.update(
        solution_checkpoint_fields(
            "pre_polish", pre_polish_best, codec.size
        )
    )
    arrays.update(
        solution_checkpoint_fields(
            "polished", polished_solution, codec.size
        )
    )
    if set(arrays) != set(MODEL_STATE_ARRAY_KEYS):
        raise AssertionError("Internal model-state checkpoint schema mismatch")
    return arrays


def integral_checkpoint_array(
    arrays: dict[str, np.ndarray], name: str
) -> np.ndarray:
    values = arrays[name]
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"Checkpoint field {name} is not integer-valued")
    return rounded.astype(np.int64)


def scalar_checkpoint_integer(
    arrays: dict[str, np.ndarray], name: str
) -> int:
    if arrays[name].shape != (1,):
        raise ValueError(f"Checkpoint field {name} is not a scalar array")
    return int(integral_checkpoint_array(arrays, name)[0])


def require_binary_checkpoint_array(
    arrays: dict[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> None:
    values = arrays[name]
    if values.shape != shape or not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"Checkpoint field {name} is not a {shape} binary array")


def validate_checkpoint_terminal_crosslinks(
    arrays: dict[str, np.ndarray], offsets: np.ndarray, codec_size: int
) -> None:
    """Rebuild terminal ranking from persisted source states and compare it."""

    def rung_state(index: int) -> dict[str, object]:
        return {
            "theta": arrays["rung_solution_thetas"][index],
            "objective": float(arrays["rung_solution_objectives"][index]),
            "optimizer_success": float(
                arrays["rung_solution_optimizer_success"][index]
            ),
            "converged": float(arrays["rung_solution_converged"][index]),
            "at_iteration_cap": float(
                arrays["rung_solution_at_iteration_cap"][index]
            ),
        }

    def named_state(prefix: str) -> dict[str, object]:
        return {
            "theta": arrays[f"{prefix}_theta"],
            "objective": float(arrays[f"{prefix}_objective"][0]),
            "optimizer_success": float(
                arrays[f"{prefix}_optimizer_success"][0]
            ),
            "converged": float(arrays[f"{prefix}_converged"][0]),
            "at_iteration_cap": float(
                arrays[f"{prefix}_at_iteration_cap"][0]
            ),
        }

    def states_match(left: dict[str, object], right: dict[str, object]) -> bool:
        return bool(
            np.array_equal(left["theta"], right["theta"])
            and left["objective"] == right["objective"]
            and left["optimizer_success"] == right["optimizer_success"]
            and left["converged"] == right["converged"]
            and left["at_iteration_cap"] == right["at_iteration_cap"]
        )

    final_indices = list(range(int(offsets[-2]), int(offsets[-1])))
    final_converged = [
        index
        for index in final_indices
        if arrays["rung_solution_converged"][index] == 1.0
    ]
    final_converged.sort(
        key=lambda index: arrays["rung_solution_objectives"][index]
    )
    if not final_converged or arrays["pre_polish_present"][0] != 1.0:
        raise ValueError("Checkpoint has no final pre-polish best solution")
    pre_polish = named_state("pre_polish")
    if not states_match(pre_polish, rung_state(final_converged[0])):
        raise ValueError("Checkpoint pre-polish state is not the final-rung best")

    raw_candidates: list[tuple[int, str, dict[str, object]]] = []
    if (
        arrays["polished_present"][0] == 1.0
        and arrays["polished_converged"][0] == 1.0
    ):
        raw_candidates.append((0, "polished", named_state("polished")))
    raw_candidates.append((1, "pre_polish_best", pre_polish))
    if len(final_converged) >= 2:
        raw_candidates.append(
            (2, "original_second", rung_state(final_converged[1]))
        )
    raw_candidates.sort(key=lambda item: (item[2]["objective"], item[0]))

    source_columns = {
        source: column
        for column, source in enumerate(TERMINAL_CANDIDATE_SOURCE_NAMES)
    }
    expected: list[dict[str, object]] = []
    for _, source, state in raw_candidates:
        for existing in expected:
            if np.allclose(
                state["theta"],
                existing["theta"],
                atol=PARAMETER_EQUIVALENCE_ATOL,
                rtol=PARAMETER_EQUIVALENCE_RTOL,
            ):
                existing["source_flags"][source_columns[source]] = 1.0
                break
        else:
            source_flags = np.zeros(
                len(TERMINAL_CANDIDATE_SOURCE_NAMES), dtype=float
            )
            source_flags[source_columns[source]] = 1.0
            expected.append({**state, "source_flags": source_flags})

    terminal_thetas = arrays["terminal_candidate_thetas"]
    if terminal_thetas.shape != (len(expected), codec_size):
        raise ValueError("Checkpoint terminal candidates do not match source states")
    for index, state in enumerate(expected):
        actual = {
            "theta": terminal_thetas[index],
            "objective": float(arrays["terminal_candidate_objectives"][index]),
            "optimizer_success": float(
                arrays["terminal_candidate_optimizer_success"][index]
            ),
            "converged": float(
                arrays["terminal_candidate_converged"][index]
            ),
            "at_iteration_cap": float(
                arrays["terminal_candidate_at_iteration_cap"][index]
            ),
        }
        if not states_match(actual, state) or not np.array_equal(
            arrays["terminal_candidate_source_flags"][index],
            state["source_flags"],
        ):
            raise ValueError(
                "Checkpoint terminal ranking/source cross-link is inconsistent"
            )


def basis_transform_matches_reconstruction(
    checkpoint_transform: np.ndarray, reconstructed_transform: np.ndarray
) -> bool:
    """Allow only numerical-library roundoff in a reconstructed SVD basis."""
    checkpoint_transform = np.asarray(checkpoint_transform, dtype=float)
    reconstructed_transform = np.asarray(reconstructed_transform, dtype=float)
    return bool(
        checkpoint_transform.shape == reconstructed_transform.shape
        and np.allclose(
            checkpoint_transform,
            reconstructed_transform,
            atol=MODEL_STATE_BASIS_ATOL,
            rtol=0.0,
        )
    )


def load_validate_c_model_state_checkpoint(
    path: Path,
    *,
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    amplitude: float,
    requested_ladder: Sequence[int],
    executed_rung_replicates: Sequence[int],
    validation_replicates: int,
    terminal_replicates: int,
    anchor_replicates: int,
    streams: StreamIDs,
    seed: int,
    store: SAAMemmapStore,
    final_replicates: int,
    chunk_replicates: int,
) -> dict[str, np.ndarray]:
    """Load, validate, and replay one fitted-C model-state checkpoint."""
    streams.validate()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(MODEL_STATE_ARRAY_KEYS):
                raise ValueError("Checkpoint fields do not match the C-state schema")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load fitted-C model-state checkpoint: {exc}") from exc
    for name, array in arrays.items():
        if array.dtype != np.dtype(np.float64):
            raise ValueError(f"Checkpoint field {name} is not a float64 array")
        if not np.isfinite(array).all():
            raise ValueError(f"Checkpoint field {name} contains a nonfinite value")

    indices = np.asarray(indices, dtype=int)
    requested_ladder = tuple(int(value) for value in requested_ladder)
    executed_rung_replicates = tuple(
        int(value) for value in executed_rung_replicates
    )
    if executed_rung_replicates != requested_ladder:
        raise ValueError("A final fitted-C checkpoint requires the complete ladder")
    if min(validation_replicates, terminal_replicates, anchor_replicates) < 2:
        raise ValueError("Checkpoint evaluation replicate counts must be at least two")
    codec = full.ParameterCodec("C", basis.rank)
    scalar_expectations = {
        "schema_version": MODEL_STATE_SCHEMA_VERSION,
        "family_code": MODEL_STATE_FAMILY_CODE_C,
        "codec_size": codec.size,
        "design_rank": basis.rank,
        "basis_rank": basis.rank,
        "basis_estimable_rank": basis.estimable_rank,
        "basis_raw_term_count": basis.raw_term_count,
        "support_cell_count": len(indices),
        "support_unit_count": len(set(layout.units[indices].tolist())),
        "coordinate_count": full.core.N_COORDINATES,
        "value_contrast_count": full.N_VALUE_CONTRASTS,
        "continuous_node_count": len(full.CONTINUOUS_NODES),
        "gate_term_count": full.GATE_TERMS,
        "seed": int(seed),
        "validation_replicates": int(validation_replicates),
        "terminal_replicates": int(terminal_replicates),
        "anchor_replicates": int(anchor_replicates),
    }
    for name, expected in scalar_expectations.items():
        if scalar_checkpoint_integer(arrays, name) != int(expected):
            raise ValueError(f"Checkpoint config mismatch for {name}")
    if arrays["amplitude"].shape != (1,) or not np.isclose(
        arrays["amplitude"][0], amplitude, atol=0.0, rtol=0.0
    ):
        raise ValueError("Checkpoint config mismatch for amplitude")

    exact_vector_expectations = {
        "requested_saa_ladder": requested_ladder,
        "executed_rung_replicates": executed_rung_replicates,
        "stream_ids": (
            streams.fit,
            streams.validation,
            streams.terminal,
            streams.anchor,
            TRUTH_TEMPLATE_STREAM,
        ),
        "basis_transform_shape": basis.transform.shape,
        "metadata_p_design_shape": layout.p_design.shape,
        "support_p_design_shape": layout.p_design[indices].shape,
        "support_indices": indices,
        "continuous_node_basis_shape": full.CONTINUOUS_NODE_BASIS.shape,
        "continuous_nodes": full.CONTINUOUS_NODES,
        "gaussian_node_weights": full.GAUSSIAN_NODE_WEIGHTS,
        "value_contrast_basis_shape": full.VALUE_CONTRAST_BASIS.shape,
    }
    for name, expected in exact_vector_expectations.items():
        expected_array = np.asarray(expected, dtype=float)
        if arrays[name].shape != expected_array.shape or not np.array_equal(
            arrays[name], expected_array
        ):
            raise ValueError(f"Checkpoint config mismatch for {name}")
    if not basis_transform_matches_reconstruction(
        arrays["basis_transform"], basis.transform
    ):
        raise ValueError("Checkpoint basis transform does not match this fit")
    if tuple(codec.shapes) != C_CODEC_BLOCK_NAMES:
        raise ValueError("Current fitted-C codec block order is incompatible")
    expected_codec_offsets = [
        codec.slices[name].start for name in C_CODEC_BLOCK_NAMES
    ] + [codec.slices[C_CODEC_BLOCK_NAMES[-1]].stop]
    expected_codec_ndim = np.asarray(
        [len(codec.shapes[name]) for name in C_CODEC_BLOCK_NAMES], dtype=float
    )
    maximum_codec_ndim = int(expected_codec_ndim.max())
    expected_codec_shapes = np.zeros(
        (len(C_CODEC_BLOCK_NAMES), maximum_codec_ndim), dtype=float
    )
    for row, name in enumerate(C_CODEC_BLOCK_NAMES):
        shape = codec.shapes[name]
        expected_codec_shapes[row, : len(shape)] = shape
    for name, expected in (
        ("codec_block_offsets", np.asarray(expected_codec_offsets, dtype=float)),
        ("codec_block_ndim", expected_codec_ndim),
        ("codec_block_shapes", expected_codec_shapes),
    ):
        if arrays[name].shape != expected.shape or not np.array_equal(
            arrays[name], expected
        ):
            raise ValueError(f"Checkpoint config mismatch for {name}")
    for name, expected in (
        ("continuous_node_basis", full.CONTINUOUS_NODE_BASIS),
        ("value_contrast_basis", full.VALUE_CONTRAST_BASIS),
    ):
        if arrays[name].shape != expected.shape or not np.array_equal(
            arrays[name], expected
        ):
            raise ValueError(f"Checkpoint config mismatch for {name}")

    offsets = integral_checkpoint_array(arrays, "rung_solution_offsets")
    counts = integral_checkpoint_array(arrays, "rung_solution_counts")
    if (
        offsets.shape != (len(executed_rung_replicates) + 1,)
        or counts.shape != (len(executed_rung_replicates),)
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or not np.array_equal(np.diff(offsets), counts)
    ):
        raise ValueError("Checkpoint rung offsets/counts are invalid")
    total_solutions = int(offsets[-1])
    if arrays["rung_solution_thetas"].shape != (total_solutions, codec.size):
        raise ValueError("Checkpoint rung theta matrix has the wrong shape")
    if arrays["rung_solution_objectives"].shape != (total_solutions,):
        raise ValueError("Checkpoint rung objective vector has the wrong shape")
    for name in (
        "rung_solution_optimizer_success",
        "rung_solution_converged",
        "rung_solution_at_iteration_cap",
    ):
        require_binary_checkpoint_array(arrays, name, (total_solutions,))

    for prefix in ("pre_polish", "polished"):
        require_binary_checkpoint_array(arrays, f"{prefix}_present", (1,))
        if arrays[f"{prefix}_theta"].shape != (codec.size,):
            raise ValueError(f"Checkpoint {prefix} theta has the wrong shape")
        if arrays[f"{prefix}_objective"].shape != (1,):
            raise ValueError(f"Checkpoint {prefix} objective has the wrong shape")
        for suffix in (
            "optimizer_success",
            "converged",
            "at_iteration_cap",
        ):
            require_binary_checkpoint_array(
                arrays, f"{prefix}_{suffix}", (1,)
            )

    terminal_thetas = arrays["terminal_candidate_thetas"]
    if terminal_thetas.ndim != 2 or terminal_thetas.shape[1] != codec.size:
        raise ValueError("Checkpoint terminal candidate matrix has the wrong shape")
    terminal_count = terminal_thetas.shape[0]
    if terminal_count < 1:
        raise ValueError("Checkpoint contains no terminal candidate")
    if arrays["terminal_candidate_objectives"].shape != (terminal_count,):
        raise ValueError("Checkpoint terminal objectives have the wrong shape")
    if np.any(np.diff(arrays["terminal_candidate_objectives"]) < 0.0):
        raise ValueError("Checkpoint terminal candidates are not fit-objective ranked")
    for name in (
        "terminal_candidate_optimizer_success",
        "terminal_candidate_converged",
        "terminal_candidate_at_iteration_cap",
    ):
        require_binary_checkpoint_array(arrays, name, (terminal_count,))
    source_shape = (terminal_count, len(TERMINAL_CANDIDATE_SOURCE_NAMES))
    require_binary_checkpoint_array(
        arrays, "terminal_candidate_source_flags", source_shape
    )
    if np.any(arrays["terminal_candidate_source_flags"].sum(axis=1) < 1.0):
        raise ValueError("A checkpoint terminal candidate has no encoded source")
    if not np.all(arrays["terminal_candidate_converged"] == 1.0):
        raise ValueError("A checkpoint terminal candidate is not converged")
    for left in range(terminal_count):
        for right in range(left + 1, terminal_count):
            if np.allclose(
                terminal_thetas[left],
                terminal_thetas[right],
                atol=PARAMETER_EQUIVALENCE_ATOL,
                rtol=PARAMETER_EQUIVALENCE_RTOL,
            ):
                raise ValueError("Checkpoint terminal candidates are not deduplicated")
    validate_checkpoint_terminal_crosslinks(arrays, offsets, codec.size)

    frozen_index = scalar_checkpoint_integer(
        arrays, "frozen_terminal_candidate_index"
    )
    if frozen_index != 0 or frozen_index >= terminal_count:
        raise ValueError("Frozen C must be fit-objective-ranked terminal candidate zero")
    if arrays["frozen_theta"].shape != (codec.size,):
        raise ValueError("Checkpoint frozen C theta has the wrong shape")
    if arrays["frozen_objective"].shape != (1,):
        raise ValueError("Checkpoint frozen C objective has the wrong shape")
    if arrays["frozen_source_flags"].shape != (
        len(TERMINAL_CANDIDATE_SOURCE_NAMES),
    ):
        raise ValueError("Checkpoint frozen C sources have the wrong shape")
    if not np.array_equal(arrays["frozen_theta"], terminal_thetas[frozen_index]):
        raise ValueError("Checkpoint frozen theta differs from its terminal candidate")
    if not np.array_equal(
        arrays["frozen_objective"],
        arrays["terminal_candidate_objectives"][frozen_index : frozen_index + 1],
    ):
        raise ValueError("Checkpoint frozen objective differs from its candidate")
    if not np.array_equal(
        arrays["frozen_source_flags"],
        arrays["terminal_candidate_source_flags"][frozen_index],
    ):
        raise ValueError("Checkpoint frozen sources differ from its candidate")
    if scalar_checkpoint_integer(
        arrays, "frozen_training_replicates"
    ) != int(final_replicates):
        raise ValueError("Checkpoint frozen training-replicate count is wrong")
    if (
        not executed_rung_replicates
        or executed_rung_replicates[-1] != int(final_replicates)
    ):
        raise ValueError("Final replicate count is not the last executed rung")

    if (
        store.stream != streams.fit
        or store.maximum_replicates < final_replicates
        or not np.array_equal(store.indices, indices)
        or store.basis.rank != basis.rank
        or not np.array_equal(store.basis.transform, basis.transform)
    ):
        raise ValueError("Open final SAA store does not match checkpoint config")
    replayed_objective, _ = chunked_objective_gradient(
        codec,
        arrays["frozen_theta"],
        store,
        final_replicates,
        chunk_replicates,
    )
    if not np.isclose(
        replayed_objective,
        arrays["frozen_objective"][0],
        atol=MODEL_STATE_OBJECTIVE_ATOL,
        rtol=MODEL_STATE_OBJECTIVE_RTOL,
    ):
        raise ValueError("Frozen C training objective does not replay")
    return arrays


def fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_fresh_publication_namespace(output_dir: Path) -> None:
    """Reject only final artifact names from an earlier production run."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    final_paths = (
        output_dir / SUMMARY_FILENAME,
        output_dir / MODEL_STATE_CHECKPOINT_FILENAME,
    )
    existing = [path.name for path in final_paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to reuse an existing population-SAA publication name: "
            + ", ".join(existing)
        )


def publish_temporary_file_without_overwrite(
    temporary_path: Path, final_path: Path
) -> None:
    """Atomically publish a same-directory temporary file only if absent."""
    try:
        os.link(temporary_path, final_path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing final artifact: {final_path}"
        ) from exc
    temporary_path.unlink()


def write_c_model_state_checkpoint(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    **validation: object,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Validate a temporary compressed NPZ, then atomically publish it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / MODEL_STATE_CHECKPOINT_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{MODEL_STATE_CHECKPOINT_FILENAME}.",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = load_validate_c_model_state_checkpoint(
            temporary_path, **validation
        )
        publish_temporary_file_without_overwrite(temporary_path, final_path)
        temporary_path = None
        fsync_directory(output_dir)
        return final_path, loaded
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def self_test(
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    seed: int,
    streams: StreamIDs,
) -> dict[str, object]:
    streams.validate()
    if os.environ.get("SLURM_JOB_ID"):
        self_test_cpus = require_slurm_cpus_per_task()
        if self_test_cpus < 2:
            raise RuntimeError(
                "The spawned-start equivalence self-test requires at least "
                "two SLURM_CPUS_PER_TASK"
            )
    if any(
        os.environ.get(name) != "1"
        for name in NUMERICAL_LIBRARY_THREAD_VARIABLES
    ):
        raise AssertionError("Numerical-library thread limits are not enforced")
    anchors = {
        "C_zero": exact_anchor_check(
            "C", 0.0, layout, basis, indices, seed, streams.anchor, 1
        ),
        "H_positive": exact_anchor_check(
            "H", 0.7, layout, basis, indices, seed, streams.anchor, 1
        ),
    }
    if not all(item["pass"] for item in anchors.values()):
        raise AssertionError("An exact analytic anchor failed")
    truth = truth_template(layout, indices, 0.7, seed)
    repeated_a = conditional_group_draw(
        truth, 1, seed, streams.fit, 0
    )
    repeated_b = conditional_group_draw(
        truth, 1, seed, streams.fit, 0
    )
    independent = conditional_group_draw(
        truth, 1, seed, streams.validation, 0
    )
    np.testing.assert_array_equal(repeated_a.active, repeated_b.active)
    np.testing.assert_array_equal(repeated_a.log_values, repeated_b.log_values)
    np.testing.assert_array_equal(repeated_a.outcomes, repeated_b.outcomes)
    if np.array_equal(repeated_a.active, independent.active) and np.array_equal(
        repeated_a.log_values, independent.log_values
    ):
        raise AssertionError("Independent synthetic streams unexpectedly coincide")

    low_probability = np.full(
        (128, full.core.N_COORDINATES), 1.0e-3, dtype=float
    )
    low_support_a, low_rounds_a = sample_conditional_nonempty_support(
        low_probability, np.random.default_rng(314159)
    )
    low_support_b, low_rounds_b = sample_conditional_nonempty_support(
        low_probability, np.random.default_rng(314159)
    )
    np.testing.assert_array_equal(low_support_a, low_support_b)
    if low_rounds_a != low_rounds_b or low_rounds_a <= 16:
        raise AssertionError(
            "Conditional support rejection did not exercise continuation past round 16"
        )
    if np.any(~low_support_a.any(axis=1)):
        raise AssertionError("Conditional support continuation returned an empty row")

    underflow_active = np.zeros((1, full.core.N_COORDINATES), dtype=bool)
    underflow_active[0, :3] = True
    underflow_log_intensity = np.zeros_like(underflow_active, dtype=float)
    underflow_log_intensity[0, 2] = -1000.0
    underflow_draw = draw_from_active_log_intensity(
        underflow_active, underflow_log_intensity
    )
    if not np.array_equal(underflow_draw.active, underflow_active):
        raise AssertionError("Log-space normalization changed sampled support")
    if not np.array_equal(underflow_draw.outcomes > 0.0, underflow_active):
        raise AssertionError("Compatibility values erased underflow-prone support")
    if not underflow_draw.log_values[0, 2] < -900.0:
        raise AssertionError("Underflow-prone log composition was clipped")

    nonfinite_anchor_rejected = False
    try:
        maximum_absolute_finite_difference(
            np.asarray([np.nan]), np.asarray([0.0])
        )
    except FloatingPointError:
        nonfinite_anchor_rejected = True
    if not nonfinite_anchor_rejected:
        raise AssertionError("Nonfinite anchor differences were not rejected")

    with tempfile.TemporaryDirectory(prefix="ep11-population-saa-selftest-") as temporary:
        namespace_root = Path(temporary) / "publication-namespaces"
        clean_namespace = namespace_root / "clean"
        clean_namespace.mkdir(parents=True)
        (clean_namespace / "unrelated.txt").write_text(
            "unrelated\n", encoding="utf-8"
        )
        require_fresh_publication_namespace(clean_namespace)

        existing_json_namespace = namespace_root / "existing-json"
        existing_json_namespace.mkdir()
        (existing_json_namespace / SUMMARY_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        existing_json_rejected = False
        try:
            require_fresh_publication_namespace(existing_json_namespace)
        except FileExistsError:
            existing_json_rejected = True
        if not existing_json_rejected:
            raise AssertionError("An existing final summary JSON was accepted")

        existing_npz_namespace = namespace_root / "existing-npz"
        existing_npz_namespace.mkdir()
        (existing_npz_namespace / MODEL_STATE_CHECKPOINT_FILENAME).write_bytes(
            b"existing"
        )
        existing_npz_rejected = False
        try:
            require_fresh_publication_namespace(existing_npz_namespace)
        except FileExistsError:
            existing_npz_rejected = True
        if not existing_npz_rejected:
            raise AssertionError("An existing final C checkpoint was accepted")
        publication_namespace_checks = {
            "pass": True,
            "clean_directory_with_unrelated_file_is_allowed": True,
            "existing_final_summary_JSON_is_rejected": existing_json_rejected,
            "existing_final_C_checkpoint_NPZ_is_rejected": (
                existing_npz_rejected
            ),
        }

        store = SAAMemmapStore.create(
            Path(temporary),
            layout,
            basis,
            indices,
            amplitude=0.7,
            maximum_replicates=2,
            seed=seed,
            stream=streams.fit,
        )
        first = store.score_data(0, 1, include_outcomes=True)
        if not np.array_equal(first.active, first.outcomes > 0.0):
            raise AssertionError("Stored sampled support was not preserved")
        pool = full.PopulationPool(
            data=first,
            single_design=basis.apply(layout.p_design[indices]),
            truth=store.truth,
            replicates=1,
        )
        codec = full.ParameterCodec("C", basis.rank)
        theta = codec.pack(
            full.truth_based_blocks(codec, pool, 0.7, "gaussian")
        )
        fixed_preconditioner_data = preconditioner_score_data(store, 2, 2)
        optimizer_scale_before = full.fisher_jacobi_scale(
            codec, theta, fixed_preconditioner_data
        )
        value_one, gradient_one = chunked_objective_gradient(
            codec, theta, store, 2, 1
        )
        value_two, gradient_two = chunked_objective_gradient(
            codec, theta, store, 2, 2
        )
        np.testing.assert_allclose(value_one, value_two, atol=1.0e-11, rtol=0.0)
        np.testing.assert_allclose(
            gradient_one, gradient_two, atol=1.0e-11, rtol=0.0
        )
        optimizer_scale_after = full.fisher_jacobi_scale(
            codec, theta, preconditioner_score_data(store, 2, 2)
        )
        np.testing.assert_array_equal(optimizer_scale_before, optimizer_scale_after)
        if not np.isfinite(gradient_one).all():
            raise AssertionError("Chunked self-test gradient is nonfinite")
        maximum_rows = max(
            len(data.active) for data, _ in store.iter_score_data(2, 1)
        )
        expected_maximum_rows = 2 * len(indices)
        if maximum_rows != expected_maximum_rows:
            raise AssertionError("Chunked SAA row bound changed")
        perturbed = theta.copy()
        perturbed[0] += 1.0e-3
        perturbed_value, _ = chunked_objective_gradient(
            codec, perturbed, store, 2, 1
        )
        boundary = full.numerical_boundary_diagnostics(codec, theta, first)
        probe_solutions = [
            full.OptimizerSolution(
                theta=value.copy(),
                objective=float(objective),
                optimizer_success=True,
                converged=True,
                iterations=0,
                evaluations=1,
                gradient_infinity_norm=0.0,
                projected_gradient_infinity_norm=0.0,
                scaled_projected_gradient_infinity_norm=0.0,
                projected_gradient_block_infinity_norms={},
                optimizer_scale_minimum=1.0,
                optimizer_scale_maximum=1.0,
                numerical_boundary_diagnostics=boundary,
                at_iteration_cap=False,
                message="self_test_probe",
            )
            for value, objective in (
                (theta, value_one),
                (perturbed, perturbed_value),
            )
        ]
        paired_probe = evaluate_solutions(
            probe_solutions,
            layout,
            basis,
            indices,
            0.7,
            seed,
            streams.validation,
            2,
        )[0]["paired_C_log_density_minus_next_solution"]
        if not np.isfinite(
            [float(paired_probe["mean"]), float(paired_probe["mcse"])]
        ).all():
            raise AssertionError("Paired validation summary is nonfinite")

        terminal_probe = rank_terminal_candidates(
            [
                ("polished", replace(probe_solutions[0], objective=2.0)),
                ("pre_polish_best", replace(probe_solutions[0], objective=3.0)),
                ("original_second", replace(probe_solutions[1], objective=1.0)),
            ]
        )
        if len(terminal_probe) != 2:
            raise AssertionError("Parameter-equivalent terminal candidates remain")
        if terminal_probe[0].sources != ("original_second",):
            raise AssertionError("Terminal candidates were not fit-objective ranked")
        if set(terminal_probe[1].sources) != {"polished", "pre_polish_best"}:
            raise AssertionError("Equivalent terminal candidate sources were lost")

        checkpoint_fit = full.FamilyFit(
            family="C",
            amplitude=0.7,
            solutions=tuple(probe_solutions),
            parameter_count=codec.size,
            design_rank=basis.rank,
        )
        checkpoint_ranked = checkpoint_fit.ranked_solutions
        checkpoint_terminal = rank_terminal_candidates(
            [
                ("polished", checkpoint_ranked[0]),
                ("pre_polish_best", checkpoint_ranked[0]),
                ("original_second", checkpoint_ranked[1]),
            ]
        )
        checkpoint_arrays = build_c_model_state_arrays(
            layout=layout,
            basis=basis,
            indices=indices,
            amplitude=0.7,
            requested_ladder=(2,),
            executed_rung_replicates=(2,),
            validation_replicates=2,
            terminal_replicates=2,
            anchor_replicates=2,
            streams=streams,
            seed=seed,
            fits=(checkpoint_fit,),
            pre_polish_best=checkpoint_ranked[0],
            polished_solution=checkpoint_ranked[0],
            terminal_candidates=checkpoint_terminal,
        )
        checkpoint_directory = Path(temporary) / "model-state-roundtrip"
        checkpoint_path, loaded_checkpoint = write_c_model_state_checkpoint(
            checkpoint_directory,
            checkpoint_arrays,
            layout=layout,
            basis=basis,
            indices=indices,
            amplitude=0.7,
            requested_ladder=(2,),
            executed_rung_replicates=(2,),
            validation_replicates=2,
            terminal_replicates=2,
            anchor_replicates=2,
            streams=streams,
            seed=seed,
            store=store,
            final_replicates=2,
            chunk_replicates=1,
        )
        if checkpoint_path.name != MODEL_STATE_CHECKPOINT_FILENAME:
            raise AssertionError("Model-state checkpoint filename changed")
        if not all(
            np.issubdtype(value.dtype, np.floating)
            and np.isfinite(value).all()
            for value in loaded_checkpoint.values()
        ):
            raise AssertionError("Checkpoint round trip produced a non-float array")
        np.testing.assert_array_equal(
            loaded_checkpoint["frozen_theta"],
            checkpoint_terminal[0].solution.theta,
        )
        checkpoint_validation: dict[str, object] = {
            "layout": layout,
            "basis": basis,
            "indices": indices,
            "amplitude": 0.7,
            "requested_ladder": (2,),
            "executed_rung_replicates": (2,),
            "validation_replicates": 2,
            "terminal_replicates": 2,
            "anchor_replicates": 2,
            "streams": streams,
            "seed": seed,
            "store": store,
            "final_replicates": 2,
            "chunk_replicates": 1,
        }

        def checkpoint_is_rejected(
            candidate_path: Path, **overrides: object
        ) -> bool:
            validation = {**checkpoint_validation, **overrides}
            try:
                load_validate_c_model_state_checkpoint(
                    candidate_path, **validation
                )
            except ValueError:
                return True
            return False

        roundoff_transform = basis.transform.copy()
        roundoff_transform[0, 0] += 0.5 * MODEL_STATE_BASIS_ATOL
        roundoff_basis = replace(basis, transform=roundoff_transform)
        roundoff_store = replace(store, basis=roundoff_basis)
        load_validate_c_model_state_checkpoint(
            checkpoint_path,
            **{
                **checkpoint_validation,
                "basis": roundoff_basis,
                "store": roundoff_store,
            },
        )
        checkpoint_basis_roundoff_accepted = True

        above_tolerance_transform = basis.transform.copy()
        above_tolerance_transform[0, 0] += 2.0 * MODEL_STATE_BASIS_ATOL
        above_tolerance_basis = replace(
            basis, transform=above_tolerance_transform
        )
        above_tolerance_store = replace(store, basis=above_tolerance_basis)
        checkpoint_basis_change_rejected = checkpoint_is_rejected(
            checkpoint_path,
            basis=above_tolerance_basis,
            store=above_tolerance_store,
        )
        if not checkpoint_basis_change_rejected:
            raise AssertionError("Material checkpoint-basis change was accepted")

        wrong_family = {
            name: value.copy() for name, value in loaded_checkpoint.items()
        }
        wrong_family["family_code"][0] = float(ord("H"))
        wrong_family_path = checkpoint_directory / "wrong-family.npz"
        np.savez_compressed(wrong_family_path, **wrong_family)
        wrong_family_rejected = checkpoint_is_rejected(wrong_family_path)
        if not wrong_family_rejected:
            raise AssertionError("Wrong-family checkpoint tamper was accepted")

        changed_theta = {
            name: value.copy() for name, value in loaded_checkpoint.items()
        }
        original_frozen_theta = changed_theta["frozen_theta"].copy()
        changed_theta["frozen_theta"][0] += 0.25
        changed_theta["terminal_candidate_thetas"][0, 0] += 0.25
        matching_rung_solutions = np.all(
            changed_theta["rung_solution_thetas"] == original_frozen_theta,
            axis=1,
        )
        if not np.any(matching_rung_solutions):
            raise AssertionError("Frozen theta is not linked to a rung solution")
        changed_theta["rung_solution_thetas"][matching_rung_solutions, 0] += 0.25
        for prefix in ("pre_polish", "polished"):
            if np.array_equal(
                changed_theta[f"{prefix}_theta"], original_frozen_theta
            ):
                changed_theta[f"{prefix}_theta"][0] += 0.25
        changed_theta_path = checkpoint_directory / "changed-theta.npz"
        np.savez_compressed(changed_theta_path, **changed_theta)
        changed_theta_error: str | None = None
        try:
            load_validate_c_model_state_checkpoint(
                changed_theta_path, **checkpoint_validation
            )
        except ValueError as exc:
            changed_theta_error = str(exc)
        changed_theta_rejected = (
            changed_theta_error == "Frozen C training objective does not replay"
        )
        if not changed_theta_rejected:
            raise AssertionError(
                "Changed-theta checkpoint did not reach objective-replay rejection: "
                f"{changed_theta_error}"
            )

        wrong_shape = {
            name: value.copy() for name, value in loaded_checkpoint.items()
        }
        wrong_shape["frozen_theta"] = wrong_shape["frozen_theta"][:-1]
        wrong_shape_path = checkpoint_directory / "wrong-shape.npz"
        np.savez_compressed(wrong_shape_path, **wrong_shape)
        wrong_shape_rejected = checkpoint_is_rejected(wrong_shape_path)
        if not wrong_shape_rejected:
            raise AssertionError("Wrong-shape checkpoint tamper was accepted")

        nonfinite = {
            name: value.copy() for name, value in loaded_checkpoint.items()
        }
        nonfinite["basis_transform"][0, 0] = np.nan
        nonfinite_path = checkpoint_directory / "nonfinite.npz"
        np.savez_compressed(nonfinite_path, **nonfinite)
        nonfinite_rejected = checkpoint_is_rejected(nonfinite_path)
        if not nonfinite_rejected:
            raise AssertionError("Nonfinite checkpoint tamper was accepted")

        config_mismatch_rejected = checkpoint_is_rejected(
            checkpoint_path, seed=seed + 1
        )
        if not config_mismatch_rejected:
            raise AssertionError("Checkpoint config mismatch was accepted")
        checkpoint_roundtrip = {
            "pass": True,
            "schema_version": MODEL_STATE_SCHEMA_VERSION,
            "all_arrays_are_finite_float_arrays": True,
            "compressed_NPZ_round_trip": True,
            "wrong_family_tamper_rejected": wrong_family_rejected,
            "wrong_shape_tamper_rejected": wrong_shape_rejected,
            "nonfinite_array_tamper_rejected": nonfinite_rejected,
            "config_mismatch_tamper_rejected": config_mismatch_rejected,
            "basis_roundoff_within_tolerance_replays": (
                checkpoint_basis_roundoff_accepted
            ),
            "basis_change_above_tolerance_rejected": (
                checkpoint_basis_change_rejected
            ),
            "changed_theta_training_objective_replay_tamper_rejected": (
                changed_theta_rejected
            ),
            "frozen_selection": (
                "deduplicated_terminal_candidate_fit_objective_rank_1"
            ),
        }

        # Exercise the actual SciPy/memmap worker path with a deliberately
        # reduced rank-one design and three metadata rows.  One optimizer
        # iteration is sufficient to test dispatch, start identity, objective
        # replay, parameter collection, and all emitted diagnostics without
        # turning the structural self-test into a production fit.
        small_indices = np.asarray(indices[: min(3, len(indices))], dtype=int)
        restricted_design = np.zeros_like(layout.p_design[small_indices])
        source_design = np.asarray(layout.p_design[small_indices], dtype=float)
        selected_column = int(
            np.argmax(np.linalg.norm(source_design, axis=0))
        )
        restricted_design[:, selected_column] = source_design[:, selected_column]
        small_basis = full.DesignBasis.from_support(
            restricted_design,
            full.equal_unit_weights(layout, small_indices),
        )
        small_store = SAAMemmapStore.create(
            Path(temporary) / "parallel-equivalence",
            layout,
            small_basis,
            small_indices,
            amplitude=0.7,
            maximum_replicates=1,
            seed=seed,
            stream=streams.fit,
        )
        small_data = small_store.score_data(0, 1, include_outcomes=True)
        small_pool = full.PopulationPool(
            data=small_data,
            single_design=small_basis.apply(
                layout.p_design[small_indices]
            ),
            truth=small_store.truth,
            replicates=1,
        )
        small_codec = full.ParameterCodec("C", small_basis.rank)
        analytic_start = full.clip_to_bounds(
            small_codec.pack(
                full.truth_based_blocks(
                    small_codec, small_pool, 0.7, "gaussian"
                )
            ),
            small_codec.bounds(),
        )
        neutral_start = np.zeros(small_codec.size, dtype=float)
        neutral_start[small_codec.slices["log_sigma"]] = math.log(
            small_store.truth.noise_sigma
        )
        neutral_start = full.clip_to_bounds(
            neutral_start, small_codec.bounds()
        )
        equivalence_tasks = tuple(
            CStartTask(
                start_index=start_index,
                store=small_store,
                initial_theta=start.copy(),
                replicates=1,
                chunk_replicates=1,
                preconditioner_replicates=1,
                max_iterations=1,
            )
            for start_index, start in (
                (101, analytic_start),
                (202, neutral_start),
            )
        )
        initial_snapshots = tuple(
            task.initial_theta.copy() for task in equivalence_tasks
        )
        sequential_solutions = run_c_start_tasks(
            equivalence_tasks,
            start_workers=1,
            progress_label="self-test sequential",
        )
        parallel_solutions = run_c_start_tasks(
            equivalence_tasks,
            start_workers=2,
            progress_label="self-test parallel",
        )
        for task, snapshot in zip(equivalence_tasks, initial_snapshots):
            np.testing.assert_array_equal(task.initial_theta, snapshot)
        equivalence_comparisons = [
            optimizer_solution_equivalence(sequential, parallel)
            for sequential, parallel in zip(
                sequential_solutions, parallel_solutions
            )
        ]
        if not all(item["pass"] for item in equivalence_comparisons):
            raise AssertionError(
                "Sequential and spawned optimizer-start results differ"
            )
        parallel_start_equivalence = {
            "pass": True,
            "multiprocessing_context": "spawn",
            "starts": len(equivalence_tasks),
            "workers": 2,
            "metadata_rows": len(small_indices),
            "design_rank": small_basis.rank,
            "parameter_count": small_codec.size,
            "SAA_replicates": 1,
            "optimizer_iterations": 1,
            "comparisons": equivalence_comparisons,
            "scope_limit": (
                "actual_memmap_objective_preconditioner_SciPy_and_diagnostic_"
                "path_on_a_reduced_deterministic_problem_not_a_convergence_test"
            ),
        }

    legacy_start_schedule = resolve_fresh_start_schedule(2, None)
    revised_start_schedule = resolve_fresh_start_schedule(2, 3)
    revised_fresh_by_level, revised_warm_by_level = start_source_counts_by_level(
        4, 3, 2
    )
    invalid_first_level_start_schedule_rejected = False
    try:
        resolve_fresh_start_schedule(2, 1)
    except ValueError:
        invalid_first_level_start_schedule_rejected = True
    if legacy_start_schedule != (2, 2):
        raise AssertionError("Legacy uniform fresh-start schedule changed")
    if revised_start_schedule != (3, 2):
        raise AssertionError("Revised first-rung fresh-start schedule changed")
    if revised_fresh_by_level != (3, 2, 2, 2):
        raise AssertionError("Revised per-rung fresh-start counts changed")
    if revised_warm_by_level != (0, 1, 1, 1):
        raise AssertionError("Revised per-rung warm-start counts changed")
    if not invalid_first_level_start_schedule_rejected:
        raise AssertionError("An invalid first-rung fresh-start count was accepted")

    return {
        "status": "pass",
        "uses_real_projection_outcomes": False,
        "A8_units": len(set(layout.units[indices].tolist())),
        "A8_cells": len(indices),
        "A8_design_rank": basis.rank,
        "C_parameter_count": full.ParameterCodec("C", basis.rank).size,
        "H_parameter_count": full.ParameterCodec("H", basis.rank).size,
        "streams": streams.as_dict(),
        "anchors": anchors,
        "parallel_start_equivalence": parallel_start_equivalence,
        "model_state_checkpoint_roundtrip": checkpoint_roundtrip,
        "publication_namespace": publication_namespace_checks,
        "fresh_start_schedule": {
            "legacy_default_first_and_later": list(legacy_start_schedule),
            "revised_first_and_later": list(revised_start_schedule),
            "revised_fresh_by_level": list(revised_fresh_by_level),
            "revised_warm_by_level": list(revised_warm_by_level),
        },
        "checks": {
            "conditional_draw_replays_exactly_within_stream": True,
            "validation_stream_is_independent_of_fit_stream": True,
            "group_strata_are_integrated_with_probability_weights": True,
            "chunk_size_one_and_two_objective_match": True,
            "chunk_size_one_and_two_gradient_match": True,
            "fixed_preconditioner_prefix_is_independent_of_chunk_size": True,
            "resident_objective_rows_are_bounded_by_chunk_replicates": True,
            "paired_validation_difference_and_MCSE_are_finite": True,
            "nonfinite_anchor_differences_are_rejected": True,
            "sampled_support_survives_softmax_underflow": True,
            "conditional_support_rejection_continues_only_unresolved_rows": True,
            "conditional_support_rejection_continues_past_round_16": True,
            "terminal_candidates_are_deduplicated_and_fit_objective_ranked": True,
            "terminal_stream_is_not_opened_by_self_test": True,
            "numerical_library_thread_limits_are_forced_to_one": True,
            "sequential_and_spawned_optimizer_starts_match": True,
            "optimizer_start_inputs_are_not_mutated": True,
            "model_state_checkpoint_round_trip_passes": True,
            "model_state_checkpoint_tampering_is_rejected": True,
            "fresh_publication_namespace_is_enforced": True,
            "legacy_uniform_fresh_start_schedule_is_preserved": True,
            "revised_three_then_two_fresh_start_schedule_is_exact": True,
            "revised_zero_then_one_warm_start_schedule_is_exact": True,
            "invalid_first_level_fresh_start_count_is_rejected": True,
        },
    }


def write_summary(output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / SUMMARY_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            dir=output_dir,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file_without_overwrite(temporary_path, final_path)
        temporary_path = None
        fsync_directory(output_dir)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--amplitude", type=float, default=1.25)
    parser.add_argument("--saa-ladder", default="128,512")
    parser.add_argument("--chunk-replicates", type=int, default=32)
    parser.add_argument("--preconditioner-replicates", type=int, default=32)
    parser.add_argument("--initialization-replicates", type=int, default=16)
    parser.add_argument(
        "--starts",
        type=int,
        default=2,
        help="Fresh starts at later ladder levels before adding the warm start",
    )
    parser.add_argument(
        "--first-level-fresh-starts",
        type=int,
        default=None,
        help="Fresh starts at the first ladder level; defaults to --starts",
    )
    parser.add_argument("--start-workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--polish-iterations", type=int, default=600)
    parser.add_argument("--validation-replicates", type=int, default=128)
    parser.add_argument("--terminal-replicates", type=int, default=512)
    parser.add_argument("--anchor-replicates", type=int, default=2)
    parser.add_argument("--target-mcse", type=float, default=0.005)
    parser.add_argument("--start-stability", type=float, default=0.002)
    parser.add_argument("--ladder-stability", type=float, default=0.005)
    parser.add_argument("--terminal-stability", type=float, default=0.005)
    parser.add_argument("--polish-improvement", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--fit-stream", type=int, default=FIT_STREAM)
    parser.add_argument("--validation-stream", type=int, default=VALIDATION_STREAM)
    parser.add_argument("--terminal-stream", type=int, default=TERMINAL_STREAM)
    parser.add_argument("--anchor-stream", type=int, default=ANCHOR_STREAM)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    streams = StreamIDs(
        fit=args.fit_stream,
        validation=args.validation_stream,
        terminal=args.terminal_stream,
        anchor=args.anchor_stream,
    )
    streams.validate()
    if not (0 <= args.seed <= MAX_EXACT_FLOAT_INTEGER):
        raise ValueError(
            "--seed must be nonnegative and exactly representable in float64"
        )
    if args.start_workers < 1:
        raise ValueError("--start-workers must be a positive integer")
    (
        first_level_fresh_starts,
        later_level_fresh_starts,
    ) = resolve_fresh_start_schedule(
        args.starts,
        args.first_level_fresh_starts,
    )
    if not args.self_test:
        if args.output_dir is None:
            raise ValueError("--output-dir is required")
        require_fresh_publication_namespace(args.output_dir)

    full.core.validate_vocabulary(args.vocabulary)
    layout = full.core.load_metadata(args.inventory)
    roles = full.panel_b_fixed_roles(layout)
    indices = np.asarray(roles.audit, dtype=int)
    basis = full.DesignBasis.from_support(
        layout.p_design[indices], full.equal_unit_weights(layout, indices)
    )
    if args.self_test:
        print(
            json.dumps(
                self_test(layout, basis, indices, args.seed, streams),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The population-SAA fit must run inside a Slurm allocation")
    slurm_cpus_per_task = require_slurm_cpus_per_task()
    if args.start_workers > slurm_cpus_per_task:
        raise RuntimeError(
            f"--start-workers={args.start_workers} exceeds "
            f"SLURM_CPUS_PER_TASK={slurm_cpus_per_task}"
        )
    if args.scratch_dir is None:
        scratch_root = os.environ.get("SCRATCH")
        if not scratch_root:
            raise ValueError("--scratch-dir is required when SCRATCH is unset")
        args.scratch_dir = (
            Path(scratch_root)
            / "br_autoresearch/episode11_projection_types_vs_gradients/population_saa"
        )
    ladder = parse_ladder(args.saa_ladder)
    fresh_starts_by_level, warm_starts_by_level = start_source_counts_by_level(
        len(ladder),
        first_level_fresh_starts,
        later_level_fresh_starts,
    )
    if any(value > MAX_EXACT_FLOAT_INTEGER for value in ladder):
        raise ValueError("SAA ladder values must be exactly representable in float64")
    if not np.isfinite(args.amplitude) or args.amplitude <= 0.0:
        raise ValueError("The population-C SAA pilot requires a positive amplitude")
    if (
        args.chunk_replicates < 1
        or args.preconditioner_replicates < 1
        or args.initialization_replicates < 1
    ):
        raise ValueError(
            "Chunk, preconditioner, and initialization replicate counts must be positive"
        )
    if args.preconditioner_replicates > ladder[0]:
        raise ValueError(
            "--preconditioner-replicates must not exceed the first SAA ladder level"
        )
    if min(args.max_iterations, args.polish_iterations) < 1:
        raise ValueError("Use positive iteration limits")
    if min(
        args.validation_replicates,
        args.terminal_replicates,
        args.anchor_replicates,
    ) < 2:
        raise ValueError("Anchor, validation, and terminal stages need at least two replicates")
    if max(
        args.validation_replicates,
        args.terminal_replicates,
        args.anchor_replicates,
    ) > MAX_EXACT_FLOAT_INTEGER:
        raise ValueError(
            "Evaluation replicate counts must be exactly representable in float64"
        )
    thresholds = (
        args.target_mcse,
        args.start_stability,
        args.ladder_stability,
        args.terminal_stability,
        args.polish_improvement,
    )
    if not np.isfinite(thresholds).all() or any(value <= 0.0 for value in thresholds):
        raise ValueError("All numerical pilot thresholds must be finite and positive")

    anchors = {
        "C_amplitude_zero": exact_anchor_check(
            "C",
            0.0,
            layout,
            basis,
            indices,
            args.seed,
            streams.anchor,
            args.anchor_replicates,
        ),
        "H_requested_amplitude": exact_anchor_check(
            "H",
            args.amplitude,
            layout,
            basis,
            indices,
            args.seed,
            streams.anchor,
            args.anchor_replicates,
        ),
    }
    maximum_scheduled_starts = max(
        fresh + warm
        for fresh, warm in zip(fresh_starts_by_level, warm_starts_by_level)
    )
    summary: dict[str, object] = {
        "diagnostic": "EP11_group_stratified_population_C_SAA_pilot",
        "uses_real_projection_outcomes": False,
        "outcome_bearing_search_authorized": False,
        "Panel_A_population_calibration_complete": False,
        "terminal_population_calibration_pass": False,
        "scope": "implemented_smoke_family_on_fixed_provisional_A8_metadata_support",
        "D12_to_A8_transport_evaluated": False,
        "amplitude": float(args.amplitude),
        "A8_units": len(set(layout.units[indices].tolist())),
        "A8_cells": len(indices),
        "P_design_rank": basis.rank,
        "C_parameter_count": full.ParameterCodec("C", basis.rank).size,
        "SAA_ladder": list(ladder),
        "default_ladder_is_a_pilot_not_terminal_Panel_A_calibration": (
            ladder == (128, 512)
        ),
        "group_integration": "exact_two_stratum_probability_weighting_per_metadata_row",
        "support_and_value_noise_integration": "nested_sample_average_approximation",
        "objective_memory_contract": {
            "storage": "two_temporary_disk_backed_numpy_arrays",
            "maximum_resident_replicates_per_objective_chunk": args.chunk_replicates,
            "maximum_resident_rows_per_worker_objective_chunk": (
                2 * len(indices) * args.chunk_replicates
            ),
            "maximum_concurrent_optimizer_workers": min(
                args.start_workers, maximum_scheduled_starts, slurm_cpus_per_task
            ),
            "maximum_aggregate_rows_across_concurrent_worker_chunks": (
                2
                * len(indices)
                * args.chunk_replicates
                * min(
                    args.start_workers,
                    maximum_scheduled_starts,
                    slurm_cpus_per_task,
                )
            ),
        },
        "preconditioner": {
            "fit_stream_prefix_replicates": args.preconditioner_replicates,
            "independent_of_objective_chunk_replicates": True,
        },
        "streams": {
            **streams.as_dict(),
            "fit_ladder_uses_nested_prefixes": True,
            "terminal_stream_opened_only_after_final_ladder_fit": True,
        },
        "start_schedule": {
            "policy": "first_rung_fresh_override_then_later_fresh_plus_previous_best_warm_v1",
            "first_level_fresh_starts_requested": first_level_fresh_starts,
            "later_level_fresh_starts_requested": later_level_fresh_starts,
            "initialization_prefix_replicates": args.initialization_replicates,
            "maximum_iterations_per_main_start": args.max_iterations,
            "maximum_iterations_for_fixed_start_polish": args.polish_iterations,
            "first_level_scheduled_starts": first_level_fresh_starts,
            "later_levels_add_previous_level_best_as_one_extra_warm_start": True,
            "later_level_scheduled_starts": later_level_fresh_starts + 1,
            "start_workers_requested": args.start_workers,
            "SLURM_cpus_per_task": slurm_cpus_per_task,
            "multiprocessing_context": "spawn",
            "first_level_effective_workers": min(
                args.start_workers,
                first_level_fresh_starts,
                slurm_cpus_per_task,
            ),
            "later_level_effective_workers": min(
                args.start_workers,
                later_level_fresh_starts + 1,
                slurm_cpus_per_task,
            ),
            "result_collection_order": "original_start_order",
            "objective_or_draw_sharding": False,
            "polish_uses_one_fixed_start_from_final_ladder_best": True,
            "polish_start_workers": 1,
        },
        "anchors": anchors,
        "ladder": [],
        "thresholds": {
            "analytic_anchor_maximum_absolute_error": ANCHOR_TOLERANCE,
            "validation_start_stability_nats_per_neuron": args.start_stability,
            "successive_ladder_validation_change_nats_per_neuron": args.ladder_stability,
            "terminal_start_stability_nats_per_neuron": args.terminal_stability,
            "maximum_fresh_score_gain_from_polishing_nats_per_neuron": (
                args.polish_improvement
            ),
            "terminal_MCSE_nats_per_neuron": args.target_mcse,
            "physical_projected_gradient": full.PROJECTED_GRADIENT_TOLERANCE,
        },
    }
    if not all(item["pass"] for item in anchors.values()):
        summary["status"] = "technical_failure_exact_analytic_anchor"
        summary["acceptance"] = {
            "exact_analytic_anchors_pass": False,
            "terminal_population_calibration_pass": False,
            "Panel_A_population_calibration_complete": False,
        }
        write_summary(args.output_dir, summary)
        return

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ep11-population-saa-", dir=args.scratch_dir
    ) as temporary:
        print(
            f"Building nested group-stratified SAA store through R={ladder[-1]}",
            file=sys.stderr,
            flush=True,
        )
        store = SAAMemmapStore.create(
            Path(temporary),
            layout,
            basis,
            indices,
            amplitude=args.amplitude,
            maximum_replicates=ladder[-1],
            seed=args.seed,
            stream=streams.fit,
        )
        warm_start: np.ndarray | None = None
        fits: list[full.FamilyFit] = []
        validation_results: list[list[dict[str, object]]] = []
        for level_index, replicates in enumerate(ladder):
            print(
                f"Fitting population-C SAA level R={replicates}",
                file=sys.stderr,
                flush=True,
            )
            fresh_starts = fresh_starts_by_level[level_index]
            fit = fit_chunked_c(
                store,
                args.amplitude,
                replicates,
                args.chunk_replicates,
                args.preconditioner_replicates,
                args.initialization_replicates,
                fresh_starts,
                args.start_workers,
                args.max_iterations,
                args.seed,
                warm_start,
            )
            fits.append(fit)
            ranked = fit.ranked_solutions[:2]
            validation = evaluate_solutions(
                ranked,
                layout,
                basis,
                indices,
                args.amplitude,
                args.seed,
                streams.validation,
                args.validation_replicates,
            )
            validation_results.append(validation)
            start_pair = (
                None
                if len(validation) < 2
                else validation[0]["paired_C_log_density_minus_next_solution"]
            )
            previous_pair = None
            if len(fits) >= 2 and ranked and fits[-2].converged_solutions:
                cross_level = evaluate_solutions(
                    [ranked[0], fits[-2].best],
                    layout,
                    basis,
                    indices,
                    args.amplitude,
                    args.seed,
                    streams.validation,
                    args.validation_replicates,
                )
                previous_pair = cross_level[0][
                    "paired_C_log_density_minus_next_solution"
                ]
            level_summary = {
                "SAA_replicates": replicates,
                "fresh_starts_scheduled": fresh_starts,
                "previous_level_warm_starts_scheduled": warm_starts_by_level[
                    level_index
                ],
                "optimizer": optimizer_summary(fit),
                "effective_start_workers": min(
                    args.start_workers, len(fit.solutions)
                ),
                "validation_stream": streams.validation,
                "validation": validation,
                "best_two_validation_score_paired_difference": start_pair,
                "best_vs_previous_level_validation_score_paired_difference": (
                    previous_pair
                ),
                "at_least_two_physical_KKT_converged_starts": len(ranked) >= 2,
                "best_solution_numerical_boundaries_inactive": (
                    bool(ranked) and boundary_inactive(ranked[0])
                ),
            }
            summary["ladder"].append(level_summary)
            print(
                f"Completed R={replicates}: "
                f"{len(fit.converged_solutions)}/{len(fit.solutions)} starts KKT-converged",
                file=sys.stderr,
                flush=True,
            )
            if not ranked:
                break
            warm_start = ranked[0].theta.copy()

        final_fit = fits[-1]
        final_ranked = final_fit.ranked_solutions[:2]
        polish_fit: full.FamilyFit | None = None
        polish_validation: list[dict[str, object]] = []
        polishing_pair: dict[str, object] | None = None
        polishing_fresh_score_improvement: float | None = None
        pre_polish_best: full.OptimizerSolution | None = (
            final_ranked[0] if final_ranked else None
        )
        polished_solution: full.OptimizerSolution | None = None
        if pre_polish_best is not None:
            print(
                f"Polishing the final R={ladder[-1]} solution from one fixed start",
                file=sys.stderr,
                flush=True,
            )
            polish_fit = fit_chunked_c(
                store,
                args.amplitude,
                ladder[-1],
                args.chunk_replicates,
                args.preconditioner_replicates,
                args.initialization_replicates,
                later_level_fresh_starts,
                1,
                args.polish_iterations,
                args.seed,
                warm_start=None,
                fixed_initial_starts=[pre_polish_best.theta],
            )
            if polish_fit.converged_solutions:
                polished_solution = polish_fit.best
                polish_validation = evaluate_solutions(
                    [pre_polish_best, polished_solution],
                    layout,
                    basis,
                    indices,
                    args.amplitude,
                    args.seed,
                    streams.validation,
                    args.validation_replicates,
                )
                polishing_pair = polish_validation[0][
                    "paired_C_log_density_minus_next_solution"
                ]
                polishing_fresh_score_improvement = max(
                    0.0,
                    -float(polishing_pair["mean"]),
                )
        summary["polishing"] = {
            "fit_stream": streams.fit,
            "verification_stream": streams.validation,
            "maximum_iterations": args.polish_iterations,
            "start_workers": 1,
            "optimizer": (
                None if polish_fit is None else optimizer_summary(polish_fit)
            ),
            "pre_polish_and_polished_validation": polish_validation,
            "pre_polish_minus_polished_paired_validation_score": polishing_pair,
            "fresh_score_improvement_nats_per_neuron": (
                polishing_fresh_score_improvement
            ),
            "threshold_nats_per_neuron": args.polish_improvement,
        }

        terminal_candidate_inputs: list[
            tuple[str, full.OptimizerSolution]
        ] = []
        if polished_solution is not None:
            terminal_candidate_inputs.append(("polished", polished_solution))
        if pre_polish_best is not None:
            terminal_candidate_inputs.append(
                ("pre_polish_best", pre_polish_best)
            )
        if len(final_ranked) >= 2:
            terminal_candidate_inputs.append(
                ("original_second", final_ranked[1])
            )
        ranked_terminal_candidates = rank_terminal_candidates(
            terminal_candidate_inputs
        )
        terminal_candidates = ranked_terminal_candidates[:2]
        terminal_solutions = [
            candidate.solution for candidate in terminal_candidates
        ]
        print(
            f"Evaluating untouched terminal stream with {args.terminal_replicates} replicates",
            file=sys.stderr,
            flush=True,
        )
        terminal = evaluate_solutions(
            terminal_solutions,
            layout,
            basis,
            indices,
            args.amplitude,
            args.seed,
            streams.terminal,
            args.terminal_replicates,
        )
        if len(terminal) != len(terminal_candidates):
            raise AssertionError("Terminal scoring did not return every candidate")
        for result, candidate in zip(terminal, terminal_candidates):
            result["fit_stream_candidate_sources"] = list(candidate.sources)
        terminal_pair = (
            None
            if len(terminal) < 2
            else terminal[0]["paired_C_log_density_minus_next_solution"]
        )
        summary["terminal"] = {
            "stream": streams.terminal,
            "replicates": args.terminal_replicates,
            "selection_rule": (
                "deduplicate_polished_pre_polish_best_and_original_second_by_"
                "parameter_equivalence_then_rank_by_fit_stream_training_objective;_"
                "terminal_scores_never_select_parameters"
            ),
            "parameter_equivalence_tolerances": {
                "absolute": PARAMETER_EQUIVALENCE_ATOL,
                "relative": PARAMETER_EQUIVALENCE_RTOL,
            },
            "raw_candidate_pool": [
                {
                    "source": source,
                    "fit_stream_training_objective": solution.objective,
                }
                for source, solution in terminal_candidate_inputs
            ],
            "deduplicated_fit_objective_ranking": [
                {
                    "rank": rank + 1,
                    "sources": list(candidate.sources),
                    "fit_stream_training_objective": (
                        candidate.solution.objective
                    ),
                    "terminal_scored": rank < 2,
                }
                for rank, candidate in enumerate(ranked_terminal_candidates)
            ],
            "solutions": terminal,
            "best_two_score_paired_difference": terminal_pair,
        }
        if ranked_terminal_candidates:
            checkpoint_polished_solution = (
                None
                if polish_fit is None or not polish_fit.solutions
                else polish_fit.solutions[0]
            )
            executed_rungs = ladder[: len(fits)]
            checkpoint_arrays = build_c_model_state_arrays(
                layout=layout,
                basis=basis,
                indices=indices,
                amplitude=args.amplitude,
                requested_ladder=ladder,
                executed_rung_replicates=executed_rungs,
                validation_replicates=args.validation_replicates,
                terminal_replicates=args.terminal_replicates,
                anchor_replicates=args.anchor_replicates,
                streams=streams,
                seed=args.seed,
                fits=fits,
                pre_polish_best=pre_polish_best,
                polished_solution=checkpoint_polished_solution,
                terminal_candidates=ranked_terminal_candidates,
            )
            checkpoint_path, _ = write_c_model_state_checkpoint(
                args.output_dir,
                checkpoint_arrays,
                layout=layout,
                basis=basis,
                indices=indices,
                amplitude=args.amplitude,
                requested_ladder=ladder,
                executed_rung_replicates=executed_rungs,
                validation_replicates=args.validation_replicates,
                terminal_replicates=args.terminal_replicates,
                anchor_replicates=args.anchor_replicates,
                streams=streams,
                seed=args.seed,
                store=store,
                final_replicates=executed_rungs[-1],
                chunk_replicates=args.chunk_replicates,
            )
            frozen_candidate = ranked_terminal_candidates[0]
            summary["model_state_checkpoint"] = {
                "filename": checkpoint_path.name,
                "schema_version": MODEL_STATE_SCHEMA_VERSION,
                "family": "C",
                "format": "compressed_NPZ_with_finite_float_arrays_only",
                "write_status": (
                    "training_objective_replayed_then_atomic_write_completed"
                ),
                "stream_id_array_order": list(MODEL_STATE_STREAM_NAMES),
                "C_codec_block_array_order": list(C_CODEC_BLOCK_NAMES),
                "terminal_candidate_source_flag_columns": list(
                    TERMINAL_CANDIDATE_SOURCE_NAMES
                ),
                "frozen_theta": {
                    "NPZ_array": "frozen_theta",
                    "identical_terminal_candidate_array": (
                        "terminal_candidate_thetas[0]"
                    ),
                    "zero_based_terminal_candidate_index": 0,
                    "deduplicated_fit_objective_rank": 1,
                    "fit_stream_candidate_sources": list(
                        frozen_candidate.sources
                    ),
                    "fit_stream_training_objective": (
                        frozen_candidate.solution.objective
                    ),
                    "selection_rule": (
                        "minimum_final_fit_stream_training_objective_after_"
                        "parameter_equivalence_deduplication"
                    ),
                    "terminal_scores_used_for_parameter_selection": False,
                },
            }
        else:
            summary["model_state_checkpoint"] = {
                "filename": None,
                "schema_version": MODEL_STATE_SCHEMA_VERSION,
                "family": "C",
                "write_status": "not_written_no_converged_terminal_candidate",
                "frozen_theta": None,
            }

    ladder_complete = len(fits) == len(ladder)
    two_start_pass = ladder_complete and all(
        len(fit.converged_solutions) >= 2 for fit in fits
    )
    boundaries_pass = ladder_complete and all(
        boundary_inactive(fit.best) for fit in fits if fit.converged_solutions
    ) and all(fit.converged_solutions for fit in fits)
    start_stability_pass = ladder_complete and all(
        len(values) >= 2
        and abs(
            float(
                values[0]["paired_C_log_density_minus_next_solution"]["mean"]
            )
        ) < args.start_stability
        and float(
            values[0]["paired_C_log_density_minus_next_solution"]["mcse"]
        ) <= args.target_mcse
        for values in validation_results
    )
    final_level_summary = summary["ladder"][-1]
    final_ladder_pair = final_level_summary[
        "best_vs_previous_level_validation_score_paired_difference"
    ]
    ladder_change: float | None = (
        None
        if final_ladder_pair is None
        else abs(float(final_ladder_pair["mean"]))
    )
    terminal_stability_pass = bool(
        terminal_pair is not None
        and abs(float(terminal_pair["mean"])) < args.terminal_stability
        and float(terminal_pair["mcse"]) <= args.target_mcse
    )
    polishing_pass = bool(
        polish_fit is not None
        and polish_fit.converged_solutions
        and boundary_inactive(polish_fit.best)
        and polishing_fresh_score_improvement is not None
        and polishing_fresh_score_improvement <= args.polish_improvement
        and polishing_pair is not None
        and float(polishing_pair["mcse"]) <= args.target_mcse
    )
    terminal_mcse_pass = bool(
        terminal
        and float(terminal[0]["mcse_truth_minus_C_nats_per_neuron"])
        <= args.target_mcse
    )
    pilot_acceptance = {
        "exact_analytic_anchors_pass": True,
        "all_fit_validation_terminal_streams_are_distinct": len(
            {
                streams.fit,
                streams.validation,
                streams.terminal,
                streams.anchor,
                TRUTH_TEMPLATE_STREAM,
            }
        )
        == 5,
        "SAA_ladder_completed": ladder_complete,
        "at_least_two_physical_KKT_converged_starts_at_every_level": two_start_pass,
        "best_solution_numerical_boundaries_inactive_at_every_level": boundaries_pass,
        "best_two_validation_scores_differ_below_threshold_at_every_level": (
            start_stability_pass
        ),
        "last_ladder_increase_changes_fresh_score_below_threshold_with_precise_pair": (
            ladder_change is not None
            and ladder_change < args.ladder_stability
            and final_ladder_pair is not None
            and float(final_ladder_pair["mcse"]) <= args.target_mcse
        ),
        "fixed_start_polishing_fresh_score_improvement_below_threshold": (
            polishing_pass
        ),
        "best_two_terminal_scores_differ_below_threshold": terminal_stability_pass,
        "terminal_truth_minus_C_MCSE_at_most_target": terminal_mcse_pass,
    }
    acceptance = {
        **pilot_acceptance,
        "recommended_terminal_SAA_ladder_128_512_2048_8192_completed": (
            ladder == (128, 512, 2048, 8192)
        ),
        "fitted_H_population_optimizer_recovery_completed": False,
        "terminal_population_calibration_pass": False,
        "Panel_A_population_calibration_complete": False,
    }
    summary["pilot_acceptance"] = pilot_acceptance
    summary["acceptance"] = acceptance
    summary["last_ladder_validation_change_nats_per_neuron"] = ladder_change
    summary["Panel_A_population_calibration_complete"] = False
    remaining = []
    if ladder != (128, 512, 2048, 8192):
        remaining.append(
            "complete_the_recommended_128_512_2048_8192_population_C_SAA_ladder"
        )
    remaining.extend(
        [
            "demonstrate_fitted_H_population_optimizer_recovery",
            "apply_all_prespecified_Panel_A_acceptance_checks",
        ]
    )
    summary["remaining_before_Panel_A_completion"] = remaining
    summary["status"] = (
        "pass_population_C_SAA_pilot_gates_not_Panel_A_calibration"
        if all(pilot_acceptance.values())
        else "revise_population_C_SAA_pilot_gates"
    )
    write_summary(args.output_dir, summary)


if __name__ == "__main__":
    main()
