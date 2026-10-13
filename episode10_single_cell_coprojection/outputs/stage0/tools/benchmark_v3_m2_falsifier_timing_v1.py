#!/usr/bin/env python3
"""Target-free incremental timing supplement for the EP10 v4 falsifiers.

The program constructs a wholly synthetic 194-row objective in memory.  It
does not load observed target sets, actual K values, protected/final/unassigned
data, SWCs, or an existing M0/M1/M2 result.  The physical fold cache is built
only as the already-paid scaffold required by the frozen Rademacher transform;
it is deliberately outside the four incremental measurements.

Only aggregate resource measurements and contract witness counts are emitted.
No row identity, realized set, coefficient, probability, or fold result can
enter the JSON report.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import v3_m2_permutation_adapter_v1 as permutation  # noqa: E402
import v3_m2_permutation_m1_v1 as permutation_m1  # noqa: E402
import v3_m2_rademacher_adapter_v1 as rademacher  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


EXPECTED_ROWS = 194
EXPECTED_GROUPS = 12
EXPECTED_FOLDS = 12
EXPECTED_RANK = 104
FIXTURE_SEED = 2026092917

UNIT_RADEMACHER_CORE = "Rademacher_core"
UNIT_RADEMACHER_LOW_RANK = "Rademacher_low_rank"
UNIT_RADEMACHER_HYBRID = (
    "Rademacher_hybrid_including_internal_precursor"
)
UNIT_PERMUTATION = "one_outer_only_permutation_M1_plus_configured_M2_fit"
REQUIRED_UNITS = (
    UNIT_RADEMACHER_CORE,
    UNIT_RADEMACHER_LOW_RANK,
    UNIT_RADEMACHER_HYBRID,
    UNIT_PERMUTATION,
)

R_CORE_PROBES = (
    (2026092801, controller.Configuration("lasso", 0.001)),
    (2026092802, controller.Configuration("elastic_net", 0.001, alpha=0.25)),
    (2026092803, controller.Configuration("elastic_net", 0.001, alpha=0.5)),
    (2026092804, controller.Configuration("elastic_net", 0.001, alpha=0.75)),
    (2026092805, controller.Configuration("group_lasso", 0.001)),
)
R_LOW_RANK_PROBE = (
    2026092806,
    controller.Configuration("low_rank", 0.001, rank=8),
)
R_HYBRID_PROBE = (
    2026092807,
    controller.Configuration(
        "sparse_plus_low_rank",
        0.001,
        rank=8,
        sparse_rho=0.001,
        sparse_edges=tuple(range(16)),
    ),
)
R_HYBRID_PRECURSOR_RHO = 0.1
PERMUTATION_CONFIGURATION = controller.Configuration("group_lasso", 0.1)
PERMUTATION_SEED = 2026092813
PROBE_KEYS = tuple(f"core_{index}" for index in range(1, 6)) + (
    "low_rank", "hybrid", "permutation"
)
MAX_WORKERS = len(PROBE_KEYS)

NUMERIC_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

PROHIBITED_REPORT_KEYS = frozenset(
    {
        "cell_key",
        "identifier",
        "k",
        "k_i_obs",
        "observed_index",
        "candidate_index",
        "membership",
        "membership_matrix",
        "realized_set",
        "realized_positive_set",
        "probabilities",
        "predictions",
        "coefficients",
        "beta",
        "fold_results",
        "recipient_to_donor",
        "selected_lambdas",
    }
)


class FalsifierTimingError(RuntimeError):
    """A target-free timing or aggregate-output invariant failed closed."""


@dataclass(frozen=True)
class Measurement:
    """One aggregate measurement with process-lifetime peak-RSS semantics."""

    wall_seconds: float
    cpu_seconds: float
    peak_rss_gib: float
    scratch_gib: float = 0.0

    def __post_init__(self) -> None:
        values = (
            float(self.wall_seconds),
            float(self.cpu_seconds),
            float(self.peak_rss_gib),
            float(self.scratch_gib),
        )
        if (
            not all(math.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 0.0
            or values[2] <= 0.0
            or values[3] < 0.0
        ):
            raise FalsifierTimingError("aggregate measurement is malformed")

    def as_report(self) -> dict[str, Any]:
        return {
            "status": "pass",
            "wall_seconds": float(self.wall_seconds),
            "cpu_seconds": float(self.cpu_seconds),
            "peak_rss_gib": float(self.peak_rss_gib),
            "peak_rss_semantics": "process_lifetime_high_water_conservative",
            "scratch_gib": float(self.scratch_gib),
        }


@dataclass(frozen=True)
class TimedResult:
    result: Any
    measurement: Measurement


@dataclass(frozen=True)
class SyntheticFixture:
    cells: tuple[m1.Cell, ...]
    moments: tuple[m1.UniformCellMoments, ...]
    candidates: m1.CandidateLibrary
    geometry: m1.GeometryEngine
    pairs: m2.PairFeatureLibrary
    theta: np.ndarray


@dataclass(frozen=True)
class ProbeTiming:
    measurement: Measurement
    fold_count: int
    stable_rank_count: int
    precursor_fit_count: int


_T = TypeVar("_T")


def _peak_rss_gib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  Sherlock is Linux, while the
    # alternate branch keeps the helper locally testable.
    divisor = 1024.0**3 if sys.platform == "darwin" else 1024.0**2
    return max(raw / divisor, np.finfo(float).tiny)


def _time_call(function: Callable[[], _T]) -> TimedResult:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = function()
    wall = max(time.perf_counter() - wall_start, np.finfo(float).eps)
    cpu = max(time.process_time() - cpu_start, 0.0)
    return TimedResult(result, Measurement(wall, cpu, _peak_rss_gib(), 0.0))


def componentwise_max(values: Sequence[Measurement]) -> Measurement:
    supplied = tuple(values)
    if not supplied:
        raise FalsifierTimingError("componentwise maximum is empty")
    return Measurement(
        max(value.wall_seconds for value in supplied),
        max(value.cpu_seconds for value in supplied),
        max(value.peak_rss_gib for value in supplied),
        max(value.scratch_gib for value in supplied),
    )


def serial_sum(values: Sequence[Measurement]) -> Measurement:
    supplied = tuple(values)
    if not supplied:
        raise FalsifierTimingError("serial measurement composition is empty")
    return Measurement(
        math.fsum(value.wall_seconds for value in supplied),
        math.fsum(value.cpu_seconds for value in supplied),
        max(value.peak_rss_gib for value in supplied),
        math.fsum(value.scratch_gib for value in supplied),
    )


def repeat_conservative(value: Measurement, count: int) -> Measurement:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise FalsifierTimingError("measurement repetition count is invalid")
    return Measurement(
        value.wall_seconds * count,
        value.cpu_seconds * count,
        value.peak_rss_gib,
        value.scratch_gib * count,
    )


def _canonical_execution_inventory() -> tuple[tuple[str, str], ...]:
    group_position = {group: index for index, group in enumerate(m1.GROUP_ORDER)}

    def identifier_key(value: str) -> tuple[int, str]:
        try:
            return (int(value), value)
        except ValueError:
            return (sys.maxsize, value)

    return tuple(
        sorted(
            controller.FROZEN_CELL_INVENTORY.entries,
            key=lambda entry: (
                group_position[entry[1]],
                identifier_key(entry[0]),
            ),
        )
    )


def build_synthetic_fixture(*, seed: int = FIXTURE_SEED) -> SyntheticFixture:
    """Build the deterministic target-free 194-row timing objective."""

    inventory = _canonical_execution_inventory()
    if (
        len(inventory) != EXPECTED_ROWS
        or len(m1.GROUP_ORDER) != EXPECTED_GROUPS
        or sum(m1.GROUP_SIZES.values()) != EXPECTED_ROWS
    ):
        raise FalsifierTimingError("frozen target-blind dimensions changed")

    candidates = m1.CandidateLibrary()
    pair_geometry = m1.pair_geometry_from_coordinates(
        m1.synthetic_target_coordinates()
    )
    geometry = m1.GeometryEngine(candidates, pair_geometry)
    pairs = m2.PairFeatureLibrary(candidates)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    normal_block = rng.standard_normal((EXPECTED_ROWS, 3))
    sampling_uniforms = rng.random(EXPECTED_ROWS)
    local_by_group = {group: 0 for group in m1.GROUP_ORDER}
    group_ordinal = {group: index for index, group in enumerate(m1.GROUP_ORDER)}

    provisional: list[m1.Cell] = []
    for row_index, (identifier, group) in enumerate(inventory):
        local = local_by_group[group]
        local_by_group[group] += 1
        # Four interior K values give the residual pair space its exact generic
        # rank while leaving repeated (group, segment, K) permutation strata.
        k = 5 + ((local // 2 + group_ordinal[group]) % 4)
        cell = m1._synthetic_cell(
            row_index,
            group,
            k,
            normal_block[row_index],
            float(sampling_uniforms[row_index]),
        )
        provisional.append(
            replace(
                cell,
                identifier=identifier,
                group=group,
                segment_unmapped=local % 2,
                observed_index=0,
            )
        )
    if local_by_group != m1.GROUP_SIZES:
        raise FalsifierTimingError("synthetic fixture group sizes changed")

    provisional_rows = tuple(provisional)
    moments = tuple(geometry.uniform_moments(cell) for cell in provisional_rows)
    prepared = m1.prepare_data(provisional_rows, moments)
    theta = np.asarray(m1.planted_parameters(), dtype=float).copy()
    # The multiplier preserves a moderate global signal after within-stratum
    # permutation, keeping the deterministic qualification fixture away from
    # either ridge-grid boundary without making the outcome nearly degenerate.
    theta *= 2.0
    theta.setflags(write=False)
    cells: list[m1.Cell] = []
    for index, cell in enumerate(provisional_rows):
        observed_index = m1.sample_candidate_index(
            theta,
            prepared,
            index,
            candidates,
            geometry,
            float(sampling_uniforms[index]),
        )
        cells.append(replace(cell, observed_index=int(observed_index)))
    rows = tuple(cells)
    if len(rows) != EXPECTED_ROWS:
        raise FalsifierTimingError("synthetic fixture row count changed")
    return SyntheticFixture(rows, moments, candidates, geometry, pairs, theta)


def build_physical_scaffold(
    fixture: SyntheticFixture,
) -> executor.PreparedTrialCache:
    """Build the already-paid physical transforms required by Rademacher v4."""

    folds = executor.logo_fold_specs(fixture.cells)
    offsets: dict[str, executor.FixedM1Offset] = {}
    for spec in folds:
        prepared = m1.prepare_data(
            fixture.cells,
            fixture.moments,
            spec.training_indexes,
        )
        offsets[spec.held_out_group] = executor.FixedM1Offset(
            held_out_group=spec.held_out_group,
            training_indexes=spec.training_indexes,
            theta=fixture.theta,
            design_transform=prepared.design_transform,
            geometry_scales=prepared.geometry_scales,
            fit_identity=(
                f"target_free_falsifier_timing_physical_fold_{spec.ordinal:02d}"
            ),
        )
    cache = executor.prepare_trial_cache(
        fixture.cells,
        fixture.moments,
        fixture.candidates,
        fixture.geometry,
        fixture.pairs,
        MappingProxyType(offsets),
        mode=executor.SYNTHETIC_TEST_MODE,
        expected_rank=EXPECTED_RANK,
    )
    if len(cache.folds) != EXPECTED_FOLDS:
        raise FalsifierTimingError("physical scaffold fold count changed")
    return cache


def _validate_rademacher_rank(fitted: rademacher.RademacherFoldFit) -> None:
    diagnostics = fitted.transform.diagnostics
    ranks = diagnostics.get("rank_at_tau_over_10_tau_ten_tau")
    if (
        not isinstance(ranks, Mapping)
        or tuple(ranks.get(key) for key in ("tau_over_10", "tau", "ten_tau"))
        != (EXPECTED_RANK, EXPECTED_RANK, EXPECTED_RANK)
        or diagnostics.get("rank_stable") is not True
        or diagnostics.get("expected_rank_match") is not True
    ):
        raise FalsifierTimingError("Rademacher projected rank is not stably 104")


def _score_rademacher_validation(
    beta: np.ndarray,
    fold: executor.PreparedFold,
    prepared: m2.PreparedM2,
    fixture: SyntheticFixture,
    features: rademacher.RademacherFeatureLibrary,
) -> None:
    evaluation = m2.evaluate_m2(
        beta,
        fold.fixed_m1.theta,
        prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        need_predictions=True,
    )
    values = evaluation.log_probability_nats
    if values is None or values.shape != (len(prepared.cells),) or not np.all(
        np.isfinite(values)
    ):
        raise FalsifierTimingError("Rademacher held-out application is malformed")
    baseline = np.asarray(fold.baseline_log_probability_nats, dtype=float)
    if baseline.shape != values.shape or not np.all(np.isfinite(baseline)):
        raise FalsifierTimingError("Rademacher M1 held-out baseline is malformed")
    gain_sum = math.fsum((values - baseline).tolist())
    if not math.isfinite(gain_sum):
        raise FalsifierTimingError("Rademacher M2-minus-M1 gain is nonfinite")


def _prepare_rademacher_fold(
    fixture: SyntheticFixture,
    physical_fold: executor.PreparedFold,
    features: rademacher.RademacherFeatureLibrary,
) -> tuple[rademacher.RademacherFoldFit, m2.PreparedM2]:
    fitted = rademacher.fit_rademacher_fold(
        fixture.cells,
        fixture.moments,
        fixture.candidates,
        fixture.geometry,
        features,
        physical_fold.training,
        fixture.pairs,
        indexes=physical_fold.spec.training_indexes,
        expected_rank=EXPECTED_RANK,
    )
    validation_indexes = physical_fold.spec.validation_indexes
    validation = rademacher.apply_rademacher_validation(
        fitted,
        tuple(fixture.cells[index] for index in validation_indexes),
        tuple(fixture.moments[index] for index in validation_indexes),
    )
    _validate_rademacher_rank(fitted)
    return fitted, validation


def _core_penalty_kind(
    configuration: controller.Configuration,
) -> tuple[str, float]:
    if configuration.family == "lasso":
        return "lasso", 1.0
    if configuration.family == "elastic_net" and configuration.alpha is not None:
        return "elastic_net", float(configuration.alpha)
    if configuration.family == "group_lasso":
        return "group_lasso", 1.0
    raise FalsifierTimingError("Rademacher core probe configuration is invalid")


def _run_rademacher_core_fold(
    fixture: SyntheticFixture,
    physical_fold: executor.PreparedFold,
    features: rademacher.RademacherFeatureLibrary,
    configuration: controller.Configuration,
) -> Mapping[str, int]:
    fitted, validation = _prepare_rademacher_fold(
        fixture, physical_fold, features
    )
    penalty_kind, l1_ratio = _core_penalty_kind(configuration)
    zero_correction = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    zero_correction.setflags(write=False)
    maximum = m2.fold_lambda_maximum(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        penalty_kind,
        l1_ratio=l1_ratio,
        linear_correction=zero_correction,
    )
    mapping = executor.penalty_from_configuration(configuration, maximum)
    result = m2.fit_m2(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        mapping.specification,
        linear_correction=zero_correction,
    )
    _score_rademacher_validation(
        result.beta, physical_fold, validation, fixture, features
    )
    return {"stable_rank": 1, "precursor_fits": 0}


def _run_rademacher_low_rank_fold(
    fixture: SyntheticFixture,
    physical_fold: executor.PreparedFold,
    features: rademacher.RademacherFeatureLibrary,
    configuration: controller.Configuration,
) -> Mapping[str, int]:
    if configuration.family != "low_rank" or configuration.rank is None:
        raise FalsifierTimingError("Rademacher low-rank probe is invalid")
    fitted, validation = _prepare_rademacher_fold(
        fixture, physical_fold, features
    )
    zero_correction = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    zero_correction.setflags(write=False)
    scales = optional.fold_low_rank_scales_synthetic_qualification(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        linear_correction=zero_correction,
    )
    specification = optional.LowRankSpec(
        int(configuration.rank),
        float(configuration.rho),
        scales.lambda_reference,
    )
    result = optional.fit_low_rank_synthetic_qualification(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        specification,
        linear_correction=zero_correction,
    )
    _score_rademacher_validation(
        result.beta, physical_fold, validation, fixture, features
    )
    return {"stable_rank": 1, "precursor_fits": 0}


def _run_rademacher_hybrid_fold(
    fixture: SyntheticFixture,
    physical_fold: executor.PreparedFold,
    features: rademacher.RademacherFeatureLibrary,
    configuration: controller.Configuration,
) -> Mapping[str, int]:
    if (
        configuration.family != "sparse_plus_low_rank"
        or configuration.rank is None
        or configuration.sparse_rho is None
        or len(configuration.sparse_edges) != 16
    ):
        raise FalsifierTimingError("Rademacher hybrid probe is invalid")
    fitted, validation = _prepare_rademacher_fold(
        fixture, physical_fold, features
    )
    zero_correction = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    zero_correction.setflags(write=False)
    scales = optional.fold_low_rank_scales_synthetic_qualification(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        linear_correction=zero_correction,
    )
    precursor_specification = optional.LowRankSpec(
        int(configuration.rank),
        R_HYBRID_PRECURSOR_RHO,
        scales.lambda_reference,
    )
    # Exactly one fresh same-objective precursor is constructed per fold.
    precursor = optional.fit_low_rank_synthetic_qualification(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        precursor_specification,
        linear_correction=zero_correction,
    )
    maximum = m2.fold_lambda_maximum(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        "lasso",
        linear_correction=zero_correction,
    )
    specification = optional.HybridSpec(
        int(configuration.rank),
        float(configuration.rho),
        float(configuration.sparse_rho),
        scales.lambda_reference,
        maximum.value,
        tuple(m2.PAIR_INDEX[index] for index in configuration.sparse_edges),
    )
    result = optional.fit_sparse_plus_low_rank_synthetic_qualification(
        physical_fold.fixed_m1.theta,
        fitted.prepared,
        fixture.candidates,
        fixture.geometry,
        features,
        specification,
        precursor,
        linear_correction=zero_correction,
    )
    _score_rademacher_validation(
        result.beta, physical_fold, validation, fixture, features
    )
    return {"stable_rank": 1, "precursor_fits": 1}


def measure_shared_dictionary_trial(
    seed: int,
    fold_items: Sequence[Any],
    dictionary_factory: Callable[[int], Any],
    fold_runner: Callable[[Any, Any], Mapping[str, int]],
    *,
    progress: Callable[[str, int], None] | None = None,
) -> ProbeTiming:
    """Measure one dictionary and conservatively project its 12 fold calls."""

    items = tuple(fold_items)
    if len(items) != EXPECTED_FOLDS:
        raise FalsifierTimingError("Rademacher probe requires exactly 12 folds")
    emit = progress or (lambda phase, completed: None)
    emit("dictionary_started", 0)
    dictionary_timed = _time_call(lambda: dictionary_factory(int(seed)))
    emit("dictionary_completed", 0)
    dictionary = dictionary_timed.result
    dictionary_identity = id(dictionary)
    fold_measurements: list[Measurement] = []
    stable_rank_count = 0
    precursor_fit_count = 0
    for item in items:
        if id(dictionary) != dictionary_identity:
            raise FalsifierTimingError("Rademacher dictionary identity changed")
        emit("fold_started", len(fold_measurements))
        timed = _time_call(lambda item=item: fold_runner(dictionary, item))
        witness = timed.result
        if not isinstance(witness, Mapping):
            raise FalsifierTimingError("Rademacher fold witness is malformed")
        stable_rank_count += int(witness.get("stable_rank", 0))
        precursor_fit_count += int(witness.get("precursor_fits", 0))
        fold_measurements.append(timed.measurement)
        emit("fold_completed", len(fold_measurements))
    projected_folds = repeat_conservative(
        componentwise_max(fold_measurements), EXPECTED_FOLDS
    )
    return ProbeTiming(
        serial_sum((dictionary_timed.measurement, projected_folds)),
        EXPECTED_FOLDS,
        stable_rank_count,
        precursor_fit_count,
    )


def _rademacher_dictionary(
    fixture: SyntheticFixture, seed: int
) -> rademacher.RademacherFeatureLibrary:
    route = rademacher.RademacherSeedRoute(
        "target_free_qualification", int(seed)
    )
    return rademacher.RademacherFeatureLibrary(fixture.candidates, route)


def benchmark_rademacher_units(
    fixture: SyntheticFixture,
    physical_cache: executor.PreparedTrialCache,
) -> tuple[Mapping[str, Measurement], Mapping[str, Any]]:
    core_probes: list[ProbeTiming] = []
    for seed, configuration in R_CORE_PROBES:
        probe = measure_shared_dictionary_trial(
            seed,
            physical_cache.folds,
            lambda seed, fixture=fixture: _rademacher_dictionary(fixture, seed),
            lambda features, fold, configuration=configuration: (
                _run_rademacher_core_fold(
                    fixture, fold, features, configuration
                )
            ),
        )
        core_probes.append(probe)
        gc.collect()

    low_seed, low_configuration = R_LOW_RANK_PROBE
    low_probe = measure_shared_dictionary_trial(
        low_seed,
        physical_cache.folds,
        lambda seed: _rademacher_dictionary(fixture, seed),
        lambda features, fold: _run_rademacher_low_rank_fold(
            fixture, fold, features, low_configuration
        ),
    )
    gc.collect()

    hybrid_seed, hybrid_configuration = R_HYBRID_PROBE
    hybrid_probe = measure_shared_dictionary_trial(
        hybrid_seed,
        physical_cache.folds,
        lambda seed: _rademacher_dictionary(fixture, seed),
        lambda features, fold: _run_rademacher_hybrid_fold(
            fixture, fold, features, hybrid_configuration
        ),
    )

    return _combine_rademacher_probes((*core_probes, low_probe, hybrid_probe))


def _combine_rademacher_probes(
    supplied: Sequence[ProbeTiming],
) -> tuple[Mapping[str, Measurement], Mapping[str, Any]]:
    all_probes = tuple(supplied)
    if len(all_probes) != 7 or any(
        probe.fold_count != EXPECTED_FOLDS for probe in all_probes
    ):
        raise FalsifierTimingError("Rademacher probe inventory is incomplete")
    core_probes = all_probes[:5]
    low_probe, hybrid_probe = all_probes[5:]
    hybrid_configuration = R_HYBRID_PROBE[1]
    if any(probe.stable_rank_count != EXPECTED_FOLDS for probe in all_probes):
        raise FalsifierTimingError("not every Rademacher fold had stable rank 104")
    if low_probe.precursor_fit_count != 0:
        raise FalsifierTimingError("low-rank probe unexpectedly created a precursor")
    if hybrid_probe.precursor_fit_count != EXPECTED_FOLDS:
        raise FalsifierTimingError(
            "hybrid probe did not create exactly one precursor per fold"
        )
    measurements = {
        UNIT_RADEMACHER_CORE: componentwise_max(
            tuple(probe.measurement for probe in core_probes)
        ),
        UNIT_RADEMACHER_LOW_RANK: low_probe.measurement,
        UNIT_RADEMACHER_HYBRID: hybrid_probe.measurement,
    }
    witness = {
        "core_probe_count": len(core_probes),
        "low_rank_probe_count": 1,
        "hybrid_probe_count": 1,
        "direct_seed_route": "target_free_qualification",
        "frozen_seed_count": len(all_probes),
        "dictionaries_per_probe": 1,
        "folds_per_probe": EXPECTED_FOLDS,
        "one_dictionary_shared_across_each_probe": True,
        "stable_rank_104_folds": EXPECTED_FOLDS * len(all_probes),
        "hybrid_internal_precursor_fits": hybrid_probe.precursor_fit_count,
        "hybrid_internal_precursor_reported_as_trial": False,
        "hybrid_target_rank": int(hybrid_configuration.rank),
        "hybrid_precursor_rank": int(hybrid_configuration.rank),
        "hybrid_precursor_rho": R_HYBRID_PRECURSOR_RHO,
        "hybrid_precursor_route": "rank_changed_mandatory_rho_0.1",
        "same_objective_zero_correction_used": True,
        "physical_normalization_reused": True,
        "fold_aggregation": "componentwise_max_times_12",
        "probe_aggregation": "componentwise_max",
    }
    return measurements, witness


def _coverage_binding() -> controller.CoverageBinding:
    return controller.CoverageBinding(
        tuple((index,) for index in range(controller.COVERAGE_TRIALS)),
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0", 0),
        tuple("0x0.0p+0" for _ in range(controller.PAIR_COUNT)),
    )


def _permutation_trial() -> controller.Trial:
    return controller.Trial(
        valid_index=40,
        role="association",
        kind="permutation",
        phase="postcoverage",
        falsifier=True,
        configuration=PERMUTATION_CONFIGURATION,
        seed=PERMUTATION_SEED,
        cycle=2,
        coverage_binding=_coverage_binding(),
    )


def validate_permutation_accounting(execution: Any) -> int:
    boundary_count = int(execution.boundary_count)
    if (
        execution.warm_path_count != permutation_m1.EXPECTED_WARM_PATHS
        or execution.zero_refit_count != permutation_m1.EXPECTED_ZERO_REFITS
        or execution.full_development_refit_run is not False
    ):
        raise FalsifierTimingError(
            "permutation did not run exact 66 warm and 12 zero-start refits"
        )
    if boundary_count < 0 or boundary_count >= 6:
        raise FalsifierTimingError(
            "permutation fixture failed the inherited M1 boundary gate"
        )
    return boundary_count


def _run_permutation_m2_fold(
    fold: executor.PreparedFold,
    cache: executor.PreparedTrialCache,
) -> Mapping[str, int]:
    configuration = PERMUTATION_CONFIGURATION
    maximum = m2.fold_lambda_maximum(
        fold.fixed_m1.theta,
        fold.training,
        cache.candidates,
        cache.geometry,
        cache.pairs,
        "group_lasso",
    )
    mapping = executor.penalty_from_configuration(configuration, maximum)
    result = m2.fit_m2(
        fold.fixed_m1.theta,
        fold.training,
        cache.candidates,
        cache.geometry,
        cache.pairs,
        mapping.specification,
    )
    evaluation = m2.evaluate_m2(
        result.beta,
        fold.fixed_m1.theta,
        fold.validation,
        cache.candidates,
        cache.geometry,
        cache.pairs,
        need_predictions=True,
    )
    values = evaluation.log_probability_nats
    if values is None or values.shape != (
        len(fold.validation.cells),
    ) or not np.all(np.isfinite(values)):
        raise FalsifierTimingError("permutation M2 held-out application is malformed")
    baseline = np.asarray(fold.baseline_log_probability_nats, dtype=float)
    if baseline.shape != values.shape or not np.all(np.isfinite(baseline)):
        raise FalsifierTimingError("permutation M1 held-out baseline is malformed")
    gain_sum = math.fsum((values - baseline).tolist())
    if not math.isfinite(gain_sum):
        raise FalsifierTimingError("permutation M2-minus-M1 gain is nonfinite")
    return {"completed": 1}


def benchmark_permutation_unit(
    fixture: SyntheticFixture,
    *,
    progress: Callable[[str, int], None] | None = None,
) -> tuple[Measurement, Mapping[str, Any]]:
    emit = progress or (lambda phase, completed: None)
    trial = _permutation_trial()
    emit("assignment_started", 0)
    assignment_timed = _time_call(
        lambda: permutation.build_observed_permutation_assignment(
            fixture.cells, trial
        )
    )
    emit("assignment_completed", 0)
    # The M1 implementation reports per-group tuning events to stdout.  They
    # are useful in production but are deliberately suppressed here so the
    # supplement has aggregate-only output and logs.
    emit("permutation_M1_started", 0)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            m1_timed = _time_call(
                lambda: permutation_m1.fit_permuted_outer_m1(
                    fixture.cells,
                    assignment_timed.result,
                    fixture.candidates,
                    fixture.geometry,
                    identity_prefix="target_free_falsifier_timing_v1",
                )
            )
    execution = m1_timed.result
    boundary_count = validate_permutation_accounting(execution)
    emit("permutation_M1_completed", 0)
    emit("permutation_cache_started", 0)
    cache_timed = _time_call(
        lambda: permutation_m1.prepare_permuted_m2_cache(
            execution,
            fixture.candidates,
            fixture.geometry,
            mode=executor.SYNTHETIC_TEST_MODE,
            execution_inventory=None,
            expected_rank=EXPECTED_RANK,
        )
    )
    cache = cache_timed.result
    if len(cache.folds) != EXPECTED_FOLDS:
        raise FalsifierTimingError("permutation M2 cache fold count changed")
    emit("permutation_cache_completed", 0)
    fold_measurements: list[Measurement] = []
    completed = 0
    for fold in cache.folds:
        emit("permutation_M2_fold_started", completed)
        timed = _time_call(lambda fold=fold: _run_permutation_m2_fold(fold, cache))
        completed += int(timed.result.get("completed", 0))
        fold_measurements.append(timed.measurement)
        emit("permutation_M2_fold_completed", completed)
    if completed != EXPECTED_FOLDS:
        raise FalsifierTimingError("configured permutation M2 folds are incomplete")
    projected_m2 = repeat_conservative(
        componentwise_max(fold_measurements), EXPECTED_FOLDS
    )
    measurement = serial_sum(
        (
            assignment_timed.measurement,
            m1_timed.measurement,
            cache_timed.measurement,
            projected_m2,
        )
    )
    witness = {
        "mapping_count": 1,
        "direct_seed": PERMUTATION_SEED,
        "mapping_shared_by_all_fits": True,
        "warm_path_fits": int(execution.warm_path_count),
        "zero_start_outer_refits": int(execution.zero_refit_count),
        "full_development_M1_fit_run": bool(
            execution.full_development_refit_run
        ),
        "boundary_selections": boundary_count,
        "boundary_gate_maximum_allowed": 5,
        "configured_M2_folds": completed,
        "configured_model_family": PERMUTATION_CONFIGURATION.family,
        "own_fresh_M1_baseline_used": True,
        "fold_aggregation": "componentwise_max_times_12",
    }
    return measurement, witness


def require_slurm_one_thread() -> Mapping[str, str]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise FalsifierTimingError("timing supplement must run inside Slurm")
    try:
        allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", ""))
    except ValueError as error:
        raise FalsifierTimingError("Slurm CPU allocation is malformed") from error
    if allocated_cpus < 1:
        raise FalsifierTimingError("Slurm CPU allocation is empty")
    values: dict[str, str] = {}
    for name in NUMERIC_THREAD_VARIABLES:
        value = os.environ.get(name)
        if value != "1":
            raise FalsifierTimingError(
                f"numeric thread environment is not one for {name}"
            )
        values[name] = value
    return MappingProxyType(values)


def assert_aggregate_safe(value: Any, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FalsifierTimingError("aggregate report contains a nonstring key")
            if key.lower() in PROHIBITED_REPORT_KEYS:
                raise FalsifierTimingError(
                    f"aggregate report contains prohibited key at {path}"
                )
            assert_aggregate_safe(child, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            assert_aggregate_safe(child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise FalsifierTimingError(
        f"aggregate report contains a non-JSON or nonfinite value at {path}"
    )


def build_report(
    measurements: Mapping[str, Measurement],
    rademacher_witness: Mapping[str, Any],
    permutation_witness: Mapping[str, Any],
    thread_values: Mapping[str, str],
) -> Mapping[str, Any]:
    if tuple(measurements) != REQUIRED_UNITS:
        raise FalsifierTimingError("timing supplement unit inventory changed")
    report: dict[str, Any] = {
        "status": "pass",
        "report_kind": "v3_m2_v4_target_free_falsifier_timing_supplement_v1",
        "aggregate_only": True,
        "fixture": {
            "source": "deterministic_in_memory_synthetic",
            "rows": EXPECTED_ROWS,
            "groups": EXPECTED_GROUPS,
            "folds": EXPECTED_FOLDS,
            "expected_residual_rank": EXPECTED_RANK,
            "synthetic_seed": FIXTURE_SEED,
            "real_outcomes_opened": False,
            "actual_K_projection_opened": False,
            "protected_final_unassigned_or_SWC_opened": False,
            "existing_M0_M1_or_M2_result_opened": False,
        },
        "numeric_threads": dict(thread_values),
        "aggregation": {
            "fold_rule": "componentwise_maximum_measured_fold_times_12",
            "probe_rule": "componentwise_maximum_across_exact_probes",
            "serial_components": "wall_cpu_scratch_sum_peak_rss_maximum",
        },
        "measured_units": {
            unit: measurements[unit].as_report() for unit in REQUIRED_UNITS
        },
        "rademacher_witness": dict(rademacher_witness),
        "permutation_witness": dict(permutation_witness),
        "combination_note": (
            "incremental supplement only; combine with the existing valid "
            "core/optional scale report without rerunning its heavy baseline"
        ),
    }
    assert_aggregate_safe(report)
    return report


def _probe_specification(key: str) -> tuple[int, controller.Configuration]:
    if key not in PROBE_KEYS:
        raise FalsifierTimingError("unknown timing probe")
    if key.startswith("core_"):
        return R_CORE_PROBES[PROBE_KEYS.index(key)]
    if key == "low_rank":
        return R_LOW_RANK_PROBE
    if key == "hybrid":
        return R_HYBRID_PROBE
    return PERMUTATION_SEED, PERMUTATION_CONFIGURATION


def _probe_identity(key: str) -> Mapping[str, Any]:
    seed, configuration = _probe_specification(key)
    # These are the frozen scientific inputs, not a new source-hash gate.
    # JSON normalization makes sparse-axis tuples identical after a round trip.
    return json.loads(json.dumps({
        "contract": "v4_target_free_falsifier_timing_v1",
        "probe": key,
        "fixture_seed": FIXTURE_SEED,
        "rows": EXPECTED_ROWS,
        "groups": EXPECTED_GROUPS,
        "folds": EXPECTED_FOLDS,
        "expected_residual_rank": EXPECTED_RANK,
        "direct_seed": seed,
        "configuration": asdict(configuration),
        "hybrid_precursor_rho": R_HYBRID_PRECURSOR_RHO if key == "hybrid" else None,
        "numeric_threads": {name: "1" for name in NUMERIC_THREAD_VARIABLES},
        "fold_aggregation": "componentwise_max_times_12",
        "probe_aggregation": "componentwise_max",
    }))


def _validate_completed_probe(
    key: str, payload: Mapping[str, Any],
) -> tuple[Measurement, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise FalsifierTimingError("completed timing probe is not an object")
    assert_aggregate_safe(payload)
    if (
        set(payload) != {"status", "report_kind", "identity", "measurement", "witness"}
        or payload.get("status") != "pass"
        or payload.get("report_kind") != "v3_m2_falsifier_timing_completed_probe_v1"
        or payload.get("identity") != _probe_identity(key)
    ):
        raise FalsifierTimingError("completed timing probe identity changed")
    measured = payload.get("measurement")
    witness = payload.get("witness")
    if not isinstance(measured, Mapping) or not isinstance(witness, Mapping):
        raise FalsifierTimingError("completed timing probe is malformed")
    try:
        measurement = Measurement(*(float(measured[field]) for field in (
            "wall_seconds", "cpu_seconds", "peak_rss_gib", "scratch_gib"
        )))
    except (KeyError, ValueError, TypeError) as error:
        raise FalsifierTimingError("completed probe measurement is malformed") from error
    if dict(measured) != measurement.as_report():
        raise FalsifierTimingError("completed probe measurement semantics changed")
    if key != "permutation":
        expected = {
            "fold_count": EXPECTED_FOLDS,
            "stable_rank_count": EXPECTED_FOLDS,
            "precursor_fit_count": EXPECTED_FOLDS if key == "hybrid" else 0,
        }
    else:
        boundary = witness.get("boundary_selections")
        if type(boundary) is not int or not 0 <= boundary < 6:
            raise FalsifierTimingError("completed permutation boundary gate failed")
        expected = {
            "mapping_count": 1,
            "direct_seed": PERMUTATION_SEED,
            "mapping_shared_by_all_fits": True,
            "warm_path_fits": permutation_m1.EXPECTED_WARM_PATHS,
            "zero_start_outer_refits": permutation_m1.EXPECTED_ZERO_REFITS,
            "full_development_M1_fit_run": False,
            "boundary_selections": boundary,
            "boundary_gate_maximum_allowed": 5,
            "configured_M2_folds": EXPECTED_FOLDS,
            "configured_model_family": PERMUTATION_CONFIGURATION.family,
            "own_fresh_M1_baseline_used": True,
            "fold_aggregation": "componentwise_max_times_12",
        }
    if dict(witness) != expected:
        raise FalsifierTimingError("completed timing probe witness is incomplete")
    return measurement, witness


def _completed_probe_payload(
    key: str, measurement: Measurement, witness: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = {
        "status": "pass",
        "report_kind": "v3_m2_falsifier_timing_completed_probe_v1",
        "identity": _probe_identity(key),
        "measurement": measurement.as_report(),
        "witness": dict(witness),
    }
    _validate_completed_probe(key, payload)
    return payload


def _probe_failure_payload(
    key: str, phase: str, completed: int, error: BaseException,
) -> Mapping[str, Any]:
    """Transport scalar diagnostics, never a solver exception or fitted state."""
    failure: dict[str, Any] = {
        "status": "failed", "probe": key, "failed_during": phase,
        "completed_fold_count": completed,
        "exception_type": type(error).__name__,
    }
    if isinstance(error, m1.UncertifiedLineSearchFailure):
        # The source exception contains only nested scalar solver telemetry.
        # Its constructor is not compatible with exception unpickling: args
        # has the message but __init__ also requires scalar_context.
        try:
            context = m1._scalar_mapping(error.scalar_context)
            assert_aggregate_safe(context)
            failure["optimizer_scalar_context"] = context
            failure["optimizer_scalar_context_status"] = "retained"
        except (m1.M1Error, FalsifierTimingError, TypeError, ValueError):
            failure["optimizer_scalar_context_status"] = "rejected_nonaggregate"
        kind = getattr(error, "production_fit_kind", None)
        fit_key = getattr(error, "production_fit_key", None)
        if kind == "permutation_in_memory_warm":
            keys = tuple(
                f"leave_two__{left}__{right}"
                for index, left in enumerate(m1.GROUP_ORDER)
                for right in m1.GROUP_ORDER[index + 1:]
            )
        elif kind == "permutation_in_memory_zero":
            keys = tuple(f"outer_refit__{group}" for group in m1.GROUP_ORDER)
        else:
            keys = ()
        if fit_key in keys:
            failure["production_fit_kind"] = kind
            failure["production_fit_ordinal"] = keys.index(fit_key) + 1
    assert_aggregate_safe(failure)
    return failure


def _run_probe(key: str, checkpoint_directory: str) -> Mapping[str, Any]:
    """Own one complete probe and its process-local numerical objects."""
    require_slurm_one_thread()
    root = Path(checkpoint_directory)
    completed_path = root / f"{key}.completed.json"
    if completed_path.exists() or completed_path.is_symlink():
        raise FalsifierTimingError("completed probe already exists; request explicit reuse")
    last_progress = {"phase": "not_started", "completed_fold_count": 0}

    def progress(phase: str, completed: int) -> None:
        last_progress.update(phase=phase, completed_fold_count=completed)
        payload = {
            "status": "incomplete" if phase != "completed" else "pass",
            "probe": key,
            "phase": phase,
            "completed_fold_count": completed,
            "recorded_unix_seconds": time.time(),
        }
        assert_aggregate_safe(payload)
        _write_json_atomic(root / f"{key}.progress.json", payload)

    try:
        progress("fixture_started", 0)
        fixture = build_synthetic_fixture()
        progress("fixture_completed", 0)
        if key == "permutation":
            measurement, witness = benchmark_permutation_unit(fixture, progress=progress)
        else:
            progress("physical_scaffold_started", 0)
            cache = build_physical_scaffold(fixture)
            progress("physical_scaffold_completed", 0)
            seed, configuration = _probe_specification(key)
            runner = (
                _run_rademacher_core_fold if key.startswith("core_") else
                _run_rademacher_low_rank_fold if key == "low_rank" else
                _run_rademacher_hybrid_fold
            )
            probe = measure_shared_dictionary_trial(
                seed,
                cache.folds,
                lambda value: _rademacher_dictionary(fixture, value),
                lambda features, fold: runner(fixture, fold, features, configuration),
                progress=progress,
            )
            measurement = probe.measurement
            witness = {
                "fold_count": probe.fold_count,
                "stable_rank_count": probe.stable_rank_count,
                "precursor_fit_count": probe.precursor_fit_count,
            }
        payload = _completed_probe_payload(key, measurement, witness)
        # Completion is durable before the parent sees a result. Progress I/O
        # occurs after _time_call has stopped, never inside a measured fit.
        _write_json_atomic(completed_path, payload)
        progress("completed", EXPECTED_FOLDS)
        return payload
    except BaseException as error:
        failure = _probe_failure_payload(
            key, last_progress["phase"], last_progress["completed_fold_count"], error,
        )
        _write_json_atomic(root / f"{key}.progress.json", failure)
        # A custom solver exception may not round-trip through multiprocessing.
        # Returning JSON scalars leaves unrelated workers alive and productive.
        return failure


def _assemble_probe_reports(
    payloads: Mapping[str, Mapping[str, Any]], thread_values: Mapping[str, str],
) -> Mapping[str, Any]:
    if set(payloads) != set(PROBE_KEYS):
        raise FalsifierTimingError("all eight complete timing probes are required")
    probes = []
    for key in PROBE_KEYS[:-1]:
        measurement, witness = _validate_completed_probe(key, payloads[key])
        probes.append(ProbeTiming(
            measurement,
            witness["fold_count"], witness["stable_rank_count"], witness["precursor_fit_count"],
        ))
    r_measurements, r_witness = _combine_rademacher_probes(probes)
    permutation_measurement, permutation_witness = _validate_completed_probe(
        "permutation", payloads["permutation"]
    )
    return build_report(
        {**r_measurements, UNIT_PERMUTATION: permutation_measurement},
        r_witness, permutation_witness, thread_values,
    )


def _run_checkpointed_benchmark(
    workers: int, checkpoint_directory: Path, reuse_completed_from: Path | None,
) -> Mapping[str, Any]:
    thread_values = require_slurm_one_thread()
    if (
        type(workers) is not int or not 1 <= workers <= MAX_WORKERS
        or workers > int(os.environ["SLURM_CPUS_PER_TASK"])
    ):
        raise FalsifierTimingError("timing workers exceed the allocation or eight probes")
    root = Path(os.path.abspath(checkpoint_directory))
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FalsifierTimingError("checkpoint directory must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Mapping[str, Any]] = {}
    if reuse_completed_from is not None:
        source = Path(os.path.abspath(reuse_completed_from))
        if source.is_symlink() or not source.is_dir():
            raise FalsifierTimingError("reuse directory must be an existing real directory")
        for key in PROBE_KEYS:
            path = source / f"{key}.completed.json"
            if path.is_symlink():
                raise FalsifierTimingError("completed probe must not be a symlink")
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                _validate_completed_probe(key, payload)
                payloads[key] = payload
                if source != root:
                    destination = root / path.name
                    if destination.exists() or destination.is_symlink():
                        raise FalsifierTimingError("reuse would overwrite a completed probe")
                    _write_json_atomic(destination, payload)
    missing = tuple(key for key in PROBE_KEYS if key not in payloads)
    failures: list[Mapping[str, Any]] = []

    def collect(key: str, result: Mapping[str, Any]) -> None:
        if result.get("status") == "failed":
            assert_aggregate_safe(result)
            if result.get("probe") != key:
                raise FalsifierTimingError("failed timing probe identity changed")
            failures.append(result)
        else:
            _validate_completed_probe(key, result)
            payloads[key] = result

    if workers == 1:
        for key in missing:
            collect(key, _run_probe(key, str(root)))
    elif missing:
        # Spawn keeps mutable caches, identity registries and CPU clocks local.
        # A probe never crosses process boundaries between its twelve folds.
        with ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            pending = {pool.submit(_run_probe, key, str(root)): key for key in missing}
            for future in as_completed(pending):
                key = pending[future]
                collect(key, future.result())
    if failures:
        error = FalsifierTimingError("one or more timing probes failed")
        error.probe_failures = sorted(failures, key=lambda item: PROBE_KEYS.index(item["probe"]))
        error.completed_probe_count = len(payloads)
        raise error
    return _assemble_probe_reports(payloads, thread_values)


def run_benchmark(
    *, workers: int = 1, checkpoint_directory: Path | None = None,
    reuse_completed_from: Path | None = None,
) -> Mapping[str, Any]:
    if checkpoint_directory is not None:
        return _run_checkpointed_benchmark(workers, checkpoint_directory, reuse_completed_from)
    if workers != 1 or reuse_completed_from is not None:
        raise FalsifierTimingError("parallel or resumed timing requires a checkpoint directory")
    thread_values = require_slurm_one_thread()
    fixture = build_synthetic_fixture()
    # This cache is the exact physical source reused by every R transform.  It
    # is an existing-baseline scaffold, not an incremental falsifier unit.
    physical_cache = build_physical_scaffold(fixture)
    r_measurements, r_witness = benchmark_rademacher_units(
        fixture, physical_cache
    )
    permutation_measurement, permutation_witness = benchmark_permutation_unit(
        fixture
    )
    measurements = {
        UNIT_RADEMACHER_CORE: r_measurements[UNIT_RADEMACHER_CORE],
        UNIT_RADEMACHER_LOW_RANK: r_measurements[UNIT_RADEMACHER_LOW_RANK],
        UNIT_RADEMACHER_HYBRID: r_measurements[UNIT_RADEMACHER_HYBRID],
        UNIT_PERMUTATION: permutation_measurement,
    }
    return build_report(
        measurements,
        r_witness,
        permutation_witness,
        thread_values,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = Path(os.path.abspath(path))
    absolute.parent.mkdir(parents=True, exist_ok=True)
    temporary = absolute.with_name(f".{absolute.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FalsifierTimingError("temporary report path already exists")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, absolute)
    finally:
        if temporary.exists():
            temporary.unlink()


def _blocked_report(error: BaseException) -> Mapping[str, Any]:
    report = {
        "status": "blocked",
        "report_kind": "v3_m2_v4_target_free_falsifier_timing_supplement_v1",
        "aggregate_only": True,
        "reason_code": "target_free_timing_execution_failed_closed",
        "exception_type": type(error).__name__,
        "protected_final_unassigned_or_SWC_opened": False,
        "real_outcomes_opened": False,
    }
    if hasattr(error, "probe_failures"):
        report["probe_failures"] = list(error.probe_failures)
        report["completed_probe_count"] = error.completed_probe_count
    assert_aggregate_safe(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-completed-probes", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = run_benchmark(
            workers=arguments.workers,
            checkpoint_directory=arguments.checkpoint_dir or (
                arguments.output.parent / "falsifier_timing_checkpoints"
            ),
            reuse_completed_from=arguments.resume_completed_probes,
        )
    except BaseException as error:
        _write_json_atomic(arguments.output, _blocked_report(error))
        print(
            f"falsifier timing supplement failed closed: {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    _write_json_atomic(arguments.output, report)
    print("EP10 v4 target-free falsifier timing supplement: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
