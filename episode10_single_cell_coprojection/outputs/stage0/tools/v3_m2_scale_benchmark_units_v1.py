#!/usr/bin/env python3
"""Central, membership-blind execution bridge for the EP10 M2 scale benchmark.

The benchmark scaffold issues one process-local session and binds it to one
exact synthetic context.  The first public hook executes every benchmark unit
once; the executor and optional-family hooks then expose disjoint aggregate
views of that same run.  Numerical state remains in memory and no hook returns
cell-level values, candidate indexes, memberships, or fitted coefficients.
"""

from __future__ import annotations

import math
import resource
import sys
import time
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import fit_v3_m1 as m1
import fit_v3_m2 as m2
import fit_v3_m2_optional as optional
import v3_m2_benchmark_replay_v1 as replay
import v3_m2_observed_session_v1 as observed
import v3_m2_optional_diagnostic_v1 as diagnostic
import v3_m2_trial_executor_v1 as executor
import v3_search_controller_v1 as controller


EXPECTED_ROWS = 194
EXPECTED_FOLDS = 12
EXPECTED_RANK = 104
EXPECTED_SEED = 2026092901
REPEAT_OBJECTIVE_ABSOLUTE_TOLERANCE = 1.0e-12
REPEAT_PROBABILITY_MAXIMUM_ABSOLUTE_TOLERANCE = 1.0e-10
REPEAT_PAIR_CONTRAST_MAXIMUM_ABSOLUTE_TOLERANCE = 1.0e-10
RANK_NEIGHBOR_SOURCE_RANK = 6
RANK_NEIGHBOR_TARGET_RANK = 8

UNIT_SYNTHETIC = "synthetic_membership_generation"
UNIT_M0 = "fresh_M0_replay_refit"
UNIT_M1 = "fresh_M1_replay_refit"
UNIT_SETUP = "shared_12_fold_transform_and_offset_cache_setup"
UNIT_REFERENCE = "reference_12_fold_trial"
UNIT_CORE = "core_12_fold_trial"
UNIT_LOW_RANK = "low_rank_12_fold_trial"
UNIT_HYBRID = "hybrid_12_fold_trial"
UNIT_DIAGNOSTIC = "optional_trigger_diagnostic"
UNIT_REFIT = "selected_full_development_refit_pair"
UNIT_REPORT = "controller_selection_and_aggregate_report"

EXECUTOR_UNITS = (
    UNIT_SYNTHETIC,
    UNIT_M0,
    UNIT_M1,
    UNIT_SETUP,
    UNIT_REFERENCE,
    UNIT_CORE,
    UNIT_DIAGNOSTIC,
    UNIT_REFIT,
    UNIT_REPORT,
)
OPTIONAL_UNITS = (UNIT_LOW_RANK, UNIT_HYBRID)
REQUIRED_UNITS = EXECUTOR_UNITS + OPTIONAL_UNITS

_MEASUREMENT_SEAL = object()
_SESSION_SEAL = object()
_RUN_SEAL = object()


class ScaleBenchmarkError(RuntimeError):
    """A central scale-benchmark execution invariant failed closed."""


def _make_weak_identity_registry(label: str) -> tuple[Any, Any]:
    records: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}

    def register(instance: Any, metadata: Any) -> None:
        identity = id(instance)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            current = records.get(identity)
            if current is not None and current[0] is reference:
                records.pop(identity, None)

        records[identity] = (weakref.ref(instance, remove), metadata)

    def lookup(instance: Any) -> Any:
        record = records.get(id(instance))
        if record is None or record[0]() is not instance:
            raise ScaleBenchmarkError(f"{label} was not issued by this bridge")
        return record[1]

    return register, lookup


_register_measurement_identity, _lookup_measurement_identity = (
    _make_weak_identity_registry("aggregate measurement")
)
_register_session_identity, _lookup_session_identity = _make_weak_identity_registry(
    "scale benchmark session"
)
_register_run_identity, _lookup_run_identity = _make_weak_identity_registry(
    "scale benchmark run"
)


@dataclass(frozen=True, init=False)
class AggregateMeasurement:
    """One whole-operation timing with process-lifetime RSS semantics."""

    wall_seconds: float
    cpu_seconds: float
    peak_rss_gib: float
    scratch_gib: float
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        wall_seconds: float,
        cpu_seconds: float,
        peak_rss_gib: float,
        scratch_gib: float = 0.0,
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _MEASUREMENT_SEAL:
            raise ScaleBenchmarkError(
                "aggregate measurements must be issued by the timing bridge"
            )
        values = tuple(
            float(value)
            for value in (wall_seconds, cpu_seconds, peak_rss_gib, scratch_gib)
        )
        if (
            not all(math.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 0.0
            or values[2] <= 0.0
            or values[3] < 0.0
        ):
            raise ScaleBenchmarkError("aggregate timing is malformed or nonfinite")
        object.__setattr__(self, "wall_seconds", values[0])
        object.__setattr__(self, "cpu_seconds", values[1])
        object.__setattr__(self, "peak_rss_gib", values[2])
        object.__setattr__(self, "scratch_gib", values[3])
        object.__setattr__(self, "_seal", _seal)

    def benchmark_unit(self) -> Mapping[str, Any]:
        _require_measurement(self)
        return {
            "status": "pass",
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_gib": self.peak_rss_gib,
            "peak_rss_semantics": "process_lifetime_high_water_conservative",
            "scratch_gib": self.scratch_gib,
            "real_membership_opened": False,
            "existing_M0_artifact_or_telemetry_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        }


def _measurement_integrity(
    measurement: AggregateMeasurement,
) -> tuple[object, ...]:
    return (
        measurement._seal is _MEASUREMENT_SEAL,
        measurement.wall_seconds.hex(),
        measurement.cpu_seconds.hex(),
        measurement.peak_rss_gib.hex(),
        measurement.scratch_gib.hex(),
    )


def _issue_measurement(
    wall_seconds: float,
    cpu_seconds: float,
    peak_rss_gib: float,
    scratch_gib: float = 0.0,
    *,
    scope: str = "internal_operation",
    owner_session: _ScaleBenchmarkSession | None = None,
    payload_snapshot: object | None = None,
) -> AggregateMeasurement:
    measurement = AggregateMeasurement(
        wall_seconds,
        cpu_seconds,
        peak_rss_gib,
        scratch_gib,
        _seal=_MEASUREMENT_SEAL,
    )
    owner_reference = None if owner_session is None else weakref.ref(owner_session)
    _register_measurement_identity(
        measurement,
        (
            scope,
            owner_reference,
            payload_snapshot,
            _measurement_integrity(measurement),
        ),
    )
    return measurement


def _require_measurement(
    measurement: AggregateMeasurement,
    *,
    expected_scope: str | None = None,
    expected_owner: _ScaleBenchmarkSession | None = None,
    expected_payload_snapshot: object | None = None,
) -> AggregateMeasurement:
    if type(measurement) is not AggregateMeasurement:
        raise ScaleBenchmarkError("aggregate measurement has the wrong concrete type")
    (
        scope,
        owner_reference,
        payload_snapshot,
        expected_integrity,
    ) = _lookup_measurement_identity(measurement)
    if _measurement_integrity(measurement) != expected_integrity:
        raise ScaleBenchmarkError("aggregate measurement changed after issuance")
    if expected_scope is not None and scope != expected_scope:
        raise ScaleBenchmarkError("aggregate measurement has the wrong issuance scope")
    if expected_owner is not None and (
        owner_reference is None or owner_reference() is not expected_owner
    ):
        raise ScaleBenchmarkError("aggregate measurement belongs to another session")
    if (
        expected_payload_snapshot is not None
        and payload_snapshot != expected_payload_snapshot
    ):
        raise ScaleBenchmarkError(
            "aggregate measurement belongs to another constructor outcome"
        )
    return measurement


@dataclass(frozen=True)
class _Timed:
    result: Any
    measurement: AggregateMeasurement


@dataclass(frozen=True)
class _PreparedWorkload:
    candidates: m1.CandidateLibrary
    geometry: m1.GeometryEngine
    pairs: m2.PairFeatureLibrary
    moments: tuple[m1.UniformCellMoments, ...]
    fixed_offsets: Mapping[str, executor.FixedM1Offset]
    cache: executor.PreparedTrialCache
    core_executor: executor.CoreTrialExecutor
    linear_corrections: tuple[np.ndarray, ...]
    named_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _OptionalTrialOutcome:
    configuration: controller.Configuration
    gain_bits_per_cell: float
    fits: tuple[Any, ...]


@dataclass(frozen=True)
class _DiagnosticOutcome:
    evidence: controller.CoverageEvidence
    witness: Mapping[str, Any]


@dataclass(frozen=True)
class _FullPrepared:
    fixed_m1_theta: np.ndarray
    prepared: m2.PreparedM2
    linear_correction: np.ndarray


class _WorstCaseObservedLedgerCallback:
    """Score-free callback used only to time the 80-trial session ledger.

    The returned scalar scores deterministically activate the hybrid branch and
    keep improving through the patience phase so the session reaches the exact
    cap.  No fitter, held-out evaluator, or observed outcome is called here;
    numerical trial time is already represented by the separate family units.
    """

    def __call__(self, trial: controller.Trial) -> observed.SyntheticTrialOutcome:
        configuration = trial.configuration
        if trial.role == "reference" or trial.falsifier:
            gain = 0.0
        else:
            gain = 0.01
            if trial.valid_index == 2:
                gain = 0.05
            if (
                configuration is not None
                and configuration.family == "low_rank"
                and configuration.rank == 2
                and configuration.rho == 0.1
            ):
                gain = 0.08
            if (
                configuration is not None
                and trial.kind == "anchor"
                and configuration.family == "sparse_plus_low_rank"
                and configuration.rho == 0.001
                and configuration.sparse_rho == 0.001
            ):
                gain = 0.12
            if trial.kind == "successor" and trial.phase == "patience":
                gain = 0.20 + 0.003 * trial.valid_index

        contrasts = [0.0] * controller.PAIR_COUNT
        if trial.valid_index == 2:
            contrasts[1] = 0.5
            contrasts[7] = 0.3
            contrasts[11] = 0.2
        result = controller.TrialResult(
            trial,
            float(gain),
            1 if trial.role == "association" else 0,
        )
        return observed.SyntheticTrialOutcome(result, tuple(contrasts))


@dataclass(frozen=True, init=False)
class ScaleBenchmarkRun:
    """One issued central run; only aggregate unit views are public."""

    _context_reference: weakref.ReferenceType[Any] = field(repr=False)
    _measurements: Mapping[str, AggregateMeasurement] = field(repr=False)
    _aggregate_witness: Mapping[str, Any] = field(repr=False)
    _numerical_state: tuple[Any, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        context: Any,
        measurements: Mapping[str, AggregateMeasurement],
        aggregate_witness: Mapping[str, Any],
        numerical_state: tuple[Any, ...],
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _RUN_SEAL:
            raise ScaleBenchmarkError("scale benchmark runs are bridge-issued only")
        if set(measurements) != set(REQUIRED_UNITS):
            raise ScaleBenchmarkError("central run has an incomplete unit inventory")
        for value in measurements.values():
            _require_measurement(value)
        object.__setattr__(self, "_context_reference", weakref.ref(context))
        object.__setattr__(
            self,
            "_measurements",
            MappingProxyType(dict(measurements)),
        )
        object.__setattr__(
            self,
            "_aggregate_witness",
            MappingProxyType(dict(aggregate_witness)),
        )
        object.__setattr__(self, "_numerical_state", numerical_state)
        object.__setattr__(self, "_seal", _seal)

    def _unit_view(self, names: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        _require_run(self)
        return {
            name: dict(self._measurements[name].benchmark_unit()) for name in names
        }


def _immutable_snapshot(value: Any) -> object:
    if value is None or isinstance(value, (str, bytes, bool, int, np.integer)):
        return value if not isinstance(value, np.integer) else int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ScaleBenchmarkError("aggregate witness contains a nonfinite float")
        return ("float", number.hex())
    if isinstance(value, tuple):
        return ("tuple", tuple(_immutable_snapshot(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_immutable_snapshot(item) for item in value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ScaleBenchmarkError("aggregate witness keys must be strings")
        return (
            "mapping",
            tuple(
                (key, _immutable_snapshot(value[key])) for key in sorted(value)
            ),
        )
    raise ScaleBenchmarkError("aggregate witness contains a nonaggregate object")


def _run_integrity(run: ScaleBenchmarkRun) -> tuple[object, ...]:
    context = run._context_reference()
    return (
        run._seal is _RUN_SEAL,
        id(context) if context is not None else None,
        tuple(
            (
                name,
                id(measurement),
                _measurement_integrity(_require_measurement(measurement)),
            )
            for name, measurement in sorted(run._measurements.items())
        ),
        _immutable_snapshot(run._aggregate_witness),
        tuple(id(value) for value in run._numerical_state),
    )


def _require_run(run: ScaleBenchmarkRun) -> ScaleBenchmarkRun:
    if type(run) is not ScaleBenchmarkRun:
        raise ScaleBenchmarkError("central benchmark run has the wrong concrete type")
    expected = _lookup_run_identity(run)
    if _run_integrity(run) != expected or run._context_reference() is None:
        raise ScaleBenchmarkError("central benchmark run changed after issuance")
    return run


def _issue_run(
    context: Any,
    measurements: Mapping[str, AggregateMeasurement],
    aggregate_witness: Mapping[str, Any],
    numerical_state: tuple[Any, ...],
) -> ScaleBenchmarkRun:
    run = ScaleBenchmarkRun(
        context,
        measurements,
        aggregate_witness,
        numerical_state,
        _seal=_RUN_SEAL,
    )
    _register_run_identity(run, _run_integrity(run))
    return run


class _ScaleBenchmarkSession:
    """Mutable one-shot holder embedded in exactly one immutable context."""

    __slots__ = (
        "_seal",
        "_context_reference",
        "_context_snapshot",
        "_run",
        "_running",
        "_failed",
        "__weakref__",
    )

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _SESSION_SEAL:
            raise ScaleBenchmarkError("scale benchmark sessions are bridge-issued only")
        self._seal = _seal
        self._context_reference: weakref.ReferenceType[Any] | None = None
        self._context_snapshot: tuple[object, ...] | None = None
        self._run: ScaleBenchmarkRun | None = None
        self._running = False
        self._failed = False

    def __copy__(self) -> None:
        raise ScaleBenchmarkError("scale benchmark sessions cannot be copied")

    def __deepcopy__(self, memo: object) -> None:
        del memo
        raise ScaleBenchmarkError("scale benchmark sessions cannot be copied")

    def __reduce__(self) -> None:
        raise ScaleBenchmarkError("scale benchmark sessions cannot be serialized")


def _session_integrity(session: _ScaleBenchmarkSession) -> tuple[object, ...]:
    context = (
        None
        if session._context_reference is None
        else session._context_reference()
    )
    return (
        session._seal is _SESSION_SEAL,
        id(context) if context is not None else None,
        session._context_snapshot,
        id(session._run) if session._run is not None else None,
        bool(session._running),
        bool(session._failed),
    )


def _require_issued_session(
    session: _ScaleBenchmarkSession,
) -> _ScaleBenchmarkSession:
    if type(session) is not _ScaleBenchmarkSession:
        raise ScaleBenchmarkError("scale benchmark session has the wrong concrete type")
    expected = _lookup_session_identity(session)
    if _session_integrity(session) != expected:
        raise ScaleBenchmarkError("scale benchmark session changed after issuance")
    return session


def _refresh_session_integrity(session: _ScaleBenchmarkSession) -> None:
    _register_session_identity(session, _session_integrity(session))


def _peak_rss_gib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**3 if sys.platform == "darwin" else 1024.0**2
    value = raw / divisor
    if not math.isfinite(value) or value <= 0.0:
        raise ScaleBenchmarkError("process peak RSS measurement is invalid")
    return value


def time_synthetic_constructor(
    session: object,
    pair_geometry: Any,
    operation: Callable[[], Any],
) -> tuple[Any, AggregateMeasurement]:
    """Time one callback and bind timing to that callback's exact outcome."""

    issued_session = _require_issued_session(session)
    if issued_session._context_reference is not None:
        raise ScaleBenchmarkError(
            "synthetic constructor timing must precede final context binding"
        )
    if not callable(operation):
        raise ScaleBenchmarkError("synthetic constructor operation is not callable")
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    outcome = operation()
    try:
        payload_snapshot = _synthetic_outcome_snapshot(
            outcome.cells,
            outcome.membership_matrix,
            pair_geometry,
        )
    except AttributeError as error:
        raise ScaleBenchmarkError(
            "synthetic constructor did not return the required outcome"
        ) from error
    measurement = _issue_measurement(
        time.perf_counter() - wall_started,
        time.process_time() - cpu_started,
        _peak_rss_gib(),
        0.0,
        scope="synthetic_membership_constructor",
        owner_session=issued_session,
        payload_snapshot=payload_snapshot,
    )
    return outcome, measurement


def _time_operation(operation: Callable[[], Any]) -> _Timed:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = operation()
    return _Timed(
        result,
        _issue_measurement(
            time.perf_counter() - wall_started,
            time.process_time() - cpu_started,
            _peak_rss_gib(),
        ),
    )


def _from_replay_timing(value: replay.AggregateTiming) -> AggregateMeasurement:
    if type(value) is not replay.AggregateTiming:
        raise ScaleBenchmarkError("fresh replay timing has the wrong type")
    return _issue_measurement(
        value.wall_seconds,
        value.cpu_seconds,
        value.peak_rss_gib,
        value.scratch_gib,
    )


def _componentwise_max(
    values: Sequence[AggregateMeasurement],
) -> AggregateMeasurement:
    supplied = tuple(values)
    if not supplied:
        raise ScaleBenchmarkError("componentwise timing maximum is empty")
    for value in supplied:
        _require_measurement(value)
    return _issue_measurement(
        max(value.wall_seconds for value in supplied),
        max(value.cpu_seconds for value in supplied),
        max(value.peak_rss_gib for value in supplied),
        max(value.scratch_gib for value in supplied),
    )


def _serial_sum(values: Sequence[AggregateMeasurement]) -> AggregateMeasurement:
    supplied = tuple(values)
    if not supplied:
        raise ScaleBenchmarkError("serial timing composition is empty")
    for value in supplied:
        _require_measurement(value)
    return _issue_measurement(
        math.fsum(value.wall_seconds for value in supplied),
        math.fsum(value.cpu_seconds for value in supplied),
        max(value.peak_rss_gib for value in supplied),
        math.fsum(value.scratch_gib for value in supplied),
    )


def issue_scale_benchmark_session() -> object:
    """Issue one opaque process-local context session."""

    session = _ScaleBenchmarkSession(_seal=_SESSION_SEAL)
    _refresh_session_integrity(session)
    return session


def _array_snapshot(value: Any) -> tuple[object, ...]:
    array = np.asarray(value)
    return (
        id(value),
        tuple(array.shape),
        array.dtype.str,
        tuple(array.strides),
        bool(array.flags.writeable),
        array.tobytes(order="A"),
    )


def _synthetic_outcome_snapshot(
    cells_value: Sequence[Any],
    membership_value: np.ndarray,
    pair_geometry: Any,
) -> tuple[object, ...]:
    cells = tuple(cells_value)
    membership = np.asarray(membership_value)
    if (
        len(cells) != EXPECTED_ROWS
        or membership.shape != (EXPECTED_ROWS, m1.PANEL_SIZE)
        or membership.dtype != np.dtype(np.uint8)
        or membership.flags.writeable
    ):
        raise ScaleBenchmarkError("synthetic constructor outcome is malformed")
    geometry_arrays: list[tuple[object, ...]] = []
    for label in ("distance", "cross_side", "homotopic"):
        if not hasattr(pair_geometry, label):
            raise ScaleBenchmarkError("pair geometry is malformed")
        geometry_arrays.append(_array_snapshot(getattr(pair_geometry, label)))
    cell_snapshots: list[tuple[object, ...]] = []
    for cell in cells:
        try:
            scalar = (
                str(cell.identifier),
                str(cell.group),
                str(cell.layer),
                str(cell.hemisphere),
                tuple(float(value).hex() for value in cell.soma_um),
                int(cell.k),
                int(cell.observed_index),
                int(cell.segment_unmapped),
                float(cell.sampling_uniform).hex(),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ScaleBenchmarkError(
                "synthetic context contains a malformed cell"
            ) from error
        cell_snapshots.append(
            (id(cell), scalar, _array_snapshot(cell.soma_distance_mm))
        )
    return (
        id(cells_value),
        tuple(cell_snapshots),
        _array_snapshot(membership_value),
        id(pair_geometry),
        tuple(geometry_arrays),
        tuple(pair_geometry.ordered_edges),
    )


def _context_integrity_snapshot(
    context: Any,
    session: _ScaleBenchmarkSession,
) -> tuple[object, ...]:
    try:
        cells_value = context.cells
        pair_geometry = context.pair_geometry
        seed = context.synthetic_seed
        folds = context.expected_folds
        rank = context.expected_residual_rank
        generation = context.synthetic_generation_timing
    except (AttributeError, TypeError) as error:
        raise ScaleBenchmarkError("scale benchmark context is malformed") from error
    if (
        seed != EXPECTED_SEED
        or folds != EXPECTED_FOLDS
        or rank != EXPECTED_RANK
    ):
        raise ScaleBenchmarkError("scale benchmark context invariants changed")
    outcome_snapshot = _synthetic_outcome_snapshot(
        cells_value,
        context.membership_matrix,
        pair_geometry,
    )
    _require_measurement(
        generation,
        expected_scope="synthetic_membership_constructor",
        expected_owner=session,
        expected_payload_snapshot=outcome_snapshot,
    )
    return (
        id(context),
        outcome_snapshot,
        int(seed),
        int(folds),
        int(rank),
        id(generation),
        _measurement_integrity(generation),
    )


def _context_session(context: Any) -> _ScaleBenchmarkSession:
    try:
        session = context.scale_benchmark_session
    except AttributeError as error:
        raise ScaleBenchmarkError("benchmark context omitted its exact session") from error
    return _require_issued_session(session)


def bind_scale_benchmark_context(context: Any) -> None:
    """Bind an issued session to exactly one final immutable context."""

    session = _context_session(context)
    if session._context_reference is not None:
        raise ScaleBenchmarkError("scale benchmark session is already bound")
    session._context_reference = weakref.ref(context)
    session._context_snapshot = _context_integrity_snapshot(context, session)
    _refresh_session_integrity(session)


def _require_bound_session(context: Any) -> _ScaleBenchmarkSession:
    session = _context_session(context)
    if (
        session._context_reference is None
        or session._context_reference() is not context
        or session._context_snapshot is None
        or session._context_snapshot
        != _context_integrity_snapshot(context, session)
    ):
        raise ScaleBenchmarkError(
            "benchmark context is unbound, replaced, or changed after binding"
        )
    return session


def _dct_columns(rank: int) -> np.ndarray:
    rows = np.arange(m1.PANEL_SIZE, dtype=float)[:, np.newaxis]
    columns = np.arange(rank, dtype=float)[np.newaxis, :]
    result = np.cos(math.pi * (rows + 0.5) * columns / m1.PANEL_SIZE)
    result[:, 0] *= 1.0 / math.sqrt(float(m1.PANEL_SIZE))
    if rank > 1:
        result[:, 1:] *= math.sqrt(2.0 / float(m1.PANEL_SIZE))
    return result


def _planted_low_rank_state() -> optional.LowRankState:
    rank = 8
    signs = np.where(np.arange(rank) % 2 == 0, 1.0, -1.0)
    magnitudes = 0.08 / np.sqrt(np.arange(1, rank + 1, dtype=float))
    return optional.LowRankState(_dct_columns(rank), signs * magnitudes)


def _planted_hybrid_state(
    named_edges: tuple[tuple[int, int], ...],
) -> tuple[optional.HybridState, optional.HybridSpec]:
    sparse = 0.045 * np.where(
        np.arange(len(named_edges)) % 2 == 0,
        1.0,
        -1.0,
    )
    specification = optional.HybridSpec(
        8,
        1.0,
        1.0,
        1.0,
        1.0,
        named_edges,
    )
    return optional.HybridState(_planted_low_rank_state(), sparse), specification


def _adapt_fixed_offsets(
    value: replay.M1ReplayResult,
) -> Mapping[str, executor.FixedM1Offset]:
    if type(value) is not replay.M1ReplayResult:
        raise ScaleBenchmarkError("fresh M1 replay result has the wrong type")
    offsets: dict[str, executor.FixedM1Offset] = {}
    for group in m1.GROUP_ORDER:
        source = value.outer_offsets[group]
        if type(source) is not replay.FoldBoundM1Offset:
            raise ScaleBenchmarkError("fresh M1 fold offset is malformed")
        offsets[group] = executor.FixedM1Offset(
            held_out_group=source.held_out_group,
            training_indexes=source.training_indexes,
            theta=source.theta,
            design_transform=source.design_transform,
            geometry_scales=source.geometry_scales,
            fit_identity=source.fit_identity,
        )
    return MappingProxyType(offsets)


def _build_prepared_workload(
    context: Any,
    m1_result: replay.M1ReplayResult,
) -> _PreparedWorkload:
    candidates = m1.CandidateLibrary()
    geometry = m1.GeometryEngine(candidates, context.pair_geometry)
    pairs = m2.PairFeatureLibrary(candidates)
    cells = tuple(context.cells)
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    offsets = _adapt_fixed_offsets(m1_result)
    cache = executor.prepare_trial_cache(
        cells,
        moments,
        candidates,
        geometry,
        pairs,
        offsets,
        mode=executor.SYNTHETIC_TEST_MODE,
        expected_rank=EXPECTED_RANK,
    )
    named_edges = tuple(m2.PAIR_INDEX[:16])
    planted_state, coordinate_specification = _planted_hybrid_state(named_edges)
    corrections: list[np.ndarray] = []
    for fold in cache.folds:
        planted_beta = optional.hybrid_effective_beta(
            planted_state,
            coordinate_specification,
            fold.training.transform,
        )
        correction = m2.population_linear_correction(
            planted_beta,
            fold.fixed_m1.theta,
            fold.training,
            candidates,
            geometry,
            pairs,
        )
        frozen = np.array(correction, dtype=float, copy=True)
        if frozen.shape != (m2.PAIR_DIMENSION,) or not np.all(np.isfinite(frozen)):
            raise ScaleBenchmarkError("fold synthetic correction is malformed")
        frozen.setflags(write=False)
        corrections.append(frozen)
    return _PreparedWorkload(
        candidates,
        geometry,
        pairs,
        moments,
        offsets,
        cache,
        executor.CoreTrialExecutor(cache),
        tuple(corrections),
        named_edges,
    )


def _validation_gain(
    beta: np.ndarray,
    fold: executor.PreparedFold,
    workload: _PreparedWorkload,
) -> float:
    evaluation = m2.evaluate_m2(
        beta,
        fold.fixed_m1.theta,
        fold.validation,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        need_predictions=True,
    )
    if evaluation.log_probability_nats is None:
        raise ScaleBenchmarkError("optional held-out probabilities were omitted")
    gains = (
        np.asarray(evaluation.log_probability_nats, dtype=float)
        - fold.baseline_log_probability_nats
    ) / math.log(2.0)
    if not np.all(np.isfinite(gains)):
        raise ScaleBenchmarkError("optional held-out gains are nonfinite")
    return math.fsum(float(value) for value in gains) / gains.size


def _run_low_rank_trial(
    configuration: controller.Configuration,
    workload: _PreparedWorkload,
) -> _OptionalTrialOutcome:
    if (
        configuration.family != "low_rank"
        or configuration.rank not in optional.LOW_RANKS
    ):
        raise ScaleBenchmarkError("low-rank benchmark configuration is malformed")
    fits: list[optional.LowRankOptimizerResult] = []
    gains: list[float] = []
    for fold, correction in zip(
        workload.cache.folds,
        workload.linear_corrections,
        strict=True,
    ):
        scales = optional.fold_low_rank_scales_synthetic_qualification(
            fold.fixed_m1.theta,
            fold.training,
            workload.candidates,
            workload.geometry,
            workload.pairs,
            linear_correction=correction,
        )
        specification = optional.LowRankSpec(
            int(configuration.rank),
            float(configuration.rho),
            scales.lambda_reference,
        )
        fit = optional.fit_low_rank_synthetic_qualification(
            fold.fixed_m1.theta,
            fold.training,
            workload.candidates,
            workload.geometry,
            workload.pairs,
            specification,
            linear_correction=correction,
        )
        fits.append(fit)
        gains.append(_validation_gain(fit.beta, fold, workload))
    return _OptionalTrialOutcome(
        configuration,
        math.fsum(gains) / EXPECTED_FOLDS,
        tuple(fits),
    )


def _score_selected_low_rank(
    outcomes: Sequence[_OptionalTrialOutcome],
) -> _OptionalTrialOutcome:
    supplied = tuple(outcomes)
    expected = controller.low_rank_anchor_configurations()
    if tuple(outcome.configuration for outcome in supplied) != expected:
        raise ScaleBenchmarkError("low-rank benchmark anchor inventory changed")
    return min(
        supplied,
        key=lambda outcome: (
            -outcome.gain_bits_per_cell,
            *controller.configuration_order_key(outcome.configuration),
        ),
    )


def _hybrid_configurations(
    rank: int,
    named_edges: tuple[tuple[int, int], ...],
) -> tuple[controller.Configuration, ...]:
    indexes = tuple(m2.PAIR_TO_INDEX[edge] for edge in named_edges)
    return tuple(
        controller.Configuration(
            "sparse_plus_low_rank",
            rho_low,
            rank=rank,
            sparse_rho=rho_sparse,
            sparse_edges=indexes,
        )
        for rho_low, rho_sparse in (
            (1.0, 1.0),
            (1.0, 0.001),
            (0.001, 1.0),
            (0.001, 0.001),
        )
    )


def _run_hybrid_trial(
    configuration: controller.Configuration,
    selected_low_rank: _OptionalTrialOutcome,
    workload: _PreparedWorkload,
) -> _OptionalTrialOutcome:
    if (
        configuration.family != "sparse_plus_low_rank"
        or configuration.rank != selected_low_rank.configuration.rank
        or len(configuration.sparse_edges) != 16
    ):
        raise ScaleBenchmarkError("hybrid benchmark configuration is malformed")
    fits: list[optional.HybridOptimizerResult] = []
    gains: list[float] = []
    for fold, correction, anchor in zip(
        workload.cache.folds,
        workload.linear_corrections,
        selected_low_rank.fits,
        strict=True,
    ):
        scales = optional.fold_low_rank_scales_synthetic_qualification(
            fold.fixed_m1.theta,
            fold.training,
            workload.candidates,
            workload.geometry,
            workload.pairs,
            linear_correction=correction,
        )
        lasso = m2.fold_lambda_maximum(
            fold.fixed_m1.theta,
            fold.training,
            workload.candidates,
            workload.geometry,
            workload.pairs,
            "lasso",
            linear_correction=correction,
        )
        specification = optional.HybridSpec(
            int(configuration.rank),
            float(configuration.rho),
            float(configuration.sparse_rho),
            scales.lambda_reference,
            lasso.value,
            workload.named_edges,
        )
        fit = optional.fit_sparse_plus_low_rank_synthetic_qualification(
            fold.fixed_m1.theta,
            fold.training,
            workload.candidates,
            workload.geometry,
            workload.pairs,
            specification,
            anchor,
            linear_correction=correction,
        )
        fits.append(fit)
        gains.append(_validation_gain(fit.beta, fold, workload))
    return _OptionalTrialOutcome(
        configuration,
        math.fsum(gains) / EXPECTED_FOLDS,
        tuple(fits),
    )


def _diagnose_optional_trigger(
    context: Any,
    workload: _PreparedWorkload,
    reference_execution: executor.TrialExecution,
    core_executions: Sequence[executor.TrialExecution],
) -> _DiagnosticOutcome:
    if len(core_executions) != 15:
        raise ScaleBenchmarkError("diagnostic requires all 15 core anchors")
    winner = controller.choose_incumbent(
        tuple(execution.result for execution in core_executions)
    )
    winner_execution = next(
        execution for execution in core_executions if execution.result is winner
    )
    rows: list[diagnostic.PairResidualRow] = []
    membership = np.asarray(context.membership_matrix, dtype=np.uint8)
    for fold in workload.cache.folds:
        for local_index, global_index in enumerate(fold.spec.validation_indexes):
            cell = context.cells[global_index]
            rows.append(
                diagnostic.PairResidualRow(
                    str(cell.identifier),
                    str(cell.group),
                    int(cell.k),
                    membership[global_index],
                    fold.baseline_pair_probabilities[local_index],
                )
            )
    result = diagnostic.diagnose_optional_families(rows)
    localized = controller.localized_sparse_selection(
        winner_execution.equal_group_pair_contrasts
    )
    activation = controller.optional_activation(result.diagnostics, localized)
    evidence = controller.bind_coverage_evidence(
        (
            reference_execution.result,
            *(execution.result for execution in core_executions),
        ),
        result.diagnostics,
        winner_execution.equal_group_pair_contrasts,
    )
    if (
        evidence.diagnostic != result.diagnostics
        or evidence.equal_group_pair_contrasts
        != winner_execution.equal_group_pair_contrasts
        or evidence.localized != localized
        or evidence.activation != activation
        or len(evidence.core_results) != controller.COVERAGE_TRIALS
        or evidence.core_results[-1].coverage_binding is not evidence.binding
    ):
        raise ScaleBenchmarkError("diagnostic coverage evidence failed exact binding")
    return _DiagnosticOutcome(
        evidence,
        {
            "status": "pass",
            "core_winner_configuration": winner.trial.configuration.key,
            "localized_edge_count": len(localized.edges),
            "low_rank_triggered": activation.low_rank,
            "hybrid_triggered": activation.sparse_plus_low_rank,
            "coverage_evidence_bound": True,
        },
    )


def _full_prepared(
    context: Any,
    workload: _PreparedWorkload,
    full_fit: replay.FullDevelopmentM1Fit,
) -> _FullPrepared:
    if type(full_fit) is not replay.FullDevelopmentM1Fit:
        raise ScaleBenchmarkError("fresh full-development M1 fit is malformed")
    transform, prepared = m2.fit_residual_transform(
        tuple(context.cells),
        workload.moments,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        tuple(range(EXPECTED_ROWS)),
        expected_rank=EXPECTED_RANK,
    )
    if (
        prepared.transform is not transform
        or full_fit.training_indexes != tuple(range(EXPECTED_ROWS))
        or not np.array_equal(
            transform.design_transform.soma_mean,
            full_fit.design_transform.soma_mean,
        )
        or not np.array_equal(
            transform.design_transform.soma_scale,
            full_fit.design_transform.soma_scale,
        )
        or not np.array_equal(transform.geometry_scales, full_fit.geometry_scales)
    ):
        raise ScaleBenchmarkError(
            "full-development M1 coordinates do not match the full M2 transform"
        )
    planted_state, coordinate_specification = _planted_hybrid_state(
        workload.named_edges
    )
    planted_beta = optional.hybrid_effective_beta(
        planted_state,
        coordinate_specification,
        prepared.transform,
    )
    correction = m2.population_linear_correction(
        planted_beta,
        full_fit.theta,
        prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
    )
    frozen = np.array(correction, dtype=float, copy=True)
    frozen.setflags(write=False)
    return _FullPrepared(full_fit.theta, prepared, frozen)


def _equal_group_pair_contrast(
    beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    workload: _PreparedWorkload,
) -> np.ndarray:
    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    by_group: dict[str, list[np.ndarray]] = {
        group: [] for group in m1.GROUP_ORDER
    }
    for cell_index, cell in enumerate(prepared.cells):
        fitted = m2.candidate_distribution(
            beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            workload.candidates,
            workload.geometry,
            workload.pairs,
        )
        baseline = m2.candidate_distribution(
            zero,
            fixed_m1_theta,
            prepared,
            cell_index,
            workload.candidates,
            workload.geometry,
            workload.pairs,
        )
        indicators = workload.pairs.basis(cell.k).indicators
        by_group[cell.group].append(
            fitted.probabilities @ indicators
            - baseline.probabilities @ indicators
        )
    group_means: list[np.ndarray] = []
    for group in m1.GROUP_ORDER:
        rows = by_group[group]
        if not rows:
            raise ScaleBenchmarkError("full refit pair contrast missed a group")
        group_means.append(
            np.asarray(
                [
                    math.fsum(float(row[index]) for row in rows) / len(rows)
                    for index in range(m2.PAIR_DIMENSION)
                ],
                dtype=float,
            )
        )
    result = np.asarray(
        [
            math.fsum(float(row[index]) for row in group_means) / EXPECTED_FOLDS
            for index in range(m2.PAIR_DIMENSION)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(result)):
        raise ScaleBenchmarkError("full refit pair contrast is nonfinite")
    return result


def _validate_repeat(
    cold: Any,
    warm: Any,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    workload: _PreparedWorkload,
) -> Mapping[str, float]:
    objective_error = abs(float(cold.objective) - float(warm.objective))
    comparison = m2.compare_conditional_probabilities(
        cold.beta,
        warm.beta,
        fixed_m1_theta,
        prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
    )
    probability_error = float(
        comparison["maximum_candidate_probability_absolute_error"]
    )
    cold_contrast = _equal_group_pair_contrast(
        cold.beta, fixed_m1_theta, prepared, workload
    )
    warm_contrast = _equal_group_pair_contrast(
        warm.beta, fixed_m1_theta, prepared, workload
    )
    contrast_error = float(np.max(np.abs(cold_contrast - warm_contrast)))
    if (
        not all(
            math.isfinite(value)
            for value in (objective_error, probability_error, contrast_error)
        )
        or objective_error > REPEAT_OBJECTIVE_ABSOLUTE_TOLERANCE
        or probability_error > REPEAT_PROBABILITY_MAXIMUM_ABSOLUTE_TOLERANCE
        or contrast_error > REPEAT_PAIR_CONTRAST_MAXIMUM_ABSOLUTE_TOLERANCE
    ):
        raise ScaleBenchmarkError(
            "full-development cold/warm repeat exceeded a frozen tolerance"
        )
    return {
        "objective_absolute_difference": objective_error,
        "candidate_probability_maximum_absolute_difference": probability_error,
        "pair_contrast_maximum_absolute_difference": contrast_error,
    }


def _core_full_pair(
    configuration: controller.Configuration,
    full: _FullPrepared,
    workload: _PreparedWorkload,
) -> Mapping[str, float]:
    ratio = float(configuration.alpha) if configuration.family == "elastic_net" else 1.0
    maximum = m2.fold_lambda_maximum(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        configuration.family,
        l1_ratio=ratio,
        linear_correction=full.linear_correction,
    )
    specification = executor.penalty_from_configuration(
        configuration, maximum
    ).specification
    cold = m2.fit_m2(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        specification,
        initial=None,
        linear_correction=full.linear_correction,
    )
    warm = m2.fit_m2(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        specification,
        initial=cold.beta,
        linear_correction=full.linear_correction,
    )
    return _validate_repeat(
        cold,
        warm,
        full.fixed_m1_theta,
        full.prepared,
        workload,
    )


def _low_rank_full_pair(
    configuration: controller.Configuration,
    full: _FullPrepared,
    workload: _PreparedWorkload,
) -> Mapping[str, float]:
    scales = optional.fold_low_rank_scales_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        linear_correction=full.linear_correction,
    )
    specification = optional.LowRankSpec(
        int(configuration.rank),
        float(configuration.rho),
        scales.lambda_reference,
    )
    cold = optional.fit_low_rank_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        specification,
        linear_correction=full.linear_correction,
    )
    warm = optional.refit_low_rank_from_converged_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        specification,
        cold,
        linear_correction=full.linear_correction,
    )
    if warm.telemetry.get("initialization") != (
        "exact_registered_converged_low_rank_result"
    ):
        raise ScaleBenchmarkError("low-rank warm repeat did not use the cold result")
    return _validate_repeat(
        cold,
        warm,
        full.fixed_m1_theta,
        full.prepared,
        workload,
    )


def _hybrid_full_pair(
    configuration: controller.Configuration,
    selected_low_rank_configuration: controller.Configuration,
    full: _FullPrepared,
    workload: _PreparedWorkload,
) -> Mapping[str, float]:
    scales = optional.fold_low_rank_scales_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        linear_correction=full.linear_correction,
    )
    anchor_specification = optional.LowRankSpec(
        int(selected_low_rank_configuration.rank),
        float(selected_low_rank_configuration.rho),
        scales.lambda_reference,
    )
    selected_anchor = optional.fit_low_rank_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        anchor_specification,
        linear_correction=full.linear_correction,
    )
    lasso = m2.fold_lambda_maximum(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        "lasso",
        linear_correction=full.linear_correction,
    )
    specification = optional.HybridSpec(
        int(configuration.rank),
        float(configuration.rho),
        float(configuration.sparse_rho),
        scales.lambda_reference,
        lasso.value,
        workload.named_edges,
    )
    cold = optional.fit_sparse_plus_low_rank_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        specification,
        selected_anchor,
        linear_correction=full.linear_correction,
    )
    warm = optional.refit_sparse_plus_low_rank_from_converged_synthetic_qualification(
        full.fixed_m1_theta,
        full.prepared,
        workload.candidates,
        workload.geometry,
        workload.pairs,
        specification,
        selected_anchor,
        cold,
        linear_correction=full.linear_correction,
    )
    if (
        warm.telemetry.get("initialization")
        != "exact_registered_converged_hybrid_result"
        or float(warm.telemetry.get("initial_sparse_lipschitz", math.nan))
        != float(cold.sparse_lipschitz)
    ):
        raise ScaleBenchmarkError("hybrid warm repeat did not carry the cold state")
    return _validate_repeat(
        cold,
        warm,
        full.fixed_m1_theta,
        full.prepared,
        workload,
    )


def _rank_neighbor_full_refit_configurations(
    representative_hybrid: controller.Configuration,
) -> tuple[controller.Configuration, controller.Configuration]:
    """Return the fixed maximum-rank v3 neighbor target and its precursor.

    This target-free timing scenario represents the ordered-lattice step
    r_star=6 -> r=8.  The target keeps the measured hybrid penalty corner and
    sparse edges, while its freshly fit precursor is necessarily (rank 8,
    rho=.1), never the rank-6 winner and never a padded or truncated state.
    """

    if (
        type(representative_hybrid) is not controller.Configuration
        or representative_hybrid.family != "sparse_plus_low_rank"
        or representative_hybrid.rho not in optional.RHO_GRID
        or representative_hybrid.sparse_rho not in optional.RHO_GRID
        or len(representative_hybrid.sparse_edges) != 16
    ):
        raise ScaleBenchmarkError(
            "rank-neighbor full-refit representative is malformed"
        )
    target = controller.Configuration(
        "sparse_plus_low_rank",
        float(representative_hybrid.rho),
        rank=RANK_NEIGHBOR_TARGET_RANK,
        sparse_rho=float(representative_hybrid.sparse_rho),
        sparse_edges=tuple(representative_hybrid.sparse_edges),
    )
    precursor = controller.Configuration(
        "low_rank",
        0.1,
        rank=RANK_NEIGHBOR_TARGET_RANK,
    )
    return target, precursor


def _slowest_by_wall(
    configurations: Sequence[controller.Configuration],
    timings: Sequence[AggregateMeasurement],
) -> controller.Configuration:
    pairs = tuple(zip(configurations, timings, strict=True))
    if not pairs:
        raise ScaleBenchmarkError("slowest benchmark configuration inventory is empty")
    winner_configuration, winner_timing = pairs[0]
    for configuration, timing in pairs[1:]:
        if timing.wall_seconds > winner_timing.wall_seconds:
            winner_configuration, winner_timing = configuration, timing
    return winner_configuration


def _run_report_unit(workload: _PreparedWorkload) -> Mapping[str, object]:
    """Time cap-80 chronology bookkeeping plus controller report replay.

    The synthetic session callback performs no fit.  Consequently this unit
    accounts for history/cache reauthentication, stopping, terminal selection,
    and report conformance without double-counting any family-trial numerical
    work measured by the reference/core/optional units.
    """

    owner = executor.CoreTrialExecutor(workload.cache)
    session = observed.start_synthetic_qualification_session(owner)
    callback = _WorstCaseObservedLedgerCallback()
    authority = observed.issue_synthetic_qualification_authority(
        session,
        callback,
        scopes=("core", "optional", "falsifier"),
        diagnostic_override=controller.OptionalDiagnostics(0.40, 16.0, 0.80, 9),
    )
    for expected_index in range(1, controller.MAXIMUM_VALID_TRIALS + 1):
        scored = session.run_next_synthetic_qualification(authority)
        if scored.trial.valid_index != expected_index:
            raise ScaleBenchmarkError(
                "observed-session bookkeeping history lost chronological order"
            )
    if (
        session.status != observed.INCOMPLETE_BUDGET_STOP
        or session.scientific_conclusion != observed.INCOMPLETE_CONCLUSION
        or session.valid_trial_count != controller.MAXIMUM_VALID_TRIALS
        or session.invalid_attempt_count != 0
        or session.next_trial is not None
    ):
        raise ScaleBenchmarkError(
            "observed-session bookkeeping did not reach the exact cap disposition"
        )
    terminal = observed.authorize_terminal_selection(session)
    winner = observed.require_terminal_selection(owner, terminal)
    if (
        terminal.disposition != observed.INCOMPLETE_BUDGET_STOP
        or terminal.scientific_conclusion != observed.INCOMPLETE_CONCLUSION
        or terminal.valid_trial_count != controller.MAXIMUM_VALID_TRIALS
        or terminal.invalid_attempt_count != 0
        or winner is not terminal.winner_result
    ):
        raise ScaleBenchmarkError(
            "observed-session terminal report failed exact-cap reauthentication"
        )

    result = controller.run_target_free_replay_conformance(1, "hybrid")
    if (
        result.get("replicate_index") != 1
        or result.get("optional_branch") != "hybrid"
        or result.get("started_empty") is not True
        or result.get("all_required_branches_replayed") is not True
        or result.get("substream_determinism_verified") is not True
    ):
        raise ScaleBenchmarkError("controller report replay conformance did not pass")
    return {
        "observed_session": {
            "valid_trial_count": controller.MAXIMUM_VALID_TRIALS,
            "invalid_attempt_count": 0,
            "disposition": observed.INCOMPLETE_BUDGET_STOP,
            "scientific_conclusion": observed.INCOMPLETE_CONCLUSION,
            "numerical_fits_invoked_by_callback": 0,
            "synthetic_ledger_is_not_production_coverage_authority": True,
        },
        "controller_replay_conformance": result,
    }


def _execute_scale_benchmark_run(context: Any) -> ScaleBenchmarkRun:
    generation_receipt = replay.validate_synthetic_generation(context)
    fresh = replay.run_fresh_replay_benchmark(context)
    if (
        fresh.m0.result.generation_receipt != generation_receipt
        or fresh.m1.result.generation_receipt != generation_receipt
    ):
        raise ScaleBenchmarkError("fresh replay results do not share one generation receipt")
    m0_measurement = _from_replay_timing(fresh.m0.timing)
    m1_measurement = _from_replay_timing(fresh.m1.timing)

    setup = _time_operation(
        lambda: _build_prepared_workload(context, fresh.m1.result)
    )
    workload = setup.result
    coverage_trials = controller.coverage_trials()
    if len(coverage_trials) != 16:
        raise ScaleBenchmarkError("core benchmark coverage inventory changed")
    reference = _time_operation(
        lambda: workload.core_executor.execute(coverage_trials[0])
    )

    core_timed: list[_Timed] = []
    for trial in coverage_trials[1:]:
        core_timed.append(
            _time_operation(lambda trial=trial: workload.core_executor.execute(trial))
        )
    if len(core_timed) != 15:
        raise ScaleBenchmarkError("core benchmark anchor inventory changed")
    core_executions = tuple(timed.result for timed in core_timed)
    core_measurements = tuple(timed.measurement for timed in core_timed)
    core_configurations = tuple(
        execution.trial.configuration for execution in core_executions
    )
    if any(configuration is None for configuration in core_configurations):
        raise ScaleBenchmarkError("core anchor omitted its configuration")
    core_configurations = tuple(
        configuration for configuration in core_configurations if configuration is not None
    )

    diagnosed = _time_operation(
        lambda: _diagnose_optional_trigger(
            context,
            workload,
            reference.result,
            core_executions,
        )
    )

    low_configurations = controller.low_rank_anchor_configurations()
    if len(low_configurations) != 8:
        raise ScaleBenchmarkError("low-rank benchmark anchor inventory changed")
    low_timed = tuple(
        _time_operation(
            lambda configuration=configuration: _run_low_rank_trial(
                configuration, workload
            )
        )
        for configuration in low_configurations
    )
    low_outcomes = tuple(timed.result for timed in low_timed)
    selected_low_rank = _score_selected_low_rank(low_outcomes)

    hybrid_configurations = _hybrid_configurations(
        int(selected_low_rank.configuration.rank),
        workload.named_edges,
    )
    if len(hybrid_configurations) != 4:
        raise ScaleBenchmarkError("hybrid benchmark anchor inventory changed")
    hybrid_timed = tuple(
        _time_operation(
            lambda configuration=configuration: _run_hybrid_trial(
                configuration,
                selected_low_rank,
                workload,
            )
        )
        for configuration in hybrid_configurations
    )

    full_preparation = _time_operation(
        lambda: _full_prepared(context, workload, fresh.m1.result.full_fit)
    )
    slow_core = _slowest_by_wall(core_configurations, core_measurements)
    slow_low = _slowest_by_wall(
        low_configurations,
        tuple(timed.measurement for timed in low_timed),
    )
    slow_hybrid = _slowest_by_wall(
        hybrid_configurations,
        tuple(timed.measurement for timed in hybrid_timed),
    )
    core_pair = _time_operation(
        lambda: _core_full_pair(slow_core, full_preparation.result, workload)
    )
    low_pair = _time_operation(
        lambda: _low_rank_full_pair(slow_low, full_preparation.result, workload)
    )
    hybrid_pair = _time_operation(
        lambda: _hybrid_full_pair(
            slow_hybrid,
            selected_low_rank.configuration,
            full_preparation.result,
            workload,
        )
    )
    rank_neighbor_target, rank_neighbor_precursor = (
        _rank_neighbor_full_refit_configurations(slow_hybrid)
    )
    rank_neighbor_hybrid_pair = _time_operation(
        lambda: _hybrid_full_pair(
            rank_neighbor_target,
            rank_neighbor_precursor,
            full_preparation.result,
            workload,
        )
    )
    family_pair_maximum = _componentwise_max(
        (
            core_pair.measurement,
            low_pair.measurement,
            hybrid_pair.measurement,
            rank_neighbor_hybrid_pair.measurement,
        )
    )
    refit_measurement = _serial_sum(
        (m1_measurement, full_preparation.measurement, family_pair_maximum)
    )
    report = _time_operation(lambda: _run_report_unit(workload))

    measurements = {
        UNIT_SYNTHETIC: context.synthetic_generation_timing,
        UNIT_M0: m0_measurement,
        UNIT_M1: m1_measurement,
        UNIT_SETUP: setup.measurement,
        UNIT_REFERENCE: reference.measurement,
        UNIT_CORE: _componentwise_max(core_measurements),
        UNIT_LOW_RANK: _componentwise_max(
            tuple(timed.measurement for timed in low_timed)
        ),
        UNIT_HYBRID: _componentwise_max(
            tuple(timed.measurement for timed in hybrid_timed)
        ),
        UNIT_DIAGNOSTIC: diagnosed.measurement,
        UNIT_REFIT: refit_measurement,
        UNIT_REPORT: report.measurement,
    }
    repeat_witness = {
        "core": core_pair.result,
        "low_rank": low_pair.result,
        "hybrid": hybrid_pair.result,
        "hybrid_rank_neighbor_r6_to_r8": rank_neighbor_hybrid_pair.result,
    }
    aggregate_witness = {
        "status": "pass",
        "fixed_core_anchors_measured": len(core_timed),
        "fixed_low_rank_anchors_measured": len(low_timed),
        "fixed_hybrid_anchors_measured": len(hybrid_timed),
        "shared_M1_replays": 1,
        "shared_12_fold_caches": 1,
        "diagnostic": diagnosed.result.witness,
        "fixed_optional_units_are_worst_case_target_free_timing_probes": True,
        "optional_probe_execution_is_coverage_evidence_authorization": False,
        "production_optional_activation_rank_and_edges_remain_outcome_bound": True,
        "rank_neighbor_full_refit_timing_probe": {
            "source_selected_rank": RANK_NEIGHBOR_SOURCE_RANK,
            "target_rank": RANK_NEIGHBOR_TARGET_RANK,
            "target_rank_precursor_rho": 0.1,
            "fresh_precursor_then_cold_and_warm_hybrid": True,
            "target_free_timing_probe_not_selection_authority": True,
        },
        "repeat_tolerances_passed": True,
        "observed_session_cap80_bookkeeping_measured_without_fits": True,
        "synthetic_ledger_timing_is_not_production_coverage_authority": True,
        "controller_report_replay_passed": True,
    }
    return _issue_run(
        context,
        measurements,
        aggregate_witness,
        (
            fresh,
            workload,
            reference.result,
            core_executions,
            diagnosed.result,
            selected_low_rank,
            tuple(timed.result for timed in hybrid_timed),
            full_preparation.result,
            repeat_witness,
            report.result,
        ),
    )


def _run_for_context(context: Any) -> ScaleBenchmarkRun:
    session = _require_bound_session(context)
    if session._failed:
        raise ScaleBenchmarkError("scale benchmark session previously failed closed")
    if session._run is not None:
        run = _require_run(session._run)
        if run._context_reference() is not context:
            raise ScaleBenchmarkError("scale benchmark run belongs to another context")
        return run
    if session._running:
        raise ScaleBenchmarkError("scale benchmark session cannot execute reentrantly")
    session._running = True
    _refresh_session_integrity(session)
    try:
        run = _execute_scale_benchmark_run(context)
    except Exception:
        session._failed = True
        session._running = False
        _refresh_session_integrity(session)
        raise
    session._running = False
    session._run = run
    _refresh_session_integrity(session)
    return _require_run(run)


def executor_scale_units(context: Any) -> Mapping[str, Mapping[str, Any]]:
    """Return the executor-owned aggregate view of one exact central run."""

    return _run_for_context(context)._unit_view(EXECUTOR_UNITS)


def optional_scale_units(context: Any) -> Mapping[str, Mapping[str, Any]]:
    """Return the optional-family aggregate view of the same central run."""

    return _run_for_context(context)._unit_view(OPTIONAL_UNITS)


__all__ = (
    "AggregateMeasurement",
    "EXECUTOR_UNITS",
    "OPTIONAL_UNITS",
    "REQUIRED_UNITS",
    "ScaleBenchmarkError",
    "ScaleBenchmarkRun",
    "bind_scale_benchmark_context",
    "executor_scale_units",
    "issue_scale_benchmark_session",
    "optional_scale_units",
    "time_synthetic_constructor",
)
