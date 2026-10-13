#!/usr/bin/env python3
"""Fresh in-memory M0/M1 replay support for the EP10 M2 scale benchmark.

This module is deliberately narrower than either production runner.  It accepts
only an already synthetic ``BenchmarkContext``-shaped object, performs no input
loading, checkpointing, artifact reads, or artifact writes, and returns an
aggregate-only timing view separately from the in-memory numerical results.

Importing :mod:`fit_v3_m0_v7` is intentional: that wrapper installs the frozen
V7 optimizer route in its ``base`` M0 module.  Every M0 call below is made
through that patched base module.  The M1 orchestration is reused unchanged,
with factories that compute every path in memory and ignore the checkpoint
argument without inspecting it.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import resource
import sys
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import fit_v3_m0_v7 as m0_v7
import fit_v3_m1_real_v1 as m1_real


EXPECTED_ROWS = 194
EXPECTED_FOLDS = 12
EXPECTED_PANEL_SIZE = 16
EXPECTED_RESIDUAL_RANK = 104
EXPECTED_SYNTHETIC_SEED = 2026092901
EXPECTED_M1_WARM_PATHS = 78
EXPECTED_M1_ZERO_REFITS = 13
_EXPECTED_FROZEN_INVENTORY_SHA256 = (
    "f9c25dddc3e98012c9d72345761de1cd19a5bfa7fbd22d3216f5a0d194b27fcd"
)
M0_UNIT_NAME = "fresh_M0_replay_refit"
M1_UNIT_NAME = "fresh_M1_replay_refit"

# This target-blind ID/group vector is copied from the frozen EP10 M2 inventory
# so replay validation does not need to open that TSV (or any other input).
_FROZEN_CELL_IDS_BY_GROUP = MappingProxyType(
    {
        "212074": ("14076", "14077", "1923", "1924", "7102", "7103"),
        "220187": (
            "10000", "10001", "10002", "12172", "12173", "12175", "12181",
            "12182", "12183", "12186", "14085", "14086", "14088", "14094",
            "14095", "14096", "14097", "14098", "17287", "17289", "17290",
            "17294", "1947", "1948", "1949", "1950", "1951", "1952", "1972",
            "9988", "9989", "9990", "9997", "9998", "9999",
        ),
        "220188": (
            "10012", "10014", "10015", "10016", "12194", "12196", "12197",
            "14111", "14112", "14115", "14117", "14118", "14119", "17312",
            "17313", "17315", "17316", "17317", "1981", "1986", "1987",
            "1988", "1989", "1991", "7117",
        ),
        "220190": (
            "10032", "10038", "10042", "12229", "12230", "12231", "12232",
            "12233", "12236", "14138", "14139", "17347", "17348", "17349",
            "17350", "2000",
        ),
        "221091": (
            "10546", "10548", "10549", "10550", "10551", "10552", "10553",
            "10554", "10555", "10556", "10557", "10558", "10559", "10560",
            "10561", "10562", "10563", "10564", "10565", "10566", "10567",
            "10568", "10569", "10570", "10571", "10572", "10573", "10574",
            "12349", "2909", "2910", "7297", "7304",
        ),
        "221225": (
            "10686", "10687", "10688", "10695", "10696", "10698", "14682",
            "14683", "14684", "14685", "14686", "14687", "14688", "14689",
            "14690", "14691", "14692", "14693", "14694", "14704", "14705",
            "14710", "14711", "14712", "14713", "14720", "14721", "14722",
            "14723", "14724", "14725", "14726", "14727", "14729", "2955",
            "2958", "2959", "2961", "7318",
        ),
        "221418": (
            "10839", "10842", "10843", "3288", "3289", "3290", "3292",
            "3293", "3294", "3295", "3296", "3297",
        ),
        "221514": ("17691", "17693", "17694", "17695", "17696"),
        "221539": ("15537",),
        "221588": ("4187", "4188", "4189", "4200", "4204", "4205", "4206"),
        "221751": (
            "11590", "11593", "5525", "5526", "5527", "5528", "5529",
            "5530", "5532", "5533", "5535", "5536", "5537",
        ),
        "233281": ("18168", "18171"),
    }
)
_FROZEN_INVENTORY_ENTRIES = tuple(
    sorted(
        (identifier, group)
        for group, identifiers in _FROZEN_CELL_IDS_BY_GROUP.items()
        for identifier in identifiers
    )
)
_EXPECTED_LEAVE_TWO_KEYS = tuple(
    f"leave_two__{left}__{right}"
    for left, right in itertools.combinations(m1_real.m1.GROUP_ORDER, 2)
)
_EXPECTED_LEAVE_ONE_KEYS = tuple(
    f"leave_one__{group}" for group in m1_real.m1.GROUP_ORDER
)
_EXPECTED_OUTER_REFIT_KEYS = tuple(
    f"outer_refit__{group}" for group in m1_real.m1.GROUP_ORDER
)
_EXPECTED_WARM_KEYS = _EXPECTED_LEAVE_TWO_KEYS + _EXPECTED_LEAVE_ONE_KEYS
_EXPECTED_ZERO_KEYS = _EXPECTED_OUTER_REFIT_KEYS + ("full_refit",)
if (
    len(_FROZEN_INVENTORY_ENTRIES) != EXPECTED_ROWS
    or len({identifier for identifier, _ in _FROZEN_INVENTORY_ENTRIES})
    != EXPECTED_ROWS
    or {
        group: len(_FROZEN_CELL_IDS_BY_GROUP.get(group, ()))
        for group in m1_real.m1.GROUP_ORDER
    }
    != m1_real.m1.GROUP_SIZES
    or len(_EXPECTED_WARM_KEYS) != EXPECTED_M1_WARM_PATHS
    or len(_EXPECTED_ZERO_KEYS) != EXPECTED_M1_ZERO_REFITS
):
    raise ImportError("embedded fresh-replay inventory or fit accounting changed")

_NO_CHECKPOINT_IO = object()


class FreshReplayError(RuntimeError):
    """A benchmark-only fresh replay invariant failed closed."""


@dataclass(frozen=True)
class AggregateTiming:
    """One whole-operation measurement, never a per-cell measurement.

    ``peak_rss_gib`` is the conservative process-lifetime high-water mark
    observed after the operation, not an isolated incremental allocation.
    """

    wall_seconds: float
    cpu_seconds: float
    peak_rss_gib: float
    scratch_gib: float = 0.0

    def benchmark_unit(self) -> Mapping[str, Any]:
        """Return the exact aggregate fields accepted by the scale benchmark."""
        return {
            "status": "pass",
            "wall_seconds": float(self.wall_seconds),
            "cpu_seconds": float(self.cpu_seconds),
            "peak_rss_gib": float(self.peak_rss_gib),
            "peak_rss_semantics": "process_lifetime_high_water_conservative",
            "scratch_gib": float(self.scratch_gib),
            "real_membership_opened": False,
            "existing_M0_artifact_or_telemetry_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        }


@dataclass(frozen=True)
class SyntheticGenerationReceipt:
    """Non-disclosive in-memory proof of the exact synthetic replay outcome."""

    replay_identity: str
    generator: str
    seed: int
    rows: int
    inventory_sha256: str
    uniforms_sha256: str
    candidate_indexes_sha256: str
    membership_sha256: str


@dataclass(frozen=True)
class M0ReplayResult:
    """Aggregate M0 replay outcome plus the fresh full-development fit."""

    cross_fit_cells: int
    outer_fold_count: int
    outer_boundary_count: int
    outer_selected_lambdas: tuple[float, ...]
    full_selected_lambda: float
    full_selection: Mapping[str, Any]
    full_design_transform: Any
    full_optimizer_result: Any
    generation_receipt: SyntheticGenerationReceipt


@dataclass(frozen=True)
class FoldBoundM1Offset:
    """One fresh M1 fit bound to its exact outer-LOGO training rows."""

    held_out_group: str
    training_indexes: tuple[int, ...]
    theta: np.ndarray
    design_transform: Any
    geometry_scales: np.ndarray
    fit_identity: str


@dataclass(frozen=True)
class FullDevelopmentM1Fit:
    """Fresh full-development M1 parameters and their fitted coordinates."""

    training_indexes: tuple[int, ...]
    ridge_lambda: float
    theta: np.ndarray
    design_transform: Any
    geometry_scales: np.ndarray
    fit_identity: str
    optimizer_result: Any


@dataclass(frozen=True)
class M1ReplayResult:
    """Reusable fresh M1 offsets without retaining per-cell predictions."""

    outer_offsets: Mapping[str, FoldBoundM1Offset]
    full_fit: FullDevelopmentM1Fit
    cross_fit_cells: int
    outer_fold_count: int
    outer_boundary_count: int
    warm_path_count: int
    zero_refit_count: int
    generation_receipt: SyntheticGenerationReceipt


@dataclass
class _M1FactoryAccounting:
    """Per-call accounting for factories actually invoked by orchestration."""

    warm_keys: list[str] = field(default_factory=list)
    zero_keys: list[str] = field(default_factory=list)

    def record_warm(self, key: Any, checkpoint_root: Any) -> None:
        if checkpoint_root is not _NO_CHECKPOINT_IO:
            raise FreshReplayError("M1 warm path received a checkpoint-capable root")
        self.warm_keys.append(_validate_factory_key(key))

    def record_zero(self, key: Any, checkpoint_root: Any) -> None:
        if checkpoint_root is not _NO_CHECKPOINT_IO:
            raise FreshReplayError("M1 zero refit received a checkpoint-capable root")
        self.zero_keys.append(_validate_factory_key(key))

    def validate_complete(self) -> None:
        if (
            len(self.warm_keys) != EXPECTED_M1_WARM_PATHS
            or len(set(self.warm_keys)) != EXPECTED_M1_WARM_PATHS
            or set(self.warm_keys) != set(_EXPECTED_WARM_KEYS)
        ):
            raise FreshReplayError(
                "fresh M1 replay did not invoke exactly 78 distinct warm paths"
            )
        if (
            len(self.zero_keys) != EXPECTED_M1_ZERO_REFITS
            or len(set(self.zero_keys)) != EXPECTED_M1_ZERO_REFITS
            or set(self.zero_keys) != set(_EXPECTED_ZERO_KEYS)
        ):
            raise FreshReplayError(
                "fresh M1 replay did not invoke exactly 13 distinct zero refits"
            )


@dataclass(frozen=True)
class TimedReplay:
    """An in-memory replay result paired with its whole-operation timing."""

    result: Any
    timing: AggregateTiming


@dataclass(frozen=True)
class FreshReplayBenchmarkResult:
    """The two complete fresh prefix replays needed by an M2 benchmark."""

    m0: TimedReplay
    m1: TimedReplay

    def aggregate_units(self) -> Mapping[str, Mapping[str, Any]]:
        """Expose timings only; numerical arrays cannot enter the report."""
        return {
            M0_UNIT_NAME: self.m0.timing.benchmark_unit(),
            M1_UNIT_NAME: self.m1.timing.benchmark_unit(),
        }


def _require_exact_integer(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FreshReplayError(f"{label} must be the exact integer {expected}")
    if int(value) != expected:
        raise FreshReplayError(f"{label} must equal {expected}")


def _readonly_float_copy(
    value: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise FreshReplayError(f"{label} is malformed or nonfinite")
    copied = array.copy()
    copied.setflags(write=False)
    return copied


def _sha256_array(value: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _inventory_sha256(entries: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for identifier, group in entries:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\t")
        digest.update(group.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _revalidate_synthetic_generation(
    cells: tuple[Any, ...],
    membership: np.ndarray,
    pair_geometry: Any,
    library: Any,
    seed: int,
) -> SyntheticGenerationReceipt:
    """Reproduce the deterministic draw without reading any external input."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    expected_uniforms = rng.random(EXPECTED_ROWS)
    observed_uniforms = np.asarray(
        [getattr(cell, "sampling_uniform", math.nan) for cell in cells],
        dtype=np.float64,
    )
    if (
        observed_uniforms.shape != (EXPECTED_ROWS,)
        or not np.all(np.isfinite(observed_uniforms))
        or not np.array_equal(observed_uniforms, expected_uniforms)
    ):
        raise FreshReplayError(
            "synthetic sampling uniforms do not match the fixed PCG64 receipt"
        )

    bases = tuple(library.basis(int(cell.k)) for cell in cells)
    needs_scored_sampling = any(int(basis.count) > 1 for basis in bases)
    prepared = None
    geometry = None
    theta = None
    if needs_scored_sampling:
        try:
            geometry = m1_real.m1.GeometryEngine(library, pair_geometry)
            moments = tuple(geometry.uniform_moments(cell) for cell in cells)
            prepared = m1_real.m1.prepare_data(cells, moments)
            theta = m1_real.m1.planted_parameters()
        except (m1_real.m1.M1Error, TypeError, ValueError) as error:
            raise FreshReplayError(
                "synthetic generation receipt could not rebuild planted M1 inputs"
            ) from error

    expected_indexes = np.empty(EXPECTED_ROWS, dtype=np.int64)
    for row_index, (cell, basis, uniform) in enumerate(
        zip(cells, bases, expected_uniforms, strict=True)
    ):
        if int(basis.count) == 1:
            expected_index = 0
        else:
            if prepared is None or geometry is None or theta is None:
                raise FreshReplayError(
                    "synthetic generation receipt omitted planted M1 inputs"
                )
            try:
                expected_index = int(
                    m1_real.m1.sample_candidate_index(
                        theta,
                        prepared,
                        row_index,
                        library,
                        geometry,
                        float(uniform),
                    )
                )
            except (m1_real.m1.M1Error, TypeError, ValueError) as error:
                raise FreshReplayError(
                    "synthetic generation receipt could not reproduce a candidate draw"
                ) from error
        expected_indexes[row_index] = expected_index
        if int(cell.observed_index) != expected_index:
            raise FreshReplayError(
                "synthetic candidate index does not match the planted-M1 receipt"
            )
        expected_row = np.asarray(basis.membership[expected_index], dtype=np.uint8)
        if not np.array_equal(membership[row_index], expected_row):
            raise FreshReplayError(
                "synthetic membership does not match the planted-M1 receipt"
            )

    inventory_digest = _inventory_sha256(_FROZEN_INVENTORY_ENTRIES)
    if inventory_digest != _EXPECTED_FROZEN_INVENTORY_SHA256:
        raise FreshReplayError("embedded frozen ID/group inventory digest changed")
    membership_digest = _sha256_array(membership, "u1")
    replay_identity = f"synthetic_seed_{seed}"
    return SyntheticGenerationReceipt(
        replay_identity=replay_identity,
        generator="numpy_SeedSequence_PCG64_planted_M1",
        seed=seed,
        rows=EXPECTED_ROWS,
        inventory_sha256=inventory_digest,
        uniforms_sha256=_sha256_array(expected_uniforms, "<f8"),
        candidate_indexes_sha256=_sha256_array(expected_indexes, "<i8"),
        membership_sha256=membership_digest,
    )


def _validated_synthetic_context(
    context: Any, library: Any
) -> tuple[tuple[Any, ...], np.ndarray, SyntheticGenerationReceipt]:
    """Validate the complete synthetic interface without opening any input."""
    try:
        cells = tuple(context.cells)
        membership = np.asarray(context.membership_matrix)
        pair_geometry = context.pair_geometry
        seed = context.synthetic_seed
        expected_folds = context.expected_folds
        expected_rank = context.expected_residual_rank
    except (AttributeError, TypeError) as error:
        raise FreshReplayError("synthetic benchmark context is malformed") from error

    _require_exact_integer(expected_folds, EXPECTED_FOLDS, "expected folds")
    _require_exact_integer(
        expected_rank, EXPECTED_RESIDUAL_RANK, "expected residual rank"
    )
    _require_exact_integer(seed, EXPECTED_SYNTHETIC_SEED, "synthetic seed")
    seed = int(seed)
    if pair_geometry is None:
        raise FreshReplayError("synthetic context omitted pair geometry")
    if len(cells) != EXPECTED_ROWS:
        raise FreshReplayError("fresh replay requires exactly 194 synthetic cells")
    if membership.shape != (EXPECTED_ROWS, EXPECTED_PANEL_SIZE):
        raise FreshReplayError("synthetic membership matrix has the wrong shape")
    if membership.dtype != np.dtype(np.uint8):
        raise FreshReplayError("synthetic membership matrix must use uint8")
    if membership.flags.writeable:
        raise FreshReplayError("synthetic membership matrix must be read-only")
    if not np.all((membership == 0) | (membership == 1)):
        raise FreshReplayError("synthetic membership matrix is not binary")

    inventory_entries: list[tuple[str, str]] = []
    observed_group_sizes = {group: 0 for group in m1_real.m1.GROUP_ORDER}
    for row_index, cell in enumerate(cells):
        identifier = str(getattr(cell, "identifier", ""))
        group = str(getattr(cell, "group", ""))
        layer = str(getattr(cell, "layer", ""))
        hemisphere = str(getattr(cell, "hemisphere", ""))
        inventory_entries.append((identifier, group))
        if not identifier:
            raise FreshReplayError("synthetic cell identifier is empty")
        if group not in observed_group_sizes:
            raise FreshReplayError("synthetic cell group is outside the frozen order")
        observed_group_sizes[group] += 1
        if layer not in m1_real.m1.LAYERS:
            raise FreshReplayError("synthetic cell layer is outside the frozen levels")
        if hemisphere not in {"left", "right"}:
            raise FreshReplayError("synthetic cell hemisphere is invalid")

        soma = np.asarray(getattr(cell, "soma_um", ()), dtype=float)
        distances = np.asarray(getattr(cell, "soma_distance_mm", ()), dtype=float)
        if soma.shape != (3,) or not np.all(np.isfinite(soma)):
            raise FreshReplayError("synthetic cell soma coordinates are malformed")
        if distances.shape != (EXPECTED_PANEL_SIZE,) or not np.all(
            np.isfinite(distances)
        ):
            raise FreshReplayError("synthetic cell soma distances are malformed")

        k = getattr(cell, "k", None)
        observed_index = getattr(cell, "observed_index", None)
        segment = getattr(cell, "segment_unmapped", None)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise FreshReplayError("synthetic cell K is not an integer")
        if not 0 <= int(k) <= EXPECTED_PANEL_SIZE:
            raise FreshReplayError("synthetic cell K is outside 0..16")
        if isinstance(observed_index, bool) or not isinstance(
            observed_index, (int, np.integer)
        ):
            raise FreshReplayError("synthetic observed index is not an integer")
        if isinstance(segment, bool) or not isinstance(segment, (int, np.integer)):
            raise FreshReplayError("synthetic segment flag is not an integer")
        if int(segment) not in (0, 1):
            raise FreshReplayError("synthetic segment flag is not binary")

        basis = library.basis(int(k))
        candidate_count = int(basis.count)
        candidate_index = int(observed_index)
        if not 0 <= candidate_index < candidate_count:
            raise FreshReplayError("synthetic observed index is outside exact Omega_i")
        row = np.asarray(membership[row_index], dtype=np.uint8)
        expected_row = np.asarray(basis.membership[candidate_index], dtype=np.uint8)
        if (
            int(np.sum(row, dtype=np.int64)) != int(k)
            or expected_row.shape != (EXPECTED_PANEL_SIZE,)
            or not np.array_equal(row, expected_row)
        ):
            raise FreshReplayError(
                "synthetic membership is not the cell's observed fixed-K candidate"
            )

    if tuple(sorted(inventory_entries)) != _FROZEN_INVENTORY_ENTRIES:
        raise FreshReplayError(
            "synthetic cells do not match the exact frozen ID/group inventory"
        )
    if observed_group_sizes != m1_real.m1.GROUP_SIZES:
        raise FreshReplayError("synthetic conservative-group sizes changed")
    frozen_membership = membership.copy()
    frozen_membership.setflags(write=False)
    receipt = _revalidate_synthetic_generation(
        cells, frozen_membership, pair_geometry, library, seed
    )
    return cells, frozen_membership, receipt


def _m0_records_from_validated(
    cells: Sequence[Any], membership: np.ndarray
) -> tuple[Any, ...]:
    records = []
    for row_index, cell in enumerate(cells):
        records.append(
            m0_v7.base.Record(
                provider_neuron_id=str(cell.identifier),
                filename=f"in_memory_synthetic_{row_index:03d}.swc",
                group=str(cell.group),
                sample_id=str(cell.group),
                source_acronym=str(cell.layer),
                hemisphere=str(cell.hemisphere),
                soma_um=tuple(float(value) for value in cell.soma_um),
                segment_unmapped=int(cell.segment_unmapped),
                terminal_unmapped_report_only=0,
                outcome=tuple(int(value) for value in membership[row_index]),
                k=int(cell.k),
            )
        )
    return tuple(records)


def validate_synthetic_generation(context: Any) -> SyntheticGenerationReceipt:
    """Return a receipt only after exact inventory and planted-draw replay."""
    library = m1_real.m1.CandidateLibrary()
    _, _, receipt = _validated_synthetic_context(context, library)
    return receipt


def context_to_m0_records(context: Any) -> tuple[Any, ...]:
    """Convert an exactly revalidated synthetic context to M0 records."""
    library = m1_real.m1.CandidateLibrary()
    cells, membership, _ = _validated_synthetic_context(context, library)
    return _m0_records_from_validated(cells, membership)


def _validate_m0_cross_fit(block: Any, records: Sequence[Any]) -> None:
    eta = np.asarray(block.eta, dtype=float)
    log_probability = np.asarray(block.log_probability_nats, dtype=float)
    marginals = np.asarray(block.marginals, dtype=float)
    expected_k = np.asarray([record.k for record in records], dtype=float)
    if not (
        eta.shape == (EXPECTED_ROWS, EXPECTED_PANEL_SIZE)
        and log_probability.shape == (EXPECTED_ROWS,)
        and marginals.shape == (EXPECTED_ROWS, EXPECTED_PANEL_SIZE)
        and np.all(np.isfinite(eta))
        and np.all(np.isfinite(log_probability))
        and np.all(np.isfinite(marginals))
        and np.all(log_probability <= 1.0e-12)
        and np.all(marginals >= -1.0e-12)
        and np.all(marginals <= 1.0 + 1.0e-12)
        and np.allclose(
            np.sum(marginals, axis=1), expected_k, rtol=0.0, atol=1.0e-10
        )
    ):
        raise FreshReplayError("fresh M0 cross-fit prediction is malformed")


def run_fresh_m0_replay(context: Any) -> M0ReplayResult:
    """Run complete nested M0 LOGO and a separately tuned fresh full refit."""
    m0_v7.FIT_TELEMETRY.clear()
    if m0_v7.base.fit_lbfgs is not m0_v7.fit_lbfgs_v7:
        raise FreshReplayError("the frozen M0 V7 route is not installed")
    library = m1_real.m1.CandidateLibrary()
    cells, membership, receipt = _validated_synthetic_context(context, library)
    records = _m0_records_from_validated(cells, membership)
    prediction, outer_details_raw, boundary_count = m0_v7.base.nested_outer_logo(
        records
    )
    _validate_m0_cross_fit(prediction, records)

    outer_details = tuple(outer_details_raw)
    if len(outer_details) != EXPECTED_FOLDS:
        raise FreshReplayError("fresh M0 nested replay did not return 12 folds")
    by_group: dict[str, Mapping[str, Any]] = {}
    for detail in outer_details:
        if not isinstance(detail, Mapping):
            raise FreshReplayError("fresh M0 outer detail is malformed")
        group = str(detail.get("held_out_group", ""))
        selected = float(detail.get("selected_lambda", math.nan))
        if group in by_group or group not in m0_v7.base.GROUP_SIZES:
            raise FreshReplayError("fresh M0 outer fold inventory changed")
        if selected not in m0_v7.base.LAMBDAS:
            raise FreshReplayError("fresh M0 selected lambda is outside the grid")
        by_group[group] = detail
    if set(by_group) != set(m0_v7.base.GROUP_SIZES):
        raise FreshReplayError("fresh M0 outer folds do not cover all groups")
    expected_boundary_count = sum(
        bool(detail.get("selected_lambda_is_grid_boundary"))
        for detail in outer_details
    )
    if (
        isinstance(boundary_count, bool)
        or int(boundary_count) != expected_boundary_count
    ):
        raise FreshReplayError("fresh M0 boundary count is inconsistent")
    if int(boundary_count) >= 6:
        raise m0_v7.base.BoundaryReview(
            "ridge-grid boundary selected in at least six fresh M0 outer folds"
        )

    full_selected, full_selection = m0_v7.base.tune_lambda_logo(records)
    full_selected = float(full_selected)
    if full_selected not in m0_v7.base.LAMBDAS:
        raise FreshReplayError("fresh M0 full lambda is outside the grid")
    full_transform, full_fit = m0_v7.base._fit_single(records, full_selected)
    coefficients = np.asarray(full_fit.coefficients, dtype=float)
    if coefficients.shape != (17, 15) or not np.all(np.isfinite(coefficients)):
        raise FreshReplayError("fresh M0 full-fit coefficients are malformed")
    if not isinstance(full_selection, Mapping):
        raise FreshReplayError("fresh M0 full selection is malformed")

    return M0ReplayResult(
        cross_fit_cells=EXPECTED_ROWS,
        outer_fold_count=EXPECTED_FOLDS,
        outer_boundary_count=int(boundary_count),
        outer_selected_lambdas=tuple(
            float(by_group[group]["selected_lambda"])
            for group in m0_v7.base.GROUP_SIZES
        ),
        full_selected_lambda=full_selected,
        full_selection=MappingProxyType(dict(full_selection)),
        full_design_transform=full_transform,
        full_optimizer_result=full_fit,
        generation_receipt=receipt,
    )


def _validate_factory_key(key: Any) -> str:
    value = str(key)
    if not value or m1_real.CHECKPOINT_KEY_PATTERN.fullmatch(value) is None:
        raise FreshReplayError("in-memory M1 fit key is invalid")
    return value


def in_memory_m1_warm_factory(
    key: str,
    indexes: Sequence[int],
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Any,
) -> Any:
    """Fit one complete descending M1 path without touching checkpoints."""
    del checkpoint_root
    fit_key = _validate_factory_key(key)
    selected_indexes = tuple(int(index) for index in indexes)
    prepared = m1_real._prepare_training(cells, moments, selected_indexes)
    try:
        fits = m1_real.m1.fit_path(prepared, library, geometry)
    except m1_real.m1.M1Error as error:
        error.production_fit_kind = "benchmark_in_memory_warm"
        error.production_fit_key = fit_key
        raise
    if tuple(fits) != m1_real.m1.LAMBDAS_DESCENDING:
        raise FreshReplayError("in-memory M1 warm path lambda order changed")
    for ridge_lambda, result in fits.items():
        m1_real._validate_optimizer_result(result, float(ridge_lambda))
    return m1_real.TrainingBundle(
        fit_key, selected_indexes, prepared, fits, False
    )


def in_memory_m1_zero_factory(
    key: str,
    indexes: Sequence[int],
    ridge_lambda: float,
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Any,
) -> Any:
    """Fit one selected M1 model from exact zero without checkpoint I/O."""
    del checkpoint_root
    fit_key = _validate_factory_key(key)
    ridge = float(ridge_lambda)
    if ridge not in m1_real.m1.LAMBDAS_DESCENDING:
        raise FreshReplayError("in-memory M1 zero-refit lambda is outside the grid")
    selected_indexes = tuple(int(index) for index in indexes)
    prepared = m1_real._prepare_training(cells, moments, selected_indexes)
    try:
        result = m1_real.m1.fit_newton(
            prepared, library, geometry, ridge, initial=None
        )
    except m1_real.m1.M1Error as error:
        error.production_fit_kind = "benchmark_in_memory_zero"
        error.production_fit_key = fit_key
        error.production_fit_lambda = ridge
        raise
    m1_real._validate_optimizer_result(result, ridge)
    return m1_real.TrainingBundle(
        fit_key, selected_indexes, prepared, {ridge: result}, False
    )


def _copy_m1_coordinates(
    bundle: Any, result: Any
) -> tuple[np.ndarray, Any, np.ndarray]:
    theta = _readonly_float_copy(
        result.theta, (m1_real.m1.PARAMETER_DIMENSION,), "fresh M1 theta"
    )
    design_mean = _readonly_float_copy(
        bundle.prepared.design_transform.soma_mean,
        (9,),
        "fresh M1 soma mean",
    )
    design_scale = _readonly_float_copy(
        bundle.prepared.design_transform.soma_scale,
        (9,),
        "fresh M1 soma scale",
    )
    if np.any(design_scale <= 0.0):
        raise FreshReplayError("fresh M1 soma scale is not positive")
    geometry_scales = _readonly_float_copy(
        bundle.prepared.geometry_scales,
        (m1_real.m1.GEOMETRY_DIMENSION,),
        "fresh M1 geometry scales",
    )
    if np.any(geometry_scales <= 0.0):
        raise FreshReplayError("fresh M1 geometry scale is not positive")
    transform = m1_real.m1.DesignTransform(design_mean, design_scale)
    return theta, transform, geometry_scales


def _validate_m1_cross_fit(cross_fit: Any, cells: Sequence[Any]) -> None:
    log_probability = np.asarray(
        cross_fit.prediction.log_probability_nats, dtype=float
    )
    marginals = np.asarray(cross_fit.prediction.marginals, dtype=float)
    expected_k = np.asarray([cell.k for cell in cells], dtype=float)
    if not (
        log_probability.shape == (EXPECTED_ROWS,)
        and marginals.shape == (EXPECTED_ROWS, EXPECTED_PANEL_SIZE)
        and np.all(np.isfinite(log_probability))
        and np.all(np.isfinite(marginals))
        and np.all(log_probability <= 1.0e-12)
        and np.all(marginals >= -1.0e-12)
        and np.all(marginals <= 1.0 + 1.0e-12)
        and np.allclose(
            np.sum(marginals, axis=1), expected_k, rtol=0.0, atol=1.0e-10
        )
    ):
        raise FreshReplayError("fresh M1 cross-fit prediction is malformed")


def run_fresh_m1_replay(context: Any) -> M1ReplayResult:
    """Run complete fresh nested M1 selection, outer refits, and full refit."""
    library = m1_real.m1.CandidateLibrary()
    cells, _, receipt = _validated_synthetic_context(context, library)
    geometry = m1_real.m1.GeometryEngine(library, context.pair_geometry)
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    accounting = _M1FactoryAccounting()

    def counted_warm_factory(
        key: str,
        indexes: Sequence[int],
        factory_cells: Sequence[Any],
        factory_moments: Sequence[Any],
        factory_library: Any,
        factory_geometry: Any,
        checkpoint_root: Any,
    ) -> Any:
        accounting.record_warm(key, checkpoint_root)
        return in_memory_m1_warm_factory(
            key,
            indexes,
            factory_cells,
            factory_moments,
            factory_library,
            factory_geometry,
            checkpoint_root,
        )

    def counted_zero_factory(
        key: str,
        indexes: Sequence[int],
        ridge_lambda: float,
        factory_cells: Sequence[Any],
        factory_moments: Sequence[Any],
        factory_library: Any,
        factory_geometry: Any,
        checkpoint_root: Any,
    ) -> Any:
        accounting.record_zero(key, checkpoint_root)
        return in_memory_m1_zero_factory(
            key,
            indexes,
            ridge_lambda,
            factory_cells,
            factory_moments,
            factory_library,
            factory_geometry,
            checkpoint_root,
        )

    selection = m1_real.select_outer_lambdas(
        cells,
        moments,
        library,
        geometry,
        _NO_CHECKPOINT_IO,
        warm_factory=counted_warm_factory,
        predictor=m1_real._production_predictor,
    )
    cross_fit = m1_real.refit_outer_and_predict(
        cells,
        moments,
        library,
        geometry,
        _NO_CHECKPOINT_IO,
        selection,
        zero_factory=counted_zero_factory,
        predictor=m1_real._production_predictor,
    )
    _validate_m1_cross_fit(cross_fit, cells)
    full = m1_real.tune_and_refit_full_development(
        cells,
        moments,
        library,
        geometry,
        _NO_CHECKPOINT_IO,
        cross_fit.warm_cache,
        warm_factory=counted_warm_factory,
        zero_factory=counted_zero_factory,
        predictor=m1_real._production_predictor,
    )

    accounting.validate_complete()
    if (
        set(selection.warm_cache) != set(_EXPECTED_LEAVE_TWO_KEYS)
        or set(cross_fit.warm_cache) != set(_EXPECTED_LEAVE_TWO_KEYS)
        or set(full.warm_cache) != set(_EXPECTED_WARM_KEYS)
    ):
        raise FreshReplayError("fresh M1 warm-cache inventory changed")
    if set(selection.selected_lambdas) != set(m1_real.m1.GROUP_ORDER):
        raise FreshReplayError("fresh M1 outer selection inventory changed")
    if set(cross_fit.outer_refits) != set(m1_real.m1.GROUP_ORDER):
        raise FreshReplayError("fresh M1 outer refit inventory changed")
    offsets: dict[str, FoldBoundM1Offset] = {}
    for ordinal, group in enumerate(m1_real.m1.GROUP_ORDER, start=1):
        expected_indexes = tuple(
            index for index, cell in enumerate(cells) if cell.group != group
        )
        bundle = cross_fit.outer_refits[group]
        if bundle.indexes != expected_indexes or bundle.checkpoint_reused is not False:
            raise FreshReplayError("fresh M1 outer refit is not fold-bound in memory")
        ridge = float(selection.selected_lambdas[group])
        if ridge not in bundle.fits or ridge not in m1_real.m1.LAMBDAS_DESCENDING:
            raise FreshReplayError("fresh M1 outer refit lambda binding changed")
        result = bundle.fits[ridge]
        theta, transform, geometry_scales = _copy_m1_coordinates(bundle, result)
        offsets[group] = FoldBoundM1Offset(
            held_out_group=group,
            training_indexes=expected_indexes,
            theta=theta,
            design_transform=transform,
            geometry_scales=geometry_scales,
            fit_identity=(
                f"{receipt.replay_identity}__M1_outer_refit_{ordinal:02d}"
            ),
        )

    full_ridge = float(full.selected_lambda)
    full_bundle = full.full_refit
    full_indexes = tuple(range(EXPECTED_ROWS))
    if (
        full_ridge not in m1_real.m1.LAMBDAS_DESCENDING
        or full_bundle.indexes != full_indexes
        or full_bundle.checkpoint_reused is not False
        or full_ridge not in full_bundle.fits
    ):
        raise FreshReplayError("fresh M1 full refit binding changed")
    full_optimizer = full_bundle.fits[full_ridge]
    full_theta, full_transform, full_geometry_scales = _copy_m1_coordinates(
        full_bundle, full_optimizer
    )
    full_fit = FullDevelopmentM1Fit(
        training_indexes=full_indexes,
        ridge_lambda=full_ridge,
        theta=full_theta,
        design_transform=full_transform,
        geometry_scales=full_geometry_scales,
        fit_identity=f"{receipt.replay_identity}__M1_full_development_refit",
        optimizer_result=full_optimizer,
    )
    return M1ReplayResult(
        outer_offsets=MappingProxyType(offsets),
        full_fit=full_fit,
        cross_fit_cells=EXPECTED_ROWS,
        outer_fold_count=EXPECTED_FOLDS,
        outer_boundary_count=int(cross_fit.boundary_count),
        warm_path_count=len(accounting.warm_keys),
        zero_refit_count=len(accounting.zero_keys),
        generation_receipt=receipt,
    )


def _peak_rss_gib() -> float:
    """Return the conservative process-lifetime resident-set high-water mark."""
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**3 if sys.platform == "darwin" else 1024.0**2
    value = raw / divisor
    if not math.isfinite(value) or value <= 0.0:
        raise FreshReplayError("process peak RSS measurement is invalid")
    return value


def _time_operation(operation: Callable[[], Any]) -> TimedReplay:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = operation()
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    peak_rss = _peak_rss_gib()
    if not (
        math.isfinite(wall_seconds)
        and wall_seconds > 0.0
        and math.isfinite(cpu_seconds)
        and cpu_seconds >= 0.0
    ):
        raise FreshReplayError("aggregate replay timing is invalid")
    return TimedReplay(
        result=result,
        timing=AggregateTiming(
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_gib=peak_rss,
            scratch_gib=0.0,
        ),
    )


def time_fresh_m0_replay(context: Any) -> TimedReplay:
    """Time one complete fresh M0 nested replay plus full refit."""
    return _time_operation(lambda: run_fresh_m0_replay(context))


def time_fresh_m1_replay(context: Any) -> TimedReplay:
    """Time one complete fresh M1 nested replay plus full refit."""
    return _time_operation(lambda: run_fresh_m1_replay(context))


def run_fresh_replay_benchmark(context: Any) -> FreshReplayBenchmarkResult:
    """Run and separately time the complete fresh M0 and M1 replay prefix."""
    return FreshReplayBenchmarkResult(
        m0=time_fresh_m0_replay(context),
        m1=time_fresh_m1_replay(context),
    )


def aggregate_benchmark_units(
    replay: FreshReplayBenchmarkResult,
) -> Mapping[str, Mapping[str, Any]]:
    """Return only aggregate resource metrics for benchmark report merging."""
    if not isinstance(replay, FreshReplayBenchmarkResult):
        raise FreshReplayError("aggregate timings require a fresh replay result")
    return replay.aggregate_units()


__all__ = (
    "AggregateTiming",
    "EXPECTED_M1_WARM_PATHS",
    "EXPECTED_M1_ZERO_REFITS",
    "EXPECTED_SYNTHETIC_SEED",
    "FoldBoundM1Offset",
    "FreshReplayBenchmarkResult",
    "FreshReplayError",
    "FullDevelopmentM1Fit",
    "M0ReplayResult",
    "M1ReplayResult",
    "SyntheticGenerationReceipt",
    "TimedReplay",
    "aggregate_benchmark_units",
    "context_to_m0_records",
    "in_memory_m1_warm_factory",
    "in_memory_m1_zero_factory",
    "run_fresh_m0_replay",
    "run_fresh_m1_replay",
    "run_fresh_replay_benchmark",
    "time_fresh_m0_replay",
    "time_fresh_m1_replay",
    "validate_synthetic_generation",
)
