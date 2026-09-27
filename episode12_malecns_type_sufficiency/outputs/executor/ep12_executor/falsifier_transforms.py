"""Deterministic, conservation-checked transforms for EP12 falsifier trials.

These primitives contain no MaleCNS loader and do not choose scientific pass
thresholds.  They transform already-authorized development arrays, preserve
the invariants declared by each control, and emit enough information for a
caller to fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .policy import digest_bytes, digest_object


Array = np.ndarray


class FalsifierTransformError(ValueError):
    """A requested falsifier transform cannot preserve its contract."""


def _counts(value: Array) -> Array:
    result = np.asarray(value)
    if result.ndim != 2 or not np.issubdtype(result.dtype, np.integer):
        raise FalsifierTransformError("counts must be a two-dimensional integer array")
    if np.any(result < 0):
        raise FalsifierTransformError("counts must be nonnegative")
    return result.astype(np.int64, copy=True)


def _labels(value: Sequence[object], rows: int, name: str) -> Array:
    result = np.asarray(value, dtype=object)
    if result.ndim != 1 or len(result) != rows:
        raise FalsifierTransformError(f"{name} must contain one value per row")
    return result


def _array_digest(value: Array) -> str:
    contiguous = np.ascontiguousarray(value)
    return digest_object(
        {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "bytes_sha256": digest_bytes(contiguous.tobytes()),
        }
    )


@dataclass(frozen=True)
class TransformReceipt:
    transform: str
    seed: int | None
    input_sha256: str
    output_sha256: str
    row_totals_preserved: bool
    column_totals_preserved: bool
    total_mass_preserved: bool
    details: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform,
            "seed": self.seed,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "row_totals_preserved": self.row_totals_preserved,
            "column_totals_preserved": self.column_totals_preserved,
            "total_mass_preserved": self.total_mass_preserved,
            "details": self.details,
        }


def binary_edge_counts(counts: Array) -> tuple[Array, TransformReceipt]:
    """Replace positive weights by one; this intentionally changes strengths."""

    source = _counts(counts)
    result = (source > 0).astype(np.int64)
    receipt = TransformReceipt(
        transform="binary_edge_counts",
        seed=None,
        input_sha256=_array_digest(source),
        output_sha256=_array_digest(result),
        row_totals_preserved=bool(np.array_equal(source.sum(1), result.sum(1))),
        column_totals_preserved=bool(np.array_equal(source.sum(0), result.sum(0))),
        total_mass_preserved=bool(source.sum() == result.sum()),
        details={"positive_support_preserved": bool(np.array_equal(source > 0, result > 0))},
    )
    return result, receipt


def shuffle_side_membership(
    sides: Sequence[object],
    strata: Sequence[object],
    *,
    seed: int,
) -> tuple[Array, dict[str, object]]:
    """Shuffle L/R membership within each stratum while preserving side counts."""

    side_values = np.asarray(sides, dtype=object)
    stratum_values = _labels(strata, len(side_values), "strata")
    if side_values.ndim != 1 or any(value not in {"L", "R"} for value in side_values):
        raise FalsifierTransformError("sides must contain only L and R")
    result = side_values.copy()
    rng = np.random.default_rng(int(seed))
    before: dict[str, dict[str, int]] = {}
    after: dict[str, dict[str, int]] = {}
    for stratum in sorted(set(stratum_values.tolist()), key=str):
        indices = np.flatnonzero(stratum_values == stratum)
        if len(indices) < 2:
            raise FalsifierTransformError(f"side-shuffle stratum has fewer than two rows: {stratum}")
        before[str(stratum)] = {
            "L": int(np.sum(side_values[indices] == "L")),
            "R": int(np.sum(side_values[indices] == "R")),
        }
        result[indices] = result[indices][rng.permutation(len(indices))]
        after[str(stratum)] = {
            "L": int(np.sum(result[indices] == "L")),
            "R": int(np.sum(result[indices] == "R")),
        }
    if before != after:
        raise FalsifierTransformError("side shuffle failed to preserve stratum side counts")
    return result, {
        "transform": "within_stratum_side_membership_shuffle",
        "seed": int(seed),
        "stratum_side_counts_before": before,
        "stratum_side_counts_after": after,
        "changed_row_count": int(np.sum(result != side_values)),
    }


def shuffle_partner_identities_on_side(
    counts: Array,
    sides: Sequence[object],
    typed_columns: Sequence[int],
    *,
    shuffled_side: str,
    seed: int,
) -> tuple[Array, TransformReceipt]:
    """Permute typed partner identities on one side and retain endpoint columns."""

    source = _counts(counts)
    side_values = _labels(sides, source.shape[0], "sides")
    if shuffled_side not in {"L", "R"}:
        raise FalsifierTransformError("shuffled_side must be L or R")
    columns = np.asarray(list(typed_columns), dtype=np.int64)
    if columns.ndim != 1 or len(columns) < 2 or len(set(columns.tolist())) != len(columns):
        raise FalsifierTransformError("typed_columns must contain at least two unique columns")
    if np.any(columns < 0) or np.any(columns >= source.shape[1]):
        raise FalsifierTransformError("typed column is outside the count matrix")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(columns)
    result = source.copy()
    row_mask = side_values == shuffled_side
    result[np.ix_(row_mask, columns)] = source[np.ix_(row_mask, permutation)]
    row_ok = bool(np.array_equal(source.sum(1), result.sum(1)))
    total_ok = bool(source.sum() == result.sum())
    if not row_ok or not total_ok:
        raise FalsifierTransformError("partner shuffle failed to preserve neuron totals")
    receipt = TransformReceipt(
        transform="typed_partner_identity_shuffle_on_one_side",
        seed=int(seed),
        input_sha256=_array_digest(source),
        output_sha256=_array_digest(result),
        row_totals_preserved=row_ok,
        column_totals_preserved=bool(np.array_equal(source.sum(0), result.sum(0))),
        total_mass_preserved=total_ok,
        details={
            "shuffled_side": shuffled_side,
            "typed_columns": columns.tolist(),
            "permuted_source_columns": permutation.tolist(),
            "endpoint_columns_unchanged": bool(
                np.array_equal(
                    source[:, np.setdiff1d(np.arange(source.shape[1]), columns)],
                    result[:, np.setdiff1d(np.arange(source.shape[1]), columns)],
                )
            ),
        },
    )
    return result, receipt


def margin_preserving_switch_randomization(
    counts: Array,
    strata: Sequence[object],
    *,
    seed: int,
    requested_switches: int | None = None,
) -> tuple[Array, TransformReceipt]:
    """Apply exact integer 2x2 switches within strata.

    A switch subtracts equal mass from two diagonal cells and adds it to the
    off-diagonal cells.  It therefore preserves every row total and every
    partner-column total exactly.  The default number of requested switches is
    derived from the observed support (one request per nonzero cell), rather
    than introducing a scientific threshold.
    """

    source = _counts(counts)
    stratum_values = _labels(strata, source.shape[0], "strata")
    result = source.copy()
    attempts = int(np.count_nonzero(source)) if requested_switches is None else int(requested_switches)
    if attempts < 1:
        raise FalsifierTransformError("at least one switch must be requested")
    rng = np.random.default_rng(int(seed))
    completed = 0
    unique_strata = sorted(set(stratum_values.tolist()), key=str)
    eligible = [
        value for value in unique_strata if np.sum(stratum_values == value) >= 2
    ]
    if not eligible:
        raise FalsifierTransformError("no stratum contains two rows for an exact switch")
    for _ in range(attempts):
        stratum = eligible[int(rng.integers(len(eligible)))]
        rows = np.flatnonzero(stratum_values == stratum)
        block = result[rows]
        positive = np.argwhere(block > 0)
        if len(positive) < 2:
            continue
        order = rng.permutation(len(positive))
        chosen: tuple[int, int, int, int] | None = None
        for left_index in order:
            r1, c1 = positive[left_index]
            for right_index in order:
                r2, c2 = positive[right_index]
                if r1 != r2 and c1 != c2:
                    chosen = (int(r1), int(c1), int(r2), int(c2))
                    break
            if chosen is not None:
                break
        if chosen is None:
            continue
        r1, c1, r2, c2 = chosen
        maximum = min(int(block[r1, c1]), int(block[r2, c2]))
        delta = int(rng.integers(1, maximum + 1))
        global_r1 = int(rows[r1])
        global_r2 = int(rows[r2])
        result[global_r1, c1] -= delta
        result[global_r2, c2] -= delta
        result[global_r1, c2] += delta
        result[global_r2, c1] += delta
        completed += 1
    row_ok = bool(np.array_equal(source.sum(1), result.sum(1)))
    column_ok = bool(np.array_equal(source.sum(0), result.sum(0)))
    total_ok = bool(source.sum() == result.sum())
    if not row_ok or not column_ok or not total_ok or np.any(result < 0):
        raise FalsifierTransformError("integer switch randomization violated exact margins")
    if completed == 0:
        raise FalsifierTransformError("no nontrivial exact integer switch was available")
    receipt = TransformReceipt(
        transform="within_stratum_integer_2x2_switch_randomization",
        seed=int(seed),
        input_sha256=_array_digest(source),
        output_sha256=_array_digest(result),
        row_totals_preserved=row_ok,
        column_totals_preserved=column_ok,
        total_mass_preserved=total_ok,
        details={
            "requested_switches": attempts,
            "completed_switches": completed,
            "stratum_count": len(unique_strata),
            "derived_default": requested_switches is None,
        },
    )
    return result, receipt


def ablate_covariate_columns(
    covariates: Array,
    columns: Iterable[int],
    *,
    block_id: str,
) -> tuple[Array, dict[str, object]]:
    """Zero a declared common nuisance block for all three model families."""

    source = np.asarray(covariates, dtype=float)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise FalsifierTransformError("covariates must be a finite two-dimensional array")
    selected = np.asarray(sorted(set(int(value) for value in columns)), dtype=np.int64)
    if not block_id or len(selected) == 0:
        raise FalsifierTransformError("a named nonempty covariate block is required")
    if np.any(selected < 0) or np.any(selected >= source.shape[1]):
        raise FalsifierTransformError("covariate ablation column is outside the design")
    result = source.copy()
    result[:, selected] = 0.0
    return result, {
        "transform": "common_nuisance_block_ablation",
        "block_id": block_id,
        "columns": selected.tolist(),
        "applied_to_models": ["T", "U", "M"],
        "nonselected_columns_unchanged": bool(
            np.array_equal(
                source[:, np.setdiff1d(np.arange(source.shape[1]), selected)],
                result[:, np.setdiff1d(np.arange(source.shape[1]), selected)],
            )
        ),
    }


def pool_partner_columns(
    counts: Array,
    columns: Sequence[int],
    *,
    pool_column: int,
) -> tuple[Array, TransformReceipt]:
    """Move a partner-family block into one explicit retained-mass column."""

    source = _counts(counts)
    selected = np.asarray(sorted(set(int(value) for value in columns)), dtype=np.int64)
    if len(selected) == 0 or pool_column in selected:
        raise FalsifierTransformError("pooled columns must be nonempty and exclude the pool column")
    if np.any(selected < 0) or np.any(selected >= source.shape[1]):
        raise FalsifierTransformError("partner column is outside the count matrix")
    if pool_column < 0 or pool_column >= source.shape[1]:
        raise FalsifierTransformError("pool column is outside the count matrix")
    result = source.copy()
    result[:, pool_column] += result[:, selected].sum(axis=1)
    result[:, selected] = 0
    row_ok = bool(np.array_equal(source.sum(1), result.sum(1)))
    total_ok = bool(source.sum() == result.sum())
    if not row_ok or not total_ok:
        raise FalsifierTransformError("partner pooling failed to retain endpoint mass")
    receipt = TransformReceipt(
        transform="explicit_partner_family_pooling",
        seed=None,
        input_sha256=_array_digest(source),
        output_sha256=_array_digest(result),
        row_totals_preserved=row_ok,
        column_totals_preserved=bool(np.array_equal(source.sum(0), result.sum(0))),
        total_mass_preserved=total_ok,
        details={
            "pooled_columns": selected.tolist(),
            "pool_column": int(pool_column),
            "pooled_mass": int(source[:, selected].sum()),
        },
    )
    return result, receipt
