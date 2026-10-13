"""Conditional added-term EDF for EP18's existing joint weighted ridge model.

No response or held-out data enter this array-only component. The caller must
provide the already-expanded training designs, including their actual event
times and common temporal basis, explicit sample weights and base penalties.
All slots in one call share that base. Slot coefficient geometry is explicit:
this module does not turn raw RBF kernel columns into an RKHS feature map.

For Z0=sqrt(W)X0, Za=sqrt(W)Xa and P0=diag(base_penalties), the joint objective
is ||sqrt(W)Y-Z0*b-Za*a||_F^2 + b'P0*b + lambda*a'a. The primary capacity is
the trace of the fitted added term's influence, not the change in total model
EDF when the base coefficients are refitted. Both are returned, per sensor
contrast and conditional on the supplied transforms, weights and penalties.

This exact spectral implementation uses dense rectangular arrays/SVD. Its
resource cost at the full episode dimensions remains unmeasured; it is not a
matrix-free full-scale capacity implementation or a scientific-release gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping

import numpy as np


RIDGE_LOG10_GRID = tuple(range(-6, 7))
RIDGE_GRID = tuple(10.0 ** power for power in RIDGE_LOG10_GRID)
MAXIMUM_RELATIVE_DF_DIFFERENCE = 0.05
EDF_DEFINITIONS = ("term_edf", "incremental_model_edf")


def _real(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric data")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _tuple(values):
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class SlotCapacityCurve:
    slot_name: str
    slot_columns: int
    schur_singular_values: tuple[float, ...]
    term_edf: tuple[float, ...]
    incremental_model_edf: tuple[float, ...]
    total_model_edf: tuple[float, ...]


@dataclass(frozen=True)
class SlotCapacityResult:
    ridge_penalties: tuple[float, ...]
    base_columns: int
    base_numerical_rank: int
    base_singular_values: tuple[float, ...]
    base_rank_relative_tolerance: float
    base_rank_threshold: float
    base_model_edf: float
    curves: tuple[SlotCapacityCurve, ...]

    def curve(self, slot_name):
        for curve in self.curves:
            if curve.slot_name == slot_name:
                return curve
        raise ValueError(f"Unknown added slot: {slot_name}")


@dataclass(frozen=True)
class _BaseFactors:
    rows: int
    columns: int
    root_weights: np.ndarray
    data_left: np.ndarray
    penalty_left: np.ndarray
    singular_values: tuple[float, ...]
    numerical_rank: int
    rank_relative_tolerance: float
    rank_threshold: float
    model_edf: float


def _factor_base(*, base_design, sample_weights, base_penalties,
                 base_rank_relative_tolerance):
    """Shared exact augmented-base SVD; no EEG target or penalty selection."""
    base = _real(base_design, 2, "base design")
    rows, columns = base.shape
    if rows < 1:
        raise ValueError("Training design must contain rows")
    weights = _real(sample_weights, 1, "sample weights")
    penalties = _real(base_penalties, 1, "base penalties")
    if (weights.shape != (rows,) or np.any(weights < 0)
            or not np.any(weights > 0)):
        raise ValueError("Supply common nonnegative weights with retained training support")
    if penalties.shape != (columns,) or np.any(penalties < 0):
        raise ValueError("Supply a nonnegative penalty for every base column")
    tolerance = base_rank_relative_tolerance
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or not 0 <= tolerance < 1):
        raise ValueError("Supply an explicit finite base SVD relative tolerance in [0,1)")
    root_weights = np.sqrt(weights)[:, None]
    with np.errstate(over="raise", invalid="raise"):
        weighted_base = root_weights * base
        augmented_base = np.vstack((weighted_base, np.diag(np.sqrt(penalties))))
    if columns:
        left, singular, _ = np.linalg.svd(augmented_base, full_matrices=False)
        threshold = float(tolerance * singular[0])
        retained = singular > threshold
        data_left = left[:rows, retained]
        penalty_left = left[rows:, retained]
    else:
        singular = np.empty(0)
        threshold = 0.0
        retained = np.zeros(0, dtype=bool)
        data_left, penalty_left = np.empty((rows, 0)), np.empty((0, 0))
    for values in (root_weights, data_left, penalty_left):
        values.setflags(write=False)
    return _BaseFactors(rows, columns, root_weights, data_left, penalty_left,
        _tuple(singular), int(np.count_nonzero(retained)), float(tolerance), threshold,
        float(np.sum(data_left ** 2)))


def _evaluate_slot(base, name, values, *, projection_chunk_rows=None,
                   leverage_penalty_indices=()):
    """One exact slot factorization, EDF curves and optional joint leverages.

    The ordinary public capacity route keeps its original full-row projection
    and reduction order. The selected-design diagnostic route projects rows in
    explicit bounded chunks; that changes only floating-point summation order,
    not the factorization, rows, singular directions or EDF definition.
    Joint leverage includes the base refit: diag(H0)+diag(E(T+lambda I)^-1 E').
    Its sum is total-model EDF, NOT primary added-term EDF.
    """
    if not isinstance(base, _BaseFactors) or not isinstance(name, str) or not name:
        raise ValueError("Supply the shared exact base and a named added slot")
    if projection_chunk_rows is not None and (isinstance(projection_chunk_rows, (bool, np.bool_))
            or not isinstance(projection_chunk_rows, Integral) or projection_chunk_rows < 1):
        raise ValueError("Projection chunk rows must be a positive integer")
    indices = tuple(leverage_penalty_indices)
    if (len(indices) != len(set(indices)) or any(isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral) or not 0 <= i < len(RIDGE_GRID) for i in indices)):
        raise ValueError("Leverages require distinct registered penalty indices")
    rows = base.rows
    slot = _real(values, 2, f"{name} added-slot design")
    if slot.shape[0] != rows:
        raise ValueError("All slots need the same training rows and weights as the base")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        weighted_slot = base.root_weights * slot
        coordinates = base.data_left.T @ weighted_slot
        residual_design = weighted_slot - base.data_left @ coordinates
        penalty_response = base.penalty_left @ coordinates
        schur_root = np.vstack((residual_design, penalty_response))
    # These copies no longer contribute to any operation or return value.
    del slot, weighted_slot, coordinates, penalty_response
    lambdas = np.asarray(RIDGE_GRID)
    columns = schur_root.shape[1]
    base_leverage = np.sum(base.data_left ** 2, axis=1) if indices else None
    leverage = {int(index): base_leverage.copy() for index in indices}
    if columns:
        unused_left, schur_singular, right = np.linalg.svd(schur_root, full_matrices=False)
        del unused_left, schur_root
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            eigenvalues = schur_singular ** 2
            denominators = eigenvalues[:, None] + lambdas[None, :]
            term_df = np.sum(eigenvalues[:, None] / denominators, axis=0)
            chunk = rows if projection_chunk_rows is None else int(projection_chunk_rows)
            incremental_numerators = np.zeros(len(schur_singular))
            for start in range(0, rows, chunk):
                stop = min(rows, start + chunk)
                squared = (residual_design[start:stop] @ right.T) ** 2
                incremental_numerators += np.sum(squared, axis=0)
                for index in indices:
                    leverage[index][start:stop] += np.sum(squared / denominators[:, index], axis=1)
            increment_df = np.sum(incremental_numerators[:, None] / denominators, axis=0)
    else:
        schur_singular = np.empty(0)
        term_df = np.zeros(len(RIDGE_GRID))
        increment_df = np.zeros(len(RIDGE_GRID))
    total_df = base.model_edf + increment_df
    if not all(np.isfinite(values).all() for values in (term_df, increment_df, total_df, *leverage.values())):
        raise ValueError("EDF or joint leverage is not numerically representable")
    for vector in leverage.values():
        vector.setflags(write=False)
    return (SlotCapacityCurve(name, columns, _tuple(schur_singular), _tuple(term_df),
                              _tuple(increment_df), _tuple(total_df)), leverage)


def _capacity_result(base, curves):
    return SlotCapacityResult(RIDGE_GRID, base.columns, base.numerical_rank, base.singular_values,
        base.rank_relative_tolerance, base.rank_threshold, base.model_edf, tuple(curves))


def slot_capacity_curves(*, base_design, slot_designs: Mapping[str, np.ndarray],
                         sample_weights, base_penalties,
                         base_rank_relative_tolerance):
    """Return both EDF curves on exactly the thirteen registered penalties.

    The SVD of [Z0; sqrt(P0)] identifies the base coefficient range using the
    caller's explicit float64 rank tolerance. This supports rank-deficient,
    unpenalized nuisance columns without a hidden ridge term. Discarded base
    directions are reported through the complete singular spectrum and rank;
    the caller must qualify this numerical range against the actual joint fit.

    For B=(Z0'Z0+P0)^+ Z0'Za and E=Za-Z0*B, T=E'E+B'P0*B=Za'R0*Za,
    where R0=I-H_base. Right singular vectors v of F=[E; sqrt(P0)*B] give
    eigenvalues s^2 of T without clipping PSD eigenvalues. Added-term EDF is
    sum s^2/(s^2+lambda); incremental model EDF instead has ||E*v||^2 in the
    numerator. No EEG residual target, stochastic trace, new penalty point,
    feature rescaling or penalty selection is introduced.
    """
    if (not slot_designs or any(not isinstance(name, str) or not name
                               for name in slot_designs)):
        raise ValueError("Supply named added-slot designs")
    base = _factor_base(base_design=base_design, sample_weights=sample_weights,
                        base_penalties=base_penalties,
                        base_rank_relative_tolerance=base_rank_relative_tolerance)
    curves = [_evaluate_slot(base, name, values)[0] for name, values in slot_designs.items()]
    return _capacity_result(base, curves)


@dataclass(frozen=True)
class CapacityMatch:
    reference_penalty_index: int
    comparison_penalty_index: int
    reference_penalty: float
    comparison_penalty: float
    reference_edf: float
    comparison_edf: float
    relative_difference: float


@dataclass(frozen=True)
class CapacityMatches:
    reference_slot: str
    comparison_slot: str
    comparison_kind: str
    df_definition: str
    operative_term_capacity: bool
    status: str
    zero_reference_penalty_indices: tuple[int, ...]
    unmatched_reference_penalty_indices: tuple[int, ...]
    admissible_pairs: tuple[CapacityMatch, ...]


def admissible_capacity_pairs(result: SlotCapacityResult, *, reference_slot,
                              comparison_slot, comparison_kind, df_definition):
    """List feasible grid pairs without choosing or fitting a penalty.

    For comparison_kind='label', reference <= label <= 1.05*reference.
    For 'false_group', absolute relative difference must be <=0.05.
    The recorded EP18 definition is term_edf; incremental_model_edf remains
    an explicitly labelled diagnostic. Zero reference EDF has undefined
    relative capacity and produces no admissible pair. A no-pair status is
    underidentified for these supplied designs, not an episode terminal.
    """
    if not isinstance(result, SlotCapacityResult) or result.ridge_penalties != RIDGE_GRID:
        raise ValueError("Use one common-base result with the exact registered ridge grid")
    if reference_slot == comparison_slot:
        raise ValueError("Reference and comparison must name different added slots")
    if comparison_kind not in ("label", "false_group"):
        raise ValueError("comparison_kind must be label or false_group")
    if df_definition not in EDF_DEFINITIONS:
        raise ValueError("Specify term_edf or incremental_model_edf explicitly")
    reference = getattr(result.curve(reference_slot), df_definition)
    comparison = getattr(result.curve(comparison_slot), df_definition)
    pairs, zero, unmatched = [], [], []
    for i, ref_df in enumerate(reference):
        if ref_df == 0.0:
            zero.append(i)
            unmatched.append(i)
            continue
        found = False
        for j, other_df in enumerate(comparison):
            difference = other_df - ref_df
            within = abs(difference) <= MAXIMUM_RELATIVE_DF_DIFFERENCE * ref_df
            if within and (comparison_kind == "false_group" or difference >= 0.0):
                pairs.append(CapacityMatch(i, j, RIDGE_GRID[i], RIDGE_GRID[j],
                                          ref_df, other_df, difference / ref_df))
                found = True
        if not found:
            unmatched.append(i)
    return CapacityMatches(reference_slot, comparison_slot, comparison_kind,
                           df_definition, df_definition == "term_edf",
                           "admissible_pairs" if pairs else "underidentified",
                           tuple(zero), tuple(unmatched), tuple(pairs))
