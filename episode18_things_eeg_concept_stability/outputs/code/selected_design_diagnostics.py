"""Exact selected-design checks for EP18's eight frozen false partitions.

Analyst definition selected prospectively on 2026-10-12, not a claim that the
policy uniquely defines the word 'spectrum': compare the singular values s of
F=[(I-H0)Za; sqrt(P0)B], the same conditional design used by slot_capacity.
Keep s**2 as a diagnostic; the .10 relative spectrum bound applies to s.
Leverage means the diagonal of the full joint weighted-data smoother, including
the refitted base, at each model's selected registered penalty. Zero-weight
rows remain in the computed operator but not the leverage distribution.

Both distributions must pass the existing standardized mean/Wasserstein-1/
quantile bounds AND the additional relative-quantile bound. Applying both is
the prospectively selected interpretation of 'every tolerance', not uniquely
prescribed source wording. Quantiles use linear interpolation at the exact five
registered probabilities. The true distribution's sample SD is the scale;
zero SD requires sorted empirical distributions equal within 1e-12. A zero
reference quantile in the additional relative check likewise uses 1e-12
absolute equality. No epsilon, stochastic trace or singular-value truncation.

Stage one factors the common weighted/penalized base once. Stage two evaluates
one true/false slot at a time, retaining only its spectrum/EDF/selected leverage
after releasing its dense factors. Stage three compares exactly eight slots;
stage four requires all six outer folds and the identical frozen bank IDs.
No callback supplies the mathematics or acceptance decision. Callers materialize
the real expanded designs with their original event times and exact common
weights/base penalties; this module does not read data, select a design/bank,
refit EEG, regenerate failed partitions, or qualify feature/context matching.

The SVD is still dense. estimate_array_bytes can reject impossible shapes
BEFORE caller expansion; later guards bound explicit array estimates, excluding
native LAPACK workspace and caller-owned unrelated caches. Row chunking bounds
E@V projection temporaries, not the underlying dense SVD. A coordinator-owned
hard memory/time limit remains necessary. Resource refusal is not an observed
scientific matching failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
import re
from typing import Mapping

import numpy as np

from slot_capacity import (_BaseFactors, _capacity_result, _evaluate_slot,
    _factor_base, RIDGE_GRID, SlotCapacityCurve, admissible_capacity_pairs)


RECIPE_ID = "conditional_design_singular_values_and_joint_leverage_20261012"
QUANTILES = (.05, .25, .50, .75, .95)
STANDARDIZED_LIMIT = .10
RELATIVE_QUANTILE_LIMIT = .10
ZERO_ABSOLUTE_LIMIT = 1e-12
FALSE_SLOTS = tuple(f"false_group_{index}" for index in range(1, 9))


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def estimate_array_bytes(*, rows, base_columns, slot_columns, projection_chunk_rows):
    """Conservative dense-array admission estimate, not native workspace RAM.

    This pure shape calculation can run before any real design is expanded.
    Includes the caller's current base/slot input arrays, float64 copies,
    weighted designs, augmented roots, SVD factors, boolean-selected base
    factors, square-reduction and projection/division temporaries. Other
    caller-owned designs/caches remain outside this one-slot estimate.
    """
    n = _positive(rows, "rows")
    p = _nonnegative(base_columns, "base columns")
    q = _nonnegative(slot_columns, "slot columns")
    chunk = min(n, _positive(projection_chunk_rows, "projection chunk rows"))
    # The base caller input, copied base, weighted base, augmented root, SVD
    # left factor, selected left factor and its square can coexist. Reserve
    # eight n*p terms rather than excluding input/reduction storage silently.
    base = 8 * (8*n*p + 6*p*p + 10*n + 10*p)
    slot = 8 * (2*n*p + 2*p*p + 6*n*q + 6*p*q + 3*q*q
                + 3*chunk*q + 24*n + 24*q + 10*p)
    return {"base_factorization_bytes": base, "slot_factorization_bytes": slot,
            "required_array_bytes": max(base, slot), "native_workspace_included": False}


@dataclass(frozen=True)
class SelectedDesignBase:
    recipe_id: str
    factors: _BaseFactors = field(repr=False, compare=False)
    array_byte_cap: int
    projection_chunk_rows: int


def prepare_selected_base(*, recipe_id, base_design, sample_weights, base_penalties,
                          base_rank_relative_tolerance, array_byte_cap, projection_chunk_rows):
    """Factor once before sequential true/eight-false evaluation; no responses."""
    if recipe_id != RECIPE_ID:
        raise ValueError("Select the prospectively recorded diagnostic definition explicitly")
    limit = _positive(array_byte_cap, "array byte cap")
    chunk = _positive(projection_chunk_rows, "projection chunk rows")
    shape = np.shape(base_design)
    if len(shape) != 2:
        raise ValueError("Base design must have rows and columns")
    estimate = estimate_array_bytes(rows=shape[0], base_columns=shape[1], slot_columns=0,
                                    projection_chunk_rows=chunk)
    if estimate["required_array_bytes"] > limit:
        raise MemoryError("Dense base exceeds the explicit array admission allowance")
    factors = _factor_base(base_design=base_design, sample_weights=sample_weights,
                          base_penalties=base_penalties,
                          base_rank_relative_tolerance=base_rank_relative_tolerance)
    if np.count_nonzero(factors.root_weights[:, 0]) < 2:
        raise ValueError("Distribution matching requires at least two retained training rows")
    return SelectedDesignBase(RECIPE_ID, factors, limit, chunk)


@dataclass(frozen=True)
class SelectedSlotDiagnostics:
    shared_base: SelectedDesignBase = field(repr=False, compare=False)
    slot_name: str
    selected_penalty_index: int
    curve: SlotCapacityCurve
    # Includes exact structural null directions when columns exceed root rows;
    # numerical singular values are never dropped/rounded/clipped to zero.
    design_singular_values: tuple[float, ...]
    conditional_gram_eigenvalues: tuple[float, ...]
    joint_leverage: np.ndarray = field(repr=False, compare=False)
    array_admission_bytes: int

    @property
    def selected_term_edf(self):
        return self.curve.term_edf[self.selected_penalty_index]


def evaluate_selected_slot(shared_base, *, slot_name, slot_design, selected_penalty_index):
    """Complete all 13 EDF values and exact selected joint leverage for one slot."""
    if not isinstance(shared_base, SelectedDesignBase) or shared_base.recipe_id != RECIPE_ID:
        raise ValueError("Use this recipe's prepared shared base")
    index = _nonnegative(selected_penalty_index, "selected penalty index")
    if index >= len(RIDGE_GRID):
        raise ValueError("The selected penalty must be one of the original thirteen")
    shape = np.shape(slot_design)
    if len(shape) != 2 or shape[0] != shared_base.factors.rows or shape[1] < 2:
        raise ValueError("Slot requires the same training rows and at least two design columns")
    estimate = estimate_array_bytes(rows=shape[0], base_columns=shared_base.factors.columns,
        slot_columns=shape[1], projection_chunk_rows=shared_base.projection_chunk_rows)
    if estimate["required_array_bytes"] > shared_base.array_byte_cap:
        raise MemoryError("Dense slot exceeds the explicit array admission allowance")
    curve, leverages = _evaluate_slot(shared_base.factors, slot_name, slot_design,
        projection_chunk_rows=shared_base.projection_chunk_rows, leverage_penalty_indices=(index,))
    spectrum = curve.schur_singular_values + (0.,) * (curve.slot_columns - len(curve.schur_singular_values))
    eigenvalues = tuple(float(value * value) for value in spectrum)
    if not np.isfinite(eigenvalues).all():
        raise ValueError("Conditional Gram spectrum is not numerically representable")
    return SelectedSlotDiagnostics(shared_base, slot_name, index, curve, spectrum,
        eigenvalues, leverages[index], estimate["required_array_bytes"])


@dataclass(frozen=True)
class DistributionMatch:
    count: int
    reference_sample_sd: float
    reference_quantiles: tuple[float, ...]
    comparison_quantiles: tuple[float, ...]
    absolute_mean_difference: float
    wasserstein_1: float
    absolute_quantile_differences: tuple[float, ...]
    standardized_mean_difference: float | None
    standardized_wasserstein_1: float | None
    standardized_quantile_differences: tuple[float, ...] | None
    relative_quantile_differences: tuple[float | None, ...]
    zero_scale_exact_equality: bool | None
    standardized_pass: bool
    relative_quantile_pass: bool
    passed: bool


def compare_design_distribution(reference, comparison):
    """Exact empirical equal-size W1 and all fixed distribution/relative bounds."""
    raw_reference, raw_comparison = np.asarray(reference), np.asarray(comparison)
    if (raw_reference.dtype.kind not in "biuf" or raw_comparison.dtype.kind not in "biuf"
            or raw_reference.ndim != 1 or raw_comparison.shape != raw_reference.shape
            or len(raw_reference) < 2):
        raise ValueError("Compare equally sized real design distributions with sample-SD support")
    ref, other = np.sort(raw_reference.astype(np.float64)), np.sort(raw_comparison.astype(np.float64))
    if (not np.isfinite(ref).all() or not np.isfinite(other).all()
            or np.any(ref < 0) or np.any(other < 0)):
        raise ValueError("Design singular values and joint leverage must be finite/nonnegative")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scale = 0. if np.all(ref == ref[0]) else float(np.std(ref, ddof=1))
        ref_q = np.quantile(ref, QUANTILES, method="linear")
        other_q = np.quantile(other, QUANTILES, method="linear")
        mean_difference = float(abs(np.mean(other) - np.mean(ref)))
        wasserstein = float(np.mean(np.abs(other - ref)))
        differences = np.abs(other_q - ref_q)
        if scale:
            standardized = (mean_difference / scale, wasserstein / scale, differences / scale)
            if not all(np.isfinite(value).all() for value in standardized):
                raise ValueError("Standardized design discrepancy is not representable")
            std_mean, std_w1 = map(float, standardized[:2])
            std_q = tuple(map(float, standardized[2]))
            zero_equal = None
            standard_pass = max(std_mean, std_w1, *std_q) <= STANDARDIZED_LIMIT
        else:
            std_mean = std_w1 = std_q = None
            zero_equal = bool(np.all(np.abs(other - ref) <= ZERO_ABSOLUTE_LIMIT))
            standard_pass = zero_equal
        relative, relative_pass = [], True
        for reference_q, difference in zip(ref_q, differences):
            if reference_q == 0:
                relative.append(None)
                relative_pass &= bool(difference <= ZERO_ABSOLUTE_LIMIT)
            else:
                value = float(difference / reference_q)
                if not np.isfinite(value):
                    raise ValueError("Relative design discrepancy is not representable")
                relative.append(value)
                relative_pass &= value <= RELATIVE_QUANTILE_LIMIT
    return DistributionMatch(len(ref), scale, tuple(map(float, ref_q)), tuple(map(float, other_q)),
        mean_difference, wasserstein, tuple(map(float, differences)), std_mean, std_w1, std_q,
        tuple(relative), zero_equal, bool(standard_pass), bool(relative_pass),
        bool(standard_pass and relative_pass))


@dataclass(frozen=True)
class FalseSlotMatch:
    false_slot: str
    reference_penalty_index: int
    comparison_penalty_index: int
    reference_term_edf: float
    comparison_term_edf: float
    term_edf_relative_difference: float | None
    term_edf_pass: bool
    spectrum: DistributionMatch
    joint_leverage: DistributionMatch
    passed: bool


@dataclass(frozen=True)
class SelectedOuterFoldMatch:
    recipe_id: str
    outer_index: int
    frozen_partition_fingerprints: tuple[str, ...]
    common_concept_penalty_index: int
    false_matches: tuple[FalseSlotMatch, ...]
    passed: bool


def compare_selected_outer_fold(*, outer_index, true_slot, false_slots: Mapping[str, SelectedSlotDiagnostics],
                                frozen_partition_fingerprints):
    """Check one common concept index and all eight frozen slots, with no retuning.

    Fingerprints bind the caller's already-frozen literal partitions across
    folds. This math component does not manufacture or authenticate membership.
    """
    outer = _nonnegative(outer_index, "outer index")
    identities = tuple(frozen_partition_fingerprints)
    if (outer >= 6 or not isinstance(true_slot, SelectedSlotDiagnostics)
            or true_slot.slot_name != "true_concept" or set(false_slots) != set(FALSE_SLOTS)):
        raise ValueError("One true slot, exactly eight false slots and one original outer fold required")
    if len(identities) != 8 or len(set(identities)) != 8 or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in identities):
        raise ValueError("Supply eight distinct frozen partition fingerprints in bank order")
    shared = true_slot.shared_base
    values = tuple(false_slots[name] for name in FALSE_SLOTS)
    if any(not isinstance(value, SelectedSlotDiagnostics) or value.shared_base is not shared
           or value.slot_name != name or value.curve.slot_columns != true_slot.curve.slot_columns
           for name, value in zip(FALSE_SLOTS, values)):
        raise ValueError("Every matched slot must retain the identical base, rows/weights and slot dimension")
    capacity = _capacity_result(shared.factors, (true_slot.curve,) + tuple(value.curve for value in values))
    retained = shared.factors.root_weights[:, 0] > 0
    matches = []
    for name, value in zip(FALSE_SLOTS, values):
        feasible = admissible_capacity_pairs(capacity, reference_slot="true_concept", comparison_slot=name,
                                            comparison_kind="false_group", df_definition="term_edf")
        pair = next((match for match in feasible.admissible_pairs
            if match.reference_penalty_index == true_slot.selected_penalty_index
            and match.comparison_penalty_index == value.selected_penalty_index), None)
        spectrum = compare_design_distribution(true_slot.design_singular_values, value.design_singular_values)
        leverage = compare_design_distribution(true_slot.joint_leverage[retained], value.joint_leverage[retained])
        reference_df, comparison_df = true_slot.selected_term_edf, value.selected_term_edf
        relative = (comparison_df - reference_df) / reference_df if reference_df > 0 else None
        matches.append(FalseSlotMatch(name, true_slot.selected_penalty_index, value.selected_penalty_index,
            reference_df, comparison_df, relative, pair is not None, spectrum, leverage,
            pair is not None and spectrum.passed and leverage.passed))
    return SelectedOuterFoldMatch(RECIPE_ID, outer, identities, true_slot.selected_penalty_index,
                                   tuple(matches), all(match.passed for match in matches))


def assemble_selected_outer_folds(folds):
    """Complete selected-design result only; no feature/context/episode release."""
    values = tuple(folds)
    if (len(values) != 6 or any(not isinstance(value, SelectedOuterFoldMatch) for value in values)
            or {value.outer_index for value in values} != set(range(6))):
        raise ValueError("All six distinct original outer folds are mandatory")
    first = values[0]
    if any(value.recipe_id != RECIPE_ID or value.frozen_partition_fingerprints != first.frozen_partition_fingerprints
           or tuple(match.false_slot for match in value.false_matches) != FALSE_SLOTS for value in values):
        raise ValueError("The same literal frozen bank and diagnostic definition must survive every fold")
    ordered = tuple(sorted(values, key=lambda value: value.outer_index))
    return {"recipe_id": RECIPE_ID, "scope": "selected_design_spectrum_joint_leverage_term_edf_only",
            "status": "selected_design_checks_pass" if all(value.passed for value in ordered)
                      else "selected_design_match_failure",
            "outer_folds": ordered, "false_bank_fully_qualified": False,
            "EEG_scored": False, "bank_regeneration_or_penalty_fallback": False}
