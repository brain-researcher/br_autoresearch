"""Inactive EP18 adapter for the existing nested-tuning capacity call sites.

No files, EEG targets, residuals, fitted coefficients or scores are read here.
The caller must pass an existing fully prepared continuous_tuning split cache
and the same explicit TuningRecipe. This module does not install a hook,
monkeypatch the controller, choose a trial/recipe, or authorize a release.

Proposed integration (existing controller remains unchanged):
* Baseline-search _inner/_outer replace _capacity with capacity_for_split.
* After one complete baseline is fixed, prepare its six outer training sets
  and call outer_capacity_and_match with the retained inner choices. Exactly
  six results go to the unchanged assemble_selected_outer_folds, within the
  original complete-cohort precomparison qualification and decision order.
* The fit driver then consumes the returned capacity/feasible/match original
  types. No compared-model fit/score may bypass the existing qualification.

Merely replacing the existing _outer call is insufficient: the present
fixed_panel runner scores inner true/false grids earlier. Its cohort driver
still needs this separate precomparison pass and the original exposure gates.

The inner-selected concept index is known before outer factorization, but the
eight false indices depend on outer capacity. Therefore the one-pass outer
route retains one true and all 13*8 false joint-leverage vectors (105*n*8
bytes), then selects the original match without another QR/SVD. Label capacity
is still evaluated on all 13 penalties. The shared base is factored once and
all ten added slots are evaluated sequentially. No penalty fallback occurs.

All distribution mathematics and acceptance constants come from the existing
selected_design_diagnostics and slot_capacity helpers. Full conditional design
singular values and their squared eigenvalue diagnostics are returned for all
nine categorical slots; only structurally absent square-padding values are 0.

The explicit array allowance includes backend arrays plus a conservative
comparison reserve of 8*(32*n + 128*q_max + 4096) bytes. Existing caller-owned
operator/feature/EEG caches and native BLAS thread buffers remain outside it.
Backend work remains O(n*p^2 + p^3) per base and
O(n*p*q + n*q^2 + q^3) per slot; throughput is unmeasured. No full-model fit,
61-target solver, 30-person aggregation, original 19--44-trial search manager,
controls, persistence or one-shot scientific-release mechanism is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import re
from types import MappingProxyType

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import selected_design_diagnostics as diagnostics
from slot_capacity import SlotCapacityResult, admissible_capacity_pairs
from streamed_slot_capacity import SequenceRows, streamed_slot_capacity_curves


@dataclass(frozen=True)
class StreamedResources:
    """Caller-selected numerical work bounds, not episode-budget authority."""
    chunk_rows: int
    qr_block_size: int
    array_byte_cap: int

    def __post_init__(self):
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1
               for value in (self.chunk_rows, self.qr_block_size, self.array_byte_cap)):
            raise ValueError("Explicit positive streamed chunk/block/array limits are required")


@dataclass(frozen=True)
class StreamedOuterCapacity:
    capacity: SlotCapacityResult
    feasible: tuple[tuning.CapacityOptions, ...]
    match: tuning.MatchedIndices
    selected_design: diagnostics.SelectedOuterFoldMatch
    design_singular_values: object
    conditional_gram_eigenvalues: object
    comparison_array_reserve: int


def _views(cache, recipe, resources, check):
    if (not isinstance(recipe, tuning.TuningRecipe) or not isinstance(resources, StreamedResources)
            or not callable(check) or set(cache.train_designs) != set(folds.FIXED_PANEL_MODELS)):
        raise ValueError("Use the complete existing 15-model split and explicit streamed recipe")
    check()
    base = cache.train_designs["label_linear"]
    shared = SequenceRows(base, 0, base.shape[1])
    slots = {name: SequenceRows(cache.train_designs[name], base.shape[1],
                                cache.train_designs[name].shape[1])
             for name in tuning.MATCHED_MODELS}
    return shared, slots


def _capacity(cache, recipe, resources, check, shared, slots, requests, reserve):
    allowance = resources.array_byte_cap - reserve
    if allowance <= 0:
        raise MemoryError("Comparison reserve exceeds the explicit streamed array allowance")
    return streamed_slot_capacity_curves(base_design=shared, slot_designs=slots,
        sample_weights=cache.weights, base_penalties=cache.penalties["label_linear"],
        base_rank_relative_tolerance=recipe.base_rank_relative_tolerance,
        chunk_rows=resources.chunk_rows, qr_block_size=resources.qr_block_size,
        array_byte_cap=allowance, leverage_penalty_indices=requests, budget_check=check)


def capacity_for_split(cache, recipe, check, *, resources):
    """Drop-in body for the original inner _capacity call; no dense expansion.

    Final selected outer diagnostics require outer_capacity_and_match instead.
    No changes to all_feasible_tuples, _match, baseline selection or grid fitting
    are needed at the inner call site.
    """
    shared, slots = _views(cache, recipe, resources, check)
    result, _ = _capacity(cache, recipe, resources, check, shared, slots,
                         {name: () for name in slots}, 0)
    return result


def outer_capacity_and_match(cache, choice, recipe, check, *, resources,
                             diagnostic_recipe_id, frozen_partition_fingerprints):
    """One original outer rematch and all eight selected design checks.

    This reports a failed design check without fitting, retuning, changing the
    chosen concept index, regenerating a bank, or declaring full qualification.
    The caller enforces the existing release/underidentified decision order.
    """
    shared, slots = _views(cache, recipe, resources, check)
    split = cache.split
    if (not isinstance(split, folds.FoldSplit) or split.inner_index is not None
            or isinstance(split.outer_index, (bool, np.bool_))
            or not isinstance(split.outer_index, Integral) or not 0 <= split.outer_index < 6
            or split != folds._split(int(split.outer_index))
            or not isinstance(choice, tuning.BaselineChoice)
            or diagnostic_recipe_id != diagnostics.RECIPE_ID):
        raise ValueError("Use the original outer split, fixed inner choice and selected diagnostic recipe")
    concept_index = tuning._index(choice.concept_index)
    tuning._index(choice.image_index)
    identities = tuple(frozen_partition_fingerprints)
    if (len(identities) != 8 or len(set(identities)) != 8 or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in identities)):
        raise ValueError("Supply the same eight literal frozen partition fingerprints in bank order")
    categorical = ("true_concept",) + tuning.FALSE_MODELS
    width = slots["true_concept"].columns
    if width < 2 or any(slots[name].columns != width for name in categorical):
        raise ValueError("All categorical slots require the identical complete dimension")
    weights = np.asarray(cache.weights)
    if weights.shape != (shared.rows,) or np.count_nonzero(weights > 0) < 2:
        raise ValueError("Selected design distributions need the original retained training rows")
    maximum_width = max(slot.columns for slot in slots.values())
    reserve = 8 * (32*shared.rows + 128*maximum_width + 4096)
    requests = {name: (() if name == "label_capacity_matched" else
                       (concept_index,) if name == "true_concept" else tuple(range(13)))
                for name in slots}
    capacity, leverages = _capacity(cache, recipe, resources, check, shared, slots, requests, reserve)
    feasible = tuning.all_feasible_tuples(capacity)
    option = next((value for value in feasible if value.concept_index == concept_index), None)
    if option is None:
        raise ValueError("Underidentified: the selected inner concept index fails outer-training capacity recheck")
    match = tuning._match(recipe, split, capacity, option)
    spectra, eigenvalues = {}, {}
    for name in categorical:
        check()
        curve = capacity.curve(name)
        spectrum = curve.schur_singular_values + (0.,) * (curve.slot_columns-len(curve.schur_singular_values))
        squared = tuple(float(value*value) for value in spectrum)
        if not np.isfinite(squared).all():
            raise ValueError("Conditional Gram spectrum is not numerically representable")
        spectra[name], eigenvalues[name] = spectrum, squared
    retained = weights > 0
    true_curve = capacity.curve("true_concept")
    reference_df = true_curve.term_edf[concept_index]
    reference_leverage = leverages["true_concept"][concept_index][retained]
    results = []
    for name, index in zip(tuning.FALSE_MODELS, match.false_indices):
        check()
        curve = capacity.curve(name)
        admissible = admissible_capacity_pairs(capacity, reference_slot="true_concept",
            comparison_slot=name, comparison_kind="false_group", df_definition="term_edf")
        pair = next((value for value in admissible.admissible_pairs
                     if value.reference_penalty_index == concept_index
                     and value.comparison_penalty_index == index), None)
        spectrum = diagnostics.compare_design_distribution(spectra["true_concept"], spectra[name])
        leverage = diagnostics.compare_design_distribution(reference_leverage,
                                                           leverages[name][index][retained])
        comparison_df = curve.term_edf[index]
        relative = (comparison_df-reference_df)/reference_df if reference_df > 0 else None
        results.append(diagnostics.FalseSlotMatch(name, concept_index, index, reference_df,
            comparison_df, relative, pair is not None, spectrum, leverage,
            pair is not None and spectrum.passed and leverage.passed))
    selected = diagnostics.SelectedOuterFoldMatch(diagnostics.RECIPE_ID, int(split.outer_index),
        identities, concept_index, tuple(results), all(value.passed for value in results))
    check()
    return StreamedOuterCapacity(capacity, feasible, match, selected,
        MappingProxyType(spectra), MappingProxyType(eigenvalues), reserve)


def assemble_outer_checks(results):
    """Preserve the existing six-fold/same-bank completeness and scope checks."""
    values = tuple(results)
    if any(not isinstance(value, StreamedOuterCapacity) for value in values):
        raise ValueError("Use complete streamed outer-capacity records")
    return diagnostics.assemble_selected_outer_folds(value.selected_design for value in values)
