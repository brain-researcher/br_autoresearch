"""Explicit-recipe nested tuning and exact added-term EDF composition for EP18.

No empirical reader, scientific defaults, recipe selection or release authority
is supplied. Every original split is prepared once. Its designs, original 61D
targets, sample weights, nuisance fit and training whitener are reused across
the registered thirteen penalties. Baseline selection receives only baseline
validation losses. Fold-local capacity matching receives training-design EDF
only, never EEG losses. The same design-only matcher is reapplied on outer
training at the selected concept-grid index; failure does not trigger retuning.

All feasible capacity tuples are represented exactly as one Cartesian product
of label/eight-false penalty-index sets per common concept index. Expanding that
product could require 13**10 records; no tuple is dropped by this representation.
The matcher chooses independently within each fold. Requiring the same full
tuple in all five inner folds is deliberately not an eligibility condition.

Exact EDF still uses the existing dense SVD implementation. The caller's byte
limit admits retained dense training matrices, NOT the native SVD workspace,
all cached EEG, or peak process memory. A bounded compute allocation must admit
those additional costs. No stochastic EDF, capacity relaxation, retry, hidden
ridge, partial participant completion or empirical execution is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

import continuous_fold_runner as folds
from continuous_sequence_model import JointSequenceDesign
from slot_capacity import (RIDGE_GRID, SlotCapacityResult, admissible_capacity_pairs,
                           slot_capacity_curves)
from training_transforms import CommonSensorSpace, fit_residual_whitener


FALSE_MODELS = tuple(f"false_group_{index}" for index in range(1, 9))
MATCHED_MODELS = ("label_capacity_matched", "true_concept") + FALSE_MODELS
GRID_MODELS = ("image",) + MATCHED_MODELS


@dataclass(frozen=True)
class TuningPreparation:
    training_sequence_ids: tuple[str, ...]
    build_designs: Callable


@dataclass(frozen=True)
class BasePenaltyPlan:
    nuisance: tuple[float, ...]
    reduced_task_nuisance: tuple[float, ...]
    label_linear: tuple[float, ...]


@dataclass(frozen=True)
class PenaltyDimensions:
    split: folds.FoldSplit
    columns: Mapping[str, int]


@dataclass(frozen=True)
class CapacityOptions:
    concept_index: int
    label_indices: tuple[int, ...]
    false_indices: tuple[tuple[int, ...], ...]

    @property
    def tuple_count(self):
        count = len(self.label_indices)
        for values in self.false_indices:
            count *= len(values)
        return count


@dataclass(frozen=True)
class MatchedIndices:
    concept_index: int
    label_index: int
    false_indices: tuple[int, ...]


@dataclass(frozen=True)
class MatchView:
    split: folds.FoldSplit
    feasible: CapacityOptions
    capacity: SlotCapacityResult
    tie_break: str


@dataclass(frozen=True)
class BaselineChoice:
    concept_index: int
    image_index: int


@dataclass(frozen=True)
class BaselineSelectionView:
    outer_index: int
    feasible_concept_indices: tuple[int, ...]
    matched_label_inner_losses: Mapping[int, tuple[float, ...]]
    matched_label_mean_losses: Mapping[int, float]
    image_inner_losses: tuple[tuple[float, ...], ...]
    image_mean_losses: tuple[float, ...]
    tie_break: str


@dataclass(frozen=True)
class ImageConceptPenaltyView:
    split: folds.FoldSplit
    selected_image_index: int
    selected_concept_index: int
    nuisance_penalties: tuple[float, ...]
    image_columns: int
    full_columns: int
    registered_grid: tuple[float, ...]


@dataclass(frozen=True)
class TuningRecipe:
    """All callbacks/policies must be fixed before examining compared outcomes.

    base_penalties receives dimensions only; covariance receives training-only
    nuisance residuals. match_capacity sees EDF only. select_baseline sees the
    restricted view above. image_concept_penalties explicitly fixes the entire
    explanatory model penalty vector without held-out losses. Callbacks are a
    procedural API boundary, not OS isolation or proof of external provenance.
    """
    base_penalties: Callable[[PenaltyDimensions], BasePenaltyPlan]
    covariance: Callable
    match_capacity: Callable[[MatchView], MatchedIndices]
    select_baseline: Callable[[BaselineSelectionView], BaselineChoice]
    image_concept_penalties: Callable[[ImageConceptPenaltyView], tuple[float, ...]]
    fit_weight_scale: str
    fit_weight_multiplier: float
    normalization: str
    tie_break: str
    base_rank_relative_tolerance: float
    dense_design_byte_limit: int


@dataclass(frozen=True)
class InnerGridResult:
    split: folds.FoldSplit
    training_sequence_ids: tuple[str, ...]
    heldout_sequence_ids: tuple[str, ...]
    raw_sequence_losses: Mapping[str, np.ndarray]
    normalized_losses: Mapping[str, np.ndarray]
    capacity: SlotCapacityResult
    feasible: tuple[CapacityOptions, ...]
    matches: Mapping[int, MatchedIndices]


@dataclass(frozen=True)
class OuterTuningResult:
    split: folds.FoldSplit
    choice: BaselineChoice
    match: MatchedIndices
    capacity: SlotCapacityResult
    feasible: tuple[CapacityOptions, ...]
    errors: folds.FoldErrors
    normalized_losses: Mapping[str, float]


@dataclass(frozen=True)
class FittedTunedFold:
    result: OuterTuningResult
    preparation: TuningPreparation
    sensor_space: CommonSensorSpace
    whitener: object
    penalties: Mapping[str, np.ndarray]
    fits: Mapping[str, object]


@dataclass(frozen=True)
class TunedParticipant:
    participant_id: str
    phase: str
    inner: tuple[tuple[InnerGridResult, ...], ...]
    outer: tuple[OuterTuningResult, ...]
    normalized_participant_losses: Mapping[str, float]


@dataclass
class _SplitCache:
    split: folds.FoldSplit
    training: tuple
    heldout: tuple
    preparation: TuningPreparation
    train_designs: Mapping
    test_designs: Mapping
    targets: np.ndarray
    heldout_targets: tuple
    weights: np.ndarray
    penalties: Mapping
    nuisance_fit: object
    nuisance_losses: np.ndarray
    whitener: object


def normalize_losses(raw_sequence_losses, nuisance_sequence_losses, *, normalization):
    if normalization != "ratio_of_equal_sequence_means":
        raise ValueError("Select ratio_of_equal_sequence_means explicitly")
    raw, nuisance = np.asarray(raw_sequence_losses), np.asarray(nuisance_sequence_losses)
    if (raw.dtype.kind not in "iuf" or nuisance.dtype.kind not in "iuf"
            or nuisance.ndim != 1 or not len(nuisance)
            or raw.ndim not in (1, 2) or raw.shape[-1] != len(nuisance)
            or not np.isfinite(raw).all() or not np.isfinite(nuisance).all()
            or np.any(raw < 0) or np.any(nuisance < 0)):
        raise ValueError("Finite nonnegative paired sequence errors are required")
    denominator = float(np.mean(nuisance))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Nuisance mean error must be strictly positive; no epsilon is added")
    result = np.mean(raw, axis=-1) / denominator
    if not np.isfinite(result).all():
        raise ValueError("Normalized loss is not finite")
    return result


def all_feasible_tuples(capacity):
    """Lossless factorization of every jointly feasible registered-grid tuple."""
    comparisons = ("label_capacity_matched",) + FALSE_MODELS
    by_model = {}
    for name in comparisons:
        matches = admissible_capacity_pairs(capacity, reference_slot="true_concept",
            comparison_slot=name, comparison_kind="label" if name == comparisons[0] else "false_group",
            df_definition="term_edf")
        by_model[name] = {i: tuple(pair.comparison_penalty_index for pair in matches.admissible_pairs
                                  if pair.reference_penalty_index == i) for i in range(13)}
    return tuple(CapacityOptions(i, by_model[comparisons[0]][i],
                 tuple(by_model[name][i] for name in FALSE_MODELS))
                 for i in range(13) if all(by_model[name][i] for name in comparisons))


def _index(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < 13:
        raise ValueError("Choose an index of the exact thirteen registered penalties")
    return int(value)


def _match(recipe, split, capacity, option):
    selected = recipe.match_capacity(MatchView(split, option, capacity, recipe.tie_break))
    if (not isinstance(selected, MatchedIndices) or _index(selected.concept_index) != option.concept_index
            or _index(selected.label_index) not in option.label_indices
            or len(selected.false_indices) != 8
            or any(_index(index) not in allowed for index, allowed in zip(selected.false_indices, option.false_indices))):
        raise ValueError("Design-only matcher selected a tuple outside the common concept capacity set")
    return MatchedIndices(int(selected.concept_index), int(selected.label_index), tuple(map(int, selected.false_indices)))


def _penalty(values, width):
    raw = np.asarray(values)
    if (raw.dtype.kind not in "iuf" or raw.shape != (width,)
            or not np.isfinite(raw).all() or np.any(raw < 0)):
        raise ValueError("Explicit finite nonnegative penalties must cover every joint column")
    return folds._readonly(raw)


def _same_term(first, second):
    return (np.array_equal(first.features, second.features)
        and first.basis_dimension == second.basis_dimension and len(first.kernels) == len(second.kernels)
        and all(a[:2] == b[:2] and np.array_equal(a[2], b[2]) for a, b in zip(first.kernels, second.kernels)))


def _verify_base(base, expanded):
    for a, b in zip(base.sequences, expanded.sequences):
        count = len(a.schema[1])
        if (a.sequence_id != b.sequence_id or not np.array_equal(a.nuisance, b.nuisance)
                or b.schema[1][:count] != a.schema[1] or len(b.schema[1]) <= count or b.shape[1] < a.shape[1]
                or any(not _same_term(a.terms[name], b.terms[name]) for name in a.terms)):
            raise ValueError("Matched models must append a slot to the identical penalized base")


def _score(cache, model, fit, check):
    errors = []
    for index, item in enumerate(cache.heldout):
        check()
        operator = JointSequenceDesign([cache.test_designs[model].sequences[index]])
        residual = cache.heldout_targets[index] - operator @ fit.coefficients
        selected = cache.whitener.apply(residual)[item.retained_mask]
        with np.errstate(over="raise", invalid="raise"):
            error = float(np.mean(selected * selected))
        if not np.isfinite(error):
            raise ValueError("Nonfinite held-out loss cannot produce a completed fold")
        errors.append(error)
    return folds._readonly(errors)


def _prepare(sequences, split, sensor, recipe, prepare_training, limits, check):
    training = tuple(item for item in sequences if item.block in split.training_blocks)
    heldout = tuple(item for item in sequences if item.block in split.heldout_blocks)
    preparation = prepare_training(training, split=split, budget_check=check)
    if (not isinstance(preparation, TuningPreparation)
            or tuple(preparation.training_sequence_ids) != tuple(item.sequence_id for item in training)
            or not callable(preparation.build_designs)):
        raise ValueError("Preparation must bind exactly this split's training sequences")
    train = folds._designs(training, preparation, folds.FIXED_PANEL_MODELS, check)
    test = folds._designs(heldout, preparation, folds.FIXED_PANEL_MODELS, check)
    for name in train:
        if train[name].sequences[0].schema != test[name].sequences[0].schema:
            raise ValueError("Held-out columns must retain the frozen training definitions")
    for collection in (train, test):
        _verify_base(collection["image"], collection["label_linear"])
        for name in MATCHED_MODELS:
            _verify_base(collection["label_linear"], collection[name])
        _verify_base(collection["image"], collection["image_plus_true_concept"])
        # The explanatory concept slot must be the same actual time-expanded term.
        for concept, combined, base, image in zip(collection["true_concept"].sequences,
                collection["image_plus_true_concept"].sequences,
                collection["label_linear"].sequences, collection["image"].sequences):
            c_names, i_names = tuple(concept.terms)[len(base.terms):], tuple(combined.terms)[len(image.terms):]
            if (concept.schema[1][len(base.terms):] != combined.schema[1][len(image.terms):]
                    or any(not _same_term(concept.terms[a], combined.terms[b]) for a, b in zip(c_names, i_names))):
                raise ValueError("Image-plus-concept must preserve the same actual concept slot")
    plan = recipe.base_penalties(PenaltyDimensions(split, MappingProxyType({name: value.shape[1] for name, value in train.items()})))
    if not isinstance(plan, BasePenaltyPlan):
        raise ValueError("An explicit immutable shared-base penalty plan is required")
    penalties = {name: _penalty(getattr(plan, name), train[name].shape[1])
                 for name in ("nuisance", "reduced_task_nuisance", "label_linear")}
    if not np.array_equal(penalties["label_linear"][:len(penalties["nuisance"])], penalties["nuisance"]):
        raise ValueError("The shared label base must preserve the nuisance penalties")
    weight = recipe.fit_weight_multiplier / (len(training) if recipe.fit_weight_scale == "mean_sequences" else 1)
    weights = folds._readonly(np.concatenate([item.retained_mask * (weight / np.count_nonzero(item.retained_mask))
                                              for item in training]))
    targets = np.concatenate([sensor.apply(item.sensor_samples, channel_order=item.channel_order) for item in training])
    heldout_targets = tuple(sensor.apply(item.sensor_samples, channel_order=item.channel_order) for item in heldout)
    nuisance_fit = folds._fit(train["nuisance"], targets, weights, penalties["nuisance"], limits, check)
    residual = (targets - train["nuisance"] @ nuisance_fit.coefficients)[weights > 0]
    residual.setflags(write=False)
    covariance_weights = folds._readonly(weights[weights > 0])
    choice = recipe.covariance(residual, covariance_weights)
    if not isinstance(choice, folds.WhiteningChoices):
        raise ValueError("All training covariance choices must be explicit")
    whitener = fit_residual_whitener(residual, sample_weights=covariance_weights,
        center_covariance=choice.center_covariance, covariance_denominator=choice.covariance_denominator,
        shrinkage_target=choice.shrinkage_target, shrinkage_strength=choice.shrinkage_strength)
    cache = _SplitCache(split, training, heldout, preparation, train, test, targets, heldout_targets,
        weights, MappingProxyType(penalties), nuisance_fit, np.empty(0), whitener)
    cache.nuisance_losses = _score(cache, "nuisance", nuisance_fit, check)
    normalize_losses(cache.nuisance_losses, cache.nuisance_losses, normalization=recipe.normalization)
    return cache


def _expand(design, start, check):
    result = np.empty((design.shape[0], design.shape[1] - start), dtype=np.float64)
    basis = np.zeros(design.shape[1])
    for column in range(start, design.shape[1]):
        check()
        basis[column] = 1.
        result[:, column - start] = design @ basis
        basis[column] = 0.
    return result


def _capacity(cache, recipe, check):
    base = cache.train_designs["label_linear"]
    columns = base.shape[1] + sum(cache.train_designs[name].shape[1] - base.shape[1] for name in MATCHED_MODELS)
    if 8 * base.shape[0] * columns > recipe.dense_design_byte_limit:
        raise MemoryError("Exact dense EDF training matrices exceed the caller's admitted byte limit")
    check()
    dense_base = _expand(base, 0, check)
    slots = {name: _expand(cache.train_designs[name], base.shape[1], check) for name in MATCHED_MODELS}
    result = slot_capacity_curves(base_design=dense_base, slot_designs=slots,
        sample_weights=cache.weights, base_penalties=cache.penalties["label_linear"],
        base_rank_relative_tolerance=recipe.base_rank_relative_tolerance)
    check()
    return result


def _grid_penalty(cache, model, index):
    base = cache.penalties["nuisance"] if model == "image" else cache.penalties["label_linear"]
    return folds._readonly(np.concatenate((base, np.full(cache.train_designs[model].shape[1] - len(base), RIDGE_GRID[index]))))


def _inner(cache, recipe, phase, limits, check):
    capacity = _capacity(cache, recipe, check)
    options = all_feasible_tuples(capacity)
    matches = MappingProxyType({option.concept_index: _match(recipe, cache.split, capacity, option) for option in options})
    losses = {"nuisance": folds._readonly(cache.nuisance_losses[None, :])}
    for model in ("reduced_task_nuisance", "label_linear"):
        fit = folds._fit(cache.train_designs[model], cache.targets, cache.weights, cache.penalties[model], limits, check)
        losses[model] = folds._readonly(_score(cache, model, fit, check)[None, :])
    names = ("image", "label_capacity_matched") if phase == "baseline" else GRID_MODELS
    for model in names:
        values = []
        for index in range(13):
            fit = folds._fit(cache.train_designs[model], cache.targets, cache.weights,
                             _grid_penalty(cache, model, index), limits, check)
            values.append(_score(cache, model, fit, check))
        losses[model] = folds._readonly(values)
    normalized = {name: folds._readonly(normalize_losses(values, cache.nuisance_losses,
                                    normalization=recipe.normalization)) for name, values in losses.items()}
    return InnerGridResult(cache.split, tuple(item.sequence_id for item in cache.training),
        tuple(item.sequence_id for item in cache.heldout), MappingProxyType(losses),
        MappingProxyType(normalized), capacity, options, matches)


def _selection(inner, outer, recipe):
    feasible = tuple(index for index in range(13) if all(index in result.matches for result in inner))
    if not feasible:
        raise ValueError("Underidentified: no common concept index has a local match in every inner fold")
    label = {index: tuple(float(result.normalized_losses["label_capacity_matched"][result.matches[index].label_index])
                         for result in inner) for index in feasible}
    images = tuple(tuple(map(float, result.normalized_losses["image"])) for result in inner)
    view = BaselineSelectionView(outer, feasible, MappingProxyType(label),
        MappingProxyType({index: float(np.mean(values)) for index, values in label.items()}),
        images, tuple(map(float, np.mean(images, axis=0))), recipe.tie_break)
    choice = recipe.select_baseline(view)
    if (not isinstance(choice, BaselineChoice) or _index(choice.concept_index) not in feasible):
        raise ValueError("Baseline selector must choose an inner-feasible concept index")
    return BaselineChoice(_index(choice.concept_index), _index(choice.image_index))


def _outer(cache, choice, recipe, phase, sensor, limits, retain_fit, check):
    capacity = _capacity(cache, recipe, check)
    feasible = all_feasible_tuples(capacity)
    option = next((item for item in feasible if item.concept_index == choice.concept_index), None)
    if option is None:
        raise ValueError("Underidentified: the selected inner concept index fails outer-training capacity recheck")
    match = _match(recipe, cache.split, capacity, option)
    penalties = dict(cache.penalties)
    penalties["image"] = _grid_penalty(cache, "image", choice.image_index)
    penalties["label_capacity_matched"] = _grid_penalty(cache, "label_capacity_matched", match.label_index)
    if phase == "fixed_panel":
        penalties["true_concept"] = _grid_penalty(cache, "true_concept", choice.concept_index)
        penalties.update({name: _grid_penalty(cache, name, index) for name, index in zip(FALSE_MODELS, match.false_indices)})
        view = ImageConceptPenaltyView(cache.split, choice.image_index, choice.concept_index,
            tuple(map(float, penalties["nuisance"])), cache.train_designs["image"].shape[1],
            cache.train_designs["image_plus_true_concept"].shape[1], RIDGE_GRID)
        full = _penalty(recipe.image_concept_penalties(view), view.full_columns)
        nuisance_width = len(view.nuisance_penalties)
        if (not np.array_equal(full[:nuisance_width], penalties["nuisance"])
                or any(value not in RIDGE_GRID for value in full[nuisance_width:])):
            raise ValueError("Explicit explanatory penalties must preserve nuisance and use the registered grid")
        for group in (full[nuisance_width:view.image_columns], full[view.image_columns:]):
            if len(group) and not np.all(group == group[0]):
                raise ValueError("The explanatory image base and added concept slot each require one scalar ridge")
        penalties["image_plus_true_concept"] = full
    fits = {"nuisance": cache.nuisance_fit}
    errors = {"nuisance": cache.nuisance_losses}
    for name, vector in penalties.items():
        if name != "nuisance":
            fits[name] = folds._fit(cache.train_designs[name], cache.targets, cache.weights, vector, limits, check)
            errors[name] = _score(cache, name, fits[name], check)
    fold_errors = folds.FoldErrors(cache.split, tuple(item.sequence_id for item in cache.training),
        tuple(item.sequence_id for item in cache.heldout),
        tuple(int(np.count_nonzero(item.retained_mask)) for item in cache.heldout), MappingProxyType(errors))
    normalized = MappingProxyType({name: float(normalize_losses(values, cache.nuisance_losses,
                      normalization=recipe.normalization)) for name, values in errors.items()})
    result = OuterTuningResult(cache.split, choice, match, capacity, feasible, fold_errors, normalized)
    check()
    retain_fit(FittedTunedFold(result, cache.preparation, sensor, cache.whitener,
                              MappingProxyType(penalties), MappingProxyType(fits)))
    return result


def run_tuned_participant(*, participant_id, sequences, frozen_channel_order, phase,
                          recipe, prepare_training, solver_limits, retain_fit, budget_check):
    """Complete all 30 inner plus six outer splits with fixed original model panels.

    Baseline phase creates concept/false training designs for outcome-blind EDF
    but never fits or scores those responses. Fixed-panel inner grids retain
    their raw losses in results, but cannot expose them to baseline selection.
    The explicit explanatory penalty policy runs only on outer training.
    All 13 grid fits are evaluated even if some cannot form a matched tuple.
    """
    if (not isinstance(participant_id, str) or not participant_id or phase not in ("baseline", "fixed_panel")
            or not isinstance(recipe, TuningRecipe) or not isinstance(solver_limits, folds.SolverLimits)
            or not all(callable(value) for value in (prepare_training, retain_fit, budget_check,
                recipe.base_penalties, recipe.covariance, recipe.match_capacity,
                recipe.select_baseline, recipe.image_concept_penalties))
            or recipe.fit_weight_scale not in ("mean_sequences", "sum_sequences")
            or recipe.normalization != "ratio_of_equal_sequence_means"
            or not isinstance(recipe.tie_break, str) or not recipe.tie_break.strip()
            or isinstance(recipe.dense_design_byte_limit, bool)
            or not isinstance(recipe.dense_design_byte_limit, Integral) or recipe.dense_design_byte_limit <= 0):
        raise ValueError("Supply the explicit immutable tuning recipe, phase and bounded callbacks")
    folds._scalar(recipe.fit_weight_multiplier, "fitting-weight multiplier")
    tolerance = recipe.base_rank_relative_tolerance
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("Supply an explicit exact-SVD relative rank tolerance")
    for name in ("atol", "btol", "conlim"):
        folds._scalar(getattr(solver_limits, name), name)
    if (solver_limits.conlim <= 1 or isinstance(solver_limits.maxiter, bool)
            or not isinstance(solver_limits.maxiter, Integral) or solver_limits.maxiter < 1):
        raise ValueError("Supply explicit solver condition and iteration bounds")
    budget_check()
    sensor = CommonSensorSpace(frozen_channel_order)
    sequences = folds._roster(sequences, sensor, budget_check)
    all_inner, outer = [], []
    for outer_index in range(6):
        inner = []
        for inner_index in range(6):
            if inner_index == outer_index:
                continue
            cache = _prepare(sequences, folds._split(outer_index, inner_index), sensor,
                             recipe, prepare_training, solver_limits, budget_check)
            inner.append(_inner(cache, recipe, phase, solver_limits, budget_check))
            del cache
        choice = _selection(tuple(inner), outer_index, recipe)
        cache = _prepare(sequences, folds._split(outer_index), sensor,
                         recipe, prepare_training, solver_limits, budget_check)
        outer.append(_outer(cache, choice, recipe, phase, sensor, solver_limits, retain_fit, budget_check))
        del cache
        all_inner.append(tuple(inner))
    participant_losses = MappingProxyType({name: float(np.mean([result.normalized_losses[name] for result in outer]))
                                          for name in outer[0].normalized_losses})
    budget_check()
    return TunedParticipant(participant_id, phase, tuple(all_inner), tuple(outer), participant_losses)
