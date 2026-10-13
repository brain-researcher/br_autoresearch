"""Complete EP18 participant fold composition over explicitly supplied arrays.

This is not an empirical reader or execution/release authority. A coordinated
caller supplies all 72 physical sequences, already filtered/resampled with
sequence boundaries respected, their common fixed masks, and a prospectively
selected training-preparation callback. That callback receives training
sequences only. It must fit every learned feature/preprocessing transform
there, and return a frozen design builder and explicit penalties/whitening
choices. The callback is not an OS access boundary.

All 30 inner and six outer split compositions run, serially. Outer preparation sees
only its five completed inner results, not outer held-out EEG. No recipe,
penalty, tie rule, covariance default or filter is selected here. Candidate
models fit the original 61D EEG jointly with nuisance predictors; nuisance
residuals are used solely to estimate one shared training-only whitening
metric. The existing fixed 62-to-61 sensor projection is reused. Each split
evaluates one explicit supplied penalty assignment per model. This is a
fixed-recipe composition, not the registered thirteen-penalty search or
capacity-matching controller; those remain caller prerequisites.

Results contain paired per-sequence raw whitened MSEs. No ambiguous choice
between averaging sequence ratios and normalizing mean errors is made here.
There is no participant contrast, scientific release, terminal classification,
retry or missing-fold fallback. A caller-owned sink retains fitted objects for
subsequent no-refit ablation; this module performs no file I/O. A failed run may
leave earlier sink artifacts, which do not constitute a complete participant.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from continuous_sequence_model import JointRidgeFit, JointSequenceDesign, SequenceDesign, fit_joint_ridge
from training_transforms import CommonSensorSpace, ResidualWhitener, fit_residual_whitener


BLOCK_PAIRS = tuple((index, index + 6) for index in range(6))
BASELINE_MODELS = ("nuisance", "reduced_task_nuisance", "image", "label_linear", "label_capacity_matched")
FIXED_PANEL_MODELS = BASELINE_MODELS + ("true_concept",) + tuple(
    f"false_group_{index}" for index in range(1, 9)) + ("image_plus_true_concept",)


@dataclass(frozen=True)
class PhysicalSequence:
    """A complete prepared physical sequence, not an isolated image epoch.

    preparation_input contains the caller's permitted feature/event inputs.
    Identifiers/block membership must come from the frozen source join. The
    runner verifies their complete roster; it cannot authenticate raw filtering
    support or data provenance from caller-supplied arrays.
    """
    sequence_id: str
    block: int
    within_block: int
    sample_times_seconds: np.ndarray
    sensor_samples: np.ndarray
    channel_order: tuple[str, ...]
    retained_mask: np.ndarray
    preparation_input: object


@dataclass(frozen=True)
class SequencePredictors:
    """Design-builder view: no held-out EEG or residuals are supplied."""
    sequence_id: str
    block: int
    within_block: int
    sample_times_seconds: np.ndarray
    preparation_input: object


@dataclass(frozen=True)
class FoldSplit:
    outer_index: int
    inner_index: int | None
    training_blocks: tuple[int, ...]
    heldout_blocks: tuple[int, int]


@dataclass(frozen=True)
class SolverLimits:
    atol: float
    btol: float
    conlim: float
    maxiter: int


@dataclass(frozen=True)
class WhiteningChoices:
    center_covariance: bool
    covariance_denominator: float
    shrinkage_target: np.ndarray
    shrinkage_strength: float


@dataclass(frozen=True)
class FittedFoldPreparation:
    """Training-bound preparation, returned once per exact split.

    build_designs(sequence) returns every model in the selected phase. All full
    models must share the nuisance matrix, sample axis and temporal definitions.
    The reduced-task control is the explicit exception to nuisance equality.
    whitening_choices receives only retained training nuisance residuals and weights.
    per_sequence_fit_weight explicitly fixes the loss/penalty scale; it is not
    silently normalized by the number of training sequences.
    """
    training_sequence_ids: tuple[str, ...]
    build_designs: Callable[[SequencePredictors], Mapping[str, SequenceDesign]]
    penalties: Mapping[str, np.ndarray]
    whitening_choices: Callable[[np.ndarray, np.ndarray], WhiteningChoices]
    per_sequence_fit_weight: float


@dataclass(frozen=True)
class FoldErrors:
    split: FoldSplit
    training_sequence_ids: tuple[str, ...]
    heldout_sequence_ids: tuple[str, ...]
    retained_sample_counts: tuple[int, ...]
    # Each vector has the same sequence order; no normalized loss is implied.
    whitened_mse: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class FittedFold:
    split: FoldSplit
    preparation: FittedFoldPreparation
    sensor_space: CommonSensorSpace
    whitener: ResidualWhitener
    fits: Mapping[str, JointRidgeFit]
    errors: FoldErrors


@dataclass(frozen=True)
class ParticipantFolds:
    participant_id: str
    phase: str
    inner: tuple[tuple[FoldErrors, ...], ...]
    outer: tuple[FoldErrors, ...]


def _readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be an explicit finite positive scalar")
    return float(value)


def _roster(sequences, sensor_space, check):
    sequences = tuple(sequences)
    if len(sequences) != 72 or any(not isinstance(item, PhysicalSequence) for item in sequences):
        raise ValueError("Supply all twelve blocks and six physical sequences per block")
    locations, ids, frozen = set(), set(), []
    for item in sequences:
        check()
        if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in (item.block, item.within_block))
                or not 0 <= item.block < 12 or not 0 <= item.within_block < 6
                or (item.block, item.within_block) in locations
                or not isinstance(item.sequence_id, str) or not item.sequence_id
                or item.sequence_id in ids):
            raise ValueError("Unique frozen physical-sequence and block identities are required")
        locations.add((item.block, item.within_block))
        ids.add(item.sequence_id)
        times, samples, mask = (np.asarray(item.sample_times_seconds),
                                np.asarray(item.sensor_samples), np.asarray(item.retained_mask))
        names = tuple(item.channel_order)
        if (times.ndim != 1 or times.dtype.kind not in "iuf" or not len(times)
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or samples.dtype.kind not in "iuf" or samples.shape != (len(times), len(names))
                or not np.isfinite(samples).all() or len(set(names)) != len(names)
                or not set(sensor_space.channel_order).issubset(names)
                or mask.dtype != np.dtype(bool) or mask.shape != times.shape or not mask.any()):
            raise ValueError("Each sequence needs finite aligned EEG, channels, times and one fixed mask")
        if len(times) > 1 and not np.allclose(np.diff(times), .004, rtol=0., atol=1e-10):
            raise ValueError("Prepared continuous samples must retain the fixed 250-Hz grid")
        sample_view = samples.view()
        sample_view.setflags(write=False)
        frozen.append(PhysicalSequence(item.sequence_id, int(item.block), int(item.within_block),
            _readonly(times), sample_view, names, _readonly(mask), item.preparation_input))
    if locations != {(block, index) for block in range(12) for index in range(6)}:
        raise ValueError("The complete twelve-block physical-sequence roster is required")
    return tuple(sorted(frozen, key=lambda item: (item.block, item.within_block)))


def _split(outer, inner=None):
    heldout = BLOCK_PAIRS[outer if inner is None else inner]
    excluded = set(heldout) | (set(BLOCK_PAIRS[outer]) if inner is not None else set())
    return FoldSplit(outer, inner, tuple(block for block in range(12) if block not in excluded), heldout)


def _designs(items, preparation, model_names, check):
    collected = {name: [] for name in model_names}
    temporal_definition = None
    for item in items:
        check()
        view = SequencePredictors(item.sequence_id, item.block, item.within_block,
                                  item.sample_times_seconds, item.preparation_input)
        supplied = preparation.build_designs(view)
        if not isinstance(supplied, Mapping) or set(supplied) != set(model_names):
            raise ValueError("Every selected model must use the same complete sequence roster")
        reference = supplied["nuisance"]
        if not isinstance(reference, SequenceDesign) or reference.terms:
            raise ValueError("Supply the explicit time-expanded nuisance matrix without feature terms")
        for name in model_names:
            design = supplied[name]
            if (not isinstance(design, SequenceDesign) or design.sequence_id != item.sequence_id
                    or not np.array_equal(design.sample_times, item.sample_times_seconds)):
                raise ValueError("All model designs must preserve each physical sequence's sample axis")
            if name != "reduced_task_nuisance" and not np.array_equal(design.nuisance, reference.nuisance):
                raise ValueError("Compared models must share the exact nuisance design")
            if name == "reduced_task_nuisance" and design.terms:
                raise ValueError("The reduced task control contains only its explicit nuisance columns")
            for term in design.schema[1]:
                definition = term[2], term[3], term[4]
                if term[3] != (0., .8):
                    raise ValueError("Primary event designs retain the complete zero-to-800-ms support")
                if temporal_definition is None:
                    temporal_definition = definition
                elif definition != temporal_definition:
                    raise ValueError("All compared models must share the same frozen temporal basis")
            collected[name].append(design)
    return {name: JointSequenceDesign(parts) for name, parts in collected.items()}


def _fit(design, targets, weights, penalties, limits, check):
    """Reuse the original solver; check the caller's budget between contrasts."""
    coefficients, stops, iterations = [], [], []
    for column in range(61):
        check()
        fit = fit_joint_ridge(design, targets[:, column], sample_weights=weights,
            penalties=penalties, atol=limits.atol, btol=limits.btol,
            conlim=limits.conlim, maxiter=limits.maxiter)
        coefficients.append(fit.coefficients)
        stops.extend(fit.stop_codes)
        iterations.extend(fit.iterations)
    check()
    return JointRidgeFit(_readonly(np.column_stack(coefficients)), tuple(stops), tuple(iterations))


def _run_fold(sequences, split, inner_results, sensor_space, model_names,
              prepare_training, limits, retain_fit, check):
    training = tuple(item for item in sequences if item.block in split.training_blocks)
    heldout = tuple(item for item in sequences if item.block in split.heldout_blocks)
    training_ids = tuple(item.sequence_id for item in training)
    check()
    preparation = prepare_training(training, split=split, inner_results=inner_results,
                                   budget_check=check)
    if (not isinstance(preparation, FittedFoldPreparation)
            or tuple(preparation.training_sequence_ids) != training_ids
            or not callable(preparation.build_designs) or not callable(preparation.whitening_choices)
            or set(preparation.penalties) != set(model_names)):
        raise ValueError("Preparation must bind exactly this split's training sequences and model choices")
    sequence_weight = _scalar(preparation.per_sequence_fit_weight, "per-sequence fitting weight")
    train_designs = _designs(training, preparation, model_names, check)
    test_designs = _designs(heldout, preparation, model_names, check)
    penalties = {}
    for name in model_names:
        check()
        if train_designs[name].sequences[0].schema != test_designs[name].sequences[0].schema:
            raise ValueError("Held-out designs must apply the frozen training column definitions")
        raw = np.asarray(preparation.penalties[name])
        if (raw.dtype.kind not in "iuf" or raw.shape != (train_designs[name].shape[1],)
                or not np.isfinite(raw).all() or np.any(raw < 0)):
            raise ValueError("Supply every explicit joint-column penalty before fitting")
        penalties[name] = _readonly(raw)
    targets = np.concatenate([sensor_space.apply(item.sensor_samples, channel_order=item.channel_order)
                              for item in training])
    weights = np.concatenate([np.asarray(item.retained_mask, dtype=float)
                              * (sequence_weight / np.count_nonzero(item.retained_mask))
                              for item in training])
    fits = {"nuisance": _fit(train_designs["nuisance"], targets, weights,
                              penalties["nuisance"], limits, check)}
    retained_training = weights > 0
    residuals = (targets - train_designs["nuisance"] @ fits["nuisance"].coefficients)[retained_training]
    residuals.setflags(write=False)
    weights.setflags(write=False)
    covariance_weights = _readonly(weights[retained_training])
    choices = preparation.whitening_choices(residuals, covariance_weights)
    if not isinstance(choices, WhiteningChoices):
        raise ValueError("Every covariance/shrinkage choice must be explicit and training-only")
    check()
    whitener = fit_residual_whitener(residuals, sample_weights=covariance_weights,
        center_covariance=choices.center_covariance, covariance_denominator=choices.covariance_denominator,
        shrinkage_target=choices.shrinkage_target, shrinkage_strength=choices.shrinkage_strength)
    del residuals
    for name in model_names:
        if name != "nuisance":
            fits[name] = _fit(train_designs[name], targets, weights, penalties[name], limits, check)
    del targets
    errors = {name: [] for name in model_names}
    for index, item in enumerate(heldout):
        check()
        observed = sensor_space.apply(item.sensor_samples, channel_order=item.channel_order)
        for name in model_names:
            check()
            design = JointSequenceDesign([test_designs[name].sequences[index]])
            prediction = design @ fits[name].coefficients
            whitened = whitener.apply(observed - prediction)
            retained = whitened[item.retained_mask]
            with np.errstate(over="raise", invalid="raise"):
                mse = float(np.mean(retained * retained))
            if not np.isfinite(mse):
                raise ValueError("Nonfinite sequence error cannot become a partial fold result")
            errors[name].append(mse)
    result = FoldErrors(split, training_ids, tuple(item.sequence_id for item in heldout),
        tuple(int(np.count_nonzero(item.retained_mask)) for item in heldout),
        MappingProxyType({name: _readonly(value) for name, value in errors.items()}))
    check()
    retain_fit(FittedFold(split, preparation, sensor_space, whitener, MappingProxyType(fits), result))
    check()
    return result


def run_participant_folds(*, participant_id, sequences, frozen_channel_order,
                          phase, prepare_training, solver_limits, retain_fit, budget_check):
    """Run every original nested split; return after all 36 compositions complete.

    `phase='baseline'` exposes no true/false-group models. `phase='fixed_panel'`
    requires the whole registered model panel, including all eight false-group
    models, but does not grant its one-time scientific release. Preparation is
    responsible for the frozen recipe, complete registered penalty search,
    inner selection and matched slot EDF;
    this function neither authenticates nor substitutes those qualifications.
    retain_fit is called once per successful split; failures are never retried.
    A bounded allocation/process must contain the longest individual solver
    call; the supplied budget callback is checked between all 61D solves.
    """
    if (not isinstance(participant_id, str) or not participant_id
            or phase not in ("baseline", "fixed_panel")
            or not all(callable(value) for value in (prepare_training, retain_fit, budget_check))
            or not isinstance(solver_limits, SolverLimits)):
        raise ValueError("Explicit participant, phase, preparation, fit sink and resource checks required")
    for name in ("atol", "btol", "conlim"):
        _scalar(getattr(solver_limits, name), name)
    if (solver_limits.conlim <= 1 or isinstance(solver_limits.maxiter, (bool, np.bool_))
            or not isinstance(solver_limits.maxiter, Integral) or solver_limits.maxiter < 1):
        raise ValueError("Supply explicit finite solver condition and iteration bounds")
    budget_check()
    sensor_space = CommonSensorSpace(frozen_channel_order)
    sequences = _roster(sequences, sensor_space, budget_check)
    names = BASELINE_MODELS if phase == "baseline" else FIXED_PANEL_MODELS
    all_inner, outer = [], []
    for outer_index in range(6):
        inner = tuple(_run_fold(sequences, _split(outer_index, inner_index), (), sensor_space,
                    names, prepare_training, solver_limits, retain_fit, budget_check)
                    for inner_index in range(6) if inner_index != outer_index)
        all_inner.append(inner)
        outer.append(_run_fold(sequences, _split(outer_index), inner, sensor_space,
                     names, prepare_training, solver_limits, retain_fit, budget_check))
    budget_check()
    return ParticipantFolds(participant_id, phase, tuple(all_inner), tuple(outer))
