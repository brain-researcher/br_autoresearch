"""Separately admitted, exact-constant adaptation of the EP21 brain comparator.

This pure library has no reader, admission, scheduling or publication authority.
It reuses the original row/fold binding, float32 penalized primal solver,
float64 augmented prediction and image-weighted alpha selection. Only a zero
sample SD with exactly constant finite fitting values AND an exactly matching
original float32 mean receives scale one. Positive SDs are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

import brain_loss as original

NORMALIZATION_POLICY = "float32_ddof1_exact_constant_exact_mean_zero_sd_effective_one"
BrainLossNotEvaluable = original.BrainLossNotEvaluable


@dataclass(frozen=True)
class NormalizationState:
    mean: np.ndarray
    raw_sd: np.ndarray
    effective_scale: np.ndarray
    replacement_mask: np.ndarray
    fitting_images: int

    @property
    def replacement_count(self) -> int:
        return int(np.count_nonzero(self.replacement_mask))


@dataclass(frozen=True)
class ConstantBrainLossFit:
    # Deliberately not a BrainLossFit subclass: original publication rejects it.
    summary: original.BrainPairSummary
    image_ids: tuple[str, ...]
    source_native_voxel_ids: tuple[int, ...]
    target_native_voxel_ids: tuple[int, ...]
    matrix: np.ndarray
    fold_source_normalization: tuple[NormalizationState, ...]
    fold_target_normalization: tuple[NormalizationState, ...]
    source_normalization: NormalizationState
    target_normalization: NormalizationState


def normalization_state(values, label) -> NormalizationState:
    """Compute original moments, admitting only the adopted exact zero case.

    The caller supplies fitting rows in their original sorted image order.
    Neither held-out values nor another fold's replacement mask enters here.
    """
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32")
            or values.ndim != 2 or len(values) < 2 or values.shape[1] < 1):
        raise ValueError("float32_fitting_rows_on_full_native_axis_required")
    if not np.isfinite(values).all():
        raise BrainLossNotEvaluable("nonfinite_required_training_response:" + label)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(values, axis=0)[None, :]
        raw_sd = np.std(values, axis=0, ddof=1)[None, :]
    if (not np.isfinite(mean).all() or not np.isfinite(raw_sd).all()
            or np.any(raw_sd < 0)):
        raise BrainLossNotEvaluable("nonfinite_or_negative_required_training_scale:" + label)
    replacement = raw_sd == 0
    if np.any(replacement):
        columns = np.flatnonzero(replacement[0])
        # Compare stored float32 values exactly; no tolerance or substituted mean.
        first = values[0, columns]
        if not np.all(values[:, columns] == first[None, :]):
            raise BrainLossNotEvaluable("zero_sd_nonconstant_required_training_column:" + label)
        if not np.all(mean[0, columns] == first):
            raise BrainLossNotEvaluable("zero_sd_constant_mean_mismatch:" + label)
    effective = raw_sd.copy()
    effective[replacement] = np.float32(1)
    return NormalizationState(*(original._readonly(value) for value in
        (mean, raw_sd, effective, replacement)), fitting_images=len(values))


def _normalize(values, state, label):
    return original._normalize(values, state.mean, state.effective_scale, label)


def fit_brain_pair(source, target, folds, *, source_native_ids, target_native_ids,
                   check: Callable[[], None]) -> ConstantBrainLossFit:
    """Complete all 50 fixed CV fits and the selected final refit, or fail.

    All target coordinates remain in the per-image MSE denominator. Guarded
    coordinates have response-unit residuals, not unit-variance residuals.
    """
    if not callable(check):
        raise ValueError("caller_allocation_deadline_check_required")
    check()
    original._validate_folds(folds)
    if (not isinstance(source, original.TauTrainResponses)
            or not isinstance(target, original.TauTrainResponses)
            or (source.participant, target.participant) not in original.PAIR_ORDER):
        raise ValueError("one_fixed_directed_nonself_pair_required")
    x, source_ids = original._aligned_values(source, folds, source_native_ids)
    y, target_ids = original._aligned_values(target, folds, target_native_ids)
    scores = np.empty((len(original.ALPHAS), 5), dtype=np.float64)
    source_states, target_states = [], []
    for fold, validation_indices in enumerate(folds.validation_indices):
        check()
        validation = np.asarray(validation_indices)
        training_mask = np.ones(original.TRAINING_IMAGES, dtype=bool)
        training_mask[validation] = False
        x_fit, y_fit = x[training_mask], y[training_mask]
        x_state = normalization_state(x_fit, f"source_fold_{fold}")
        y_state = normalization_state(y_fit, f"target_fold_{fold}")
        source_states.append(x_state)
        target_states.append(y_state)
        x_fit = _normalize(x_fit, x_state, "source_fit")
        y_fit = _normalize(y_fit, y_state, "target_fit")
        x_validation = _normalize(x[validation], x_state, "source_validation")
        y_validation = _normalize(y[validation], y_state, "target_validation")
        gram, right = original._primal_system(x_fit, y_fit, check)
        for index, alpha in enumerate(original.ALPHAS):
            matrix = original._solve_primal(gram, right, alpha, check)
            check()
            prediction = original._predict_standardized(x_validation, matrix)
            with np.errstate(over="ignore", invalid="ignore"):
                per_image = np.mean((prediction - y_validation) ** 2, axis=1)
                score = np.mean(per_image)
            if not np.isfinite(per_image).all() or not np.isfinite(score):
                raise BrainLossNotEvaluable(f"nonfinite_required_validation_MSE:fold_{fold}:alpha_{alpha}")
            scores[index, fold] = score
            del matrix, prediction, per_image
        del gram, right, x_fit, y_fit, x_validation, y_validation
    selected, aggregate = original._choose_alpha(scores)
    check()
    x_state = normalization_state(x, "source_final")
    y_state = normalization_state(y, "target_final")
    gram, right = original._primal_system(_normalize(x, x_state, "source_final"),
                                        _normalize(y, y_state, "target_final"), check)
    matrix = original._solve_primal(gram, right, selected, check)
    check()
    summary = original.BrainPairSummary(source.participant, target.participant, True,
        "all_50_CV_fits_and_selected_final_refit_complete", selected, aggregate,
        tuple(tuple(float(value) for value in row) for row in scores), completed_fits=51)
    return ConstantBrainLossFit(summary, folds.image_ids, source_ids, target_ids,
        original._readonly(matrix), tuple(source_states), tuple(target_states), x_state, y_state)


def _physical_prediction(values, matrix, source_mean, source_scale, target_mean,
                         target_scale, check):
    check()
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32")
            or values.ndim != 2 or values.shape[1] != source_mean.shape[1]):
        raise ValueError("float32_source_responses_on_complete_native_axis_required")
    normalized = original._normalize(values, source_mean, source_scale, "physical_prediction")
    predicted = original._predict_standardized(normalized, matrix)
    check()
    with np.errstate(over="ignore", invalid="ignore"):
        physical = predicted * target_scale + target_mean
    if physical.dtype != np.float64 or not np.isfinite(physical).all():
        raise BrainLossNotEvaluable("nonfinite_physical_target_prediction")
    return physical


def predict_physical(fit, source_values, *, source_native_ids, check):
    """Restore full target-native physical units before any decoder normalizer.

    This pure operation grants no test-data access and changes no decoder axis.
    Float64 prediction/restoration is retained, as in the original readout.
    """
    if not isinstance(fit, ConstantBrainLossFit) or not callable(check):
        raise ValueError("constant_comparator_fit_and_caller_check_required")
    if original._native_ids(source_native_ids) != fit.source_native_voxel_ids:
        raise ValueError("exact_fixed_native_voxel_order_required")
    return _physical_prediction(source_values, fit.matrix,
        fit.source_normalization.mean, fit.source_normalization.effective_scale,
        fit.target_normalization.mean, fit.target_normalization.effective_scale, check)


def fit_brain_panel(responses: Mapping[str, original.TauTrainResponses], *,
                    authenticated_tau_train_image_ids, expected_native_axes,
                    check, on_pair_completed):
    """Visit all 20 fixed pairs, retaining unavailable dispositions explicitly."""
    if not callable(check) or not callable(on_pair_completed):
        raise ValueError("caller_check_and_completed_pair_consumer_required")
    check()
    if (set(responses) != set(original.PARTICIPANTS)
            or set(expected_native_axes) != set(original.PARTICIPANTS)):
        raise ValueError("all_five_fixed_participants_and_native_axes_required")
    folds = original.make_common_folds(authenticated_tau_train_image_ids)
    for participant in original.PARTICIPANTS:
        if (not isinstance(responses[participant], original.TauTrainResponses)
                or responses[participant].participant != participant):
            raise ValueError("response_participant_key_binding_required")
        original._row_order(responses[participant], folds, expected_native_axes[participant])
    summaries = []
    for source, target in original.PAIR_ORDER:
        check()
        try:
            result = fit_brain_pair(responses[source], responses[target], folds,
                source_native_ids=expected_native_axes[source],
                target_native_ids=expected_native_axes[target], check=check)
        except BrainLossNotEvaluable as error:
            summaries.append(original.BrainPairSummary(source, target, False, str(error)))
            continue
        on_pair_completed(result)
        summaries.append(result.summary)
        del result
    check()
    return tuple(summaries)
