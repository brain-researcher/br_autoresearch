"""UNACTIVATED fixed EP21 brain-loss CV library over admitted image averages.

No file I/O, empirical reader, authorization, CLI, scheduling or publication.
The future caller must admit tau-train access, independently image-average each
person after role assignment, bind the complete fixed native axes and provide
an allocation/deadline check. These typed arrays do not establish permission.

Source: pinned V1.0.0 NCC_brain_loss/ncc_train.py132-169 and FastL2LiR0.9
__sub_fit's no-feature-selection branch. Fit uses float32 dense primal ridge
and a penalized appended intercept. Pinned ncc_test.py105-109 appends default
float64 ones for prediction; CV preserves that promotion. No dual switch,
voxel selection, zero-scale deletion, grid repair or automatic retry occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Callable, Mapping

import numpy as np

from schedule import PAIR_ORDER, PARTICIPANTS

TRAINING_IMAGES = 897
ALPHAS = (620, 640, 660, 740, 780, 800, 820, 1000, 1200, 1400)
FOLD_SIZES = (180, 180, 179, 179, 179)


class BrainLossNotEvaluable(RuntimeError):
    """A required numerical cell failed; retain its pair, never a smaller fit."""


@dataclass(frozen=True)
class TauTrainResponses:
    """Caller-admitted separately averaged float32 rows, one per image.

    Native IDs retain their external fixed order; they are not the content
    decoder's subject-unique eligible-axis mapping. Values must remain unchanged
    throughout a call. This container does not verify raw repetition averaging.
    """
    participant: str
    image_ids: tuple[str, ...]
    native_voxel_ids: tuple[int, ...]
    values: np.ndarray
    role: str = "tau_train"
    aggregation: str = "separately_image_averaged_after_role_assignment"


@dataclass(frozen=True)
class CommonImageFolds:
    image_ids: tuple[str, ...]
    validation_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BrainPairSummary:
    source: str
    target: str
    available: bool
    reason: str
    selected_alpha: int | None = None
    validation_mse: tuple[float, ...] = ()
    fold_mse_by_alpha: tuple[tuple[float, ...], ...] = ()
    validation_image_counts: tuple[int, ...] = FOLD_SIZES
    completed_fits: int | None = None


@dataclass(frozen=True)
class BrainLossFit:
    summary: BrainPairSummary
    image_ids: tuple[str, ...]
    source_native_voxel_ids: tuple[int, ...]
    target_native_voxel_ids: tuple[int, ...]
    matrix: np.ndarray  # [source native voxel + penalized bias, target native voxel]
    source_mean: np.ndarray  # [1, source native voxel]
    source_std: np.ndarray
    target_mean: np.ndarray  # [1, target native voxel]
    target_std: np.ndarray


def _image_ids(values):
    if isinstance(values, str):
        raise ValueError("897_distinct_image_IDs_required")
    ids = tuple(values)
    if (len(ids) != TRAINING_IMAGES or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != TRAINING_IMAGES):
        raise ValueError("897_distinct_image_IDs_required")
    return ids


def _native_ids(values):
    ids = tuple(values)
    if (not ids or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                       or value < 0 for value in ids) or len(set(ids)) != len(ids)):
        raise ValueError("complete_unique_integer_native_axis_required")
    return tuple(int(value) for value in ids)


def make_common_folds(authenticated_tau_train_image_ids) -> CommonImageFolds:
    """Bytewise-sort IDs, RandomState42 permutation, then position modulo five.

    Preserve sorted order inside each fitting/validation subset, as in the
    source's label sorting. The RNG changes membership only and is local.
    """
    ids = tuple(sorted(_image_ids(authenticated_tau_train_image_ids), key=lambda value: value.encode("utf-8")))
    permutation = np.random.RandomState(42).permutation(TRAINING_IMAGES)
    membership = np.empty(TRAINING_IMAGES, dtype=np.int64)
    membership[permutation] = np.arange(TRAINING_IMAGES) % 5
    folds = tuple(tuple(int(index) for index in np.flatnonzero(membership == fold)) for fold in range(5))
    if tuple(map(len, folds)) != FOLD_SIZES:
        raise ValueError("fixed_five_fold_image_support_required")
    return CommonImageFolds(ids, folds)


def _validate_folds(folds):
    if not isinstance(folds, CommonImageFolds) or folds != make_common_folds(folds.image_ids):
        raise ValueError("exact_common_tau_train_fold_assignment_required")


def _row_order(response, folds, expected_native_ids):
    if (not isinstance(response, TauTrainResponses) or response.participant not in PARTICIPANTS
            or response.role != "tau_train"
            or response.aggregation != "separately_image_averaged_after_role_assignment"):
        raise ValueError("admitted_separately_averaged_tau_train_responses_required")
    image_ids = _image_ids(response.image_ids)
    if set(image_ids) != set(folds.image_ids):
        raise ValueError("exact_authenticated_tau_train_image_identity_required")
    native_ids = _native_ids(response.native_voxel_ids)
    if native_ids != _native_ids(expected_native_ids):
        raise ValueError("exact_fixed_native_voxel_order_required")
    positions = {image_id: index for index, image_id in enumerate(image_ids)}
    return tuple(positions[image_id] for image_id in folds.image_ids), native_ids


def _aligned_values(response, folds, expected_native_ids):
    order, native_ids = _row_order(response, folds, expected_native_ids)
    values = response.values
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32")
            or values.shape != (TRAINING_IMAGES, len(native_ids))):
        raise ValueError("float32_image_averages_on_complete_fixed_native_axis_required")
    if not np.isfinite(values).all():
        raise BrainLossNotEvaluable("nonfinite_required_tau_train_response:" + response.participant)
    return values[np.asarray(order)], native_ids


def _moments(values, label):
    # Preserve the source's NumPy float32 means/std arithmetic, with no epsilon.
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(values, axis=0)[None, :]
        std = np.std(values, axis=0, ddof=1)[None, :]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise BrainLossNotEvaluable("zero_or_nonfinite_required_training_scale:" + label)
    return mean, std


def _normalize(values, mean, std, label):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (values - mean) / std
    if normalized.dtype != np.dtype("float32") or not np.isfinite(normalized).all():
        raise BrainLossNotEvaluable("nonfinite_required_standardized_response:" + label)
    return normalized


def _primal_system(source_normalized, target_normalized, check):
    check()
    # Exact FastL2LiR0.9 no-feature-selection layout. Reuse these products across
    # the ten alphas of one fold; never reuse them across different image folds.
    augmented = np.hstack((source_normalized, np.ones((len(source_normalized), 1), dtype=np.float32)))
    gram = np.matmul(augmented.T, augmented)
    right = np.matmul(augmented.T, target_normalized)
    if gram.dtype != np.float32 or right.dtype != np.float32 or not np.isfinite(gram).all() or not np.isfinite(right).all():
        raise BrainLossNotEvaluable("nonfinite_float32_primal_system")
    return gram, right


def _solve_primal(gram, right, alpha, check):
    if type(alpha) is not int or alpha not in ALPHAS:
        raise ValueError("fixed_brain_loss_alpha_required")
    check()  # The caller's allocation/deadline failure must propagate unchanged.
    try:
        matrix = np.linalg.solve(gram + alpha * np.eye(gram.shape[0], dtype=np.float32), right)
    except np.linalg.LinAlgError as error:
        raise BrainLossNotEvaluable("required_float32_primal_solve_failed") from error
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise BrainLossNotEvaluable("nonfinite_float32_brain_mapping")
    return matrix


def _predict_standardized(source_normalized, matrix):
    # Deliberately no dtype on ones: pinned ncc_test.py promotes this augmented
    # prediction matrix to float64 while keeping the trained matrix float32.
    augmented = np.hstack((source_normalized, np.ones((len(source_normalized), 1))))
    prediction = np.matmul(augmented, matrix)
    if prediction.dtype != np.float64 or not np.isfinite(prediction).all():
        raise BrainLossNotEvaluable("nonfinite_brain_validation_prediction")
    return prediction


def _choose_alpha(fold_mse):
    if fold_mse.shape != (len(ALPHAS), 5) or not np.isfinite(fold_mse).all() or np.any(fold_mse < 0):
        raise BrainLossNotEvaluable("incomplete_or_nonfinite_required_CV_scores")
    scores = np.sum(fold_mse * np.asarray(FOLD_SIZES, dtype=np.float64)[None, :], axis=1) / TRAINING_IMAGES
    if not np.isfinite(scores).all():
        raise BrainLossNotEvaluable("nonfinite_image_weighted_CV_score")
    chosen = min(range(len(ALPHAS)), key=lambda index: (float(scores[index]), ALPHAS[index]))
    return ALPHAS[chosen], tuple(float(value) for value in scores)


def _readonly(value):
    value.setflags(write=False)
    return value


def fit_brain_pair(source: TauTrainResponses, target: TauTrainResponses, folds: CommonImageFolds, *,
                   source_native_ids, target_native_ids, check: Callable[[], None]) -> BrainLossFit:
    """Fifty fixed CV fits and one final refit, all on exactly897 tau-train IDs.

    Expected native axes and the folds must originate in caller-admitted frozen
    metadata. This function can enforce equality, not authenticate those facts.
    A scientific failure returns no partial selected model and removes no rows
    or columns. Budget, cancellation and arbitrary programming errors propagate.
    """
    if not callable(check):
        raise ValueError("caller_allocation_deadline_check_required")
    check()
    _validate_folds(folds)
    if (not isinstance(source, TauTrainResponses) or not isinstance(target, TauTrainResponses)
            or (source.participant, target.participant) not in PAIR_ORDER):
        raise ValueError("one_fixed_directed_nonself_pair_required")
    x, source_ids = _aligned_values(source, folds, source_native_ids)
    y, target_ids = _aligned_values(target, folds, target_native_ids)
    scores = np.empty((len(ALPHAS), 5), dtype=np.float64)
    for fold, validation_indices in enumerate(folds.validation_indices):
        check()
        validation = np.asarray(validation_indices)
        training_mask = np.ones(TRAINING_IMAGES, dtype=bool)
        training_mask[validation] = False
        x_fit, y_fit = x[training_mask], y[training_mask]
        x_mean, x_std = _moments(x_fit, f"source_fold_{fold}")
        y_mean, y_std = _moments(y_fit, f"target_fold_{fold}")
        x_fit = _normalize(x_fit, x_mean, x_std, "source_fit")
        y_fit = _normalize(y_fit, y_mean, y_std, "target_fit")
        x_validation = _normalize(x[validation], x_mean, x_std, "source_validation")
        y_validation = _normalize(y[validation], y_mean, y_std, "target_validation")
        gram, right = _primal_system(x_fit, y_fit, check)
        for index, alpha in enumerate(ALPHAS):
            matrix = _solve_primal(gram, right, alpha, check)
            check()
            prediction = _predict_standardized(x_validation, matrix)
            with np.errstate(over="ignore", invalid="ignore"):
                per_image = np.mean((prediction - y_validation) ** 2, axis=1)
                score = np.mean(per_image)
            if not np.isfinite(per_image).all() or not np.isfinite(score):
                raise BrainLossNotEvaluable(f"nonfinite_required_validation_MSE:fold_{fold}:alpha_{alpha}")
            scores[index, fold] = score
            del matrix, prediction, per_image
        del gram, right, x_fit, y_fit, x_validation, y_validation
    selected, aggregate = _choose_alpha(scores)
    check()
    x_mean, x_std = _moments(x, "source_final")
    y_mean, y_std = _moments(y, "target_final")
    gram, right = _primal_system(_normalize(x, x_mean, x_std, "source_final"),
                                  _normalize(y, y_mean, y_std, "target_final"), check)
    matrix = _solve_primal(gram, right, selected, check)
    check()
    summary = BrainPairSummary(source.participant, target.participant, True,
        "all_50_CV_fits_and_selected_final_refit_complete", selected, aggregate,
        tuple(tuple(float(value) for value in row) for row in scores), completed_fits=51)
    return BrainLossFit(summary, folds.image_ids, source_ids, target_ids, _readonly(matrix),
                        _readonly(x_mean), _readonly(x_std), _readonly(y_mean), _readonly(y_std))


def fit_brain_panel(responses: Mapping[str, TauTrainResponses], *, authenticated_tau_train_image_ids,
                    expected_native_axes: Mapping[str, tuple[int, ...]], check: Callable[[], None],
                    on_pair_completed: Callable[[BrainLossFit], None]) -> tuple[BrainPairSummary, ...]:
    """Visit all20 pairs once; retain summaries and stream each successful model.

    The callback owns any later admitted publication and must not retain all
    matrices accidentally. It is outside the scientific-failure catch. No
    complete panel is returned after a budget/callback/programming failure.
    Required scientifically unavailable pairs remain explicit in the20 rows;
    available-subset means or a complete quantitative finding are not produced.
    """
    if not callable(check) or not callable(on_pair_completed):
        raise ValueError("caller_check_and_completed_pair_consumer_required")
    check()
    if set(responses) != set(PARTICIPANTS) or set(expected_native_axes) != set(PARTICIPANTS):
        raise ValueError("all_five_fixed_participants_and_native_axes_required")
    folds = make_common_folds(authenticated_tau_train_image_ids)
    for participant in PARTICIPANTS:
        if not isinstance(responses[participant], TauTrainResponses) or responses[participant].participant != participant:
            raise ValueError("response_participant_key_binding_required")
        _row_order(responses[participant], folds, expected_native_axes[participant])
    summaries = []
    for source, target in PAIR_ORDER:
        check()
        try:
            result = fit_brain_pair(responses[source], responses[target], folds,
                source_native_ids=expected_native_axes[source], target_native_ids=expected_native_axes[target], check=check)
        except BrainLossNotEvaluable as error:
            summaries.append(BrainPairSummary(source, target, False, str(error)))
            continue
        on_pair_completed(result)
        summaries.append(result.summary)
        del result
    check()
    return tuple(summaries)
