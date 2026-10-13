"""EP17 fixed linking arithmetic; arrays only, with no source reader or selector.

The caller owns role isolation, image/concept alignment, fitting partitions,
the fixed four-model menu, resource limits and any paired-trial bookkeeping.
This module must never receive calibration/audit arrays for model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

LAYERS = ("layer1", "layer2", "layer3", "layer4")
PCA_RANKS = (32, 64, 128)
RIDGE_PENALTIES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
RELATIVE_SINGULAR_THRESHOLD = 1e-12


class PredictionError(ValueError):
    """Invalid whole input/fit/score; never authorizes row or voxel deletion."""


def require(condition, code):
    if not condition:
        raise PredictionError(code)


def matrix(value, name, minimum_rows=1):
    raw = np.asarray(value)
    require(raw.dtype.kind in "iuf" and raw.ndim == 2, f"{name}_must_be_real_matrix")
    result = np.asarray(raw, dtype=np.float64)
    require(result.shape[0] >= minimum_rows and result.shape[1] > 0, f"{name}_empty_or_too_short")
    require(np.isfinite(result).all(), f"{name}_nonfinite")
    return result


def readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    require(np.isfinite(result).all(), "nonfinite_fitted_state")
    result.setflags(write=False)
    return result


def centered_training(value):
    mean = np.mean(value, axis=0, dtype=np.float64)
    # An exactly constant supplied column is exactly constant mathematically;
    # summing repeated non-binary decimals must not create a residual column.
    constant = np.all(value == value[0], axis=0)
    mean[constant] = value[0, constant]
    centered = value - mean
    require(np.isfinite(mean).all() and np.isfinite(centered).all(), "centering_nonfinite")
    return mean, centered


def valid_rank(rank):
    require(isinstance(rank, Integral) and not isinstance(rank, (bool, np.bool_))
            and rank in PCA_RANKS, "unregistered_PCA_rank")
    return int(rank)


def valid_penalty(penalty):
    require(isinstance(penalty, (int, float, np.integer, np.floating))
            and not isinstance(penalty, (bool, np.bool_)), "unregistered_ridge_penalty")
    value = float(penalty)
    require(math.isfinite(value) and value in RIDGE_PENALTIES, "unregistered_ridge_penalty")
    return value


@dataclass(frozen=True)
class PCAFit:
    feature_mean: np.ndarray
    loadings: np.ndarray
    score_scale: float
    singular_values: np.ndarray
    numerical_rank: int
    training_images: int

    @property
    def rank(self):
        return self.loadings.shape[1]

    def transform(self, features):
        features = matrix(features, "features")
        require(features.shape[1] == self.feature_mean.size, "feature_width_changed")
        scores = ((features - self.feature_mean) @ self.loadings) * self.score_scale
        require(np.isfinite(scores).all(), "transformed_scores_nonfinite")
        return scores


def fit_pca(training_features, rank):
    """Fit only training rows; full deterministic SVD, unwhitened scores."""
    rank = valid_rank(rank)
    features = matrix(training_features, "training_features", minimum_rows=2)
    feature_mean, centered = centered_training(features)
    try:
        # Economy storage of every singular triplet, not a randomized or
        # truncated SVD algorithm. No held-out array enters this operation.
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise PredictionError("PCA_SVD_failed") from error
    require(np.isfinite(singular).all() and np.isfinite(right).all(), "PCA_SVD_nonfinite")
    numerical_rank = int(np.count_nonzero(singular > RELATIVE_SINGULAR_THRESHOLD * singular[0]))
    require(rank <= numerical_rank, "PCA_rank_unavailable")
    require(rank == len(singular) or singular[rank - 1] != singular[rank], "PCA_cutoff_exactly_degenerate")
    loadings = right[:rank].T.copy()
    largest = np.argmax(np.abs(loadings), axis=0)
    signs = np.where(loadings[largest, np.arange(rank)] < 0, -1.0, 1.0)
    loadings *= signs
    scores = centered @ loadings
    energy = float(np.sum(scores * scores, dtype=np.float64))
    require(math.isfinite(energy) and energy > 0, "PCA_score_energy_invalid")
    scale = math.sqrt(features.shape[0] / energy)
    require(math.isfinite(scale) and scale > 0, "PCA_score_scale_invalid")
    return PCAFit(readonly(feature_mean), readonly(loadings), scale, readonly(singular),
                  numerical_rank, features.shape[0])


@dataclass(frozen=True)
class RidgeFit:
    score_mean: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    penalty: float
    training_images: int

    def predict(self, scores):
        scores = matrix(scores, "prediction_scores")
        require(scores.shape[1] == self.score_mean.size, "ridge_feature_width_changed")
        result = (scores - self.score_mean) @ self.coefficients + self.target_mean
        require(np.isfinite(result).all(), "ridge_prediction_nonfinite")
        return result


def fit_ridge(training_scores, training_targets, penalty):
    """Mean squared error + lambda*||B||², with an unpenalized intercept.

    PCA training scores have zero mathematical mean. Centering their supplied
    float64 values again absorbs rounding drift and makes the array API's
    unpenalized-intercept semantics explicit; held-out rows never set a mean.
    """
    penalty = valid_penalty(penalty)
    scores = matrix(training_scores, "training_scores", minimum_rows=2)
    targets = matrix(training_targets, "training_targets", minimum_rows=2)
    require(scores.shape[0] == targets.shape[0], "training_image_count_mismatch")
    score_mean, centered_scores = centered_training(scores)
    target_mean, centered_targets = centered_training(targets)
    n_images = scores.shape[0]
    gram = centered_scores.T @ centered_scores
    gram.flat[::gram.shape[0] + 1] += n_images * penalty
    cross = centered_scores.T @ centered_targets
    require(np.isfinite(gram).all() and np.isfinite(cross).all(), "ridge_system_nonfinite")
    try:
        coefficients = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError as error:
        raise PredictionError("ridge_solve_failed") from error
    return RidgeFit(readonly(score_mean), readonly(target_mean), readonly(coefficients),
                    penalty, n_images)


def hyperparameter_tie_key(layer, rank, penalty):
    """Order exact objective ties only; this is not a hyperparameter selector."""
    require(layer in LAYERS, "unregistered_layer")
    return LAYERS.index(layer), valid_rank(rank), -valid_penalty(penalty)


@dataclass(frozen=True)
class ErrorSums:
    sse: np.ndarray
    sst: np.ndarray
    images: int


def error_sums(observed, predicted, development_training_mean):
    """Per-voxel equal-image sums, preserving the proper baseline for each row.

    The baseline is either one training-mean vector or an (images,voxels)
    matrix of the corresponding training means for out-of-fold predictions.
    A held-out response mean is never estimated here.
    """
    observed = matrix(observed, "observed")
    predicted = matrix(predicted, "predicted")
    require(observed.shape == predicted.shape, "prediction_shape_mismatch")
    raw = np.asarray(development_training_mean)
    require(raw.dtype.kind in "iuf", "training_baseline_must_be_real")
    mean = np.asarray(raw, dtype=np.float64)
    require(mean.shape in ((observed.shape[1],), observed.shape), "training_baseline_shape_mismatch")
    require(np.isfinite(mean).all(), "training_baseline_nonfinite")
    residual = observed - predicted
    baseline_residual = observed - mean
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)
    sst = np.sum(baseline_residual * baseline_residual, axis=0, dtype=np.float64)
    require(np.isfinite(sse).all() and np.isfinite(sst).all(), "score_sums_nonfinite")
    return ErrorSums(readonly(sse), readonly(sst), observed.shape[0])


def pooled_r2(partitions):
    """Ratio of pooled sums, not a mean of fold R²; negative values retained."""
    partitions = tuple(partitions)
    require(bool(partitions), "empty_score_partitions")
    shape = None
    total_sse = total_sst = None
    for item in partitions:
        require(isinstance(item, ErrorSums) and isinstance(item.images, Integral)
                and not isinstance(item.images, (bool, np.bool_)) and item.images > 0,
                "invalid_error_sums")
        sse, sst = np.asarray(item.sse), np.asarray(item.sst)
        require(sse.ndim == 1 and sse.size > 0 and sse.shape == sst.shape
                and sse.dtype.kind in "iuf" and sst.dtype.kind in "iuf", "error_sum_shape_mismatch")
        require(np.isfinite(sse).all() and np.isfinite(sst).all()
                and (sse >= 0).all() and (sst >= 0).all(), "invalid_error_sums")
        if shape is None:
            shape = sse.shape
            total_sse = np.zeros(shape, dtype=np.float64)
            total_sst = np.zeros(shape, dtype=np.float64)
        require(sse.shape == shape, "pooled_voxel_count_mismatch")
        total_sse += sse
        total_sst += sst
    require(np.isfinite(total_sse).all() and np.isfinite(total_sst).all()
            and (total_sst > 0).all(), "pooled_denominator_invalid")
    result = 1.0 - total_sse / total_sst
    require(np.isfinite(result).all(), "R2_nonfinite")
    return result


def linear_cka(features, responses):
    """Centered linear CKA, with original cross-voxel response scales retained."""
    features = matrix(features, "CKA_features", minimum_rows=2)
    responses = matrix(responses, "CKA_responses", minimum_rows=2)
    require(features.shape[0] == responses.shape[0], "CKA_image_count_mismatch")
    _, x = centered_training(features)
    _, y = centered_training(responses)
    cross = x.T @ y
    x_gram, y_gram = x.T @ x, y.T @ y
    numerator = float(np.sum(cross * cross, dtype=np.float64))
    denominator = float(np.linalg.norm(x_gram, ord="fro") * np.linalg.norm(y_gram, ord="fro"))
    require(math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0,
            "CKA_denominator_or_numerator_invalid")
    result = numerator / denominator
    require(math.isfinite(result), "CKA_nonfinite")
    return result
