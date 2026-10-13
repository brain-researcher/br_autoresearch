"""EP17 fixed calibration arithmetic on arrays; no readers, fitting or scoring.

One call represents one participant, ROI and B/C/D stage. The trusted caller
supplies the complete fixed calibration-image roster and original voxel axis.
Invalid columns retain their positions; no image or voxel is deleted here.
Source/role identity and cross-stage intersection remain the caller's job.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MINIMUM_CALIBRATION_IMAGES = 120
RELATIVE_VARIANCE_THRESHOLD = 1e-12
PEARSON_BOUNDARY_TOLERANCE = 1e-12
FISHER_BOUNDARY_DISTANCE = 1e-7
CEILING_FLOOR = 0.10


class CalibrationError(ValueError):
    """A whole-input identity/shape/order error, never permission to drop rows."""


def require(condition, code):
    if not condition:
        raise CalibrationError(code)


def matrix(value, name, minimum_rows=2):
    value = np.asarray(value)
    require(value.dtype.kind in "iuf" and value.ndim == 2, f"{name}_must_be_real_matrix")
    require(value.shape[0] >= minimum_rows and value.shape[1] > 0, f"{name}_too_short_or_empty")
    return np.asarray(value, dtype=np.float64)


def immutable(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def center(values):
    """Centered accumulation, with an exact branch for constant input columns."""
    mean = np.mean(values, axis=0, dtype=np.float64)
    constant = np.all(values == values[0], axis=0)
    mean[constant] = values[0, constant]
    return mean, values - mean


@dataclass(frozen=True)
class VarianceQualification:
    finite_inputs: np.ndarray
    finite_moments: np.ndarray
    positive_variance: np.ndarray
    above_relative_threshold: np.ndarray
    valid: np.ndarray
    population_variance: np.ndarray
    sample_variance: np.ndarray
    mean_square: np.ndarray


def qualify_variance(image_means):
    """Qualify complete per-voxel image vectors, with no pairwise deletion.

    The public calibration API applies this to all calibration image means,
    and separately to the two half-mean vectors. This helper also supports
    later, explicitly scoped fold/block checks without changing its rule.
    """
    values = matrix(image_means, "image_means")
    n_images, width = values.shape
    finite = np.isfinite(values).all(axis=0)
    pop = np.full(width, np.nan)
    sample = np.full(width, np.nan)
    square = np.full(width, np.nan)
    moments = np.zeros(width, dtype=bool)
    positive = np.zeros(width, dtype=bool)
    above = np.zeros(width, dtype=bool)
    indices = np.flatnonzero(finite)
    if len(indices):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            selected = values[:, indices]
            _, centered = center(selected)
            sums = np.sum(centered * centered, axis=0, dtype=np.float64)
            variance = sums / n_images
            unbiased = sums / (n_images - 1)
            mean_square = np.mean(selected * selected, axis=0, dtype=np.float64)
        moments[indices] = np.isfinite(variance) & np.isfinite(unbiased) & np.isfinite(mean_square)
        positive[indices] = variance > 0
        above[indices] = variance > RELATIVE_VARIANCE_THRESHOLD * mean_square
        eligible = moments[indices] & positive[indices] & above[indices]
        accepted = indices[eligible]
        pop[accepted], sample[accepted], square[accepted] = variance[eligible], unbiased[eligible], mean_square[eligible]
    valid = finite & moments & positive & above
    return VarianceQualification(*(immutable(array) for array in
        (finite, moments, positive, above, valid, pop, sample, square)))


@dataclass(frozen=True)
class RepeatSummary:
    repeat_counts: np.ndarray
    half_counts: np.ndarray
    finite_input_voxels: np.ndarray
    finite_summary_voxels: np.ndarray
    image_means: np.ndarray
    within_image_sample_variances: np.ndarray
    half_1_means: np.ndarray
    half_2_means: np.ndarray


def summarize_repeats(repeated_responses, retained_acquisition_order):
    """Equal-repeat summaries, preserving every input image and voxel.

    Both arguments are ordered sequences with one array per fixed image.
    Response arrays are (retained repetitions, original voxels). Order arrays
    are unique nonnegative integer metadata keys within each image; gaps are
    allowed. Halves use sorted retained position, never source-ordinal parity.
    No source row is generated, omitted or duplicated to balance the halves.
    """
    responses, orders = tuple(repeated_responses), tuple(retained_acquisition_order)
    require(len(responses) == len(orders) and len(responses) >= MINIMUM_CALIBRATION_IMAGES,
            "complete_calibration_roster_requires_at_least_120_images")
    width, finite = None, None
    matrices, permutations, counts = [], [], []
    for response, order in zip(responses, orders):
        response = matrix(response, "retained_responses")
        if width is None:
            width = response.shape[1]
            finite = np.ones(width, dtype=bool)
        require(response.shape[1] == width, "original_voxel_axis_width_changed")
        order = np.asarray(order)
        require(order.dtype.kind in "iu" and order.ndim == 1
                and order.size == response.shape[0] and np.all(order >= 0)
                and np.unique(order).size == order.size, "retained_acquisition_order_invalid")
        finite &= np.isfinite(response).all(axis=0)
        matrices.append(response)
        permutations.append(np.argsort(order, kind="stable"))
        counts.append(response.shape[0])
    counts = np.asarray(counts, dtype=np.int64)
    half_counts = np.column_stack(((counts + 1) // 2, counts // 2))
    shape = len(responses), width
    means, within, first, second = (np.full(shape, np.nan) for _ in range(4))
    # A nonfinite value invalidates that entire original column across the
    # complete image roster. Never recover a statistic from its finite rows.
    indices = np.flatnonzero(finite)
    for index, (response, permutation) in enumerate(zip(matrices, permutations)):
        if not len(indices):
            break
        values = response[permutation][:, indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean, centered = center(values)
            variance = np.sum(centered * centered, axis=0, dtype=np.float64) / (len(values) - 1)
            half_1, _ = center(values[::2])
            half_2, _ = center(values[1::2])
        means[index, indices], within[index, indices] = mean, variance
        first[index, indices], second[index, indices] = half_1, half_2
    summaries_finite = finite & np.isfinite(means).all(axis=0) & np.isfinite(within).all(axis=0)
    summaries_finite &= np.isfinite(first).all(axis=0) & np.isfinite(second).all(axis=0)
    for output in (means, within, first, second):
        output[:, ~summaries_finite] = np.nan
    return RepeatSummary(*(immutable(array) for array in
        (counts, half_counts, finite, summaries_finite, means, within, first, second)))


@dataclass(frozen=True)
class FisherResult:
    valid: np.ndarray
    pearson_r: np.ndarray
    fisher_z: np.ndarray
    roundoff_to_boundary: np.ndarray
    fisher_boundary_adjusted: np.ndarray


def fisher_transform(correlations):
    """Keep Pearson for ranking; separately apply the fixed Fisher boundary."""
    correlations = np.asarray(correlations)
    require(correlations.dtype.kind in "iuf" and correlations.ndim == 1 and correlations.size > 0,
            "correlations_must_be_real_voxel_vector")
    raw = np.asarray(correlations, dtype=np.float64)
    valid = np.isfinite(raw) & (np.abs(raw) <= 1 + PEARSON_BOUNDARY_TOLERANCE)
    r, z = np.full(raw.shape, np.nan), np.full(raw.shape, np.nan)
    roundoff = np.zeros(raw.shape, dtype=bool)
    adjusted = np.zeros(raw.shape, dtype=bool)
    r[valid] = np.clip(raw[valid], -1., 1.)
    roundoff[valid] = r[valid] != raw[valid]
    for_z = np.clip(r[valid], -1 + FISHER_BOUNDARY_DISTANCE, 1 - FISHER_BOUNDARY_DISTANCE)
    adjusted[valid] = for_z != r[valid]
    z[valid] = np.arctanh(for_z)
    return FisherResult(*(immutable(array) for array in (valid, r, z, roundoff, adjusted)))


@dataclass(frozen=True)
class ReliabilityResult:
    half_1_variance: VarianceQualification
    half_2_variance: VarianceQualification
    transformed: FisherResult


def repeat_reliability(half_1_means, half_2_means):
    first = matrix(half_1_means, "half_1_means")
    second = matrix(half_2_means, "half_2_means")
    require(first.shape == second.shape and first.shape[0] >= MINIMUM_CALIBRATION_IMAGES,
            "complete_equal_half_image_roster_required")
    qualification_1, qualification_2 = qualify_variance(first), qualify_variance(second)
    indices = np.flatnonzero(qualification_1.valid & qualification_2.valid)
    r = np.full(first.shape[1], np.nan)
    if len(indices):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _, a = center(first[:, indices])
            _, b = center(second[:, indices])
            aa = np.sum(a * a, axis=0, dtype=np.float64)
            bb = np.sum(b * b, axis=0, dtype=np.float64)
            ab = np.sum(a * b, axis=0, dtype=np.float64)
            r[indices] = (ab / np.sqrt(aa)) / np.sqrt(bb)
    return ReliabilityResult(qualification_1, qualification_2, fisher_transform(r))


@dataclass(frozen=True)
class CalibrationResult:
    image_count: int
    repeat_counts: np.ndarray
    half_counts: np.ndarray
    finite_input_voxels: np.ndarray
    image_mean_variance: VarianceQualification
    response_eligible: np.ndarray
    reliability_valid: np.ndarray
    pearson_r: np.ndarray
    fisher_z: np.ndarray
    fisher_boundary_adjusted: np.ndarray
    pearson_roundoff_to_boundary: np.ndarray
    half_1_variance_valid: np.ndarray
    half_2_variance_valid: np.ndarray
    ceiling_valid: np.ndarray
    mean_sampling_noise_variance: np.ndarray
    noise_ceiling: np.ndarray
    ceiling_denominator: np.ndarray

    @property
    def fisher_boundary_adjusted_count(self):
        return int(np.count_nonzero(self.fisher_boundary_adjusted))


def calibrate(repeated_responses, retained_acquisition_order):
    """All fixed images, one participant/stage; maps retain the original axis.

    response_eligible supplies that stage's finite/nondegenerate calibration
    qualification. The later builder intersects B/C/D by original column ID.
    A half-degenerate column can have a valid response/ceiling but invalid
    Pearson: the reliable-support selector must use reliability_valid too.
    """
    summary = summarize_repeats(repeated_responses, retained_acquisition_order)
    variance = qualify_variance(summary.image_means)
    eligible = summary.finite_input_voxels & summary.finite_summary_voxels & variance.valid
    reliability = repeat_reliability(summary.half_1_means, summary.half_2_means)
    transformed = reliability.transformed
    r_valid = eligible & transformed.valid
    width = len(eligible)
    r, z, noise, ceiling, denominator = (np.full(width, np.nan) for _ in range(5))
    r[r_valid], z[r_valid] = transformed.pearson_r[r_valid], transformed.fisher_z[r_valid]
    indices = np.flatnonzero(eligible)
    ceiling_valid = np.zeros(width, dtype=bool)
    if len(indices):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # Every image gets weight1/M, regardless of its number of repeats.
            estimate = np.mean(summary.within_image_sample_variances[:, indices]
                               / summary.repeat_counts[:, None], axis=0, dtype=np.float64)
            values = 1. - estimate / variance.sample_variance[indices]
        good = np.isfinite(estimate) & (estimate >= 0) & np.isfinite(values)
        accepted = indices[good]
        noise[accepted], ceiling[accepted] = estimate[good], values[good]
        denominator[accepted] = np.maximum(values[good], CEILING_FLOOR)
        ceiling_valid[accepted] = True
    adjusted = transformed.fisher_boundary_adjusted & r_valid
    rounded = transformed.roundoff_to_boundary & r_valid
    return CalibrationResult(len(summary.repeat_counts), summary.repeat_counts, summary.half_counts,
        summary.finite_input_voxels, variance, immutable(eligible), immutable(r_valid),
        immutable(r), immutable(z), immutable(adjusted), immutable(rounded),
        reliability.half_1_variance.valid, reliability.half_2_variance.valid,
        immutable(ceiling_valid), immutable(noise), immutable(ceiling), immutable(denominator))
