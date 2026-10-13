"""EP17 fixed measurement-edge arithmetic; arrays only, never an outcome reader.

The caller supplies correctly aligned participant/ROI arrays from qualified
role-filtered handoffs. These functions do not establish reader isolation,
choose a candidate, or perform the one-shot audit. A support or ceiling edge
accepts one common voxel-advantage bank, so it cannot request endpoint refits.
"""
from dataclasses import dataclass

import numpy as np

SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-06")
ROIS = ("left", "right")
GAIN_VARIANCE_FLOOR = 1e-8
SVD_RELATIVE_TOLERANCE = 1e-12


class MeasurementError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MeasurementError(message)


def vector(value, name):
    raw = np.asarray(value)
    require(raw.ndim == 1 and raw.size > 0 and raw.dtype.kind in "iuf", name + "_shape")
    result = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(result).all(), name + "_nonfinite")
    return result


def fixed_array(value):
    result = np.array(value, dtype=np.float64, copy=True)
    require(np.isfinite(result).all(), "derived_array_nonfinite")
    result.setflags(write=False)
    return result


def mean_weights(value, size, name):
    result = vector(value, name)
    require(result.size == size and (result >= 0).all()
            and np.isclose(result.sum(), 1.0, rtol=0, atol=1e-12), name + "_not_mean_weights")
    return result


def equal_participant_roi_mean(values):
    """Input is already a within-ROI scalar, never pooled native voxel rows."""
    require(set(values) == {(s, r) for s in SUBJECTS for r in ROIS}, "eight_participant_ROIs_required")
    rows = np.asarray([[values[s, r] for r in ROIS] for s in SUBJECTS], dtype=np.float64)
    require(rows.shape == (4, 2) and np.isfinite(rows).all(), "ROI_values_invalid")
    value = float(np.mean(np.mean(rows, axis=1)))
    require(np.isfinite(value), "ROI_mean_nonfinite")
    return value


@dataclass(frozen=True)
class LinearEdge:
    start: float
    end: float
    change: float
    operator_weights: np.ndarray


def support_edge(common_advantage, anatomical_weights, reliable_weights):
    """A_N to R_N on one unchanged D-stage advantage vector."""
    d = vector(common_advantage, "common_advantage")
    a = mean_weights(anatomical_weights, d.size, "anatomical")
    r = mean_weights(reliable_weights, d.size, "reliable")
    start, end = float(a @ d), float(r @ d)
    require(np.isfinite(start) and np.isfinite(end), "support_score_nonfinite")
    return LinearEdge(start, end, end - start, fixed_array(r - a))


def ceiling_edge(common_advantage, ceiling_denominators):
    """Mean of per-voxel ratios, with no renormalization of positive weights."""
    d = vector(common_advantage, "common_advantage")
    denominator = vector(ceiling_denominators, "ceiling_denominators")
    require(d.shape == denominator.shape and (denominator >= 0.10).all(), "ceiling_floor_or_identity")
    raw, normalized = float(np.mean(d)), float(np.mean(d / denominator))
    require(np.isfinite(raw) and np.isfinite(normalized), "ceiling_score_nonfinite")
    return LinearEdge(raw, normalized, normalized - raw,
                      fixed_array((1.0 / denominator - 1.0) / d.size))


@dataclass(frozen=True)
class LinearForecast:
    development_change: float
    operator_weights: np.ndarray

    def observed_change(self, new_concept_advantage):
        d = vector(new_concept_advantage, "new_concept_advantage")
        require(d.shape == self.operator_weights.shape, "forecast_voxel_identity_changed")
        value = float(self.operator_weights @ d)
        require(np.isfinite(value), "forecast_observation_nonfinite")
        return value


def freeze_linear_forecast(development_advantage, operator_weights):
    """Freeze from development only; new-concept observations cannot enter."""
    d = vector(development_advantage, "development_advantage")
    weights = vector(operator_weights, "operator_weights")
    require(d.shape == weights.shape, "forecast_voxel_identity_changed")
    value = float(weights @ d)
    require(np.isfinite(value), "forecast_nonfinite")
    return LinearForecast(value, fixed_array(weights))


@dataclass(frozen=True)
class GainDesign:
    matrix: np.ndarray
    standardized_gain: np.ndarray
    low_indices: np.ndarray
    high_indices: np.ndarray
    residual_gain_variance: float


def beta_gain_design(calibration_gain, spatial_bins, original_columns):
    """Eight fixed bin intercepts plus the standardized calibration gain."""
    gain = vector(calibration_gain, "calibration_gain")
    bins, columns = np.asarray(spatial_bins), np.asarray(original_columns)
    require(bins.shape == columns.shape == gain.shape and bins.dtype.kind in "iu"
            and columns.dtype.kind in "iu" and (columns >= 0).all()
            and len(np.unique(columns)) == len(columns), "gain_identity_invalid")
    require(set(bins.tolist()) == set(range(1, 9)), "eight_nonempty_spatial_bins_required")
    centered = gain - np.mean(gain)
    if np.all(gain == gain[0]):
        centered[:] = 0.0
    variance = float(np.mean(centered * centered))
    require(np.isfinite(variance) and variance > 0, "calibration_gain_variance_zero")
    z = centered / np.sqrt(variance)
    intercepts = np.column_stack([bins == b for b in range(1, 9)]).astype(np.float64)
    bin_means = np.array([np.mean(z[bins == b]) for b in range(1, 9)])
    residual = z - intercepts @ bin_means
    residual_variance = float(np.mean(residual * residual))
    require(np.isfinite(residual_variance) and residual_variance >= GAIN_VARIANCE_FLOOR,
            "residual_gain_variance_below_floor")
    design = np.column_stack([intercepts, z])
    singular = np.linalg.svd(design, compute_uv=False)
    require(np.isfinite(singular).all()
            and np.count_nonzero(singular > SVD_RELATIVE_TOLERANCE * singular[0]) == 9,
            "gain_design_rank_deficient")
    order = np.lexsort((columns, gain))
    split = len(order) // 2
    low, high = order[:split].copy(), order[split:].copy()
    low.setflags(write=False)
    high.setflags(write=False)
    return GainDesign(fixed_array(design), fixed_array(z), low, high, residual_variance)


def beta_high_minus_low(pooled_voxel_stage_change, design):
    change = vector(pooled_voxel_stage_change, "pooled_voxel_stage_change")
    require(isinstance(design, GainDesign) and len(change) == design.matrix.shape[0], "gain_design_identity_changed")
    value = float(np.mean(change[design.high_indices]) - np.mean(change[design.low_indices]))
    require(np.isfinite(value), "high_low_change_nonfinite")
    return value


@dataclass(frozen=True)
class BetaForecast:
    coefficients: np.ndarray
    predicted_voxel_change: np.ndarray
    development_block_mean_forecast: float
    crossfit_predicted_block_means: np.ndarray
    crossfit_observed_block_means: np.ndarray
    crossfit_absolute_errors: np.ndarray


def fit_beta_forecast(development_block_voxel_changes, design):
    """Twelve/twelve block cross-fit and final24-block OLS, within one ROI.

    The supplied targets are blockwise changes from frozen training-baseline
    R². They are not the pooled-image primary beta effect or voxel replicates.
    """
    require(isinstance(design, GainDesign), "gain_design_required")
    raw = np.asarray(development_block_voxel_changes)
    require(raw.dtype.kind in "iuf" and raw.shape == (24, design.matrix.shape[0]), "24_development_blocks_required")
    changes = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(changes).all(), "block_stage_changes_nonfinite")

    def fit(target):
        coefficient, _, rank, _ = np.linalg.lstsq(design.matrix, target, rcond=SVD_RELATIVE_TOLERANCE)
        require(rank == 9 and np.isfinite(coefficient).all(), "gain_OLS_invalid")
        predicted = design.matrix @ coefficient
        require(np.isfinite(predicted).all(), "gain_OLS_prediction_nonfinite")
        return coefficient, predicted

    crossfit = np.empty(24, dtype=np.float64)
    for train, test in ((slice(0, 12), slice(12, 24)), (slice(12, 24), slice(0, 12))):
        _, predicted = fit(np.mean(changes[train], axis=0))
        crossfit[test] = np.mean(predicted)
    coefficient, predicted = fit(np.mean(changes, axis=0))
    observed = np.mean(changes, axis=1)
    return BetaForecast(fixed_array(coefficient), fixed_array(predicted), float(np.mean(predicted)),
                        fixed_array(crossfit), fixed_array(observed), fixed_array(np.abs(observed - crossfit)))
