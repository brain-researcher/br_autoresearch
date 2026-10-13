"""Compose EP17's adopted development forecasts on already admitted arrays.

The caller supplies a development EndpointStatistic, the explicit reference
block order frozen from metadata, and (for beta edges) calibration-qualified
GainDesigns bound to original columns. Nothing is read, released or persisted.
The reference order must match exactly; this code never sorts or repairs it.
Metadata equality cannot authenticate the original freeze, calibration origin,
pair qualification, reader isolation or eligibility of the source responses.

Support/ceiling forecasts are the frozen development edge carried forward.
Beta forecasts retain the separate equal-block target and 12/12 cross-fit.
Cross-fit errors use participant-by-heldout-block rows, averaging the two ROI
predictions and observations BEFORE taking absolute error. The fitted spatial
bin intercepts make the aggregate forecast a repeatability quantity, not a
test of incremental gain-map explanation. Pooled primary endpoints and H are
unchanged. No sign, cross-fit pass threshold or scientific claim is chosen.

The required resource callback admits each allocating/numerical step. Each
existing measurement.fit_beta_forecast call is one bounded per-group operation
(three nine-column OLS fits); the numerical core itself is unchanged. Callback
or numerical failure aborts without a partial forecast. No audit input is
accepted, and the size-control edge has no adopted explanatory forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

import endpoint_evaluator as endpoint
import measurement_core as measurement


LINEAR_EDGES = MappingProxyType({
    "A_N_to_R_N": ("D_AN_raw", "D_RN_raw", .02),
    "raw_to_ceiling_normalized": ("D_VNC_raw", "D_VNC_normalized", .05),
})
BETA_EDGES = MappingProxyType({"B_to_C": "H_BC", "C_to_D": "H_CD"})


class ForecastError(ValueError):
    """Unresolved or inconsistent inputs are not scientific negative evidence."""


def require(condition, code):
    if not condition:
        raise ForecastError(code)


@dataclass(frozen=True)
class ColumnBoundGainDesign:
    """A supplied, previously qualified calibration design on native columns.

    The private operator producer exports A_all plus edge-specific design,
    standardized_gain, low_columns, high_columns and residual_gain_variance.
    A caller converts those original low/high IDs to positions on this exact
    A_all axis. It must select the correct edge's frozen calibration artifact;
    matching strata alone cannot authenticate an otherwise different gain map.
    """
    column_ids: np.ndarray
    design: measurement.GainDesign


@dataclass(frozen=True)
class ResourceEvent:
    operation: str
    when: str
    edge_id: str
    group: tuple | None = None
    shape: tuple = ()


def _event(check, operation, edge_id, when="before", *, group=None, shape=()):
    check(ResourceEvent(operation, when, edge_id, group, shape))


def _call(check, operation, edge_id, function, *args, group=None, shape=()):
    _event(check, operation, edge_id, group=group, shape=shape)
    result = function(*args)
    _event(check, operation, edge_id, "after", group=group, shape=shape)
    return result


@dataclass(frozen=True)
class DevelopmentForecast:
    """Calculated development values, not a persisted final relation lock.

    eligible records only the selected operator's calibration applicability.
    Missing inputs or malformed designs raise instead of becoming inapplicable.
    An eligible return does not assert that cross-fit qualification, relation
    selection, precision, audit access or a scientific acceptance gate passed.
    The caller separately supplies its frozen direction to the audit adjudicator.
    """
    endpoint_family: endpoint.EndpointFamily
    pair_id: str
    edge_id: str
    reference_block_order: tuple
    eligible: bool
    eligibility_reason: str
    target: str | None
    value: float | None
    absolute_error_limit: float | None
    participant_forecasts: np.ndarray | None
    gain_designs: Mapping
    group_fits: Mapping
    crossfit_predicted: np.ndarray | None
    crossfit_observed: np.ndarray | None
    crossfit_absolute_errors: np.ndarray | None


def _unavailable(statistic, edge, order, reason):
    return DevelopmentForecast(statistic.family, statistic.pair_id, edge, order, False, reason,
                               None, None, None, None, MappingProxyType({}), MappingProxyType({}),
                               None, None, None)


def _bind_design(value, support, component):
    require(isinstance(value, ColumnBoundGainDesign), "column_bound_calibration_design_required")
    columns = endpoint.column_ids(value.column_ids, "gain_design_columns")
    require(np.array_equal(columns, support.columns), "gain_design_original_columns_changed")
    design = value.design
    require(isinstance(design, measurement.GainDesign), "calibration_GainDesign_required")
    matrix = endpoint.frozen_array(design.matrix)
    gain = endpoint.frozen_array(design.standardized_gain)
    low = endpoint.column_ids(design.low_indices, "gain_low_positions")
    high = endpoint.column_ids(design.high_indices, "gain_high_positions")
    require(matrix.shape == (len(columns), 9) and gain.shape == columns.shape,
            "gain_design_shape_changed")
    require(np.array_equal(matrix[:, -1], gain), "standardized_gain_column_changed")
    bins = matrix[:, :8]
    expected_bins = np.column_stack([support.spatial_bins == bin_id for bin_id in range(1, 9)])
    require(np.array_equal(bins, expected_bins), "frozen_spatial_bin_assignment_changed")
    frozen_low, frozen_high = support.strata[component]
    require(np.array_equal(low, frozen_low) and np.array_equal(high, frozen_high),
            "frozen_gain_strata_changed")
    variance = design.residual_gain_variance
    require(np.ndim(variance) == 0 and np.isfinite(variance)
            and float(variance) >= measurement.GAIN_VARIANCE_FLOOR,
            "calibration_gain_design_not_qualified")
    # Full-rank numerical qualification belongs to the admitted calibration
    # design; the unchanged OLS core also refuses a rank other than nine.
    frozen = measurement.GainDesign(matrix, gain, low, high, float(variance))
    return ColumnBoundGainDesign(columns, frozen)


def _freeze_fit(fit):
    return replace(fit, coefficients=endpoint.frozen_array(fit.coefficients),
                   predicted_voxel_change=endpoint.frozen_array(fit.predicted_voxel_change),
                   crossfit_predicted_block_means=endpoint.frozen_array(fit.crossfit_predicted_block_means),
                   crossfit_observed_block_means=endpoint.frozen_array(fit.crossfit_observed_block_means),
                   crossfit_absolute_errors=endpoint.frozen_array(fit.crossfit_absolute_errors))


def _linear_summary(statistic, edge):
    before, after, _ = LINEAR_EDGES[edge]
    effects = statistic.participant_effects(np.ones(24, dtype=int))
    changes = effects[:, statistic.component_ids.index(after)] - effects[:, statistic.component_ids.index(before)]
    return endpoint.frozen_array(changes)


def _beta_summary(fits):
    forecasts = np.empty(4, dtype=np.float64)
    predicted = np.empty((4, 24), dtype=np.float64)
    observed = np.empty_like(predicted)
    for index, person in enumerate(endpoint.SUBJECTS):
        left, right = (fits[person, roi] for roi in endpoint.ROIS)
        forecasts[index] = (left.development_block_mean_forecast + right.development_block_mean_forecast) / 2.
        predicted[index] = (left.crossfit_predicted_block_means + right.crossfit_predicted_block_means) / 2.
        observed[index] = (left.crossfit_observed_block_means + right.crossfit_observed_block_means) / 2.
    return tuple(endpoint.frozen_array(value) for value in
                 (forecasts, predicted, observed, np.abs(observed - predicted)))


def build_development_forecast(statistic, edge_id, *, reference_block_order,
                               gain_designs=None, resource_check):
    """Calculate exactly one registered forecast without selecting a relation.

    reference_block_order is independently supplied from the frozen metadata
    order, including its original label types. It must match statistic.blocks
    exactly. gain_designs maps all eight participant/ROI groups to bound designs
    for the selected beta edge. Linear edges need no gain input; a supplied
    unrelated design panel is refused. There is no audit application method.
    """
    require(callable(resource_check), "external_resource_callback_required")
    require(isinstance(statistic, endpoint.EndpointStatistic) and statistic.phase == "development",
            "development_statistic_required")
    require(edge_id in (*LINEAR_EDGES, *BETA_EDGES, "A_all_to_A_N"), "unregistered_forecast_edge")
    _event(resource_check, "begin", edge_id)
    require(isinstance(reference_block_order, (tuple, list)), "explicit_reference_block_order_required")
    order = tuple(reference_block_order)
    actual_order = tuple(block.block_id for block in statistic.blocks)
    require(len(order) == 24 and order == actual_order
            and all((isinstance(given, str) and isinstance(bound, str)) or
                    (isinstance(given, Integral) and not isinstance(given, (bool, np.bool_))
                     and isinstance(bound, Integral) and not isinstance(bound, (bool, np.bool_)))
                    for given, bound in zip(order, actual_order)),
            "frozen_reference_block_order_changed")

    if edge_id == "A_all_to_A_N":
        return _unavailable(statistic, edge_id, order, "size_control_has_no_adopted_explanatory_forecast")
    if edge_id in LINEAR_EDGES:
        require(gain_designs is None, "linear_edge_does_not_use_gain_designs")
        before, after, error_limit = LINEAR_EDGES[edge_id]
        if before not in statistic.primary_endpoint_ids or after not in statistic.primary_endpoint_ids:
            return _unavailable(statistic, edge_id, order, "calibration_ceiling_support_inapplicable")
        forecasts = _call(resource_check, "linear_edge_forecasts", edge_id, _linear_summary,
                          statistic, edge_id, shape=(4,))
        result = DevelopmentForecast(statistic.family, statistic.pair_id, edge_id, order, True,
            "frozen_operator_applicable", "aggregate_edge_change", float(np.mean(forecasts)), error_limit,
            forecasts, MappingProxyType({}), MappingProxyType({}), None, None, None)
    else:
        component = BETA_EDGES[edge_id]
        if component not in statistic.family.h_component_ids:
            return _unavailable(statistic, edge_id, order, "calibration_beta_gain_inapplicable")
        require(isinstance(gain_designs, Mapping) and set(gain_designs) == set(endpoint.GROUPS),
                "eight_calibration_gain_designs_required")
        bound = {}
        for group in endpoint.GROUPS:
            support = statistic.family._supports[group]
            bound[group] = _call(resource_check, "bind_calibration_design", edge_id, _bind_design,
                                 gain_designs[group], support, component, group=group,
                                 shape=(len(support.columns), 9))
        fits = {}
        for group in endpoint.GROUPS:
            shape = (24, len(bound[group].column_ids))
            changes = _call(resource_check, "beta_block_changes", edge_id,
                            statistic.group_beta_block_changes, group, edge_id, group=group, shape=shape)
            fit = _call(resource_check, "fit_beta_forecast", edge_id, measurement.fit_beta_forecast,
                        changes, bound[group].design, group=group, shape=shape)
            fits[group] = _call(resource_check, "freeze_beta_fit", edge_id, _freeze_fit, fit,
                                group=group, shape=shape)
        forecasts, predicted, observed, errors = _call(resource_check, "aggregate_beta_forecasts", edge_id,
                                                      _beta_summary, fits, shape=(4, 24))
        result = DevelopmentForecast(statistic.family, statistic.pair_id, edge_id, order, True,
            "frozen_operator_applicable_crossfit_pass_not_adjudicated", "equal_block_mean_stage_change",
            float(np.mean(forecasts)), .02, forecasts, MappingProxyType(bound), MappingProxyType(fits),
            predicted, observed, errors)
    require(np.isfinite(result.value), "forecast_nonfinite")
    _event(resource_check, "complete", edge_id, "after")
    return result
