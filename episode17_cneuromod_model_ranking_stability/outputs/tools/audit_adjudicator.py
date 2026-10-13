"""Apply EP17's fixed primary audit gates to supplied, matched calculations.

Rules: GOAL.md, 'Development search and one-shot audit'; SEARCH_POLICY.yaml,
'audit'; outputs/measurement_contract_specification.md, 'Exact audit behavior
and forecast counterparts'; prediction_qualification_specification.md's adopted
margin/width table. This module creates no method freeze and opens no outcome.
The caller supplies an EXISTING frozen relation, its calibration family and
development forecast, plus an already admitted audit EndpointStatistic/bounds.

Persisted freeze provenance, precise development evidence, checkpoint exposure,
predictor/role correspondence, data separation and one-shot access remain caller
obligations. A passing numerical gate is not candidate readiness or permission
to audit. No final heldout-nonreplication claim is returned without its missing
development qualification. The policy does not operationally define 'stable
participant or concept-block heterogeneity'; failures needing that classification
remain explicit. Secondary correlation/CKA/conditionality are not adjudicated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

import endpoint_evaluator as endpoint
import uncertainty_core as uncertainty

WIDTHS = {"raw_R2": .02, "ceiling_normalized_R2": .10, "beta_high_minus_low_gain": .02}
EXPLAINED_EDGES = {
    "B_to_C": ("B_Aall_raw", "C_Aall_raw", "H_BC", .02),
    "C_to_D": ("C_Aall_raw", "D_Aall_raw", "H_CD", .02),
    "A_N_to_R_N": ("D_AN_raw", "D_RN_raw", None, .02),
    "raw_to_ceiling_normalized": ("D_VNC_raw", "D_VNC_normalized", None, .05),
}


class AdjudicationError(ValueError):
    """Unmatched or unresolved inputs are not scientific negative evidence."""


def require(condition, code):
    if not condition:
        raise AdjudicationError(code)


def finite_scalar(value, name):
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value), name + "_invalid")
    return float(value)


def direction(value):
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value in (-1, 1),
            "frozen_direction_required")
    return int(value)


@dataclass(frozen=True)
class FrozenForecast:
    """The already frozen scalar forecast, in its registered edge units.

    The caller must have averaged its eight development participant/ROI
    forecasts with equal ROI and then equal participant weight. This scalar
    interface cannot verify that upstream aggregation or when it was frozen.
    Error is computed on the aggregate, never averaged absolute group errors.
    """
    value: float
    direction: int
    absolute_error_limit: float


@dataclass(frozen=True)
class FrozenAuditRelation:
    """Caller-supplied frozen decisions; no missing decision gets a default.

    component_ids/primary_endpoint_ids describe the entire frozen INFERENCE
    family. required_components/directions describe its relation-specific J.
    margins and maximum_full_widths cover every inference component, including
    eligible H terms that may not enter J. Only the registered fixed values are
    accepted; these arguments cannot amend a scientific threshold.
    """
    pair_id: str
    relation: str
    endpoint_family: endpoint.EndpointFamily
    inference_family: str
    component_ids: tuple[str, ...]
    primary_endpoint_ids: tuple[str, ...]
    required_components: tuple[str, ...]
    directions: Mapping[str, int]
    margins: Mapping[str, float]
    maximum_full_widths: Mapping[str, float]
    edge_id: str | None
    forecast: FrozenForecast | None

    def __post_init__(self):
        for name in ("component_ids", "primary_endpoint_ids", "required_components"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("directions", "margins", "maximum_full_widths"):
            require(isinstance(getattr(self, name), Mapping), name + "_required")
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class ForecastCheck:
    predicted: float
    observed: float
    registered_direction: int
    prediction_direction_pass: bool
    observation_direction_pass: bool
    absolute_error: float
    absolute_error_limit: float
    error_pass: bool

    @property
    def passed(self):
        return self.prediction_direction_pass and self.observation_direction_pass and self.error_pass


@dataclass(frozen=True)
class AuditGateResult:
    status: str
    relation: str
    pair_id: str
    required_components: tuple[str, ...]
    signed_bounds_pass: bool
    participant_component_pass: tuple[tuple[bool, ...], ...]
    common_passing_participants: tuple[str, ...]
    common_participant_gate_pass: bool
    leave_one_participant_effects: np.ndarray
    leave_one_block_effects: np.ndarray
    leave_one_participant_gate_pass: bool
    leave_one_block_gate_pass: bool
    forecast: ForecastCheck | None
    primary_audit_gates_pass: bool
    reversal_endpoint_gates_pass: bool | None
    precision_adequate: bool
    component_precision_adequate: tuple[bool, ...]
    equivalent_required_components: tuple[str, ...]
    opposite_required_components: tuple[str, ...]
    family_interval_classification: str
    unresolved_requirements: tuple[str, ...]


def _snapshot_bounds(bounds):
    require(isinstance(bounds, uncertainty.SimultaneousBounds) and bounds.phase == "audit", "audit_bounds_required")
    critical = finite_scalar(bounds.critical_value, "max_t_critical_value")
    require(critical >= 0, "max_t_critical_value_invalid")
    values = {}
    for name in ("estimate", "standard_error", "lower", "upper", "margins"):
        values[name] = endpoint.frozen_array(getattr(bounds, name))
        require(values[name].shape == (len(bounds.component_ids),), "bounds_component_shape_changed")
    require((values["standard_error"] > 0).all() and (values["margins"] > 0).all(), "bounds_scale_invalid")
    radius = critical * values["standard_error"]
    require(np.isfinite(radius).all()
            and np.array_equal(values["lower"], values["estimate"] - radius)
            and np.array_equal(values["upper"], values["estimate"] + radius), "bounds_not_registered_max_t_form")
    return replace(bounds, component_ids=tuple(bounds.component_ids), score_scales=tuple(bounds.score_scales),
                   primary_endpoint_ids=tuple(bounds.primary_endpoint_ids), critical_value=critical, **values)


def _bind_relation(relation, statistic, bounds):
    require(isinstance(relation, FrozenAuditRelation), "frozen_relation_required")
    require(isinstance(statistic, endpoint.EndpointStatistic) and statistic.phase == "audit", "audit_statistic_required")
    require(statistic.family is relation.endpoint_family, "frozen_calibration_family_changed")
    require(relation.pair_id == statistic.pair_id, "frozen_model_pair_changed")
    require(relation.pair_id in ("P1", "P2", "P3"), "random_falsifier_is_not_a_primary_relation")
    require(relation.relation in ("contract_robust_relation", "predicted_measurement_reversal"), "primary_relation_not_registered")
    require(relation.inference_family in ("F_raw", "F_raw_normalized"), "frozen_inference_family_required")
    normalized = "D_VNC_normalized" in statistic.primary_endpoint_ids
    require(relation.inference_family != "F_raw_normalized" or normalized, "normalized_family_inapplicable")
    indices = tuple(i for i, key in enumerate(statistic.component_ids)
                    if relation.inference_family != "F_raw" or key != "D_VNC_normalized")
    ids = tuple(statistic.component_ids[i] for i in indices)
    scales = tuple(statistic.score_scales[i] for i in indices)
    primary = tuple(key for key in statistic.primary_endpoint_ids if key in ids)
    require(relation.component_ids == bounds.component_ids == ids
            and relation.primary_endpoint_ids == bounds.primary_endpoint_ids == primary
            and bounds.score_scales == scales, "frozen_inference_roster_or_scale_changed")
    require(set(relation.margins) == set(ids) and set(relation.maximum_full_widths) == set(ids), "complete_frozen_thresholds_required")
    margins = np.array([finite_scalar(relation.margins[key], "margin") for key in ids])
    widths = np.array([finite_scalar(relation.maximum_full_widths[key], "full_width") for key in ids])
    require(np.array_equal(margins, [uncertainty.MARGINS[scale] for scale in scales])
            and np.array_equal(bounds.margins, margins), "adopted_margins_changed")
    require(np.array_equal(widths, [WIDTHS[scale] for scale in scales]), "adopted_full_widths_changed")

    if relation.relation == "contract_robust_relation":
        require(primary == statistic.primary_endpoint_ids, "robustness_requires_all_eligible_primary_endpoints")
        required = primary
        require(relation.edge_id is None and relation.forecast is None, "robustness_has_no_selected_forecast")
    else:
        require(relation.edge_id in EXPLAINED_EDGES,
                "explained_edge_unregistered_or_size_control_forecast_unspecified")
        start, end, high_low, error_limit = EXPLAINED_EDGES[relation.edge_id]
        required = (start, end) + (() if high_low is None else (high_low,))
        require(all(key in ids for key in required), "selected_reversal_or_beta_explanation_inapplicable")
        require(isinstance(relation.forecast, FrozenForecast), "frozen_forecast_required")
        finite_scalar(relation.forecast.value, "forecast_value")
        direction(relation.forecast.direction)
        require(finite_scalar(relation.forecast.absolute_error_limit, "forecast_error_limit") == error_limit,
                "adopted_forecast_error_limit_changed")
    require(relation.required_components == required and set(relation.directions) == set(required),
            "frozen_required_components_changed")
    signs = np.array([direction(relation.directions[key]) for key in required], dtype=int)
    if relation.relation == "contract_robust_relation":
        require((signs == signs[0]).all(), "robustness_direction_must_be_common")
    else:
        require(signs[0] == -signs[1], "reversal_endpoint_directions_must_oppose")
    estimates = statistic(np.ones(4, dtype=int), np.ones(12, dtype=int))
    require(np.array_equal(bounds.estimate, estimates[list(indices)]), "bounds_do_not_match_audit_statistic")
    return indices, required, signs, margins, widths


def adjudicate_primary_audit(relation, statistic, bounds):
    """Return the selected relation's gates; never select a different relation.

    Width limits determine whether a FAILURE can be called adequately precise.
    They are not an extra positive gate. A decisive negative audit pattern still
    needs qualified precise development evidence before a nonreplication claim.
    """
    bounds = _snapshot_bounds(bounds)
    indices, required, signs, margins, widths = _bind_relation(relation, statistic, bounds)
    full_locations = {key: i for i, key in enumerate(statistic.component_ids)}
    inference_locations = {key: i for i, key in enumerate(bounds.component_ids)}
    take = [full_locations[key] for key in required]
    bound_take = [inference_locations[key] for key in required]
    required_margins = margins[bound_take]
    lower, upper = bounds.lower[bound_take], bounds.upper[bound_take]
    signed_lower = np.where(signs > 0, lower, -upper)
    bound_pass = signed_lower > required_margins
    all_people, all_blocks = np.ones(4, dtype=int), np.ones(12, dtype=int)
    person_effects = statistic.participant_effects(all_blocks)[:, take]
    person_pass = person_effects * signs > required_margins
    passing_people = tuple(person for i, person in enumerate(endpoint.SUBJECTS) if person_pass[i].all())
    person_gate = len(passing_people) >= 3
    leave_people = []
    for i in range(4):
        multiplicities = all_people.copy()
        multiplicities[i] = 0
        leave_people.append(statistic(multiplicities, all_blocks)[take])
    leave_blocks = []
    for i in range(12):
        multiplicities = all_blocks.copy()
        multiplicities[i] = 0
        leave_blocks.append(statistic(all_people, multiplicities)[take])
    leave_people, leave_blocks = endpoint.frozen_array(leave_people), endpoint.frozen_array(leave_blocks)
    people_direction_pass = leave_people * signs > 0
    blocks_direction_pass = leave_blocks * signs > 0
    people_gate, blocks_gate = bool(people_direction_pass.all()), bool(blocks_direction_pass.all())

    forecast_check, endpoint_reversal = None, None
    if relation.relation == "predicted_measurement_reversal":
        if relation.edge_id in endpoint.BETA_EDGES:
            observed = statistic.beta_block_mean(relation.edge_id, all_people, all_blocks)
        else:
            # Both support and ceiling targets apply the same frozen weights
            # to the shared D advantage. No mean of per-group absolute errors.
            start, end = required[:2]
            point = statistic(all_people, all_blocks)
            observed = float(point[full_locations[end]] - point[full_locations[start]])
        frozen = relation.forecast
        predicted = float(frozen.value)
        absolute_error = abs(observed - predicted)
        forecast_check = ForecastCheck(predicted, observed, int(frozen.direction),
            bool(frozen.direction * predicted > 0), bool(frozen.direction * observed > 0),
            absolute_error, float(frozen.absolute_error_limit), bool(absolute_error <= frozen.absolute_error_limit))
        # Whether the reversal itself repeats is distinct from its explanatory
        # H component and forecast. Keep the locked edge and directions.
        endpoint_reversal = bool(bound_pass[:2].all() and np.count_nonzero(person_pass[:, :2].all(axis=1)) >= 3
                                 and people_direction_pass[:, :2].all() and blocks_direction_pass[:, :2].all())
    passed = bool(bound_pass.all() and person_gate and people_gate and blocks_gate
                  and (forecast_check is None or forecast_check.passed))
    component_precise = bounds.upper - bounds.lower <= widths
    # Audit adequacy belongs to the frozen relation's required set J. An
    # unselected H remains in the simultaneous family and width diagnostics,
    # but cannot veto a primary stability/equivalence conclusion.
    precise = bool(component_precise[bound_take].all())
    equivalent = tuple(key for i, key in enumerate(required)
                       if lower[i] >= -required_margins[i] and upper[i] <= required_margins[i])
    opposite = tuple(key for i, key in enumerate(required)
                     if (upper[i] < -required_margins[i] if signs[i] > 0 else lower[i] > required_margins[i]))
    unresolved = ()
    if passed:
        status = "primary_audit_gates_pass"
    elif endpoint_reversal:
        status = "unexplained_measurement_reversal"
    elif not precise:
        status = "audit_underidentified"
    elif equivalent or opposite:
        status = "adequate_negative_pattern_requires_development_qualification"
        unresolved = ("precise_development_evidence_not_adjudicated",)
    else:
        status = "unresolved_scientific_classification"
        unresolved = ("failed_gate_is_not_an_operationally_defined_stable_heterogeneity_pattern",
                      "no_decisive_required_component_equivalence_or_opposite_margin_pattern")
    directed_edges = tuple(edge for edge in statistic.family.directed_edges
                           if all(key in bounds.primary_endpoint_ids for key in edge))
    classification = uncertainty.relation_decision(bounds, directed_edges)
    # This whole-family descriptor cannot rescue failure of the frozen edge.
    return AuditGateResult(status, relation.relation, statistic.pair_id, required,
        bool(bound_pass.all()), tuple(tuple(bool(value) for value in row) for row in person_pass),
        passing_people, person_gate, leave_people, leave_blocks, people_gate, blocks_gate,
        forecast_check, passed, endpoint_reversal, precise, tuple(bool(value) for value in component_precise),
        equivalent, opposite, classification, unresolved)
