"""EP17 selected audit secondary families from the SAME primary-application bank.

Implements measurement_contract_specification.md's secondary families and the
audit counterpart in prediction_qualification_specification.md. The input must
be the existing qualified AuditPredictionResult from the already authorized
one-shot application. Its locked descriptors choose at most two checks; each
selected SCALE still uses its entire eligible six-component inference family.
No other pair/scale can be requested through this API. An empty descriptor
tuple produces no statistic and performs no secondary array scoring.

The observed arrays are the exact immutable snapshots retained by primary
application. No replacement observation argument, reader, audit grant, fit,
transform application, lock writer, persistence or scientific adjudicator is
provided. Structural checks do not authenticate caller-supplied provenance or
prove an existing persisted lock was valid. Readonly bytes-backed matrices are
reused by reference; mapping shells and small block coordinates are frozen.

Unlike development CKA, audit CKA has NO fold mixture. Each model's single
all-development rank64 feature bank is centered once over the resampled audit
images for each whole-support calculation. All 32 anatomical-bank CKAs are
averaged. Whole-block and participant multiplicities are literal replication;
absent people contribute zero, and positive-mass undefined results refuse the
entire family. The unchanged uncertainty engine supplies the fixed10000 draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

import audit_prediction as application
import development_endpoints as development
import development_secondary as secondary
import endpoint_evaluator as endpoint
import nested_prediction as nested
import uncertainty_core as uncertainty


class AuditSecondaryError(ValueError):
    """Refuse the selected family; never delete, add or substitute a check."""


def require(condition, code):
    if not condition:
        raise AuditSecondaryError(code)


def _reuse_matrix(value, shape):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float64)
            and value.shape == shape and not value.flags.writeable,
            "existing_immutable_audit_matrix_required")
    owner = value
    while isinstance(owner, np.ndarray):
        owner = owner.base
    require(isinstance(owner, bytes), "existing_immutable_audit_matrix_required")
    require(np.isfinite(value).all(), "audit_secondary_matrix_nonfinite")
    return value


def _layout(applied):
    require(isinstance(applied, application.AuditPredictionResult)
            and isinstance(applied.design, application.AuditDesign), "existing_audit_application_required")
    require(applied.pair_id in ("P1", "P2", "P3"), "one_selected_primary_pair_required")
    statistic = applied.statistic
    require(isinstance(statistic, endpoint.EndpointStatistic) and statistic.phase == "audit"
            and statistic.pair_id == applied.pair_id, "primary_audit_statistic_binding_changed")
    family = statistic.family
    require(isinstance(family, endpoint.EndpointFamily), "frozen_endpoint_family_required")
    blocks = endpoint.bind_blocks(applied.blocks, "audit")
    require(blocks == statistic.blocks, "primary_audit_block_order_changed")
    require({block.block_id for block in blocks} == set(applied.design.block_ids),
            "audit_design_block_binding_changed")
    for block in blocks:
        rows = [i for i, label in enumerate(applied.design.block_ids) if label == block.block_id]
        require(block.image_ids == tuple(applied.design.image_ids[i] for i in rows)
                and block.concept_ids == tuple(applied.design.concept_ids[i] for i in rows),
                "audit_design_image_binding_changed")
    ids = secondary.COMPONENTS if "D_VNC_raw" in family.primary_endpoint_ids else secondary.COMPONENTS[:-1]
    checks = applied.secondary_checks
    require(isinstance(checks, tuple) and len(checks) <= 2
            and all(isinstance(check, tuple) and len(check) == 2
                    and check[0] in secondary.SCALES and check[1] in ids for check in checks)
            and len(set(checks)) == len(checks), "at_most_two_frozen_secondary_checks_required")
    return family, blocks, ids, checks


@dataclass(frozen=True)
class _AuditBank:
    family: endpoint.EndpointFamily
    pair_id: str
    model_order: tuple
    responses: Mapping
    predictions: Mapping
    features: Mapping
    bank_positions: Mapping
    block_coordinates: np.ndarray
    component_ids: tuple
    resource_check: object

    def group_scores(self, scale, model, group, weights):
        fixed, check = self.family._supports[group], self.resource_check
        context = dict(model_id=model, group=group)
        if scale == "heldout_correlation":
            z = {stage: nested._call(check, "weighted_fisher_correlation", secondary.weighted_fisher_correlation,
                self.responses[group][stage], self.predictions[model][group][stage], weights,
                context={**context, "stage": stage}) for stage in endpoint.STAGES}
            d = z["D"]
            scores = [float(np.mean(z[stage], dtype=np.float64)) for stage in endpoint.STAGES]
            scores.extend((float(d @ fixed.anatomical_weights), float(d @ fixed.reliable_weights)))
            if "D_VNC" in self.component_ids:
                scores.append(float(np.mean(d[fixed.ceiling_positions], dtype=np.float64)))
            return np.asarray(scores, dtype=np.float64)
        features = self.features[model]
        scores = [nested._call(check, "weighted_linear_cka", secondary.weighted_linear_cka,
            features, self.responses[group][stage], weights, context={**context, "stage": stage})
            for stage in endpoint.STAGES]
        delivered, binding = self.responses[group]["D"], {**context, "stage": "D"}
        scores.append(secondary.mean_bank_cka(features, delivered, weights, self.bank_positions[group],
                                              resource_check=check, context=binding))
        reliable = np.flatnonzero(fixed.reliable_weights > 0)
        selected = secondary._slice_columns(check, delivered, reliable, binding)
        scores.append(nested._call(check, "weighted_linear_cka", secondary.weighted_linear_cka,
                                  features, selected, weights, context=binding))
        if "D_VNC" in self.component_ids:
            selected = secondary._slice_columns(check, delivered, fixed.ceiling_positions, binding)
            scores.append(nested._call(check, "weighted_linear_cka", secondary.weighted_linear_cka,
                                      features, selected, weights, context=binding))
        return np.asarray(scores, dtype=np.float64)


@dataclass(frozen=True, init=False, eq=False)
class AuditSecondaryStatistic:
    def __init__(self, bank, scale, selected_components):
        require(scale in secondary.SCALES and selected_components
                and all(component in bank.component_ids for component in selected_components),
                "selected_secondary_family_required")
        for name, value in (("_bank", bank), ("scale", scale), ("pair_id", bank.pair_id),
                            ("model_order", bank.model_order), ("component_ids", bank.component_ids),
                            ("score_scales", (scale,) * len(bank.component_ids)),
                            ("selected_components", tuple(selected_components)), ("phase", "audit")):
            object.__setattr__(self, name, value)

    def _people(self, block_multiplicities, included):
        bank = self._bank
        nested._event(bank.resource_check, "audit_secondary_statistic")
        blocks = endpoint.multiplicities(block_multiplicities, 12, "audit_block_multiplicities")
        weights = blocks[bank.block_coordinates]
        values = np.zeros((4, len(self.component_ids)), dtype=np.float64)
        first, second = self.model_order
        for index, person in enumerate(endpoint.SUBJECTS):
            if not included[index]:
                continue
            for roi in endpoint.ROIS:
                group = (person, roi)
                a = bank.group_scores(self.scale, first, group, weights)
                b = bank.group_scores(self.scale, second, group, weights)
                values[index] += (a - b) / 2
        require(np.isfinite(values).all(), "audit_secondary_statistic_nonfinite")
        return values

    def __call__(self, people_multiplicities, block_multiplicities):
        people = endpoint.multiplicities(people_multiplicities, 4, "people_multiplicities")
        values = self._people(block_multiplicities, people > 0)
        return (people @ values) / people.sum()

    def participant_effects(self, block_multiplicities):
        return self._people(block_multiplicities, np.ones(4, dtype=bool))

    def crossed_bounds(self):
        nested._event(self._bank.resource_check, "audit_secondary_crossed_bounds")
        return uncertainty.crossed_bounds(self, self.component_ids, self.score_scales, "audit")


@dataclass(frozen=True)
class AuditSecondary:
    secondary_checks: tuple
    statistics: Mapping


def compose_audit_secondary(applied, anatomical_banks, *, resource_check):
    """Bind selected scale families without copying the admitted audit matrices.

    ``anatomical_banks`` may be None unless CKA was prelocked. Fisher-z needs
    only the already frozen inclusion weights. Any CKA family requires every
    original AnatomicalBank and all32 actual memberships. Existing result and
    lock authenticity remain caller preconditions, not this array adapter's
    claimed guarantee. No numerical secondary score runs during binding.
    """
    require(callable(resource_check), "external_resource_callback_required")
    nested._event(resource_check, "begin_audit_secondary")
    family, blocks, ids, checks = _layout(applied)
    if not checks:
        nested._event(resource_check, "complete_audit_secondary", "after")
        return AuditSecondary(checks, MappingProxyType({}))
    models = endpoint.PAIR_SIGNS[applied.pair_id]
    scales = tuple(dict.fromkeys(scale for scale, _ in checks))
    development._mapping(applied.columns, endpoint.GROUPS, "complete_audit_column_groups_required")
    development._mapping(applied.responses, endpoint.GROUPS, "complete_retained_audit_responses_required")
    observed, predicted, features = {}, {}, {}
    for group in endpoint.GROUPS:
        require(np.array_equal(endpoint.column_ids(applied.columns[group], "audit_columns"),
                               family._supports[group].columns), "audit_original_column_binding_changed")
        development._mapping(applied.responses[group], endpoint.STAGES, "complete_retained_audit_B_C_D_required")
        observed[group] = {}
        width = len(family._supports[group].columns)
        for stage in endpoint.STAGES:
            context = dict(group=group, stage=stage, shape=(application.AUDIT_IMAGES, width))
            observed[group][stage] = nested._call(resource_check, "validate_audit_secondary_array", _reuse_matrix,
                applied.responses[group][stage], context["shape"], context=context)
    if "heldout_correlation" in scales:
        development._mapping(applied.predictions, models, "only_selected_pair_predictions_required")
        for model in models:
            development._mapping(applied.predictions[model], endpoint.GROUPS, "complete_audit_prediction_groups_required")
            predicted[model] = {}
            for group in endpoint.GROUPS:
                development._mapping(applied.predictions[model][group], endpoint.STAGES, "complete_audit_prediction_B_C_D_required")
                predicted[model][group] = {}
                for stage in endpoint.STAGES:
                    context = dict(model_id=model, group=group, stage=stage,
                                   shape=(application.AUDIT_IMAGES, len(family._supports[group].columns)))
                    predicted[model][group][stage] = nested._call(resource_check, "validate_audit_secondary_array",
                        _reuse_matrix, applied.predictions[model][group][stage], context["shape"], context=context)
    banks = MappingProxyType({})
    if "linear_CKA" in scales:
        banks = nested._call(resource_check, "bind_audit_secondary_banks", secondary._bind_banks, family, anatomical_banks)
        development._mapping(applied.cka_features, models, "only_selected_pair_CKA_features_required")
        for model in models:
            context = dict(model_id=model, rank=64, shape=(application.AUDIT_IMAGES, 64))
            features[model] = nested._call(resource_check, "validate_audit_secondary_array", _reuse_matrix,
                                           applied.cka_features[model], context["shape"], context=context)
    nested._event(resource_check, "bind_audit_secondary_coordinates")
    lookup = {block.block_id: index for index, block in enumerate(blocks)}
    coordinates = endpoint.frozen_array([lookup[label] for label in applied.design.block_ids], integer=True)
    bank = _AuditBank(family, applied.pair_id, models, nested._nested_map(observed), nested._nested_map(predicted),
                      MappingProxyType(features), banks, coordinates, ids, resource_check)
    statistics = MappingProxyType({scale: AuditSecondaryStatistic(bank, scale,
        tuple(component for selected_scale, component in checks if selected_scale == scale)) for scale in scales})
    nested._event(resource_check, "complete_audit_secondary", "after")
    return AuditSecondary(checks, statistics)
