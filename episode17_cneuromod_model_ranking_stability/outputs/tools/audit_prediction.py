"""Apply an existing EP17 primary pair's frozen predictors to supplied arrays.

Array-only implementation of measurement_contract_specification.md's 'Exact
audit behavior' and prediction_qualification_specification.md's common linking
method. This is not a reader, persisted-lock validator, one-shot access grant,
fitter or audit opener. The caller supplies already authorized sealed_audit
arrays, qualified immutable all-development fits and an existing frozen relation.
Counts/IDs and coefficient-column bindings cannot authenticate their provenance.
The caller must have validated the complete relation before this application:
required components, directions, margins, widths and forecast. This module
matches its pair/calibration/inference roster only; full relation validation
and scientific gates belong to audit_adjudicator. A malformed lock is not
made safe to open by passing this module's narrower array-binding checks.

All original columns, people, ROIs and stages are mandatory. Response variance
on all images is required. Under the adopted October12 primary-R² exception,
individual blocks require finite inputs/moments and positive finite SST around
the stored DEVELOPMENT target mean, without a centered-variance veto.
Failure refuses the entire application. Audit
responses never determine a predictor, intercept, transform, support or repair.
One D prediction bank serves every support/ceiling endpoint. No uncertainty or
adjudication runs here; use the existing frozen relation/statistic interface.

At most two already locked secondary descriptors are retained. No secondary
score is computed here. Returned rank64 features use each model's single frozen
all-development transform; a separate selected-secondary caller must retain
the complete eligible family for inference on any selected scale.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

import audit_adjudicator as adjudicator
import development_endpoints as development
import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction

AUDIT_IMAGES = 606
SECONDARY_SCALES = ("heldout_correlation", "linear_CKA")
SECONDARY_COMPONENTS = ("B_Aall", "C_Aall", "D_Aall", "D_AN", "D_RN", "D_VNC")


class AuditPredictionError(ValueError):
    """Whole-application refusal; no selection, deletion or secondary rescue."""


def require(condition, code):
    if not condition:
        raise AuditPredictionError(code)


@dataclass(frozen=True)
class AuditDesign:
    image_ids: tuple
    concept_ids: tuple
    roles: tuple
    block_ids: tuple

    def __post_init__(self):
        for name in ("image_ids", "concept_ids", "roles", "block_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        require(all(len(getattr(self, name)) == AUDIT_IMAGES for name in
                    ("image_ids", "concept_ids", "roles", "block_ids")), "complete_606_audit_images_required")
        require(all(isinstance(value, str) and value for value in self.image_ids + self.concept_ids),
                "audit_image_concept_identity_invalid")
        require(len(set(self.image_ids)) == AUDIT_IMAGES, "duplicate_audit_image_identity")
        require(set(self.roles) == {"sealed_audit"}, "sealed_audit_role_only")
        require(all(nested._label(value) for value in self.block_ids), "audit_block_label_invalid")
        assignments = {}
        for concept, block in zip(self.concept_ids, self.block_ids):
            require(concept not in assignments or assignments[concept] == block, "audit_concept_crosses_blocks")
            assignments[concept] = block
        require(len(assignments) == 120, "all_120_audit_concepts_required")
        require(len(set(self.block_ids)) == 12 and
                all(sum(value == block for value in assignments.values()) == 10 for block in set(self.block_ids)),
                "twelve_10_concept_audit_blocks_required")


@dataclass(frozen=True)
class AuditFeaturePanel:
    rows: AuditDesign
    layer: str
    values: np.ndarray


@dataclass(frozen=True)
class AuditResponsePanel:
    rows: AuditDesign
    column_ids: np.ndarray
    stages: Mapping


@dataclass(frozen=True)
class FrozenAuditPredictor:
    """Caller binds coefficients from final_predictors to their original IDs."""
    predictor: nested.FrozenPredictor
    column_ids: Mapping


@dataclass(frozen=True)
class AuditPredictionResult:
    design: AuditDesign
    blocks: tuple
    pair_id: str
    columns: Mapping
    predictions: Mapping
    baselines: Mapping  # One immutable development-mean vector per group/stage.
    cka_features: Mapping  # One (606,64) bank per selected model.
    paired_errors: Mapping
    statistic: endpoint.EndpointStatistic
    secondary_checks: tuple
    responses: Mapping  # Exact existing immutable snapshots, retained without recopying.


def _relation_layout(relation, secondary_checks):
    require(isinstance(relation, adjudicator.FrozenAuditRelation), "existing_frozen_audit_relation_required")
    require(relation.pair_id in ("P1", "P2", "P3"), "one_selected_primary_pair_required")
    family = relation.endpoint_family
    require(isinstance(family, endpoint.EndpointFamily), "frozen_endpoint_family_required")
    require(relation.relation in ("contract_robust_relation", "predicted_measurement_reversal"),
            "registered_frozen_relation_required")
    require(relation.inference_family in ("F_raw", "F_raw_normalized"), "frozen_inference_family_required")
    require(relation.inference_family != "F_raw_normalized" or "D_VNC_normalized" in family.primary_endpoint_ids,
            "normalized_family_inapplicable")
    ids = tuple(key for key in family.component_ids
                if relation.inference_family != "F_raw" or key != "D_VNC_normalized")
    primary = tuple(key for key in family.primary_endpoint_ids if key in ids)
    require(relation.component_ids == ids and relation.primary_endpoint_ids == primary,
            "frozen_inference_roster_changed")
    eligible = SECONDARY_COMPONENTS if "D_VNC_raw" in family.primary_endpoint_ids else SECONDARY_COMPONENTS[:-1]
    require(isinstance(secondary_checks, tuple) and len(secondary_checks) <= 2
            and all(isinstance(check, tuple) and len(check) == 2 and check[0] in SECONDARY_SCALES
                    and check[1] in eligible for check in secondary_checks)
            and len(set(secondary_checks)) == len(secondary_checks), "at_most_two_frozen_secondary_checks_required")
    return family, endpoint.PAIR_SIGNS[relation.pair_id], secondary_checks


def _layout(design, development_design, predictors, features, responses, family, models, block_order):
    require(isinstance(design, AuditDesign) and isinstance(development_design, nested.DevelopmentDesign),
            "frozen_audit_and_development_designs_required")
    require(not set(design.image_ids).intersection(development_design.image_ids)
            and not set(design.concept_ids).intersection(development_design.concept_ids),
            "audit_development_identity_overlap")
    development._mapping(predictors, models, "only_selected_pair_predictors_required")
    development._mapping(features, models, "only_selected_pair_features_required")
    development._mapping(responses, endpoint.GROUPS, "complete_audit_response_groups_required")
    order = tuple(block_order)
    require(len(order) == 12 and all(nested._label(label) for label in order)
            and len(set(order)) == 12 and set(order) == set(design.block_ids),
            "complete_frozen_audit_block_order_required")
    block_rows = tuple(np.array([i for i, label in enumerate(design.block_ids) if label == block], dtype=int)
                       for block in order)
    blocks = tuple(endpoint.ConceptBlock(block, tuple(design.image_ids[i] for i in rows),
                    tuple(design.concept_ids[i] for i in rows)) for block, rows in zip(order, block_rows))
    blocks = endpoint.bind_blocks(blocks, "audit")
    for group in endpoint.GROUPS:
        panel = responses[group]
        require(isinstance(panel, AuditResponsePanel) and panel.rows == design, "audit_response_row_binding_changed")
        require(np.array_equal(endpoint.column_ids(panel.column_ids, "audit_response_columns"),
                               family._supports[group].columns), "audit_original_column_binding_changed")
        development._mapping(panel.stages, endpoint.STAGES, "complete_audit_B_C_D_required")
    for model in models:
        panel = features[model]
        require(isinstance(panel, AuditFeaturePanel) and panel.rows == design, "audit_feature_row_binding_changed")
    return blocks, block_rows


def _pca_snapshot(fit, width, rank):
    require(isinstance(fit, prediction.PCAFit) and fit.rank == rank
            and fit.training_images == nested.DEVELOPMENT_IMAGES and fit.numerical_rank >= rank
            and fit.feature_mean.shape == (width,) and fit.loadings.shape == (width, rank)
            and np.isfinite(fit.score_scale) and fit.score_scale > 0,
            "frozen_all_development_PCA_changed")
    return nested._frozen_pca(fit)


def _freeze_predictor(wrapper, feature, development_design, family):
    require(isinstance(wrapper, FrozenAuditPredictor) and isinstance(wrapper.predictor, nested.FrozenPredictor),
            "bound_frozen_predictor_required")
    value = wrapper.predictor
    require(tuple(value.training_image_ids) == development_design.image_ids
            and tuple(value.training_concept_ids) == development_design.concept_ids,
            "all_development_training_roster_changed")
    require(value.setting in nested.SETTINGS and isinstance(value.selection, nested.Selection)
            and value.selection.setting == value.setting, "frozen_selected_setting_changed")
    require(feature.layer == value.setting.layer, "audit_selected_layer_changed")
    development._mapping(wrapper.column_ids, endpoint.GROUPS, "complete_coefficient_column_axes_required")
    development._mapping(value.ridges, endpoint.GROUPS, "complete_frozen_ridge_groups_required")
    width, rank = nested.FEATURE_WIDTHS[value.setting.layer], value.setting.rank
    pca, cka = _pca_snapshot(value.pca, width, rank), _pca_snapshot(value.cka_pca, width, 64)
    common_rank = min(rank, 64)
    require(np.array_equal(pca.feature_mean, cka.feature_mean)
            and np.array_equal(pca.loadings[:, :common_rank], cka.loadings[:, :common_rank]),
            "encoding_CKA_frozen_basis_changed")
    ridges = {}
    for group in endpoint.GROUPS:
        columns = endpoint.column_ids(wrapper.column_ids[group], "coefficient_columns")
        require(np.array_equal(columns, family._supports[group].columns), "coefficient_original_column_binding_changed")
        development._mapping(value.ridges[group], endpoint.STAGES, "complete_frozen_ridge_B_C_D_required")
        ridges[group] = {}
        for stage in endpoint.STAGES:
            fit = value.ridges[group][stage]
            require(isinstance(fit, prediction.RidgeFit) and fit.training_images == nested.DEVELOPMENT_IMAGES
                    and fit.penalty == value.setting.penalty and fit.score_mean.shape == (rank,)
                    and fit.target_mean.shape == (len(columns),) and fit.coefficients.shape == (rank, len(columns)),
                    "frozen_all_development_ridge_changed")
            ridges[group][stage] = nested._frozen_ridge(fit)
    return replace(value, pca=pca, cka_pca=cka, ridges=nested._nested_map(ridges))


def _block_sst(observed, mean):
    # Development's helper requires equal matrix shapes; this immutable
    # broadcast is a view of the stored mean, never an audit-fitted intercept.
    sums = development._sum_squared(observed, np.broadcast_to(mean, observed.shape))
    require(bool((sums > 0).all()), "audit_block_SST_nonpositive")
    return sums


def apply_frozen_audit(design, development_design, predictors, features, responses, relation, *,
                       block_order, secondary_checks, resource_check):
    """Return one frozen pair's audit bank and primary EndpointStatistic.

    This consumes supplied arrays only. ``secondary_checks`` is an explicit
    tuple (possibly empty) of at most two (scale,component) descriptors, not
    permission to choose or calculate extra checks. No secondary is executed.
    The existing relation must already be fully validated by the caller; this
    entry point checks its pair/family/roster, not its forecast or gate metadata.
    ``baselines[group][stage]`` is the stored 1D development target-mean vector
    common to all audit images; ``cka_features[model]`` has shape(606,64).
    ``responses`` retains the same immutable snapshots used for primary sums,
    so a later selected-secondary caller needs no replacement observation bank.
    """
    require(callable(resource_check), "external_resource_callback_required")
    nested._event(resource_check, "begin_audit_application")
    family, models, checks = _relation_layout(relation, secondary_checks)
    blocks, block_rows = _layout(design, development_design, predictors, features, responses,
                                 family, models, block_order)
    frozen = {model: nested._call(resource_check, "freeze_audit_predictor", _freeze_predictor,
        predictors[model], features[model], development_design, family, context=dict(model_id=model))
        for model in models}
    first, second = models
    baselines = {group: {} for group in endpoint.GROUPS}
    for group in endpoint.GROUPS:
        for stage in endpoint.STAGES:
            a, b = frozen[first].ridges[group][stage].target_mean, frozen[second].ridges[group][stage].target_mean
            require(np.array_equal(a, b), "paired_development_target_mean_changed")
            baselines[group][stage] = a
    baselines = nested._nested_map(baselines)
    # Qualify every frozen response column and block before applying a model.
    observed, sst = {group: {} for group in endpoint.GROUPS}, {group: {} for group in endpoint.GROUPS}
    for group in endpoint.GROUPS:
        width = len(family._supports[group].columns)
        for stage in endpoint.STAGES:
            context = dict(group=group, stage=stage, shape=(AUDIT_IMAGES, width))
            values = nested._call(resource_check, "snapshot_audit_response", nested._frozen_matrix,
                responses[group].stages[stage], AUDIT_IMAGES, width, "audit_response", context=context)
            observed[group][stage] = values
            nested._call(resource_check, "qualify_audit_response_variance", nested._qualify_response_variance,
                         values, context=context)
            nested._event(resource_check, "allocate_audit_block_sst", **{**context, "shape": (12, width)})
            sums = np.empty((12, width), dtype=np.float64)
            for index, rows in enumerate(block_rows):
                selected = nested._slice_rows(resource_check, "slice_audit_response", values, rows, context)
                nested._call(resource_check, "qualify_audit_block_moments", nested._qualify_primary_block_moments,
                             selected, context={**context, "shape": selected.shape})
                sums[index] = nested._call(resource_check, "audit_block_sst", _block_sst,
                    selected, baselines[group][stage], context=context)
            sst[group][stage] = nested._call(resource_check, "freeze_audit_block_sst",
                                            endpoint.frozen_array, sums, context=context)
    predicted, cka_features, sse = {}, {}, {}
    for model in models:
        fit = frozen[model]
        context = dict(model_id=model, layer=fit.setting.layer, rank=fit.setting.rank,
                       penalty=fit.setting.penalty)
        values = nested._call(resource_check, "snapshot_audit_features", nested._frozen_matrix,
            features[model].values, AUDIT_IMAGES, nested.FEATURE_WIDTHS[fit.setting.layer],
            "audit_features", context=context)
        scores = nested._call(resource_check, "apply_audit_pca", fit.pca.transform, values, context=context)
        transformed = nested._call(resource_check, "apply_audit_cka_pca", fit.cka_pca.transform,
                                   values, context={**context, "rank": 64})
        cka_features[model] = nested._call(resource_check, "freeze_audit_cka_features", nested._frozen_matrix,
            transformed, AUDIT_IMAGES, 64, "audit_cka_features", context={**context, "rank": 64})
        predicted[model], sse[model] = {}, {}
        for group in endpoint.GROUPS:
            predicted[model][group], sse[model][group] = {}, {}
            width = len(family._supports[group].columns)
            for stage in endpoint.STAGES:
                binding = {**context, "group": group, "stage": stage, "shape": (AUDIT_IMAGES, width)}
                prediction_values = nested._call(resource_check, "predict_audit",
                    fit.ridges[group][stage].predict, scores, context=binding)
                prediction_values = nested._call(resource_check, "freeze_audit_prediction", nested._frozen_matrix,
                    prediction_values, AUDIT_IMAGES, width, "audit_prediction", context=binding)
                predicted[model][group][stage] = prediction_values
                nested._event(resource_check, "allocate_audit_block_sse", **{**binding, "shape": (12, width)})
                sums = np.empty((12, width), dtype=np.float64)
                for index, rows in enumerate(block_rows):
                    sums[index] = nested._call(resource_check, "audit_block_sse", development._sum_squared,
                        nested._slice_rows(resource_check, "slice_audit_response", observed[group][stage], rows, binding),
                        nested._slice_rows(resource_check, "slice_audit_prediction", prediction_values, rows, binding),
                        context=binding)
                sse[model][group][stage] = nested._call(resource_check, "freeze_audit_block_sse",
                                                       endpoint.frozen_array, sums, context=binding)
    nested._event(resource_check, "bind_audit_errors")
    columns = MappingProxyType({group: family._supports[group].columns for group in endpoint.GROUPS})
    sse, sst = nested._nested_map(sse), nested._nested_map(sst)
    errors = MappingProxyType({group: endpoint.PairedBlockErrors(models, columns[group], blocks,
        sse[first][group], sse[second][group], sst[group]) for group in endpoint.GROUPS})
    statistic = nested._call(resource_check, "bind_audit_endpoint",
        lambda: family.bind("audit", errors, pair_id=relation.pair_id))
    result = AuditPredictionResult(design, blocks, relation.pair_id, columns, nested._nested_map(predicted),
        baselines, MappingProxyType(cka_features), errors, statistic, checks, nested._nested_map(observed))
    nested._event(resource_check, "complete_audit_application", "after")
    return result
