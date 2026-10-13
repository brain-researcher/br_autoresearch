"""Compose EP17 development prediction banks into its fixed primary endpoints.

This is the missing array bridge between nested_prediction and
endpoint_evaluator, under GOAL.md/SEARCH_POLICY.yaml and the adopted
prediction_qualification_specification.md and measurement_contract_specification.md.
It consumes already qualified, role-safe development arrays and an EXISTING
calibration-frozen EndpointFamily; it opens no files or audit inputs, fits no
model, computes no CKA and chooses no support, endpoint, pair or method freeze.

The caller supplies the SAME response panels used for nested fitting. Identity
metadata and matching training means cannot authenticate all response values or
prediction provenance. Complete source qualification, equal-repeat aggregation,
effective reader isolation, predictor binding and resource authority remain
caller obligations. Failure returns no partial comparison or deletion rescue.

The separately frozen block order is required, because it also identifies the
bootstrap draw coordinates and fixed beta-forecast block split. It must not be
inferred from first appearance in the image roster. Labels retain their types;
integer 1 and string '1' are distinct, with no string-casting collision.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction


class DevelopmentEndpointError(ValueError):
    """Refuse the complete development composition without revising its family."""


def require(condition, code):
    if not condition:
        raise DevelopmentEndpointError(code)


@dataclass(frozen=True)
class DevelopmentEndpoints:
    blocks: tuple
    model_sse: Mapping
    common_sst: Mapping
    paired_errors: Mapping
    statistics: Mapping


def _mapping(values, keys, code):
    require(isinstance(values, Mapping) and set(values) == set(keys), code)


def _label(value):
    return (isinstance(value, str) and bool(value)) or (
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)))


def _bind_layout(result, responses, family, block_order):
    require(isinstance(result, nested.NestedPredictionResult), "nested_prediction_result_required")
    require(isinstance(result.design, nested.DevelopmentDesign), "frozen_development_design_required")
    require(isinstance(family, endpoint.EndpointFamily), "frozen_endpoint_family_required")
    _mapping(responses, endpoint.GROUPS, "complete_response_groups_required")
    _mapping(result.columns, endpoint.GROUPS, "complete_prediction_groups_required")
    _mapping(result.baselines, endpoint.GROUPS, "complete_prediction_groups_required")
    _mapping(result.predictions, nested.MODEL_IDS, "complete_prediction_models_required")
    order = tuple(block_order)
    require(len(order) == 24 and all(_label(label) for label in order)
            and len(set(order)) == 24 and set(order) == set(result.design.block_ids),
            "complete_frozen_block_order_required")
    for model in nested.MODEL_IDS:
        _mapping(result.predictions[model], endpoint.GROUPS, "complete_prediction_groups_required")
        for group in endpoint.GROUPS:
            _mapping(result.predictions[model][group], endpoint.STAGES, "complete_B_C_D_required")
    for group in endpoint.GROUPS:
        panel = responses[group]
        require(isinstance(panel, nested.ResponsePanel) and panel.rows == result.design,
                "response_row_binding_changed")
        expected = family._supports[group].columns
        source = endpoint.column_ids(panel.column_ids, "response_columns")
        predicted = endpoint.column_ids(result.columns[group], "prediction_columns")
        require(np.array_equal(source, expected) and np.array_equal(predicted, expected),
                "original_column_binding_changed")
        _mapping(panel.stages, endpoint.STAGES, "complete_B_C_D_required")
        _mapping(result.baselines[group], endpoint.STAGES, "complete_B_C_D_required")
    indices = tuple(np.array([i for i, label in enumerate(result.design.block_ids) if label == block], dtype=int)
                    for block in order)
    blocks = tuple(endpoint.ConceptBlock(block,
                        tuple(result.design.image_ids[i] for i in rows),
                        tuple(result.design.concept_ids[i] for i in rows))
                   for block, rows in zip(order, indices))
    # Reuse the existing complete whole-concept/image checks before any score.
    return endpoint.bind_blocks(blocks, "development"), indices


def _snapshot(value, width, name):
    matrix = prediction.matrix(value, name)
    require(matrix.shape == (nested.DEVELOPMENT_IMAGES, width), name + "_shape_changed")
    return endpoint.frozen_array(matrix)


def _sum_squared(left, right):
    require(left.shape == right.shape, "residual_shape_changed")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = left - right
        sums = np.sum(residual * residual, axis=0, dtype=np.float64)
    require(np.isfinite(sums).all() and (sums >= 0).all(), "block_error_sums_invalid")
    return sums


def _mean(values):
    # Exactly the same float64 accumulation/constant-column convention and
    # original row order as nested_prediction._training_mean/fit_ridge.
    mean, _ = prediction.centered_training(values)
    return mean


def _baseline_matches(values, mean):
    require(bool((values == mean).all()), "outer_training_baseline_changed")


def _group_stage(result, responses, group, stage, block_indices, fold_indices, check):
    width = len(result.columns[group])
    context = dict(group=group, stage=stage, shape=(nested.DEVELOPMENT_IMAGES, width))
    # Snapshot only one group/stage, and later one model at a time. In
    # particular, do not duplicate the four complete OOF prediction banks.
    observed = nested._call(check, "snapshot_response", _snapshot, responses[group].stages[stage],
                            width, "response", context=context)
    baseline = nested._call(check, "snapshot_baseline", _snapshot, result.baselines[group][stage],
                            width, "baseline", context=context)
    for folds, training, heldout in fold_indices:
        binding = {**context, "training_folds": folds}
        mean = nested._call(check, "outer_training_mean", _mean,
            nested._slice_rows(check, "slice_responses", observed, training, binding), context=binding)
        nested._call(check, "verify_outer_baseline", _baseline_matches,
            nested._slice_rows(check, "slice_baselines", baseline, heldout, binding), mean, context=binding)
    nested._event(check, "allocate_block_sst", **{**context, "shape": (24, width)})
    sst = np.empty((24, width), dtype=np.float64)
    for index, rows in enumerate(block_indices):
        sst[index] = nested._call(check, "block_sst", _sum_squared,
            nested._slice_rows(check, "slice_responses", observed, rows, context),
            nested._slice_rows(check, "slice_baselines", baseline, rows, context),
            context={**context, "shape": (len(rows), width)})
        require(bool((sst[index] > 0).all()), "block_SST_nonpositive")
    sst = nested._call(check, "freeze_block_sst", endpoint.frozen_array, sst,
                       context={**context, "shape": (24, width)})
    model_sse = {}
    for model in nested.MODEL_IDS:
        binding = {**context, "model_id": model}
        predicted = nested._call(check, "snapshot_prediction", _snapshot,
            result.predictions[model][group][stage], width, "prediction", context=binding)
        nested._event(check, "allocate_block_sse", **{**binding, "shape": (24, width)})
        sse = np.empty((24, width), dtype=np.float64)
        for index, rows in enumerate(block_indices):
            sse[index] = nested._call(check, "block_sse", _sum_squared,
                nested._slice_rows(check, "slice_responses", observed, rows, binding),
                nested._slice_rows(check, "slice_predictions", predicted, rows, binding),
                context={**binding, "shape": (len(rows), width)})
        model_sse[model] = nested._call(check, "freeze_block_sse", endpoint.frozen_array, sse,
                                      context={**binding, "shape": (24, width)})
        del predicted
    return model_sse, sst


def compose_development_endpoints(result, responses, family, *, block_order, resource_check):
    """Return all four fixed signed pairs, with shared block error sums.

    ``block_order`` is the caller's already frozen 24-label metadata order.
    ``responses`` must have the same complete DevelopmentDesign and original
    A_all columns as the prediction bank/family. Common baselines are verified
    against other-five-fold means and then used as stored, never repaired.
    The existing EndpointFamily alone applies support/ceiling/H operators and
    its registered pooled-R²/ROI/participant calculations. No uncertainty
    bootstrap, candidate selection, forecasting fit or audit opening occurs.
    """
    require(callable(resource_check), "external_resource_callback_required")
    nested._event(resource_check, "begin")
    blocks, block_indices = _bind_layout(result, responses, family, block_order)
    fold_indices = tuple((tuple(other for other in result.design.fold_labels if other != fold),
                          np.array([i for i, label in enumerate(result.design.fold_ids) if label != fold], dtype=int),
                          np.array([i for i, label in enumerate(result.design.fold_ids) if label == fold], dtype=int))
                         for fold in result.design.fold_labels)
    model_sse = {model: {group: {} for group in endpoint.GROUPS} for model in nested.MODEL_IDS}
    common_sst = {group: {} for group in endpoint.GROUPS}
    for group in endpoint.GROUPS:
        for stage in endpoint.STAGES:
            sums, common_sst[group][stage] = _group_stage(result, responses, group, stage,
                block_indices, fold_indices, resource_check)
            for model in nested.MODEL_IDS:
                model_sse[model][group][stage] = sums[model]
    model_sse, common_sst = nested._nested_map(model_sse), nested._nested_map(common_sst)
    errors, statistics = {}, {}
    for pair, model_order in endpoint.PAIR_SIGNS.items():
        first, second = model_order
        paired = MappingProxyType({group: endpoint.PairedBlockErrors(model_order,
            family._supports[group].columns, blocks, model_sse[first][group], model_sse[second][group],
            common_sst[group]) for group in endpoint.GROUPS})
        errors[pair] = paired
        # Endpoint binding snapshots only the small block-by-voxel sums. It
        # recomputes neither prediction banks nor per-model SSE/common SST.
        statistics[pair] = nested._call(resource_check, "bind_endpoint_pair",
            lambda: family.bind("development", paired, pair_id=pair))
    value = DevelopmentEndpoints(blocks, model_sse, common_sst,
                                 MappingProxyType(errors), MappingProxyType(statistics))
    nested._event(resource_check, "complete", "after")
    return value
