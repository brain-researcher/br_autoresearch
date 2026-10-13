"""EP17's fixed nested development prediction bank, on supplied arrays only.

Implements prediction_qualification_specification.md, 'Adopted common linking
method', with the model_pair_registry_specification.md feature roster. The
caller supplies the existing frozen development assignment and calibration
A_all columns, complete equal-repetition response means and role-safe features.
Matching metadata/counts cannot authenticate eligibility, preprocessing, equal
repeat aggregation, source identity, exposure qualification or reader isolation.
This module is not a reader, a resource grant, a method-freeze writer or an
audit opener. No calibration/audit response or audit feature input is accepted.

All four models, 84 settings and all required partitions are mandatory. Feature
feasibility for every partition precedes neural selection. A failed resource
callback or numerical operation aborts the entire call: no reduced grid,
participant/voxel deletion or partially qualified result is returned.

Outputs retain one B/C/D OOF prediction bank per model, original row/column
bindings and one shared per-image development-training baseline per group/stage.
Support and ceiling operations occur downstream on the SAME D bank. Endpoint
inference, CKA scoring, predictor persistence and audit application are separate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

import calibration_core as calibration
import endpoint_evaluator as endpoint
import prediction_core as core

MODEL_IDS = ("DINO_R50", "SWAV_R50_800", "SUP_R50_V1", "RANDOM_R50_S17")
LAYERS = core.LAYERS
PCA_RANKS = core.PCA_RANKS
RIDGE_PENALTIES = core.RIDGE_PENALTIES
FEATURE_WIDTHS = MappingProxyType(dict(zip(LAYERS, (256, 512, 1024, 2048))))
GROUPS, STAGES = endpoint.GROUPS, endpoint.STAGES
DEVELOPMENT_IMAGES = 2423


class NestedPredictionError(ValueError):
    """Whole-configuration refusal; no selective rescue or partial readiness."""


def require(condition, code):
    if not condition:
        raise NestedPredictionError(code)


@dataclass(frozen=True)
class Setting:
    layer: str
    rank: int
    penalty: float


SETTINGS = tuple(Setting(layer, rank, penalty) for layer in LAYERS
                 for rank in PCA_RANKS for penalty in RIDGE_PENALTIES)


def _label(value):
    return (isinstance(value, str) and bool(value)) or (
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)))


@dataclass(frozen=True)
class DevelopmentDesign:
    """Complete caller-frozen reference roster, in its original image order.

    Fold/block labels are preserved, not renumbered. Their iteration order is
    first occurrence in the supplied frozen image roster, independent of scores.
    The known 2423-image count is necessary, not proof of authentic identities.
    """
    image_ids: tuple
    concept_ids: tuple
    roles: tuple
    fold_ids: tuple
    block_ids: tuple

    def __post_init__(self):
        for field in ("image_ids", "concept_ids", "roles", "fold_ids", "block_ids"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        require(all(len(getattr(self, field)) == DEVELOPMENT_IMAGES for field in
                    ("image_ids", "concept_ids", "roles", "fold_ids", "block_ids")),
                "complete_2423_development_images_required")
        require(all(isinstance(value, str) and value for value in self.image_ids + self.concept_ids),
                "image_concept_identity_invalid")
        require(len(set(self.image_ids)) == DEVELOPMENT_IMAGES, "duplicate_image_identity")
        require(set(self.roles) == {"development"}, "development_role_only")
        require(all(_label(value) for value in self.fold_ids + self.block_ids), "fold_block_label_invalid")
        concept_assignment = {}
        for concept, fold, block in zip(self.concept_ids, self.fold_ids, self.block_ids):
            require(concept not in concept_assignment or concept_assignment[concept] == (fold, block),
                    "concept_crosses_fold_or_block")
            concept_assignment[concept] = (fold, block)
        require(len(concept_assignment) == 480, "all_480_development_concepts_required")
        require(len(self.fold_labels) == 6 and all(sum(f == label for f, _ in concept_assignment.values()) == 80
                    for label in self.fold_labels), "six_80_concept_folds_required")
        require(len(self.block_labels) == 24 and all(sum(b == label for _, b in concept_assignment.values()) == 20
                    for label in self.block_labels), "twenty_four_20_concept_blocks_required")

    @property
    def fold_labels(self):
        return tuple(dict.fromkeys(self.fold_ids))

    @property
    def block_labels(self):
        return tuple(dict.fromkeys(self.block_ids))


@dataclass(frozen=True)
class FeaturePanel:
    rows: DevelopmentDesign
    layers: Mapping


@dataclass(frozen=True)
class ResponsePanel:
    rows: DevelopmentDesign
    column_ids: np.ndarray
    stages: Mapping


@dataclass(frozen=True)
class ResourceEvent:
    """Public operation metadata only; no response values or model scores.

    A required external callback receives before/after events and must raise
    when the actual allocation, deadline or memory budget no longer permits
    work. It owns accounting/termination; a callback is not physical isolation.
    No timeout thread, scheduler access or hidden resource default is created.
    """
    operation: str
    when: str
    model_id: str | None = None
    training_folds: tuple = ()
    layer: str | None = None
    rank: int | None = None
    penalty: float | None = None
    group: tuple | None = None
    stage: str | None = None
    shape: tuple = ()


def _event(check, operation, when="before", **context):
    check(ResourceEvent(operation, when, **context))


def _call(check, operation, function, *args, context=None):
    context = {} if context is None else context
    _event(check, operation, **context)
    value = function(*args)
    _event(check, operation, "after", **context)
    return value


def _slice_rows(check, operation, values, rows, context=None):
    """Obtain a potentially large fancy-index copy only after admission."""
    context = {**({} if context is None else context), "shape": (len(rows), values.shape[1])}
    _event(check, operation, **context)
    selected = values[rows]
    _event(check, operation, "after", **context)
    return selected


def _frozen_matrix(value, rows, columns, name):
    matrix = core.matrix(value, name)
    require(matrix.shape == (rows, columns), name + "_shape_changed")
    return endpoint.frozen_array(matrix)


def _frozen_pca(fit):
    require(isinstance(fit, core.PCAFit), "PCA_fit_required")
    return replace(fit, feature_mean=endpoint.frozen_array(fit.feature_mean),
                   loadings=endpoint.frozen_array(fit.loadings),
                   singular_values=endpoint.frozen_array(fit.singular_values))


def _frozen_ridge(fit):
    require(isinstance(fit, core.RidgeFit), "ridge_fit_required")
    return replace(fit, score_mean=endpoint.frozen_array(fit.score_mean),
                   target_mean=endpoint.frozen_array(fit.target_mean),
                   coefficients=endpoint.frozen_array(fit.coefficients))


def derive_rank_pca(maximum_fit, training_features, rank):
    """Reuse a matching full-SVD fit; recompute the rank-specific score scale.

    Internal composition requires the same training rows that produced the
    maximum fit. All three exact cutoffs are checked before neural objectives;
    this cannot replace a failed 128-dimensional fit with a smaller rank.
    """
    rank = core.valid_rank(rank)
    require(isinstance(maximum_fit, core.PCAFit) and maximum_fit.rank == max(PCA_RANKS),
            "full_maximum_rank_PCA_required")
    features = core.matrix(training_features, "training_features", minimum_rows=2)
    require(features.shape == (maximum_fit.training_images, maximum_fit.feature_mean.size),
            "PCA_training_shape_changed")
    singular = maximum_fit.singular_values
    require(rank <= maximum_fit.numerical_rank, "PCA_rank_unavailable")
    require(rank == len(singular) or singular[rank - 1] != singular[rank],
            "PCA_cutoff_exactly_degenerate")
    if rank == maximum_fit.rank:
        return maximum_fit
    loadings = maximum_fit.loadings[:, :rank]
    scores = (features - maximum_fit.feature_mean) @ loadings
    energy = float(np.sum(scores * scores, dtype=np.float64))
    require(math.isfinite(energy) and energy > 0, "PCA_score_energy_invalid")
    scale = math.sqrt(features.shape[0] / energy)
    require(math.isfinite(scale) and scale > 0, "PCA_score_scale_invalid")
    return core.PCAFit(maximum_fit.feature_mean, loadings, scale, singular,
                       maximum_fit.numerical_rank, maximum_fit.training_images)


def pooled_inner_objective(partitions):
    """Pooled voxel R², then voxel→ROI→participant→stage equal means.

    ``partitions[(person, roi, stage)]`` supplies five inner ErrorSums for an
    outer selection, or six for the all-development selection. No averaging
    of fold R², weighting by group voxel counts or clipping is performed.
    """
    keys = tuple((*group, stage) for group in GROUPS for stage in STAGES)
    require(isinstance(partitions, Mapping) and set(partitions) == set(keys), "complete_inner_panel_required")
    counts = {len(partitions[key]) for key in keys}
    require(len(counts) == 1 and next(iter(counts)) in (5, 6), "five_or_six_inner_partitions_required")
    values = {key: float(np.mean(core.pooled_r2(partitions[key]), dtype=np.float64)) for key in keys}
    stage_values = []
    for stage in STAGES:
        person_values = [float(np.mean([values[person, roi, stage] for roi in endpoint.ROIS], dtype=np.float64))
                         for person in endpoint.SUBJECTS]
        stage_values.append(float(np.mean(person_values, dtype=np.float64)))
    result = float(np.mean(stage_values, dtype=np.float64))
    require(math.isfinite(result), "inner_objective_nonfinite")
    return result


@dataclass(frozen=True)
class Selection:
    setting: Setting
    objectives: Mapping


@dataclass(frozen=True)
class FrozenPredictor:
    """All-development fit only; no audit-application or persistence authority."""
    setting: Setting
    pca: core.PCAFit
    cka_pca: core.PCAFit
    ridges: Mapping
    selection: Selection
    training_image_ids: tuple
    training_concept_ids: tuple


@dataclass(frozen=True)
class NestedPredictionResult:
    design: DevelopmentDesign
    columns: Mapping
    predictions: Mapping
    baselines: Mapping
    outer_selections: Mapping
    outer_cka_transforms: Mapping
    final_predictors: Mapping


def _nested_map(values):
    return MappingProxyType({key: _nested_map(value) if isinstance(value, dict) else value
                             for key, value in values.items()})


def _bind_inputs(design, features, responses, frozen_columns, check):
    require(isinstance(design, DevelopmentDesign), "frozen_development_design_required")
    require(isinstance(features, Mapping) and set(features) == set(MODEL_IDS), "all_four_registered_models_required")
    require(isinstance(responses, Mapping) and set(responses) == set(GROUPS), "all_eight_response_groups_required")
    require(isinstance(frozen_columns, Mapping) and set(frozen_columns) == set(GROUPS),
            "all_eight_frozen_column_axes_required")
    bound_features, bound_responses, columns = {}, {}, {}
    for model in MODEL_IDS:
        panel = features[model]
        require(isinstance(panel, FeaturePanel) and panel.rows == design, "feature_row_binding_changed")
        require(isinstance(panel.layers, Mapping) and set(panel.layers) == set(LAYERS), "all_four_feature_layers_required")
        bound_features[model] = {}
        for layer in LAYERS:
            context = dict(model_id=model, layer=layer, shape=(DEVELOPMENT_IMAGES, FEATURE_WIDTHS[layer]))
            bound_features[model][layer] = _call(check, "snapshot_features", _frozen_matrix,
                panel.layers[layer], DEVELOPMENT_IMAGES, FEATURE_WIDTHS[layer], "features", context=context)
    for group in GROUPS:
        panel = responses[group]
        require(isinstance(panel, ResponsePanel) and panel.rows == design, "response_row_binding_changed")
        columns[group] = endpoint.column_ids(frozen_columns[group], "frozen_A_all")
        supplied = endpoint.column_ids(panel.column_ids, "response_columns")
        require(np.array_equal(supplied, columns[group]), "original_column_binding_changed")
        require(isinstance(panel.stages, Mapping) and set(panel.stages) == set(STAGES), "all_three_response_stages_required")
        bound_responses[group] = {}
        for stage in STAGES:
            context = dict(group=group, stage=stage, shape=(DEVELOPMENT_IMAGES, len(columns[group])))
            bound_responses[group][stage] = _call(check, "snapshot_responses", _frozen_matrix,
                panel.stages[stage], DEVELOPMENT_IMAGES, len(columns[group]), "responses", context=context)
    return bound_features, bound_responses, columns


def _row_indices(design):
    folds = design.fold_labels
    training = {subset: np.array([i for i, label in enumerate(design.fold_ids) if label in subset], dtype=int)
                for count in (4, 5, 6) for subset in combinations(folds, count)}
    validation = {label: np.array([i for i, fold in enumerate(design.fold_ids) if fold == label], dtype=int)
                  for label in folds}
    blocks = {label: np.array([i for i, block in enumerate(design.block_ids) if block == label], dtype=int)
              for label in design.block_labels}
    return training, validation, blocks


def _feature_feasibility(features, training, check):
    bank = {}
    for model in MODEL_IDS:
        for folds, rows in training.items():
            for layer in LAYERS:
                context = dict(model_id=model, training_folds=folds, layer=layer,
                               rank=max(PCA_RANKS), shape=(len(rows), FEATURE_WIDTHS[layer]))
                values = _slice_rows(check, "slice_features", features[model][layer], rows, context)
                maximum = _call(check, "fit_pca", core.fit_pca, values, max(PCA_RANKS), context=context)
                maximum = _call(check, "freeze_pca", _frozen_pca, maximum, context=context)
                bank[model, folds, layer] = {}
                for rank in PCA_RANKS:
                    bank[model, folds, layer][rank] = _call(check, "derive_rank_pca", derive_rank_pca,
                        maximum, values, rank, context={**context, "rank": rank})
    _event(check, "all_features_feasible", "after")
    return bank


def _qualify_response_variance(values):
    require(bool(calibration.qualify_variance(values).valid.all()), "response_partition_variance_invalid")


def _qualify_primary_block_moments(values):
    """Adopted Oct12 exception: primary block SST, not centered variance.

    Keep the original finite-input and computed-moment checks. This helper
    alone does not admit a block: its original positive finite training-mean
    SST check is also mandatory. Training/validation and secondary statistics
    continue to use their unchanged variance rules.
    """
    qualification = calibration.qualify_variance(values)
    require(bool(qualification.finite_inputs.all() and qualification.finite_moments.all()),
            "primary_block_response_moments_invalid")


def _training_mean(values):
    mean, _ = core.centered_training(values)
    return endpoint.frozen_array(mean)


def _qualify_sst(values, mean):
    sums = core.error_sums(values, values, mean)
    require(bool((sums.sst > 0).all()), "response_partition_SST_invalid")


def _prepare_responses(design, responses, training, validation, blocks, check):
    means = {folds: {group: {} for group in GROUPS} for folds in training}
    baselines = {group: {} for group in GROUPS}
    for group in GROUPS:
        for stage in STAGES:
            values = responses[group][stage]
            common = dict(group=group, stage=stage)
            for folds, rows in training.items():
                context = {**common, "training_folds": folds, "shape": (len(rows), values.shape[1])}
                selected = _slice_rows(check, "slice_responses", values, rows, context)
                _call(check, "qualify_response_variance", _qualify_response_variance, selected, context=context)
                means[folds][group][stage] = _call(check, "training_mean", _training_mean, selected, context=context)
            for rows in validation.values():
                _call(check, "qualify_response_variance", _qualify_response_variance,
                      _slice_rows(check, "slice_responses", values, rows, common),
                      context={**common, "shape": (len(rows), values.shape[1])})
            for rows in blocks.values():
                _call(check, "qualify_primary_block_moments", _qualify_primary_block_moments,
                      _slice_rows(check, "slice_responses", values, rows, common),
                      context={**common, "shape": (len(rows), values.shape[1])})
            for folds in training:
                for heldout, rows in validation.items():
                    if heldout not in folds:
                        _call(check, "qualify_scoring_SST", _qualify_sst,
                              _slice_rows(check, "slice_responses", values, rows,
                                          {**common, "training_folds": folds}), means[folds][group][stage],
                              context={**common, "training_folds": folds, "shape": (len(rows), values.shape[1])})
            _event(check, "allocate_shared_baseline", **common, shape=values.shape)
            baseline = np.empty_like(values)
            for heldout, rows in validation.items():
                folds = tuple(label for label in design.fold_labels if label != heldout)
                baseline[rows] = means[folds][group][stage]
            # Each uncertainty block can cross outer folds. Its SST must use
            # every image's matching outer-training mean, never a block mean.
            for rows in blocks.values():
                _call(check, "qualify_block_SST", _qualify_sst,
                      _slice_rows(check, "slice_responses", values, rows, common),
                      _slice_rows(check, "slice_baselines", baseline, rows, common),
                      context={**common, "shape": (len(rows), values.shape[1])})
            baselines[group][stage] = _call(check, "freeze_shared_baseline", endpoint.frozen_array,
                                           baseline, context={**common, "shape": baseline.shape})
    return means, baselines


def _select(model, selection_folds, features, responses, training, validation, pcas, means, check):
    objectives = {}
    for layer in LAYERS:
        for rank in PCA_RANKS:
            scores = {}
            for heldout in selection_folds:
                folds = tuple(label for label in selection_folds if label != heldout)
                pca = pcas[model, folds, layer][rank]
                context = dict(model_id=model, training_folds=folds, layer=layer, rank=rank)
                train_rows, test_rows = training[folds], validation[heldout]
                train_scores = _call(check, "transform_training", pca.transform,
                                     _slice_rows(check, "slice_features", features[model][layer], train_rows, context),
                                     context={**context, "shape": (len(train_rows), FEATURE_WIDTHS[layer])})
                test_scores = _call(check, "transform_validation", pca.transform,
                                    _slice_rows(check, "slice_features", features[model][layer], test_rows, context),
                                    context={**context, "shape": (len(test_rows), FEATURE_WIDTHS[layer])})
                scores[heldout] = (folds, train_scores, test_scores)
            for penalty in RIDGE_PENALTIES:
                setting = Setting(layer, rank, penalty)
                partitions = {(*group, stage): [] for group in GROUPS for stage in STAGES}
                for heldout in selection_folds:
                    folds, train_scores, test_scores = scores[heldout]
                    train_rows, test_rows = training[folds], validation[heldout]
                    for group in GROUPS:
                        for stage in STAGES:
                            values = responses[group][stage]
                            context = dict(model_id=model, training_folds=folds, layer=layer, rank=rank,
                                           penalty=penalty, group=group, stage=stage)
                            fit = _call(check, "fit_ridge", core.fit_ridge, train_scores,
                                        _slice_rows(check, "slice_responses", values, train_rows, context), penalty,
                                        context={**context, "shape": (len(train_rows), values.shape[1])})
                            require(np.array_equal(fit.target_mean, means[folds][group][stage]),
                                    "model_specific_training_baseline_changed")
                            predicted = _call(check, "predict_validation", fit.predict, test_scores,
                                              context={**context, "shape": (len(test_rows), values.shape[1])})
                            sums = _call(check, "inner_error_sums", core.error_sums,
                                         _slice_rows(check, "slice_responses", values, test_rows, context),
                                         predicted, means[folds][group][stage], context=context)
                            partitions[(*group, stage)].append(sums)
                objectives[setting] = _call(check, "inner_objective", pooled_inner_objective, partitions,
                    context=dict(model_id=model, training_folds=selection_folds, layer=layer, rank=rank, penalty=penalty))
    require(set(objectives) == set(SETTINGS), "full_registered_grid_not_completed")
    winner = min(SETTINGS, key=lambda setting: (-objectives[setting],
                 core.hyperparameter_tie_key(setting.layer, setting.rank, setting.penalty)))
    return Selection(winner, MappingProxyType(objectives))


def _refit(model, folds, selection, features, responses, training, pcas, means, check,
           *, test_rows=None, destination=None):
    setting = selection.setting
    pca = pcas[model, folds, setting.layer][setting.rank]
    rows = training[folds]
    context = dict(model_id=model, training_folds=folds, layer=setting.layer,
                   rank=setting.rank, penalty=setting.penalty)
    scores = _call(check, "transform_refit_training", pca.transform,
                   _slice_rows(check, "slice_features", features[model][setting.layer], rows, context),
                   context={**context, "shape": (len(rows), FEATURE_WIDTHS[setting.layer])})
    test_scores = None if test_rows is None else _call(check, "transform_outer_heldout", pca.transform,
        _slice_rows(check, "slice_features", features[model][setting.layer], test_rows, context),
        context={**context, "shape": (len(test_rows), FEATURE_WIDTHS[setting.layer])})
    ridges = {group: {} for group in GROUPS}
    for group in GROUPS:
        for stage in STAGES:
            values = responses[group][stage]
            binding = {**context, "group": group, "stage": stage, "shape": (len(rows), values.shape[1])}
            fit = _call(check, "fit_ridge", core.fit_ridge, scores,
                        _slice_rows(check, "slice_responses", values, rows, binding), setting.penalty, context=binding)
            require(np.array_equal(fit.target_mean, means[folds][group][stage]), "model_specific_training_baseline_changed")
            if test_rows is None:
                ridges[group][stage] = _call(check, "freeze_ridge", _frozen_ridge, fit, context=binding)
            else:
                predicted = _call(check, "predict_outer_heldout", fit.predict, test_scores,
                                  context={**binding, "shape": (len(test_rows), values.shape[1])})
                require(predicted.shape == (len(test_rows), values.shape[1]) and np.isfinite(predicted).all(),
                        "outer_prediction_missing_or_nonfinite")
                destination[group][stage][test_rows] = predicted
    return pca, pcas[model, folds, setting.layer][64], _nested_map(ridges)


def run_nested_prediction(design, features, responses, frozen_columns, *, resource_check):
    """Build the complete registered development bank and final predictors.

    The required callback may raise at any event. No function argument changes
    the model menu, grid, participants, image count or fold/block counts. All
    result arrays are immutable snapshots, with one common baseline rather than
    a repeated model-specific copy. No files are read or written here.
    """
    require(callable(resource_check), "external_resource_callback_required")
    _event(resource_check, "begin")
    features, responses, columns = _bind_inputs(design, features, responses, frozen_columns, resource_check)
    training, validation, blocks = _row_indices(design)
    pcas = _feature_feasibility(features, training, resource_check)
    means, baselines = _prepare_responses(design, responses, training, validation, blocks, resource_check)
    predictions, outer_selections, outer_cka, final_predictors = {}, {}, {}, {}
    for model in MODEL_IDS:
        bank = {group: {} for group in GROUPS}
        for group in GROUPS:
            for stage in STAGES:
                _event(resource_check, "allocate_OOF_bank", model_id=model, group=group, stage=stage,
                       shape=responses[group][stage].shape)
                bank[group][stage] = np.full_like(responses[group][stage], np.nan)
        outer_selections[model], outer_cka[model] = {}, {}
        for heldout, test_rows in validation.items():
            folds = tuple(label for label in design.fold_labels if label != heldout)
            selected = _select(model, folds, features, responses, training, validation, pcas, means, resource_check)
            _, cka, _ = _refit(model, folds, selected, features, responses, training, pcas, means,
                               resource_check, test_rows=test_rows, destination=bank)
            outer_selections[model][heldout], outer_cka[model][heldout] = selected, cka
        for group in GROUPS:
            for stage in STAGES:
                bank[group][stage] = _call(resource_check, "freeze_OOF_bank", endpoint.frozen_array,
                    bank[group][stage], context=dict(model_id=model, group=group, stage=stage,
                                                     shape=bank[group][stage].shape))
        predictions[model] = bank
        all_folds = design.fold_labels
        selected = _select(model, all_folds, features, responses, training, validation, pcas, means, resource_check)
        pca, cka, ridges = _refit(model, all_folds, selected, features, responses, training, pcas, means, resource_check)
        final_predictors[model] = FrozenPredictor(selected.setting, pca, cka, ridges, selected,
                                                 design.image_ids, design.concept_ids)
    result = NestedPredictionResult(design, MappingProxyType(columns), _nested_map(predictions),
        _nested_map(baselines), _nested_map(outer_selections), _nested_map(outer_cka), MappingProxyType(final_predictors))
    _event(resource_check, "complete", "after")
    return result
