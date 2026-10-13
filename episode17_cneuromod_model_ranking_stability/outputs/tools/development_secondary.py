"""EP17's separate development Fisher-z and linear-CKA families, arrays only.

Implements measurement_contract_specification.md's secondary scales and the
prospective 'Development secondary resampling composition — October 9' in
prediction_qualification_specification.md. No fitting, support selection,
ceiling division, audit application, reader or final scientific gate occurs.
All arrays are supplied by the caller under the existing development boundary.
Metadata checks cannot authenticate a feature/response source, the original
bank freeze or effective reader isolation. Resource admission remains external.

Whole-block multiplicities represent literal image-row replication. Fisher-z
is computed per voxel on pooled OOF rows before fixed support averaging. CKA
uses each outer fold's encoding-selected layer and frozen rank64 transform,
weighted centering, then resampled image-mass fold averaging. All 32 anatomical
subset CKAs are averaged, never replaced by CKA of mean-weighted columns.
Zero-mass people/folds contribute zero; every positive-mass statistic must be
defined. Separate complete families use uncertainty_core's fixed 10000 draws.

Binding retains immutable snapshots of the response and four prediction banks
plus transformed heldout features. This is deliberately an admitted-memory
operation, not a claim that any particular allocation can hold/run the family.
Each statistic recomputes the exact weighted kernels; no runtime is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

import calibration_core as calibration
import development_endpoints as development
import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction
import support_core as support
import uncertainty_core as uncertainty

COMPONENTS = ("B_Aall", "C_Aall", "D_Aall", "D_AN", "D_RN", "D_VNC")
SCALES = ("heldout_correlation", "linear_CKA")


class DevelopmentSecondaryError(ValueError):
    """Refuse the complete affected family without deleting or replacing data."""


def require(condition, code):
    if not condition:
        raise DevelopmentSecondaryError(code)


def _weights(value, count):
    raw = np.asarray(value)
    require(raw.dtype.kind in "iuf" and raw.shape == (count,), "image_multiplicities_invalid")
    weights = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(weights).all() and (weights >= 0).all()
            and (weights == np.floor(weights)).all()
            and np.isfinite(weights.sum()) and weights.sum() > 0,
            "image_multiplicities_empty_or_invalid")
    return weights


def _centered(value, weights):
    # Positive rows are precisely the rows present in literal replication.
    selected = value[weights > 0]
    mass = weights[weights > 0]
    total = mass.sum(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.sum(selected * mass[:, None], axis=0, dtype=np.float64) / total
        constant = np.all(selected == selected[0], axis=0)
        mean[constant] = selected[0, constant]
        centered = selected - mean
    require(np.isfinite(mean).all() and np.isfinite(centered).all(), "weighted_centering_nonfinite")
    return selected, centered, mass


def weighted_fisher_correlation(observed, predicted, image_weights):
    """Per-voxel Fisher-z; centered float64, fixed variance/Fisher conventions."""
    observed = prediction.matrix(observed, "secondary_observed")
    predicted = prediction.matrix(predicted, "secondary_predicted")
    require(observed.shape == predicted.shape, "secondary_prediction_shape_changed")
    weights = _weights(image_weights, len(observed))
    first, x, mass = _centered(observed, weights)
    second, y, _ = _centered(predicted, weights)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xx = np.sum(mass[:, None] * x * x, axis=0, dtype=np.float64)
        yy = np.sum(mass[:, None] * y * y, axis=0, dtype=np.float64)
        xy = np.sum(mass[:, None] * x * y, axis=0, dtype=np.float64)
        raw_xx = np.sum(mass[:, None] * first * first, axis=0, dtype=np.float64)
        raw_yy = np.sum(mass[:, None] * second * second, axis=0, dtype=np.float64)
        valid = (np.isfinite(xx) & np.isfinite(yy) & np.isfinite(xy)
                 & np.isfinite(raw_xx) & np.isfinite(raw_yy)
                 & (xx > 0) & (yy > 0) & (xx > 1e-12 * raw_xx) & (yy > 1e-12 * raw_yy))
        require(bool(valid.all()), "prediction_correlation_variance_undefined")
        correlations = (xy / np.sqrt(xx)) / np.sqrt(yy)
    transformed = calibration.fisher_transform(correlations)
    require(bool(transformed.valid.all()) and np.isfinite(transformed.fisher_z).all(),
            "prediction_correlation_undefined")
    return transformed.fisher_z


def weighted_linear_cka(features, responses, image_weights):
    """Centered linear CKA equal to explicit row replication; no voxel scaling."""
    features = prediction.matrix(features, "secondary_features")
    responses = prediction.matrix(responses, "secondary_responses")
    require(len(features) == len(responses), "CKA_image_binding_changed")
    weights = _weights(image_weights, len(features))
    _, x, mass = _centered(features, weights)
    _, y, _ = _centered(responses, weights)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        root_mass = np.sqrt(mass)[:, None]
        x = x * root_mass
        y = y * root_mass
        cross = x.T @ y
        x_gram = x.T @ x
        # Frobenius norms of Y'Y and YY' agree; choose smaller storage without
        # any normalization or approximation of delivered response columns.
        y_gram = y.T @ y if y.shape[1] <= y.shape[0] else y @ y.T
        numerator = np.sum(cross * cross, dtype=np.float64)
        x_norm = np.sqrt(np.sum(x_gram * x_gram, dtype=np.float64))
        y_norm = np.sqrt(np.sum(y_gram * y_gram, dtype=np.float64))
        denominator = x_norm * y_norm
        require(np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0,
                "CKA_denominator_undefined")
        value = float(numerator / denominator)
    require(np.isfinite(value), "CKA_nonfinite")
    return value


def _slice_columns(check, values, columns, context):
    nested._event(check, "slice_secondary_columns", **{**context, "shape": (len(values), len(columns))})
    result = values[:, columns]
    nested._event(check, "slice_secondary_columns", "after", **context)
    return result


def mean_bank_cka(features, responses, image_weights, bank_positions, *, resource_check, context=None):
    """Arithmetic mean of every admitted bank's whole-support CKA."""
    require(callable(resource_check), "external_resource_callback_required")
    require(len(bank_positions) == len(support.BANKS), "all_32_anatomical_banks_required")
    context = {} if context is None else context
    values = []
    for columns in bank_positions:
        require(len(columns) == support.N, "anatomical_bank_size_changed")
        selected = _slice_columns(resource_check, responses, columns, context)
        values.append(nested._call(resource_check, "weighted_linear_cka", weighted_linear_cka,
                                  features, selected, image_weights, context=context))
    return float(np.mean(values, dtype=np.float64))


def _bind_banks(family, supplied):
    development._mapping(supplied, endpoint.GROUPS, "complete_anatomical_bank_groups_required")
    output = {}
    for group in endpoint.GROUPS:
        bank, fixed = supplied[group], family._supports[group]
        require(isinstance(bank, support.AnatomicalBank)
                and (bank.participant, bank.roi) == group, "anatomical_bank_group_changed")
        ids = endpoint.column_ids(bank.column_ids, "anatomical_structural_columns")
        require(bool((ids < support.PARTICIPANT_COLUMNS[group[0]]).all()),
                "anatomical_structural_columns_outside_original_axis")
        require(len(bank.spatial_bins) == len(ids) and all(b in support.BINS for b in bank.spatial_bins),
                "anatomical_spatial_bins_invalid")
        eligible = endpoint.column_ids(bank.a_all_ids, "anatomical_A_all")
        require(set(eligible) == set(fixed.columns) and set(eligible).issubset(set(ids)),
                "anatomical_A_all_changed")
        structural = {int(ident): index for index, ident in enumerate(ids)}
        lookup = {int(ident): index for index, ident in enumerate(fixed.columns)}
        require(all(bank.spatial_bins[structural[int(ident)]] == support.BINS[int(bin_id) - 1]
                    for ident, bin_id in zip(fixed.columns, fixed.spatial_bins)),
                "anatomical_spatial_binding_changed")
        require(len(bank.banks) == len(support.BANKS), "all_32_anatomical_banks_required")
        positions, counts = [], np.zeros(len(ids), dtype=np.int64)
        for members in bank.banks:
            members = endpoint.column_ids(members, "anatomical_bank_members")
            require(len(members) == support.N and all(int(x) in lookup for x in members),
                    "anatomical_bank_membership_changed")
            positions.append(endpoint.frozen_array([lookup[int(x)] for x in members], integer=True))
            require(all(sum(bank.spatial_bins[structural[int(x)]] == b for x in members) == support.Q
                        for b in support.BINS), "anatomical_bank_quota_changed")
            for ident in members:
                counts[structural[int(ident)]] += 1
        declared = np.asarray(bank.inclusion_counts)
        require(declared.dtype.kind in "iu" and declared.shape == counts.shape
                and np.array_equal(declared, counts), "anatomical_inclusion_counts_changed")
        aligned = np.array([counts[structural[int(x)]] for x in fixed.columns], dtype=np.float64)
        require(np.array_equal(aligned / (len(support.BANKS) * support.N), fixed.anatomical_weights),
                "anatomical_mean_weights_changed")
        output[group] = tuple(positions)
    return MappingProxyType(output)


def _bind_features(result, features, fold_rows, check):
    development._mapping(features, nested.MODEL_IDS, "complete_feature_models_required")
    development._mapping(result.outer_selections, nested.MODEL_IDS, "complete_outer_selections_required")
    development._mapping(result.outer_cka_transforms, nested.MODEL_IDS, "complete_outer_CKA_transforms_required")
    transformed = {}
    for model in nested.MODEL_IDS:
        panel = features[model]
        require(isinstance(panel, nested.FeaturePanel) and panel.rows == result.design,
                "feature_row_binding_changed")
        development._mapping(panel.layers, nested.LAYERS, "all_registered_feature_layers_required")
        qualified = {}
        for layer in nested.LAYERS:
            nested._event(check, "qualify_secondary_features", model_id=model, layer=layer)
            values = prediction.matrix(panel.layers[layer], "secondary_feature_panel")
            require(values.shape == (nested.DEVELOPMENT_IMAGES, nested.FEATURE_WIDTHS[layer]),
                    "secondary_feature_shape_changed")
            qualified[layer] = values
        development._mapping(result.outer_selections[model], result.design.fold_labels,
                             "complete_outer_selections_required")
        development._mapping(result.outer_cka_transforms[model], result.design.fold_labels,
                             "complete_outer_CKA_transforms_required")
        transformed[model] = {}
        for fold, rows in fold_rows.items():
            selection = result.outer_selections[model][fold]
            require(isinstance(selection, nested.Selection) and selection.setting in nested.SETTINGS,
                    "registered_outer_selection_required")
            layer = selection.setting.layer
            fit = result.outer_cka_transforms[model][fold]
            require(isinstance(fit, prediction.PCAFit) and fit.rank == 64
                    and fit.feature_mean.shape == (nested.FEATURE_WIDTHS[layer],)
                    and fit.loadings.shape == (nested.FEATURE_WIDTHS[layer], 64)
                    and fit.numerical_rank >= 64
                    and fit.training_images == nested.DEVELOPMENT_IMAGES - len(rows),
                    "selected_layer_rank64_transform_changed")
            context = dict(model_id=model, layer=layer, rank=64,
                           training_folds=tuple(x for x in result.design.fold_labels if x != fold))
            snapshot = nested._call(check, "snapshot_outer_cka_transform", nested._frozen_pca, fit, context=context)
            require(np.isfinite(snapshot.score_scale) and snapshot.score_scale > 0,
                    "selected_layer_rank64_transform_changed")
            heldout = nested._slice_rows(check, "slice_secondary_features", qualified[layer], rows, context)
            scores = nested._call(check, "outer_cka_transform", snapshot.transform, heldout, context=context)
            transformed[model][fold] = nested._call(check, "freeze_outer_cka_features",
                nested._frozen_matrix, scores, len(rows), 64, "outer_cka_features", context=context)
    return nested._nested_map(transformed)


@dataclass(frozen=True)
class _SecondaryBank:
    family: endpoint.EndpointFamily
    responses: Mapping
    predictions: Mapping
    features: Mapping
    bank_positions: Mapping
    fold_rows: Mapping
    block_coordinates: np.ndarray
    resource_check: object
    component_ids: tuple

    def group_scores(self, scale, model, group, image_weights):
        check = self.resource_check
        fixed = self.family._supports[group]
        context = dict(model_id=model, group=group)
        if scale == "heldout_correlation":
            scores = {}
            for stage in endpoint.STAGES:
                scores[stage] = nested._call(check, "weighted_fisher_correlation", weighted_fisher_correlation,
                    self.responses[group][stage], self.predictions[model][group][stage], image_weights,
                    context={**context, "stage": stage})
            d = scores["D"]
            values = [float(np.mean(scores[stage], dtype=np.float64)) for stage in endpoint.STAGES]
            values.extend((float(d @ fixed.anatomical_weights), float(d @ fixed.reliable_weights)))
            if "D_VNC" in self.component_ids:
                values.append(float(np.mean(d[fixed.ceiling_positions], dtype=np.float64)))
            return np.asarray(values, dtype=np.float64)
        total = np.zeros(len(self.component_ids), dtype=np.float64)
        mass_total = float(image_weights.sum())
        for fold, rows in self.fold_rows.items():
            weights = image_weights[rows]
            mass = float(weights.sum())
            if mass == 0:
                continue
            features = self.features[model][fold]
            stage_values, delivered = [], None
            for stage in endpoint.STAGES:
                binding = {**context, "stage": stage}
                delivered = nested._slice_rows(check, "slice_secondary_responses",
                    self.responses[group][stage], rows, binding)
                stage_values.append(nested._call(check, "weighted_linear_cka", weighted_linear_cka,
                    features, delivered, weights, context=binding))
            # The final stage loop value is the SAME delivered D response bank.
            binding = {**context, "stage": "D"}
            stage_values.append(mean_bank_cka(features, delivered, weights, self.bank_positions[group],
                                             resource_check=check, context=binding))
            reliable = np.flatnonzero(fixed.reliable_weights > 0)
            selected = _slice_columns(check, delivered, reliable, binding)
            stage_values.append(nested._call(check, "weighted_linear_cka", weighted_linear_cka,
                                             features, selected, weights, context=binding))
            if "D_VNC" in self.component_ids:
                selected = _slice_columns(check, delivered, fixed.ceiling_positions, binding)
                stage_values.append(nested._call(check, "weighted_linear_cka", weighted_linear_cka,
                                                 features, selected, weights, context=binding))
            total += (mass / mass_total) * np.asarray(stage_values, dtype=np.float64)
        return total


@dataclass(frozen=True, init=False, eq=False)
class SecondaryStatistic:
    def __init__(self, bank, scale, pair_id):
        require(scale in SCALES and pair_id in endpoint.PAIR_SIGNS, "secondary_family_identity_invalid")
        object.__setattr__(self, "_bank", bank)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "model_order", endpoint.PAIR_SIGNS[pair_id])
        object.__setattr__(self, "component_ids", bank.component_ids)
        object.__setattr__(self, "score_scales", (scale,) * len(bank.component_ids))

    def _people(self, block_multiplicities, included):
        bank = self._bank
        nested._event(bank.resource_check, "secondary_statistic")
        block_weights = endpoint.multiplicities(block_multiplicities, 24, "block_multiplicities")
        image_weights = block_weights[bank.block_coordinates]
        values = np.zeros((4, len(self.component_ids)), dtype=np.float64)
        first, second = self.model_order
        for index, person in enumerate(endpoint.SUBJECTS):
            if not included[index]:
                continue
            for roi in endpoint.ROIS:
                group = (person, roi)
                a = bank.group_scores(self.scale, first, group, image_weights)
                b = bank.group_scores(self.scale, second, group, image_weights)
                values[index] += (a - b) / 2
        require(np.isfinite(values).all(), "secondary_statistic_nonfinite")
        return values

    def __call__(self, people_multiplicities, block_multiplicities):
        people = endpoint.multiplicities(people_multiplicities, 4, "people_multiplicities")
        values = self._people(block_multiplicities, people > 0)
        return (people @ values) / people.sum()

    def participant_effects(self, block_multiplicities):
        """Explicit diagnostics require all four people, unlike an absent draw."""
        return self._people(block_multiplicities, np.ones(4, dtype=bool))

    def crossed_bounds(self):
        """Separate complete scale/pair family; no secondary-to-primary promotion."""
        nested._event(self._bank.resource_check, "secondary_crossed_bounds")
        return uncertainty.crossed_bounds(self, self.component_ids, self.score_scales, "development")


@dataclass(frozen=True)
class DevelopmentSecondary:
    blocks: tuple
    statistics: Mapping


def compose_development_secondary(result, features, responses, family, anatomical_banks, *,
                                  block_order, resource_check):
    """Bind complete development inputs and return both scales/all signed pairs.

    Actual 32-bank memberships are required in support_core.AnatomicalBank
    records. Their structural axis may contain columns outside A_all; IDs are
    joined explicitly and all outside inclusion counts must be zero. Freeze
    identity/authenticity is caller-owned. block_order is the independently
    frozen original 24-block coordinate order, never sorted or inferred.
    """
    require(callable(resource_check), "external_resource_callback_required")
    nested._event(resource_check, "begin_secondary")
    blocks, _ = development._bind_layout(result, responses, family, block_order)
    banks = nested._call(resource_check, "bind_anatomical_banks", _bind_banks, family, anatomical_banks)
    nested._event(resource_check, "bind_secondary_rows")
    fold_rows = MappingProxyType({fold: endpoint.frozen_array(
        [i for i, label in enumerate(result.design.fold_ids) if label == fold], integer=True)
        for fold in result.design.fold_labels})
    block_lookup = {block.block_id: index for index, block in enumerate(blocks)}
    coordinates = endpoint.frozen_array([block_lookup[label] for label in result.design.block_ids], integer=True)
    cka_features = _bind_features(result, features, fold_rows, resource_check)
    observed = {group: {} for group in endpoint.GROUPS}
    predicted = {model: {group: {} for group in endpoint.GROUPS} for model in nested.MODEL_IDS}
    for group in endpoint.GROUPS:
        width = len(family._supports[group].columns)
        for stage in endpoint.STAGES:
            context = dict(group=group, stage=stage, shape=(nested.DEVELOPMENT_IMAGES, width))
            observed[group][stage] = nested._call(resource_check, "snapshot_secondary_response",
                development._snapshot, responses[group].stages[stage], width, "secondary_response", context=context)
            for model in nested.MODEL_IDS:
                predicted[model][group][stage] = nested._call(resource_check, "snapshot_secondary_prediction",
                    development._snapshot, result.predictions[model][group][stage], width,
                    "secondary_prediction", context={**context, "model_id": model})
    ids = COMPONENTS if "D_VNC_raw" in family.primary_endpoint_ids else COMPONENTS[:-1]
    bank = _SecondaryBank(family, nested._nested_map(observed), nested._nested_map(predicted),
                          cka_features, banks, fold_rows, coordinates, resource_check, ids)
    nested._event(resource_check, "bind_secondary_families")
    statistics = MappingProxyType({scale: MappingProxyType({pair: SecondaryStatistic(bank, scale, pair)
        for pair in endpoint.PAIR_SIGNS}) for scale in SCALES})
    nested._event(resource_check, "complete_secondary", "after")
    return DevelopmentSecondary(blocks, statistics)
