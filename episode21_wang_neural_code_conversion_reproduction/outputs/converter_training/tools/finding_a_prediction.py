"""Dormant Finding A prediction bridge over caller-admitted arrays and models.

No reader, training, CLI, grant, scoring or collection-completeness decision.
The caller authenticates roles, paired image identities, final artifacts, the
pinned converter factory and decoder blocks before calling, and supplies its
allocation/cancellation check. All 224 images and the training-fixed native
axes are retained. A within-source input can be reused for all outgoing pairs.

The adopted EP21 decoder interface casts normalized inputs to float32 before
FastL2LiR's default float64 prediction. This differs from the literal tagged
NCC brain caller, which forwards its float64 normalized input directly. Content
prediction retains the tagged float32 physical inverse/forward normalization
round trip. Returned float64 features are pre-serialization intermediates;
source_file_features explicitly applies the tagged save_array float32 cast.
Neither numerical bitwise equivalence across BLAS implementations nor complete
layer, nineteen-layer, pair-panel or Finding A coverage is asserted here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from decoder_interface import NativeAxis, DecoderBlock, active_shape, assembly
import final_brain_artifact as brain_artifacts
import final_content_artifact as content_artifacts
from schedule import PARTICIPANTS, PairInitialization, initialize_pair


TEST_IMAGES = 224
_ROWS = 64
_AXIS_FIELDS = ("full_native_ids", "eligible_native_ids", "eligible_full_columns", "mean", "std")


@dataclass(frozen=True)
class TauTestResponses:
    participant: str
    image_ids: tuple[str, ...]
    native_voxel_ids: tuple[int, ...]
    values: np.ndarray
    role: str = "tau_test"
    aggregation: str = "separately_image_averaged_after_role_assignment"


@dataclass(frozen=True)
class PreparedDecoderInput:
    method: str
    source: str
    target: str | None
    decoder_axis: NativeAxis
    image_ids: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class FeaturePredictionBlock:
    """Only the named channel's explicit F/C units, with all 224 image rows.

    standardized_features and features are float64 predictor intermediates.
    source_file_features is the float32 representation used by tagged output
    files; a later source-style metric adapter must respect that boundary.
    No sampled training-channel schedule determines these evaluation units.
    """

    method: str
    source: str
    target: str | None
    decoder_participant: str
    image_ids: tuple[str, ...]
    layer: str
    channel: int | None
    full_layer_Fortran_index: np.ndarray
    full_layer_C_index: np.ndarray
    standardized_features: np.ndarray
    features: np.ndarray
    source_file_features: np.ndarray


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    value.setflags(write=False)
    return value


def _ids(values, count):
    _require(isinstance(values, tuple) and len(values) == count
             and all(isinstance(value, str) and bool(value) for value in values)
             and len(set(values)) == count, "complete_unique_image_identity_tuple_required")
    return values


def _disjoint(test_ids, training_ids, count):
    _ids(training_ids, count)
    _require(not set(test_ids).intersection(training_ids), "test_training_image_overlap_forbidden")


def _finite(values, check):
    # All public matrices are at most 224 rows or one <=256-unit block.
    for first in range(0, len(values), _ROWS):
        check()
        _require(bool(np.isfinite(values[first:first + _ROWS]).all()),
                 "nonfinite_required_prediction_input_or_output")


def _axis(axis, check):
    """Reuse the frozen training-axis contract, copying only small metadata."""
    check()
    record = content_artifacts._axis_snapshot(axis, check)
    _require(record["participant"] in PARTICIPANTS, "fixed_decoder_participant_required")
    return NativeAxis(record["participant"],
                      *(_readonly(record[name].numpy()) for name in _AXIS_FIELDS))


def _artifact_axis(record, check):
    return _axis(NativeAxis(record["participant"],
                           *(record[name].detach().numpy() for name in _AXIS_FIELDS)), check)


def _same_axis(actual, expected):
    _require(actual.participant == expected.participant
             and all(np.array_equal(getattr(actual, name), getattr(expected, name))
                     for name in _AXIS_FIELDS), "exact_final_model_decoder_axis_and_moments_required")


def _responses(responses, participant, full_native_ids, expected_ids, check):
    check()
    _ids(expected_ids, TEST_IMAGES)
    _require(isinstance(responses, TauTestResponses)
             and responses.participant == participant and participant in PARTICIPANTS
             and responses.role == "tau_test"
             and responses.aggregation == "separately_image_averaged_after_role_assignment",
             "caller_admitted_separately_image_averaged_tau_test_responses_required")
    _ids(responses.image_ids, TEST_IMAGES)
    _require(responses.image_ids == expected_ids, "exact_224_paired_test_image_order_required")
    _require(isinstance(responses.native_voxel_ids, tuple)
             and responses.native_voxel_ids == tuple(full_native_ids),
             "complete_final_model_native_axis_and_order_required")
    values = responses.values
    _require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float32")
             and values.shape == (TEST_IMAGES, len(full_native_ids)),
             "complete_float32_test_response_matrix_required")
    _finite(values, check)
    return values


def _normalize_unique(values, axis, check):
    result = np.empty((TEST_IMAGES, len(axis.eligible_native_ids)), dtype=np.float32)
    for first in range(0, TEST_IMAGES, _ROWS):
        check()
        result[first:first + _ROWS] = ((values[first:first + _ROWS, axis.eligible_full_columns]
                                      - axis.mean) / axis.std)
    _finite(result, check)
    return result


def prepare_within_source(responses, axis, *, decoder_training_image_ids,
                          expected_test_image_ids, check):
    """Prepare one reusable source baseline using its own training moments."""
    check()
    axis = _axis(axis, check)
    values = _responses(responses, axis.participant, axis.full_native_ids,
                        expected_test_image_ids, check)
    _disjoint(expected_test_image_ids, decoder_training_image_ids, 4712)
    normalized = _normalize_unique(values, axis, check)
    return PreparedDecoderInput("within_source", axis.participant, None, axis,
                                expected_test_image_ids, _readonly(normalized))


def prepare_content(responses, artifact, target_axis, *, expected_test_image_ids,
                    converter_factory, check):
    """Run one final frozen model, preserving the tagged float32 round trip.

    The supplied factory is the already-qualified pinned Converter. Restoring
    its initialization binding preserves caller RNG; final weights replace the
    initialized weights before eval/no_grad inference on all 224 images. No
    factory/source loading or learned normalization occurs here.
    """
    check()
    content_artifacts._validate_artifact(artifact, check)
    source_axis = _artifact_axis(artifact["source_axis"], check)
    target_axis = _axis(target_axis, check)
    _same_axis(target_axis, _artifact_axis(artifact["target_axis"], check))
    values = _responses(responses, source_axis.participant, source_axis.full_native_ids,
                        expected_test_image_ids, check)
    for name in ("source_training_image_ids", "target_decoder_training_image_ids"):
        _disjoint(expected_test_image_ids, artifact[name], 4712)
    normalized = _normalize_unique(values, source_axis, check)
    initial = artifact["initialization"]
    record = PairInitialization(**{name: initial[name] for name in (
        "pair_index", "source", "target", "input_width", "hidden_width", "output_width",
        "torch_version", "cpu_rng_before", "cpu_rng_after")})
    check()
    model = initialize_pair(record, converter_factory)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    check()
    with torch.no_grad():
        predicted = model(torch.from_numpy(normalized))
    check()
    _require(isinstance(predicted, torch.Tensor) and predicted.device.type == "cpu"
             and predicted.dtype == torch.float32
             and tuple(predicted.shape) == (TEST_IMAGES, len(target_axis.eligible_native_ids)),
             "pinned_float32_converter_output_shape_required")
    predicted = predicted.detach().numpy()
    _finite(predicted, check)
    # Source NCC_test.py restores physical float32 responses; utils.py then
    # centers/scales again. Forwarding pred_B directly loses this round trip.
    normalized_target = np.empty_like(predicted)
    for first in range(0, TEST_IMAGES, _ROWS):
        check()
        physical = predicted[first:first + _ROWS] * target_axis.std + target_axis.mean
        normalized_target[first:first + _ROWS] = (physical - target_axis.mean) / target_axis.std
    _finite(normalized_target, check)
    return PreparedDecoderInput("content_loss", source_axis.participant,
        target_axis.participant, target_axis, expected_test_image_ids, _readonly(normalized_target))


def prepare_brain(responses, artifact, target_axis, *, target_decoder_training_image_ids,
                  expected_test_image_ids, check):
    """Map all native target columns, then enter the target decoder's scale.

    Tau normalization is float32. Default float64 ones promote the ridge
    prediction and physical target restoration to float64. Target subject-
    unique normalization remains float64 until the adopted decoder-input cast.
    Column blocks limit coefficient promotion; no native column is omitted.
    """
    check()
    brain_artifacts._validate_artifact(artifact, check)
    source, target = artifact["summary"]["source"], artifact["summary"]["target"]
    target_axis = _axis(target_axis, check)
    _require(target_axis.participant == target, "brain_target_decoder_participant_required")
    model_target_ids = artifact["target_native_voxel_ids"]
    _require(set(model_target_ids) == set(target_axis.full_native_ids.tolist()),
             "complete_brain_target_and_decoder_native_axes_required")
    # Brain artifacts retain supplied native order; decoder axes are sorted.
    # Map by identity, without requiring or silently imposing a new brain order.
    positions = {native: column for column, native in enumerate(model_target_ids)}
    eligible_columns = np.asarray([positions[native] for native in target_axis.eligible_native_ids],
                                  dtype=np.int64)
    values = _responses(responses, source, artifact["source_native_voxel_ids"],
                        expected_test_image_ids, check)
    _disjoint(expected_test_image_ids, artifact["training_image_ids"], 897)
    _disjoint(expected_test_image_ids, target_decoder_training_image_ids, 4712)
    source_mean, source_std = (artifact[name].detach().numpy() for name in ("source_mean", "source_std"))
    tau_source = np.empty_like(values)
    for first in range(0, TEST_IMAGES, _ROWS):
        check()
        tau_source[first:first + _ROWS] = (values[first:first + _ROWS] - source_mean) / source_std
    _finite(tau_source, check)
    # Match the pinned appended unscaled, penalized bias coordinate.
    augmented = np.hstack((tau_source, np.ones((TEST_IMAGES, 1))))
    matrix = artifact["matrix"].detach().numpy()
    target_mean, target_std = (artifact[name].detach().numpy() for name in ("target_mean", "target_std"))
    physical = np.empty((TEST_IMAGES, len(model_target_ids)), dtype=np.float64)
    for first in range(0, len(model_target_ids), assembly.MAX_BLOCK_UNITS):
        check()
        last = first + assembly.MAX_BLOCK_UNITS
        projected = np.matmul(augmented, matrix[:, first:last])
        physical[:, first:last] = projected * target_std[:, first:last] + target_mean[:, first:last]
        check()
    _finite(physical, check)
    normalized_target = np.empty((TEST_IMAGES, len(eligible_columns)), dtype=np.float32)
    for first in range(0, TEST_IMAGES, _ROWS):
        check()
        # Do not cast the physical response before this float64 normalization.
        normalized_target[first:first + _ROWS] = (
            (physical[first:first + _ROWS, eligible_columns] - target_axis.mean) / target_axis.std)
    _finite(normalized_target, check)
    return PreparedDecoderInput("brain_loss", source, target, target_axis,
                                expected_test_image_ids, _readonly(normalized_target))


def _integer_array(value, shape, message):
    _require(isinstance(value, np.ndarray) and value.dtype.kind in "iu" and value.shape == shape,
             message)
    return value


def _decoder_block(block, axis, check):
    check()
    _require(isinstance(block, DecoderBlock) and block.participant == axis.participant,
             "exact_decoder_participant_block_required")
    shape, layer_units = active_shape(block.layer, block.channel)
    weights = block.weights
    _require(isinstance(weights, np.ndarray) and weights.dtype == np.dtype("float32")
             and weights.ndim == 2 and 1 <= weights.shape[0] <= assembly.MAX_BLOCK_UNITS
             and 1 <= weights.shape[1] <= len(axis.eligible_native_ids),
             "bounded_float32_decoder_coefficients_required")
    units = weights.shape[0]
    _require(type(block.unit_start) is int and 0 <= block.unit_start
             and block.unit_start + units <= layer_units, "contiguous_in_layer_decoder_units_required")
    columns = _integer_array(block.eligible_columns, weights.shape, "integer_decoder_columns_required")
    ids = _integer_array(block.selected_native_voxel_ids, weights.shape, "integer_selected_native_ids_required")
    _require(bool((columns >= 0).all()) and bool((columns < len(axis.eligible_native_ids)).all())
             and np.array_equal(axis.eligible_native_ids[columns], ids)
             and all(len(set(row.tolist())) == len(row) for row in columns),
             "unique_selected_coefficients_on_frozen_native_axis_required")
    spatial = np.arange(block.unit_start, block.unit_start + units, dtype=np.int64)
    expected_f = spatial if len(shape) == 1 else spatial * shape[0] + block.channel
    expected_c = (spatial if len(shape) == 1 else block.channel * layer_units
                  + (spatial % shape[1]) * shape[2] + spatial // shape[1])
    for actual, expected in ((block.full_layer_Fortran_index, expected_f),
                             (block.full_layer_C_index, expected_c)):
        _integer_array(actual, (units,), "integer_F_and_C_feature_coordinates_required")
        _require(np.array_equal(actual, expected), "exact_F_and_C_feature_coordinates_required")
    for name in ("bias", "target_mean", "target_std"):
        value = getattr(block, name)
        _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float32")
                 and value.shape == (units,), "frozen_float32_decoder_bias_and_feature_moments_required")
        _finite(value, check)
    _require(bool((block.target_std > 0).all()), "positive_frozen_feature_scales_required")
    _finite(weights, check)
    return units, expected_f, expected_c


def predict_decoder_block(prepared, block, *, check, row_chunk_size=64):
    """Predict <=256 explicit units using native-order default FastL2LiR math.

    Scatter ranked selected weights into the complete training-eligible axis,
    preserving zero weights at other eligible columns. A rank-ordered sparse
    dot or the float32 training-loss projector is not the evaluation predictor.
    All channels are available; the caller owns complete evaluation coverage.
    """
    check()
    _require(isinstance(prepared, PreparedDecoderInput)
             and prepared.method in ("within_source", "content_loss", "brain_loss")
             and prepared.source in PARTICIPANTS, "prepared_fixed_method_decoder_input_required")
    _require(type(row_chunk_size) is int and 1 <= row_chunk_size <= TEST_IMAGES,
             "bounded_prediction_row_chunk_required")
    axis = _axis(prepared.decoder_axis, check)
    _require((prepared.method == "within_source" and prepared.target is None
              and axis.participant == prepared.source)
             or (prepared.method != "within_source" and prepared.target == axis.participant
                 and prepared.target in PARTICIPANTS and prepared.target != prepared.source),
             "method_source_target_decoder_binding_required")
    _ids(prepared.image_ids, TEST_IMAGES)
    values = prepared.values
    _require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float32")
             and values.shape == (TEST_IMAGES, len(axis.eligible_native_ids)),
             "complete_float32_prepared_decoder_input_required")
    _finite(values, check)
    units, full_f, full_c = _decoder_block(block, axis, check)
    check()
    dense = np.zeros((len(axis.eligible_native_ids), units), dtype=np.float32)
    for unit in range(units):
        dense[block.eligible_columns[unit], unit] = block.weights[unit]
    standardized = np.empty((TEST_IMAGES, units), dtype=np.float64)
    for first in range(0, TEST_IMAGES, row_chunk_size):
        check()
        stop = min(TEST_IMAGES, first + row_chunk_size)
        # FastL2LiR 0.9 casts only X; stored W and b remain float32.
        x = values[first:stop].astype(np.float64)
        standardized[first:stop] = (np.matmul(x, dense)
            + np.matmul(np.ones((stop - first, 1), dtype=np.float64), block.bias[None, :]))
        check()
    _finite(standardized, check)
    features = standardized * block.target_std + block.target_mean
    _finite(features, check)
    with np.errstate(over="ignore", invalid="ignore"):
        source_file_features = features.astype(np.float32)
    _finite(source_file_features, check)
    return FeaturePredictionBlock(prepared.method, prepared.source, prepared.target,
        axis.participant, prepared.image_ids, block.layer, block.channel,
        _readonly(full_f), _readonly(full_c), _readonly(standardized),
        _readonly(features), _readonly(source_file_features))
