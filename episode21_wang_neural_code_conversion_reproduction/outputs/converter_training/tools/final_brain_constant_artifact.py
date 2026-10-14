"""Final publication for the separately labeled exact-constant comparator.

Only completed ConstantBrainLossFit objects are accepted. Original artifacts
are never converted or relabeled. This module records all fold/final moments
on both complete native axes; its validation cannot independently authenticate
the caller's input identities or replay the fitting-value equality guard.
"""
from __future__ import annotations

from dataclasses import asdict
import os

import numpy as np
import torch

import brain_loss as original
from brain_loss_constant import (
    ConstantBrainLossFit, NormalizationState, NORMALIZATION_POLICY, _physical_prediction,
)
from final_brain_artifact import SOURCE_IMPLEMENTATION, CV_RULE, _snapshot
from final_content_artifact import _CheckedWriter, _file_context, _finite_tensor, _require

KIND = "ep21_final_brain_constant_pair"
SCHEMA_VERSION = 1
NORMALIZATION = {
    "policy": NORMALIZATION_POLICY,
    "role": "tau_train",
    "aggregation": "separately_image_averaged_after_role_assignment",
    "axis": 0, "ddof": 1, "dtype": "float32",
    "CV_moments": "source_and_target_fitting_rows_only_per_fold",
    "final_moments": "all_897_tau_train_images",
    "positive_finite_sd": "unchanged_including_tiny_positive_values",
    "zero_sd_guard": "finite_exact_constant_and_original_float32_mean_equals_first_value",
    "guarded_effective_scale": 1,
    "other_invalid_scale": "pair_not_evaluable_no_column_dropping",
    "validation_support": "all_target_coordinates_including_guarded_response_unit_residuals",
    "prediction": "restore_physical_target_units_before_separate_fixed_decoder_normalization",
}


def _snapshot_state(state, width, check):
    _require(isinstance(state, NormalizationState), "constant_comparator_normalization_state_required")
    shape = (1, width)
    result = {name: _snapshot(getattr(state, name), shape, check)
              for name in ("mean", "raw_sd", "effective_scale")}
    mask = state.replacement_mask
    _require(isinstance(mask, np.ndarray) and mask.dtype == np.dtype("bool")
             and mask.shape == shape, "complete_boolean_replacement_mask_required")
    check()
    result["replacement_mask"] = torch.from_numpy(np.array(mask, copy=True, order="C"))
    result["replacement_count"] = state.replacement_count
    result["fitting_images"] = state.fitting_images
    return result


def _validate_state(state, width, fitting_images, check):
    _require(isinstance(state, dict), "constant_comparator_normalization_record_required")
    _require(type(state["fitting_images"]) is int and state["fitting_images"] == fitting_images,
             "exact_fold_or_final_fitting_image_count_required")
    shape = (1, width)
    for name in ("mean", "raw_sd", "effective_scale"):
        value = state[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and value.layout == torch.strided
                 and tuple(value.shape) == shape, "complete_CPU_float32_normalization_record_required")
        _finite_tensor(value, check)
    mask = state["replacement_mask"]
    _require(isinstance(mask, torch.Tensor) and mask.device.type == "cpu"
             and mask.dtype == torch.bool and mask.layout == torch.strided
             and tuple(mask.shape) == shape, "complete_CPU_boolean_replacement_mask_required")
    raw, effective = state["raw_sd"], state["effective_scale"]
    _require(bool((raw >= 0).all()) and bool((effective > 0).all())
             and torch.equal(mask, raw == 0)
             and torch.equal(effective, torch.where(mask, torch.ones_like(raw), raw)),
             "raw_SD_effective_scale_and_exact_zero_mask_must_agree")
    _require(type(state["replacement_count"]) is int
             and state["replacement_count"] == int(torch.count_nonzero(mask)),
             "replacement_count_must_match_full_native_mask")


def _validate_artifact(artifact, check):
    check()
    _require(isinstance(artifact, dict) and artifact.get("kind") == KIND
             and type(artifact.get("schema_version")) is int
             and artifact["schema_version"] == SCHEMA_VERSION
             and artifact.get("complete") is True,
             "final_constant_brain_pair_artifact_required")
    summary = artifact["summary"]
    pair = summary["source"], summary["target"]
    index = artifact["pair_index"]
    _require(type(index) is int and 0 <= index < len(original.PAIR_ORDER)
             and pair == original.PAIR_ORDER[index]
             and artifact["canonical_pair_order"] == original.PAIR_ORDER,
             "canonical_twenty_pair_binding_required")
    _require(summary["available"] is True and type(summary["completed_fits"]) is int
             and summary["completed_fits"] == 51
             and summary["reason"] == "all_50_CV_fits_and_selected_final_refit_complete",
             "all_50_CV_fits_and_final_refit_required")
    folds = original.make_common_folds(artifact["training_image_ids"])
    _require(artifact["training_image_ids"] == folds.image_ids
             and artifact["validation_indices"] == folds.validation_indices,
             "exact_897_image_order_and_common_folds_required")
    _require(artifact["source_implementation"] == SOURCE_IMPLEMENTATION
             and artifact["normalization"] == NORMALIZATION and artifact["CV_rule"] == CV_RULE,
             "fixed_constant_comparator_source_and_normalization_rules_required")
    _require(tuple(summary["validation_image_counts"]) == original.FOLD_SIZES,
             "all_five_validation_image_counts_required")
    selected, aggregate = original._choose_alpha(np.asarray(summary["fold_mse_by_alpha"], dtype=np.float64))
    _require(type(summary["selected_alpha"]) is int and summary["selected_alpha"] == selected
             and tuple(summary["validation_mse"]) == aggregate,
             "complete_image_weighted_CV_selection_required")
    source_ids = original._native_ids(artifact["source_native_voxel_ids"])
    target_ids = original._native_ids(artifact["target_native_voxel_ids"])
    _require(artifact["source_native_voxel_ids"] == source_ids
             and artifact["target_native_voxel_ids"] == target_ids,
             "complete_fixed_native_axes_required")
    matrix = artifact["matrix"]
    _require(isinstance(matrix, torch.Tensor) and matrix.device.type == "cpu"
             and matrix.dtype == torch.float32 and matrix.layout == torch.strided
             and tuple(matrix.shape) == (len(source_ids) + 1, len(target_ids)),
             "complete_CPU_float32_brain_matrix_required")
    _finite_tensor(matrix, check)
    for axis, width in (("source", len(source_ids)), ("target", len(target_ids))):
        states = artifact["fold_" + axis + "_normalization"]
        _require(isinstance(states, tuple) and len(states) == 5,
                 "all_five_fold_normalization_records_required")
        for count, state in zip(original.FOLD_SIZES, states):
            _validate_state(state, width, original.TRAINING_IMAGES - count, check)
        _validate_state(artifact[axis + "_normalization"], width, original.TRAINING_IMAGES, check)
    return artifact


def prepare_final_brain_constant_artifact(fit, *, expected_pair,
        authenticated_tau_train_image_ids, source_native_ids, target_native_ids, check):
    """Snapshot a completed amended fit, preserving every raw/effective scale."""
    _require(callable(check), "caller_allocation_check_required")
    check()
    _require(isinstance(fit, ConstantBrainLossFit)
             and isinstance(fit.summary, original.BrainPairSummary),
             "completed_constant_brain_loss_fit_required")
    _require(isinstance(expected_pair, tuple) and expected_pair in original.PAIR_ORDER
             and (fit.summary.source, fit.summary.target) == expected_pair,
             "caller_fixed_directed_pair_binding_required")
    folds = original.make_common_folds(authenticated_tau_train_image_ids)
    source_ids, target_ids = original._native_ids(source_native_ids), original._native_ids(target_native_ids)
    _require(fit.image_ids == folds.image_ids, "exact_authenticated_897_tau_train_images_required")
    _require(fit.source_native_voxel_ids == source_ids and fit.target_native_voxel_ids == target_ids,
             "exact_caller_complete_native_axis_order_required")
    artifact = {
        "kind": KIND, "schema_version": SCHEMA_VERSION, "complete": True,
        "summary": asdict(fit.summary), "pair_index": original.PAIR_ORDER.index(expected_pair),
        "canonical_pair_order": original.PAIR_ORDER,
        "training_image_ids": folds.image_ids, "validation_indices": folds.validation_indices,
        "source_native_voxel_ids": source_ids, "target_native_voxel_ids": target_ids,
        "source_implementation": dict(SOURCE_IMPLEMENTATION),
        "normalization": dict(NORMALIZATION), "CV_rule": dict(CV_RULE),
        "matrix": _snapshot(fit.matrix, (len(source_ids) + 1, len(target_ids)), check),
    }
    for axis, width in (("source", len(source_ids)), ("target", len(target_ids))):
        artifact["fold_" + axis + "_normalization"] = tuple(
            _snapshot_state(state, width, check)
            for state in getattr(fit, "fold_" + axis + "_normalization"))
        artifact[axis + "_normalization"] = _snapshot_state(
            getattr(fit, axis + "_normalization"), width, check)
    return _validate_artifact(artifact, check)


def predict_physical(artifact, source_values, *, source_native_ids, check):
    """Predict physical units from an explicitly admitted amended artifact.

    This neither opens data nor normalizes/selects the separate decoder axis.
    """
    _validate_artifact(artifact, check)
    _require(original._native_ids(source_native_ids) == artifact["source_native_voxel_ids"],
             "exact_fixed_native_voxel_order_required")
    source, target = artifact["source_normalization"], artifact["target_normalization"]
    return _physical_prediction(source_values, artifact["matrix"].numpy(),
        source["mean"].numpy(), source["effective_scale"].numpy(),
        target["mean"].numpy(), target["effective_scale"].numpy(), check)


def write_final_brain_constant_artifact(path, artifact, *, budget, maximum_bytes):
    """Exclusive bounded publication; preserve every failed partial/final file."""
    path = _file_context(path, budget, maximum_bytes)
    _validate_artifact(artifact, budget.check)
    if os.path.lexists(path):
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        torch.save(artifact, _CheckedWriter(stream, budget.check, maximum_bytes))
        budget.check()
        stream.flush()
        os.fsync(stream.fileno())
    budget.check()
    os.link(partial, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
        partial.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def load_final_brain_constant_artifact(path, *, budget, maximum_bytes):
    path = _file_context(path, budget, maximum_bytes)
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size <= maximum_bytes,
                 "final_artifact_exceeds_caller_storage_bound")
        budget.check()
        artifact = torch.load(stream, map_location="cpu", weights_only=True)
    return _validate_artifact(artifact, budget.check)
