"""UNACTIVATED final-only publication for the fixed brain-loss comparator.

Accepts an already completed BrainLossFit; never fits, admits tau access,
discovers payloads or retries. The future admitted caller supplies the fixed
pair, authenticated 897 image IDs and complete native axes independently of
the fit. Equality to those bindings is checked here, not their authenticity.
Per-fold moments are not returned by the fitting library: preserve its exact
fold membership, normalization rule and CV scores, plus all final moments.
"""
from __future__ import annotations

from dataclasses import asdict
import os

import numpy as np
import torch

from brain_loss import (
    ALPHAS, FOLD_SIZES, BrainLossFit, BrainPairSummary,
    _choose_alpha, _native_ids, make_common_folds,
)
from final_content_artifact import _CheckedWriter, _file_context, _finite_tensor, _require
from schedule import PAIR_ORDER


SOURCE_IMPLEMENTATION = {
    "tag": "V1.0.0",
    "commit": "edbff02edd6a08d3c6c829af45a957fd14bdc460",
    "training": "NCC_brain_loss/ncc_train.py",
    "prediction": "NCC_brain_loss/ncc_test.py",
    "ridge": "FastL2LiR0.9_no_feature_selection_float32_primal_penalized_bias",
    "adapted_CV_policy": "outputs/protocol_decisions_2026-09-30.yaml",
}
NORMALIZATION = {
    "role": "tau_train",
    "aggregation": "separately_image_averaged_after_role_assignment",
    "axis": 0,
    "ddof": 1,
    "dtype": "float32",
    "CV_moments": "source_and_target_fitting_rows_only_per_fold",
    "final_moments": "all_897_tau_train_images",
    "zero_or_nonfinite_scale": "pair_not_evaluable_no_column_dropping",
}
CV_RULE = {
    "alphas": ALPHAS,
    "image_order": "UTF8_bytewise_sorted",
    "fold_assignment": "NumPy_RandomState42_permutation_position_modulo5",
    "validation_counts": FOLD_SIZES,
    "validation_metric": "mean_squared_error_across_all_target_voxels_per_image",
    "aggregation": "validation_image_count_weighted_mean_across_five_folds",
    "selection": "minimum_score_then_smaller_alpha",
    "prediction_augmented_dtype": "float64",
}


def _snapshot(value, shape, check):
    check()
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float32")
             and value.shape == shape, "complete_float32_brain_matrix_or_moments_required")
    result = torch.from_numpy(np.array(value, copy=True, order="C"))
    _finite_tensor(result, check)
    return result


def _validate_artifact(artifact, check):
    check()
    _require(isinstance(artifact, dict) and artifact.get("kind") == "ep21_final_brain_pair"
             and artifact.get("complete") is True, "final_brain_pair_artifact_required")
    summary = artifact["summary"]
    pair = summary["source"], summary["target"]
    index = artifact["pair_index"]
    _require(type(index) is int and 0 <= index < len(PAIR_ORDER)
             and pair == PAIR_ORDER[index] and artifact["canonical_pair_order"] == PAIR_ORDER,
             "canonical_twenty_pair_binding_required")
    _require(summary["available"] is True and type(summary["completed_fits"]) is int
             and summary["completed_fits"] == 51
             and summary["reason"] == "all_50_CV_fits_and_selected_final_refit_complete",
             "all_50_CV_fits_and_final_refit_required")
    folds = make_common_folds(artifact["training_image_ids"])
    _require(artifact["training_image_ids"] == folds.image_ids
             and artifact["validation_indices"] == folds.validation_indices,
             "exact_897_image_order_and_common_folds_required")
    _require(artifact["source_implementation"] == SOURCE_IMPLEMENTATION
             and artifact["normalization"] == NORMALIZATION
             and artifact["CV_rule"] == CV_RULE,
             "fixed_brain_source_and_normalization_rules_required")
    _require(tuple(summary["validation_image_counts"]) == FOLD_SIZES,
             "all_five_validation_image_counts_required")
    selected, aggregate = _choose_alpha(np.asarray(summary["fold_mse_by_alpha"], dtype=np.float64))
    _require(type(summary["selected_alpha"]) is int and summary["selected_alpha"] == selected
             and tuple(summary["validation_mse"]) == aggregate,
             "complete_image_weighted_CV_selection_required")
    source_ids = _native_ids(artifact["source_native_voxel_ids"])
    target_ids = _native_ids(artifact["target_native_voxel_ids"])
    _require(artifact["source_native_voxel_ids"] == source_ids
             and artifact["target_native_voxel_ids"] == target_ids,
             "complete_fixed_native_axes_required")
    shapes = {"matrix": (len(source_ids) + 1, len(target_ids)),
              "source_mean": (1, len(source_ids)), "source_std": (1, len(source_ids)),
              "target_mean": (1, len(target_ids)), "target_std": (1, len(target_ids))}
    for name, shape in shapes.items():
        value = artifact[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and value.layout == torch.strided
                 and tuple(value.shape) == shape, "complete_CPU_float32_brain_matrix_or_moments_required")
        _finite_tensor(value, check)
        if name.endswith("_std"):
            _require(bool((value > 0).all()), "positive_complete_tau_train_scales_required")
    return artifact


def prepare_final_brain_artifact(fit, *, expected_pair, authenticated_tau_train_image_ids,
                                 source_native_ids, target_native_ids, check):
    """Snapshot the final mapping and training-only metadata without file I/O.

    Native axes retain caller order, including nonmonotone IDs. They are the
    entire frozen brain-comparator ROI, never content-decoder eligible columns.
    CPU copies prevent subsequent caller mutation changing the saved model;
    their memory must fit the future caller's separately admitted resources.
    """
    _require(callable(check), "caller_allocation_check_required")
    check()
    _require(isinstance(fit, BrainLossFit) and isinstance(fit.summary, BrainPairSummary),
             "completed_brain_loss_fit_required")
    _require(isinstance(expected_pair, tuple) and expected_pair in PAIR_ORDER
             and (fit.summary.source, fit.summary.target) == expected_pair,
             "caller_fixed_directed_pair_binding_required")
    folds = make_common_folds(authenticated_tau_train_image_ids)
    source_ids, target_ids = _native_ids(source_native_ids), _native_ids(target_native_ids)
    _require(fit.image_ids == folds.image_ids, "exact_authenticated_897_tau_train_images_required")
    _require(fit.source_native_voxel_ids == source_ids and fit.target_native_voxel_ids == target_ids,
             "exact_caller_complete_native_axis_order_required")
    _require(fit.summary.available is True and type(fit.summary.completed_fits) is int
             and fit.summary.completed_fits == 51,
             "all_50_CV_fits_and_final_refit_required")
    artifact = {
        "kind": "ep21_final_brain_pair", "complete": True,
        "summary": asdict(fit.summary), "pair_index": PAIR_ORDER.index(expected_pair),
        "canonical_pair_order": PAIR_ORDER,
        "training_image_ids": folds.image_ids, "validation_indices": folds.validation_indices,
        "source_native_voxel_ids": source_ids, "target_native_voxel_ids": target_ids,
        "source_implementation": dict(SOURCE_IMPLEMENTATION),
        "normalization": dict(NORMALIZATION), "CV_rule": dict(CV_RULE),
    }
    for name, shape in (("matrix", (len(source_ids) + 1, len(target_ids))),
                        ("source_mean", (1, len(source_ids))), ("source_std", (1, len(source_ids))),
                        ("target_mean", (1, len(target_ids))), ("target_std", (1, len(target_ids)))):
        artifact[name] = _snapshot(getattr(fit, name), shape, check)
    return _validate_artifact(artifact, check)


def write_final_brain_artifact(path, artifact, *, budget, maximum_bytes):
    """Publish exclusively, with the same bounded writer as content artifacts.

    Preserve failed .partial files and any final file. A hard link publishes
    atomically without overwriting; fsync failure is propagated even if the
    final name is already visible. No directories, receipts or retries arise.
    """
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


def load_final_brain_artifact(path, *, budget, maximum_bytes):
    """Read an explicitly admitted final model on CPU, with no fitting/resume."""
    path = _file_context(path, budget, maximum_bytes)
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size <= maximum_bytes,
                 "final_artifact_exceeds_caller_storage_bound")
        budget.check()
        artifact = torch.load(stream, map_location="cpu", weights_only=True)
    return _validate_artifact(artifact, budget.check)
