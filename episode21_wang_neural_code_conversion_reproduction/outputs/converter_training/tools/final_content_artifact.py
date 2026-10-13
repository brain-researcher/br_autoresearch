"""UNACTIVATED final-only content-converter artifact integration.

No driver, CLI, admission, fit, payload reader or automatic continuation. The
future admitted caller may prepare the final model after fit_content_pair
returns, then explicitly publish it within that caller's compute/storage
budget. The empirical execution adapter remains unavailable. This helper does
not authenticate a supplied completion or grant later-stage access.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path

import numpy as np
import torch

from decoder_interface import ContentPair, NativeAxis, assembly, require_image_order
from schedule import (
    ChannelSchedule, ITERATIONS, LAYER_ORDER, PAIR_ORDER, PairInitialization, SEED,
    make_channel_schedule,
)
from train_content import CompletedContentFit, LEARNING_RATE, TRAINING_ROWS


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _state_shapes(input_width, output_width):
    hidden = input_width // 2
    return {"model.0.weight": (hidden, input_width), "model.0.bias": (hidden,),
            "model.3.weight": (hidden, hidden), "model.3.bias": (hidden,),
            "model.6.weight": (output_width, hidden), "model.6.bias": (output_width,)}


def _finite_tensor(tensor, check):
    # Avoid a model-sized finiteness mask; check the last Adam update too.
    flattened = tensor.reshape(-1)
    for first in range(0, flattened.numel(), 262144):
        check()
        _require(bool(torch.isfinite(flattened[first:first+262144]).all()),
                 "nonfinite_final_artifact_tensor")


def _copy_tensor(value, check):
    check()
    result = value.detach().to(device="cpu", copy=True).contiguous()
    _finite_tensor(result, check)
    return result


def _axis_snapshot(axis, check):
    _require(isinstance(axis, NativeAxis), "actual_frozen_native_axis_required")
    result = {"participant": axis.participant}
    for name in ("full_native_ids", "eligible_native_ids", "eligible_full_columns", "mean", "std"):
        raw = np.asarray(getattr(axis, name))
        if name in ("mean", "std"):
            _require(raw.dtype == np.dtype("float32"), "retained_float32_training_moments_required")
        else:
            _require(raw.dtype.kind in "iu" and raw.ndim == 1 and bool((raw >= 0).all()),
                     "nonnegative_native_ID_or_column_vector_required")
            cast = raw.astype(np.int64)
            _require(np.array_equal(raw, cast) and bool((cast >= 0).all()), "native_axis_integer_overflow")
            raw = cast
        result[name] = torch.from_numpy(np.array(raw, copy=True))
        _finite_tensor(result[name], check)
    _validate_axis(result)
    return result


def _validate_axis(axis):
    full, eligible, columns = (axis[name] for name in
                              ("full_native_ids", "eligible_native_ids", "eligible_full_columns"))
    for values in (full, eligible, columns):
        _require(isinstance(values, torch.Tensor) and values.device.type == "cpu"
                 and values.dtype == torch.int64 and values.ndim == 1,
                 "CPU_integer_native_axis_required")
    width = eligible.numel()
    _require(width >= 4 and columns.shape == eligible.shape and full.numel() >= width
             and bool((full >= 0).all()) and bool((full[1:] > full[:-1]).all())
             and bool((eligible[1:] > eligible[:-1]).all())
             and bool((columns >= 0).all()) and bool((columns < full.numel()).all())
             and bool((columns[1:] > columns[:-1]).all())
             and torch.equal(full[columns], eligible), "exact_ordered_eligible_native_axis_required")
    for name in ("mean", "std"):
        value = axis[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and value.shape == (width,)
                 and bool(torch.isfinite(value).all()), "fixed_eligible_training_moments_required")
    _require(bool((axis["std"] > 0).all()), "positive_frozen_eligible_scales_required")
    return width


def _validate_artifact(artifact, check):
    """Validate this final checkpoint's scientific bindings, not stage access."""
    check()
    _require(isinstance(artifact, dict) and artifact.get("kind") == "ep21_final_content_pair"
             and artifact.get("complete") is True, "final_content_pair_artifact_required")
    initial = artifact["initialization"]
    index = initial["pair_index"]
    _require(type(initial["seed"]) is int and initial["seed"] == SEED
             and type(index) is int and 0 <= index < len(PAIR_ORDER)
             and (initial["source"], initial["target"]) == PAIR_ORDER[index]
             and tuple(artifact["canonical_pair_order"]) == PAIR_ORDER, "canonical_twenty_pair_binding_required")
    source, target = artifact["source_axis"], artifact["target_axis"]
    input_width, output_width = _validate_axis(source), _validate_axis(target)
    _require((source["participant"], target["participant"]) == PAIR_ORDER[index]
             and initial["input_width"] == input_width and initial["output_width"] == output_width
             and initial["hidden_width"] == input_width // 2, "model_and_eligible_axis_widths_must_match")
    for name in ("cpu_rng_before", "cpu_rng_after"):
        value = initial[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                 "canonical_CPU_initialization_states_required")
    _require(isinstance(initial["torch_version"], str) and bool(initial["torch_version"]),
             "initialization_runtime_binding_required")
    completion = artifact["completion"]
    _require(type(completion["iterations"]) is int and completion["iterations"] == ITERATIONS
             and type(completion["rows"]) is int and completion["rows"] == TRAINING_ROWS,
             "all_1024_full_4712_row_updates_required_before_final_artifact")
    _require(completion["backend"] in ("sparse", "dense")
             and completion["bitwise_source_equivalence"] is False
             and math.isfinite(completion["last_step_loss_before_update"])
             and completion["last_step_loss_before_update"] >= 0
             and completion["final_used_learning_rate"] == LEARNING_RATE / 512,
             "fixed_final_update_metadata_required")
    rows, decoder_rows = artifact["source_training_image_ids"], artifact["target_decoder_training_image_ids"]
    for values in (rows, decoder_rows):
        _require(isinstance(values, tuple) and len(values) == TRAINING_ROWS
                 and len(set(values)) == TRAINING_ROWS
                 and all(isinstance(value, str) and bool(value) for value in values),
                 "exact_4712_unique_image_identity_binding_required")
    _require(not set(rows).intersection(decoder_rows), "source_converter_target_decoder_overlap_forbidden")
    schedule = artifact["channel_schedule"]
    _require(type(schedule["seed"]) is int and schedule["seed"] == SEED
             and tuple(schedule["layer_order"]) == LAYER_ORDER
             and isinstance(schedule["python_version"], str) and bool(schedule["python_version"]),
             "shared_source_schedule_runtime_and_layer_order_required")
    # Preserve the exact stored schedule even for later inspection in another
    # runtime; generation/replay was bound when preparing the final artifact.
    channels = schedule["channels_by_layer"]
    _require(isinstance(channels, tuple) and len(channels) == len(LAYER_ORDER)
             and all(isinstance(row, tuple) and len(row) == ITERATIONS for row in channels),
             "complete_shared_channel_schedule_required")
    for layer, row in zip(LAYER_ORDER, channels):
        if layer.startswith("fc"):
            _require(row == (None,)*ITERATIONS, "all_FC_units_required_each_iteration")
        else:
            count = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}[int(layer[4])]
            _require(all(type(value) is int for value in row)
                     and all(sorted(row[first:first+count]) == list(range(count))
                             for first in range(0, ITERATIONS, count)), "complete_convolutional_permutation_cycles_required")
    expected = _state_shapes(input_width, output_width)
    state = artifact["state_dict"]
    _require(isinstance(state, dict) and set(state) == set(expected), "exact_tagged_converter_state_required")
    for name, shape in expected.items():
        tensor = state[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                 and tensor.dtype == torch.float32 and tensor.layout == torch.strided
                 and tuple(tensor.shape) == shape, "tagged_float32_state_dimensions_required")
        _finite_tensor(tensor, check)
    _require(artifact["target_decoder_manifest"] == assembly.MANIFEST
             and artifact["source_training_state"] == assembly.required_state(source["participant"]).record()
             and artifact["target_decoder_state"] == assembly.required_state(target["participant"]).record()
             and artifact["source_feature_root"] == str(assembly.BASE / "outputs/steward_provisioning"
                                                       / "regular_vgg19_sealed" / source["participant"]),
             "existing_frozen_decoder_and_training_state_references_required")
    _require(artifact["content_loss_rule"] == "normalized_inverse_squared_active_block_norms_times_summed_squared_error",
             "corrected_source_loss_binding_required")
    return artifact


def prepare_final_content_artifact(model, completion, pair, initialization, channel_schedule, *, check):
    """Snapshot one completed fit, with no file I/O or new numerical fitting.

    CPU copies protect saved final weights from later caller mutation. They
    require additional memory that the future converter budget must cover.
    Source normalized responses and decoder coefficients are never duplicated.
    The caller must have consumed the existing explicit loss-correction evidence
    during future stage admission; this function does not grant that admission.
    """
    _require(callable(check), "caller_allocation_check_required")
    check()
    _require(isinstance(completion, CompletedContentFit) and completion.iterations == ITERATIONS
             and completion.rows == TRAINING_ROWS, "completed_full_fit_required")
    _require(isinstance(pair, ContentPair) and isinstance(initialization, PairInitialization),
             "existing_content_pair_and_canonical_initialization_required")
    _require(isinstance(channel_schedule, ChannelSchedule)
             and channel_schedule == make_channel_schedule(), "exact_shared_source_schedule_required")
    _require(str(torch.__version__) == initialization.torch_version, "captured_initialization_runtime_required")
    assembly.verify_catalog(pair.manifest.fragments, pair.manifest.states)
    _require(pair.target_state in pair.manifest.states, "target_state_must_belong_to_existing_decoder_catalog")
    source_ids = require_image_order(pair.source.image_ids, pair.source.image_ids)
    target_ids = tuple(pair.target_normalization.training_image_ids.tolist())
    source_axis = _axis_snapshot(pair.source.axis, check)
    target_axis = _axis_snapshot(pair.target_axis, check)
    _require(pair.source.normalized_X.shape == (TRAINING_ROWS, source_axis["eligible_native_ids"].numel()),
             "actual_full_source_matrix_axis_required")
    _require(pair.target_normalization.participant == target_axis["participant"]
             and pair.target_state.participant == target_axis["participant"], "target_decoder_participant_binding_required")
    columns = target_axis["eligible_full_columns"].numpy()
    for name, expected in (("full_native_ids", pair.target_normalization.native_ids),
                           ("eligible_native_ids", pair.target_normalization.eligible_ids),
                           ("mean", pair.target_normalization.mean[columns]),
                           ("std", pair.target_normalization.std[columns])):
        _require(np.array_equal(target_axis[name].numpy(), expected), "target_axis_must_equal_actual_decoder_normalization")
    _require(isinstance(model, torch.nn.Module), "final_pinned_converter_model_required")
    state = model.state_dict()
    expected = _state_shapes(initialization.input_width, initialization.output_width)
    _require(set(state) == set(expected) and all(tuple(state[key].shape) == shape
             and state[key].dtype == torch.float32 for key, shape in expected.items()),
             "exact_tagged_final_model_state_and_dimensions_required")
    initial = {key: getattr(initialization, key) for key in (
        "pair_index", "source", "target", "input_width", "hidden_width", "output_width", "torch_version")}
    initial.update(seed=SEED, cpu_rng_before=_copy_tensor(initialization.cpu_rng_before, check),
                   cpu_rng_after=_copy_tensor(initialization.cpu_rng_after, check))
    artifact = {
        "kind": "ep21_final_content_pair", "complete": True,
        "state_dict": {name: _copy_tensor(tensor, check) for name, tensor in state.items()},
        "completion": asdict(completion), "initialization": initial,
        "canonical_pair_order": PAIR_ORDER,
        "channel_schedule": {"seed": SEED, "layer_order": LAYER_ORDER, "python_version": channel_schedule.python_version,
                             "channels_by_layer": channel_schedule.channels_by_layer},
        "source_axis": source_axis, "target_axis": target_axis,
        "source_training_image_ids": source_ids, "target_decoder_training_image_ids": target_ids,
        "source_feature_root": str(pair.source.feature_root),
        "source_training_state": assembly.required_state(source_axis["participant"]).record(),
        "target_decoder_state": pair.target_state.record(), "target_decoder_manifest": assembly.MANIFEST,
        "content_loss_rule": "normalized_inverse_squared_active_block_norms_times_summed_squared_error",
    }
    return _validate_artifact(artifact, check)


class _CheckedWriter:
    def __init__(self, stream, check, maximum_bytes):
        self.stream, self.check, self.maximum_bytes = stream, check, maximum_bytes

    def write(self, data):
        view = memoryview(data).cast("B")
        _require(self.stream.tell()+view.nbytes <= self.maximum_bytes, "final_artifact_exceeds_caller_storage_bound")
        total = 0
        while total < view.nbytes:
            self.check()
            written = self.stream.write(view[total:total+8*1024**2])
            if written is None or written <= 0:
                raise OSError("final artifact write made no progress")
            total += written
        return total

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.stream.tell()


def _file_context(path, budget, maximum_bytes):
    assembly._require_reader_context(budget)
    _require(type(maximum_bytes) is int and maximum_bytes > 0, "explicit_caller_artifact_byte_bound_required")
    path = Path(path)
    _require(path.is_absolute() and path.suffix == ".pt", "explicit_absolute_final_pt_path_required")
    return path


def write_final_content_artifact(path, artifact, *, budget, maximum_bytes):
    """Publish once, without replacing a final file or failed partial attempt.

    A sibling .partial is written exclusively and synced before a hard link
    creates the final name atomically with no overwrite. Failures before cleanup
    preserve the partial for diagnosis. A failure after link publication may
    leave the final visible, including after partial removal if the last directory
    sync fails; it never replaces or retries it. The future coordinator owns that
    closeout. No directories or separate manifests/receipts are created here.
    """
    path = _file_context(path, budget, maximum_bytes)
    _validate_artifact(artifact, budget.check)
    if os.path.lexists(path):
        raise FileExistsError(path)
    partial = path.with_name(path.name+".partial")
    with partial.open("xb") as stream:
        torch.save(artifact, _CheckedWriter(stream, budget.check, maximum_bytes))
        budget.check()
        stream.flush()
        os.fsync(stream.fileno())
    budget.check()
    os.link(partial, path)  # EEXIST is intentional; os.replace would overwrite.
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
        partial.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def load_final_content_artifact(path, *, budget, maximum_bytes):
    """Read a caller-authorized final checkpoint on CPU; never resume training."""
    path = _file_context(path, budget, maximum_bytes)
    with path.open("rb") as stream:
        _require(os.fstat(stream.fileno()).st_size <= maximum_bytes, "final_artifact_exceeds_caller_storage_bound")
        budget.check()
        artifact = torch.load(stream, map_location="cpu", weights_only=True)
    return _validate_artifact(artifact, budget.check)
