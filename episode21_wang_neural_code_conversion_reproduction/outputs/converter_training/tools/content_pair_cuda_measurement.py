"""Separately admitted four-update CUDA measurement of the unchanged full pair.

No grant, CLI, payload reader, model output, restart or scientific score. The
original CPU measurement and shared numerical libraries remain unchanged.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import time
from types import MappingProxyType

import torch

import content_pair_driver as driver
import empirical_content_inputs as inputs
from content_pair_measurement import _snapshot, _difference, _versions, _MeasurementComplete

SOURCE, TARGET = "sub-03", "sub-01"
DEVICE = "cuda:0"
UPDATES, BLOCK_UNITS, ROW_CHUNK_SIZE = 4, 256, 512
OPERATION = "measure_content_pair_cuda_four_updates"
CACHE_OPERATION = "measure_content_pair_cuda_cached_four_updates"
CACHE_MODE = "bounded_channel_two_pass_cache"
CACHE_LIMITS = {"maximum_channel_bytes": 1073741824, "maximum_retained_bytes": 4294967296}


def _cached_error_details(error, feature_read_mode):
    """Report exact fixed cache guards only, never arbitrary exception text."""
    if (feature_read_mode != CACHE_MODE or type(error) is not ValueError
            or len(error.args) != 1 or type(error.args[0]) is not str or len(error.args[0]) > 128):
        return {}
    known = {
        "explicit_channel_budget_check_required",
        "admitted_full_source_feature_and_flag_shapes_required",
        "selected_channel_and_flags_exceed_1GiB_before_read",
        "owned_bounded_contiguous_selected_channel_required",
        "owned_bounded_selected_channel_flag_bytes_required",
        "selected_channel_and_flags_exceed_1GiB",
        "contiguous_channel_reader_closed",
        "feature_block_must_match_selected_channel",
        "bounded_selected_channel_block_required",
        "exact_selected_channel_F_C_coordinates_required",
        "nonfinite required source-feature flag",
        "nonfinite required source feature",
        "fixed positive target-decoder feature scales required",
        "nonfinite target-standardized source feature",
        "existing_admitted_provider_and_checks_required",
        "positive_retained_cache_bound_at_most_4GiB_required",
        "positive_complete_row_count_required",
        "unique_layer_roster_required",
        "ordered_contiguous_LossBlock_required",
        "admitted_NumPy_LossBlock_arrays_required",
        "complete_dtype_shape_and_active_support_required",
        "retained_cache_byte_cap_before_copy",
        "exact_owning_cache_copies_required",
        "cache_closed_or_failed",
        "one_exhausted_cache_pass_at_a_time_required",
        "explicit_ActiveLayer_required",
        "fixed_layer_pass_order_required",
        "complete_active_layer_before_cache_replay_required",
        "same_iteration_layer_channel_units_required",
        "complete_original_active_layer_size_required",
    }
    return {"error_code": error.args[0]} if error.args[0] in known else {}


def _options(reserved_final_path, feature_read_mode="bounded_column_stripe"):
    path = Path(reserved_final_path)
    if not path.is_absolute() or path.suffix != ".pt":
        raise ValueError("explicit_absolute_reserved_final_pt_path_required")
    if feature_read_mode not in ("bounded_column_stripe", CACHE_MODE):
        raise ValueError("fixed_CUDA_feature_read_mode_required")
    options = {"device": DEVICE, "block_units": BLOCK_UNITS,
        "row_chunk_size": ROW_CHUNK_SIZE, "backend": "sparse", "updates": UPDATES,
        "source_rows": 4712, "layer_count": 19, "final_path": path, "maximum_bytes": 1}
    if feature_read_mode == CACHE_MODE:
        options.update(feature_read_mode=CACHE_MODE, **CACHE_LIMITS)
    return MappingProxyType(options)


def _cuda_snapshot():
    # Complete queued work before the host clock and allocator counters. These
    # are Torch allocator measurements, not all physical driver/library usage.
    torch.cuda.synchronize(0)
    return {**_snapshot(), "gpu_allocator": {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}}


def _cuda_difference(start, end):
    return {**_difference(start, end), "gpu_allocator": dict(end["gpu_allocator"])}


def _runtime():
    versions = _versions()
    if versions["torch_intraop_threads"] != 1 or versions["torch_interop_threads"] != 1:
        raise ValueError("measurement_requires_one_Torch_intraop_and_interop_thread")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"):
        raise ValueError("one_visible_CUDA_device_float32_TF32_off_no_AMP_required")
    properties = torch.cuda.get_device_properties(0)
    signature = inspect.signature(torch.optim.Adam).parameters
    defaults = {name: signature[name].default if name in signature else "not_exposed"
                for name in ("eps", "weight_decay", "amsgrad", "maximize", "foreach",
                             "capturable", "differentiable", "fused")}
    return {**versions, "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "device": DEVICE,
        "device_name": properties.name, "device_total_memory_bytes": int(properties.total_memory),
        "device_capability": [int(properties.major), int(properties.minor)],
        "matmul_allow_TF32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_TF32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "AMP_enabled": bool(torch.is_autocast_enabled()),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "Adam_declared_configuration": {"lr": driver.training.LEARNING_RATE,
                                        "betas": [0.5, 0.999], **defaults}}


def measure_content_pair_cuda(execution_context, *, require_admitted_context,
                              prepare_inputs, budget, reserved_final_path,
                              feature_read_mode="bounded_column_stripe"):
    """Measure four full CUDA Adam/LR updates, stopping before any final copy.

    The caller verifies the actual GPU allocation, device/host memory caps and
    deadline before this function. Explicit operation admission precedes input
    preparation and all CUDA inspection here. No measurement can become a
    production fit or authorize a different device/backend on failure.
    """
    if not callable(prepare_inputs):
        raise ValueError("explicit_admitted_input_preparation_callback_required")
    options = _options(reserved_final_path, feature_read_mode)
    operation = CACHE_OPERATION if feature_read_mode == CACHE_MODE else OPERATION
    identity = inputs._admit(execution_context, require_admitted_context,
        operation=operation, source=SOURCE, target=TARGET, budget=budget,
        execution_options=options)
    provider_options = {}
    if feature_read_mode == CACHE_MODE:
        # This import and factory selection follow explicit cached-operation
        # admission; the historical route imports or supplies neither.
        import content_channel_cache
        if (content_channel_cache.MAX_CHANNEL_BYTES != CACHE_LIMITS["maximum_channel_bytes"]
                or content_channel_cache.MAX_RETAINED_BYTES != CACHE_LIMITS["maximum_retained_bytes"]):
            raise ValueError("fixed_admitted_content_cache_limits_required")
        provider_options["provider_factory"] = content_channel_cache.make_cached_provider
    path = driver.final._file_context(options["final_path"], budget, options["maximum_bytes"])
    if any(os.path.lexists(value) for value in (path, path.with_name(path.name + ".partial"))):
        raise FileExistsError("reserved_measurement_destination_already_exists")

    start = _cuda_snapshot()
    report = {"kind": "ep21_content_pair_cuda_resource_measurement", "status": "incomplete",
        "source": SOURCE, "target": TARGET, "pair_index": 4, "requested_updates": UPDATES,
        "completed_updates": 0, "source_rows": 4712, "layers_per_update": 19,
        "device": DEVICE, "backend": "sparse", "block_units": BLOCK_UNITS, "row_chunk_size": ROW_CHUNK_SIZE,
        "allocation_identity": {"job_id": identity[0], "hostname": identity[1]},
        "actual_versions": None, "input_preparation_callback_seconds": None,
        "updates": [], "completed_update_logical_feature_bytes": 0,
        "partial_update_logical_feature_bytes": None, "production_iterations": 1024,
        "production_fit_completed": False, "model_published": False,
        "snapshot_and_write_cost_measured": False, "resumable": False,
        "initial_peak_rss_bytes": start["peak_rss_bytes"], "initial_gpu_allocator": None,
        "kernel_io_counters_start": start["kernel_io"], "error_type": None,
        "limitations": [
            "Four complete updates are a disposable measurement, not a production fit or resumable prefix.",
            "The same full rows, canonical initialization, channel schedule and sparse arithmetic are retained.",
            "CPU/CUDA floating-point or repeated-CUDA bitwise equivalence is not claimed.",
            "GPU timings synchronize cuda:0; update one includes pair setup and first Adam allocation.",
            "GPU peaks reset once before input preparation and retain any existing allocator baseline.",
            "GPU counters cover the Torch allocator, not all physical driver/library GPU consumption.",
            "Adam flags report constructor defaults; None does not identify the internal selected kernel.",
            "Host peak RSS is process-lifetime; kernel I/O counters are not a physical filesystem meter.",
            "Logical feature bytes cover completed updates only; partial reads and final copy/write are unmeasured.",
        ]}
    stage = "runtime_configuration"
    previous = None
    logical_bytes = ()
    gpu_ready = False

    def iteration_finished(event):
        nonlocal previous
        completed, iteration = event.get("completed_iterations"), event.get("iteration")
        expected = len(report["updates"]) + 1
        if (type(completed) is not int or type(iteration) is not int
                or completed != expected or iteration != expected - 1 or completed > UPDATES):
            raise ValueError("noncanonical_measurement_iteration_callback")
        # Record the confirmed committed step before a memory/deadline refusal.
        report["completed_updates"] = completed
        now = _cuda_snapshot()
        report["updates"].append({"completed_update": completed, **_cuda_difference(previous, now),
            "includes_pair_setup_and_first_Adam_allocation": completed == 1,
            "logical_feature_bytes": logical_bytes[iteration]})
        report["completed_update_logical_feature_bytes"] += logical_bytes[iteration]
        previous = now
        budget.check()
        if completed == UPDATES:
            raise _MeasurementComplete()

    try:
        report["actual_versions"] = _runtime()
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        gpu_ready = True
        initial_gpu = _cuda_snapshot()
        report["initial_gpu_allocator"] = dict(initial_gpu["gpu_allocator"])
        budget.check()
        stage = "input_preparation"
        preparation_start = time.monotonic()
        bound = prepare_inputs()
        torch.cuda.synchronize(0)
        report["input_preparation_callback_seconds"] = time.monotonic() - preparation_start
        if (not isinstance(bound, inputs.BoundContentInputs)
                or bound.execution_context is not execution_context
                or bound.require_admitted_context is not require_admitted_context
                or bound.budget is not budget or bound.allocation_identity != identity
                or bound.source.axis.participant != SOURCE or bound.target != TARGET
                or bound.source.normalized_X.shape != (4712, 12897)
                or len(bound.pair.target_axis.eligible_native_ids) != 12294):
            raise ValueError("measurement_requires_the_exact_full_admitted_pair_context")
        initial = bound.panel.initializations[4]
        if initial.pair_index != 4 or (initial.source, initial.target) != (SOURCE, TARGET):
            raise ValueError("canonical_pair_four_initialization_required")
        inputs._admit(execution_context, require_admitted_context, operation=operation,
            source=SOURCE, target=TARGET, budget=budget, execution_options=options,
            allocation_identity=identity)
        if inputs._read_loss_correction() != bound.loss_correction:
            raise ValueError("loss_correction_changed_after_measurement_input_binding")
        logical_bytes = tuple(2 * driver.training.TRAINING_ROWS * 4 * sum(
            driver.interface.active_layer_units(layer, channel)
            for layer, channel in bound.panel.channel_schedule.blocks(index)) for index in range(UPDATES))
        stage = "pair_setup_and_updates"
        previous = _cuda_snapshot()
        driver.train_and_publish_content_pair(bound.panel, bound.source, target=TARGET,
            budget=budget, device=DEVICE, block_units=BLOCK_UNITS, row_chunk_size=ROW_CHUNK_SIZE,
            backend="sparse", final_path=path, maximum_bytes=1, on_iteration=iteration_finished,
            **provider_options)
        raise RuntimeError("driver_returned_without_the_measurement_stop")
    except _MeasurementComplete:
        report["status"] = "complete_measurement"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["failed_stage"] = stage
        report.update(_cached_error_details(error, feature_read_mode))
    # An asynchronous CUDA failure can make later synchronization fail too. Keep
    # the original failure and confirmed counters rather than losing the report.
    try:
        finish = _cuda_snapshot() if gpu_ready else _snapshot()
        report["total"] = _cuda_difference(start, finish) if gpu_ready else _difference(start, finish)
    except Exception as error:
        finish = _snapshot()
        report["total"] = _difference(start, finish)
        report["gpu_final_snapshot_error_type"] = type(error).__name__
        if report["status"] == "complete_measurement":
            report.update(status="incomplete", error_type=type(error).__name__, failed_stage="final_GPU_snapshot")
    report["kernel_io_counters_end"] = finish["kernel_io"]
    return report
