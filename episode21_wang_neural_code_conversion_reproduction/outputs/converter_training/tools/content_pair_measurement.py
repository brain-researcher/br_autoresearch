"""UNACTIVATED four-update resource measurement; no grant or output writer.

The unchanged full-pair driver is interrupted by its existing iteration
callback after four complete Adam/LR updates. This is an explicitly separate
measurement operation, never a shortened production fit or resumable model.
It requires its own external admission before input preparation. The current
decoder grant does not admit it. Production remains all 1,024 updates.

Only operational metadata is returned. Training loss is neither copied nor
reported; no model, optimizer, response, prediction or feature value is saved.
The reserved final destination is checked by the existing driver, but the
callback raises before either final-artifact preparation or publication.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import resource
import time
from types import MappingProxyType

import content_pair_driver as driver
import empirical_content_inputs as inputs


SOURCE = "sub-03"
TARGET = "sub-01"
UPDATES = 4
BLOCK_UNITS = 256
ROW_CHUNK_SIZE = 512
OPERATION = "measure_content_pair_four_updates"
_IO_KEYS = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes",
            "cancelled_write_bytes")


class _MeasurementComplete(Exception):
    """Private control flow after update four, before final publication."""


def _kernel_io():
    """Bounded counters for this process; not a filesystem traffic meter."""
    try:
        with open("/proc/self/io", "rb") as stream:
            raw = stream.read(16385)
        if len(raw) > 16384:
            return None
        fields = dict(line.split(":", 1) for line in raw.decode("ascii").splitlines())
        return {key: int(fields[key].strip()) for key in _IO_KEYS}
    except (OSError, UnicodeError, ValueError, KeyError):
        return None


def _snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Sherlock is Linux: ru_maxrss is KiB and is a process-lifetime high-water
    # mark, including preparation/imports, not an independently reset step peak.
    return {"monotonic": time.monotonic(), "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime, "peak_rss_bytes": int(usage.ru_maxrss) * 1024,
            "kernel_io": _kernel_io()}


def _difference(start, end):
    before, after = start["kernel_io"], end["kernel_io"]
    return {"elapsed_seconds": end["monotonic"] - start["monotonic"],
            "user_cpu_seconds": end["user_seconds"] - start["user_seconds"],
            "system_cpu_seconds": end["system_seconds"] - start["system_seconds"],
            "peak_rss_bytes": end["peak_rss_bytes"],
            "kernel_io_counter_delta": (None if before is None or after is None else
                                        {key: after[key] - before[key] for key in _IO_KEYS})}


def _versions():
    import h5py
    import numpy
    import scipy
    import torch
    return {"python": platform.python_version(), "numpy": str(numpy.__version__),
            "scipy": str(scipy.__version__), "h5py": str(h5py.__version__),
            "torch": str(torch.__version__), "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def _options(reserved_final_path):
    path = Path(reserved_final_path)
    if not path.is_absolute() or path.suffix != ".pt":
        raise ValueError("explicit_absolute_reserved_final_pt_path_required")
    return MappingProxyType({"device": "cpu", "block_units": BLOCK_UNITS,
        "row_chunk_size": ROW_CHUNK_SIZE, "backend": "sparse", "updates": UPDATES,
        "source_rows": driver.training.TRAINING_ROWS, "layer_count": len(driver.schedule.LAYER_ORDER),
        "final_path": path, "maximum_bytes": 1})


def measure_content_pair(execution_context, *, require_admitted_context,
                         prepare_inputs, budget, reserved_final_path):
    """Measure the fixed full-size pair through four updates, without saving it.

    ``prepare_inputs()`` returns BoundContentInputs from the existing binder;
    the callback may reuse a previously admitted value, in which case its time
    does not measure the earlier preparation. The prospective profile is one
    CPU/16GiB/two hours, but this library selects no allocation. Its separate
    caller-owned budget enforces the admitted deadline and memory bound.

    Initial admission failures raise before preparation. Subsequent ordinary
    failures return an incomplete operational record with only exception type;
    no retry or resumable state exists. Process kill/OOM can prevent any report.
    Update-one timing includes pair checks, model/tensor/provider setup and
    first Adam allocation. Updates two through four are callback-to-callback
    durations, including the small measurement callback overhead. The exact
    logical feature-byte count covers completed updates only; partial-update
    reads and actual physical filesystem traffic are not inferred.
    """
    if not callable(prepare_inputs):
        raise ValueError("explicit_admitted_input_preparation_callback_required")
    options = _options(reserved_final_path)
    identity = inputs._admit(execution_context, require_admitted_context,
        operation=OPERATION, source=SOURCE, target=TARGET, budget=budget,
        execution_options=options)
    # Fail before preparation if the never-published destination is occupied.
    path = driver.final._file_context(options["final_path"], budget, options["maximum_bytes"])
    if any(os.path.lexists(value) for value in (path, path.with_name(path.name + ".partial"))):
        raise FileExistsError("reserved_measurement_destination_already_exists")

    start = _snapshot()
    report = {"kind": "ep21_content_pair_resource_measurement", "status": "incomplete",
        "source": SOURCE, "target": TARGET, "requested_updates": UPDATES,
        "completed_updates": 0, "source_rows": options["source_rows"],
        "layers_per_update": options["layer_count"], "device": "cpu", "backend": "sparse",
        "block_units": BLOCK_UNITS, "row_chunk_size": ROW_CHUNK_SIZE,
        "allocation_identity": {"job_id": identity[0], "hostname": identity[1]},
        "actual_versions": None, "input_preparation_callback_seconds": None,
        "updates": [], "completed_update_logical_feature_bytes": 0,
        "partial_update_logical_feature_bytes": None,
        "production_iterations": driver.schedule.ITERATIONS,
        "production_fit_completed": False, "model_published": False,
        "snapshot_and_write_cost_measured": False, "resumable": False,
        "initial_peak_rss_bytes": start["peak_rss_bytes"],
        "kernel_io_counters_start": start["kernel_io"], "error_type": None,
        "limitations": [
            "Four updates are a separately admitted resource measurement, not a production fit.",
            "Preparation timing covers only this invocation of prepare_inputs.",
            "First-update time includes pair setup and Adam initialization.",
            "Later update times include callback overhead; channels follow the unchanged schedule.",
            "Peak RSS is the process-lifetime high-water mark, not a reset per-update peak.",
            "Kernel I/O counters can undercount network filesystem traffic and include metadata.",
            "Logical bytes count complete feature requests, not disk bytes or decoder/metadata traffic.",
            "Partial-update logical reads and final snapshot/write costs are unmeasured.",
            "No production walltime, memory guarantee, bitwise equivalence or scientific result follows.",
        ]}
    stage = "input_preparation"
    previous = None
    logical_bytes = ()

    def iteration_finished(event):
        nonlocal previous
        # Read only operation counters. In particular do not retain/copy the
        # incoming dictionary, training_loss or learning_rate_used.
        completed = event.get("completed_iterations")
        iteration = event.get("iteration")
        expected = len(report["updates"]) + 1
        if (type(completed) is not int or type(iteration) is not int
                or completed != expected or iteration != expected - 1 or completed > UPDATES):
            raise ValueError("noncanonical_measurement_iteration_callback")
        now = _snapshot()
        update = {"completed_update": completed, **_difference(previous, now),
                  "includes_pair_setup_and_first_Adam_allocation": completed == 1,
                  "logical_feature_bytes": logical_bytes[iteration]}
        report["updates"].append(update)
        report["completed_updates"] = completed
        report["completed_update_logical_feature_bytes"] += logical_bytes[iteration]
        previous = now
        # Adam and the LR update have already committed before this callback.
        # A newly exceeded deadline/RSS cap must not erase that confirmed work.
        budget.check()
        if completed == UPDATES:
            raise _MeasurementComplete()

    try:
        report["actual_versions"] = _versions()
        if (report["actual_versions"]["torch_intraop_threads"] != 1
                or report["actual_versions"]["torch_interop_threads"] != 1):
            raise ValueError("measurement_requires_one_Torch_intraop_and_interop_thread")
        preparation_start = time.monotonic()
        bound = prepare_inputs()
        report["input_preparation_callback_seconds"] = time.monotonic() - preparation_start
        if (not isinstance(bound, inputs.BoundContentInputs)
                or bound.execution_context is not execution_context
                or bound.require_admitted_context is not require_admitted_context
                or bound.budget is not budget or bound.allocation_identity != identity
                or bound.source.axis.participant != SOURCE or bound.target != TARGET):
            raise ValueError("measurement_requires_the_exact_admitted_pair_context_and_allocation")
        inputs._admit(execution_context, require_admitted_context, operation=OPERATION,
            source=SOURCE, target=TARGET, budget=budget, execution_options=options,
            allocation_identity=identity)
        if inputs._read_loss_correction() != bound.loss_correction:
            raise ValueError("loss_correction_changed_after_measurement_input_binding")
        # This is metadata arithmetic over the complete canonical schedule.
        # The existing full driver/loop still validate that schedule and every
        # actual provider pass; no generated array or empirical score is read.
        logical_bytes = tuple(2 * 4 * driver.training.TRAINING_ROWS * sum(
            driver.interface.active_layer_units(layer, channel)
            for layer, channel in bound.panel.channel_schedule.blocks(index))
            for index in range(UPDATES))
        stage = "pair_setup_and_updates"
        previous = _snapshot()
        driver.train_and_publish_content_pair(bound.panel, bound.source, target=TARGET,
            budget=budget, device="cpu", block_units=BLOCK_UNITS, row_chunk_size=ROW_CHUNK_SIZE,
            backend="sparse", final_path=path, maximum_bytes=1,
            on_iteration=iteration_finished)
        # Unreachable for the unchanged driver: the callback raises at four.
        raise RuntimeError("driver_returned_without_the_measurement_stop")
    except _MeasurementComplete:
        report["status"] = "complete_measurement"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["failed_stage"] = stage
    finish = _snapshot()
    report["total"] = _difference(start, finish)
    report["kernel_io_counters_end"] = finish["kernel_io"]
    return report
