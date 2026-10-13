#!/usr/bin/env python3
"""EP17 bounded full-row/column-stripe copying; full execution needs a measured bound.

pilot003 is fixed sub-01/B, all original rows, original voxels [0,512).
Full attempt002 requires successful pilot003, explicit --max-seconds and a
current-job capacity record. Root selects full execution separately in the
experiment log; this module never schedules, retries or releases readers.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import h5py
import numpy as np

import materialize_neural_roles as original
import materialize_neural_roles_full as row_copy
from neural_role_private_paths import private_directory

core = original.core
EPISODE, CACHE_PARENT = original.EPISODE, original.CACHE_PARENT
RUNS = EPISODE / "outputs/runs/role_materialization"
PRIVATE_PARENT = CACHE_PARENT / "private/role_materialization"
HEADERS = RUNS / "compute_job_47086577/all_source_layout_headers.json"
COLUMNS = 512
CHUNK_ROWS = 1024
FULL_STRIPES = 5169
FULL_METADATA_RESERVE = 1024**3
PILOT_METADATA_RESERVE = 16 * 1024**2
LAYOUT = "full_rows_column_stripes_v1"


def paths(mode):
    name = "pilot003" if mode == "pilot" else "attempt002"
    return PRIVATE_PARENT / name, RUNS / name


def local_dataset(handle, shape):
    core.require(isinstance(handle.get("modelmd", getlink=True), h5py.HardLink), "source_dataset_not_local_hardlink")
    dataset = handle["modelmd"]
    core.require(isinstance(dataset, h5py.Dataset) and not dataset.is_virtual and not dataset.external,
                 "source_dataset_indirection")
    core.require(tuple(dataset.shape) == tuple(shape) and dataset.dtype.kind == "f" and dataset.dtype.itemsize == 4,
                 "source_shape_or_dtype_changed")
    return dataset


def column_bound(shape, chunks, start, stop):
    core.require(tuple(chunks) == (shape[0], 1, 1, 1) and 0 <= start < stop <= shape[3],
                 "observed_column_layout_required")
    return math.prod(shape[:3]) * (stop - start) * 4


def saved_headers():
    record = original.load_json(HEADERS)
    core.require(record.get("scope") == "twelve_named_source_headers_only" and
                 record.get("neural_values_read") is False and len(record.get("products", [])) == 12,
                 "fixed_header_observation_required")
    result = {}
    for item in record["products"]:
        key = item["participant"], item["stage"]
        core.require(key not in result and key[0] in original.SHAPES and key[1] in row_copy.STAGES,
                     "header_identity_changed")
        shape = original.SHAPES[key[0]]
        core.require(item["source_relative"] == row_copy.source_relative(*key) and tuple(item["shape"]) == shape and
                     tuple(item["chunks"]) == (shape[0], 1, 1, 1) and item["dtype_str"] == "<f4" and
                     item["compression"] == "gzip" and item["compression_opts"] == 3,
                     "saved_column_header_changed")
        result[key] = item
    return result


def inspect_header(root, observed):
    subject, stage = observed["participant"], observed["stage"]
    path = row_copy.source_path(root, subject, stage)
    with h5py.File(path, "r", rdcc_nbytes=original.CACHE_BYTES) as handle:
        dataset = local_dataset(handle, observed["shape"])
        header = {"shape": list(dataset.shape), "dtype_str": dataset.dtype.str,
                  "chunks": list(dataset.chunks) if dataset.chunks else None,
                  "compression": dataset.compression, "compression_opts": dataset.compression_opts}
    core.require(all(header[key] == observed[key] for key in header), "source_header_changed_after_observation")
    return path


def map_payloads(routed):
    result = {}
    for role in original.ROLES:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=original.MAP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for subject in sorted(routed):
            selected = [row for row in routed[subject] if row["role"] == role]
            for local, row in enumerate(selected):
                writer.writerow({**row, "local_row_index_zero_based": local})
        result[role] = stream.getvalue().encode("utf-8")
    return result


def padded_bytes(rows, columns):
    if rows == 0:
        return 0
    cr, cc = min(CHUNK_ROWS, rows), min(COLUMNS, columns)
    return math.ceil(rows / cr) * math.ceil(columns / cc) * cr * cc * 4


def storage_plan(mode, routed, maps, preserved_bytes):
    count = 0
    for subject, rows in routed.items():
        width = COLUMNS if mode == "pilot" else original.SHAPES[subject][3]
        count += sum(padded_bytes(sum(row["role"] == role for row in rows), width) for role in original.ROLES)
    count *= 1 if mode == "pilot" else 3
    metadata = PILOT_METADATA_RESERVE if mode == "pilot" else FULL_METADATA_RESERVE
    # Engineering slack is not a proven HDF5 metadata bound; OutputBudget
    # independently enforces the hard actual-file-byte ceiling during writes.
    total = count + sum(map(len, maps.values())) + preserved_bytes + metadata + row_copy.REPORT_RESERVE
    cap = original.BYTE_CAP if mode == "pilot" else row_copy.FULL_CAP
    core.require(total <= cap, "padded_output_plan_exceeds_cap")
    return {"padded_array_payload_bytes": count, "exact_index_map_bytes": sum(map(len, maps.values())),
            "preserved_attempt_bytes": preserved_bytes, "engineering_metadata_slack_bytes": metadata,
            "terminal_report_reserve_bytes": row_copy.REPORT_RESERVE, "planned_bytes": total, "output_byte_cap": cap}


def failed_report_bytes():
    total = 0
    expected = {"pilot001": "private_directory_permissions_invalid", "pilot002": "source_layout_exceeds_decoded_bound"}
    for name, code in expected.items():
        path = RUNS / name / "pilot_report.json"
        report = original.load_json(path)
        core.require(report.get("status") == "stopped" and report.get("neural_read_attempted") is False and
                     report.get("neural_values_read") is False and report.get("errors") == [{"code": code}],
                     "preserved_preaccess_attempt_changed")
        core.require(path.is_file() and not path.is_symlink(), "preserved_report_not_ordinary")
        total += path.stat().st_size
    return total


def validate_completed_pilot(routed):
    private, public = paths("pilot")
    report_path = public / "pilot_report.json"
    report = original.load_json(report_path)
    n = original.SHAPES[original.SUBJECT][0]
    core.require(report.get("status") == "completed" and report.get("pilot_copy_complete") is True and
                 report.get("errors") == [] and report.get("copy_layout") == LAYOUT and
                 report.get("private_attempt") == str(private) and report.get("public_attempt") == str(public) and
                 report.get("role_assignment_attempt") == str(original.ASSIGNMENT) and
                 report.get("pilot_scope") == {"participant": original.SUBJECT, "stage": original.STAGE,
                                               "rows_start": 0, "rows_stop_exclusive": n,
                                               "voxels_start": 0, "voxels_stop_exclusive": COLUMNS} and
                 report.get("logical_source_payload_bytes") == n * COLUMNS * 4 and
                 report.get("all_copy_bytes_equal") is True and report.get("neural_read_completed") is True,
                 "completed_column_pilot_required")
    for field in ("source_open_header_seconds", "source_read_seconds", "route_write_readback_seconds", "elapsed_seconds", "peak_rss_kib"):
        value = report.get(field)
        core.require(type(value) in (float, int) and math.isfinite(value) and value > 0, "pilot_resource_observation_missing")
    core.require(report["peak_rss_kib"] < 2 * 1024**2, "pilot_exceeds_two_gib")
    outputs = report.get("products", [])
    core.require(len(outputs) == 1 and outputs[0].get("complete") is True and len(outputs[0].get("outputs", [])) == 3,
                 "pilot_products_incomplete")
    size = 0
    expected_maps = map_payloads({original.SUBJECT: routed[original.SUBJECT]})
    for role in original.ROLES:
        path = private / role / "source_index_map.csv"
        size += row_copy.ordinary_private_file(path, private)
        core.require(original.stage.bounded_read(path) == expected_maps[role], "pilot_index_map_changed")
        rows = sum(row["role"] == role for row in routed[original.SUBJECT])
        entries = [entry for entry in outputs[0]["outputs"] if entry.get("role") == role]
        core.require(len(entries) == 1 and entries[0].get("rows") == rows and entries[0].get("copy_bytes_equal") is True,
                     "pilot_role_scope_changed")
        if rows:
            array = private / role / original.SUBJECT / f"{original.STAGE}.h5"
            array_size = row_copy.ordinary_private_file(array, private)
            core.require(entries[0].get("array_file") == str(array) and entries[0].get("array_file_bytes") == array_size and
                         entries[0].get("shape") == [rows, 1, 1, COLUMNS] and entries[0].get("dtype_str") == "<f4",
                         "pilot_array_identity_changed")
            size += array_size
    core.require(size == report.get("current_attempt_private_output_bytes") and size <= original.BYTE_CAP,
                 "pilot_output_size_changed")
    return report, size + report_path.stat().st_size


def output_dataset(handle, shape, dtype):
    dataset = local_dataset(handle, shape)
    core.require(dataset.dtype.str == dtype and dataset.chunks == (min(CHUNK_ROWS, shape[0]), 1, 1, min(COLUMNS, shape[3])) and
                 dataset.compression is None and dataset.scaleoffset is None and not dataset.shuffle and not dataset.fletcher32,
                 "private_output_representation_changed")
    return dataset


def access(report, item, rows, start, stop, reused):
    now = original.utc_now()
    report["neural_read_attempted"] = report["neural_values_read"] = True
    if report["first_neural_access_utc"] is None:
        report["first_neural_access_utc"] = now
    audit = sum(row["role"] == "sealed_audit" for row in rows)
    if audit:
        report["raw_audit_bytes_accessed"] = True
        if report["first_raw_audit_access_utc"] is None:
            report["first_raw_audit_access_utc"] = now
    report["last_attempted_read_scope"] = {"participant": item["participant"], "stage": item["stage"],
                                            "rows_start": 0, "rows_stop_exclusive": len(rows),
                                            "voxels_start": start, "voxels_stop_exclusive": stop,
                                            "decoded_voxels_start": start, "decoded_voxels_stop_exclusive": stop,
                                            "from_preserved_pilot": reused}
    item["audit_source_rows_per_stripe"] = audit


def reused_stripe(rows, dtype, check):
    private, _ = paths("pilot")
    result = np.empty((len(rows), 1, 1, COLUMNS), dtype=np.dtype(dtype))
    for role in original.ROLES:
        indices = [i for i, row in enumerate(rows) if row["role"] == role]
        if not indices:
            continue
        check()
        path = private / role / original.SUBJECT / f"{original.STAGE}.h5"
        row_copy.ordinary_private_file(path, private)
        with h5py.File(path, "r", rdcc_nbytes=original.CACHE_BYTES) as handle:
            dataset = output_dataset(handle, (len(indices), 1, 1, COLUMNS), dtype)
            saved = dataset[...]
        core.require(saved.dtype.str == dtype, "pilot_read_representation_changed")
        result.view(np.uint8)[indices] = saved.view(np.uint8)
        del saved
    return result


def verify_stripe(path, shape, dtype, start, selected):
    with h5py.File(path, "r", rdcc_nbytes=original.CACHE_BYTES) as handle:
        dataset = output_dataset(handle, shape, dtype)
        copied = dataset[:, :, :, start:start + selected.shape[3]]
    core.require(copied.shape == selected.shape and copied.dtype.str == selected.dtype.str and
                 np.array_equal(copied.view(np.uint8), selected.view(np.uint8)), "private_copy_byte_mismatch")


def copy_product(root, observed, rows, mode, private, budget, report, check):
    subject, stage = observed["participant"], observed["stage"]
    columns = COLUMNS if mode == "pilot" else observed["shape"][3]
    item = {"participant": subject, "stage": stage, "shape": [len(rows), 1, 1, columns],
            "source_voxel_count": observed["shape"][3], "dtype_str": observed["dtype_str"],
            "logical_source_payload_bytes": 0, "source_columns_read": 0, "pilot_columns_reused": 0,
            "verified_columns": 0, "complete": False, "outputs": []}
    report["products"].append(item)
    check()
    path = inspect_header(root, observed)
    outputs = {}
    for role in original.ROLES:
        indices = [i for i, row in enumerate(rows) if row["role"] == role]
        entry = {"role": role, "rows": len(indices), "array_file": None, "empty_role": not indices,
                 "verified_columns": 0, "copy_bytes_equal": not indices}
        item["outputs"].append(entry)
        if not indices:
            continue
        directory = private / role / subject
        private_directory(directory, private)
        partial = directory / f"{stage}.h5.partial"
        shape = (len(indices), 1, 1, columns)
        with budget.write(partial), original.new_private_file(partial) as stream:
            with h5py.File(stream, "w", rdcc_nbytes=original.CACHE_BYTES) as handle:
                handle.create_dataset("modelmd", shape=shape, dtype=np.dtype(observed["dtype_str"]),
                                      chunks=(min(CHUNK_ROWS, len(indices)), 1, 1, min(COLUMNS, columns)))
        outputs[role] = (partial, shape, indices, entry)
    with h5py.File(path, "r", rdcc_nbytes=original.CACHE_BYTES) as source:
        dataset = local_dataset(source, observed["shape"])
        core.require(dataset.dtype.str == observed["dtype_str"] and list(dataset.chunks) == observed["chunks"] and
                     dataset.compression == observed["compression"] and dataset.compression_opts == observed["compression_opts"],
                     "source_header_changed_before_stripe")
        for start in range(0, columns, COLUMNS):
            stop = min(start + COLUMNS, columns)
            check()
            bound = column_bound(observed["shape"], observed["chunks"], start, stop)
            core.require(bound <= original.BYTE_CAP, "column_decoded_work_exceeds_cap")
            reused = mode == "full" and subject == original.SUBJECT and stage == original.STAGE and start == 0
            access(report, item, rows, start, stop, reused)
            began = time.monotonic()
            block = reused_stripe(rows, observed["dtype_str"], check) if reused else dataset[:, :, :, start:stop]
            report["pilot_read_seconds" if reused else "source_read_seconds"] += time.monotonic() - began
            core.require(block.shape == (len(rows), 1, 1, stop - start) and block.dtype.str == observed["dtype_str"],
                         "source_read_representation_changed")
            if reused:
                item["pilot_columns_reused"] += stop - start
            else:
                item["source_columns_read"] += stop - start
                item["logical_source_payload_bytes"] += block.nbytes
            check()
            began = time.monotonic()
            for partial, shape, indices, entry in outputs.values():
                check()
                selected = block[indices]
                with budget.write(partial), h5py.File(partial, "r+", rdcc_nbytes=original.CACHE_BYTES) as target:
                    output_dataset(target, shape, observed["dtype_str"])[:, :, :, start:stop] = selected
                    target.flush()
                verify_stripe(partial, shape, observed["dtype_str"], start, selected)
                entry["verified_columns"] += stop - start
                del selected
            del block
            item["verified_columns"] += stop - start
            report["route_write_readback_seconds"] += time.monotonic() - began
    check()
    core.require(item["verified_columns"] == columns and sum(entry["rows"] for entry in item["outputs"]) == len(rows),
                 "column_partition_incomplete")
    for partial, shape, indices, entry in outputs.values():
        core.require(entry["verified_columns"] == columns, "role_column_partition_incomplete")
        check()
        final = partial.with_suffix("")
        budget.publish(partial, final)
        entry.update({"array_file": str(final), "array_file_bytes": final.stat().st_size,
                      "shape": list(shape), "dtype_str": observed["dtype_str"], "copy_bytes_equal": True})
    item["complete"] = True


def run(mode, capacity_path, max_seconds=None):
    core.require(mode in ("pilot", "full"), "invalid_copy_mode")
    core.require((mode == "pilot" and max_seconds is None) or
                 (mode == "full" and type(max_seconds) is int and 1 <= max_seconds <= 240 * 3600 - 70),
                 "explicit_full_time_bound_required")
    duration = 120 if mode == "pilot" else max_seconds
    private, public = paths(mode)
    public.parent.mkdir(parents=True, exist_ok=True)
    public.mkdir(exist_ok=False)
    report = original.initial_report()
    report.update({"mode": mode, "copy_layout": LAYOUT, "full_copy_complete": False,
                   "private_attempt": str(private), "public_attempt": str(public), "products": [],
                   "internal_seconds": duration, "external_seconds_required": duration + 60,
                   "output_byte_cap": original.BYTE_CAP if mode == "pilot" else row_copy.FULL_CAP,
                   "source_read_seconds": 0.0, "pilot_read_seconds": 0.0, "route_write_readback_seconds": 0.0,
                   "logical_source_payload_bytes": 0, "source_open_header_seconds": 0.0,
                   "neural_access_flag_is_conservative_if_read_fails": True})
    if mode == "pilot":
        report["pilot_scope"] = {"participant": original.SUBJECT, "stage": original.STAGE,
                                 "rows_start": 0, "rows_stop_exclusive": original.SHAPES[original.SUBJECT][0],
                                 "voxels_start": 0, "voxels_stop_exclusive": COLUMNS}
    started = time.monotonic()
    budget = None
    def check():
        core.require(time.monotonic() < started + duration, "attempt_deadline")
    def alarm(_signum, _frame):
        raise core.Refusal("attempt_deadline")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(duration)
    try:
        root, routed = row_copy.load_full_plan()
        check()
        observed = saved_headers()
        chosen = [(original.SUBJECT, original.STAGE)] if mode == "pilot" else sorted(observed)
        if mode == "full":
            completed, pilot_bytes = validate_completed_pilot(routed)
            report["completed_pilot_costs"] = {key: completed[key] for key in
                                                ("source_open_header_seconds", "source_read_seconds", "route_write_readback_seconds",
                                                 "elapsed_seconds", "peak_rss_kib")}
        else:
            pilot_bytes = 0
            routed = {original.SUBJECT: routed[original.SUBJECT]}
        preserved = failed_report_bytes() + pilot_bytes
        capacity = original.check_capacity(original.load_json(capacity_path), CACHE_PARENT, report["slurm_job_id"])
        core.require(min(capacity["filesystem_available_bytes"], capacity["quota_available_bytes"]) -
                     capacity["concurrent_reserved_bytes"] >= report["output_byte_cap"], "insufficient_capacity_after_reservations")
        report["capacity_observation"] = capacity
        maps = map_payloads(routed)
        report["storage_plan"] = storage_plan(mode, routed, maps, preserved)
        core.require("/*/outputs/cache/" in (EPISODE.parent / ".gitignore").read_text().splitlines(), "private_tree_not_ignored")
        core.require(not private.exists() and not private.is_symlink(), "private_attempt_already_exists")
        opened = time.monotonic()
        headers = []
        for key in chosen:
            check()
            header = observed[key]
            inspect_header(root, header)
            columns = COLUMNS if mode == "pilot" else header["shape"][3]
            for start in range(0, columns, COLUMNS):
                core.require(column_bound(header["shape"], header["chunks"], start, min(start + COLUMNS, columns)) <= original.BYTE_CAP,
                             "column_decoded_work_exceeds_cap")
            headers.append(header)
        report["source_open_header_seconds"] = time.monotonic() - opened
        report["source_header_preflight_complete"] = True
        report["source_headers"] = headers
        stripes = sum(math.ceil((COLUMNS if mode == "pilot" else header["shape"][3]) / COLUMNS) for header in headers)
        core.require(stripes == (1 if mode == "pilot" else FULL_STRIPES), "full_column_scope_changed")
        report["planned_stripes_including_reused_pilot"] = stripes
        check()
        private_directory(private, CACHE_PARENT)
        budget = row_copy.OutputBudget(preserved, cap=report["output_byte_cap"])
        for role, payload in maps.items():
            directory = private / role
            private_directory(directory, private)
            partial = directory / "source_index_map.csv.partial"
            with budget.write(partial), original.new_private_file(partial) as stream:
                stream.write(payload)
            budget.publish(partial, directory / "source_index_map.csv")
        del maps
        for header in headers:
            copy_product(root, header, routed[header["participant"]], mode, private, budget, report, check)
        check()
        report["logical_source_payload_bytes"] = sum(item["logical_source_payload_bytes"] for item in report["products"])
        reused = sum(item["pilot_columns_reused"] for item in report["products"])
        core.require(len(report["products"]) == (1 if mode == "pilot" else 12) and
                     all(item["complete"] for item in report["products"]) and reused == (0 if mode == "pilot" else COLUMNS),
                     "copy_scope_incomplete")
        report.update({"all_copy_bytes_equal": True, "neural_read_completed": True, "status": "completed",
                       "pilot_copy_complete": mode == "pilot", "full_copy_complete": mode == "full",
                       "pilot_columns_reused": reused})
        check()
    except core.Refusal as error:
        report.update({"status": "stopped", "pilot_copy_complete": False, "full_copy_complete": False})
        report["errors"].append(error.detail)
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, OverflowError, RuntimeError, MemoryError):
        report.update({"status": "stopped", "pilot_copy_complete": False, "full_copy_complete": False})
        report["errors"].append({"code": "private_column_copy_processing_failure"})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
    report.update({"elapsed_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "downstream_use_requires_this_terminal_report_completed": True,
                   "completion_does_not_activate_reader_handoffs": True})
    if budget is not None:
        report["current_attempt_private_output_bytes"] = sum(budget.sizes.values())
        report["accounted_bytes_including_preserved_attempts"] = budget.total
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    core.require(len(encoded.encode("utf-8")) <= row_copy.REPORT_RESERVE, "public_report_byte_cap")
    filename = "pilot_report.json" if mode == "pilot" else "materialization_report.json"
    with (public / filename).open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({key: report[key] for key in ("status", "pilot_copy_complete", "full_copy_complete",
                                                  "neural_values_read", "raw_audit_bytes_accessed", "elapsed_seconds", "peak_rss_kib")}))
    return 0 if report["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--capacity-observation", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    core.require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "verified_compute_allocation_required")
    core.require(all(os.environ.get(name) == "1" for name in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")), "single_thread_environment_required")
    core.require(args.capacity_observation.is_file() and args.capacity_observation.resolve().is_relative_to(RUNS.resolve()),
                 "capacity_record_path_outside_scope")
    return run(args.mode, args.capacity_observation, args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
