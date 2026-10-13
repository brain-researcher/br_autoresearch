#!/usr/bin/env python3
"""Unselected EP17 full literal-copy implementation; run only after pilot review.

The coordinator must select a finite --max-seconds from actual pilot costs,
provide current capacity bound to the same Slurm job, and wrap this command in
an external timeout of max-seconds + 60, with a ten-second kill grace. This
module never dispatches jobs, retries, releases readers or evaluates a model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time

import h5py
import numpy as np

import materialize_neural_roles as pilot
from neural_role_private_paths import private_directory

core, stage, chronology = pilot.core, pilot.stage, pilot.chronology
EPISODE, CACHE_PARENT = pilot.EPISODE, pilot.CACHE_PARENT
PRIVATE = CACHE_PARENT / "private/role_materialization/attempt001"
PUBLIC = EPISODE / "outputs/runs/role_materialization/attempt001"
PILOT_PRIVATE, PILOT_PUBLIC = pilot.PRIVATE, pilot.PUBLIC
PILOT_REPORT = PILOT_PUBLIC / "pilot_report.json"
STAGES = {"B": "TYPEB_FITHRF.mat", "C": "TYPEC_FITHRF_GLMDENOISE.mat",
          "D": "TYPED_FITHRF_GLMDENOISE_RR.mat"}
FULL_CAP = 160 * 1024**3
REPORT_RESERVE = 1024**2
EXPECTED_PAYLOAD_BYTES = 132332156640
EXPECTED_BLOCKS = 2346


def select_pilot_attempt(name):
    global PILOT_PRIVATE, PILOT_PUBLIC, PILOT_REPORT
    core.require(name in ("pilot001", "pilot002"), "unsupported_pilot_attempt")
    PILOT_PRIVATE = CACHE_PARENT / "private/role_materialization" / name
    PILOT_PUBLIC = EPISODE / "outputs/runs/role_materialization" / name
    PILOT_REPORT = PILOT_PUBLIC / "pilot_report.json"


def load_full_plan():
    """Parse each frozen metadata input once; retain all four exact routings."""
    report = pilot.load_json(pilot.ASSIGNMENT / "assignment_report.json")
    core.require(report.get("status") == "completed" and report.get("metadata_assignments_complete") is True and
                 report.get("errors") == [] and report.get("preserved_metadata") ==
                 {"concepts": 720, "images": 4320, "presentations": 49980, "runs": 833} and
                 report.get("role_counts") == dict(zip(pilot.ROLES, (480, 120, 120))), "complete_full001_roles_required")
    basis = pilot.load_json(core.METADATA / "metadata_report.json")
    events = pilot.load_json(stage.EVENTS / "reconciliation_report.json")
    stage.check_reports(basis, events)
    _, runs = stage.table_rows(stage.bounded_read(core.METADATA / "run_metadata.csv"), stage.RUN_FIELDS)
    stage.check_roster(runs, events)
    _, original = stage.table_rows(stage.bounded_read(core.METADATA / "presentation_identity.csv"), stage.PRESENTATION_FIELDS)
    _, event_rows = stage.table_rows(stage.bounded_read(stage.EVENTS / "event_metadata.csv"), core.OUTPUT_FIELDS)
    core.require(len(original) == len(event_rows) == 49980, "fixed_presentation_count_changed")
    source_rows = stage.check_presentations(original, event_rows, runs)
    del original, event_rows
    eligible = pilot.load_json(chronology.QUALIFICATION / "qualification_report.json")
    core.require(eligible.get("status") == "completed" and eligible.get("eligibility_complete") is True and
                 eligible.get("eligible_720_concept_universe_verified") is True and eligible.get("errors") == [] and
                 eligible.get("qualified_concept_count") == 720 and eligible.get("qualified_image_count") == 3634 and
                 eligible.get("source_metadata_attempt") == str(core.METADATA) and
                 eligible.get("event_projection_attempt") == str(stage.EVENTS), "completed_eligibility_required")
    history = pilot.load_json(pilot.CHRONOLOGY / "chronology_report.json")
    core.require(history.get("status") == "completed" and history.get("errors") == [] and
                 all(history.get(flag) is True for flag in ("source_chronology_complete", "retained_half_qualification_complete",
                                                           "global_chronology_complete", "all_observed_ordinals_qualified")) and
                 history.get("preserved_presentations") == 49980 and history.get("preserved_runs") == 833 and
                 history.get("source_metadata_attempt") == str(core.METADATA) and
                 history.get("event_projection_attempt") == str(stage.EVENTS) and
                 history.get("eligibility_attempt") == str(chronology.QUALIFICATION), "completed_chronology_required")
    def table(path, fields):
        return stage.table_rows(stage.bounded_read(path), fields)[1]
    concepts = pilot.check_concepts(table(pilot.ASSIGNMENT / "concept_assignments.csv", pilot.CONCEPT_FIELDS))
    images = table(pilot.ASSIGNMENT / "image_assignments.csv", (*stage.IMAGE_FIELDS, *pilot.ROLE_FIELDS))
    core.require(len(images) == 4320 and sum(row["eligible_image"] == "True" for row in images) == 3634,
                 "assigned_image_counts_changed")
    routed = pilot.check_routing(concepts, images,
                                table(chronology.QUALIFICATION / "image_qualification.csv", stage.IMAGE_FIELDS),
                                table(pilot.ASSIGNMENT / "presentation_assignments.csv", (*pilot.PRESENTATION_FIELDS, *pilot.ROLE_FIELDS)),
                                table(pilot.CHRONOLOGY / "presentation_chronology.csv", pilot.PRESENTATION_FIELDS),
                                source_rows, {subject: shape[0] for subject, shape in pilot.SHAPES.items()})
    root = Path(os.environ["OAK"]) / pilot.SOURCE_SUFFIX
    core.require(basis.get("source") == str(root), "source_root_changed")
    for subject, shape in pilot.SHAPES.items():
        axes = next(person for person in basis["participants"] if person["participant"] == subject)["stage_axes"]
        for stage_name in STAGES:
            axis = axes[stage_name]
            core.require(axis["file"] == source_relative(subject, stage_name) and axis["dataset"] == "modelmd" and
                         tuple(axis["stored_shape"]) == shape and axis["dtype"] == "float32", "saved_source_header_changed")
    return root, routed


def source_relative(subject, stage_name):
    return f"THINGS/glmsingle/{subject}/glmsingle/output/T1w/{STAGES[stage_name]}"


def source_path(root, subject, stage_name):
    path = (root / source_relative(subject, stage_name)).resolve(strict=True)
    core.require(root.is_dir() and path.is_relative_to(root.resolve()) and path.is_file(), "source_resolution_outside_root")
    return path


def header_plan(handle, shape):
    dataset, header = pilot.checked_dataset(handle, shape, stop=min(pilot.ROWS, shape[0]))
    blocks = []
    for start in range(0, shape[0], pilot.ROWS):
        stop = min(start + pilot.ROWS, shape[0])
        bound = pilot.layout_bound(shape, dataset.chunks, dataset.dtype.itemsize, start=start, stop=stop)
        core.require(bound["decoded_work_bound_bytes"] <= pilot.BYTE_CAP, "source_layout_exceeds_decoded_bound")
        blocks.append({"start": start, "stop": stop, **bound})
    return {"shape": header["shape"], "dtype_str": header["dtype_str"], "chunks": header["chunks"], "blocks": blocks}


def preflight_sources(root, check):
    products = []
    for subject, shape in sorted(pilot.SHAPES.items()):
        for stage_name in STAGES:
            check()
            path = source_path(root, subject, stage_name)
            with h5py.File(path, "r", rdcc_nbytes=pilot.CACHE_BYTES) as handle:
                header = header_plan(handle, shape)
            products.append({"participant": subject, "stage": stage_name, "source_file": str(path), **header})
    core.require(len(products) == 12 and sum(len(item["blocks"]) for item in products) == EXPECTED_BLOCKS and
                 sum(math.prod(item["shape"]) * 4 for item in products) == EXPECTED_PAYLOAD_BYTES,
                 "full_source_scope_changed")
    return products


def ordinary_private_file(path, parent):
    core.require(path.is_relative_to(parent), "private_file_outside_attempt")
    directories = [parent]
    for part in path.parent.relative_to(parent).parts:
        directories.append(directories[-1] / part)
    for current in directories:
        info = current.lstat()
        core.require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700,
                     "private_directory_permissions_invalid")
    info = path.lstat()
    core.require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
                 "private_file_representation_invalid")
    return info.st_size


def validate_pilot(report, routed):
    """Operational report and index maps only; no pilot neural values here."""
    core.require(report.get("status") == "completed" and report.get("pilot_copy_complete") is True and
                 report.get("all_copy_bytes_equal") is True and report.get("neural_read_completed") is True and
                 report.get("errors") == [] and report.get("private_attempt") == str(PILOT_PRIVATE) and
                 report.get("public_attempt") == str(PILOT_PUBLIC) and report.get("role_assignment_attempt") == str(pilot.ASSIGNMENT) and
                 report.get("logical_source_payload_bytes") == pilot.ROWS * math.prod(pilot.SHAPES[pilot.SUBJECT][1:]) * 4,
                 "completed_fixed_pilot_required")
    core.require(report.get("logical_copied_scope") == {"participant": pilot.SUBJECT, "stage": pilot.STAGE,
                                                        "rows_start": 0, "rows_stop_exclusive": pilot.ROWS},
                 "pilot_scope_changed")
    for field in ("source_open_header_seconds", "source_read_seconds", "route_write_readback_seconds", "elapsed_seconds", "peak_rss_kib"):
        value = report.get(field)
        core.require(type(value) in (int, float) and math.isfinite(value) and value > 0, "pilot_resource_observation_missing")
    core.require(report["peak_rss_kib"] < 2 * 1024**2, "pilot_exceeds_two_gib")
    outputs = report.get("outputs", [])
    core.require(len(outputs) == 3 and {item["role"] for item in outputs} == set(pilot.ROLES), "pilot_role_products_changed")
    size = 0
    for role in pilot.ROLES:
        rows = [row for row in routed[pilot.SUBJECT][:pilot.ROWS] if row["role"] == role]
        entry = next(item for item in outputs if item["role"] == role)
        core.require(entry.get("participant") == pilot.SUBJECT and entry.get("stage") == pilot.STAGE and
                     entry.get("rows") == len(rows) and entry.get("empty_role") == (not rows) and
                     entry.get("copy_bytes_equal") is True, "pilot_role_scope_changed")
        mapping = PILOT_PRIVATE / role / "source_index_map.csv"
        size += ordinary_private_file(mapping, PILOT_PRIVATE)
        fields, copied_rows = stage.table_rows(stage.bounded_read(mapping), pilot.MAP_FIELDS)
        core.require(fields == list(pilot.MAP_FIELDS) and len(copied_rows) == len(rows), "pilot_index_map_changed")
        for local, (saved, original) in enumerate(zip(copied_rows, rows)):
            expected = {field: str(original.get(field, "")) for field in pilot.MAP_FIELDS}
            expected["local_row_index_zero_based"] = str(local)
            core.require(saved == expected, "pilot_index_map_changed")
        if rows:
            path = PILOT_PRIVATE / role / pilot.SUBJECT / f"{pilot.STAGE}.h5"
            actual_bytes = ordinary_private_file(path, PILOT_PRIVATE)
            size += actual_bytes
            core.require(entry.get("array_file") == str(path) and entry.get("array_file_bytes") == actual_bytes and
                         entry.get("shape") == [len(rows), *pilot.SHAPES[pilot.SUBJECT][1:]] and
                         entry.get("dtype_str") == report["source_layout"]["dtype_str"], "pilot_array_identity_changed")
        else:
            core.require(entry.get("array_file") is None, "empty_pilot_role_has_array")
    core.require(size == report.get("private_output_bytes") and size <= pilot.BYTE_CAP, "pilot_output_size_changed")
    return size + PILOT_REPORT.stat().st_size


class OutputBudget:
    """Track only known attempt files; enforce aggregate bound during each write."""
    def __init__(self, preserved_bytes, cap=FULL_CAP, report_reserve=REPORT_RESERVE):
        self.preserved_bytes, self.cap, self.report_reserve = preserved_bytes, cap, report_reserve
        self.sizes = {}
        core.require(0 <= preserved_bytes < cap - report_reserve, "full_output_byte_cap")

    @property
    def total(self):
        return self.preserved_bytes + sum(self.sizes.values())

    @contextmanager
    def write(self, path):
        previous = self.sizes.get(path, 0)
        core.require((not path.exists() and previous == 0) or
                     (path.is_file() and not path.is_symlink() and path.stat().st_size == previous), "output_file_size_changed")
        allowed = previous + self.cap - self.report_reserve - self.total
        core.require(allowed > previous, "full_output_byte_cap")
        old = resource.getrlimit(resource.RLIMIT_FSIZE)
        soft = allowed if old[1] == resource.RLIM_INFINITY else min(allowed, old[1])
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, old[1]))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, old)
            if path.exists():
                info = path.lstat()
                core.require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
                             "private_file_representation_invalid")
                self.sizes[path] = info.st_size
            core.require(self.total + self.report_reserve <= self.cap, "full_output_byte_cap")

    def publish(self, partial, final):
        pilot.publish_file(partial, final)
        self.sizes[final] = self.sizes.pop(partial)


def write_full_maps(routed, private, budget, check):
    for role in pilot.ROLES:
        check()
        directory = private / role
        private_directory(directory, private)
        partial = directory / "source_index_map.csv.partial"
        with budget.write(partial), pilot.new_private_file(partial) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=pilot.MAP_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for subject in sorted(routed):
                    selected = [row for row in routed[subject] if row["role"] == role]
                    for local, row in enumerate(selected):
                        writer.writerow({**row, "local_row_index_zero_based": local})
        budget.publish(partial, directory / "source_index_map.csv")


def note_access(report, item, rows, block, *, reused=False):
    now = pilot.utc_now()
    first, stop = block["start"], block["stop"]
    dfirst, dstop = (first, stop) if reused else (block["decoded_row_start"], block["decoded_row_stop_exclusive"])
    audit = sum(row["role"] == "sealed_audit" for row in rows[dfirst:dstop])
    report["neural_read_attempted"] = report["neural_values_read"] = True
    if report["first_neural_access_utc"] is None:
        report["first_neural_access_utc"] = now
    if audit:
        report["raw_audit_bytes_accessed"] = True
        if report["first_raw_audit_access_utc"] is None:
            report["first_raw_audit_access_utc"] = now
    report["last_attempted_read_scope"] = {"participant": item["participant"], "stage": item["stage"],
                                            "rows_start": first, "rows_stop_exclusive": stop,
                                            "decoded_rows_start": dfirst, "decoded_rows_stop_exclusive": dstop,
                                            "from_preserved_pilot": reused, "all_source_voxel_columns": True}
    item["logical_audit_rows_attempted"] += sum(row["role"] == "sealed_audit" for row in rows[first:stop])
    item["decoded_audit_row_accesses_attempted"] += audit


def read_pilot_block(product, rows, report, item, check):
    block_plan = product["blocks"][0]
    note_access(report, item, rows, block_plan, reused=True)
    shape = (pilot.ROWS, *product["shape"][1:])
    result = np.empty(shape, dtype=np.dtype(product["dtype_str"]))
    for role in pilot.ROLES:
        check()
        indices = [i for i, row in enumerate(rows[:pilot.ROWS]) if row["role"] == role]
        if not indices:
            continue
        path = PILOT_PRIVATE / role / pilot.SUBJECT / f"{pilot.STAGE}.h5"
        ordinary_private_file(path, PILOT_PRIVATE)
        with h5py.File(path, "r", rdcc_nbytes=pilot.CACHE_BYTES) as handle:
            dataset, _ = pilot.checked_dataset(handle, (len(indices), *shape[1:]), stop=len(indices))
            core.require(dataset.dtype.str == product["dtype_str"] and dataset.compression is None and
                         dataset.scaleoffset is None and dataset.chunks == (min(16, len(indices)), 1, 1, shape[3]),
                         "pilot_array_representation_changed")
            selected = dataset[...]
        core.require(selected.dtype.str == result.dtype.str and selected.shape == (len(indices), *shape[1:]),
                     "pilot_read_representation_changed")
        # Byte-view assignment avoids a float conversion, preserving NaN bits.
        result.view(np.uint8)[indices] = selected.view(np.uint8)
        del selected
    return result


def verify_segment(path, shape, dtype, start, selected):
    with h5py.File(path, "r", rdcc_nbytes=pilot.CACHE_BYTES) as handle:
        dataset, _ = pilot.checked_dataset(handle, shape, stop=min(pilot.ROWS, shape[0]))
        core.require(dataset.dtype.str == dtype and dataset.compression is None and dataset.scaleoffset is None and
                     dataset.shuffle is False and dataset.fletcher32 is False and
                     dataset.chunks == (min(16, shape[0]), 1, 1, shape[3]), "private_output_representation_changed")
        copied = dataset[start:start + selected.shape[0], :, :, :]
        core.require(copied.dtype.str == selected.dtype.str and copied.shape == selected.shape and
                     np.array_equal(copied.view(np.uint8), selected.view(np.uint8)), "private_copy_byte_mismatch")


def copy_product(root, product, rows, private, budget, report, check):
    subject, stage_name = product["participant"], product["stage"]
    item = {"participant": subject, "stage": stage_name, "source_file": product["source_file"],
            "shape": product["shape"], "dtype_str": product["dtype_str"],
            "original_voxel_index_start": 0, "original_voxel_index_stop_exclusive": product["shape"][3],
            "source_rows_read": 0, "pilot_rows_reused": 0, "verified_rows": 0,
            "logical_audit_rows_attempted": 0, "decoded_audit_row_accesses_attempted": 0,
            "decoded_source_work_bytes": 0, "source_read_seconds": 0.0,
            "pilot_read_seconds": 0.0, "route_write_readback_seconds": 0.0,
            "complete": False, "outputs": []}
    report["products"].append(item)
    check()
    path = source_path(root, subject, stage_name)
    core.require(str(path) == product["source_file"], "source_resolution_changed")
    with h5py.File(path, "r", rdcc_nbytes=pilot.CACHE_BYTES) as source:
        observed = header_plan(source, tuple(product["shape"]))
        core.require(observed == {key: product[key] for key in observed}, "source_header_changed_after_preflight")
        dataset = source["modelmd"]
        outputs = {}
        for role in pilot.ROLES:
            check()
            count = sum(row["role"] == role for row in rows)
            entry = {"role": role, "rows": count, "empty_role": count == 0, "array_file": None,
                     "verified_rows": 0, "copy_bytes_equal": False}
            item["outputs"].append(entry)
            if count == 0:
                entry["copy_bytes_equal"] = True
                continue
            directory = private / role / subject
            private_directory(directory, private)
            partial = directory / f"{stage_name}.h5.partial"
            shape = (count, *product["shape"][1:])
            with budget.write(partial), pilot.new_private_file(partial) as stream:
                with h5py.File(stream, "w", rdcc_nbytes=pilot.CACHE_BYTES) as target:
                    target.create_dataset("modelmd", shape=shape, dtype=np.dtype(product["dtype_str"]),
                                          chunks=(min(16, count), 1, 1, shape[3]))
            outputs[role] = (partial, shape, entry)
        for plan in product["blocks"]:
            check()
            start, stop = plan["start"], plan["stop"]
            reused = subject == pilot.SUBJECT and stage_name == pilot.STAGE and start == 0
            opened = time.monotonic()
            if reused:
                block = read_pilot_block(product, rows, report, item, check)
                item["pilot_rows_reused"] += stop - start
                item["pilot_read_seconds"] += time.monotonic() - opened
            else:
                note_access(report, item, rows, plan)
                block = dataset[start:stop, :, :, :]
                item["source_rows_read"] += stop - start
                item["decoded_source_work_bytes"] += plan["decoded_work_bound_bytes"]
                item["source_read_seconds"] += time.monotonic() - opened
            core.require(block.shape == (stop - start, *product["shape"][1:]) and block.dtype.str == product["dtype_str"],
                         "source_read_representation_changed")
            check()
            opened = time.monotonic()
            for role, (partial, shape, entry) in outputs.items():
                indices = [i for i, row in enumerate(rows[start:stop]) if row["role"] == role]
                if not indices:
                    continue
                check()
                selected = block[indices]
                local = entry["verified_rows"]
                with budget.write(partial), h5py.File(partial, "r+", rdcc_nbytes=pilot.CACHE_BYTES) as target:
                    target["modelmd"][local:local + len(indices), :, :, :] = selected
                    target.flush()
                verify_segment(partial, shape, product["dtype_str"], local, selected)
                entry["verified_rows"] += len(indices)
                del selected
            del block
            item["verified_rows"] += stop - start
            item["route_write_readback_seconds"] += time.monotonic() - opened
        check()
        core.require(item["verified_rows"] == product["shape"][0] and
                     sum(entry["verified_rows"] for entry in item["outputs"]) == product["shape"][0], "product_partition_incomplete")
        for role, (partial, shape, entry) in outputs.items():
            core.require(entry["verified_rows"] == shape[0], "role_partition_incomplete")
            check()
            final = partial.with_suffix("")
            budget.publish(partial, final)
            entry.update({"array_file": str(final), "shape": list(shape), "dtype_str": product["dtype_str"],
                          "array_file_bytes": final.stat().st_size, "copy_bytes_equal": True})
    item["complete"] = True


def check_full_capacity(record, job_id):
    observed = pilot.check_capacity(record, CACHE_PARENT, job_id)
    core.require(min(observed["filesystem_available_bytes"], observed["quota_available_bytes"]) -
                 observed["concurrent_reserved_bytes"] >= FULL_CAP, "insufficient_full_capacity_after_reservations")
    return observed


def initial_report(max_seconds):
    report = pilot.initial_report()
    report.update({"mode": "full", "pilot_copy_complete": False, "full_copy_complete": False,
                   "private_attempt": str(PRIVATE), "public_attempt": str(PUBLIC), "pilot_report": str(PILOT_REPORT),
                   "internal_seconds": max_seconds, "external_seconds_required": max_seconds + 60,
                   "output_byte_cap": FULL_CAP, "report_reserve_bytes": REPORT_RESERVE,
                   "products": [], "pilot_rows_reused": 0, "logical_source_rows_read": 0,
                   "private_output_bytes_including_preserved_pilot": None,
                   "neural_access_flag_is_conservative_if_read_fails": True})
    return report


def run_full(capacity_path, max_seconds):
    core.require(type(max_seconds) is int and 1 <= max_seconds <= 240 * 3600 - 70, "invalid_full_time_bound")
    started = time.monotonic()
    report = initial_report(max_seconds)
    budget = None
    def check():
        core.require(time.monotonic() < started + max_seconds, "attempt_deadline")
    def alarm(_signum, _frame):
        raise core.Refusal("attempt_deadline")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(max_seconds)
    try:
        root, routed = load_full_plan()
        check()
        completed_pilot = pilot.load_json(PILOT_REPORT)
        preserved = validate_pilot(completed_pilot, routed)
        report["pilot_operational_costs"] = {field: completed_pilot[field] for field in
                                             ("source_open_header_seconds", "source_read_seconds", "route_write_readback_seconds",
                                              "elapsed_seconds", "peak_rss_kib", "private_output_bytes")}
        report["capacity_observation"] = check_full_capacity(pilot.load_json(capacity_path), report["slurm_job_id"])
        core.require("/*/outputs/cache/" in (EPISODE.parent / ".gitignore").read_text().splitlines(), "private_tree_not_ignored")
        core.require(not PRIVATE.exists() and not PRIVATE.is_symlink(), "private_attempt_already_exists")
        # Every source header and every planned block is accepted before the
        # first neural read, including any read from the preserved pilot.
        products = preflight_sources(root, check)
        first = products[0]
        core.require(first["dtype_str"] == completed_pilot["source_layout"]["dtype_str"] and
                     first["shape"] == completed_pilot["source_layout"]["shape"] and
                     first["chunks"] == completed_pilot["source_layout"]["chunks"], "pilot_source_header_changed")
        report["source_header_preflight_complete"] = True
        report["source_headers"] = [{**{key: item[key] for key in
                                         ("participant", "stage", "source_file", "shape", "dtype_str", "chunks")},
                                     "planned_blocks": len(item["blocks"]),
                                     "maximum_decoded_block_bytes": max(block["decoded_work_bound_bytes"]
                                                                         for block in item["blocks"])}
                                    for item in products]
        report["planned_source_products"] = len(products)
        report["planned_source_blocks_including_reused_pilot"] = sum(len(item["blocks"]) for item in products)
        report["planned_logical_payload_bytes"] = EXPECTED_PAYLOAD_BYTES
        report["maximum_planned_decoded_block_bytes"] = max(block["decoded_work_bound_bytes"]
                                                              for item in products for block in item["blocks"])
        report["metadata_and_header_seconds"] = time.monotonic() - started
        check()
        private_directory(PRIVATE, CACHE_PARENT)
        budget = OutputBudget(preserved)
        write_full_maps(routed, PRIVATE, budget, check)
        for product in products:
            copy_product(root, product, routed[product["participant"]], PRIVATE, budget, report, check)
        check()
        report["pilot_rows_reused"] = sum(item["pilot_rows_reused"] for item in report["products"])
        report["logical_source_rows_read"] = sum(item["source_rows_read"] for item in report["products"])
        expected_rows = sum(shape[0] for shape in pilot.SHAPES.values()) * 3
        core.require(len(report["products"]) == 12 and all(item["complete"] for item in report["products"]) and
                     report["pilot_rows_reused"] == pilot.ROWS and
                     report["logical_source_rows_read"] + report["pilot_rows_reused"] == expected_rows,
                     "full_copy_scope_incomplete")
        report.update({"all_copy_bytes_equal": True, "neural_read_completed": True,
                       "full_copy_complete": True, "status": "completed"})
        check()
    except core.Refusal as error:
        report.update({"status": "stopped", "full_copy_complete": False})
        report["errors"].append(error.detail)
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, OverflowError, RuntimeError, MemoryError):
        report.update({"status": "stopped", "full_copy_complete": False})
        report["errors"].append({"code": "private_full_copy_processing_failure"})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
    report.update({"elapsed_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "downstream_use_requires_this_terminal_report_completed": True,
                   "completion_does_not_activate_reader_handoffs": True})
    if budget is not None:
        report["private_output_bytes_including_preserved_pilot"] = budget.total
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    core.require(len(encoded.encode("utf-8")) <= REPORT_RESERVE, "public_report_byte_cap")
    with (PUBLIC / "materialization_report.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({key: report[key] for key in ("status", "full_copy_complete", "neural_values_read",
                                                  "raw_audit_bytes_accessed", "elapsed_seconds", "peak_rss_kib")}))
    return 0 if report["status"] == "completed" and report["full_copy_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-observation", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--pilot-attempt", choices=("pilot001", "pilot002"), required=True)
    args = parser.parse_args()
    select_pilot_attempt(args.pilot_attempt)
    core.require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "verified_compute_allocation_required")
    core.require(all(os.environ.get(name) == "1" for name in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")), "single_thread_environment_required")
    core.require(args.capacity_observation.is_file() and args.capacity_observation.resolve().is_relative_to(
                 (EPISODE / "outputs/runs/role_materialization").resolve()), "capacity_record_path_outside_scope")
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(exist_ok=False)
    return run_full(args.capacity_observation, args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
