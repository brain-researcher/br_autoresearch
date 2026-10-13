#!/usr/bin/env python3
"""Fixed EP17 trusted-copy pilot; private bytes, public operational facts only.

Run in the coordinator's Slurm allocation, with an external 180-second timeout
and 10-second kill grace. Full materialization is deliberately not implemented.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
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

import reconcile_released_events as core
import qualify_source_roles as stage
import qualify_source_chronology as chronology

EPISODE = core.EPISODE
ROLES = ("development", "support_calibration", "sealed_audit")
ROLE_FIELDS = ("role", "outer_fold", "development_block", "audit_block")
CONCEPT_FIELDS = ("things_category_nr", "uniqueID", "provider_category_nr",
                  "original_common_image_count", "qualified_image_count", *ROLE_FIELDS)
PRESENTATION_FIELDS = (*core.OUTPUT_FIELDS, "concept_name", "condition", *chronology.ADDED_FIELDS)
ASSIGNMENT = EPISODE / "outputs/runs/concept_role_assignment/full001"
CHRONOLOGY = EPISODE / "outputs/runs/source_chronology/attempt001"
CACHE_PARENT = EPISODE / "outputs/cache"
PRIVATE = CACHE_PARENT / "private/role_materialization/pilot001"
PUBLIC = EPISODE / "outputs/runs/role_materialization/pilot001"
SOURCE_SUFFIX = "users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw"
SUBJECT = "sub-01"
STAGE = "B"
SOURCE_FILE = "THINGS/glmsingle/sub-01/glmsingle/output/T1w/TYPEB_FITHRF.mat"
SHAPES = dict(zip(stage.SUBJECTS, ((12780, 1, 1, 227214), (12780, 1, 1, 236701),
                                 (12780, 1, 1, 210349), (11640, 1, 1, 207095))))
ROWS = 64
INTERNAL_SECONDS = 120
BYTE_CAP = 256 * 1024**2
CACHE_BYTES = 16 * 1024**2
MAP_FIELDS = ("participant", "local_row_index_zero_based", "glmsingle_trial_index_zero_based",
              "expected_provider_image_path", "source_image_basename", "things_category_nr", "concept_name",
              "provider_category_nr", "provider_exemplar_nr", "condition", "eligible_image", "failure_reasons",
              *ROLE_FIELDS, "session", "run", "run_order", "trial_in_run_zero_based",
              "source_trial_number", "source_order", "source_repetition", "retained_repeat_rank", "repeat_half",
              "ordinal_status", "component_id", "component_source_history_position",
              "global_source_history_position", "retained_acquisition_rank")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(stage.bounded_read(path))


def unique_rows(rows, key, code):
    mapped = {}
    for row in rows:
        value = key(row)
        core.require(value not in mapped, code)
        mapped[value] = row
    return mapped


def check_concepts(concepts, capacities=(480, 120, 120), fold_size=80, development_size=20, audit_size=10):
    """Metadata only; capacities are fixed by the CLI, small in source-free fixtures."""
    mapped = unique_rows(concepts, lambda row: row["things_category_nr"], "duplicate_concept")
    core.require(len(mapped) == sum(capacities) and len({r["uniqueID"] for r in concepts}) == len(mapped),
                 "concept_roster_changed")
    core.require(Counter(row["role"] for row in concepts) == dict(zip(ROLES, capacities)), "role_capacity_changed")
    for row in concepts:
        core.require(row["role"] in ROLES and int(row["original_common_image_count"]) in (5, 6) and
                     1 <= int(row["qualified_image_count"]) <= int(row["original_common_image_count"]),
                     "concept_identity_or_eligibility_changed")
        if row["role"] == "development":
            core.require(row["outer_fold"] != "" and row["development_block"] != "" and row["audit_block"] == "",
                         "development_labels_changed")
        elif row["role"] == "sealed_audit":
            core.require(row["outer_fold"] == row["development_block"] == "" and row["audit_block"] != "",
                         "audit_labels_changed")
        else:
            core.require(all(row[field] == "" for field in ROLE_FIELDS[1:]), "calibration_labels_changed")
    for field, role, prefix, size in (("outer_fold", "development", "fold", fold_size),
                                      ("development_block", "development", "block", development_size),
                                      ("audit_block", "sealed_audit", "block", audit_size)):
        n = capacities[ROLES.index(role)]
        core.require(n % size == 0 and Counter(row[field] for row in concepts if row["role"] == role) ==
                     {f"{prefix}{i + 1:02d}": size for i in range(n // size)}, "fold_or_block_capacity_changed")
    return mapped


def check_routing(concepts, images, qualified_images, rows, chronology_rows, source_rows, expected_counts):
    """Exact inherited joins; incidental CSV order never controls neural row order."""
    image_map = unique_rows(images, lambda row: row["provider_image_path"], "duplicate_image")
    qualified = unique_rows(qualified_images, lambda row: row["provider_image_path"], "duplicate_qualified_image")
    core.require(set(image_map) == set(qualified), "assigned_image_roster_changed")
    concept_images = defaultdict(Counter)
    for path, row in image_map.items():
        core.require(all(row[field] == qualified[path][field] for field in stage.IMAGE_FIELDS),
                     "assigned_image_metadata_changed")
        cid = row["things_category_nr"]
        core.require(cid in concepts and row["eligible_image"] in ("True", "False"), "unknown_image_concept")
        concept = concepts[cid]
        core.require(row["concept_name"] == concept["uniqueID"] and
                     row["provider_category_nr"] == concept["provider_category_nr"] and
                     all(row[field] == concept[field] for field in ROLE_FIELDS), "image_role_conflict")
        core.require(row["eligible_image"] != "True" or
                     all(int(row[f"retained_count_{s}"]) >= 2 for s in expected_counts),
                     "eligible_image_lacks_retained_repetitions")
        concept_images[cid]["common"] += row["common_to_four"] == "True"
        concept_images[cid]["qualified"] += row["eligible_image"] == "True"
    core.require(all(concept_images[cid]["common"] == int(row["original_common_image_count"]) and
                     concept_images[cid]["qualified"] == int(row["qualified_image_count"])
                     for cid, row in concepts.items()), "concept_image_counts_changed")
    original = unique_rows(source_rows, stage.key, "duplicate_original_source_row")
    chronological = unique_rows(chronology_rows, stage.key, "duplicate_chronology_row")
    assigned = unique_rows(rows, stage.key, "duplicate_assigned_source_row")
    core.require(set(original) == set(chronological) == set(assigned), "presentation_roster_changed")
    core.require(set(assigned) == {(subject, i) for subject, n in expected_counts.items() for i in range(n)},
                 "source_indices_not_complete")
    counts = defaultdict(Counter)
    for key, row in assigned.items():
        old = chronological[key]
        core.require(all(row[field] == old[field] for field in PRESENTATION_FIELDS) and
                     all(old[field] == original[key][field] for field in (*core.OUTPUT_FIELDS, "concept_name", "condition")),
                     "original_presentation_metadata_changed")
        path = row["expected_provider_image_path"]
        core.require(path in image_map, "unknown_presentation_image")
        image = image_map[path]
        core.require(all(row[field] == image[field] for field in
                         ("things_category_nr", "concept_name", "provider_category_nr", "provider_exemplar_nr",
                          "condition", "eligible_image", *ROLE_FIELDS)) and
                     row["source_image_basename"] == image["image_name"] and row["ordinal_status"] == "valid" and
                     (row["repeat_half"] in ("1", "2") if row["eligible_image"] == "True" else row["repeat_half"] == ""),
                     "presentation_role_or_eligibility_conflict")
        counts[path][row["participant"]] += 1
    core.require(all(counts[path][s] == int(image[f"retained_count_{s}"])
                     for path, image in image_map.items() for s in expected_counts), "image_retained_counts_changed")
    routed = {}
    for subject, n in expected_counts.items():
        ordered = [assigned[(subject, i)] for i in range(n)]
        core.require({int(row["retained_acquisition_rank"]) for row in ordered} == set(range(n)),
                     "retained_acquisition_rank_incomplete")
        routed[subject] = [{**row, "failure_reasons": image_map[row["expected_provider_image_path"]]["failure_reasons"]}
                           for row in ordered]
    return routed


def load_plan():
    report = load_json(ASSIGNMENT / "assignment_report.json")
    core.require(report.get("status") == "completed" and report.get("metadata_assignments_complete") is True and
                 report.get("errors") == [] and report.get("preserved_metadata") ==
                 {"concepts": 720, "images": 4320, "presentations": 49980, "runs": 833} and
                 report.get("role_counts") == dict(zip(ROLES, (480, 120, 120))), "complete_full001_roles_required")
    basis = load_json(core.METADATA / "metadata_report.json")
    # The established source join, inlined only to retain its already parsed
    # header report. Each saved metadata input is read once, with the same cap.
    event_report = load_json(stage.EVENTS / "reconciliation_report.json")
    stage.check_reports(basis, event_report)
    _, source_runs = stage.table_rows(stage.bounded_read(core.METADATA / "run_metadata.csv"), stage.RUN_FIELDS)
    stage.check_roster(source_runs, event_report)
    _, original_rows = stage.table_rows(stage.bounded_read(core.METADATA / "presentation_identity.csv"), stage.PRESENTATION_FIELDS)
    _, event_rows = stage.table_rows(stage.bounded_read(stage.EVENTS / "event_metadata.csv"), core.OUTPUT_FIELDS)
    core.require(len(original_rows) == len(event_rows) == 49980, "fixed_presentation_count_changed")
    source_rows = stage.check_presentations(original_rows, event_rows, source_runs)
    del original_rows, event_rows
    eligible = load_json(chronology.QUALIFICATION / "qualification_report.json")
    core.require(eligible.get("status") == "completed" and eligible.get("eligibility_complete") is True and
                 eligible.get("eligible_720_concept_universe_verified") is True and eligible.get("errors") == [] and
                 eligible.get("qualified_concept_count") == 720 and eligible.get("qualified_image_count") == 3634 and
                 eligible.get("source_metadata_attempt") == str(core.METADATA) and
                 eligible.get("event_projection_attempt") == str(stage.EVENTS), "completed_eligibility_required")
    history = load_json(CHRONOLOGY / "chronology_report.json")
    core.require(history.get("status") == "completed" and history.get("errors") == [] and
                 all(history.get(flag) is True for flag in ("source_chronology_complete", "retained_half_qualification_complete",
                                                           "global_chronology_complete", "all_observed_ordinals_qualified")) and
                 history.get("preserved_presentations") == 49980 and history.get("preserved_runs") == 833 and
                 history.get("source_metadata_attempt") == str(core.METADATA) and
                 history.get("event_projection_attempt") == str(stage.EVENTS) and
                 history.get("eligibility_attempt") == str(chronology.QUALIFICATION), "completed_chronology_required")
    def table(path, fields):
        return stage.table_rows(stage.bounded_read(path), fields)[1]
    concepts = check_concepts(table(ASSIGNMENT / "concept_assignments.csv", CONCEPT_FIELDS))
    images = table(ASSIGNMENT / "image_assignments.csv", (*stage.IMAGE_FIELDS, *ROLE_FIELDS))
    core.require(len(images) == 4320 and sum(r["eligible_image"] == "True" for r in images) == 3634,
                 "assigned_image_counts_changed")
    routed = check_routing(concepts, images,
                           table(chronology.QUALIFICATION / "image_qualification.csv", stage.IMAGE_FIELDS),
                           table(ASSIGNMENT / "presentation_assignments.csv", (*PRESENTATION_FIELDS, *ROLE_FIELDS)),
                           table(CHRONOLOGY / "presentation_chronology.csv", PRESENTATION_FIELDS), source_rows,
                           {s: shape[0] for s, shape in SHAPES.items()})
    source_root = Path(os.environ["OAK"]) / SOURCE_SUFFIX
    core.require(basis.get("source") == str(source_root), "source_root_changed")
    axis = next(person for person in basis["participants"] if person["participant"] == SUBJECT)["stage_axes"][STAGE]
    core.require(axis["file"] == SOURCE_FILE and axis["dataset"] == "modelmd" and
                 tuple(axis["stored_shape"]) == SHAPES[SUBJECT] and axis["dtype"] == "float32", "saved_source_header_changed")
    return source_root, routed[SUBJECT]


def check_capacity(record, parent, job_id, now=None):
    now = now or datetime.now(timezone.utc)
    core.require(record.get("parent") == str(parent) and record.get("slurm_job_id") == job_id and
                 record.get("quota_checked") is True, "capacity_observation_scope_changed")
    observed = datetime.fromisoformat(record["observed_at_utc"].replace("Z", "+00:00"))
    core.require(observed.tzinfo is not None and observed.utcoffset().total_seconds() == 0 and observed <= now,
                 "capacity_observation_time_invalid")
    fields = ("filesystem_available_bytes", "quota_available_bytes", "concurrent_reserved_bytes")
    core.require(all(type(record.get(field)) is int and record[field] >= 0 for field in fields),
                 "capacity_observation_unknown")
    reserved = record["concurrent_reserved_bytes"]
    core.require(min(record[fields[0]], record[fields[1]]) - reserved >= BYTE_CAP,
                 "insufficient_capacity_after_reservations")
    return {field: record[field] for field in ("parent", "slurm_job_id", "observed_at_utc", "quota_checked", *fields)}


def layout_bound(shape, chunks, itemsize, start=0, stop=ROWS):
    core.require(len(shape) == 4 and shape[1:3] == (1, 1) and 0 <= start < stop <= shape[0], "invalid_source_extent")
    if chunks is None:
        return {"decoded_work_bound_bytes": (stop - start) * math.prod(shape[1:]) * itemsize,
                "decoded_row_start": start, "decoded_row_stop_exclusive": stop, "intersecting_chunks": None}
    core.require(len(chunks) == 4 and all(type(n) is int and n > 0 for n in chunks), "invalid_source_chunks")
    row_first, row_last = start // chunks[0], (stop - 1) // chunks[0]
    count = (row_last - row_first + 1) * math.prod((shape[i] + chunks[i] - 1) // chunks[i] for i in range(1, 4))
    return {"decoded_work_bound_bytes": count * math.prod(chunks) * itemsize,
            "decoded_row_start": row_first * chunks[0],
            "decoded_row_stop_exclusive": min((row_last + 1) * chunks[0], shape[0]), "intersecting_chunks": count}


def checked_dataset(handle, shape, stop=ROWS):
    core.require(isinstance(handle.get("modelmd", getlink=True), h5py.HardLink), "source_dataset_not_local_hardlink")
    dataset = handle["modelmd"]
    core.require(isinstance(dataset, h5py.Dataset) and not dataset.is_virtual and not dataset.external,
                 "source_dataset_indirection")
    core.require(tuple(dataset.shape) == tuple(shape) and dataset.dtype.kind == "f" and dataset.dtype.itemsize == 4,
                 "source_shape_or_dtype_changed")
    bound = layout_bound(tuple(dataset.shape), dataset.chunks, dataset.dtype.itemsize, stop=stop)
    core.require(bound["decoded_work_bound_bytes"] <= BYTE_CAP, "source_layout_exceeds_decoded_bound")
    return dataset, {"shape": list(dataset.shape), "dtype_str": dataset.dtype.str,
                     "chunks": list(dataset.chunks) if dataset.chunks else None, **bound}


def private_directory(path, parent):
    core.require(path.is_relative_to(parent) and parent.is_dir() and not parent.is_symlink(), "private_parent_invalid")
    current = parent
    for part in path.relative_to(parent).parts:
        current /= part
        if not current.exists():
            current.mkdir(mode=0o700)
        core.require(not current.is_symlink() and current.is_dir() and stat.S_IMODE(current.stat().st_mode) == 0o700,
                     "private_directory_permissions_invalid")


def new_private_file(path):
    return os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600), "w+b")


def publish_file(partial, final):
    # A local hard link only makes publication exclusive; never links source or another attempt.
    os.link(partial, final)
    partial.unlink()


def write_map(path, rows):
    with new_private_file(path) as raw:
        import io
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MAP_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow({**row, "local_row_index_zero_based": i})


def begin_access(report, rows, layout, stop):
    """Conservatively record exposure before an I/O call that may fail midway."""
    now = utc_now()
    decoded = rows[layout["decoded_row_start"]:layout["decoded_row_stop_exclusive"]]
    audit = sum(row["role"] == "sealed_audit" for row in decoded)
    report.update({"neural_read_attempted": True, "neural_values_read": True,
                   "neural_read_completed": False, "first_neural_access_utc": now,
                   "neural_access_flag_is_conservative_if_read_fails": True,
                   "logical_copied_scope": {"participant": SUBJECT, "stage": STAGE, "rows_start": 0, "rows_stop_exclusive": stop},
                   "conservative_decoded_scope": {"participant": SUBJECT, "stage": STAGE,
                                                  "rows_start": layout["decoded_row_start"],
                                                  "rows_stop_exclusive": layout["decoded_row_stop_exclusive"],
                                                  "all_source_voxel_columns": True},
                   "logical_audit_row_count": sum(row["role"] == "sealed_audit" for row in rows[:stop]),
                   "decoded_scope_audit_row_count": audit, "raw_audit_bytes_accessed": audit > 0,
                   "first_raw_audit_access_utc": now if audit else None})


def copy_block(dataset, rows, private, layout, report, check, stop=ROWS):
    check()
    begin_access(report, rows, layout, stop)
    started = time.monotonic()
    block = dataset[:stop, :, :, :]
    report["source_read_seconds"] = time.monotonic() - started
    report["neural_read_completed"] = True
    core.require(block.shape == (stop, *dataset.shape[1:]) and block.dtype == dataset.dtype, "source_read_representation_changed")
    check()
    started = time.monotonic()
    outputs, total_bytes = [], 0
    for role in ROLES:
        check()
        indices = [i for i, row in enumerate(rows[:stop]) if row["role"] == role]
        role_dir = private / role
        private_directory(role_dir, private)
        selected_rows = [rows[i] for i in indices]
        map_partial = role_dir / "source_index_map.csv.partial"
        write_map(map_partial, selected_rows)
        total_bytes += map_partial.stat().st_size
        entry = {"role": role, "participant": SUBJECT, "stage": STAGE, "rows": len(indices),
                 "empty_role": not indices, "array_file": None, "copy_bytes_equal": True}
        if indices:
            subject_dir = role_dir / SUBJECT
            private_directory(subject_dir, private)
            partial = subject_dir / f"{STAGE}.h5.partial"
            selected = block[indices]
            with new_private_file(partial) as stream:
                with h5py.File(stream, "w", rdcc_nbytes=CACHE_BYTES) as output:
                    destination = output.create_dataset("modelmd", shape=selected.shape, dtype=dataset.dtype,
                                                        chunks=(min(16, len(indices)), 1, 1, dataset.shape[3]))
                    destination[...] = selected
                    output.flush()
            # Independent reopened reader validates persisted HDF5 bytes and
            # local layout; no second source payload read is performed.
            with h5py.File(partial, "r", rdcc_nbytes=CACHE_BYTES) as verified:
                destination, _ = checked_dataset(verified, selected.shape, stop=len(indices))
                core.require(destination.dtype.str == selected.dtype.str and destination.compression is None and
                             destination.scaleoffset is None and
                             destination.chunks == (min(16, len(indices)), 1, 1, dataset.shape[3]),
                             "private_output_representation_changed")
                copied = destination[...]
                core.require(copied.dtype.str == selected.dtype.str and copied.shape == selected.shape and
                             np.array_equal(copied.view(np.uint8), selected.view(np.uint8)), "private_copy_byte_mismatch")
            core.require(stat.S_IMODE(partial.stat().st_mode) == 0o600, "private_file_permissions_invalid")
            total_bytes += partial.stat().st_size
            core.require(total_bytes <= BYTE_CAP, "pilot_output_byte_cap")
            check()
            final = subject_dir / f"{STAGE}.h5"
            publish_file(partial, final)
            entry.update({"array_file": str(final), "shape": list(selected.shape), "dtype_str": dataset.dtype.str,
                          "array_file_bytes": final.stat().st_size, "logical_payload_bytes": selected.nbytes})
            del selected, copied
        core.require(stat.S_IMODE(map_partial.stat().st_mode) == 0o600, "private_map_permissions_invalid")
        publish_file(map_partial, role_dir / "source_index_map.csv")
        outputs.append(entry)
    check()
    report.update({"outputs": outputs, "private_output_bytes": total_bytes,
                   "logical_source_payload_bytes": block.nbytes, "all_copy_bytes_equal": True,
                   "route_write_readback_seconds": time.monotonic() - started})
    core.require(total_bytes <= BYTE_CAP and sum(item["rows"] for item in outputs) == stop,
                 "pilot_output_scope_changed")


def finish(report, check):
    check()
    core.require(report.get("neural_read_completed") is True and report.get("all_copy_bytes_equal") is True,
                 "copy_not_complete")
    report.update({"status": "completed", "pilot_copy_complete": True})


def initial_report():
    return {"analysis_id": "role_materialization", "mode": "pilot", "status": "running", "pilot_copy_complete": False,
            "source_roles_complete": False, "physical_role_isolation_verified": False, "reader_handoffs_activated": False,
            "candidate_scoring_authorized": False, "full_neural_handoffs_materialized": False,
            "neural_read_attempted": False, "neural_values_read": False, "neural_read_completed": False,
            "raw_audit_bytes_accessed": False, "first_neural_access_utc": None, "first_raw_audit_access_utc": None,
            "scientific_audit_evaluation_count": 0, "audit_model_evaluated": False,
            "image_values_read": False, "behavioral_outcomes_read": False, "model_features_read": False,
            "finite_voxel_alignment_verified": False, "response_units_verified": False,
            "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 180, "external_kill_grace_seconds": 10,
            "workers": 1, "requested_memory_gib": 2, "output_byte_cap": BYTE_CAP, "decoded_work_byte_cap": BYTE_CAP,
            "source_raw_chunk_cache_bytes": CACHE_BYTES, "destination_raw_chunk_cache_bytes": CACHE_BYTES,
            "source_metadata_attempt": str(core.METADATA), "role_assignment_attempt": str(ASSIGNMENT),
            "private_attempt": str(PRIVATE), "public_attempt": str(PUBLIC), "network_requests": 0, "automatic_retries": 0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(), "errors": []}


def run_pilot(capacity_path):
    start = time.monotonic()
    report = initial_report()
    def check():
        core.require(time.monotonic() < start + INTERNAL_SECONDS, "attempt_deadline")
    def alarm(_signum, _frame):
        raise core.Refusal("attempt_deadline")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        root, rows = load_plan()
        check()
        report["metadata_validation_seconds"] = time.monotonic() - start
        report["capacity_observation"] = check_capacity(load_json(capacity_path), CACHE_PARENT, report["slurm_job_id"])
        core.require("/*/outputs/cache/" in (EPISODE.parent / ".gitignore").read_text().splitlines(), "private_tree_not_ignored")
        core.require(not PRIVATE.exists() and not PRIVATE.is_symlink(), "private_attempt_already_exists")
        private_directory(PRIVATE, CACHE_PARENT)
        source = (root / SOURCE_FILE).resolve(strict=True)
        core.require(root.is_dir() and source.is_relative_to(root.resolve()) and source.is_file(), "source_resolution_outside_root")
        check()
        opened = time.monotonic()
        with h5py.File(source, "r", rdcc_nbytes=CACHE_BYTES) as handle:
            dataset, layout = checked_dataset(handle, SHAPES[SUBJECT])
            report.update({"source_file": str(root / SOURCE_FILE), "source_dataset": "modelmd", "source_layout": layout,
                           "source_open_header_seconds": time.monotonic() - opened})
            copy_block(dataset, rows, PRIVATE, layout, report, check)
        finish(report, check)
        check()
    except core.Refusal as error:
        report.update({"status": "stopped", "pilot_copy_complete": False})
        report["errors"].append(error.detail)
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, OverflowError, RuntimeError, MemoryError):
        report.update({"status": "stopped", "pilot_copy_complete": False})
        report["errors"].append({"code": "private_copy_processing_failure"})
    finally:
        signal.alarm(0)
    report.update({"elapsed_seconds": time.monotonic() - start,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "downstream_use_requires_this_terminal_report_completed": True,
                   "completion_does_not_activate_reader_handoffs": True})
    with (PUBLIC / "pilot_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "pilot_copy_complete": report["pilot_copy_complete"],
                      "neural_values_read": report["neural_values_read"], "raw_audit_bytes_accessed": report["raw_audit_bytes_accessed"],
                      "elapsed_seconds": report["elapsed_seconds"], "peak_rss_kib": report["peak_rss_kib"]}))
    return 0 if report["status"] == "completed" and report["pilot_copy_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot",), required=True)
    parser.add_argument("--capacity-observation", type=Path, required=True)
    args = parser.parse_args()
    core.require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "verified_compute_allocation_required")
    core.require(all(os.environ.get(name) == "1" for name in
                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")), "single_thread_environment_required")
    core.require(args.capacity_observation.is_file() and args.capacity_observation.resolve().is_relative_to(
                 (EPISODE / "outputs/runs/role_materialization").resolve()), "capacity_record_path_outside_scope")
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(exist_ok=False)
    return run_pilot(args.capacity_observation)


if __name__ == "__main__":
    raise SystemExit(main())
