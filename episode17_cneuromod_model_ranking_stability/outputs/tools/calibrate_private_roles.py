#!/usr/bin/env python3
"""Bounded trusted-builder calibration, exclusively from private role copies.

The coordinator selects each invocation separately. Pilot is the first 128
original columns of sub-01/left, all fixed eligible images and B/C/D. Full
requires its completed cost pilot and an explicit finite --max-seconds. Neither
mode releases readers, fits candidates, reads audit responses or launches jobs.
Run with one CPU, 2 GiB, and an external timeout of max-seconds + 60 (+10 kill).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import gzip
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

import calibration_core as numerical
import support_core as support
import materialize_neural_roles as original
import materialize_neural_roles_full as copied
import materialize_neural_roles_columns as columns
import qualify_structural_support as structural
from neural_role_private_paths import private_directory

core = original.core
EPISODE = original.EPISODE
COPY_PRIVATE = original.CACHE_PARENT / "private/role_materialization/attempt002"
COPY_PUBLIC = EPISODE / "outputs/runs/role_materialization/attempt002"
STRUCTURAL = EPISODE / "outputs/runs/structural_support/attempt001"
PRIVATE_PARENT = original.CACHE_PARENT / "private/calibration_support"
RUNS = EPISODE / "outputs/runs/calibration_support"
ROLE = "support_calibration"
STAGES = ("B", "C", "D")
PILOT_COLUMNS = 128
PILOT_SECONDS = 300
MEMORY_BYTES = 2 * 1024**3
CAPS = {"pilot": 64 * 1024**2, "full": 256 * 1024**2}
VECTOR_FIELDS = tuple(field.name for field in fields(numerical.CalibrationResult)
                      if field.name not in ("image_count", "repeat_counts", "half_counts", "image_mean_variance"))
VARIANCE_FIELDS = tuple(field.name for field in fields(numerical.VarianceQualification))


def paths(mode):
    name = "pilot001" if mode == "pilot" else "attempt001"
    return PRIVATE_PARENT / name, RUNS / name


def validate_copy(report):
    core.require(report.get("status") == "completed" and report.get("mode") == "full" and
                 report.get("full_copy_complete") is True and report.get("all_copy_bytes_equal") is True and
                 report.get("neural_read_completed") is True and report.get("errors") == [] and
                 report.get("copy_layout") == columns.LAYOUT and
                 report.get("private_attempt") == str(COPY_PRIVATE) and
                 report.get("public_attempt") == str(COPY_PUBLIC) and
                 report.get("role_assignment_attempt") == str(original.ASSIGNMENT),
                 "completed_private_full_attempt002_required")
    products = report.get("products", [])
    expected = {(subject, stage) for subject in support.PARTICIPANT_COLUMNS for stage in STAGES}
    core.require(len(products) == 12 and {(p["participant"], p["stage"]) for p in products} == expected,
                 "copied_product_roster_changed")
    result = {}
    for item in products:
        subject, stage = item["participant"], item["stage"]
        width = support.PARTICIPANT_COLUMNS[subject]
        core.require(item.get("complete") is True and item.get("verified_columns") == width and
                     item.get("dtype_str") == "<f4", "private_copy_product_incomplete")
        entries = item.get("outputs", [])
        core.require(len(entries) == 3 and {entry["role"] for entry in entries} == set(original.ROLES),
                     "private_copy_role_roster_changed")
        entry = next(entry for entry in entries if entry["role"] == ROLE)
        path = COPY_PRIVATE / ROLE / subject / f"{stage}.h5"
        core.require(entry.get("copy_bytes_equal") is True and entry.get("empty_role") is False and
                     entry.get("verified_columns") == width and entry.get("array_file") == str(path) and
                     entry.get("shape") == [entry.get("rows"), 1, 1, width] and
                     type(entry.get("rows")) is int and entry["rows"] > 0 and entry.get("dtype_str") == "<f4",
                     "private_calibration_copy_changed")
        result[(subject, stage)] = entry
    return result


def metadata_plan(copy_entries):
    # This helper reads frozen projected CSV/JSON metadata only. Its returned
    # mixed-source root is deliberately discarded and never opened here.
    _, all_routed = copied.load_full_plan()
    routed = {subject: [row for row in rows if row["role"] == ROLE]
              for subject, rows in all_routed.items()}
    del all_routed
    path = COPY_PRIVATE / ROLE / "source_index_map.csv"
    copied.ordinary_private_file(path, COPY_PRIVATE)
    _, actual = original.stage.table_rows(original.stage.bounded_read(path), original.MAP_FIELDS)
    expected = [{field: str(local) if field == "local_row_index_zero_based" else row[field]
                 for field in original.MAP_FIELDS}
                for subject in sorted(routed) for local, row in enumerate(routed[subject])]
    core.require(actual == expected, "private_calibration_index_map_changed")
    core.require(all(copy_entries[(subject, stage)]["rows"] == len(rows)
                     for subject, rows in routed.items() for stage in STAGES), "private_calibration_rows_changed")
    return {subject: image_groups(rows) for subject, rows in routed.items()}, routed


def image_groups(rows):
    """Fixed metadata-only eligibility, with complete acquisition-ordered repeats."""
    groups = defaultdict(list)
    for local, row in enumerate(rows):
        core.require(row["role"] == ROLE and row["eligible_image"] in ("True", "False"),
                     "calibration_role_or_eligibility_changed")
        if row["eligible_image"] == "True":
            groups[row["expected_provider_image_path"]].append((local, row))
    core.require(len(groups) >= numerical.MINIMUM_CALIBRATION_IMAGES, "fixed_calibration_image_roster_too_small")
    result = []
    for image, repeated in sorted(groups.items()):
        ordered = sorted(repeated, key=lambda item: int(item[1]["retained_acquisition_rank"]))
        ranks = [int(row["retained_acquisition_rank"]) for _, row in ordered]
        core.require(len(ordered) >= 2 and len(set(ranks)) == len(ranks) and min(ranks) >= 0 and
                     all(row["ordinal_status"] == "valid" and int(row["retained_repeat_rank"]) == i and
                         int(row["repeat_half"]) == 1 + i % 2 for i, (_, row) in enumerate(ordered)),
                     "calibration_repeat_chronology_changed")
        result.append({"image": image, "rows": [local for local, _ in ordered], "order": ranks})
    return result


def structural_columns():
    report = original.load_json(STRUCTURAL / "structural_support_report.json")
    core.require(report.get("status") == "completed" and report.get("structural_bins_complete") is True and
                 report.get("N_and_quotas_frozen") is True and report.get("errors") == [] and
                 report.get("q_per_bin") == support.Q and report.get("N_per_participant_ROI") == support.N and
                 report.get("output_metadata_rows") == 8558, "frozen_structural_bins_required")
    raw = original.stage.bounded_read(STRUCTURAL / "bin_columns.csv.gz", cap=2 * 1024**2)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(4 * 1024**2 + 1)
    core.require(len(expanded) <= 4 * 1024**2, "structural_metadata_byte_cap")
    _, rows = original.stage.table_rows(expanded, (*structural.FIELDS, "spatial_bin"))
    groups = defaultdict(list)
    for row in rows:
        subject, roi = row["participant"], row["nominated_hemisphere"]
        core.require(subject in support.PARTICIPANT_COLUMNS and roi in support.ROI_LABELS and
                     int(row["atlas_label"]) == support.ROI_LABELS[roi] and row["spatial_bin"] in support.BINS,
                     "structural_identity_changed")
        groups[(subject, roi)].append(support.StructuralColumn(
            subject, roi, int(row["model_column_zero_based"]), row["spatial_bin"], int(row["atlas_label"])))
    core.require(set(groups) == {(subject, roi) for subject in support.PARTICIPANT_COLUMNS for roi in support.ROI_LABELS},
                 "structural_group_roster_changed")
    for (subject, roi), values in groups.items():
        expected_count = structural.EXPECTED_COUNTS[subject][1 if roi == "left" else 2]
        ids = [value.column_id for value in values]
        core.require(len(ids) == expected_count and len(ids) == len(set(ids)) and
                     all(0 <= value < support.PARTICIPANT_COLUMNS[subject] for value in ids),
                     "structural_original_columns_changed")
        values.sort(key=lambda value: value.column_id)
    return dict(groups)


def selected_columns(mode, groups):
    if mode == "pilot":
        selected = tuple(row.column_id for row in groups[("sub-01", "left")][:PILOT_COLUMNS])
        core.require(len(selected) == PILOT_COLUMNS, "fixed_cost_pilot_columns_missing")
        return {"sub-01": selected}
    return {subject: tuple(sorted(row.column_id for roi in support.ROI_LABELS for row in groups[(subject, roi)]))
            for subject in support.PARTICIPANT_COLUMNS}


def stripe_plan(ids, shape):
    core.require(len(ids) > 0 and len(ids) == len(set(ids)) and tuple(ids) == tuple(sorted(ids)) and
                 all(0 <= value < shape[3] for value in ids), "selected_original_columns_invalid")
    grouped = defaultdict(list)
    for position, ident in enumerate(ids):
        grouped[(ident // columns.COLUMNS) * columns.COLUMNS].append((position, ident))
    result = []
    for start, entries in sorted(grouped.items()):
        stop = min(start + columns.COLUMNS, shape[3])
        rows_per_chunk = min(columns.CHUNK_ROWS, shape[0])
        decoded = math.ceil(shape[0] / rows_per_chunk) * rows_per_chunk * columns.COLUMNS * 4
        core.require(decoded <= 16 * 1024**2, "calibration_stripe_decoded_bound")
        result.append({"start": start, "stop": stop, "positions": [p for p, _ in entries],
                       "offsets": [ident - start for _, ident in entries],
                       "row_chunks": math.ceil(shape[0] / rows_per_chunk), "decoded_bytes": decoded})
    return result


def preflight(entries, selected, check):
    """Finish every selected private header and stripe plan before any values."""
    result = []
    for subject, ids in selected.items():
        for stage in STAGES:
            check()
            entry = entries[(subject, stage)]
            path = COPY_PRIVATE / ROLE / subject / f"{stage}.h5"
            actual = copied.ordinary_private_file(path, COPY_PRIVATE)
            core.require(actual == entry["array_file_bytes"], "private_calibration_file_size_changed")
            with h5py.File(path, "r", rdcc_nbytes=original.CACHE_BYTES) as handle:
                columns.output_dataset(handle, entry["shape"], entry["dtype_str"])
            result.append({"participant": subject, "stage": stage, "path": str(path),
                           "shape": entry["shape"], "dtype_str": entry["dtype_str"],
                           "file_bytes": actual, "column_ids": list(ids),
                           "stripes": stripe_plan(ids, entry["shape"])})
    return result


def calibrate_product(dataset, plan, groups, report, check):
    vectors = {}
    for stripe in plan["stripes"]:
        check()
        report["calibration_values_read"] = True  # Conservative if I/O fails.
        began = time.monotonic()
        block = dataset[:, 0, 0, stripe["start"]:stripe["stop"]]
        report["private_read_seconds"] += time.monotonic() - began
        core.require(block.shape == (plan["shape"][0], stripe["stop"] - stripe["start"]) and
                     block.dtype.str == plan["dtype_str"], "private_calibration_read_representation_changed")
        began = time.monotonic()
        repeated = [block[np.ix_(item["rows"], stripe["offsets"])] for item in groups]
        result = numerical.calibrate(repeated, [np.asarray(item["order"], dtype=np.int64) for item in groups])
        output = {name: getattr(result, name) for name in VECTOR_FIELDS}
        output.update({"variance_" + name: getattr(result.image_mean_variance, name) for name in VARIANCE_FIELDS})
        for name, value in output.items():
            if name not in vectors:
                vectors[name] = np.empty(len(plan["column_ids"]), dtype=value.dtype)
            vectors[name][stripe["positions"]] = value
        vectors["repeat_counts"] = result.repeat_counts
        vectors["half_counts"] = result.half_counts
        report["numerical_seconds"] += time.monotonic() - began
        report["completed_physical_stripes"] += 1
        report["completed_decoded_bound_bytes"] += stripe["decoded_bytes"]
        del block, repeated, result, output
        check()
    return vectors


def construct_supports(groups, selected, products):
    artifacts = {}
    for (subject, roi), structural_group in sorted(groups.items()):
        ids = np.asarray(selected[subject], dtype=np.int64)
        qualified = np.logical_and.reduce([products[(subject, stage)]["response_eligible"] for stage in STAGES])
        index = {int(ident): i for i, ident in enumerate(ids)}
        a_all = tuple(row.column_id for row in structural_group if qualified[index[row.column_id]])
        d = products[(subject, "D")]
        reliable = {row.column_id: float(d["pearson_r"][index[row.column_id]])
                    if d["reliability_valid"][index[row.column_id]] else math.nan for row in structural_group}
        anatomical = support.build_anatomical_bank(structural_group, a_all)
        ranked = support.build_reliable_support(structural_group, a_all, reliable)
        prefix = f"{subject}_{roi}_"
        artifacts[prefix + "column_ids"] = np.asarray(anatomical.column_ids, dtype=np.int64)
        artifacts[prefix + "spatial_bins"] = np.asarray(anatomical.spatial_bins)
        artifacts[prefix + "A_all"] = np.asarray(a_all, dtype=np.int64)
        artifacts[prefix + "A_N_banks"] = np.asarray(anatomical.banks, dtype=np.int64)
        artifacts[prefix + "A_N_inclusion_counts"] = np.asarray(anatomical.inclusion_counts, dtype=np.int64)
        artifacts[prefix + "R_N"] = np.asarray(ranked.selected_ids, dtype=np.int64)
        artifacts[prefix + "V_NC"] = np.asarray([ident for ident in a_all if d["ceiling_valid"][index[ident]]], dtype=np.int64)
    return artifacts


def write_private(path, payload, budget):
    partial = path.with_suffix(path.suffix + ".partial")
    with budget.write(partial), original.new_private_file(partial) as stream:
        stream.write(payload)
    budget.publish(partial, path)


def save_arrays(path, arrays, budget, check):
    partial = path.with_suffix(path.suffix + ".partial")
    with budget.write(partial), original.new_private_file(partial) as stream:
        np.savez(stream, **arrays)
    check()
    with np.load(partial, allow_pickle=False) as saved:
        core.require(set(saved.files) == set(arrays), "private_calibration_artifact_keys_changed")
        for key, expected in arrays.items():
            check()
            actual = saved[key]
            core.require(actual.shape == expected.shape and actual.dtype == expected.dtype and
                         actual.tobytes() == expected.tobytes(), "private_calibration_artifact_bytes_changed")
    budget.publish(partial, path)


def validate_cost_pilot(full_binding):
    private, public = paths("pilot")
    report = original.load_json(public / "calibration_report.json")
    core.require(report.get("status") == "completed" and report.get("pilot_complete") is True and
                 report.get("mode") == "pilot" and report.get("errors") == [] and
                 report.get("private_attempt") == str(private) and report.get("full_copy_attempt") == str(COPY_PRIVATE) and
                 report.get("selected_original_columns") == PILOT_COLUMNS and
                 report.get("completed_physical_stripes") == report.get("planned_physical_stripes") and
                 report.get("derived_arrays_verified") is True, "completed_matching_calibration_cost_pilot_required")
    for field in ("elapsed_seconds", "peak_rss_kib", "private_read_seconds", "numerical_seconds"):
        value = report.get(field)
        core.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     "calibration_cost_observation_missing")
    core.require(report["peak_rss_kib"] < MEMORY_BYTES // 1024, "calibration_cost_pilot_memory_bound")
    plan_path = private / "read_plan.json"
    size = copied.ordinary_private_file(plan_path, private)
    previous = original.load_json(plan_path)
    core.require(previous.get("binding") == full_binding, "calibration_cost_pilot_binding_changed")
    size += copied.ordinary_private_file(private / "calibration_maps.npz", private)
    return size + (public / "calibration_report.json").stat().st_size


def run(mode, max_seconds=None):
    core.require(mode in CAPS, "calibration_mode_invalid")
    seconds = PILOT_SECONDS if mode == "pilot" else max_seconds
    core.require(type(seconds) is int and 0 < seconds <= 7200 and
                 (mode != "pilot" or max_seconds in (None, PILOT_SECONDS)), "explicit_calibration_time_bound_required")
    private, public = paths(mode)
    core.require(not private.exists() and not private.is_symlink() and not public.exists() and not public.is_symlink(),
                 "calibration_attempt_already_exists")
    public.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"analysis_id": "calibration_support", "mode": mode, "status": "running", "errors": [],
              "private_attempt": str(private), "full_copy_attempt": str(COPY_PRIVATE),
              "pilot_complete": False, "calibration_support_complete": False, "derived_arrays_verified": False,
              "calibration_values_read": False, "development_values_read": False, "raw_audit_bytes_accessed": False,
              "mixed_source_values_read": False, "scientific_audit_evaluation_count": 0,
              "reader_handoffs_activated": False, "candidate_scoring_authorized": False,
              "private_read_seconds": 0., "numerical_seconds": 0., "completed_physical_stripes": 0,
              "completed_decoded_bound_bytes": 0, "cpus": 1, "memory_bound_bytes": MEMORY_BYTES,
              "internal_seconds": seconds, "external_seconds_required": seconds + 60,
              "external_kill_grace_seconds": 10, "output_byte_cap": CAPS[mode],
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname()}
    def check():
        core.require(time.monotonic() - started < seconds, "calibration_deadline")
    def alarm(_signum, _frame):
        raise core.Refusal("calibration_deadline")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(seconds)
    budget = None
    try:
        entries = validate_copy(original.load_json(COPY_PUBLIC / "materialization_report.json"))
        images, routed = metadata_plan(entries)
        core.require(len({tuple(item["image"] for item in value) for value in images.values()}) == 1,
                     "calibration_image_roster_differs_between_participants")
        structural_groups = structural_columns()
        selected = selected_columns(mode, structural_groups)
        # Bind the fixed prospective pilot even in full mode; no response-based
        # pilot membership, changed chronology or altered bins may be reused.
        binding = {"full_copy_attempt": str(COPY_PRIVATE), "role_assignment": str(original.ASSIGNMENT),
                   "structural_attempt": str(STRUCTURAL),
                   "pilot_column_ids": list(selected_columns("pilot", structural_groups)["sub-01"]),
                   "eligible_images": images,
                   "structural_columns": {f"{subject}_{roi}": [[r.column_id, r.spatial_bin, r.atlas_label] for r in rows]
                                          for (subject, roi), rows in sorted(structural_groups.items())}}
        preserved = validate_cost_pilot(binding) if mode == "full" else 0
        plans = preflight(entries, selected, check)
        report.update({"selected_original_columns": sum(map(len, selected.values())),
                       "eligible_image_count": len(next(iter(images.values()))), "products_planned": len(plans),
                       "planned_physical_stripes": sum(len(p["stripes"]) for p in plans),
                       "planned_storage_chunks": sum(s["row_chunks"] for p in plans for s in p["stripes"]),
                       "planned_decoded_bound_bytes": sum(s["decoded_bytes"] for p in plans for s in p["stripes"])})
        del routed
        private_directory(private, original.CACHE_PARENT)
        budget = copied.OutputBudget(preserved, cap=CAPS[mode])
        write_private(private / "read_plan.json", (json.dumps({"binding": binding, "products": plans},
                      sort_keys=True, allow_nan=False) + "\n").encode(), budget)
        products, artifacts = {}, {}
        for plan in plans:
            check()
            path = Path(plan["path"])
            core.require(copied.ordinary_private_file(path, COPY_PRIVATE) == plan["file_bytes"],
                         "private_calibration_file_size_changed")
            with h5py.File(path, "r", rdcc_nbytes=original.CACHE_BYTES) as handle:
                dataset = columns.output_dataset(handle, plan["shape"], plan["dtype_str"])
                key = plan["participant"], plan["stage"]
                result = calibrate_product(dataset, plan, images[key[0]], report, check)
                products[key] = result
                prefix = "_".join(key) + "_"
                artifacts[prefix + "column_ids"] = np.asarray(plan["column_ids"], dtype=np.int64)
                artifacts.update({prefix + name: value for name, value in result.items()})
        for subject in selected:
            artifacts[subject + "_eligible_images"] = np.asarray([item["image"] for item in images[subject]])
        # Preserve private maps even when an unchanged quota makes full support
        # invalid; public output exposes only the invalid_support stop code.
        save_arrays(private / "calibration_maps.npz", artifacts, budget, check)
        report["derived_arrays_verified"] = True
        if mode == "full":
            supports = construct_supports(structural_groups, selected, products)
            save_arrays(private / "supports.npz", supports, budget, check)
        check()
        report.update(status="completed", pilot_complete=mode == "pilot", calibration_support_complete=mode == "full")
    except support.SupportError:
        report.update(status="stopped", errors=[{"code": "invalid_support"}])
    except core.Refusal as error:
        report.update(status="stopped", errors=[{"code": error.detail["code"]}])
    except (OSError, ValueError, TypeError, KeyError, OverflowError, MemoryError):
        report.update(status="stopped", errors=[{"code": "private_calibration_processing_failure"}])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
    report.update(elapsed_seconds=time.monotonic() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  private_output_bytes=sum(budget.sizes.values()) if budget else 0,
                  accounted_output_bytes_including_pilot=budget.total if budget else 0)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    core.require(len(encoded.encode()) <= copied.REPORT_RESERVE, "calibration_public_report_byte_cap")
    with (public / "calibration_report.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({key: report[key] for key in ("status", "mode", "pilot_complete", "calibration_support_complete",
                                                  "calibration_values_read", "elapsed_seconds", "peak_rss_kib")}))
    return 0 if report["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    core.require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "verified_compute_allocation_required")
    core.require(os.environ.get("SLURM_CPUS_PER_TASK") == "1" and all(os.environ.get(name) == "1" for name in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
                 "single_cpu_thread_environment_required")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = MEMORY_BYTES if soft == resource.RLIM_INFINITY else min(soft, MEMORY_BYTES)
    core.require(hard == resource.RLIM_INFINITY or hard >= target, "two_gib_memory_limit_unavailable")
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    return run(args.mode, args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
