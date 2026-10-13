#!/usr/bin/env python3
"""Freeze EP17 geometry bins from the completed public voxel-axis table only."""

from __future__ import annotations

import argparse
import csv
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

EPISODE = Path(__file__).resolve().parents[2]
INPUT = EPISODE / "outputs/runs/anatomical_axis/attempt002"
OUTPUT = EPISODE / "outputs/runs/structural_support/attempt001"
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-06")
ROIS = {"left": 1007, "right": 2007}
EXPECTED_COLUMNS = 881359
EXPECTED_COUNTS = dict(zip(SUBJECTS, ((227214, 1169, 1120), (236701, 1103, 942),
                                    (210349, 1174, 1182), (207095, 1032, 836))))
INTERNAL_SECONDS = 60
REPORT_CAP = 256 * 1024
AXIS_CAP = 256 * 1024 * 1024
OUTPUT_CAP = 2 * 1024 * 1024
LINE_CAP = 1024
FIELDS = ("participant", "model_column_zero_based", "c_flat_index_zero_based",
          "i", "j", "k", "x_mm", "y_mm", "z_mm", "atlas_label", "nominated_hemisphere")


class Refusal(Exception):
    pass


def require(condition, code):
    if not condition:
        raise Refusal(code)


def check(deadline):
    require(time.monotonic() < deadline, "attempt_deadline")


def alarm(signum, frame):
    raise Refusal("attempt_deadline_or_termination")


def integer(token):
    value = int(token)
    require(str(value) == token, "noncanonical_axis_integer")
    return value


def load_report():
    path = INPUT / "anatomical_axis_report.json"
    require(path.stat().st_size <= REPORT_CAP, "source_report_byte_cap")
    with path.open("rb") as stream:
        payload = stream.read(REPORT_CAP + 1)
    require(len(payload) <= REPORT_CAP, "source_report_byte_cap")
    report = json.loads(payload)
    require(report.get("analysis_id") == "anatomical_axis" and report.get("status") == "completed"
            and report.get("structural_voxel_axis_complete") is True
            and report.get("anatomical_fusiform_mapping_complete") is True
            and report.get("errors") == [] and report.get("refusals") == [], "anatomical_axis_incomplete")
    require(report.get("mask_column_rule") == "original_union_boolean_C_order"
            and report.get("modelmd_coordinate_rule") == "modelmd[p,0,0,j]"
            and report.get("table_rows_written") == EXPECTED_COLUMNS, "original_axis_contract_changed")
    count_bytes = report.get("table_uncompressed_bytes_written")
    require(type(count_bytes) is int and 0 < count_bytes <= AXIS_CAP, "axis_byte_bound_changed")
    participants = report.get("participants", [])
    require([p.get("participant") for p in participants] == list(SUBJECTS), "participant_roster_changed")
    for item in participants:
        expected = EXPECTED_COUNTS[item["participant"]]
        count = item.get("original_model_columns")
        require(type(count) is int and count == expected[0] and item.get("axis_rows_written") == count
                and item.get("BCD_header_column_counts_agree") is True, "participant_axis_count_changed")
        geometry = item["mask_geometry"]
        shape = geometry.get("shape")
        require(isinstance(shape, list) and len(shape) == 3
                and all(type(n) is int and n > 0 for n in shape) and count <= math.prod(shape), "axis_shape_invalid")
        affine = geometry.get("selected_affine_mm")
        require(isinstance(affine, list) and len(affine) == 4
                and all(isinstance(row, list) and len(row) == 4 for row in affine)
                and all(math.isfinite(float(x)) for row in affine for x in row)
                and affine[3] == [0, 0, 0, 1], "axis_world_geometry_invalid")
        for index, (roi, label) in enumerate(ROIS.items(), 1):
            observed = item["ROIs"][roi]
            require(observed.get("label") == label and observed.get("refusal") is None
                    and type(observed.get("original_axis_voxels")) is int
                    and observed["original_axis_voxels"] == expected[index], "nominated_roi_changed")
    require(sum(p["original_model_columns"] for p in participants) == EXPECTED_COLUMNS,
            "total_original_axis_count_changed")
    return report


class Row:
    __slots__ = ("fields", "column", "y", "z", "bin")

    def __init__(self, fields, column, y, z):
        self.fields, self.column, self.y, self.z = tuple(fields), column, y, z
        self.bin = None


def read_axis(report, deadline, telemetry):
    """One bounded complete pass; retain only the two exact nominated labels."""
    specs = report["participants"]
    bins = {(subject, roi): [] for subject in SUBJECTS for roi in ROIS}
    retained = []
    telemetry.update(axis_rows_read=0, axis_uncompressed_bytes_read=0, nominated_rows_retained=0)
    state, column, previous_flat = 0, 0, -1

    def lines(stream):
        while True:
            raw = stream.readline(LINE_CAP + 1)
            if not raw:
                break
            require(len(raw) <= LINE_CAP, "axis_line_bound")
            telemetry["axis_uncompressed_bytes_read"] += len(raw)
            require(telemetry["axis_uncompressed_bytes_read"] <= report["table_uncompressed_bytes_written"], "axis_byte_count_exceeded")
            yield raw.decode("utf-8")

    source = INPUT / "voxel_axis.csv.gz"
    require(source.stat().st_size <= AXIS_CAP, "compressed_axis_byte_cap")
    with gzip.open(source, "rb") as stream:
        reader = csv.reader(lines(stream), strict=True)
        require(tuple(next(reader)) == FIELDS, "axis_csv_header_changed")
        for fields in reader:
            if telemetry["axis_rows_read"] % 4096 == 0:
                check(deadline)
            require(len(fields) == len(FIELDS) and state < len(SUBJECTS), "axis_row_shape_or_count_changed")
            spec = specs[state]
            subject = SUBJECTS[state]
            require(fields[0] == subject, "axis_participant_order_changed")
            numbers = [integer(token) for token in fields[1:6]]
            observed_column, flat, i, j, k = numbers
            shape = spec["mask_geometry"]["shape"]
            require(observed_column == column and 0 <= i < shape[0] and 0 <= j < shape[1]
                    and 0 <= k < shape[2] and flat == (i * shape[1] + j) * shape[2] + k
                    and flat > previous_flat, "original_axis_identity_changed")
            world = [float(token) for token in fields[6:9]]
            require(all(math.isfinite(value) for value in world), "nonfinite_world_coordinate")
            affine = spec["mask_geometry"]["selected_affine_mm"]
            voxel = (i, j, k, 1)
            predicted = [sum(a * b for a, b in zip(row, voxel)) for row in affine[:3]]
            require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) for a, b in zip(world, predicted)), "axis_world_coordinate_changed")
            label = integer(fields[9])
            roi = "left" if label == 1007 else "right" if label == 2007 else ""
            require(fields[10] == roi, "axis_hemisphere_label_changed")
            if roi:
                row = Row(fields, column, world[1], world[2])
                retained.append(row)
                bins[(subject, roi)].append(row)
                telemetry["nominated_rows_retained"] += 1
            telemetry["axis_rows_read"] += 1
            previous_flat = flat
            column += 1
            if column == spec["original_model_columns"]:
                state, column, previous_flat = state + 1, 0, -1
    require(state == len(SUBJECTS) and column == 0 and telemetry["axis_rows_read"] == EXPECTED_COLUMNS,
            "incomplete_original_axis")
    require(telemetry["axis_uncompressed_bytes_read"] == report["table_uncompressed_bytes_written"], "axis_byte_count_changed")
    for spec in specs:
        for roi in ROIS:
            require(len(bins[(spec["participant"], roi)]) == spec["ROIs"][roi]["original_axis_voxels"], "nominated_roi_count_changed")
    check(deadline)
    return retained, bins


def assign_bins(rows):
    ordered = sorted(rows, key=lambda row: (row.y, row.column))
    base, remainder = divmod(len(ordered), 4)
    start, counts = 0, []
    for group in range(4):
        size = base + (group < remainder)
        selected = sorted(ordered[start:start + size], key=lambda row: (row.z, row.column))
        lower = (size + 1) // 2
        for index, row in enumerate(selected):
            row.bin = 2 * group + 1 + (index >= lower)
        counts.extend((lower, size - lower))
        start += size
    require(start == len(rows), "bin_partition_incomplete")
    return counts


def select_quota(counts):
    require(len(counts) == 64 and all(type(n) is int and n >= 0 for n in counts), "bin_count_contract_changed")
    q = min(32, min(counts) // 2)
    require(q >= 8, "structural_quota_below_minimum")
    return q, 8 * q


class CappedBuffer(io.BytesIO):
    def write(self, data):
        require(self.tell() + len(data) <= OUTPUT_CAP - REPORT_CAP, "combined_output_byte_cap")
        return super().write(data)


def encode_rows(rows, deadline):
    buffer = CappedBuffer()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=1, mtime=0) as compressed:
        batch = io.StringIO(newline="")
        writer = csv.writer(batch, lineterminator="\n")
        writer.writerow((*FIELDS, "spatial_bin"))
        for index, row in enumerate(rows):
            require(type(row.bin) is int and 1 <= row.bin <= 8, "unassigned_structural_column")
            writer.writerow((*row.fields, f"BIN{row.bin:02d}"))
            if index % 1024 == 1023:
                check(deadline)
                compressed.write(batch.getvalue().encode("utf-8"))
                batch.seek(0)
                batch.truncate(0)
        compressed.write(batch.getvalue().encode("utf-8"))
    check(deadline)
    return buffer.getvalue()


def finish(report, deadline):
    report.update(status="completed", structural_bins_complete=True, N_and_quotas_frozen=True)
    check(deadline)


def run_attempt(out):
    started = time.monotonic()
    deadline = started + INTERNAL_SECONDS
    report = {"analysis_id": "structural_support", "status": "running", "structural_bins_complete": False,
              "N_and_quotas_frozen": False, "source_report": str(INPUT / "anatomical_axis_report.json"),
              "source_axis": str(INPUT / "voxel_axis.csv.gz"), "source_axis_complete_passes": 0,
              "source_arrays_read": False, "neural_values_read": False, "calibration_executed": False,
              "A_all_finite_support_complete": False, "reliable_support_complete": False,
              "anatomical_bank_constructed": False, "source_roles_complete": False,
              "candidate_scoring_run": False, "audit_opened": False,
              "source_components_pruned": False, "nominated_labels": ROIS,
              "quota_rule": "min(32,floor(min_all_64_structural_bin_counts/2));N=8*q;q>=8",
              "bin_rule": "four_y_groups_then_two_z_halves;column_ties;earlier_remainders",
              "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 90,
              "combined_output_byte_cap": OUTPUT_CAP, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "hostname": socket.gethostname(), "participants": [], "errors": []}
    previous = {sig: signal.signal(sig, alarm) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(INTERNAL_SECONDS)
    encoded = None
    try:
        source = load_report()
        retained, bins = read_axis(source, deadline, report)
        report["source_axis_complete_passes"] = 1
        all_counts = []
        for spec in source["participants"]:
            item = {"participant": spec["participant"], "original_model_columns": spec["original_model_columns"], "ROIs": {}}
            for roi, label in ROIS.items():
                counts = assign_bins(bins[(spec["participant"], roi)])
                all_counts.extend(counts)
                item["ROIs"][roi] = {"label": label, "original_axis_voxels": sum(counts),
                                      "bin_counts": {f"BIN{i+1:02d}": value for i, value in enumerate(counts)}}
            report["participants"].append(item)
        report["minimum_structural_bin_count"] = min(all_counts)
        report["q_per_bin"], report["N_per_participant_ROI"] = select_quota(all_counts)
        encoded = encode_rows(retained, deadline)
        report["output_metadata_rows"] = len(retained)
        report["compressed_table_bytes"] = len(encoded)
        check(deadline)
        with (out / "bin_columns.csv.gz").open("xb") as stream:
            stream.write(encoded)
        check(deadline)
        finish(report, deadline)
    except (Refusal, OSError, ValueError, TypeError, KeyError, OverflowError, MemoryError, csv.Error, StopIteration) as error:
        report.update(status="stopped", structural_bins_complete=False, N_and_quotas_frozen=False)
        report["errors"].append(str(error) if isinstance(error, Refusal) else type(error).__name__)
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    report["elapsed_seconds"] = time.monotonic() - started
    report["maximum_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if report["elapsed_seconds"] >= INTERNAL_SECONDS:
        report.update(status="stopped", structural_bins_complete=False, N_and_quotas_frozen=False)
        report["errors"].append("attempt_deadline")
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Table encoding reserves REPORT_CAP before any output publication.
    require(len(payload) <= REPORT_CAP and len(payload) + (len(encoded) if encoded is not None else 0) <= OUTPUT_CAP,
            "combined_output_byte_cap")
    with (out / "structural_support_report.json").open("xb") as stream:
        stream.write(payload)
    print(json.dumps({key: report.get(key) for key in ("status", "structural_bins_complete", "N_and_quotas_frozen", "q_per_bin", "N_per_participant_ROI")}))
    return 0 if report["status"] == "completed" and report["structural_bins_complete"] and report["N_and_quotas_frozen"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "requires_coordinator_owned_compute")
    require(args.out.resolve() == OUTPUT.resolve() and not args.out.exists(), "fresh_selected_attempt_required")
    args.out.mkdir(parents=True)
    return run_attempt(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
