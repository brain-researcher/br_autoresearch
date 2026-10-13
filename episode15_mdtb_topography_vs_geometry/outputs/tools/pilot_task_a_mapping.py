#!/usr/bin/env python3
"""Map one fixed participant's Task A to frozen support; export cost/QC only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import time
import zipfile

import numpy as np

from build_anatomical_support import (GRID_KEYS, MAX_MEMBER_BYTES, SCRATCH_BUDGET_BYTES,
                                      SOURCE_RELATIVE, lattice_from_header, read_image,
                                      scale_values, selected_sform, strict_trilinear,
                                      world_to_voxel)
from join_source_metadata import PLAN_COLUMNS, beta_member, validate_session


SUPPORT_POINTS = 208
CONDITION_COUNT = 29
TOTAL_A_REGRESSORS = 736
MAX_PLAN_BYTES = 16 * 1024 * 1024


def choose_participant(joined, anatomy, support_report):
    """Select before data access, without a user-specified participant override."""
    ids = sorted(row["participant_id"] for row in joined["participants"])
    if (len(ids) != 24 or len(set(ids)) != 24
            or any(not re.fullmatch(r"sub-\d{2}", pid) for pid in ids)
            or ids != sorted(row["participant_id"] for row in anatomy["participants"])
            or ids != sorted(support_report["participant_ids"])
            or not joined["metadata_join_complete"]
            or not joined["all_selected_Task_A_headers_equal_native_cerebellar_mask_grid"]
            or not anatomy["forward_transform_headers_share_grid"]
            or not support_report["anatomical_intersection_complete"]
            or support_report["participants_completed"] != 24
            or support_report["common_support_points"] != SUPPORT_POINTS
            or support_report["Task_B_values_read"]):
        raise ValueError("Completed matching all24 metadata/anatomical support prerequisites required")
    return ids[0]


def select_plan(rows, pid, expected_summary, member_counts):
    """Validate the full A design, then expose only its noninstruction members."""
    selected, seen = [], set()
    for original in rows:
        if original["participant_id"] != pid or original["session"] != "ses-s1":
            continue
        row = dict(original)
        for key in ("run", "reg_id", "instruction", "common", "half", "geometry_partition"):
            row[key] = int(row[key])
        run, reg = row["run"], row["reg_id"]
        if (row["task_set"] != "A" or not 1 <= run <= 16 or reg < 0
                or row["instruction"] not in (0, 1) or row["common"] not in (0, 1)
                or bool(row["instruction"]) != (row["task_code"] == "instrct")
                or row["half"] != (1 if run <= 8 else 2)
                or row["geometry_partition"] != (run - 1) // 4 + 1
                or row["beta_member"] != beta_member(pid, "ses-s1", run, reg)
                or not row["cond_name"] or not row["task_name"]):
            raise ValueError("Task-A plan identity, instruction, run or member mismatch")
        if (run, reg) in seen:
            raise ValueError("Duplicate Task-A run/regressor in plan")
        seen.add((run, reg))
        selected.append(row)
    if not selected:
        raise ValueError("Selected participant has no Task-A plan")
    summary = validate_session(selected, member_counts, TOTAL_A_REGRESSORS)
    if summary != expected_summary or len(summary) != CONDITION_COUNT:
        raise ValueError("Task-A condition identities or run multiplicities differ from metadata")
    return sorted((row for row in selected if not row["instruction"]),
                  key=lambda row: (row["run"], row["reg_id"]))


def load_plan(path, pid, expected_summary, member_counts):
    if path.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError("Metadata plan exceeds bounded read size")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != PLAN_COLUMNS:
            raise ValueError("Unexpected metadata plan columns")
        return select_plan(reader, pid, expected_summary, member_counts)


class _CountedStream:
    def __init__(self, stream, owner):
        self.stream, self.owner = stream, owner

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def read(self, size=-1):
        if size < 0 or size > MAX_MEMBER_BYTES:
            raise ValueError("Unbounded image read denied")
        block = self.stream.read(size)
        self.owner.bytes_read += len(block)
        return block


class ExactMemberArchive:
    """Enforce the operation's narrow stream boundary, not OS confidentiality."""
    def __init__(self, archive, allowed):
        self.archive, self.allowed = archive, frozenset(allowed)
        self.opened, self.bytes_read = [], 0

    def getinfo(self, member):
        if member not in self.allowed:
            raise ValueError("Image outside the selected Task-A/forward-field allowlist")
        return self.archive.getinfo(member)

    def open(self, info):
        name = info.filename if hasattr(info, "filename") else info
        if name not in self.allowed:
            raise ValueError("Image outside the selected Task-A/forward-field allowlist")
        self.opened.append(name)
        return _CountedStream(self.archive.open(info), self)


def validate_beta_reference(a_header, mask_header):
    if any(a_header[key] != mask_header[key] for key in GRID_KEYS):
        raise ValueError("Saved Task-A grid does not equal the native cerebellar-mask grid")
    if len(a_header["shape"]) != 3:
        raise ValueError("Task-A beta must be three-dimensional")
    selected_sform(a_header)  # Source A header includes actual spatial units.


def validate_support(arrays, forward_header, expected_points=SUPPORT_POINTS):
    required = {"common_mask", "world_xyz_mm", "source_transform_voxel_ijk", "lattice_affine_mm"}
    if set(arrays) != required:
        raise ValueError("Unexpected frozen common-support fields")
    indices, world, shape, affine = lattice_from_header(forward_header)
    mask = arrays["common_mask"]
    if mask.dtype != np.dtype(bool) or mask.shape != shape or int(mask.sum()) != expected_points:
        raise ValueError("Frozen common support has changed shape/count/type")
    keep = mask.ravel(order="C")
    if (not np.array_equal(arrays["source_transform_voxel_ijk"], indices[keep])
            or not np.array_equal(arrays["world_xyz_mm"], world[keep])
            or not np.array_equal(arrays["lattice_affine_mm"], affine)):
        raise ValueError("Frozen common-support ordering or coordinates disagree with its lattice")
    return indices[keep]


def sample_beta(archive, member, a_header, mask_header, native_world):
    # Validate both expected grids before read_image reads any beta payload.
    validate_beta_reference(a_header, mask_header)
    image = read_image(archive, member, a_header)
    values, valid = strict_trilinear(image, world_to_voxel(image["affine"], native_world))
    if not np.all(valid) or not np.all(np.isfinite(values)):
        raise ValueError("Task-A beta has an invalid or nonfinite fixed-support stencil; no support shrink")
    return values, image["payload_bytes"]


def condition_halves(plan, condition_names, sample, points=SUPPORT_POINTS):
    """Mean regressors within each run, then mean the eight runs within each half."""
    conditions = list(condition_names)
    if len(set(conditions)) != len(conditions):
        raise ValueError("Duplicate condition name")
    lookup = {condition: number for number, condition in enumerate(conditions)}
    sums = np.zeros((16, len(conditions), points), dtype=np.float64)
    counts = np.zeros((16, len(conditions)), dtype=np.int64)
    expected = Counter((row["run"], row["cond_name"]) for row in plan)
    if set(expected) != {(run, condition) for run in range(1, 17) for condition in conditions}:
        raise ValueError("A condition lacks one or more complete registered runs")
    for row in plan:
        values = np.asarray(sample(row), dtype=np.float64)
        if values.shape != (points,) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid fixed-support mapped beta")
        slot = (row["run"] - 1, lookup[row["cond_name"]])
        sums[slot] += values
        counts[slot] += 1
    for (run, condition), count in expected.items():
        if counts[run - 1, lookup[condition]] != count:
            raise ValueError("Incomplete condition/run regressor count")
    run_maps = sums / counts[..., None]
    halves = np.stack((run_maps[:8].mean(axis=0), run_maps[8:].mean(axis=0)))
    if not np.all(np.isfinite(halves)):
        raise ValueError("Nonfinite Task-A half map; no imputation")
    return halves


def half_map_flags(halves):
    """Booleans only; deliberately do not compute a spatial correlation."""
    halves = np.asarray(halves)
    finite = bool(np.all(np.isfinite(halves)))
    nonconstant = finite and bool(np.all(np.any(halves != halves[..., :1], axis=-1)))
    return {"all_half_maps_finite": finite,
            "all_half_maps_nonconstant": nonconstant,
            "half_maps_exported": False,
            "Task_A_reliability_computed": False,
            "Task_A_reliability_values_exported": False}


def deadline(signum, frame):
    raise TimeoutError("Selected Task-A pilot deadline reached; no automatic retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=540)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("Internal deadline must be between 1 and 540 seconds")
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified Slurm compute allocation")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Numeric thread counts must all be one")
    outputs = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if outputs / "runs/source_qualification" not in output.parents:
        raise ValueError("Output must be a new source_qualification attempt in this episode")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    report = {
        "analysis_id": "source_qualification", "stage": "Task_A_mapping_cost_pilot",
        "specification": "outputs/task_a_mapping_pilot_specification.md",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "internal_deadline_seconds": args.max_seconds, "participants_completed": 0,
        "Task_A_beta_members_completed": 0, "beta_voxel_payload_bytes_read": 0,
        "forward_voxel_payload_bytes_read": 0, "Task_B_image_streams_opened": 0,
        "instruction_image_streams_opened": 0, "Task_B_values_read": False,
        "Task_A_reliability_computed": False, "Task_A_reliability_values_exported": False,
        "participant_roles_assigned": False, "scientific_margins_activated": False,
        "candidate_scores_computed": False, "candidate_scoring_authorized": False,
        "half_maps_exported": False, "individual_anatomy_or_native_coordinates_exported": False,
        "preparation_scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        "image_extraction_to_disk": False, "support_changed": False,
        "source_access_boundary": "exact_member_reader_in_coordinator_custodian_operation_not_OS_separation",
        "reused_NIfTI_reader": "build_anatomical_support.read_image_applied_to_forward_field_and_Task_A_betas",
        "pilot_complete": False,
    }
    checked, exit_code = None, 0
    try:
        stage = outputs / "runs/source_qualification"
        joined = json.loads((stage / "metadata_join001/source_metadata.json").read_text())
        anatomy = json.loads((stage / "attempt001/anatomical_headers.json").read_text())
        support_report = json.loads((stage / "anatomical_support001/anatomical_support_report.json").read_text())
        pid = choose_participant(joined, anatomy, support_report)
        report["participant_id"] = pid
        source = Path(os.environ["OAK"]) / SOURCE_RELATIVE
        if source.resolve() != Path(joined["source"]).resolve():
            raise ValueError("Live source root differs from completed metadata source")
        person = next(row for row in joined["participants"] if row["participant_id"] == pid)
        headers = next(row["headers"] for row in anatomy["participants"] if row["participant_id"] == pid)
        a_header = person["selected_Task_A_header"]["header"]
        validate_beta_reference(a_header, headers["desc-cereb_mask.nii"])
        forward_member = f"{pid}/anat/{pid}_space-SUIT_xfm.nii"
        with zipfile.ZipFile(source / f"MDTB_{pid}.zip") as archive:
            member_counts = Counter(info.filename for info in archive.infolist())
            if member_counts[forward_member] != 1:
                raise ValueError("Missing or duplicate selected forward field")
            expected_summary = joined["condition_multiplicity_by_session"]["ses-s1"]
            plan = load_plan(stage / "metadata_join001/regressor_members.tsv", pid, expected_summary, member_counts)
            if person["selected_Task_A_header"]["member"] not in {row["beta_member"] for row in plan}:
                raise ValueError("Saved selected Task-A header is not a planned noninstruction member")
            conditions = sorted(expected_summary)
            if conditions != joined["condition_design"]["Task_A_conditions"]:
                raise ValueError("Condition order differs from completed Task-A design")
            checked = ExactMemberArchive(archive, {forward_member} | {row["beta_member"] for row in plan})
            report.update(planned_Task_A_beta_members=len(plan), condition_count=len(conditions),
                          registered_runs=16, planned_half_maps=2 * len(conditions),
                          fixed_support_points=SUPPORT_POINTS)
            forward = read_image(checked, forward_member, headers["space-SUIT_xfm.nii"])
            report["forward_voxel_payload_bytes_read"] = forward["payload_bytes"]
            with np.load(stage / "anatomical_support001/common_support.npz", allow_pickle=False) as arrays:
                indices = validate_support(arrays, forward["header"])
            native_world = scale_values(forward, forward["values"][indices[:, 0], indices[:, 1], indices[:, 2], 0, :])
            if not np.all(np.isfinite(native_world)):
                raise ValueError("Nonfinite forward coordinates on frozen common support")
            del forward

            def sample(row):
                values, payload_bytes = sample_beta(checked, row["beta_member"], a_header,
                                                    headers["desc-cereb_mask.nii"], native_world)
                report["Task_A_beta_members_completed"] += 1
                report["beta_voxel_payload_bytes_read"] += payload_bytes
                return values

            halves = condition_halves(plan, conditions, sample)
            report.update(half_map_flags(halves))
            del halves, native_world
            if not report["all_half_maps_nonconstant"]:
                raise ValueError("Constant Task-A half map blocks later correlation; no condition exclusion")
        report.update(status="complete_first_participant_Task_A_mapping_pilot", pilot_complete=True,
                      participants_completed=1, fixed_support_preserved=True)
    except Exception as error:
        report.update(status="Task_A_mapping_pilot_failure", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      image_streams_opened=0 if checked is None else len(checked.opened),
                      uncompressed_image_stream_bytes_read=0 if checked is None else checked.bytes_read)
        (output / "task_a_mapping_pilot.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "participants_completed", "Task_A_beta_members_completed",
                                                 "elapsed_seconds", "max_rss_kib", "pilot_complete")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
