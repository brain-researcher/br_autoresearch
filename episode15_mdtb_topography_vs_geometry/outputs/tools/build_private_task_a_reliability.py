#!/usr/bin/env python3
"""Build all24 private Task-A medians; public output contains operational facts only."""
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

from pilot_task_a_mapping import (CONDITION_COUNT, SCRATCH_BUDGET_BYTES, SOURCE_RELATIVE,
                                  SUPPORT_POINTS, ExactMemberArchive, choose_participant,
                                  condition_halves, load_plan, read_image, sample_beta,
                                  scale_values, validate_beta_reference, validate_support)


PRIVATE_FIELDS = ("participant_id", "Task_A_median_centered_spatial_correlation")
PRIVATE_FILENAME = "task_a_reliability.csv"


class ReliabilityFailure(ValueError):
    """A fixed public diagnostic code without any private value."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def validated_ids(ids):
    ids = list(ids)
    if (len(ids) != 24 or len(set(ids)) != 24 or ids != sorted(ids)
            or any(not re.fullmatch(r"sub-\d{2}", pid) for pid in ids)):
        raise ValueError("The complete lexicographically ordered cohort is required")
    return ids


def reliability_median(halves):
    """Ordinary median of all29 centered spatial Pearson coefficients, unclipped."""
    maps = np.asarray(halves, dtype=np.float64)
    if maps.shape != (2, CONDITION_COUNT, SUPPORT_POINTS) or not np.all(np.isfinite(maps)):
        raise ReliabilityFailure("incomplete_or_nonfinite_Task_A_half_maps")
    # Fail on undefined arithmetic; no epsilon, scaling floor or clipping.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        centered = maps - maps.mean(axis=-1, keepdims=True)
        energies = np.sum(centered * centered, axis=-1)
        if not np.all(np.isfinite(energies)) or np.any(energies <= 0):
            raise ReliabilityFailure("undefined_Task_A_centered_spatial_energy")
        denominator = np.sqrt(energies[0]) * np.sqrt(energies[1])
        if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
            raise ReliabilityFailure("undefined_Task_A_correlation_denominator")
        correlations = np.sum(centered[0] * centered[1], axis=-1) / denominator
        if not np.all(np.isfinite(correlations)):
            raise ReliabilityFailure("nonfinite_Task_A_condition_correlation")
        median = float(np.median(correlations))
        if not np.isfinite(median):
            raise ReliabilityFailure("nonfinite_Task_A_median_correlation")
    return median


def collect_private_cohort(ids, get_halves, on_complete=lambda: None):
    """Return medians only after the whole cohort; never export a partial list."""
    records = []
    for pid in validated_ids(ids):
        halves = get_halves(pid)
        median = reliability_median(halves)
        del halves
        records.append(dict(zip(PRIVATE_FIELDS, (pid, median))))
        on_complete()
    return records


def publish_private(records, ids, private_dir):
    """Atomic no-overwrite publication, invoked only after successful all24 work."""
    ids = validated_ids(ids)
    if (len(records) != 24 or [row.get("participant_id") for row in records] != ids
            or any(set(row) != set(PRIVATE_FIELDS) for row in records)
            or not all(type(row[PRIVATE_FIELDS[1]]) is float and np.isfinite(row[PRIVATE_FIELDS[1]])
                       for row in records)):
        raise ValueError("Only the complete finite cohort can be published privately")
    private_dir = Path(private_dir)
    final = private_dir / PRIVATE_FILENAME
    temporary = private_dir / ("." + PRIVATE_FILENAME + ".partial")
    if final.exists():
        raise FileExistsError("Private result already exists; no overwrite")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PRIVATE_FIELDS)
            writer.writeheader()
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publication is atomic and refuses an existing final name.
        # Removing the temporary name leaves one complete private CSV.
        os.link(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return final


def map_participant(pid, source, stage, joined, anatomy, report):
    """Reuse the passed pilot's exact source plan and sampling arithmetic."""
    person = next(row for row in joined["participants"] if row["participant_id"] == pid)
    headers = next(row["headers"] for row in anatomy["participants"] if row["participant_id"] == pid)
    a_header = person["selected_Task_A_header"]["header"]
    validate_beta_reference(a_header, headers["desc-cereb_mask.nii"])
    forward_member = f"{pid}/anat/{pid}_space-SUIT_xfm.nii"
    checked = None
    try:
        with zipfile.ZipFile(source / f"MDTB_{pid}.zip") as archive:
            member_counts = Counter(info.filename for info in archive.infolist())
            if member_counts[forward_member] != 1:
                raise ValueError("Missing or duplicate selected forward field")
            expected_summary = joined["condition_multiplicity_by_session"]["ses-s1"]
            plan = load_plan(stage / "metadata_join001/regressor_members.tsv", pid, expected_summary, member_counts)
            if person["selected_Task_A_header"]["member"] not in {row["beta_member"] for row in plan}:
                raise ValueError("Saved Task-A header is not a planned noninstruction member")
            conditions = sorted(expected_summary)
            if conditions != joined["condition_design"]["Task_A_conditions"]:
                raise ValueError("Condition order differs from completed Task-A metadata")
            checked = ExactMemberArchive(archive, {forward_member} | {row["beta_member"] for row in plan})
            forward = read_image(checked, forward_member, headers["space-SUIT_xfm.nii"])
            report["forward_members_completed"] += 1
            report["forward_voxel_payload_bytes_read"] += forward["payload_bytes"]
            with np.load(stage / "anatomical_support001/common_support.npz", allow_pickle=False) as arrays:
                indices = validate_support(arrays, forward["header"])
            native_world = scale_values(forward, forward["values"][indices[:, 0], indices[:, 1], indices[:, 2], 0, :])
            if not np.all(np.isfinite(native_world)):
                raise ValueError("Nonfinite forward coordinates on frozen support")
            del forward

            def sample(row):
                values, count = sample_beta(checked, row["beta_member"], a_header,
                                            headers["desc-cereb_mask.nii"], native_world)
                report["Task_A_beta_members_completed"] += 1
                report["beta_voxel_payload_bytes_read"] += count
                return values

            return condition_halves(plan, conditions, sample)
    finally:
        if checked is not None:
            report["image_streams_opened"] += len(checked.opened)
            report["uncompressed_image_stream_bytes_read"] += checked.bytes_read


def initial_report(private_path, max_seconds):
    return {
        "analysis_id": "source_qualification", "stage": "private_Task_A_reliability",
        "specification": "outputs/private_task_a_reliability_specification.md",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "internal_deadline_seconds": max_seconds, "participants_expected": 24,
        "participants_completed": 0, "conditions_per_participant": CONDITION_COUNT,
        "fixed_support_points": SUPPORT_POINTS, "Task_A_beta_members_completed": 0,
        "forward_members_completed": 0, "beta_voxel_payload_bytes_read": 0,
        "forward_voxel_payload_bytes_read": 0, "image_streams_opened": 0,
        "uncompressed_image_stream_bytes_read": 0,
        "Task_B_image_streams_opened": 0, "instruction_image_streams_opened": 0,
        "Task_B_values_read": False, "participant_roles_assigned": False,
        "scientific_margins_activated": False, "candidate_scores_computed": False,
        "candidate_scoring_authorized": False, "half_maps_exported": False,
        "condition_correlations_exported": False,
        "individual_anatomy_or_native_coordinates_exported": False,
        "reliability_values_in_public_report": False,
        "private_artifact_path": str(private_path), "private_artifact_published": False,
        "private_reliability_records": 0, "all24_reliability_computable": False,
        "all24_half_maps_finite_and_nonconstant": False, "support_changed": False,
        "preparation_scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        "image_extraction_to_disk": False,
        "private_boundary": "0700_directory_0600_file_same_UID_logical_steward_handoff_not_OS_seal",
        "intended_private_reader": "future_trusted_Task_B_blind_role_assignment_only",
        "automatic_retry": False, "status": "started",
    }


def deadline(signum, frame):
    raise TimeoutError("Fixed private Task-A preparation deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=540)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("Internal deadline must be between one and 540 seconds")
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified Slurm compute allocation")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("Numeric thread counts must all be one")
    outputs = Path(__file__).resolve().parents[1]
    stage = outputs / "runs/source_qualification"
    output, private_dir = args.output_dir.resolve(), args.private_dir.resolve()
    if output.parent != stage or private_dir.parent != outputs / "cache/task_a_private_reliability":
        raise ValueError("Exact episode public/private attempt parents are required")
    if output.exists() or private_dir.exists():
        raise FileExistsError("Fresh public and private attempts are required; no overwrite")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = initial_report(private_dir / PRIVATE_FILENAME, args.max_seconds)
    exit_code = 0
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        private_dir.parent.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(mode=0o700, exist_ok=False)
        joined = json.loads((stage / "metadata_join001/source_metadata.json").read_text())
        anatomy = json.loads((stage / "attempt001/anatomical_headers.json").read_text())
        support = json.loads((stage / "anatomical_support001/anatomical_support_report.json").read_text())
        choose_participant(joined, anatomy, support)  # Reuse passed all24 prerequisites.
        ids = validated_ids(sorted(row["participant_id"] for row in joined["participants"]))
        source = Path(os.environ["OAK"]) / SOURCE_RELATIVE
        if source.resolve() != Path(joined["source"]).resolve():
            raise ValueError("Live source root differs from completed metadata source")

        def get_halves(pid):
            report["operation"] = "strict_Task_A_mapping_and_private_correlation"
            return map_participant(pid, source, stage, joined, anatomy, report)

        def completed():
            report["participants_completed"] += 1

        records = collect_private_cohort(ids, get_halves, completed)
        report.update(all24_reliability_computable=True, all24_half_maps_finite_and_nonconstant=True)
        report["operation"] = "complete_private_publication"
        publish_private(records, ids, private_dir)
        del records
        report.update(status="complete_all24_private_Task_A_reliability",
                      private_artifact_published=True, private_reliability_records=24,
                      all24_reliability_computable=True, all24_half_maps_finite_and_nonconstant=True)
    except Exception as error:
        # Never echo an exception's data, arrays or private correlation values.
        report.update(status="private_Task_A_reliability_failure", failure_category=type(error).__name__,
                      failure_code=error.code if isinstance(error, ReliabilityFailure) else
                      "bounded_preparation_or_publication_failure")
        exit_code = 1
    finally:
        signal.alarm(0)
        # This checks only existence, never reads private contents. A deadline
        # after atomic publication must not be misreported as no artifact.
        report["private_artifact_published"] = (private_dir / PRIVATE_FILENAME).is_file()
        report["private_reliability_records"] = 24 if report["private_artifact_published"] else 0
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output / "private_task_a_reliability_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "participants_completed", "Task_A_beta_members_completed",
                                                 "elapsed_seconds", "max_rss_kib", "private_artifact_published")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
