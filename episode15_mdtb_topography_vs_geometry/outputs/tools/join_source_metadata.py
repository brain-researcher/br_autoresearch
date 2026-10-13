#!/usr/bin/env python3
"""EP15 Stage 1: ZIP indexes, design tables, demography, one A header/person.

No beta/anatomy voxel is decoded. No Task B image stream is opened. This
custodian metadata operation neither assigns roles nor authorizes a candidate
to access the mixed source archives. Uses Python's standard library only.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import time
import zipfile

from inspect_registration_headers import nifti_header


SOURCE_RELATIVE = "data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0/source"
PROVIDER_NAMING = "https://github.com/DiedrichsenLab/Functional_Fusion/blob/main/Functional_Fusion/dataset.py"
SESSIONS = {"ses-s1": "A", "ses-s2": "B"}
EXPECTED_BETAS = {"ses-s1": 736, "ses-s2": 784}
REGINFO_COLUMNS = {"run", "task_name", "cond_name", "reg_id", "instruction",
                   "common", "task_code", "cond_code", "half"}
DOMAINS = {
    "CPRO": ["CPRO"],
    "prediction": ["Prediction", "PredictViol", "PredictScram"],
    "spatial_map": ["SpatialMapEasy", "SpatialMapMed", "SpatialMedDiff"],
    "movie": ["NatureMovie", "RomanceMovie", "LandscapeMovie"],
    "mental_rotation": ["MentalRotEasy", "MentalRotMed", "MentalRotDiff"],
    "emotion_processing": ["BodyMotionIntact", "BodyMotionScram"],
    "response_alternation": ["RespAltEasy", "RespAltMed", "RespAltDiff"],
}
MAX_TABLE_BYTES = 1024 * 1024
GRID_KEYS = ("shape", "pixdim", "sform_code", "sform_rows", "qform_code",
             "quaternion_and_offsets")
PLAN_COLUMNS = ["participant_id", "session", "task_set", "run", "reg_id",
                "task_name", "cond_name", "instruction", "common", "task_code",
                "cond_code", "half", "geometry_partition", "beta_member"]


def participant_ids_and_completeness(payload, expected_count=24):
    """Read demography but return no person-level demographic values."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
    required = {"participant_id", "sex", "age", "native_english", "ses-rest"}
    if set(reader.fieldnames or []) != required:
        raise ValueError("Unexpected participant metadata columns")
    rows = list(reader)
    ids = [row["participant_id"] for row in rows]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError("Expected every unique participant; no replacement or reduced N")
    if any(not re.fullmatch(r"sub-\d{2}", pid) for pid in ids):
        raise ValueError("Unsafe or unexpected participant identifier")
    missing_tokens = {"", "n/a", "na", "nan", "none"}
    counts = {key: 0 for key in ("sex", "age", "native_english", "ses-rest")}
    invalid_age = 0
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed participant metadata row")
        for key in counts:
            if row[key].strip().lower() in missing_tokens:
                counts[key] += 1
        if row["age"].strip().lower() not in missing_tokens:
            try:
                if not math.isfinite(float(row["age"])):
                    invalid_age += 1
            except ValueError:
                invalid_age += 1
    return sorted(ids), {
        "participants": len(ids), "missing_values_by_field": counts,
        "invalid_nonmissing_age_count": invalid_age,
        "age_available_for_future_role_assignment": counts["age"] == 0 and invalid_age == 0,
        "missing_categories_must_form_explicit_levels": True,
        "person_level_demographic_values_exported": False,
    }


def beta_member(pid, session, run, reg_id):
    # Official DataSet.get_data_fnames: reg_id is used as supplied (includes 0).
    return f"{pid}/func/{session}/{pid}_{session}_run-{run:02d}_reg-{reg_id:02d}_beta.nii"


def parse_reginfo(payload, pid, session):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
    if set(reader.fieldnames or []) != REGINFO_COLUMNS:
        raise ValueError(f"{pid}/{session}: unexpected reginfo columns")
    records, seen = [], set()
    for line, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{pid}/{session}: malformed design row {line}")
        try:
            run, reg_id, instruction, common, half = (
                int(row[key]) for key in ("run", "reg_id", "instruction", "common", "half"))
        except ValueError as error:
            raise ValueError(f"{pid}/{session}: noninteger design field at row {line}") from error
        if not 1 <= run <= 16 or reg_id < 0 or instruction not in (0, 1) or common not in (0, 1):
            raise ValueError(f"{pid}/{session}: invalid run/regressor/indicator")
        if half != (1 if run <= 8 else 2):
            raise ValueError(f"{pid}/{session}: source half disagrees with registered run halves")
        if (run, reg_id) in seen:
            raise ValueError(f"{pid}/{session}: duplicate run/reg_id")
        seen.add((run, reg_id))
        if bool(instruction) != (row["task_code"] == "instrct"):
            raise ValueError(f"{pid}/{session}: instruction flag/code disagreement")
        if not row["cond_name"] or not row["task_name"]:
            raise ValueError(f"{pid}/{session}: missing task or condition label")
        records.append({
            "participant_id": pid, "session": session, "task_set": SESSIONS[session],
            "run": run, "reg_id": reg_id, "task_name": row["task_name"],
            "cond_name": row["cond_name"], "instruction": instruction,
            "common": common, "task_code": row["task_code"], "cond_code": row["cond_code"],
            "half": half, "geometry_partition": (run - 1) // 4 + 1,
            "beta_member": beta_member(pid, session, run, reg_id),
        })
    if not records:
        raise ValueError(f"{pid}/{session}: empty reginfo")
    return sorted(records, key=lambda row: (row["run"], row["reg_id"]))


def validate_session(records, member_counts, expected_beta_count):
    pid, session = records[0]["participant_id"], records[0]["session"]
    expected = {row["beta_member"] for row in records}
    if len(records) != expected_beta_count:
        raise ValueError(f"{pid}/{session}: unexpected total beta/design row count")
    prefix = f"{pid}/func/{session}/{pid}_{session}_run-"
    actual = {name for name in member_counts if name.startswith(prefix) and name.endswith("_beta.nii")}
    if actual != expected or any(member_counts[name] != 1 for name in expected):
        raise ValueError(f"{pid}/{session}: missing, extra or duplicate beta members")
    if {row["run"] for row in records} != set(range(1, 17)):
        raise ValueError(f"{pid}/{session}: not every registered run is present")
    by_condition = defaultdict(list)
    for row in records:
        if not row["instruction"]:
            by_condition[row["cond_name"]].append(row)
    if not by_condition:
        raise ValueError(f"{pid}/{session}: no noninstruction conditions")
    summary = {}
    for condition, rows in sorted(by_condition.items()):
        counts = Counter(row["run"] for row in rows)
        if set(counts) != set(range(1, 17)):
            raise ValueError(f"{pid}/{session}: condition lacks one or more registered runs: {condition}")
        # Source labels must identify one task; do not alias or average task variants.
        signatures = {(row["task_name"], row["task_code"], row["cond_code"], row["common"]) for row in rows}
        if len(signatures) != 1:
            raise ValueError(f"{pid}/{session}: inconsistent condition identity: {condition}")
        signature = next(iter(signatures))
        summary[condition] = {
            "task_name": signature[0], "task_code": signature[1],
            "cond_code": signature[2], "common": signature[3],
            "regressors_per_run": [counts[run] for run in range(1, 17)],
        }
    return summary


def validate_task_sets(a, b):
    shared, b_only = set(a) & set(b), set(b) - set(a)
    contracted = {condition for conditions in DOMAINS.values() for condition in conditions}
    if b_only != contracted:
        raise ValueError("B-only condition labels differ from contract; missing=" +
                         repr(sorted(contracted - b_only)) + "; extra=" + repr(sorted(b_only - contracted)))
    if len(shared) != 14 or len(set(a) | set(b)) != 47:
        raise ValueError("Expected 14 shared and 47 distinct noninstruction conditions")
    if not any(condition.lower() == "rest" for condition in shared):
        raise ValueError("Shared task-design rest condition is missing")
    for conditions in (a, b):
        for condition, item in conditions.items():
            if item["common"] != int(condition in shared):
                raise ValueError(f"Source common flag disagrees with A/B label intersection: {condition}")
    return {
        "Task_A_conditions": sorted(a), "Task_B_conditions": sorted(b),
        "shared_conditions_postdecision_Task_B_diagnostic_only": sorted(shared),
        "Task_B_only_conditions": sorted(b_only),
        "distinct_condition_count": len(set(a) | set(b)),
        "distinct_task_name_count": len({item["task_name"] for table in (a, b) for item in table.values()}),
        "domains": DOMAINS,
        "condition_weights": {condition: {"numerator": 1, "denominator": 7 * len(conditions)}
                              for conditions in DOMAINS.values() for condition in conditions},
        "geometry_conditions_remain_separate": True,
        "pair_weight_requirement": "domain_pair_balanced; full_signed_matrix",
        "numeric_pair_weights_activated": False,
        "map_halves": [list(range(1, 9)), list(range(9, 17))],
        "geometry_partitions": [list(range(start, start + 4)) for start in (1, 5, 9, 13)],
        "primary_geometry_partition_pairs": [[1, 2], [3, 4]],
    }


def read_table(archive, member_counts, name):
    if member_counts[name] != 1:
        raise ValueError(f"Missing or duplicate reginfo member: {name}")
    if archive.getinfo(name).file_size > MAX_TABLE_BYTES:
        raise ValueError("Reginfo exceeds the bounded metadata scope")
    with archive.open(name) as stream:
        payload = stream.read(MAX_TABLE_BYTES + 1)
    if len(payload) > MAX_TABLE_BYTES:
        raise ValueError("Reginfo exceeds the bounded metadata scope")
    return payload


def inspect_participant(archive, pid, expected_counts=EXPECTED_BETAS):
    """The only image stream permitted here is one noninstruction ses-s1 header."""
    member_counts = Counter(info.filename for info in archive.infolist())
    records, summaries = {}, {}
    for session in SESSIONS:
        name = f"{pid}/func/{session}/{pid}_{session}_reginfo.tsv"
        records[session] = parse_reginfo(read_table(archive, member_counts, name), pid, session)
        summaries[session] = validate_session(records[session], member_counts, expected_counts[session])
    task_sets = validate_task_sets(summaries["ses-s1"], summaries["ses-s2"])
    first_a = next(row for row in records["ses-s1"] if not row["instruction"])
    with archive.open(first_a["beta_member"]) as stream:
        header = nifti_header(stream)
    if len(header["shape"]) != 3:
        raise ValueError(f"{pid}: expected a 3-D Task A beta header")
    return records, summaries, task_sets, {"member": first_a["beta_member"], "header": header}


def timeout_handler(signum, frame):
    raise TimeoutError("Stage 1 internal walltime bound reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=540)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified Slurm compute shell")
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("Internal bound must be between 1 and 540 seconds")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_seconds)
    start = time.monotonic()
    source = Path(os.environ["OAK"]) / SOURCE_RELATIVE
    episode_outputs = Path(__file__).resolve().parents[1]
    if episode_outputs / "runs/source_qualification" not in args.output_dir.resolve().parents:
        raise ValueError("Output must be a new attempt below episode outputs/runs/source_qualification")
    anatomy_path = episode_outputs / "runs/source_qualification/attempt001/anatomical_headers.json"
    anatomy = json.loads(anatomy_path.read_text())
    with (source / "participants.tsv").open("rb") as stream:
        payload = stream.read(MAX_TABLE_BYTES + 1)
    if len(payload) > MAX_TABLE_BYTES:
        raise ValueError("Participant metadata exceeds bound")
    ids, demography = participant_ids_and_completeness(payload)
    anatomical = {item["participant_id"]: item for item in anatomy["participants"]}
    if set(anatomical) != set(ids):
        raise ValueError("Prior all-participant anatomy headers do not match the source cohort")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "analysis_id": "source_qualification", "stage": "metadata_join",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source": str(source), "provider_beta_naming_reference": PROVIDER_NAMING,
        "provider_naming_rule": "DataSet.get_data_fnames: run-{run:02}_reg-{reg_id:02}_beta.nii",
        "demographic_completeness": demography, "participants": [],
        "beta_or_anatomy_voxel_elements_decoded": False, "Task_B_image_streams_opened": 0,
        "planned_Task_A_image_headers_per_participant": 1, "NIfTI_header_bytes_requested_per_image": 348,
        "ZIP_decompressor_may_buffer_beyond_requested_header_bytes": True,
        "private_Task_A_reliability_computed": False, "roles_assigned": False,
        "scientific_margins_activated": False, "common_support_qualified": False,
        "all_beta_headers_qualified": False, "candidate_scoring_authorized": False,
        "regressor_members_table": "regressor_members.tsv",
        "metadata_join_complete": False,
    }
    canonical_summaries = None
    try:
        with (args.output_dir / "regressor_members.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PLAN_COLUMNS, delimiter="\t")
            writer.writeheader()
            for pid in ids:
                with zipfile.ZipFile(source / f"MDTB_{pid}.zip") as archive:
                    records, summaries, task_sets, selected_header = inspect_participant(archive, pid)
                if canonical_summaries is None:
                    canonical_summaries = summaries
                    report["condition_design"] = task_sets
                    report["condition_multiplicity_by_session"] = summaries
                elif summaries != canonical_summaries:
                    raise ValueError(f"{pid}: condition identities or per-run multiplicities differ across participants")
                for session in SESSIONS:
                    writer.writerows(records[session])
                mask = anatomical[pid]["headers"]["desc-cereb_mask.nii"]
                same_grid = all(selected_header["header"][key] == mask[key] for key in GRID_KEYS)
                report["participants"].append({
                    "participant_id": pid, "beta_members_joined": {session: len(rows) for session, rows in records.items()},
                    "selected_Task_A_header": selected_header,
                    "selected_Task_A_header_equals_prior_native_cerebellar_mask_grid": same_grid,
                })
        report["metadata_join_complete"] = True
        report["all_selected_Task_A_headers_equal_native_cerebellar_mask_grid"] = all(
            item["selected_Task_A_header_equals_prior_native_cerebellar_mask_grid"] for item in report["participants"])
        report["next_numeric_stage"] = (
            "Adopt the anatomy-only common-support and strict sampling rule before anatomy values. "
            "Then qualify every Task A beta header while reading authorized Task A values for "
            "private reliability; metadata alone cannot instantiate the 12/12 roles.")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - start
        (args.output_dir / "source_metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"metadata_join_complete": True, "participants": len(report["participants"]),
                      "Task_B_image_streams_opened": 0, "voxel_elements_decoded": False,
                      "roles_assigned": False, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
