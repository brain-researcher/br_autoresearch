#!/usr/bin/env python3
"""Prepare two audit manifests from existing outcome-free metadata only.

Run inside the coordinator's admitted Slurm job after audit adoption. Read only
five fixed published JSON/TSV files, never a container, source image, private
reliability value, support array, or fitted outcome. Recorded container paths
are checked lexically, without stat/list/open. The two readers retain their
own adoption, complete-plan and prediction-freeze gates. This adapter does
not itself release inputs or authorize evaluation. Existing metadata and prior
materialization checks are reused; scientific recipes are unchanged.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

from build_condition_partition_plan import MEMBER_COLUMNS
from join_source_metadata import EXPECTED_BETAS, SESSIONS, beta_member, validate_session, validate_task_sets
from prepare_role_handoffs import ANATOMY
from run_development_point_panel import OUTPUTS, MANIFEST_BYTES, MEMBER_BYTES, _bounded, _compute, _path, _write_new


AUDIT_IDS = ("sub-02", "sub-03", "sub-04", "sub-06", "sub-09", "sub-10",
             "sub-19", "sub-22", "sub-26", "sub-28", "sub-29", "sub-30")
DEVELOPMENT_IDS = ("sub-08", "sub-12", "sub-14", "sub-15", "sub-17", "sub-18",
                   "sub-20", "sub-21", "sub-24", "sub-25", "sub-27", "sub-31")
INPUTS = {
    "materialization": "runs/role_handoffs/full001/materialization_report.json",
    "anatomy": "runs/source_qualification/attempt001/anatomical_headers.json",
    "metadata": "runs/source_qualification/metadata_join001/source_metadata.json",
    "members": "runs/source_qualification/condition_partition_plan001/role_member_plan.tsv",
    "roles": "runs/role_assignment/attempt001/participant_roles.json",
}
ROLE_BY_SESSION = {"ses-s1": "audit_calibration", "ses-s2": "sealed_audit"}
INTEGER_FIELDS = ("run", "reg_id", "instruction", "common", "half", "geometry_partition")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, allow_nan=False, indent=2) + "\n").encode()


def people(document):
    rows = document["participants"]
    result = {row["participant_id"]: row for row in rows}
    require(len(rows) == len(result) == 24 and set(result) == set(AUDIT_IDS + DEVELOPMENT_IDS),
            "original_unique24_metadata_people_required")
    return result


def prepare_payloads(final_view, check):
    documents = {}
    for key, relative in INPUTS.items():
        check()
        payload = _bounded(OUTPUTS / relative, MEMBER_BYTES if key == "members" else MANIFEST_BYTES)
        documents[key] = payload if key == "members" else json.loads(payload)
    materialization, anatomy, metadata, roles = (documents[key] for key in
                                                ("materialization", "anatomy", "metadata", "roles"))
    require(tuple(roles["audit_ids"]) == AUDIT_IDS
            and tuple(roles["development_ids"]) == DEVELOPMENT_IDS
            and roles.get("participants") == 24 and roles.get("enumeration_complete") is True,
            "original_complete12plus12_role_assignment_required")
    require(materialization.get("status") == "complete_unreleased_role_materialization"
            and materialization.get("materialization_complete") is True
            and materialization.get("workers_joined") is True
            and materialization.get("completed_containers") == 36
            and materialization.get("image_arrays_decoded") is False
            and materialization.get("scientific_audit_opened") is False,
            "completed_opaque_role_materialization_required")
    workers = materialization["worker_results"]
    require(len(workers) == 4 and all(row.get("complete") is True for row in workers),
            "all_materialization_workers_complete_required")
    private_root = OUTPUTS / "cache/private/role_handoffs/full001"
    require(_path(materialization["private_unreleased_output"]) == private_root,
            "existing_fixed_role_handoff_root_required")
    containers = {role: {} for role in ROLE_BY_SESSION.values()}
    for worker in workers:
        for item in worker["containers"]:
            role, pid = item["handoff"], item["participant_id"]
            if role == "development":
                require(pid in DEVELOPMENT_IDS, "development_role_identity_mismatch")
                continue
            require(role in containers and pid in AUDIT_IDS and pid not in containers[role],
                    "unique_original_audit_container_roles_required")
            session = "ses-s1" if role == "audit_calibration" else "ses-s2"
            path = _path(item["path"])
            require(path == private_root / role / (pid + ".zip")
                    and item.get("closed_original_CRC_verified") is True
                    and item.get("members") == EXPECTED_BETAS[session] + (len(ANATOMY) if session == "ses-s1" else 0),
                    "existing_verified_exact_role_filtered_container_required")
            containers[role][pid] = str(path)
    require(all(set(group) == set(AUDIT_IDS) for group in containers.values()),
            "both_complete12_audit_handoffs_required")
    require(anatomy.get("forward_transform_headers_share_grid") is True
            and anatomy.get("all_voxel_payloads_closed") is True
            and metadata.get("metadata_join_complete") is True
            and metadata.get("all_selected_Task_A_headers_equal_native_cerebellar_mask_grid") is True
            and metadata.get("Task_B_image_streams_opened") == 0,
            "completed_outcome_free_header_metadata_required")
    anatomical, functional = people(anatomy), people(metadata)
    bindings = {session: [] for session in ROLE_BY_SESSION}
    for pid in AUDIT_IDS:
        check()
        headers, person = anatomical[pid]["headers"], functional[pid]
        require(set(headers) == set(ANATOMY) and person["beta_members_joined"] == EXPECTED_BETAS
                and person["selected_Task_A_header_equals_prior_native_cerebellar_mask_grid"] is True,
                "original_audit_anatomy_and_A_header_metadata_required")
        bindings["ses-s1"].append(dict(participant_id=pid,
            archive_path=containers["audit_calibration"][pid], anatomy_headers=headers,
            task_a_header=person["selected_Task_A_header"]["header"]))
        bindings["ses-s2"].append(dict(participant_id=pid,
            archive_path=containers["sealed_audit"][pid]))
    table = csv.DictReader(io.StringIO(documents["members"].decode("utf-8-sig")), delimiter="\t")
    require(table.fieldnames == MEMBER_COLUMNS, "original_member_columns_required")
    selected = {session: [] for session in ROLE_BY_SESSION}
    grouped = {(pid, session): [] for pid in AUDIT_IDS for session in ROLE_BY_SESSION}
    seen = set()
    for index, row in enumerate(table):
        check()
        require(index < 24 * sum(EXPECTED_BETAS.values()) and None not in row
                and all(value is not None for value in row.values()), "bounded_complete_member_rows_required")
        pid, session = row["participant_id"], row["session"]
        require(pid in AUDIT_IDS + DEVELOPMENT_IDS and session in ROLE_BY_SESSION,
                "original_participant_and_session_required")
        converted = dict(row)
        for name in INTEGER_FIELDS:
            converted[name] = int(row[name])
        run, reg = converted["run"], converted["reg_id"]
        key = pid, session, run, reg
        require(key not in seen and 1 <= run <= 16 and reg >= 0
                and row["task_set"] == SESSIONS[session]
                and converted["half"] == (1 if run <= 8 else 2)
                and converted["geometry_partition"] == (run - 1) // 4 + 1
                and row["beta_member"] == beta_member(pid, session, run, reg),
                "unique_original_member_identity_and_partition_required")
        seen.add(key)
        if pid in DEVELOPMENT_IDS:
            require(row["participant_role"] == row["planned_handoff"] == "development",
                    "development_rows_must_keep_development_role")
            continue
        require(row["participant_role"] == "audit" and row["planned_handoff"] == ROLE_BY_SESSION[session],
                "audit_rows_must_keep_exact_session_role")
        selected[session].append(row)
        grouped[pid, session].append(converted)
    require(len(seen) == 24 * sum(EXPECTED_BETAS.values()), "all_original_member_metadata_required")
    summaries = metadata["condition_multiplicity_by_session"]
    require(set(summaries) == set(ROLE_BY_SESSION) and len(summaries["ses-s1"]) == 29
            and len(summaries["ses-s2"]) == 32, "original29A_and32B_condition_metadata_required")
    validate_task_sets(summaries["ses-s1"], summaries["ses-s2"])
    for (pid, session), rows in grouped.items():
        check()
        require(len(rows) == EXPECTED_BETAS[session], "complete_person_session_metadata_required")
        actual = validate_session(rows, Counter(row["beta_member"] for row in rows), EXPECTED_BETAS[session])
        require(actual == summaries[session], "unchanged_condition_multiplicities_required")
    payloads = {}
    for session, prefix in (("ses-s1", "calibration"), ("ses-s2", "evaluation")):
        filename = "Audit_A_members.tsv" if session == "ses-s1" else "Audit_B_members.tsv"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=MEMBER_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected[session])
        payloads[filename] = stream.getvalue().encode()
        payloads[prefix + "_manifest.json"] = encode(dict(release_bindings=bindings[session],
            member_rows_path=str(final_view / filename),
            condition_multiplicity_by_session={session: summaries[session]}))
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    parser.add_argument("--durable-byte-cap", required=True, type=int)
    args = parser.parse_args()
    _compute()
    require(math.isfinite(args.deadline_epoch) and args.deadline_epoch > time.time()
            and args.rss_limit_kib > 0 and args.durable_byte_cap > 0,
            "future_finite_deadline_and_positive_caps_required")
    output = _path(args.output_dir)
    require(output.parent == OUTPUTS / "runs/observed_map_audit" and not output.exists(),
            "fresh_direct_child_of_fixed_audit_run_root_required")
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "no_existing_timer_required")
    def check():
        require(time.time() < args.deadline_epoch
                and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= args.rss_limit_kib,
                "metadata_preparation_resource_limit_reached")
    def stop(_signal, _frame):
        raise RuntimeError("metadata_preparation_interrupted")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
        final = output / "view"
        payloads = prepare_payloads(final, check)
        payloads["preparation.json"] = encode(dict(status="complete_observed_map_audit_metadata_views",
            job_id=os.environ["SLURM_JOB_ID"], completed_utc=datetime.now(timezone.utc).isoformat(),
            audit_participant_ids=AUDIT_IDS, calibration_member_rows=12 * EXPECTED_BETAS["ses-s1"],
            evaluation_member_rows=12 * EXPECTED_BETAS["ses-s2"],
            container_paths_statted_or_opened=False, audit_values_opened=False,
            separation="procedural_same_account_not_technically_enforced"))
        require(sum(map(len, payloads.values())) <= args.durable_byte_cap, "durable_byte_cap_exceeded")
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        pending = output / ".pending"
        pending.mkdir(mode=0o700)
        for name, payload in payloads.items():
            check()
            _write_new(pending / name, payload)
        check()
        pending.rename(final)
        print(json.dumps(dict(status="complete_observed_map_audit_metadata_views", view=str(final))))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
