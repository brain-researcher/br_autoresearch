#!/usr/bin/env python3
"""Inventory exact role members, or copy the unreleased sub-08 Task-A pilot.

No image reader, array decoder, model, scorer, reader release or full-copy mode.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import stat
import time
import zipfile
import zlib

from build_condition_partition_plan import MEMBER_COLUMNS
from join_source_metadata import SOURCE_RELATIVE, EXPECTED_BETAS, SESSIONS, beta_member


OUTPUTS = Path(__file__).resolve().parents[1]
PLAN = "runs/source_qualification/condition_partition_plan001"
INPUTS = {
    "report": (PLAN+"/condition_partition_plan_report.json", 1024**2),
    "plan": (PLAN+"/condition_partition_plan.json", 4*1024**2),
    "members": (PLAN+"/role_member_plan.tsv", 32*1024**2),
    "roles": ("runs/role_assignment/attempt001/participant_roles.json", 1024**2),
    "anatomy": ("runs/source_qualification/attempt001/anatomical_headers.json", 4*1024**2),
}
ANATOMY = ("space-SUIT_xfm.nii", "desc-cereb_mask.nii", "label-GMc_probseg.nii", "T1w.nii")
HANDOFF_COUNTS = {"development": 18288, "audit_calibration": 8880, "sealed_audit": 9408}
CHUNK = 1024**2
REPORT_NAME = "handoff_report.json"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class HandoffFailure(ValueError):
    pass


def read_bounded(path, bound):
    with Path(path).open("rb") as stream:
        value = stream.read(bound+1)
    if len(value) > bound:
        raise HandoffFailure("metadata_byte_bound_exceeded")
    return value


def load_inputs(outputs=OUTPUTS):
    return {key: (payload if key == "members" else json.loads(payload))
        for key, (relative, bound) in INPUTS.items()
        for payload in [read_bounded(Path(outputs)/relative, bound)]}


def safe_member(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))
            or PurePosixPath(name).is_absolute()):
        raise HandoffFailure("unsafe_member_path")
    return name


def expected_members(inputs, check=lambda: None):
    report, plan, roles, anatomy = (inputs[key] for key in ("report", "plan", "roles", "anatomy"))
    if (report.get("status") != "complete_condition_partition_member_plan"
            or report.get("plan_complete") is not True or report.get("member_rows") != 36480
            or report.get("output_files") != ["condition_partition_plan.json", "role_member_plan.tsv"]
            or plan.get("member_rows") != 36480 or plan.get("member_table") != "role_member_plan.tsv"
            or roles.get("enumeration_complete") is not True or roles.get("participants") != 24):
        raise HandoffFailure("completed_fixed_role_and_member_plan_required")
    development, audit = roles["development_ids"], roles["audit_ids"]
    if (len(development) != 12 or len(audit) != 12 or development != sorted(set(development))
            or audit != sorted(set(audit)) or set(development)&set(audit)
            or plan.get("development_ids") != development or plan.get("audit_ids") != audit
            or development[0] != "sub-08"):
        raise HandoffFailure("fixed_disjoint_12_12_roles_required")
    ids = sorted(development+audit)
    anatomical = {row["participant_id"]: row for row in anatomy["participants"]}
    if (len(anatomy["participants"]) != 24 or set(anatomical) != set(ids)
            or any(set(ANATOMY)-set(row["headers"]) for row in anatomical.values())):
        raise HandoffFailure("prior_exact_four_anatomy_headers_for_all24_required")
    rows = csv.DictReader(io.StringIO(inputs["members"].decode("utf-8-sig")), delimiter="\t")
    if rows.fieldnames != MEMBER_COLUMNS:
        raise HandoffFailure("original_role_member_columns_required")
    records, seen, counts, runs = [], set(), Counter(), {}
    for index, row in enumerate(rows):
        if index % 1024 == 0:
            check()
        if index >= 36480 or None in row or any(value is None for value in row.values()):
            raise HandoffFailure("complete_member_rows_required")
        pid, session = row["participant_id"], row["session"]
        run, reg = int(row["run"]), int(row["reg_id"])
        role = "development" if pid in development else "audit"
        handoff = "development" if role == "development" else ("audit_calibration" if session == "ses-s1" else "sealed_audit")
        name = safe_member(row["beta_member"])
        key = (pid, session, run, reg)
        if (pid not in ids or session not in SESSIONS or not 1 <= run <= 16 or reg < 0 or key in seen
                or row["task_set"] != SESSIONS[session] or row["participant_role"] != role
                or row["planned_handoff"] != handoff or name != beta_member(pid, session, run, reg)):
            raise HandoffFailure("source_member_identity_or_role_mismatch")
        seen.add(key)
        counts[pid, session] += 1
        runs.setdefault((pid, session), set()).add(run)
        records.append({"participant_id": pid, "member": name, "handoff": handoff, "kind": SESSIONS[session]})
    if (len(records) != 36480 or any(counts[pid, session] != EXPECTED_BETAS[session]
            or runs.get((pid, session)) != set(range(1, 17)) for pid in ids for session in SESSIONS)):
        raise HandoffFailure("all_36480_functional_members_and48_sessions_required")
    for pid in ids:
        for suffix in ANATOMY:
            records.append({"participant_id": pid, "member": safe_member(f"{pid}/anat/{pid}_{suffix}"),
                "handoff": "development" if pid in development else "audit_calibration", "kind": "anatomy"})
    if dict(Counter(row["handoff"] for row in records)) != HANDOFF_COUNTS:
        raise HandoffFailure("whole_person_handoff_counts_changed")
    return sorted(records, key=lambda row: (row["participant_id"], row["member"]))


def directory(archive):
    infos = {}
    for item in archive.infolist():
        name = item.filename[:-1] if item.is_dir() else item.filename
        safe_member(name)
        if item.filename in infos:
            raise HandoffFailure("duplicate_zip_member")
        mode = stat.S_IFMT(item.external_attr >> 16)
        if mode not in (0, stat.S_IFDIR if item.is_dir() else stat.S_IFREG):
            raise HandoffFailure("unexpected_zip_member_type")
        infos[item.filename] = item
    return infos


def member_metadata(info):
    if (info.is_dir() or info.flag_bits & 1 or info.file_size <= 0 or info.compress_size < 0
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
        raise HandoffFailure("unsupported_selected_zip_member")
    return {"bytes": info.file_size, "compressed_bytes": info.compress_size,
        "zip_crc32": info.CRC, "zip_compression": info.compress_type}


def pilot_members(records):
    selected = [row for row in records if row["participant_id"] == "sub-08" and row["kind"] in ("A", "anatomy")]
    if (Counter(row["kind"] for row in selected) != {"A": 736, "anatomy": 4}
            or any(row["handoff"] != "development" for row in selected)):
        raise HandoffFailure("exact_sub08_development_736A_plus4_anatomy_required")
    return selected


def inventory(source, records, report, check=lambda: None, progress=lambda: None):
    result = []
    for pid in sorted({row["participant_id"] for row in records}):
        check()
        with zipfile.ZipFile(source/f"MDTB_{pid}.zip") as archive:
            infos = directory(archive)
            report["zip_directories_read"] += 1
            for row in (item for item in records if item["participant_id"] == pid):
                check()
                if row["member"] not in infos:
                    raise HandoffFailure("selected_source_member_missing")
                result.append({**row, **member_metadata(infos[row["member"]])})
        report["members_inventoried"] = len(result)
        progress()
    selected = pilot_members(result)
    return {"inventory_complete": True, "source": str(source), "members": result,
        "member_count": len(result), "bytes_by_handoff": {key: sum(row["bytes"] for row in result if row["handoff"] == key)
            for key in HANDOFF_COUNTS}, "total_bytes": sum(row["bytes"] for row in result),
        "pilot": {"participant_id": "sub-08", "functional_Task_A_members": 736, "anatomy_members": 4,
            "member_count": 740, "bytes": sum(row["bytes"] for row in selected),
            "compressed_bytes": sum(row["compressed_bytes"] for row in selected)}}


def bind_inventory(document, terminal, records, source):
    if (terminal.get("mode") != "inventory" or terminal.get("status") != "inventory_complete"
            or terminal.get("operation_complete") is not True or document.get("inventory_complete") is not True
            or document.get("source") != str(source) or document.get("member_count") != len(records)):
        raise HandoffFailure("completed_inventory_required")
    saved = document["members"]
    if len(saved) != len(records) or any({key: row[key] for key in original} != original for row, original in zip(saved, records)):
        raise HandoffFailure("inventory_fixed_member_plan_changed")
    selected = pilot_members(saved)
    if (any(type(row[key]) is not int or row[key] < 0 for row in selected for key in
            ("bytes", "compressed_bytes", "zip_crc32", "zip_compression"))
            or sum(row["bytes"] for row in selected) != document["pilot"]["bytes"]):
        raise HandoffFailure("invalid_pilot_size_forecast")
    return selected


def copy_members(archive, records, private, report, check=lambda: None, progress=lambda: None):
    try:
        _copy_members(archive, records, private, report, check, progress)
    except Exception:
        # A write can persist bytes before raising. Keep acknowledged-byte
        # counters distinct from the actual closed partial file left behind.
        if report.get("active_member") is not None:
            destination = private/"development"/report["active_member"]
            partial = destination.with_name(destination.name+".partial")
            try:
                report["active_partial_bytes_on_disk"] = partial.stat().st_size if partial.exists() else 0
            except OSError as error:
                report["active_partial_size_error"] = f"{type(error).__name__}: {error}"
        raise


def _copy_members(archive, records, private, report, check, progress):
    infos = directory(archive)
    # Validate all selected metadata before the first byte stream.
    for row in records:
        check()
        if row["member"] not in infos or member_metadata(infos[row["member"]]) != {
                key: row[key] for key in ("bytes", "compressed_bytes", "zip_crc32", "zip_compression")}:
            raise HandoffFailure("source_changed_since_inventory")
        if row["participant_id"] != "sub-08" or row["handoff"] != "development" or row["kind"] not in ("A", "anatomy"):
            raise HandoffFailure("pilot_may_copy_only_sub08_development_TaskA_and_anatomy")
        safe_member(row["member"])
    for row in records:
        check()
        destination = private/"development"/row["member"]
        partial = destination.with_name(destination.name+".partial")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists() or partial.exists():
            raise FileExistsError("pilot_member_already_exists")
        report["active_member"] = row["member"]
        report["active_member_bytes_read"] = 0
        report["active_member_bytes_written"] = 0
        report["active_partial_bytes_on_disk"] = None
        progress()
        check()
        report["source_member_open_attempts"] += 1
        if report["first_payload_access_attempt_utc"] is None:
            report["first_payload_access_attempt_utc"] = datetime.now(timezone.utc).isoformat()
        with archive.open(row["member"]) as reader:
            report["source_member_streams_opened"] += 1
            with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as writer:
                while True:
                    check()
                    chunk = reader.read(CHUNK)
                    if not chunk:
                        break
                    if report["first_payload_access_utc"] is None:
                        report["first_payload_access_utc"] = datetime.now(timezone.utc).isoformat()
                    report["source_bytes_read"] += len(chunk)
                    report["active_member_bytes_read"] += len(chunk)
                    if report["active_member_bytes_read"] > row["bytes"]:
                        raise HandoffFailure("member_exceeds_inventory_length")
                    check()
                    written = writer.write(chunk)
                    report["bytes_written"] += written
                    report["active_member_bytes_written"] += written
                    if written != len(chunk):
                        raise HandoffFailure("short_member_write")
        check()
        if partial.stat().st_size != row["bytes"] or report["active_member_bytes_read"] != row["bytes"]:
            raise HandoffFailure("incomplete_member_copy")
        crc, reread = 0, 0
        with partial.open("rb") as reader:
            while True:
                check()
                chunk = reader.read(CHUNK)
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
                reread += len(chunk)
                report["destination_bytes_verified"] += len(chunk)
        if crc != row["zip_crc32"] or reread != row["bytes"]:
            raise HandoffFailure("closed_copy_failed_original_ZIP_CRC_or_length")
        check()
        os.link(partial, destination)
        partial.unlink()
        report["members_copied"] += 1
        report["active_member"] = None
        progress()


def write_json(path, value):
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def run(args):
    if ("-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or any(os.environ.get(key) != "1" for key in THREAD_ENV)):
        raise RuntimeError("one_coordinator_owned_compute_CPU_and_single_numeric_threads_required")
    limit = 120 if args.mode == "inventory" else 600
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= limit:
        raise HandoffFailure("invalid_internal_deadline")
    output = args.output_dir.resolve()
    if OUTPUTS/"runs/role_handoffs" not in output.parents or output.exists():
        raise FileExistsError("fresh_role_handoff_operational_attempt_required")
    output.mkdir(parents=True)
    began, deadline = time.monotonic(), time.monotonic()+args.max_seconds
    report = {"analysis_id": "role_handoff_preparation", "mode": args.mode, "status": "started", "operation_complete": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "zip_directories_read": 0, "members_inventoried": 0, "members_copied": 0,
        "source_member_open_attempts": 0, "source_member_streams_opened": 0, "source_bytes_read": 0, "bytes_written": 0,
        "destination_bytes_verified": 0, "first_payload_access_attempt_utc": None,
        "first_payload_access_utc": None, "active_member": None,
        "bytes_written_semantics": "acknowledged write return counts; on failure closed partial size is recorded separately",
        "Task_B_streams_opened": 0, "audit_member_streams_opened": 0,
        "image_arrays_decoded": False, "participant_values_publicly_exported": False,
        "all_role_handoffs_materialized": False, "reader_isolation_established": False,
        "reader_activation": False, "candidate_scoring_authorized": False, "scientific_margins_activated": False,
        "same_UID_logical_steward_boundary_only": True, "automatic_retry": False,
        "partial_copies_are_unreleased_and_not_a_completed_handoff": True}
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("whole_handoff_operation_deadline")
    def publish():
        report["elapsed_seconds"] = time.monotonic()-began
        write_json(output/REPORT_NAME, report)
    def interrupted(*_):
        raise TimeoutError("handoff_operation_interrupted")
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    complete = False
    try:
        publish()
        records = expected_members(load_inputs(OUTPUTS), check)
        source = Path(os.environ["OAK"])/SOURCE_RELATIVE
        check()
        if args.mode == "inventory":
            document = inventory(source, records, report, check, publish)
            check()
            write_json(output/"inventory.json", document)
            report.update(inventory_file="inventory.json", total_member_bytes=document["total_bytes"],
                pilot_forecast=document["pilot"], total_members=document["member_count"])
        else:
            inventory_path = args.inventory.resolve()
            if (inventory_path.name != "inventory.json" or OUTPUTS/"runs/role_handoffs" not in inventory_path.parents):
                raise HandoffFailure("exact_episode_inventory_path_required")
            document = json.loads(read_bounded(inventory_path, 32*1024**2))
            terminal = json.loads(read_bounded(inventory_path.parent/REPORT_NAME, 1024**2))
            selected = bind_inventory(document, terminal, records, source)
            forecast = sum(row["bytes"] for row in selected)
            if type(args.max_output_bytes) is not int or not 0 < forecast <= args.max_output_bytes:
                raise HandoffFailure("pilot_forecast_exceeds_explicit_copy_byte_bound")
            private = args.private_dir.resolve()
            if OUTPUTS/"cache/private/role_handoffs" not in private.parents or private.exists():
                raise FileExistsError("fresh_UNRELEASED_private_pilot_directory_required")
            report.update(inventory_source=str(inventory_path), private_unreleased_output=str(private),
                expected_pilot_members=740, expected_pilot_bytes=forecast, maximum_output_bytes=args.max_output_bytes)
            private.mkdir(parents=True, mode=0o700)
            check()
            with zipfile.ZipFile(source/"MDTB_sub-08.zip") as archive:
                report["zip_directories_read"] += 1
                copy_members(archive, selected, private, report, check, publish)
            if report["members_copied"] != 740 or report["bytes_written"] != forecast:
                raise HandoffFailure("incomplete_selected_pilot")
        check()
        complete = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            complete = False
            report["error"] = "TimeoutError: whole_handoff_operation_deadline"
        report.update(status=args.mode+"_complete" if complete else "technical_handoff_failure",
            operation_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        publish()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_handoff_failure", operation_complete=False,
                error="TimeoutError: deadline_during_terminal_publication", finished_utc=datetime.now(timezone.utc).isoformat())
            publish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("inventory", "pilot"):
        command = commands.add_parser(mode)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--max-seconds", type=int, required=True)
        if mode == "pilot":
            command.add_argument("--inventory", type=Path, required=True)
            command.add_argument("--private-dir", type=Path, required=True)
            command.add_argument("--max-output-bytes", type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("mode", "status", "operation_complete", "members_copied",
        "members_inventoried", "bytes_written", "elapsed_seconds", "error") if key in result}, indent=2))
    return 0 if result["operation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
