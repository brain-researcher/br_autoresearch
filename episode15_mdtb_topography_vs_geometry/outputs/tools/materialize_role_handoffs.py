#!/usr/bin/env python3
"""Build36 unreleased ordinary role ZIPs, preserving literal source member bytes.

Trusted copying reads audit bytes. It does not decode arrays, open the scientific
audit, activate readers, or establish permissions against the same UID.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import stat
import threading
import time
import zipfile
import zlib

import prepare_role_handoffs as preparation


OUTPUTS = Path(__file__).resolve().parents[1]
INVENTORY_RELATIVE = "runs/role_handoffs/inventory001/inventory.json"
PILOT_RELATIVE = "runs/role_handoffs/sub08_TaskA_pilot001/handoff_report.json"
PILOT_PRIVATE = "cache/private/role_handoffs/sub08_TaskA_pilot001"
STORAGE_CAP = 64*1024**3
WORKERS = 4
MAX_SECONDS = 3300
CHUNK = 1024**2
ROLES = tuple(preparation.HANDOFF_COUNTS)


def utc():
    return datetime.now(timezone.utc).isoformat()


def container_bound(rows):
    # ZIP_STORED payload, both filenames, generous ZIP64/local/central overhead.
    return sum(row["bytes"]+256+2*len(row["member"].encode("utf-8")) for row in rows)+1024


def load_manifest():
    records = preparation.expected_members(preparation.load_inputs(OUTPUTS))
    path = OUTPUTS/INVENTORY_RELATIVE
    document = json.loads(preparation.read_bounded(path, 32*1024**2))
    terminal = json.loads(preparation.read_bounded(path.parent/preparation.REPORT_NAME, 1024**2))
    source = Path(os.environ["OAK"])/preparation.SOURCE_RELATIVE
    selected = preparation.bind_inventory(document, terminal, records, source)
    rows = document["members"]
    if any(type(row[key]) is not int or row[key] < 0 for row in rows
           for key in ("bytes", "compressed_bytes", "zip_crc32", "zip_compression")):
        raise preparation.HandoffFailure("invalid_full_member_metadata")
    if any(row["bytes"] == 0 or row["zip_crc32"] > 0xffffffff
           or row["zip_compression"] not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for row in rows):
        raise preparation.HandoffFailure("unsupported_full_member_metadata")
    sums = {role: sum(row["bytes"] for row in rows if row["handoff"] == role) for role in ROLES}
    if sums != document["bytes_by_handoff"] or sum(sums.values()) != document["total_bytes"]:
        raise preparation.HandoffFailure("inventory_size_sums_changed")
    pilot = json.loads(preparation.read_bounded(OUTPUTS/PILOT_RELATIVE, 1024**2))
    pilot_bytes = sum(row["bytes"] for row in selected)
    pilot_root = OUTPUTS/PILOT_PRIVATE
    if (pilot.get("status") != "pilot_complete" or pilot.get("mode") != "pilot"
            or pilot.get("operation_complete") is not True or pilot.get("members_copied") != 740
            or any(pilot.get(key) != pilot_bytes for key in ("bytes_written", "destination_bytes_verified", "expected_pilot_bytes"))
            or pilot.get("inventory_source") != str(path) or pilot.get("private_unreleased_output") != str(pilot_root)
            or pilot.get("Task_B_streams_opened") != 0 or pilot.get("audit_member_streams_opened") != 0):
        raise preparation.HandoffFailure("completed_fixed_sub08_pilot_required")
    groups = {(row["participant_id"], row["handoff"]) for row in rows}
    forecast = pilot_bytes+sum(container_bound([row for row in rows if (row["participant_id"], row["handoff"]) == group])
                               for group in groups)
    if len(groups) != 36 or forecast > STORAGE_CAP:
        raise preparation.HandoffFailure("selected_36_containers_and64GiB_preparation_cap_required")
    return rows, source, pilot_root, {"payload_bytes": sum(sums.values()), "bytes_by_handoff": sums,
        "retained_pilot_bytes": pilot_bytes, "conservative_total_storage_bytes": forecast,
        "storage_cap_bytes": STORAGE_CAP}


def balanced_packs(rows):
    sizes = Counter()
    for row in rows:
        sizes[row["participant_id"]] += row["bytes"]
    packs, totals = [[] for _ in range(WORKERS)], [0]*WORKERS
    for pid in sorted(sizes, key=lambda person: (-sizes[person], person)):
        index = min(range(WORKERS), key=lambda value: (totals[value], value))
        packs[index].append(pid)
        totals[index] += sizes[pid]
    return packs, totals


def role_counter():
    return {"source_open_attempts": 0, "source_streams_opened": 0, "source_bytes_read": 0,
        "pilot_open_attempts": 0, "pilot_streams_opened": 0, "pilot_bytes_read": 0,
        "first_source_access_attempt_utc": None, "first_source_byte_utc": None,
        "first_pilot_access_attempt_utc": None, "first_pilot_byte_utc": None,
        "copied_members": 0, "acknowledged_payload_bytes_written": 0, "closed_container_bytes_verified": 0}


def worker_report(index, people):
    return {"worker_index": index, "participants": people, "status": "started", "complete": False,
        "started_utc": utc(), "zip_directories_read": 0, "containers": [], "active_partial": None,
        "active_member": None, "roles": {role: role_counter() for role in ROLES},
        "raw_audit_Task_A_bytes_read": 0, "raw_audit_anatomy_bytes_read": 0, "raw_audit_Task_B_bytes_read": 0,
        "first_raw_audit_Task_A_byte_utc": None, "first_raw_audit_anatomy_byte_utc": None,
        "first_raw_audit_Task_B_byte_utc": None, "image_arrays_decoded": False,
        "scientific_audit_opened": False, "candidate_scoring_authorized": False, "reader_activation": False}


def read_payload(reader, row, counter, report, kind, check):
    read, crc = 0, 0
    while True:
        check()
        chunk = reader.read(CHUNK)
        if not chunk:
            break
        if counter["first_"+kind+"_byte_utc"] is None:
            counter["first_"+kind+"_byte_utc"] = utc()
        counter[kind+"_bytes_read"] += len(chunk)
        read += len(chunk)
        if row["handoff"] != "development":
            audit_kind = "anatomy" if row["kind"] == "anatomy" else "Task_"+row["kind"]
            report["raw_audit_"+audit_kind+"_bytes_read"] += len(chunk)
            key = "first_raw_audit_"+audit_kind+"_byte_utc"
            if report[key] is None:
                report[key] = utc()
        if read > row["bytes"]:
            raise preparation.HandoffFailure("member_exceeds_frozen_length")
        crc = zlib.crc32(chunk, crc)
        check()
        yield chunk
    if read != row["bytes"] or crc != row["zip_crc32"]:
        raise preparation.HandoffFailure("source_or_reused_pilot_size_CRC_changed")


def open_member(archive, row, pilot_root, counter):
    reuse = row["participant_id"] == "sub-08" and row["kind"] in ("A", "anatomy")
    kind = "pilot" if reuse else "source"
    counter[kind+"_open_attempts"] += 1
    if counter["first_"+kind+"_access_attempt_utc"] is None:
        counter["first_"+kind+"_access_attempt_utc"] = utc()
    if reuse:
        path = pilot_root/"development"/row["member"]
        if path.is_symlink() or pilot_root.resolve() not in path.resolve().parents:
            raise preparation.HandoffFailure("pilot_member_escaped_fixed_private_tree")
        reader = path.open("rb")
        info = os.fstat(reader.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != row["bytes"]:
            reader.close()
            raise preparation.HandoffFailure("pilot_member_not_original_sized_regular_file")
    else:
        reader = archive.open(row["member"])
    counter[kind+"_streams_opened"] += 1
    return reader, kind


def verify_container(path, rows, counter, check):
    with zipfile.ZipFile(path) as archive:
        infos = preparation.directory(archive)
        if list(infos) != [row["member"] for row in rows]:
            raise preparation.HandoffFailure("container_member_order_or_scope_changed")
        for row in rows:
            check()
            info = infos[row["member"]]
            if (info.file_size != row["bytes"] or info.CRC != row["zip_crc32"]
                    or info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1):
                raise preparation.HandoffFailure("closed_container_metadata_mismatch")
            size, crc = 0, 0
            with archive.open(info) as reader:
                while True:
                    check()
                    chunk = reader.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    crc = zlib.crc32(chunk, crc)
                    counter["closed_container_bytes_verified"] += len(chunk)
            if size != row["bytes"] or crc != row["zip_crc32"]:
                raise preparation.HandoffFailure("closed_container_payload_mismatch")


def copy_participant(pid, rows, source, private, pilot_root, report, check, publish):
    with zipfile.ZipFile(source/f"MDTB_{pid}.zip") as archive:
        infos = preparation.directory(archive)
        report["zip_directories_read"] += 1
        for row in rows:
            check()
            if row["member"] not in infos or preparation.member_metadata(infos[row["member"]]) != {
                    key: row[key] for key in ("bytes", "compressed_bytes", "zip_crc32", "zip_compression")}:
                raise preparation.HandoffFailure("source_directory_changed_since_inventory")
        for role in ROLES:
            selected = [row for row in rows if row["handoff"] == role]
            if not selected:
                continue
            check()
            target = private/role/f"{pid}.zip"
            partial = target.with_name(target.name+".partial")
            if target.exists() or partial.exists():
                raise FileExistsError("fresh_container_required")
            report["active_partial"] = str(partial)
            counter = report["roles"][role]
            maximum_container_bytes = container_bound(selected)
            with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
                os.chmod(partial, 0o600)
                for row in selected:
                    check()
                    report["active_member"] = row["member"]
                    info = zipfile.ZipInfo(row["member"])
                    info.compress_type = zipfile.ZIP_STORED
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    reader, kind = open_member(archive, row, pilot_root, counter)
                    with reader, output.open(info, "w", force_zip64=True) as writer:
                        for chunk in read_payload(reader, row, counter, report, kind, check):
                            written = writer.write(chunk)
                            counter["acknowledged_payload_bytes_written"] += written
                            if written != len(chunk):
                                raise OSError("short_container_member_write")
                    counter["copied_members"] += 1
                    report["active_member"] = None
                    if output.fp.tell() > maximum_container_bytes:
                        raise preparation.HandoffFailure("container_storage_bound_exceeded")
            check()
            size = partial.stat().st_size
            if size > maximum_container_bytes:
                raise preparation.HandoffFailure("closed_container_storage_bound_exceeded")
            verify_container(partial, selected, counter, check)
            check()
            os.link(partial, target)
            partial.unlink()
            report["containers"].append({"participant_id": pid, "handoff": role,
                "path": str(target), "members": len(selected), "payload_bytes": sum(row["bytes"] for row in selected),
                "container_bytes": size, "closed_original_CRC_verified": True})
            report["active_partial"] = None
            publish()


def copy_pack(index, people, rows, source, private, pilot_root, output, deadline, stop):
    report, began = worker_report(index, people), time.monotonic()
    def check():
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("materialization_cancelled_or_deadline")
    def publish():
        report["elapsed_seconds"] = time.monotonic()-began
        preparation.write_json(output/f"worker{index}.json", report)
    try:
        publish()
        for pid in people:
            check()
            copy_participant(pid, [row for row in rows if row["participant_id"] == pid], source,
                             private, pilot_root, report, check, publish)
        check()
        report.update(status="complete", complete=True)
    except Exception as error:
        stop.set()
        report.update(status="technical_copy_failure", complete=False, error=f"{type(error).__name__}: {error}")
    finally:
        if report["active_partial"] is not None:
            try:
                path = Path(report["active_partial"])
                report["active_partial_container_bytes_on_disk"] = path.stat().st_size if path.exists() else 0
            except OSError as error:
                report["active_partial_size_error"] = f"{type(error).__name__}: {error}"
        report["finished_utc"] = utc()
        publish()
        if report["complete"] and (stop.is_set() or time.monotonic() >= deadline):
            report.update(status="technical_copy_failure", complete=False,
                error="TimeoutError: worker_final_publication_exceeded_deadline", finished_utc=utc())
            stop.set()
            publish()
    return report


def run(args):
    if ("-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_CPUS_PER_TASK") != str(WORKERS)
            or any(os.environ.get(key) != "1" for key in preparation.THREAD_ENV)):
        raise RuntimeError("actual_four_CPU_allocation_and_single_numeric_threads_required")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("invalid_whole_materialization_deadline")
    output, private = args.output_dir.resolve(), args.private_dir.resolve()
    if (OUTPUTS/"runs/role_handoffs" not in output.parents or output.exists()
            or OUTPUTS/"cache/private/role_handoffs" not in private.parents or private.exists()):
        raise FileExistsError("fresh_public_and_UNRELEASED_private_attempts_required")
    output.mkdir(parents=True)
    began = time.monotonic()
    deadline = began+args.max_seconds
    stop = threading.Event()
    report = {"analysis_id": "role_handoff_preparation", "mode": "full_literal_materialization",
        "status": "started", "materialization_complete": False, "started_utc": utc(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "workers": WORKERS, "worker_results": [],
        "representation": "ZIP_STORED_one_participant_per_role_exact_original_paths_and_bytes",
        "raw_audit_copy_access_is_distinct_from_scientific_evaluator_opening": True,
        "scientific_audit_opened": False, "image_arrays_decoded": False, "participant_values_publicly_exported": False,
        "reader_isolation_established": False, "reader_activation": False, "candidate_scoring_authorized": False,
        "scientific_margins_activated": False, "same_UID_trusted_copy_boundary_only": True,
        "partial_outputs_are_unreleased": True, "automatic_retry": False,
        "later_search_scratch_policy_unchanged": True, "workers_joined": False}
    def check():
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("whole_materialization_deadline_or_interruption")
    def publish():
        report["elapsed_seconds"] = time.monotonic()-began
        preparation.write_json(output/"materialization_report.json", report)
    def interrupted(sig, _):
        report["termination_signal"] = sig
        stop.set()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    executor = None
    futures = {}
    try:
        publish()
        rows, source, pilot_root, forecast = load_manifest()
        check()
        report["forecast"] = forecast
        packs, totals = balanced_packs(rows)
        report.update(participant_packs=packs, pack_payload_bytes=totals,
                      private_unreleased_output=str(private), preserved_pilot=str(pilot_root))
        private.mkdir(parents=True, mode=0o700)
        for role in ROLES:
            (private/role).mkdir(mode=0o700)
        publish()
        check()
        executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="role-copy")
        for index, people in enumerate(packs):
            futures[executor.submit(copy_pack, index, people, rows, source, private, pilot_root,
                                    output, deadline, stop)] = index
        for future in as_completed(futures, timeout=max(0.001, deadline-time.monotonic())):
            result = future.result()
            if not result["complete"]:
                stop.set()
        check()
    except Exception as error:
        stop.set()
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if executor is not None:
            # Chunk checks cooperate with cancellation; external process bound
            # remains authoritative for a filesystem call that cannot return.
            executor.shutdown(wait=True, cancel_futures=True)
            report["workers_joined"] = True
        for future, index in sorted(futures.items(), key=lambda item: item[1]):
            if future.cancelled():
                report["worker_results"].append({"worker_index": index, "complete": False, "status": "cancelled_before_start"})
            else:
                try:
                    report["worker_results"].append(future.result())
                except Exception as error:
                    report["worker_results"].append({"worker_index": index, "complete": False,
                        "status": "unreported_worker_failure", "error": f"{type(error).__name__}: {error}"})
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        workers = report["worker_results"]
        complete = (not stop.is_set() and time.monotonic() < deadline and len(workers) == WORKERS
                    and all(worker["complete"] for worker in workers))
        containers = [item for worker in workers for item in worker.get("containers", [])]
        if complete:
            counts = Counter()
            for item in containers:
                counts[item["handoff"]] += item["members"]
            complete = (len(containers) == 36 and dict(counts) == preparation.HANDOFF_COUNTS
                and sum(item["payload_bytes"] for item in containers) == forecast["payload_bytes"]
                and sum(item["container_bytes"] for item in containers)+forecast["retained_pilot_bytes"] <= STORAGE_CAP)
            if not complete:
                report["error"] = "ValueError: full_member_or_storage_totals_incomplete"
        report.update(status="complete_unreleased_role_materialization" if complete else "technical_materialization_failure",
            materialization_complete=complete, finished_utc=utc(), completed_containers=len(containers))
        publish()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_materialization_failure", materialization_complete=False,
                error="TimeoutError: deadline_during_terminal_publication", finished_utc=utc())
            publish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "materialization_complete", "completed_containers",
        "workers_joined", "elapsed_seconds", "error") if key in result}, indent=2))
    return 0 if result["materialization_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
