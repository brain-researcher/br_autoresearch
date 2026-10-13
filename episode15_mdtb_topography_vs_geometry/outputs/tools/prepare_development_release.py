#!/usr/bin/env python3
"""Build explicit EP15 development views from completed metadata, without maps.

Trusted coordinator preparation, selected and recorded before invocation.
Inputs are exact explicit paths: the completed full001 materialization report,
attempt001/anatomical_headers.json, metadata_join001/source_metadata.json,
condition_partition_plan001/role_member_plan.tsv, and unchanged common_support
NPZ. No path discovery, role assignment, participant substitution or defaults.

The supplied JSON/TSV records may describe all24 people/36 containers. Select
only the fixed12 development identities. Only their recorded development ZIP
paths are stat'ed; never list or open a container, original archive, audit path,
private reliability artifact or array. The common-support file is stat'ed,
not decoded. Existing closed-container CRC verification is reused, not rerun.

Publish two four-field manifests: the complete development panel view and a
Task-A-only diagnostic view. Both reuse existing twelve containers and support;
only header metadata and filtered member tables are written. Task-B rows in
the panel table are names/design metadata, never response values. A staging
directory is renamed only after all four view files are complete. On failure
retain the incomplete attempt, never modify an earlier attempt or retry.

Publication does not select a scientific diagnostic, grant candidate scoring,
establish independent errors, satisfy the cache firewall or start execution.
Root owns release recording and each subsequent scientific/resource admission.
Separation remains procedural within the same account. Run on compute with
explicit absolute deadline/RSS/durable caps and an enclosing native timeout.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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

from run_development_point_panel import (
    ExecutionRefusal, OUTPUTS, MANIFEST_BYTES, MEMBER_BYTES, SUPPORT_BYTES,
    REPORT_BYTES, _bounded, _compute, _encoded, _path, _require, _write_new,
)


def _selected_people(document):
    from development_lopo import DEVELOPMENT_IDS
    selected = {}
    for row in document["participants"]:
        pid = row["participant_id"]
        if pid in DEVELOPMENT_IDS:
            _require(pid not in selected)
            selected[pid] = row
    _require(set(selected) == set(DEVELOPMENT_IDS))
    return selected


def select_development_metadata(materialization, anatomy, metadata, member_rows, *, check):
    """Pure metadata selection; never dereference recorded nondevelopment paths."""
    from development_handoff_reader import DEVELOPMENT_IDS, _compile_plan
    from join_source_metadata import EXPECTED_BETAS
    from prepare_role_handoffs import ANATOMY
    from run_task_a_error_support import compile_task_a_plan

    _require(materialization.get("status") == "complete_unreleased_role_materialization"
        and materialization.get("materialization_complete") is True
        and materialization.get("workers_joined") is True
        and materialization.get("completed_containers") == 36
        and materialization.get("image_arrays_decoded") is False
        and materialization.get("scientific_audit_opened") is False)
    workers = materialization["worker_results"]
    _require(len(workers) == 4 and all(worker.get("complete") is True for worker in workers))
    development_root = _path(materialization["private_unreleased_output"]) / "development"
    containers = {}
    for worker in workers:
        check()
        for item in worker["containers"]:
            if item["handoff"] != "development":
                continue
            pid = item["participant_id"]
            _require(pid in DEVELOPMENT_IDS and pid not in containers)
            path = _path(item["path"])
            _require(path == development_root / (pid + ".zip")
                and item.get("closed_original_CRC_verified") is True
                and item.get("members") == sum(EXPECTED_BETAS.values()) + len(ANATOMY)
                and type(item.get("container_bytes")) is int and item["container_bytes"] > 0
                and type(item.get("payload_bytes")) is int and 0 < item["payload_bytes"] <= item["container_bytes"])
            containers[pid] = dict(path=path, bytes=item["container_bytes"])
    _require(set(containers) == set(DEVELOPMENT_IDS))
    _require(anatomy.get("forward_transform_headers_share_grid") is True
        and anatomy.get("all_voxel_payloads_closed") is True
        and metadata.get("metadata_join_complete") is True
        and metadata.get("all_selected_Task_A_headers_equal_native_cerebellar_mask_grid") is True
        and metadata.get("Task_B_image_streams_opened") == 0)
    anatomical, functional = _selected_people(anatomy), _selected_people(metadata)
    bindings = []
    for pid in DEVELOPMENT_IDS:
        check()
        headers, person = anatomical[pid]["headers"], functional[pid]
        _require(set(headers) == set(ANATOMY)
            and person["beta_members_joined"] == EXPECTED_BETAS
            and person["selected_Task_A_header_equals_prior_native_cerebellar_mask_grid"] is True)
        bindings.append(dict(participant_id=pid, archive_path=str(containers[pid]["path"]),
            anatomy_headers=headers, task_a_header=person["selected_Task_A_header"]["header"]))
    rows = []
    for index, row in enumerate(member_rows):
        check()
        _require(index < 24 * sum(EXPECTED_BETAS.values()) and None not in row
            and all(value is not None for value in row.values()))
        if row["participant_id"] in DEVELOPMENT_IDS:
            _require(row["participant_role"] == "development" and row["planned_handoff"] == "development")
            rows.append(dict(row))
        else:
            # Never silently discard an unexpected identity labeled development.
            _require(row["participant_role"] != "development" and row["planned_handoff"] != "development")
    summaries = metadata["condition_multiplicity_by_session"]
    _compile_plan(rows, summaries, check)
    a_rows = [row for row in rows if row["session"] == "ses-s1"]
    a_summary = {"ses-s1": summaries["ses-s1"]}
    compile_task_a_plan(a_rows, a_summary, check=check)
    return bindings, rows, a_rows, summaries, a_summary, containers


def _table_bytes(rows):
    from build_condition_partition_plan import MEMBER_COLUMNS
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MEMBER_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    _require(len(payload) <= MEMBER_BYTES)
    return payload


def build_view_payloads(*, materialization_report, anatomical_headers, source_metadata,
                        role_member_plan, common_support, final_view, check):
    """Read only explicit metadata; stat exactly12 reported development files."""
    from build_condition_partition_plan import MEMBER_COLUMNS
    from development_lopo import DEVELOPMENT_IDS

    check()
    materialization = json.loads(_bounded(materialization_report, 8 * 1024**2))
    anatomy = json.loads(_bounded(anatomical_headers, MANIFEST_BYTES))
    metadata = json.loads(_bounded(source_metadata, MANIFEST_BYTES))
    table = csv.DictReader(io.StringIO(_bounded(role_member_plan, MEMBER_BYTES).decode("utf-8-sig")), delimiter="\t")
    _require(table.fieldnames == MEMBER_COLUMNS)
    bindings, rows, a_rows, summaries, a_summary, containers = select_development_metadata(
        materialization, anatomy, metadata, table, check=check)
    sizes = {}
    for pid in DEVELOPMENT_IDS:
        check()
        item = containers[pid]
        observed = item["path"].lstat()
        _require(stat.S_ISREG(observed.st_mode) and observed.st_size == item["bytes"])
        sizes[pid] = observed.st_size
    support_path = _path(common_support)
    support_stat = support_path.lstat()
    _require(stat.S_ISREG(support_stat.st_mode) and 0 < support_stat.st_size <= SUPPORT_BYTES)
    final_view = _path(final_view)
    panel = dict(release_bindings=bindings, member_rows_path=str(final_view / "development_members.tsv"),
        condition_multiplicity_by_session=summaries, common_support_path=str(support_path))
    diagnostic = dict(release_bindings=bindings, member_rows_path=str(final_view / "Task_A_members.tsv"),
        condition_multiplicity_by_session=a_summary, common_support_path=str(support_path))
    payloads = {
        "development_members.tsv": _table_bytes(rows),
        "Task_A_members.tsv": _table_bytes(a_rows),
        "point_panel_manifest.json": _encoded(panel, MANIFEST_BYTES),
        "Task_A_diagnostic_manifest.json": _encoded(diagnostic, MANIFEST_BYTES),
    }
    check()
    return payloads, dict(participant_ids=list(DEVELOPMENT_IDS), development_container_bytes=sizes,
        panel_member_rows=len(rows), Task_A_member_rows=len(a_rows),
        container_values_opened=False, container_directories_opened=False,
        audit_paths_stated_or_opened=False, support_values_opened=False,
        existing_materialization_CRC_verification_reused=True)


def execute(*, materialization_report, anatomical_headers, source_metadata, role_member_plan,
            common_support, output_dir, deadline_epoch, rss_limit_kib, durable_byte_cap):
    _compute()
    _require(math.isfinite(deadline_epoch) and deadline_epoch > time.time()
        and type(rss_limit_kib) is int and rss_limit_kib > 0
        and type(durable_byte_cap) is int and durable_byte_cap > REPORT_BYTES)
    paths = dict(materialization_report=_path(materialization_report), anatomical_headers=_path(anatomical_headers),
        source_metadata=_path(source_metadata), role_member_plan=_path(role_member_plan), common_support=_path(common_support))
    output_dir = _path(output_dir)
    _require(OUTPUTS / "runs" in output_dir.resolve().parents)
    _require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_dir.chmod(0o700)
    started = time.monotonic()
    report = dict(status="started", phase="metadata_preparation", job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        inputs={key: str(value) for key, value in paths.items()}, view_published=False,
        empirical_execution_started=False, independence_established=False,
        scientific_pass_decision=False, candidate_scoring_authorized=False,
        separation="procedural_same_account_not_technically_enforced",
        caps=dict(deadline_epoch=deadline_epoch, rss_limit_kib=rss_limit_kib, durable_byte_cap=durable_byte_cap))
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

    def check():
        _require(time.time() < deadline_epoch and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib)

    def stop(_signal, _frame):
        raise ExecutionRefusal()

    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline_epoch - time.time()))
        _write_new(output_dir / "settings.json", _encoded(report, REPORT_BYTES // 2))
        final = output_dir / "view"
        payloads, selection = build_view_payloads(**paths, final_view=final, check=check)
        _require(sum(map(len, payloads.values())) <= durable_byte_cap - REPORT_BYTES)
        report["phase"] = "view_persistence"
        pending = output_dir / ".pending"
        pending.mkdir(mode=0o700, exist_ok=False)
        for name, payload in payloads.items():
            check()
            _write_new(pending / name, payload)
        check()
        pending.rename(final)
        report.update(status="completed_development_metadata_views", view_published=True,
            selection=selection, view_bytes=sum(map(len, payloads.values())),
            point_panel_manifest=str(final / "point_panel_manifest.json"),
            Task_A_diagnostic_manifest=str(final / "Task_A_diagnostic_manifest.json"))
    except BaseException:
        report.update(status="stopped", error_code=report["phase"] + "_failed")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(view_published=(output_dir / "view").is_dir(), elapsed_seconds=time.monotonic() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        _write_new(output_dir / "operational.json", _encoded(report, REPORT_BYTES // 2))
    return 0 if report["status"] == "completed_development_metadata_views" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("materialization-report", "anatomical-headers", "source-metadata", "role-member-plan", "common-support", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    parser.add_argument("--durable-byte-cap", required=True, type=int)
    args = vars(parser.parse_args(argv))
    try:
        code = execute(**args)
    except BaseException:
        code = 1
    print("completed_development_metadata_views" if code == 0 else "stopped_development_metadata_views", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
