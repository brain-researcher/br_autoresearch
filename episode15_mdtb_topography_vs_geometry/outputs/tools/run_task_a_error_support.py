#!/usr/bin/env python3
"""EP15 fixed descriptive Task-A partition diagnostic; no scientific pass rule.

Root must select this assessment and release the twelve development inputs
before execution. The four-field manifest uses run_development_point_panel's
explicit bindings/support format, but member_rows_path contains ONLY the full
Task-A design for the twelve development people and the condition-multiplicity
object has ONLY ses-s1. There are no source defaults or directory searches.

The existing released development ZIPs are sufficient: central-directory
names may be inspected to establish unique permitted Task-A/forward members.
No Task-B, instruction, other anatomy, audit or private-reliability stream is
opened, even for headers. Root owns establishing that the supplied containers
are the released development handoffs; this allowlist is procedural, not OS
isolation. All twelve plans and all selected headers precede voxel decoding.

Use equal-regressor means within runs and equal-run means in the four frozen
partitions. Spatially center each partition/condition on all 208 locations.
For all 29 conditions report voxel-mean participant covariance
S=U(P1-P3,P2-P4), N=U(P1-P2,P3-P4), plus descriptive ranges from deleting each
participant once. Preserve negative values. No p-value, calibrated interval,
equivalence threshold, reliability scalar, candidate score or independence
certificate is produced. A shared effect identical across both sessions is
invisible to these contrasts; session changes and measurement noise are not
separately identified. Task-A findings do not authenticate GLM7 Task-B errors.

One thread, an explicit absolute deadline, RSS and durable-byte cap are
required. Root's Slurm limits and enclosing TERM/KILL timeout bound native
operations/imports. Images are streamed with the qualified 32-MiB member bound;
none are extracted. No partial scientific summary is published. A handled
failure retains operational status and any incomplete publication file, with
no retry or participant replacement. Only new attempt directories are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time
import zipfile

from run_development_point_panel import (
    ExecutionRefusal, OUTPUTS, REPORT_BYTES, _compute, _encoded, _path,
    _require, _write_new, load_released_inputs,
)


def compile_task_a_plan(rows, summaries, *, check):
    """Validate every released A row before any container or sample is opened."""
    from development_handoff_reader import DEVELOPMENT_IDS, INTEGER_FIELDS, _integer
    from local_functional_source import TASK_A_CONDITIONS
    from pilot_task_a_mapping import select_plan, TOTAL_A_REGRESSORS

    _require(set(summaries) == {"ses-s1"})
    summary = summaries["ses-s1"]
    _require(tuple(sorted(summary)) == TASK_A_CONDITIONS)
    grouped = {pid: [] for pid in DEVELOPMENT_IDS}
    for number, original in enumerate(rows):
        check()
        _require(number < len(DEVELOPMENT_IDS) * TOTAL_A_REGRESSORS)
        row = dict(original)
        _require(row.get("participant_id") in grouped and row.get("session") == "ses-s1"
            and row.get("participant_role") == "development"
            and row.get("planned_handoff") == "development")
        for field in INTEGER_FIELDS:
            row[field] = _integer(row[field])
        instruction = bool(row["instruction"])
        _require(row["cond_name"] in summary or instruction)
        run = row["run"]
        _require(1 <= run <= 16)
        count = 0 if instruction else summary[row["cond_name"]]["regressors_per_run"][run - 1]
        _require(row["endpoint_use"] == ("instruction_excluded" if instruction else "Task_A_calibration")
            and row["condition_map_included"] == int(not instruction)
            and row["regressors_in_condition_run"] == count
            and row["within_run_weight_numerator"] == int(not instruction)
            and row["within_run_weight_denominator"] == (count if count else 1))
        grouped[row["participant_id"]].append(row)
    plans, members = {}, {}
    for pid, records in grouped.items():
        check()
        _require(len(records) == TOTAL_A_REGRESSORS)
        counts = Counter(row["beta_member"] for row in records)
        plans[pid] = tuple(select_plan(records, pid, summary, counts))
        members[pid] = frozenset(counts)
    return plans, members


def partition_maps(plan, *, sample, check):
    """Qualified arithmetic, retaining four partition means instead of halves."""
    import numpy as np
    from local_functional_source import TASK_A_CONDITIONS

    lookup = {condition: number for number, condition in enumerate(TASK_A_CONDITIONS)}
    expected = Counter((row["run"], row["cond_name"]) for row in plan)
    _require(set(expected) == {(run, condition) for run in range(1, 17) for condition in lookup})
    sums = np.zeros((16, 29, 208), dtype=np.float64)
    counts = np.zeros((16, 29), dtype=np.int64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for row in plan:
            check()
            raw = np.asarray(sample(row))
            _require(raw.shape == (208,) and raw.dtype.kind in "fiu" and np.isfinite(raw).all())
            slot = row["run"] - 1, lookup[row["cond_name"]]
            sums[slot] += np.asarray(raw, dtype=np.float64)
            counts[slot] += 1
        for (run, condition), count in expected.items():
            _require(counts[run - 1, lookup[condition]] == count)
        result = (sums / counts[..., None]).reshape(4, 4, 29, 208).mean(axis=1)
    _require(np.isfinite(result).all())
    return result


def read_task_a_partitions(*, release_bindings, member_rows,
                           condition_multiplicity_by_session, common_support, check):
    """Explicit released-view reader. No Task-B stream is opened."""
    import numpy as np
    from build_anatomical_support import GRID_KEYS, selected_sform
    from development_handoff_reader import (
        DEVELOPMENT_IDS, DevelopmentReleaseBinding, _header, _native_coordinates,
    )
    from pilot_task_a_mapping import ExactMemberArchive, sample_beta, validate_beta_reference, validate_support

    bindings = tuple(release_bindings)
    _require(all(type(item) is DevelopmentReleaseBinding for item in bindings)
        and tuple(item.participant_id for item in bindings) == DEVELOPMENT_IDS)
    bindings = tuple(DevelopmentReleaseBinding(item.participant_id, _path(item.archive_path),
        deepcopy(item.anatomy_headers), deepcopy(item.task_a_header)) for item in bindings)
    _require(len({item.archive_path.resolve() for item in bindings}) == len(DEVELOPMENT_IDS)
        and all(item.archive_path.name == item.participant_id + ".zip" for item in bindings))
    plans, members = compile_task_a_plan(member_rows, condition_multiplicity_by_session, check=check)
    support = {name: np.array(value, copy=True) for name, value in common_support.items()}
    reference = bindings[0].anatomy_headers["space-SUIT_xfm.nii"]
    mapped = 0
    with ExitStack() as stack:
        prepared = {}
        for binding in bindings:
            check()
            pid, headers = binding.participant_id, binding.anatomy_headers
            _require(all(headers["space-SUIT_xfm.nii"][key] == reference[key] for key in GRID_KEYS))
            validate_beta_reference(binding.task_a_header, headers["desc-cereb_mask.nii"])
            forward_member = f"{pid}/anat/{pid}_space-SUIT_xfm.nii"
            archive = stack.enter_context(zipfile.ZipFile(binding.archive_path))
            required = members[pid] | {forward_member}
            # Closed development-B names are compared only for membership;
            # neither their metadata records nor streams are inspected further.
            counts = Counter(info.filename for info in archive.infolist() if info.filename in required)
            _require(set(counts) == required and all(count == 1 for count in counts.values()))
            checked = ExactMemberArchive(archive, {forward_member} | {row["beta_member"] for row in plans[pid]})
            forward_header = _header(checked, forward_member)
            _require(all(forward_header[key] == headers["space-SUIT_xfm.nii"][key]
                for key in GRID_KEYS + ("datatype", "voxel_offset")))
            selected_sform(forward_header)
            beta_headers = {}
            for row in plans[pid]:
                check()
                header = _header(checked, row["beta_member"])
                validate_beta_reference(header, headers["desc-cereb_mask.nii"])
                _require(all(header[key] == binding.task_a_header[key]
                    for key in GRID_KEYS + ("datatype", "voxel_offset")))
                beta_headers[row["beta_member"]] = header
            prepared[pid] = checked, forward_member, forward_header, headers["desc-cereb_mask.nii"], beta_headers
        # Historical anatomy records do not include units. Validate the fixed
        # support against the actual first forward header, after all live
        # headers match the recorded grid and their units are qualified.
        # Neither guessed units nor altered historical metadata are permitted.
        indices = validate_support(support, prepared[DEVELOPMENT_IDS[0]][2])
        # No voxel decoding occurred before every participant's header checks.
        partitions = []
        for pid in DEVELOPMENT_IDS:
            check()
            checked, forward_member, forward_header, mask_header, beta_headers = prepared[pid]
            native_world = _native_coordinates(checked, forward_member, forward_header, indices)

            def sample(row):
                nonlocal mapped
                values, _ = sample_beta(checked, row["beta_member"], beta_headers[row["beta_member"]],
                                        mask_header, native_world)
                mapped += 1
                return values

            partitions.append(partition_maps(plans[pid], sample=sample, check=check))
        check()
        report = dict(participants=12, registered_runs=16, conditions=29, support_locations=208,
            mapped_Task_A_beta_members=mapped, forward_fields=12,
            selected_stream_bytes=sum(values[0].bytes_read for values in prepared.values()),
            all_selected_headers_checked_before_voxel_decoding=True,
            Task_B_streams_opened=0, instruction_streams_opened=0, private_reliability_read=False,
            arithmetic="equal_regressors_within_run_then_equal_runs_in_four_fixed_partitions")
        return np.stack(partitions), report


def summarize_partitions(partitions, *, check):
    """Complete signed descriptive covariance and delete-one-person ranges."""
    import numpy as np
    from development_lopo import DEVELOPMENT_IDS
    from local_functional_source import TASK_A_CONDITIONS
    from participant_cross_replicate_covariance import cross_replicate_covariance as covariance

    raw = np.asarray(partitions)
    _require(raw.shape == (12, 4, 29, 208) and raw.dtype.kind in "fiu" and np.isfinite(raw).all())
    check()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        centered = np.array(raw, dtype=np.float64, copy=True)
        centered -= centered.mean(axis=-1, keepdims=True)

        def scores(values):
            s = covariance(values[:, 0] - values[:, 2], values[:, 1] - values[:, 3]).mean(axis=-1)
            n = covariance(values[:, 0] - values[:, 1], values[:, 2] - values[:, 3]).mean(axis=-1)
            return np.stack((s, n))

        point = scores(centered)
        deleted = []
        for person in range(12):
            check()
            deleted.append(scores(centered[np.arange(12) != person]))
        deleted = np.stack(deleted)
        low, high = deleted.min(axis=0), deleted.max(axis=0)
    _require(np.isfinite(point).all() and np.isfinite(deleted).all())
    return dict(participant_ids=DEVELOPMENT_IDS, partition_runs=[list(range(k, k + 4)) for k in (1, 5, 9, 13)],
        quantities={"S": "voxel_mean_U(P1-P3,P2-P4)", "N": "voxel_mean_U(P1-P2,P3-P4)"},
        condition_summaries=[dict(condition=name, S=float(point[0, c]), N=float(point[1, c]),
            S_delete_one_range=[float(low[0, c]), float(high[0, c])],
            N_delete_one_range=[float(low[1, c]), float(high[1, c])])
            for c, name in enumerate(TASK_A_CONDITIONS)],
        covariance_normalization="participant_centered_cross_product_divided_by_n_minus_1",
        spatial_centering="each_partition_condition_over_all_208_locations",
        descriptive_deletions=12, ranges_are_confidence_intervals=False,
        independence_established=False, scientific_pass_decision=False,
        limitations=["shared_effect_identical_across_sessions_is_invisible",
                     "session_signal_changes_and_measurement_noise_are_not_separated",
                     "Task_A_does_not_authenticate_Task_B_error_structure"])


def run_released_diagnostic(manifest_path, *, check, stage):
    _compute()
    stage("released_Task_A_metadata")
    inputs = load_released_inputs(manifest_path, check=check)
    stage("Task_A_mapping")
    partitions, mapping = read_task_a_partitions(**inputs, check=check)
    stage("signed_descriptive_diagnostic")
    summary = summarize_partitions(partitions, check=check)
    check()
    return summary, mapping


def execute(*, manifest_path, output_dir, deadline_epoch, rss_limit_kib, durable_byte_cap):
    _compute()
    _require(math.isfinite(deadline_epoch) and deadline_epoch > time.time()
        and type(rss_limit_kib) is int and rss_limit_kib > 0
        and type(durable_byte_cap) is int and durable_byte_cap > REPORT_BYTES)
    manifest_path, output_dir = _path(manifest_path), _path(output_dir)
    _require(OUTPUTS / "runs" in output_dir.resolve().parents)
    _require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_dir.chmod(0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(status="started", phase="startup", job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path), diagnostic_published=False, scratch_bytes=0,
        Task_B_values_read=False, audit_values_read=False, private_reliability_read=False,
        independence_established=False, scientific_pass_decision=False,
        separation="procedural_same_account_not_technically_enforced",
        caps=dict(deadline_epoch=deadline_epoch, rss_limit_kib=rss_limit_kib,
                  durable_byte_cap=durable_byte_cap))
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

    def check():
        _require(time.time() < deadline_epoch
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib)

    def stage(name):
        report["phase"] = name
        check()

    def stop(_signal, _frame):
        raise ExecutionRefusal()

    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline_epoch - time.time()))
        _write_new(output_dir / "settings.json", _encoded(report, REPORT_BYTES // 2))
        summary, mapping = run_released_diagnostic(manifest_path, check=check, stage=stage)
        stage("complete_diagnostic_persistence")
        payload = _encoded(summary, durable_byte_cap - REPORT_BYTES)
        check()
        partial, final = output_dir / "diagnostic.json.partial", output_dir / "diagnostic.json"
        _write_new(partial, payload)
        check()
        os.link(partial, final)
        report["diagnostic_published"] = True
        try:
            partial.unlink()
            report["partial_cleanup_complete"] = True
        except OSError:
            report["partial_cleanup_complete"] = False
        report.update(status="completed_Task_A_descriptive_diagnostic", mapping=mapping,
                      diagnostic_bytes=len(payload))
    except BaseException:
        report.update(status="stopped", error_code=report["phase"] + "_failed")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        _write_new(output_dir / "operational.json", _encoded(report, REPORT_BYTES // 2))
    return 0 if report["status"] == "completed_Task_A_descriptive_diagnostic" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    parser.add_argument("--durable-byte-cap", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        code = execute(manifest_path=args.manifest, output_dir=args.output_dir,
            deadline_epoch=args.deadline_epoch, rss_limit_kib=args.rss_limit_kib,
            durable_byte_cap=args.durable_byte_cap)
    except BaseException:
        code = 1
    print("completed_Task_A_diagnostic" if code == 0 else "stopped_Task_A_diagnostic", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
