"""Read-only, bounded inspection of the five October 7 I/O-failed fragments.

Run only inside the coordinator's substantive 1-CPU/7-GiB allocation. Existing
terminal accounting, status and manifest identities precede any payload access.
This reports observations, never repairs files or emits reusable/retry coverage.
No responses, features, normalized_X, Gram, tau or OOD arrays are read. Small
training normalization vectors bind selected voxel IDs; coefficient reads use
the existing assembler's validators and at most 256 rows at a time.

The coordinator supplies its already verified allocation end epoch, captures
stdout as evidence, and chooses any next action. This helper does not poll Slurm,
submit jobs, change HDF5 locking, flush/write a fragment or authorize continuation.
An explicit subset avoids rereading previously inspected fragments; omitting
--work-items still selects all five. A clean internal timeout exits 3 with an
incomplete report, never successful coverage. Other incomplete/error exits are 2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import decoder_assembly as assembly
import participant_training_state as training

BASE = assembly.BASE
FULL_ROOT = assembly.FULL_ROOT
WORKERS = {
    "work_0032": ("sub-05", "46918887_31", "46919303"),
    "work_0033": ("sub-05", "46918887_32", "46919304"),
    "work_0047": ("sub-06", "46918887_46", "46919447"),
    "work_0061": ("sub-07", "46918155_60", "46918155"),
    "work_0062": ("sub-07", "46918887_61", "46918887"),
}
BLOCK_UNITS = 256
MAX_SECONDS = 600
MAX_JSON_BYTES = 16 * 1024**2
require = assembly.require


class InspectionTimeout(TimeoutError):
    pass


class InspectionLimit(RuntimeError):
    pass


def _read_json(path):
    require(path.stat().st_size <= MAX_JSON_BYTES, "bounded operational JSON required")
    return json.loads(path.read_text())


def _path(base, relative):
    return assembly.physical_path(base, relative)


def _spans(rows):
    require(isinstance(rows, list), "explicit interval list required")
    result = [assembly.span_from_record(row) for row in rows]
    require(len({span.key for span in result}) == len(result), "duplicate channel interval")
    return result


def validate_record(work, item, status, request, terminal, base):
    """Metadata admission only; failure never causes a replacement attempt."""
    require(work in WORKERS, "only the five named I/O-failed work items are permitted")
    participant, array_id, raw_id = WORKERS[work]
    require(terminal.get("event") == "terminal_charge" and terminal.get("job_id") == array_id
            and terminal.get("allocation_job_id") == raw_id and terminal.get("state") == "FAILED"
            and terminal.get("exit_code") == "2:0" and terminal.get("allocated_cpus") == 1
            and type(terminal.get("elapsed_seconds")) is int and terminal["elapsed_seconds"] > 0
            and terminal.get("observed_utc") and terminal.get("evidence"),
            "coordinator-observed terminal parent charge required")
    require(item.get("work_item_id") == status.get("work_item_id") == request.get("work_item_id") == work
            and item.get("participant") == status.get("participant") == request.get("participant") == participant,
            "exact original work and participant identities required")
    require(status.get("schema_version") == request.get("schema_version") == "ep21.decoder_work_item.v1"
            and status.get("status") == "partial_work_item_stopped"
            and request.get("status") == "admitted_pending"
            and status.get("allocation", {}).get("job_id") == raw_id
            and request.get("allocation") == status.get("allocation")
            and status.get("allocation", {}).get("hostname") == "sh04-04n21"
            and status.get("error_type") == "OSError" and status.get("error") == "[Errno 5] Input/output error",
            "exact closed I/O-failure identity required")
    require(status.get("requested_spans") == request.get("requested_spans") == item.get("spans")
            and status.get("units_requested") == request.get("units_requested") == item.get("units_requested")
            and status.get("authorization") == str(Path(base) / "outputs/decoder_training_full_authorization.json")
            and status.get("work_manifest") == str(Path(base) / FULL_ROOT / "work_manifest.json")
            and status.get("training_role") == "subject_unique" and status.get("training_images") == 4712
            and status.get("fragment") == "channels.h5"
            and status.get("shared_state_reference") == str(Path(base) / assembly.required_state(participant).path)
            and status.get("training_row_binding_reference") == str(Path(base) / assembly.required_state(participant).binding_path),
            "unchanged request, manifest and retained training-state bindings required")
    require(all(status.get(key) is False for key in ("predictions_or_scores_computed", "tau_train_values_read",
            "tau_test_values_read", "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry")),
            "training-only data and no-successor boundary required")
    requested, committed, remaining = (_spans(status[key]) for key in
                                       ("requested_spans", "committed_spans", "remaining_spans"))
    require(requested and all(span.participant == participant for span in requested)
            and sum(span.count for span in requested) == item["units_requested"], "requested unit count mismatch")
    done = {span.key: span for span in committed}
    require(set(done) <= {span.key for span in requested}, "reported committed scope exceeds request")
    expected_remaining, expected_committed, stopped = [], [], False
    for span in requested:
        saved = done.get(span.key)
        cursor = span.start if saved is None else saved.stop
        require(saved is None or (not stopped and saved.start == span.start and cursor <= span.stop),
                "reported commits must be an ordered prefix of the fixed request")
        if saved is not None:
            expected_committed.append(saved)
        if cursor < span.stop:
            stopped = True
            expected_remaining.append(assembly.coverage.Span(*span.key, cursor, span.stop))
    require(committed == expected_committed and remaining == expected_remaining
            and type(status.get("completed_units")) is int
            and sum(span.count for span in committed) == status["completed_units"],
            "reported committed and remaining intervals must partition the original assignment")
    uncertain = status.get("uncertain_publication")
    if uncertain is not None:
        observed = assembly.span_from_record(uncertain)
        require(work in {"work_0032", "work_0061"} and expected_remaining
                and observed.key == expected_remaining[0].key and observed.start == expected_remaining[0].start
                and observed.stop <= expected_remaining[0].stop and observed.count == BLOCK_UNITS
                and uncertain.get("phase") == "completion_marker_fsync"
                and status.get("completed_units_is_lower_bound") is True
                and status.get("remaining_interval_status") == "not_authoritative_for_repair_due_to_uncertain_marker_publication",
                "exact reported ambiguous publication interval required")
    else:
        require(work not in {"work_0032", "work_0061"}
                and status.get("completed_units_is_lower_bound") is False
                and status.get("remaining_interval_status") == "exact_uncommitted_coverage",
                "reported exact-prefix metadata required")
    require(isinstance(status.get("closeout_errors"), list), "explicit closeout error list required")
    return requested


def selected_work_items(work_items=None):
    """Select only fixed identities, retaining their canonical inspection order."""
    if work_items is None:
        return list(WORKERS)
    require(isinstance(work_items, (list, tuple)) and len(work_items) > 0
            and all(isinstance(work, str) and work in WORKERS for work in work_items),
            "work-items must be a nonempty selection of the five fixed work IDs")
    require(len(set(work_items)) == len(work_items), "duplicate work-items are forbidden")
    return [work for work in WORKERS if work in work_items]


def admit_inputs(base, work_items=None):
    """Only fixed operational records; use existing journal, never a new receipt."""
    selected = selected_work_items(work_items)
    manifest = _read_json(_path(base, FULL_ROOT + "/work_manifest.json"))
    journal = _path(base, FULL_ROOT + "/submission.jsonl")
    require(journal.stat().st_size <= MAX_JSON_BYTES, "bounded existing accounting journal required")
    events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    records = []
    for work in selected:
        _, array_id, _ = WORKERS[work]
        items = [row for row in manifest["work_items"] if row.get("work_item_id") == work]
        terminals = [row for row in events if row.get("event") == "terminal_charge" and row.get("job_id") == array_id]
        require(len(items) == len(terminals) == 1, "one exact manifest assignment and terminal charge required")
        directory = FULL_ROOT + "/shards/" + work
        status = _read_json(_path(base, directory + "/work_item_status.json"))
        request = _read_json(_path(base, directory + "/requested_units.json"))
        validate_record(work, items[0], status, request, terminals[0], base)
        records.append(status)
    return records


def _error(exc):
    result = {"error_type": type(exc).__name__, "error": str(exc)}
    if isinstance(exc, OSError):
        result["errno"] = exc.errno
    if exc.__context__ is not None and exc.__context__ is not exc:
        result["earlier_error"] = {"error_type": type(exc.__context__).__name__, "error": str(exc.__context__)}
    return result


def _present_groups(handle, requested):
    import h5py
    expected = {}
    for span in requested:
        expected.setdefault(span.layer, set()).add(f"channel_{span.channel:03d}")
    require(set(handle.keys()) <= set(expected), "unexpected layer group in failed fragment")
    result = set()
    for layer in handle.keys():
        require(isinstance(handle.get(layer, getlink=True), h5py.HardLink), "nonlocal layer link forbidden")
        parent = handle[layer]
        require(isinstance(parent, h5py.Group) and set(parent.keys()) <= expected[layer],
                "unexpected channel group in failed fragment")
        for name in parent.keys():
            require(isinstance(parent.get(name, getlink=True), h5py.HardLink), "nonlocal channel link forbidden")
            result.add(f"{layer}/{name}")
    return result


def _marked_runs(markers, start):
    """At most 256 rows; preserve holes instead of declaring an invented prefix."""
    begin = None
    for offset, value in enumerate(markers):
        if value == 2 and begin is None:
            begin = start + offset
        if value != 2 and begin is not None:
            yield begin, start + offset
            begin = None
    if begin is not None:
        yield begin, start + len(markers)


def inspect_fragment(base, status, normalization, check):
    """Describe current readability and consistency, never historical fsync success.

    A valid marker observed now cannot establish which interrupted fsync reached
    durable storage at failure time. Even fully consistent observations carry
    no repaired coverage, reuse permission or retry authority.
    """
    import h5py
    import numpy as np
    work = status["work_item_id"]
    relative = FULL_ROOT + "/shards/" + work + "/channels.h5"
    status_relative = FULL_ROOT + "/shards/" + work + "/work_item_status.json"
    result = {"work_item_id": work, "fragment": relative, "status": "not_inspected", "inspectable": False,
              "inspection_complete": False, "reported_committed_units": status["completed_units"],
              "reported_committed_units_checked": 0, "observed_marker_2_units": 0,
              "ambiguous_marker_2_units_checked": 0, "channels": [],
              "historical_closeout_errors": status["closeout_errors"],
              "historical_uncertain_publication": status.get("uncertain_publication"),
              "repair_coverage_written": False, "original_artifacts_modified": False,
              "retry_authorized": False, "reuse_authorized": False}
    try:
        check()
        requested = _spans(status["requested_spans"])
        done = {span.key: span for span in _spans(status["committed_spans"])}
        uncertain = status.get("uncertain_publication")
        ambiguous = assembly.span_from_record(uncertain) if uncertain is not None else None
        with h5py.File(_path(base, relative), "r") as handle:
            result["inspectable"] = True
            assembly._attrs_equal(handle, {"schema_version": "ep21.decoder_work_item_fragment.v1",
                "participant": status["participant"], "work_item_id": work,
                "units_requested": status["units_requested"], "complete_decoder": False})
            present = _present_groups(handle, requested)
            for span in requested:
                check()
                ref = assembly.FragmentRef(relative, "grouped_channels", f"{span.layer}/channel_{span.channel:03d}",
                                           status_relative, span).validate()
                saved = done.get(span.key)
                cursor = span.start if saved is None else saved.stop
                uncertain_here = ambiguous if ambiguous is not None and ambiguous.key == span.key else None
                channel = {"layer": span.layer, "channel": span.channel,
                           "requested_interval": [span.start, span.stop], "present": ref.group in present,
                           "reported_committed_units_checked": 0, "observed_marker_2_units": 0}
                result["channels"].append(channel)
                if ref.group not in present:
                    require(saved is None and uncertain_here is None, "reported committed or ambiguous channel is absent")
                    continue
                group, origin = assembly.fragment_header(handle, ref)
                assembly._attrs_equal(group, {"spatial_Fortran_start": span.start, "spatial_Fortran_stop": span.stop,
                    "complete_decoder": False, "state_codes": "0=incomplete,2=durably_complete"})
                for begin in range(span.start, span.stop, BLOCK_UNITS):
                    check()
                    end = min(begin + BLOCK_UNITS, span.stop)
                    markers = group["state"][begin - origin:end - origin]
                    require(bool(np.isin(markers, (0, 2)).all()), "invalid completion marker")
                    spatial = np.arange(begin, end)
                    claimed = spatial < cursor
                    uncertain_mask = ((spatial >= uncertain_here.start) & (spatial < uncertain_here.stop)
                                      if uncertain_here is not None else np.zeros(len(markers), dtype=bool))
                    require(bool((markers[claimed] == 2).all()), "reported committed row lacks complete marker")
                    require(bool((markers[~claimed & ~uncertain_mask] == 0).all()),
                            "unreported complete marker outside explicit ambiguous interval")
                    marked = int(np.count_nonzero(markers == 2))
                    result["observed_marker_2_units"] += marked
                    channel["observed_marker_2_units"] += marked
                    if begin < cursor:
                        stop = min(end, cursor)
                        block = assembly.read_sparse_block(group, origin, ref, normalization, begin, stop)
                        del block
                        result["reported_committed_units_checked"] += stop - begin
                        channel["reported_committed_units_checked"] += stop - begin
                    if bool(uncertain_mask.any()):
                        observation = channel.setdefault("ambiguous_publication_observation", {
                            "interval": [uncertain_here.start, uncertain_here.stop],
                            "marker_0_units": 0, "marker_2_units": 0,
                            "marker_2_units_with_valid_coefficients": 0,
                            "historical_durability_reconciled": False})
                        selected = markers[uncertain_mask]
                        observation["marker_0_units"] += int(np.count_nonzero(selected == 0))
                        observation["marker_2_units"] += int(np.count_nonzero(selected == 2))
                        offset = max(begin, uncertain_here.start)
                        for left, right in _marked_runs(selected, offset):
                            block = assembly.read_sparse_block(group, origin, ref, normalization, left, right)
                            del block
                            observation["marker_2_units_with_valid_coefficients"] += right - left
                            result["ambiguous_marker_2_units_checked"] += right - left
        check()
        require(result["reported_committed_units_checked"] == status["completed_units"],
                "not every reported committed row was checked")
        result.update(status="consistent", inspection_complete=True)
    except InspectionTimeout as exc:
        result.update(status="timed_out", **_error(exc))
    except InspectionLimit as exc:
        # Keep a later alarm from replacing this terminal failure before run()
        # records it. Until disarming completes, this active exception remains
        # available as an alarm's context and forces a fail-closed exit.
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(status="resource_limit", **_error(exc))
    except (OSError, RuntimeError) as exc:
        # HDF5 open/read/close errors stop the batch. Do not switch locking,
        # clear consistency flags, retry access or ignore a close failure.
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(status="unreadable", **_error(exc))
    except (ValueError, KeyError, TypeError) as exc:
        result.update(status="corrupt", **_error(exc))
    return result


class Budget:
    def __init__(self, max_seconds, end_epoch):
        require(type(max_seconds) is int and 0 < max_seconds <= MAX_SECONDS
                and math.isfinite(end_epoch), "inspection bound must be 1..600 seconds")
        remaining = min(max_seconds, end_epoch - time.time() - 300)
        require(remaining > 0, "actual allocation must retain 300-second closure reserve")
        self.deadline = time.monotonic() + remaining
        self.seconds = remaining

    def check(self):
        if time.monotonic() >= self.deadline:
            raise InspectionTimeout("closed-fragment inspection time limit reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= 6.5 * 1024**3:
            raise InspectionLimit("closed-fragment inspection RSS limit reached")


def run(args):
    started = time.monotonic()
    selected = selected_work_items(args.work_items)
    host = socket.gethostname().split(".")[0]
    require(Path.cwd().resolve() == BASE and "-ln" not in host
            and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "7168"
            and not os.environ.get("SLURM_JOB_GPUS")
            and all(os.environ.get(key) == "1" for key in training.prefix.THREADS),
            "actual coordinator-owned 1-CPU/7-GiB compute shell required")
    require(os.environ.get("HDF5_USE_FILE_LOCKING") in (None, "TRUE", "1"),
            "HDF5 locking must not be disabled or weakened")
    budget = Budget(args.max_seconds, args.allocation_end_epoch)
    report = {"schema_version": "ep21.closed_io_fragment_inspection.v1",
              "observed_utc": datetime.now(timezone.utc).isoformat(),
              "allocation": {"job_id": os.environ["SLURM_JOB_ID"], "hostname": host,
                             "caller_verified_end_epoch": args.allocation_end_epoch},
              "max_seconds": budget.seconds, "block_units": BLOCK_UNITS,
              "requested_work_items": selected,
              "unrequested_work_items": [work for work in WORKERS if work not in selected],
              "quarantined_work_items": list(WORKERS),
              "status": "not_started", "fragments": [], "original_artifacts_modified": False,
              "repair_coverage_written": False, "retry_authorized": False, "reuse_authorized": False}
    previous = signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "caller must not have a competing alarm")
    def alarm(signum, frame):
        raise InspectionTimeout("closed-fragment inspection time limit reached")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, budget.seconds)
    try:
        training.authorize_full(args.authorization)
        records = admit_inputs(BASE, selected)
        normalizations = {}
        for status in records:
            budget.check()
            participant = status["participant"]
            if participant not in normalizations:
                normalizations[participant] = assembly.load_normalization(
                    BASE, assembly.required_state(participant), budget.check)
            result = inspect_fragment(BASE, status, normalizations[participant], budget.check)
            report["fragments"].append(result)
            if result["status"] in {"unreadable", "timed_out", "resource_limit"}:
                report["status"] = "stopped_" + result["status"]
                break
        else:
            budget.check()
            report["status"] = ("consistent_observations_only" if all(row["status"] == "consistent"
                                for row in report["fragments"]) else "inconsistent_observations")
    except InspectionTimeout as exc:
        report.update(status="stopped_timed_out", **_error(exc))
    except InspectionLimit as exc:
        report.update(status="stopped_resource_limit", **_error(exc))
    except (OSError, RuntimeError) as exc:
        report.update(status="stopped_unreadable_or_admission_failure", **_error(exc))
    except (ValueError, KeyError, TypeError) as exc:
        report.update(status="admission_or_normalization_inconsistent", **_error(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["attempted_work_items"] = [row["work_item_id"] for row in report["fragments"]]
    report["completely_checked_work_items"] = [row["work_item_id"] for row in report["fragments"]
                                               if row["inspection_complete"]]
    report["not_inspected_requested_work_items"] = [work for work in selected
        if work not in report["attempted_work_items"]]
    report["all_requested_fragments_completely_checked"] = (
        report["completely_checked_work_items"] == selected)
    report["all_five_fragments_completely_checked"] = (len(report["fragments"]) == len(WORKERS)
        and all(row["inspection_complete"] for row in report["fragments"]))
    report["not_inspected_work_items"] = [work for work in WORKERS
        if work not in {row["work_item_id"] for row in report["fragments"]}]
    print(json.dumps(report, indent=2), flush=True)
    # A completed diagnostic pass may discover semantic corruption. Unrelated
    # authorized fitting can continue without consuming any inspected fragment.
    # Exit 3 distinguishes only our own clean time limit; the caller may permit
    # its unrelated fit, while preserving this incomplete diagnostic evidence.
    # A timeout interrupting another exception never hides that earlier failure.
    if report["status"] in {"consistent_observations_only", "inconsistent_observations"}:
        return 0
    if report["status"] == "stopped_timed_out" and "earlier_error" not in report \
            and all(row["status"] in {"consistent", "corrupt", "timed_out"}
                    and "earlier_error" not in row for row in report["fragments"]):
        return 3
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--allocation-end-epoch", required=True, type=float,
                        help="actual scheduler end verified once by the coordinating caller")
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--work-items", nargs="+", choices=list(WORKERS),
                        help="explicit fixed work IDs to inspect; default: all five")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "inspection_not_started", **_error(exc),
                          "original_artifacts_modified": False, "retry_authorized": False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
