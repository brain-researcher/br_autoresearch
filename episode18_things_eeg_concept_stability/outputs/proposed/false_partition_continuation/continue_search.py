#!/usr/bin/env python3
"""INACTIVE EP18 adapter; requires a separately adopted deadline authority.

No authority file is supplied with this implementation. Required JSON fields
are checked in admit(): adopted status/decision record, frozen source and raw
config hashes, original bound config ID, exact predecessor, original/new
deadlines, and the fixed RESOURCES dictionary. Original producer/config/cache
identities remain unchanged. This module imports only the standard library
until admission succeeds in an actual compute allocation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

EPISODE = "episode18_things_eeg_concept_stability"
SOURCE = Path("outputs/code/false_partition_search.py")
CONFIG = Path("outputs/config/false_partition_sub07_20261012_001.json")
PREDECESSOR = Path("outputs/runs/false_partition_preparation/"
    "search_continuation_operations_20261013_001/segment_47679999/search/checkpoint.npz")
ORIGINAL_DEADLINE = 1791927000
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
RESOURCES = {"requested_cpus": 13, "maximum_allocated_cpus": 16,
             "memory_mib": 98304, "allocation_seconds": 36000,
             "cooperative_seconds": 32400, "numerical_workers": 1, "gpus": 0,
             "rss_byte_cap": 96636764160, "array_byte_cap": 85899345920,
             "maximum_reserved_CPU_seconds": 576000}
# Scheduler Start/End timestamps can differ by seconds from the requested limit.
# This admits reporting skew only; work still stops at Start+9h, with <1min tail.
SCHEDULER_TIMESTAMP_SKEW_SECONDS = 60


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_small(path, cap):
    require(path.is_file() and path.stat().st_size <= cap, "missing_or_oversized_metadata")
    return path.read_bytes()


def encoded(value):
    # Match the frozen producer's encoding; never insert the new deadline.
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def episode_root():
    return (Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch" / EPISODE).resolve()


def absolute_path(path):
    require(path.is_absolute() and ".." not in path.parts, "absolute_unambiguous_path_required")
    return path.resolve()


def admit(authority_path, output, deadline):
    """Read only bounded metadata/source bytes before any producer import."""
    require("-ln" not in socket.gethostname() and bool(os.environ.get("SLURM_JOB_ID")),
            "actual_compute_allocation_required")
    require(all(os.environ.get(key) == "1" for key in THREAD_ENV), "one_numerical_thread_required")
    require(os.environ.get("SLURM_JOB_NUM_NODES") == "1" and os.environ.get("SLURM_NTASKS") == "1",
            "single_node_single_task_required")
    require(not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_ARRAY_TASK_ID")
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "no_GPU_array_or_requeue")
    cpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    require(13 <= int(os.environ["SLURM_CPUS_PER_TASK"]) <= cpus <= 16,
            "fixed_request_13_actual_13_to_16_CPUs_required")
    require(int(os.environ["SLURM_MEM_PER_NODE"]) == RESOURCES["memory_mib"], "96_GiB_required")
    start, end = int(os.environ["SLURM_JOB_START_TIME"]), int(os.environ["SLURM_JOB_END_TIME"])
    now = time.time()
    require(start <= now < end and 0 < end-start <= (
            RESOURCES["allocation_seconds"] + SCHEDULER_TIMESTAMP_SKEW_SECONDS),
            "allocation_must_fit_ten_hours_with_at_most_sixty_seconds_timestamp_skew")

    root = episode_root()
    authority_path, output = absolute_path(authority_path), absolute_path(output)
    require(authority_path.is_relative_to(root / "outputs")
            and not authority_path.is_relative_to(root / "outputs/proposed"),
            "separately_adopted_authority_outside_proposed_directory_required")
    authority = json.loads(read_small(authority_path, 65536))
    expected = {"status", "episode", "scope", "decision_record", "original_source_sha256",
                "original_config_sha256", "original_config_id", "predecessor_job_id",
                "predecessor_checkpoint", "original_deadline_unix", "new_absolute_deadline_unix", "resources"}
    require(set(authority) == expected, "explicit_complete_authority_required")
    require(authority["status"] == "adopted" and authority["episode"] == "EP18"
            and authority["scope"] == "bounded_generator_continuation"
            and isinstance(authority["decision_record"], str) and authority["decision_record"].strip(),
            "recorded_scientist_adoption_required")
    require(authority["resources"] == RESOURCES, "adopted_resources_must_match_fixed_proposal")
    require(authority["original_deadline_unix"] == ORIGINAL_DEADLINE
            and type(authority["new_absolute_deadline_unix"]) is int
            and authority["new_absolute_deadline_unix"] > ORIGINAL_DEADLINE,
            "explicit_new_deadline_and_original_deadline_required")
    bound = min(start + RESOURCES["cooperative_seconds"], end - 60,
                authority["new_absolute_deadline_unix"] - 60)
    require(math.isfinite(deadline) and now + 60 < deadline <= bound,
            "deadline_must_fit_new_authority_nine_hours_and_cleanup_tail")
    require(authority["predecessor_job_id"] == "47679999"
            and authority["predecessor_checkpoint"] == PREDECESSOR.as_posix(),
            "exact_47679999_predecessor_required")
    require(output.is_relative_to(root / "outputs/runs/false_partition_preparation")
            and output.parent.is_dir() and not output.exists(), "fresh_existing_parent_output_required")

    source, config_path, predecessor = root / SOURCE, root / CONFIG, root / PREDECESSOR
    source_hash = hashlib.sha256(read_small(source, 128*1024)).hexdigest()
    raw_config = read_small(config_path, 65536)
    require(source_hash == authority["original_source_sha256"]
            and hashlib.sha256(raw_config).hexdigest() == authority["original_config_sha256"],
            "original_source_or_config_changed")
    config = json.loads(raw_config)
    require(config["not_after_unix"] == ORIGINAL_DEADLINE and config["participant_id"] == "sub-07",
            "original_deadline_and_participant_must_remain_frozen")
    config["code_sha256"] = source_hash
    require(hashlib.sha256(encoded(config)).hexdigest() == authority["original_config_id"],
            "original_bound_config_identity_required")
    prior = json.loads(read_small(predecessor.parent / "report.json", 1024**2))
    require(prior["status"] == "resource_incomplete" and prior["checkpoint_available"] is True
            and prior["EEG_read"] is False and prior["participant_id"] == "sub-07"
            and prior["member"] == 0 and prior["start"] == 11 and prior["swaps"] == 64556
            and prior["total_legal_proposals"] == 1164556,
            "released_predecessor_state_required")
    require(predecessor.is_file() and predecessor.stat().st_size <= config["segment_byte_cap"]//4
            and not predecessor.with_name("checkpoint.npz.pending").exists(),
            "complete_preserved_checkpoint_required")
    return root, authority, config, cpus


def load_producer(root):
    # The frozen source remains at its original path, so its own hash is unchanged.
    sys.path.insert(0, str(root / "outputs/code"))
    spec = importlib.util.spec_from_file_location("ep18_frozen_false_partition_search", root / SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args(argv)
    producer, owned, report, started = None, False, {}, time.time()
    try:
        root, authority, expected_config, cpus = admit(args.authority, args.output, args.deadline)
        producer = load_producer(root)
        producer.require_compute()
        require(all(os.environ.get(key) == "1" for key in producer.THREAD_ENV), "original_thread_check_failed")
        config = producer.load_config(root / CONFIG)
        require(config == expected_config and producer.digest(config) == authority["original_config_id"],
                "producer_original_config_binding_differs")
        os.umask(0o077)
        args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        owned = True
        report.update(status="started", phase="search", EEG_read=False, full_false_bank_qualified=False,
            checkpoint_available=False, participant_id=config["participant_id"],
            original_deadline_unix=ORIGINAL_DEADLINE,
            separately_adopted_deadline_unix=authority["new_absolute_deadline_unix"],
            execution_deadline_unix=args.deadline, predecessor_job_id="47679999",
            slurm_job_id=os.environ["SLURM_JOB_ID"], allocated_cpus=cpus, numerical_workers=1,
            maximum_reserved_CPU_seconds=cpus*RESOURCES["allocation_seconds"],
            scheduler_start_epoch=int(os.environ["SLURM_JOB_START_TIME"]),
            scheduler_end_epoch=int(os.environ["SLURM_JOB_END_TIME"]),
            scheduler_timestamp_skew_allowance_seconds=SCHEDULER_TIMESTAMP_SKEW_SECONDS,
            logical_counters_reset=False, preparation_repeated=False,
            automatic_retry=False, automatic_successor=False, scientific_release_authorized=False)
        producer.atomic_json(args.output / "settings.json", {"config": config, "phase": "search",
            "resume": str(root / PREDECESSOR), "deadline": args.deadline,
            "rss_byte_cap": RESOURCES["rss_byte_cap"], "array_byte_cap": RESOURCES["array_byte_cap"],
            "authority_path": str(args.authority), "adopted_authority": authority,
            "original_CLI_deadline_extended_by_separate_authority": True,
            "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat()})
        with producer.budget(args.deadline, RESOURCES["rss_byte_cap"]) as check:
            producer.run_search(config, args.output, root / PREDECESSOR,
                                RESOURCES["array_byte_cap"], check, report)
        require(report["status"] in ("complete_image_context_candidate_bank", "resource_incomplete",
                                     "exhausted_unavailable"), "unexpected_search_terminal_status")
        return 0 if report["status"] == "complete_image_context_candidate_bank" else (
            75 if report["status"] == "resource_incomplete" else 2)
    except BaseException as error:
        if producer is not None and isinstance(error, producer.BudgetStop):
            report.update(status="resource_incomplete", budget_stop_reason=str(error))
            return 75
        # Preserve original bounded/public error reporting; never print payloads.
        report.update(status="stopped_error", error_type=type(error).__name__)
        print("EP18 continuation refused or stopped: " + type(error).__name__, file=sys.stderr)
        return 1
    finally:
        if owned:
            report.update(elapsed_seconds=time.time()-started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finished_utc=datetime.now(timezone.utc).isoformat())
            producer.atomic_json(args.output / "report.json", report, cap=1024**2)


if __name__ == "__main__":
    raise SystemExit(main())
