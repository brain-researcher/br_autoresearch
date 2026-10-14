#!/usr/bin/env python3
"""Explicit, single-use recovery of the complete original CLIP task 5.

No retries, partial-row reuse, source images, credentials, EEG or scheduler
access. The coordinator selects the grant and launches each finite phase.
Worker delegates unchanged extract_task with ONLY its output root overlaid.
Assembly preserves the original 26-task checks and original assembled target,
resolving exactly one source directory differently. Active sources stay intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

from feature_extractors import DIMENSIONS, ImageExtractor, require_compute
from image_pixel_worker import PilotContractError, bounded_process, require, write_json
from run_image_feature_shard import (EPISODE, FAMILIES, IMAGE_COUNT, extract_task,
    load_config, read_json, validate_manifest)

SCHEMA = "ep18_clip_task005_recovery001"
ORIGINAL_CONFIG = EPISODE / "outputs/config/full_image_extraction_20261012_001.json"
TASK_INDEX = 5
EXPECTED_TASK = {"task_index": 5, "family": "clip_vit_b32_image",
                 "family_shard_index": 1, "start": 11224, "stop": 22448}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
GRANT_FIELDS = {"schema", "authorized", "original_config", "fresh_worker_root",
    "original_assembly_target", "global_deadline_unix", "worker", "assembly",
    "numerical_threads", "replacement_task_index", "automatic_retry", "reserved_cpu_hours"}
WORKER = {"cpus": 2, "memory_mib": 15360, "rss_bytes": 12*1024**3,
          "max_seconds": 14400, "artifact_byte_cap": 128*1024**2}
ASSEMBLY = {"cpus": 1, "memory_mib": 7168, "rss_bytes": 13*1024**3//2,
            "max_seconds": 3600}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def absolute(value):
    require(isinstance(value, str), "Explicit path string required")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "Explicit absolute path required")
    return path


def load_grant(path):
    require(Path(path).stat().st_size <= 65536, "Recovery grant exceeds allowance")
    grant = read_json(path)
    require(set(grant) == GRANT_FIELDS and grant["schema"] == SCHEMA
            and grant["authorized"] is True and grant["automatic_retry"] is False,
            "Explicit authorized one-use recovery grant required")
    require(absolute(grant["original_config"]) == ORIGINAL_CONFIG
            and type(grant["replacement_task_index"]) is int
            and grant["replacement_task_index"] == TASK_INDEX
            and type(grant["numerical_threads"]) is int and grant["numerical_threads"] == 1,
            "Only original CLIP task five is selected")
    require(grant["worker"] == WORKER and grant["assembly"] == ASSEMBLY
            and all(type(value) is int for limits in (grant["worker"], grant["assembly"])
                    for value in limits.values()), "Selected recovery phase resources differ")
    require(type(grant["reserved_cpu_hours"]) is int and grant["reserved_cpu_hours"] == 9,
            "Recovery reserves exactly nine new CPU hours")
    config = load_config(grant["original_config"])
    original = Path(config["run_dir"])
    fresh, target = absolute(grant["fresh_worker_root"]), absolute(grant["original_assembly_target"])
    require(fresh.parent.resolve() == original.parent.resolve() and fresh != original
            and fresh.name == "image_features_full_20261012_recovery001" and not fresh.is_symlink()
            and target == original / "assembled", "Recovery output binding differs")
    require(type(grant["global_deadline_unix"]) is int
            and grant["global_deadline_unix"] == config["not_after_unix"] == 1791927000,
            "Original episode deadline must remain unchanged")
    require(config["worker_artifact_byte_cap"][EXPECTED_TASK["family"]]
            == grant["worker"]["artifact_byte_cap"], "Original feature byte allowance changed")
    return grant, config


def admit(grant, phase, deadline):
    """Use actual allocation resources; never pretend this is the old 1-CPU job."""
    require_compute()
    limits = grant[phase]
    require(os.environ.get("SLURM_CPUS_PER_TASK") == str(limits["cpus"])
            and os.environ.get("SLURM_MEM_PER_NODE") == str(limits["memory_mib"])
            and os.environ.get("SLURM_JOB_NUM_NODES") == "1"
            and os.environ.get("SLURM_JOB_CPUS_PER_NODE") == str(limits["cpus"])
            and not os.environ.get("SLURM_JOB_GPUS")
            and all(os.environ.get(name) == "1" for name in THREADS),
            "Actual recovery allocation or single-thread limits differ")
    start, end = int(os.environ["SLURM_JOB_START_TIME"]), int(os.environ["SLURM_JOB_END_TIME"])
    now = time.time()
    maximum = min(start + limits["max_seconds"] - 60, end - 60,
                  grant["global_deadline_unix"] - 10)
    require(0 < start <= now < end and math.isfinite(deadline) and now < deadline <= maximum,
            "Recovery deadline exceeds original allocation or episode allowance")
    os.umask(0o077)


def lineage(grant, config):
    return {"schema": SCHEMA, "grant_sha256": digest(grant),
        "original_config_sha256": digest(config), "original_config": grant["original_config"],
        "replaced_task": dict(EXPECTED_TASK), "failed_job_id": "47567238",
        "preserved_failed_task": str(Path(config["run_dir"])/"tasks/task_00005"),
        "replacement_task": str(Path(grant["fresh_worker_root"])/"tasks/task_00005"),
        "original_assembly_target": grant["original_assembly_target"],
        "partial_rows_reused": 0, "automatic_retry": False,
        "worker_reserved_cpu_hours": 8, "assembly_reserved_cpu_hours": 1,
        "new_reserved_cpu_hours": grant["reserved_cpu_hours"],
        "original_91_cpu_hour_grant_reissued": False}


def original_manifest(config):
    manifest = read_json(Path(config["pixel_cache_dir"])/"manifest.json")
    validate_manifest(manifest, config)
    require(len(manifest["tasks"]) == 26 and manifest["tasks"][TASK_INDEX] == EXPECTED_TASK,
            "Original complete task roster or replacement range differs")
    require(not (Path(config["run_dir"])/"pixel_delivery_failure.json").exists(),
            "Original pixel delivery failed")
    return manifest


def progress_factory(factory=ImageExtractor):
    """Report encoding progress only; an encode return is not durable publication."""
    def create(family, assets, device):
        delegate = factory(family, assets, device=device)
        class Progress:
            def __init__(self):
                if hasattr(delegate, "torch"):
                    self.torch = delegate.torch
                self.count, self.started = 0, time.monotonic()

            def encode(self, ids, pixels):
                result = delegate.encode(ids, pixels)
                self.count += len(ids)
                if self.count % 32 == 0 or self.count == 11224:
                    print(json.dumps({"stage": "CLIP_encoding_progress", "encoded_images": self.count,
                        "durable_publication_claimed": False, "elapsed_seconds": time.monotonic()-self.started,
                        "process_cpu_seconds": time.process_time(),
                        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
                return result
        return Progress()
    return create


def run_worker(grant, config, manifest, check, on_create):
    original_task = Path(config["run_dir"])/"tasks/task_00005"
    require(not (original_task/"report.json").exists(), "Original task already has a completion report")
    fresh = Path(grant["fresh_worker_root"])
    fresh.mkdir(mode=0o700)
    on_create(fresh)
    (fresh/"tasks").mkdir(mode=0o700)
    binding = lineage(grant, config)
    write_json(fresh/"recovery_binding.json", binding, 65536)
    # This is the sole changed field. Original model/ranges/assets/caps are
    # untouched. The separately admitted recovery owns its actual clock/RSS.
    effective = dict(config, run_dir=str(fresh))
    report = extract_task(manifest, effective, TASK_INDEX, check,
                          extractor_factory=progress_factory())
    check()
    completion = {"status": "complete_replacement_task",
        "binding": binding, "report_sha256": digest(report), "job_id": report["job_id"],
        "images": EXPECTED_TASK["stop"]-EXPECTED_TASK["start"], "cpus": 2, "memory_mib": 15360}
    require(report["images"] == completion["images"], "Full original replacement coverage required")
    task = fresh/"tasks/task_00005"
    used = (report["feature_bytes"] + report["details_bytes"]
            + (task/"report.json").stat().st_size + (fresh/"recovery_binding.json").stat().st_size)
    write_json(fresh/"recovery_complete.json", completion,
               min(65536, grant["worker"]["artifact_byte_cap"]-used))
    return report


def completed_sources(manifest, config, grant, image_count=IMAGE_COUNT):
    """Every original task exactly once, with task 5 as the sole explicit override."""
    ids = validate_manifest(manifest, config, image_count)
    require(grant["replacement_task_index"] == TASK_INDEX
            and manifest["tasks"][TASK_INDEX]["family"] == EXPECTED_TASK["family"],
            "Only the registered CLIP task may be replaced")
    run, fresh = Path(config["run_dir"]), Path(grant["fresh_worker_root"])
    require(not (run/"pixel_delivery_failure.json").exists()
            and not (fresh/"failure.json").exists()
            and not (run/"tasks/task_00005/report.json").exists(),
            "Failed delivery or competing original completion cannot enter recovery")
    binding = lineage(grant, config)
    require(read_json(fresh/"recovery_binding.json") == binding, "Replacement lineage differs")
    completed = read_json(fresh/"recovery_complete.json")
    require(completed["status"] == "complete_replacement_task" and completed["binding"] == binding
            and completed["cpus"] == 2 and completed["memory_mib"] == 15360,
            "Replacement completion or allocation binding differs")
    sources = []
    for task in manifest["tasks"]:
        root = fresh if task["task_index"] == TASK_INDEX else run
        output = root/"tasks"/f"task_{task['task_index']:05d}"
        require(not output.is_symlink() and not (output/"failure.json").exists(),
                "Failed or redirected task cannot enter assembly")
        report = read_json(output/"report.json")
        require(report["status"] == "complete_image_shard"
                and all(report[key] == value for key, value in task.items()),
                "Image task binding or completion differs")
        expected = ids[task["start"]:task["stop"]]
        require(report["opaque_ids"] == expected and report["images"] == report["inferences"] == len(expected)
                and report["dimension"] == DIMENSIONS[task["family"]] and report["dtype"] == "float32"
                and report["batch_size"] == 1, "Task contains duplicate missing or altered rows")
        require((output/"features.npy").stat().st_size == report["feature_bytes"]
                and (output/"image_details.jsonl").stat().st_size == report["details_bytes"],
                "Complete task artifact size changed")
        if task["task_index"] == TASK_INDEX:
            require(completed["report_sha256"] == digest(report)
                    and completed["job_id"] == report["job_id"]
                    and completed["images"] == len(expected), "Replacement report binding differs")
        sources.append((report, output))
    return sources


def assemble_recovery(manifest, config, grant, check, image_count=IMAGE_COUNT, on_create=None):
    """Original assembly algorithm with a strict per-task source resolver."""
    import numpy as np
    sources = completed_sources(manifest, config, grant, image_count)
    output = Path(config["run_dir"])/"assembled"
    require(Path(grant["original_assembly_target"]) == output, "Original assembly target changed")
    output.mkdir(exist_ok=False, mode=0o700)
    if on_create is not None:
        on_create(output)
    started = time.monotonic()
    cap, reserve = config["assembled_byte_cap"], config["metadata_byte_cap"]
    binding = lineage(grant, config)
    used = write_json(output/"opaque_ids.json", manifest["opaque_ids"], reserve)
    source_record = {"binding": binding, "tasks": [{"task_index": report["task_index"],
        "directory": str(path), "job_id": report["job_id"], "report_sha256": digest(report)}
        for report, path in sources]}
    used += write_json(output/"recovery_sources.json", source_record, reserve)
    families = {}
    for family in FAMILIES:
        check()
        selected = [(report, path) for report, path in sources if report["family"] == family]
        feature_bytes = image_count*DIMENSIONS[family]*4 + 128
        require(used+feature_bytes+reserve <= cap, "Assembled feature allowance exceeded")
        temporary = output/(family+".partial.npy")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                          shape=(image_count, DIMENSIONS[family]))
        require(temporary.stat().st_size <= feature_bytes, "Unexpected assembled feature header size")
        used += temporary.stat().st_size
        cursor, details_bytes = 0, 0
        with (output/(family+".details.partial.jsonl")).open("xb") as destination:
            for report, source in selected:
                check()
                first, stop = report["start"], report["stop"]
                require(first == cursor, "Family has overlapping or missing ranges")
                chunk = np.load(source/"features.npy", allow_pickle=False, mmap_mode="r")
                require(chunk.dtype == np.float32 and chunk.shape == (stop-first, DIMENSIONS[family])
                        and np.isfinite(chunk).all(), "Generated feature matrix is incomplete or nonfinite")
                values[first:stop] = chunk
                del chunk
                row_count = 0
                with (source/"image_details.jsonl").open("rb") as details:
                    for line in details:
                        check()
                        item = json.loads(line)
                        require(row_count < stop-first
                                and item["opaque_image_id"] == manifest["opaque_ids"][first+row_count]
                                and set(item) == {"opaque_image_id", "details"}, "Generated detail coverage differs")
                        require(used+len(line)+reserve <= cap, "Assembled detail allowance exceeded")
                        destination.write(line)
                        used += len(line)
                        details_bytes += len(line)
                        row_count += 1
                require(row_count == stop-first, "Missing generated image details")
                cursor = stop
        require(cursor == image_count, "Full family coverage is incomplete")
        values.flush()
        del values
        check()
        temporary.rename(output/(family+".npy"))
        (output/(family+".details.partial.jsonl")).rename(output/(family+".details.jsonl"))
        families[family] = {"images": image_count, "dimension": DIMENSIONS[family],
            "feature_bytes": (output/(family+".npy")).stat().st_size,
            "details_bytes": details_bytes, "tasks": len(selected)}
    check()
    report = {"status": "all_image_features_completed", "analysis_id": "feature_preparation",
        "images": image_count, "families": families, "complete_tasks": len(sources), "batch_size": 1,
        "all_images_each_family_exactly_once": True, "artifact_bytes_before_report": used,
        "elapsed_seconds": time.monotonic()-started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "EEG_read": False, "audit_seal": False, "frozen_feature_record_emitted": False,
        "separation": "procedural_same_account_not_OS_isolation",
        "pixel_cache_cleanup": "Await separately recorded coordinator decision; this assembler deletes nothing",
        "finished_utc": datetime.now(timezone.utc).isoformat(), "recovery": binding,
        "source_record": "recovery_sources.json"}
    write_json(output/"report.json", report, min(reserve, cap-used))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--phase", choices=("worker", "assembly"), required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--local-dir", type=Path)
    args = parser.parse_args(argv)
    owned = []
    try:
        grant, config = load_grant(args.grant)
        admit(grant, args.phase, args.deadline_unix)
        require((args.local_dir is not None) == (args.phase == "worker"), "Worker alone needs local scratch")
        if args.phase == "worker":
            require(os.environ.get("L_SCRATCH_JOB") and args.local_dir.is_absolute()
                    and args.local_dir.parent.resolve() == Path(os.environ["L_SCRATCH_JOB"]).resolve(),
                    "Worker caches require the actual allocation local scratch")
            args.local_dir.mkdir(mode=0o700)
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                HF_HOME=str(args.local_dir/"hf"), XDG_CACHE_HOME=str(args.local_dir/"cache"),
                TMPDIR=str(args.local_dir))
        with bounded_process(args.deadline_unix, grant[args.phase]["rss_bytes"]) as check:
            manifest = original_manifest(config)
            report = (run_worker(grant, config, manifest, check, owned.append) if args.phase == "worker"
                      else assemble_recovery(manifest, config, grant, check, on_create=owned.append))
        print(json.dumps({"status": report["status"], "phase": args.phase,
                          "images": report["images"], "recovery": SCHEMA}), flush=True)
        return 0
    except Exception as error:
        failure = {"status": "CLIP_recovery_failure", "phase": args.phase,
                   "error_type": type(error).__name__, "automatic_retry": False}
        if isinstance(error, PilotContractError):
            failure["contract_diagnostic"] = str(error)
        # Exclusive mkdir grants ownership. Duplicate attempts never obtain it
        # and must not append a failure marker to previous outputs.
        if owned:
            write_json(owned[0]/"failure.json", failure, 65536)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
