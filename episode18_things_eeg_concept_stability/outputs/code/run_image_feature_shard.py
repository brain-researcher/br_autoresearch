#!/usr/bin/env python3
"""One finite full-extraction image family/range; pixels and opaque IDs only.

Production configuration is selected separately from the measured pilot. The
same-account boundary is procedural. No source routing, credentials or EEG is
read by this process. One model remains resident across its assigned packs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time
import zipfile

from feature_extractors import DIMENSIONS, ImageExtractor, pixels_only, validate_ids
from image_pixel_worker import PilotContractError, bounded_process, require, write_json
from qualify_model_runtime import require_compute

EPISODE = Path(__file__).resolve().parents[2]
FAMILIES = tuple(DIMENSIONS)
IMAGE_COUNT = 22448


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def positive_int(value, label):
    require(type(value) is int and value > 0, label)
    return value


def planned_tasks(config, image_count=IMAGE_COUNT):
    """Contiguous full coverage per family, independent of pixel-pack size."""
    require(set(config["family_shard_counts"]) == set(FAMILIES), "All six family shard counts are required")
    tasks = []
    for family in FAMILIES:
        count = positive_int(config["family_shard_counts"][family], "Invalid family shard count")
        require(count <= image_count, "Empty image tasks are prohibited")
        base, extra = divmod(image_count, count)
        first = 0
        for shard in range(count):
            stop = first + base + (shard < extra)
            tasks.append({"task_index": len(tasks), "family": family, "family_shard_index": shard,
                          "start": first, "stop": stop})
            first = stop
        require(first == image_count, "Family plan does not cover the complete inventory")
    return tasks


def load_config(path):
    """Read only explicit prospective resource/path choices; no private inputs."""
    config = read_json(path)
    require(config["image_count"] == IMAGE_COUNT and config["families"] == list(FAMILIES),
            "Full extraction requires all 22448 images and six unchanged families")
    for name in ("pixel_pack_images", "pixel_pack_byte_cap", "pixel_cache_byte_cap", "metadata_byte_cap",
                 "assembled_byte_cap", "run_artifact_byte_cap", "rss_bytes", "slurm_memory_mib",
                 "builder_max_seconds", "assembly_max_seconds", "max_parallel_tasks"):
        positive_int(config[name], "Invalid selected resource field: "+name)
    require(config["max_parallel_tasks"] <= 128, "Episode parallel CPU allowance exceeded")
    require(config["rss_bytes"] < config["slurm_memory_mib"]*1024**2, "RSS guard requires allocation headroom")
    require(set(config["worker_max_seconds"]) == set(FAMILIES)
            and set(config["worker_artifact_byte_cap"]) == set(FAMILIES), "Every family needs explicit resource limits")
    tasks = planned_tasks(config)
    for family in FAMILIES:
        positive_int(config["worker_max_seconds"][family], "Invalid family time allowance")
        positive_int(config["worker_artifact_byte_cap"][family], "Invalid family artifact allowance")
    reserved_seconds = config["builder_max_seconds"] + config["assembly_max_seconds"] + sum(
        config["worker_max_seconds"][task["family"]] for task in tasks)
    require(math.isfinite(config["run_cpu_hour_cap"]) and 0 < reserved_seconds/3600 <= config["run_cpu_hour_cap"],
            "Finite job reservations exceed the selected run CPU-hour cap")
    reserved_bytes = config["assembled_byte_cap"] + config["metadata_byte_cap"] + sum(
        config["worker_artifact_byte_cap"][task["family"]] for task in tasks)
    require(reserved_bytes <= config["run_artifact_byte_cap"], "Finite artifact reservations exceed the run cap")
    require(math.isfinite(config["not_after_unix"]), "Explicit global execution deadline required")
    run, cache = Path(config["run_dir"]), Path(config["pixel_cache_dir"])
    require(run.is_absolute() and run.parent.resolve() == (EPISODE/"outputs/runs/feature_preparation").resolve()
            and run.name.startswith("image_features_full_"), "Use the selected episode-local full-extraction run")
    require(cache.is_absolute() and cache.parent.resolve() == (EPISODE/"outputs/cache").resolve()
            and cache.name.startswith("image_pixels_full_"), "Use the selected restricted episode-local pixel cache")
    return config


def admit(config, phase, deadline, family=None):
    require_compute()
    require(os.environ.get("SLURM_MEM_PER_NODE") == str(config["slurm_memory_mib"])
            and not os.environ.get("SLURM_JOB_GPUS"), "Use the selected single-CPU, zero-GPU memory envelope")
    maximum = config["worker_max_seconds"][family] if phase == "worker" else config[phase+"_max_seconds"]
    require(math.isfinite(deadline) and 0 < deadline-time.time() <= maximum
            and deadline <= config["not_after_unix"], "Selected phase/global deadline exceeded")
    os.umask(0o077)


def validate_manifest(manifest, config, image_count=IMAGE_COUNT):
    require(manifest["status"] == "complete_pixel_delivery" and manifest["image_count"] == image_count,
            "A complete pixel-delivery manifest is required")
    ids = manifest["opaque_ids"]
    validate_ids(ids, image_count)
    require(ids == sorted(ids), "Opaque image order changed")
    require(manifest["tasks"] == planned_tasks(config, image_count), "Family task ranges changed")
    cursor, names = 0, set()
    for pack in manifest["packs"]:
        require(pack["start"] == cursor and cursor < pack["stop"] <= image_count,
                "Pixel packs have missing, overlapping or empty ranges")
        require(pack["name"] == f"pixels_{len(names):05d}.npz" and pack["name"] not in names,
                "Pixel pack identity changed")
        require(pack["bytes"] <= config["pixel_pack_byte_cap"], "Pixel pack exceeds its selected allowance")
        names.add(pack["name"])
        cursor = pack["stop"]
    require(cursor == image_count, "Pixel packs do not cover every image")
    require(sum(pack["bytes"] for pack in manifest["packs"]) <= config["pixel_cache_byte_cap"],
            "Pixel cache exceeds its selected allowance")
    return ids


def iterate_pixels(manifest, task, cache, config, check):
    """Stream only assigned UUIDs; pack boundaries never determine job size."""
    import numpy as np
    ids = manifest["opaque_ids"]
    for pack in manifest["packs"]:
        first, stop = max(task["start"], pack["start"]), min(task["stop"], pack["stop"])
        if first >= stop:
            continue
        path = cache/pack["name"]
        require(path.stat().st_size == pack["bytes"] and path.stat().st_size <= config["pixel_pack_byte_cap"],
                "Delivered pixel pack size changed")
        expected = ids[pack["start"]:pack["stop"]]
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            require([item.filename for item in entries] == [key+".npy" for key in expected]
                    and all(item.compress_type == zipfile.ZIP_STORED for item in entries),
                    "Pixel pack contains a different or nonopaque image inventory")
            require(sum(item.file_size for item in entries) <= config["pixel_pack_byte_cap"],
                    "Decoded pixel pack allowance exceeded")
        with np.load(path, allow_pickle=False) as arrays:
            for position in range(first, stop):
                check()
                yield ids[position], pixels_only(arrays[ids[position]])


def extract_task(manifest, config, task_index, check, extractor_factory=ImageExtractor, on_create=None):
    """All assigned images exactly once; no retry, replacement or partial success."""
    import numpy as np
    task = manifest["tasks"][task_index]
    require(task["task_index"] == task_index, "Task index differs from selected plan")
    family, first, stop = task["family"], task["start"], task["stop"]
    ids = manifest["opaque_ids"][first:stop]
    output = Path(config["run_dir"])/"tasks"/f"task_{task_index:05d}"
    output.mkdir(exist_ok=False, mode=0o700)
    if on_create is not None:
        on_create(output)
    cap = config["worker_artifact_byte_cap"][family]
    feature_bytes = len(ids)*DIMENSIONS[family]*4 + 128
    report_reserve = config["metadata_byte_cap"]
    require(feature_bytes + report_reserve < cap, "Assigned features exceed worker artifact allowance")
    started = time.monotonic()
    check()
    extractor = extractor_factory(family, manifest["assets"], device="cpu")
    if hasattr(extractor, "torch"):
        extractor.torch.set_num_interop_threads(1)
    load_seconds = time.monotonic()-started
    check()
    temporary = output/"features.partial.npy"
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(ids), DIMENSIONS[family]))
    require(temporary.stat().st_size <= feature_bytes, "Unexpected feature array header size")
    details_bytes, completed, inference_seconds = 0, 0, 0.
    with (output/"image_details.partial.jsonl").open("xb") as destination:
        for key, pixels in iterate_pixels(manifest, task, Path(config["pixel_cache_dir"]), config, check):
            require(completed < len(ids) and key == ids[completed], "Assigned image coverage changed")
            before = time.monotonic()
            values, details = extractor.encode([key], [pixels])
            inference_seconds += time.monotonic()-before
            require(values.dtype == np.float32 and values.shape == (1, DIMENSIONS[family])
                    and np.isfinite(values).all(), "Incomplete or nonfinite selected image features")
            encoded = (json.dumps({"opaque_image_id": key, "details": details}, allow_nan=False)+"\n").encode()
            require(feature_bytes + details_bytes + len(encoded) + report_reserve <= cap,
                    "Selected worker artifact allowance exceeded")
            destination.write(encoded)
            details_bytes += len(encoded)
            features[completed] = values[0]
            completed += 1
            del values, pixels
            check()
    require(completed == len(ids), "Partial image coverage cannot complete a task")
    features.flush()
    del features
    check()
    temporary.rename(output/"features.npy")
    (output/"image_details.partial.jsonl").rename(output/"image_details.jsonl")
    report = {"status": "complete_image_shard", **task, "images": completed, "inferences": completed,
              "opaque_ids": ids, "dimension": DIMENSIONS[family], "dtype": "float32", "batch_size": 1,
              "load_seconds": load_seconds, "inference_seconds": inference_seconds,
              "elapsed_seconds": time.monotonic()-started, "process_cpu_seconds": time.process_time(),
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "feature_bytes": (output/"features.npy").stat().st_size, "details_bytes": details_bytes,
              "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
              "separation": "procedural_same_account_not_OS_isolation", "EEG_or_source_metadata_read": False,
              "repeatability_evidence": "Reuse the completed fixed-singleton image pilot; no production repeats"}
    write_json(output/"report.json", report, min(report_reserve, cap-feature_bytes-details_bytes))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    owned_outputs = []
    try:
        config = load_config(args.config)
        tasks = planned_tasks(config)
        require(0 <= args.task_index < len(tasks), "Unselected image task")
        family = tasks[args.task_index]["family"]
        admit(config, "worker", args.deadline_unix, family)
        require(os.environ.get("L_SCRATCH_JOB") and args.local_dir.absolute().parent.resolve()
                == Path(os.environ["L_SCRATCH_JOB"]).resolve(), "Worker caches require this allocation's local scratch")
        args.local_dir.mkdir(exist_ok=False, mode=0o700)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                          HF_HOME=str(args.local_dir/"hf"), XDG_CACHE_HOME=str(args.local_dir/"cache"),
                          TMPDIR=str(args.local_dir))
        with bounded_process(args.deadline_unix, config["rss_bytes"]) as check:
            manifest = read_json(Path(config["pixel_cache_dir"])/"manifest.json")
            validate_manifest(manifest, config)
            report = extract_task(manifest, config, args.task_index, check, on_create=owned_outputs.append)
        print(json.dumps({key: report[key] for key in ("status", "task_index", "family", "images", "elapsed_seconds")}), flush=True)
        return 0
    except Exception as error:
        failure = {"status": "image_shard_failure", "task_index": args.task_index, "error_type": type(error).__name__}
        if isinstance(error, PilotContractError):
            failure["contract_diagnostic"] = str(error)
        if owned_outputs:
            output = owned_outputs[0]
            write_json(output/"failure.json", failure, 65536)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
