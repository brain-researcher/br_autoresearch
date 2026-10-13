#!/usr/bin/env python3
"""Assemble every frozen EP18 image and all six families, or refuse completion.

Reads only opaque pixel-delivery/task records and generated image features.
No EEG, source crosswalk, credential, outcome selection or missing-row repair.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

from feature_extractors import DIMENSIONS
from image_pixel_worker import PilotContractError, bounded_process, require, write_json
from run_image_feature_shard import FAMILIES, IMAGE_COUNT, admit, load_config, read_json, validate_manifest


def completed_tasks(manifest, config, image_count=IMAGE_COUNT):
    """Check exact selected coverage before creating any assembled artifact."""
    ids = validate_manifest(manifest, config, image_count)
    run = Path(config["run_dir"])
    require(not (run/"pixel_delivery_failure.json").exists(), "Pixel builder did not finish successfully")
    reports = []
    for task in manifest["tasks"]:
        output = run/"tasks"/f"task_{task['task_index']:05d}"
        require(not (output/"failure.json").exists(), "Failed image task cannot enter assembly")
        report = read_json(output/"report.json")
        require(report["status"] == "complete_image_shard"
                and all(report[key] == value for key, value in task.items()), "Image task binding or completion differs")
        expected = ids[task["start"]:task["stop"]]
        require(report["opaque_ids"] == expected and report["images"] == report["inferences"] == len(expected)
                and report["dimension"] == DIMENSIONS[task["family"]] and report["dtype"] == "float32"
                and report["batch_size"] == 1, "Task contains duplicate, missing or altered feature rows")
        require((output/"features.npy").stat().st_size == report["feature_bytes"]
                and (output/"image_details.jsonl").stat().st_size == report["details_bytes"],
                "Complete task artifact size changed")
        reports.append(report)
    return reports


def assemble(manifest, config, check, image_count=IMAGE_COUNT, on_create=None):
    import numpy as np
    reports = completed_tasks(manifest, config, image_count)
    run = Path(config["run_dir"])
    output = run/"assembled"
    output.mkdir(exist_ok=False, mode=0o700)
    if on_create is not None:
        on_create(output)
    started = time.monotonic()
    cap, reserve = config["assembled_byte_cap"], config["metadata_byte_cap"]
    used = write_json(output/"opaque_ids.json", manifest["opaque_ids"], reserve)
    families = {}
    for family in FAMILIES:
        check()
        family_reports = [report for report in reports if report["family"] == family]
        feature_bytes = image_count*DIMENSIONS[family]*4 + 128
        require(used+feature_bytes+reserve <= cap, "Assembled feature allowance exceeded")
        temporary = output/(family+".partial.npy")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                          shape=(image_count, DIMENSIONS[family]))
        require(temporary.stat().st_size <= feature_bytes, "Unexpected assembled feature header size")
        used += temporary.stat().st_size
        cursor, details_bytes = 0, 0
        with (output/(family+".details.partial.jsonl")).open("xb") as destination:
            for report in family_reports:
                check()
                first, stop = report["start"], report["stop"]
                require(first == cursor, "Family has overlapping or missing ranges")
                source = run/"tasks"/f"task_{report['task_index']:05d}"
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
                            "details_bytes": details_bytes, "tasks": len(family_reports)}
    check()
    report = {"status": "all_image_features_completed", "analysis_id": "feature_preparation",
              "images": image_count, "families": families, "complete_tasks": len(reports), "batch_size": 1,
              "all_images_each_family_exactly_once": True, "artifact_bytes_before_report": used,
              "elapsed_seconds": time.monotonic()-started,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "EEG_read": False, "audit_seal": False, "frozen_feature_record_emitted": False,
              "separation": "procedural_same_account_not_OS_isolation",
              "pixel_cache_cleanup": "Await separately recorded coordinator decision; this assembler deletes nothing",
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"report.json", report, min(reserve, cap-used))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    owned_outputs = []
    try:
        config = load_config(args.config)
        admit(config, "assembly", args.deadline_unix)
        with bounded_process(args.deadline_unix, config["rss_bytes"]) as check:
            manifest = read_json(Path(config["pixel_cache_dir"])/"manifest.json")
            report = assemble(manifest, config, check, on_create=owned_outputs.append)
        print(json.dumps({key: report[key] for key in ("status", "images", "complete_tasks", "elapsed_seconds")}), flush=True)
        return 0
    except Exception as error:
        failure = {"status": "image_feature_assembly_failure", "error_type": type(error).__name__}
        if isinstance(error, PilotContractError):
            failure["contract_diagnostic"] = str(error)
        if owned_outputs:
            output = owned_outputs[0]
            write_json(output/"failure.json", failure, 65536)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
